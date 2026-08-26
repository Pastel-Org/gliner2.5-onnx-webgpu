import { sigmoid } from "./gliner-boundary.mjs";

function mentionFromSpan(label, s, e, score, wordOffsets, text) {
  if (s >= e || e > wordOffsets.length || s < 0) return null;
  const charStart = wordOffsets[s].start;
  const charEnd = wordOffsets[e - 1].end;
  const surface = text.slice(charStart, charEnd);
  const lead = surface.length - surface.trimStart().length;
  const stripped = surface.trim();
  if (!stripped) return null;
  return {
    label,
    text: stripped,
    start: charStart + lead,
    end: charStart + lead + stripped.length,
    score,
    wordStart: s,
    wordEnd: e,
  };
}

/**
 * Mentions from the in-graph candidate lattice (no overlap resolution).
 * Python TypedRelationPairGenerator uses argument_threshold 0.2 on these.
 */
export function collectLatticeMentions(marg, {
  argumentThreshold = 0.0,
  pairTemperature = 1.0,
} = {}) {
  const { pairIndices, pairLogits, pairValid, candidateCount, labels, words, normalized } = marg;
  if (!pairLogits) return [];
  const Q = labels.length;
  const C = candidateCount;
  const out = [];
  for (let q = 0; q < Q; q++) {
    const seen = new Map();
    for (let c = 0; c < C; c++) {
      if (!pairValid[q * C + c]) continue;
      const s = Number(pairIndices[q * C * 2 + c * 2]);
      const e = Number(pairIndices[q * C * 2 + c * 2 + 1]);
      const p = sigmoid(pairLogits[q * C + c] / pairTemperature);
      if (p < argumentThreshold) continue;
      const key = `${s},${e}`;
      if (!seen.has(key) || p > seen.get(key).p) seen.set(key, { s, e, p });
    }
    for (const { s, e, p } of seen.values()) {
      const m = mentionFromSpan(labels[q], s, e, p, words, normalized);
      if (m) out.push(m);
    }
  }
  return out;
}

export function proposeRelationPairs(mentions, relationTypes, {
  headsPerType = 32,
  tailsPerType = 32,
  pairCap = 128,
} = {}) {
  const types = Object.entries(relationTypes);
  const pairs = [];
  types.forEach(([relType, spec], relIndex) => {
    const headLabs = new Set(spec.head || spec.heads || []);
    const tailLabs = new Set(spec.tail || spec.tails || []);
    const allowSelf = Boolean(spec.allowSelf ?? spec.allow_self);
    const rankArgs = (list, cap) => list
      .slice()
      .sort((a, b) => (a.wordEnd - b.wordEnd) || (a.wordStart - b.wordStart))
      .sort((a, b) => b.score - a.score)
      .slice(0, cap);
    const heads = rankArgs(mentions.filter((e) => headLabs.has(e.label)), headsPerType);
    const tails = rankArgs(mentions.filter((e) => tailLabs.has(e.label)), tailsPerType);
    const local = [];
    for (const h of heads) {
      for (const t of tails) {
        if (!allowSelf && h.wordStart === t.wordStart && h.wordEnd === t.wordEnd) continue;
        local.push({
          relType,
          relIndex,
          spec,
          head: h,
          tail: t,
          headStart: h.wordStart,
          headEnd: h.wordEnd,
          tailStart: t.wordStart,
          tailEnd: t.wordEnd,
        });
      }
    }
    local.sort((a, b) => (b.head.score * b.tail.score) - (a.head.score * a.tail.score));
    pairs.push(...local.slice(0, pairCap));
  });
  return pairs;
}

function nodeKey(e) {
  return `${e.label}\0${e.wordStart}\0${e.wordEnd}`;
}

function edgeAllowed(edge, stateEdges) {
  const spec = edge.spec || {};
  const uniqueHead = Boolean(spec.unique_head ?? spec.uniqueHead);
  const uniqueTail = Boolean(spec.unique_tail ?? spec.uniqueTail);
  const maxPerHead = spec.max_per_head ?? spec.maxPerHead ?? (uniqueHead ? 1 : null);
  const maxPerTail = spec.max_per_tail ?? spec.maxPerTail ?? (uniqueTail ? 1 : null);
  const hk = nodeKey(edge.head);
  const tk = nodeKey(edge.tail);
  const sameType = stateEdges.filter((e) => e.relType === edge.relType);
  if (maxPerHead != null) {
    const n = sameType.filter((e) => nodeKey(e.head) === hk).length;
    if (n >= maxPerHead) return false;
  }
  if (maxPerTail != null) {
    const n = sameType.filter((e) => nodeKey(e.tail) === tk).length;
    if (n >= maxPerTail) return false;
  }
  return true;
}

export function beamSearchRelations(scoredPairs, { beamWidth = 16, threshold = 0.5 } = {}) {
  const edges = scoredPairs
    .filter((p) => p.score >= threshold)
    .sort((a, b) => b.score + b.head.score + b.tail.score - (a.score + a.head.score + a.tail.score));

  let beam = [{ nodes: new Set(), edges: [], score: 0 }];

  for (const edge of edges) {
    const expanded = beam.slice();
    for (const st of beam) {
      const hk = nodeKey(edge.head);
      const tk = nodeKey(edge.tail);
      const id = `${edge.relType}\0${hk}\0${tk}`;
      if (st.edges.some((e) => `${e.relType}\0${nodeKey(e.head)}\0${nodeKey(e.tail)}` === id)) continue;
      if (!edgeAllowed(edge, st.edges)) continue;
      let gain = edge.score;
      const nodes = new Set(st.nodes);
      if (!nodes.has(hk)) { nodes.add(hk); gain += edge.head.score; }
      if (!nodes.has(tk)) { nodes.add(tk); gain += edge.tail.score; }
      if (gain < 0) continue;
      expanded.push({ nodes, edges: st.edges.concat(edge), score: st.score + gain });
    }
    expanded.sort((a, b) => b.score - a.score);
    const uniq = [];
    const seen = new Set();
    for (const st of expanded) {
      const sig = [...st.nodes].sort().join("|") + "#" + st.edges.map((e) => `${e.relType}:${nodeKey(e.head)}>${nodeKey(e.tail)}`).join(",");
      if (seen.has(sig)) continue;
      seen.add(sig);
      uniq.push(st);
      if (uniq.length >= beamWidth) break;
    }
    beam = uniq;
  }

  const best = beam[0] || { nodes: new Set(), edges: [], score: 0 };
  return {
    score: best.score,
    relations: best.edges.map((e) => ({
      type: e.relType,
      head: { text: e.head.text, label: e.head.label, start: e.head.start, end: e.head.end },
      tail: { text: e.tail.text, label: e.tail.label, start: e.tail.start, end: e.tail.end },
      score: e.score,
    })),
  };
}

/** Zip list-valued fields into repeated records (host assignment, not RecordHead). */
export function zipRecords(fieldHits, { strFields = [], parent = "record" } = {}) {
  const names = Object.keys(fieldHits);
  let n = 1;
  for (const name of names) {
    if (!strFields.includes(name)) n = Math.max(n, fieldHits[name].length);
  }
  const recs = [];
  for (let i = 0; i < n; i++) {
    const rec = {};
    for (const name of names) {
      const hits = fieldHits[name];
      if (strFields.includes(name)) rec[name] = hits[0] ?? null;
      else rec[name] = hits[i] ?? null;
    }
    recs.push(rec);
  }
  return { [parent]: recs };
}

export function attachAttributesFromLattice(entities, attrMarg, attrLabels) {
  const C = attrMarg.candidateCount;
  if (!C || !attrMarg.pairLogits) return entities;
  const temp = attrMarg.pairTemperature || 1;
  for (const e of entities) {
    let hit = null;
    for (let q = 0; q < attrLabels.length; q++) {
      for (let c = 0; c < C; c++) {
        if (!attrMarg.pairValid[q * C + c]) continue;
        const s = Number(attrMarg.pairIndices[q * C * 2 + c * 2]);
        const en = Number(attrMarg.pairIndices[q * C * 2 + c * 2 + 1]);
        if (s !== e.wordStart || en !== e.wordEnd) continue;
        const p = sigmoid(attrMarg.pairLogits[q * C + c] / temp);
        if (!hit || p > hit.score) hit = { label: attrLabels[q], score: p };
      }
    }
    if (hit) {
      e.attribute = hit.label;
      e.attributeScore = hit.score;
    }
  }
  return entities;
}

export function decodeAssignedRecords({
  parent, parsed, instSlots, assign, marg, toItem, anchor = 0,
}) {
  const C = marg.candidateCount;
  const F = parsed.length;
  const recs = [];
  const temp = marg.pairTemperature || 1;
  for (let i = 0; i < instSlots.length; i++) {
    const rec = {};
    for (let f = 0; f < F; f++) {
      if (f === anchor) {
        const c = instSlots[i];
        const s = Number(marg.pairIndices[f * C * 2 + c * 2]);
        const e = Number(marg.pairIndices[f * C * 2 + c * 2 + 1]);
        const p = sigmoid(marg.pairLogits[f * C + c] / temp);
        const m = mentionFromSpan(parsed[f].name, s, e, p, marg.words, marg.normalized);
        rec[parsed[f].name] = parsed[f].dtype === "list" ? (m ? [toItem(m)] : []) : (m ? toItem(m) : null);
        continue;
      }
      const width = C + 1;
      const base = i * F * width + f * width;
      const nullLogit = assign[base];
      if (parsed[f].dtype === "list") {
        const hits = [];
        const spans = [];
        for (let c = 0; c < C; c++) {
          if (!marg.pairValid[f * C + c]) continue;
          if (assign[base + 1 + c] <= nullLogit) continue;
          const p = sigmoid(marg.pairLogits[f * C + c] / temp);
          if (p < 0.5) continue;
          const s = Number(marg.pairIndices[f * C * 2 + c * 2]);
          const e = Number(marg.pairIndices[f * C * 2 + c * 2 + 1]);
          spans.push({ s, e, p, c });
        }
        spans.sort((a, b) => b.p - a.p);
        const kept = [];
        for (const sp of spans) {
          if (kept.some((k) => !(sp.e <= k.s || k.e <= sp.s))) continue;
          kept.push(sp);
        }
        kept.sort((a, b) => a.s - b.s);
        for (const sp of kept) {
          const m = mentionFromSpan(parsed[f].name, sp.s, sp.e, sp.p, marg.words, marg.normalized);
          if (m) hits.push(toItem(m));
        }
        rec[parsed[f].name] = hits;
      } else {
        let best = 0;
        let bestV = nullLogit;
        for (let c = 0; c < C; c++) {
          const v = assign[base + 1 + c];
          if (v > bestV) { bestV = v; best = c + 1; }
        }
        if (best === 0) {
          rec[parsed[f].name] = null;
          continue;
        }
        const c = best - 1;
        const s = Number(marg.pairIndices[f * C * 2 + c * 2]);
        const e = Number(marg.pairIndices[f * C * 2 + c * 2 + 1]);
        const p = sigmoid(marg.pairLogits[f * C + c] / temp);
        const m = mentionFromSpan(parsed[f].name, s, e, p, marg.words, marg.normalized);
        rec[parsed[f].name] = m ? toItem(m) : null;
      }
    }
    recs.push(rec);
  }
  for (const p of parsed) {
    const seen = new Set();
    for (const rec of recs) {
      const val = rec[p.name];
      const items = Array.isArray(val) ? val : (val == null ? [] : [val]);
      const kept = [];
      for (const item of items) {
        const key = typeof item === "string" ? item.toLowerCase() : `${item.start}-${item.end}-${String(item.text || "").toLowerCase()}`;
        if (seen.has(key)) continue;
        seen.add(key);
        kept.push(item);
      }
      rec[p.name] = Array.isArray(val) ? kept : (kept[0] ?? null);
    }
  }
  return { [parent]: recs };
}
