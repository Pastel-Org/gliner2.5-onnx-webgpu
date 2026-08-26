/**
 * JointIE host: pair proposal + beam over relation-scorer logits.
 * Neural scores come from onnx/heads.onnx. This file is combinatorial decode.
 */

export function proposeRelationPairs(entities, relationTypes, {
  headsPerType = 32,
  tailsPerType = 32,
  pairCap = 64,
} = {}) {
  const types = Object.entries(relationTypes);
  const pairs = [];
  types.forEach(([relType, spec], relIndex) => {
    const headLabs = new Set(spec.head || spec.heads || []);
    const tailLabs = new Set(spec.tail || spec.tails || []);
    const allowSelf = Boolean(spec.allowSelf ?? spec.allow_self);
    const heads = entities
      .filter((e) => headLabs.has(e.label))
      .sort((a, b) => b.score - a.score)
      .slice(0, headsPerType);
    const tails = entities
      .filter((e) => tailLabs.has(e.label))
      .sort((a, b) => b.score - a.score)
      .slice(0, tailsPerType);
    const local = [];
    for (const h of heads) {
      for (const t of tails) {
        if (!allowSelf && h.wordStart === t.wordStart && h.wordEnd === t.wordEnd) continue;
        local.push({
          relType,
          relIndex,
          head: h,
          tail: t,
          headStart: h.wordStart,
          headEnd: h.wordEnd,
          tailStart: t.wordStart,
          tailEnd: t.wordEnd,
        });
      }
    }
    local.sort((a, b) => (b.head.score + b.tail.score) - (a.head.score + a.tail.score));
    pairs.push(...local.slice(0, pairCap));
  });
  return pairs;
}

export function beamSearchRelations(entities, scoredPairs, { beamWidth = 16, threshold = 0.5 } = {}) {
  const edges = scoredPairs
    .filter((p) => p.score >= threshold)
    .sort((a, b) => b.score + b.head.score + b.tail.score - (a.score + a.head.score + a.tail.score));

  const nodeKey = (e) => `${e.label}\0${e.wordStart}\0${e.wordEnd}`;
  let beam = [{ nodes: new Set(), edges: [], score: 0 }];

  for (const edge of edges) {
    const expanded = beam.slice();
    for (const st of beam) {
      const hk = nodeKey(edge.head);
      const tk = nodeKey(edge.tail);
      const used = new Set(st.edges.map((e) => `${e.relType}\0${nodeKey(e.head)}\0${nodeKey(e.tail)}`));
      const id = `${edge.relType}\0${hk}\0${tk}`;
      if (used.has(id)) continue;
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
