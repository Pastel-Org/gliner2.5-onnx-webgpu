/**
 * GLiNER2.5 BoundaryExtractor — ONNX Runtime Web (WebGPU) inference, in the browser.
 *
 * Host-side protocol for the gliner2.5-*-onnx boundary-architecture exports
 * (DeBERTa encoder + boundary heads, opset 17). The ONNX graph runs the
 * encoder over packed input_ids and emits per-query boundary marginals
 * start_logits / end_logits [B, Q, L+1]; everything else — word splitting,
 * entity-schema packing, subword/query routing, span decode — lives here.
 *
 * Protocol reference: fastino-ai/GLiNER2 (Apache-2.0), gliner2/processor.py
 * and gliner2/models/boundary/*. Clean-room reimplementation in JS.
 *
 * Decode note: v2 graphs emit pair_logits from the in-graph reranker.
 * Span score is sigmoid(pair_logit / pair_temperature). v1 graphs fall
 * back to min(sigmoid(start), sigmoid(end)).
 *
 * Pure ES module, no Node APIs. Pair it with any tokenizer that can encode a
 * single token to subword ids (e.g. transformers.js AutoTokenizer).
 */

export const GLINER_MODELS = {
  small: {
    repo: "nicolasembleton/gliner2.5-small-v1-onnx",
    params: "74M",
    encoder: "DeBERTa-v3-xsmall",
    languages: "English",
  },
  base: {
    repo: "nicolasembleton/gliner2.5-base-v1-onnx",
    params: "194M",
    encoder: "DeBERTa-v3-base",
    languages: "English",
  },
  multi: {
    repo: "nicolasembleton/gliner2.5-multi-v1-onnx",
    params: "287M",
    encoder: "mDeBERTa-v3-base",
    languages: "Multilingual",
  },
};

/** HF resolve URL for a repo file (LFS-aware, CORS-enabled). */
export function hfFileUrl(repo, file) {
  return `https://huggingface.co/${repo}/resolve/main/${file}`;
}

/**
 * Upstream WhitespaceTokenSplitter pattern (word_splitter.py), JS port.
 * Match on the ORIGINAL text (case-insensitive) so offsets index the
 * caller's string; token VALUES are lowercased afterwards.
 */
const WORD_PATTERN = new RegExp(
  String.raw`(?:https?://[^\s]+|www\.[^\s]+)` +
    String.raw`|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}` +
    String.raw`|@[a-z0-9_]+` +
    String.raw`|[\p{L}\p{N}_]+(?:[-_][\p{L}\p{N}_]+)*` +
    String.raw`|\S`,
  "giu",
);

/** Split text into words with character offsets into the original string. */
export function splitWords(text) {
  const words = [];
  WORD_PATTERN.lastIndex = 0;
  let m;
  while ((m = WORD_PATTERN.exec(text)) !== null) {
    words.push({ text: m[0].toLowerCase(), start: m.index, end: m.index + m[0].length });
  }
  return words;
}

/** Normalize text like upstream _collate_batch: ensure terminal punctuation. */
export function normalizeText(text) {
  if (!text) return ".";
  if (text.endsWith(".") || text.endsWith("!") || text.endsWith("?")) return text;
  return text + ".";
}

/**
 * Entities / structure-field schema token layout (upstream _transform_schema).
 * Parent is a single combined token. Optional descriptions are folded into
 * that parent the same way Python does:
 *   parent + " [DESCRIPTION] " + label + ": " + desc
 * Query routing still keys off [E] marker slots at schema indices 4, 6, 8, …
 *
 * Structure extraction (extract_json fields) uses the same [E] queries with
 * a different parent name (e.g. "product"). Classification ([C]) and
 * relation ([R]) queries are a different head and are not packed here.
 */
export function buildEntitiesSchemaTokens(labels, { parent = "entities", descriptions } = {}) {
  let prompt = parent;
  if (descriptions) {
    for (const label of labels) {
      const desc = descriptions[label];
      if (desc) prompt += ` [DESCRIPTION] ${label}: ${desc}`;
    }
  }
  const tokens = ["(", "[P]", prompt, "("];
  for (const label of labels) {
    tokens.push("[E]", label);
  }
  tokens.push(")", ")");
  return tokens;
}

export function buildRelationSchemaTokens(relType, { head = "head", tail = "tail", description } = {}) {
  let prompt = relType;
  if (description) prompt += ` [DESCRIPTION] ${description}`;
  return ["(", "[P]", prompt, "(", "[R]", head, "[R]", tail, ")", ")"];
}

export function buildClassificationSchemaTokens(task, labels, { descriptions } = {}) {
  let prompt = task;
  if (descriptions) {
    for (const label of labels) {
      const desc = descriptions[label];
      if (desc) prompt += ` [DESCRIPTION] ${label}: ${desc}`;
    }
  }
  const tokens = ["(", "[P]", prompt, "("];
  for (const label of labels) tokens.push("[C]", label);
  tokens.push(")", ")");
  return tokens;
}

/**
 * Pack schema + text into input_ids and routing indices
 * (upstream _format_input_with_mapping):
 * - combined = schemaTokens + [SEP_TEXT] + textWords
 * - text word i -> first subword
 * - [E] markers -> query_marker_indices
 * - [C] markers -> cls_marker_indices
 * - [R] markers -> rel_marker_indices
 */
export function packInput(tokenize, schemaTokens, textWords) {
  const combined = [...schemaTokens, "[SEP_TEXT]", ...textWords];
  const schemaLen = schemaTokens.length;

  const inputIds = [];
  const textWordFirstPositions = [];
  const queryMarkerPositions = [];
  const clsMarkerPositions = [];
  const relMarkerPositions = [];
  let lastTextWordIndex = -1;

  for (let i = 0; i < combined.length; i++) {
    const token = combined[i];
    const subwordPos = inputIds.length;
    const subTokens = tokenize(token);

    if (i < schemaLen) {
      if (token === "[E]") queryMarkerPositions.push(subwordPos);
      if (token === "[C]") clsMarkerPositions.push(subwordPos);
      if (token === "[R]") relMarkerPositions.push(subwordPos);
    } else if (i === schemaLen) {
      // [SEP_TEXT]
    } else {
      const wordIndex = i - (schemaLen + 1);
      if (wordIndex !== lastTextWordIndex) {
        textWordFirstPositions.push(subwordPos);
        lastTextWordIndex = wordIndex;
      }
    }
    for (const id of subTokens) inputIds.push(id);
  }
  return { inputIds, textWordFirstPositions, queryMarkerPositions, clsMarkerPositions, relMarkerPositions };
}

/** Numerically stable sigmoid. */
export function sigmoid(x) {
  if (x >= 0) return 1 / (1 + Math.exp(-x));
  const e = Math.exp(x);
  return e / (1 + e);
}

/**
 * Weighted interval scheduling: max-total-score non-overlapping subset.
 * Spans use half-open word indices [start, end). Flat overlap policy.
 */
export function resolveOverlapsFlat(spans) {
  if (spans.length === 0) return [];
  const sorted = [...spans].sort((a, b) => a.end - b.end || a.start - b.start);
  const n = sorted.length;
  const p = new Int32Array(n).fill(-1);
  for (let i = 0; i < n; i++) {
    let lo = 0, hi = i - 1, cand = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (sorted[mid].end <= sorted[i].start) { cand = mid; lo = mid + 1; }
      else hi = mid - 1;
    }
    p[i] = cand;
  }
  const M = new Float64Array(n);
  M[0] = sorted[0].score;
  for (let i = 1; i < n; i++) {
    const withI = sorted[i].score + (p[i] >= 0 ? M[p[i]] : 0);
    M[i] = Math.max(M[i - 1], withI);
  }
  const out = [];
  let i = n - 1;
  while (i >= 0) {
    const withI = sorted[i].score + (p[i] >= 0 ? M[p[i]] : 0);
    if (i === 0) {
      if (M[0] > 0 || n === 1) out.push(sorted[0]);
      break;
    }
    if (withI >= M[i - 1]) {
      out.push(sorted[i]);
      i = p[i];
    } else {
      i = i - 1;
    }
  }
  return out.reverse();
}

/**
 * Decode boundary marginals into character-offset entities.
 * Boundary i (0..L) sits before word i; boundary L sits after the last word.
 * Span (i, j), i < j, covers words i..j-1. Score: min(p_start[i], p_end[j]).
 *
 * @returns {Array<{label:string,text:string,start:number,end:number,score:number}>}
 *   start/end are half-open character offsets into `text`;
 *   text.slice(start, end) === entity.text.
 */
export function decodeEntities({ startLogits, endLogits, wordCount, labels, wordOffsets, text, threshold = 0.5 }) {
  const L = wordCount;
  const stride = L + 1;
  const entities = [];
  for (let q = 0; q < labels.length; q++) {
    const spans = [];
    for (let i = 0; i < L; i++) {
      const ps = sigmoid(startLogits[q * stride + i]);
      if (ps < threshold) continue;
      for (let j = i + 1; j <= L; j++) {
        const pe = sigmoid(endLogits[q * stride + j]);
        if (pe < threshold) continue;
        const score = Math.min(ps, pe);
        if (score >= threshold) spans.push({ start: i, end: j, score });
      }
    }
    for (const { start, end, score } of resolveOverlapsFlat(spans)) {
      const charStart = wordOffsets[start].start;
      const charEnd = wordOffsets[end - 1].end;
      const surface = text.slice(charStart, charEnd);
      const lead = surface.length - surface.trimStart().length;
      const stripped = surface.trim();
      if (!stripped) continue;
      entities.push({
        label: labels[q],
        text: stripped,
        start: charStart + lead,
        end: charStart + lead + stripped.length,
        score,
        wordStart: start,
        wordEnd: end,
      });
    }
  }
  return entities;
}

/**
 * Decode v2 (pair-reranker) outputs into character-offset entities.
 * Dedupes candidate slots by (start,end) — the topk export path may emit
 * duplicates — then thresholds sigmoid(pair_logits / pairTemperature) and
 * resolves overlaps per label.
 */
export function decodeEntitiesV2({
  pairIndices, pairLogits, pairValid, candidateCount,
  labels, wordOffsets, text, threshold = 0.5, pairTemperature = 1.0,
}) {
  const Q = labels.length;
  const C = candidateCount;
  const entities = [];
  for (let q = 0; q < Q; q++) {
    const seen = new Map();
    for (let c = 0; c < C; c++) {
      if (!pairValid[q * C + c]) continue;
      const s = Number(pairIndices[q * C * 2 + c * 2]);
      const e = Number(pairIndices[q * C * 2 + c * 2 + 1]);
      const p = sigmoid(pairLogits[q * C + c] / pairTemperature);
      const key = `${s},${e}`;
      if (!seen.has(key) || p > seen.get(key).p) seen.set(key, { s, e, p });
    }
    const spans = [...seen.values()]
      .filter(({ p }) => p >= threshold)
      .map(({ s, e, p }) => ({ start: s, end: e, score: p }));
    for (const { start, end, score } of resolveOverlapsFlat(spans)) {
      if (start >= end || end > wordOffsets.length) continue;
      const charStart = wordOffsets[start].start;
      const charEnd = wordOffsets[end - 1].end;
      const surface = text.slice(charStart, charEnd);
      const lead = surface.length - surface.trimStart().length;
      const stripped = surface.trim();
      if (!stripped) continue;
      entities.push({
        label: labels[q],
        text: stripped,
        start: charStart + lead,
        end: charStart + lead + stripped.length,
        score,
        wordStart: start,
        wordEnd: end,
      });
    }
  }
  return entities;
}

/**
 * GLiNER boundary runtime over an existing ORT-web session + tokenizer.
 *
 * @param {object} opts
 * @param {import("onnxruntime-web")} opts.ort        the ort module
 * @param {import("onnxruntime-web").InferenceSession} opts.session
 * @param {(token: string) => number[]} opts.tokenize subword ids for one token
 * @param {number} [opts.pairTemperature] v2 graphs: pair logit temperature (1.0 in current exports)
 */
export class GlinerBoundaryRuntime {
  constructor({ ort, session, tokenize, pairTemperature = 1.0, headsSession = null, attrsSession = null, recordsSession = null }) {
    if (!ort || !session || !tokenize) throw new Error("ort, session and tokenize are required");
    this.ort = ort;
    this.session = session;
    this.tokenize = tokenize;
    this.pairTemperature = pairTemperature;
    this.headsSession = headsSession;
    this.attrsSession = attrsSession;
    this.recordsSession = recordsSession;
    this._cache = new Map();
    this.inputNames = new Set((session.inputNames || []).map(String));
    this.outputNames = new Set((session.outputNames || []).map(String));
  }

  /** Cached subword ids for one combined token. */
  _tokenize(token) {
    let ids = this._cache.get(token);
    if (ids === undefined) {
      ids = this.tokenize(token);
      if (this._cache.size > 100_000) this._cache.clear();
      this._cache.set(token, ids);
    }
    return ids;
  }

  /**
   * Run the encoder; return everything needed to decode spans (lets callers
   * sweep thresholds without re-running inference).
   */
  async computeMarginals(text, labels, {
    maxWords = 3800, parent = "entities", descriptions,
    schemaKind = "entities", clsTask = "label", relations,
  } = {}) {
    const normalized = normalizeText(text);
    const words = splitWords(normalized).slice(0, maxWords);
    if (words.length === 0) {
      return { normalized: "", words: [], labels, startLogits: null, endLogits: null };
    }

    let schemaTokens = schemaKind === "classification"
      ? buildClassificationSchemaTokens(clsTask, labels, { descriptions })
      : buildEntitiesSchemaTokens(labels, { parent, descriptions });
    if (relations && schemaKind !== "classification") {
      for (const [relType, spec] of Object.entries(relations)) {
        schemaTokens = schemaTokens.concat(
          ["[SEP_STRUCT]"],
          buildRelationSchemaTokens(relType, { description: spec.description }),
        );
      }
    }
    const { inputIds, textWordFirstPositions, queryMarkerPositions, clsMarkerPositions, relMarkerPositions } = packInput(
      (t) => this._tokenize(t),
      schemaTokens,
      words.map((w) => w.text),
    );

    const T = inputIds.length;
    const L = words.length;
    let Q = queryMarkerPositions.length;
    let queryIdx = queryMarkerPositions;
    let queryMask = Array.from({ length: Q }, () => 1);
    if (Q === 0) {
      Q = 1;
      queryIdx = [0];
      queryMask = [0];
    } else if (schemaKind !== "classification" && Q !== labels.length) {
      throw new Error(`query routing mismatch: ${Q} markers for ${labels.length} labels`);
    }

    const feeds = {
      input_ids: new this.ort.Tensor("int64", BigInt64Array.from(inputIds.map(BigInt)), [1, T]),
      attention_mask: new this.ort.Tensor("int64", BigInt64Array.from({ length: T }, () => 1n), [1, T]),
      text_word_indices: new this.ort.Tensor("int64", BigInt64Array.from(textWordFirstPositions.map(BigInt)), [1, L]),
      text_word_mask: new this.ort.Tensor("float32", Float32Array.from({ length: L }, () => 1), [1, L]),
      query_marker_indices: new this.ort.Tensor("int64", BigInt64Array.from(queryIdx.map(BigInt)), [1, Q]),
      query_marker_mask: new this.ort.Tensor("float32", Float32Array.from(queryMask), [1, Q]),
    };

    if (this.inputNames.has("cls_marker_indices")) {
      let K = clsMarkerPositions.length;
      let clsIdx = clsMarkerPositions;
      let clsMask = Array.from({ length: K }, () => 1);
      if (K === 0) {
        K = 1;
        clsIdx = [0];
        clsMask = [0];
      }
      feeds.cls_marker_indices = new this.ort.Tensor("int64", BigInt64Array.from(clsIdx.map(BigInt)), [1, K]);
      feeds.cls_marker_mask = new this.ort.Tensor("float32", Float32Array.from(clsMask), [1, K]);
    }

    if (this.inputNames.has("rel_marker_indices")) {
      let R = relMarkerPositions.length;
      let relIdx = relMarkerPositions;
      let relMask = Array.from({ length: R }, () => 1);
      if (R === 0) {
        R = 1;
        relIdx = [0];
        relMask = [0];
      }
      feeds.rel_marker_indices = new this.ort.Tensor("int64", BigInt64Array.from(relIdx.map(BigInt)), [1, R]);
      feeds.rel_marker_mask = new this.ort.Tensor("float32", Float32Array.from(relMask), [1, R]);
    }

    const results = await this.session.run(feeds);
    const clsLogits = results.cls_logits
      ? Array.from(results.cls_logits.data).slice(0, Math.max(clsMarkerPositions.length, 0))
      : null;
    const textStates = results.text_states ?? null;
    const relRoleStates = results.rel_role_states ?? null;
    const extra = {
      clsLogits, textStates, relRoleStates,
      relRoleCount: relMarkerPositions.length,
      queryStates: results.query_states ?? null,
      candidateStates: results.candidate_states ?? null,
    };
    if (results.pair_logits) {
      return {
        normalized,
        words,
        labels,
        startLogits: results.start_logits.data,
        endLogits: results.end_logits.data,
        pairIndices: results.pair_indices.data,
        pairLogits: results.pair_logits.data,
        pairValid: results.pair_valid.data,
        candidateCount: results.pair_logits.dims[2],
        pairTemperature: this.pairTemperature,
        ...extra,
      };
    }
    return {
      normalized,
      words,
      labels,
      startLogits: results.start_logits.data,
      endLogits: results.end_logits.data,
      ...extra,
    };
  }

  async classify(text, task, labels, { threshold = 0.5, multiLabel = false, descriptions } = {}) {
    if (!this.outputNames.has("cls_logits")) {
      const err = new Error("session has no cls_logits (need v3 export)");
      err.code = "GLINER_HEAD_MISSING";
      throw err;
    }
    const marg = await this.computeMarginals(text, labels, {
      schemaKind: "classification",
      clsTask: task,
      descriptions,
    });
    const scores = (marg.clsLogits || []).map((x) => sigmoid(x));
    if (multiLabel) {
      return labels
        .map((label, i) => ({ label, score: scores[i] ?? 0 }))
        .filter((x) => x.score >= threshold);
    }
    let best = 0;
    for (let i = 1; i < scores.length; i++) if (scores[i] > scores[best]) best = i;
    return { label: labels[best], score: scores[best] ?? 0, scores: Object.fromEntries(labels.map((l, i) => [l, scores[i] ?? 0])) };
  }

  /**
   * Score proposed (head, tail, relIndex) pairs with heads.onnx.
   * rel_role_states are concatenated pairwise into directional 2H states.
   */
  async scoreRelations(marg, pairs) {
    if (!this.headsSession) {
      const err = new Error("no heads.onnx session (need v4 export)");
      err.code = "GLINER_HEAD_MISSING";
      throw err;
    }
    if (!pairs.length) return [];
    const textT = marg.textStates;
    const roleT = marg.relRoleStates;
    if (!textT || !roleT) {
      const err = new Error("session has no text_states/rel_role_states");
      err.code = "GLINER_HEAD_MISSING";
      throw err;
    }
    const L = textT.dims[1];
    const H = textT.dims[2];
    const roleCount = Math.max(marg.relRoleCount || 0, 2);
    const nRel = Math.max(1, Math.floor(roleCount / 2));
    const roleData = roleT.data;
    const relStates = new Float32Array(nRel * 2 * H);
    for (let i = 0; i < nRel; i++) {
      const a = Math.min(i * 2, roleT.dims[1] - 1);
      const b = Math.min(i * 2 + 1, roleT.dims[1] - 1);
      for (let h = 0; h < H; h++) {
        relStates[i * 2 * H + h] = roleData[a * H + h];
        relStates[i * 2 * H + H + h] = roleData[b * H + h];
      }
    }
    const P = pairs.length;
    const headStart = new BigInt64Array(P);
    const headEnd = new BigInt64Array(P);
    const tailStart = new BigInt64Array(P);
    const tailEnd = new BigInt64Array(P);
    const relIndex = new BigInt64Array(P);
    const pairMask = new Float32Array(P);
    for (let i = 0; i < P; i++) {
      const p = pairs[i];
      headStart[i] = BigInt(p.headStart);
      headEnd[i] = BigInt(p.headEnd);
      tailStart[i] = BigInt(p.tailStart);
      tailEnd[i] = BigInt(p.tailEnd);
      relIndex[i] = BigInt(p.relIndex);
      pairMask[i] = 1;
    }
    const feeds = {
      text_states: textT,
      rel_states: new this.ort.Tensor("float32", relStates, [1, nRel, 2 * H]),
      head_start: new this.ort.Tensor("int64", headStart, [1, P]),
      head_end: new this.ort.Tensor("int64", headEnd, [1, P]),
      tail_start: new this.ort.Tensor("int64", tailStart, [1, P]),
      tail_end: new this.ort.Tensor("int64", tailEnd, [1, P]),
      rel_index: new this.ort.Tensor("int64", relIndex, [1, P]),
      pair_mask: new this.ort.Tensor("float32", pairMask, [1, P]),
    };
    const out = await this.headsSession.run(feeds);
    const logits = out.rel_logits.data;
    return pairs.map((p, i) => ({ ...p, logit: logits[i], score: sigmoid(logits[i]) }));
  }

  async scoreExplicitAttributes(text, entities, attrLabels) {
    if (!this.attrsSession) return entities;
    const marg = await this.computeMarginals(text, attrLabels, { parent: "attributes" });
    const qstates = marg.queryStates || marg.textStates;
    if (!qstates || !marg.textStates) return entities;
    const L = marg.textStates.dims[1];
    const H = marg.textStates.dims[2];
    const Q = attrLabels.length;
    const C = Math.max(entities.length, 1);
    const indices = new BigInt64Array(Q * C * 2);
    for (let q = 0; q < Q; q++) {
      for (let c = 0; c < C; c++) {
        const e = entities[c] || entities[0];
        const off = (q * C + c) * 2;
        indices[off] = BigInt(e.wordStart ?? 0);
        indices[off + 1] = BigInt(e.wordEnd ?? 1);
      }
    }
    const mask = new Float32Array(L).fill(1);
    const qmask = new Float32Array(Q).fill(1);
    const feeds = {
      text_states: marg.textStates,
      text_word_mask: new this.ort.Tensor("float32", mask, [1, L]),
      query_states: marg.queryStates,
      query_marker_mask: new this.ort.Tensor("float32", qmask, [1, Q]),
      span_indices: new this.ort.Tensor("int64", indices, [1, Q, C, 2]),
    };
    if (!marg.queryStates) return entities;
    const out = await this.attrsSession.run(feeds);
    const logits = out.attr_logits.data;
    for (let c = 0; c < entities.length; c++) {
      let best = 0;
      let bestV = -1e9;
      for (let q = 0; q < Q; q++) {
        const v = logits[q * C + c];
        if (v > bestV) { bestV = v; best = q; }
      }
      entities[c].attribute = attrLabels[best];
      entities[c].attributeScore = sigmoid(bestV);
    }
    return entities;
  }

  async scoreRecords(marg, parsed, { threshold = 0.5 } = {}) {
    if (!this.recordsSession || !marg.candidateStates || !marg.queryStates) return null;
    const Q = parsed.length;
    const C = marg.candidateCount;
    const H = marg.candidateStates.dims[3];
    const cs = marg.candidateStates.data;
    const qs = marg.queryStates.data;
    const temp = marg.pairTemperature || 1;
    const anchor = Math.max(0, parsed.findIndex((p) => p.dtype === "str"));
    const instSlots = [];
    for (let c = 0; c < C; c++) {
      if (!marg.pairValid[anchor * C + c]) continue;
      const p = sigmoid(marg.pairLogits[anchor * C + c] / temp);
      if (p >= threshold) instSlots.push(c);
    }
    if (!instSlots.length) return null;
    const N = instSlots.length;
    const inst = new Float32Array(N * H);
    const instMask = new Float32Array(N).fill(1);
    for (let i = 0; i < N; i++) {
      const src = (anchor * C + instSlots[i]) * H;
      inst.set(cs.subarray(src, src + H), i * H);
    }
    const fieldQ = new Float32Array(Q * H);
    fieldQ.set(qs.subarray(0, Q * H));
    const fieldC = new Float32Array(Q * C * H);
    fieldC.set(cs.subarray(0, Q * C * H));
    const fieldMask = new Float32Array(Q * C);
    for (let q = 0; q < Q; q++) {
      for (let c = 0; c < C; c++) fieldMask[q * C + c] = marg.pairValid[q * C + c] ? 1 : 0;
    }
    const feeds = {
      inst_states: new this.ort.Tensor("float32", inst, [1, N, H]),
      inst_mask: new this.ort.Tensor("float32", instMask, [1, N]),
      field_query_states: new this.ort.Tensor("float32", fieldQ, [1, Q, H]),
      field_cand_states: new this.ort.Tensor("float32", fieldC, [1, Q, C, H]),
      field_cand_mask: new this.ort.Tensor("float32", fieldMask, [1, Q, C]),
    };
    const out = await this.recordsSession.run(feeds);
    return { instSlots, assign: out.assign_logits.data, objectLogits: out.object_logits.data };
  }

  /**
   * Extract entities. Auto-detects v2 graphs (pair outputs present) and
   * decodes from reranked candidates; v1 graphs fall back to the marginal
   * proxy decode.
   * @returns {Promise<Array<{label:string,text:string,start:number,end:number,score:number}>>}
   */
  async extract(text, labels, { threshold = 0.5, maxWords = 3800, parent, descriptions } = {}) {
    const marg = await this.computeMarginals(text, labels, { maxWords, parent, descriptions });
    if (!marg.startLogits) return [];
    if (marg.pairLogits) {
      return decodeEntitiesV2({
        pairIndices: marg.pairIndices,
        pairLogits: marg.pairLogits,
        pairValid: marg.pairValid,
        candidateCount: marg.candidateCount,
        labels: marg.labels,
        wordOffsets: marg.words,
        text: marg.normalized,
        threshold,
        pairTemperature: marg.pairTemperature,
      });
    }
    return decodeEntities({
      startLogits: marg.startLogits,
      endLogits: marg.endLogits,
      wordCount: marg.words.length,
      labels: marg.labels,
      wordOffsets: marg.words,
      text: marg.normalized,
      threshold,
    });
  }

  /**
   * Extract entities from a v2 graph explicitly. Throws if the session does
   * not expose pair outputs (use extract() for auto-detection).
   */
  async extractV2(text, labels, { threshold = 0.5, maxWords = 3800, parent, descriptions } = {}) {
    const marg = await this.computeMarginals(text, labels, { maxWords, parent, descriptions });
    if (!marg.pairLogits) {
      throw new Error("session is not a v2 export (no pair_logits output)");
    }
    return decodeEntitiesV2({
      pairIndices: marg.pairIndices,
      pairLogits: marg.pairLogits,
      pairValid: marg.pairValid,
      candidateCount: marg.candidateCount,
      labels: marg.labels,
      wordOffsets: marg.words,
      text: marg.normalized,
      threshold,
      pairTemperature: marg.pairTemperature,
    });
  }

  /**
   * Host-side long-document scan: overlapping word chunks, remap to the
   * original (normalized) string, merge duplicate spans per label.
   * Mirrors extract_entities_long in the Python library.
   */
  async extractLong(text, labels, {
    threshold = 0.5,
    chunkSize = 384,
    chunkOverlap = 64,
    maxWords = 3800,
    parent,
    descriptions,
  } = {}) {
    const normalized = normalizeText(text);
    const words = splitWords(normalized).slice(0, maxWords);
    if (words.length <= chunkSize) {
      return this.extract(normalized, labels, { threshold, maxWords, parent, descriptions });
    }
    const all = [];
    for (let i = 0; i < words.length; ) {
      const slice = words.slice(i, Math.min(i + chunkSize, words.length));
      const origin = slice[0].start;
      const chunk = normalized.slice(origin, slice[slice.length - 1].end);
      const ents = await this.extract(chunk, labels, {
        threshold,
        maxWords: chunkSize + 8,
        parent,
        descriptions,
      });
      for (const e of ents) {
        const start = e.start + origin;
        const end = e.end + origin;
        all.push({ ...e, start, end, text: normalized.slice(start, end) });
      }
      if (i + chunkSize >= words.length) break;
      i += Math.max(1, chunkSize - chunkOverlap);
    }
    return mergeOverlappingByLabel(all);
  }
}

/** Keep the highest-scoring span when two same-label spans overlap. */
export function mergeOverlappingByLabel(ents) {
  const byLabel = new Map();
  for (const e of ents) {
    const list = byLabel.get(e.label) ?? [];
    list.push(e);
    byLabel.set(e.label, list);
  }
  const out = [];
  for (const group of byLabel.values()) {
    group.sort((a, b) => b.score - a.score);
    const kept = [];
    for (const e of group) {
      if (kept.some((k) => e.start < k.end && k.start < e.end)) continue;
      kept.push(e);
    }
    out.push(...kept);
  }
  return out.sort((a, b) => a.start - b.start || b.end - a.end);
}

/**
 * Download a model with progress, reporting bytes as they stream.
 * @returns {Promise<Uint8Array>}
 */
export async function downloadModel(url, { onProgress } = {}) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`download failed: ${res.status} ${url}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (onProgress) onProgress(received, total);
  }
  const out = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}
