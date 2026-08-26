/**
 * Python AutoExtractor-shaped JS API over the GLiNER2.5 ONNX graph.
 *
 * Neural scoring is the session. Packing, overlap, long-document merge,
 * field-as-label JSON, and (later) classification / JointIE beam live here.
 */

import { GlinerBoundaryRuntime, GLINER_MODELS, hfFileUrl, downloadModel } from "./gliner-boundary.mjs";

export { GLINER_MODELS, hfFileUrl, downloadModel };

function toEntityMap(entities, { includeConfidence = false, includeSpans = false, asList = true } = {}) {
  const out = {};
  for (const e of entities) {
    let item;
    if (includeConfidence || includeSpans) {
      item = { text: e.text };
      if (includeConfidence) item.confidence = e.score;
      if (includeSpans) {
        item.start = e.start;
        item.end = e.end;
      }
    } else {
      item = e.text;
    }
    (out[e.label] ??= []).push(item);
  }
  if (!asList) {
    for (const k of Object.keys(out)) {
      out[k] = out[k][0] ?? null;
    }
  }
  return out;
}

function parseFieldSpec(spec) {
  // "name::str::Full product name" | "features::list" | "name"
  const parts = String(spec).split("::");
  const name = parts[0];
  let dtype = "list";
  let description;
  for (const p of parts.slice(1)) {
    if (p === "str" || p === "list") dtype = p;
    else if (p.startsWith("[") && p.endsWith("]")) continue; // choices: host filter later
    else description = p;
  }
  return { name, dtype, description };
}

export class Gliner25 {
  /**
   * @param {{ ort: any, session: any, tokenize: (t: string) => number[], pairTemperature?: number }} opts
   */
  constructor({ ort, session, tokenize, pairTemperature = 1.0, graph = "v2" }) {
    this.rt = new GlinerBoundaryRuntime({ ort, session, tokenize, pairTemperature });
    this.session = session;
    this.graph = graph;
    this.outputs = new Set((session.outputNames || []).map(String));
  }

  get hasClassifier() {
    return this.outputs.has("cls_logits");
  }

  get hasCachedStates() {
    return this.outputs.has("text_states");
  }

  async extract(text, labels, opts = {}) {
    return this.rt.extract(text, labels, opts);
  }

  async extractLong(text, labels, opts = {}) {
    return this.rt.extractLong(text, labels, opts);
  }

  async extract_entities(text, labels, {
    threshold = 0.5,
    include_confidence = false,
    include_spans = false,
    descriptions,
    parent = "entities",
    maxWords = 3800,
  } = {}) {
    const entities = await this.rt.extract(text, labels, {
      threshold, descriptions, parent, maxWords,
    });
    return {
      entities: toEntityMap(entities, {
        includeConfidence: include_confidence,
        includeSpans: include_spans,
      }),
    };
  }

  async extract_entities_long(text, labels, {
    threshold = 0.5,
    chunk_size = 384,
    chunk_overlap = 64,
    include_confidence = false,
    include_spans = false,
    descriptions,
    parent = "entities",
  } = {}) {
    const entities = await this.rt.extractLong(text, labels, {
      threshold,
      chunkSize: chunk_size,
      chunkOverlap: chunk_overlap,
      descriptions,
      parent,
    });
    return {
      entities: toEntityMap(entities, {
        includeConfidence: include_confidence,
        includeSpans: include_spans,
      }),
    };
  }

  /**
   * Field-as-label JSON. Same [E] queries under `parent`. Not record-mode
   * instance formation (that needs v5). Matches extract_json on single-record
   * README examples.
   */
  async extract_json(text, structures, {
    threshold = 0.5,
    include_confidence = false,
    include_spans = false,
    descriptions = {},
  } = {}) {
    const result = {};
    for (const [parent, fields] of Object.entries(structures)) {
      const parsed = fields.map(parseFieldSpec);
      const labels = parsed.map((p) => p.name);
      const desc = { ...descriptions };
      for (const p of parsed) if (p.description) desc[p.name] = p.description;
      const entities = await this.rt.extract(text, labels, {
        threshold, parent, descriptions: desc,
      });
      const rec = {};
      for (const p of parsed) {
        const hits = entities.filter((e) => e.label === p.name).sort((a, b) => b.score - a.score);
        if (p.dtype === "str") {
          const h = hits[0];
          rec[p.name] = h
            ? (include_confidence || include_spans
              ? { text: h.text, ...(include_confidence ? { confidence: h.score } : {}), ...(include_spans ? { start: h.start, end: h.end } : {}) }
              : h.text)
            : null;
        } else {
          rec[p.name] = hits.map((h) => (
            include_confidence || include_spans
              ? { text: h.text, ...(include_confidence ? { confidence: h.score } : {}), ...(include_spans ? { start: h.start, end: h.end } : {}) }
              : h.text
          ));
        }
      }
      result[parent] = [rec];
    }
    return result;
  }

  async classify_text(_text, _task) {
    if (!this.hasClassifier) {
      const err = new Error("classify_text needs a v3 graph (cls_logits). This session is entity-only.");
      err.code = "GLINER_HEAD_MISSING";
      err.head = "classification";
      throw err;
    }
    throw new Error("classify_text decode not wired yet");
  }

  async extract_relations(_text, _types) {
    const err = new Error("extract_relations / JointIE needs a v4 graph (relation_scorer). Beam stays in JS.");
    err.code = "GLINER_HEAD_MISSING";
    err.head = "relations";
    throw err;
  }
}
