/**
 * Python AutoExtractor-shaped JS API over the GLiNER2.5 ONNX graph.
 *
 * Neural scoring is the session. Packing, overlap, long-document merge,
 * field-as-label JSON, and (later) classification / JointIE beam live here.
 */

import { GlinerBoundaryRuntime, GLINER_MODELS, hfFileUrl, downloadModel, decodeEntitiesV2, decodeEntities } from "./gliner-boundary.mjs";
import { proposeRelationPairs, beamSearchRelations, zipRecords } from "./joint-ie.mjs";

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
  constructor({ ort, session, tokenize, pairTemperature = 1.0, graph = "v2", headsSession = null }) {
    this.rt = new GlinerBoundaryRuntime({ ort, session, tokenize, pairTemperature, headsSession });
    this.session = session;
    this.headsSession = headsSession;
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
    records = false,
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
      const toItem = (h) => (
        include_confidence || include_spans
          ? { text: h.text, ...(include_confidence ? { confidence: h.score } : {}), ...(include_spans ? { start: h.start, end: h.end } : {}) }
          : h.text
      );
      if (records) {
        const fieldHits = {};
        const strFields = [];
        for (const p of parsed) {
          const hits = entities.filter((e) => e.label === p.name).sort((a, b) => a.start - b.start);
          fieldHits[p.name] = hits.map(toItem);
          if (p.dtype === "str") strFields.push(p.name);
        }
        Object.assign(result, zipRecords(fieldHits, { strFields, parent }));
        continue;
      }
      const rec = {};
      for (const p of parsed) {
        const hits = entities.filter((e) => e.label === p.name).sort((a, b) => b.score - a.score);
        if (p.dtype === "str") {
          rec[p.name] = hits[0] ? toItem(hits[0]) : null;
        } else {
          rec[p.name] = hits.map(toItem);
        }
      }
      result[parent] = [rec];
    }
    return result;
  }

  async classify_text(text, taskOrMap, labelsMaybe, opts = {}) {
    if (!this.hasClassifier) {
      const err = new Error("classify_text needs a v3 graph (cls_logits). This session is entity-only.");
      err.code = "GLINER_HEAD_MISSING";
      err.head = "classification";
      throw err;
    }
    let tasks;
    if (typeof taskOrMap === "string") {
      tasks = { [taskOrMap]: { labels: labelsMaybe, ...opts } };
    } else {
      tasks = {};
      for (const [name, spec] of Object.entries(taskOrMap)) {
        tasks[name] = Array.isArray(spec) ? { labels: spec } : spec;
      }
    }
    const out = {};
    for (const [name, spec] of Object.entries(tasks)) {
      const labels = spec.labels || spec;
      const multi = Boolean(spec.multi_label ?? spec.multiLabel);
      const threshold = spec.threshold ?? opts.threshold ?? 0.5;
      out[name] = await this.rt.classify(text, name, labels, {
        threshold,
        multiLabel: multi,
        descriptions: spec.descriptions,
      });
    }
    return out;
  }

  async extract_relations(text, types, {
    threshold = 0.5,
    labels,
    include_confidence = true,
  } = {}) {
    if (!this.rt.headsSession) {
      const err = new Error("extract_relations needs v4 heads.onnx (relation scorer). Beam stays in JS.");
      err.code = "GLINER_HEAD_MISSING";
      err.head = "relations";
      throw err;
    }
    const entityLabels = labels || [...new Set(
      Object.values(types).flatMap((s) => [...(s.head || s.heads || []), ...(s.tail || s.tails || [])]),
    )];
    const entities = await this.rt.extract(text, entityLabels, { threshold });
    const marg = await this.rt.computeMarginals(text, entityLabels, { relations: types });
    const pairs = proposeRelationPairs(entities, types);
    const scored = await this.rt.scoreRelations(marg, pairs);
    const joint = beamSearchRelations(entities, scored, { threshold });
    return {
      entities: toEntityMap(entities, { includeConfidence: include_confidence, includeSpans: true }),
      relations: joint.relations,
      score: joint.score,
    };
  }
}
