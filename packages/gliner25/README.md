# @gliner25/adapter

GLiNER2.5 (boundary architecture) in the browser: ONNX runtime, Weightlift
model definition, and a Python `AutoExtractor`-shaped JS API. Zero runtime
dependencies — you pass `onnxruntime-web` and a tokenizer loader yourself.

Models: [small](https://huggingface.co/nicolasembleton/gliner2.5-small-v1-onnx) ·
[base](https://huggingface.co/nicolasembleton/gliner2.5-base-v1-onnx) ·
[multi](https://huggingface.co/nicolasembleton/gliner2.5-multi-v1-onnx)
(revision 4 graphs: entity path + classifier + cached states + `heads.onnx` relation scorer).

## Install

```bash
npm install @gliner25/adapter
```

## Weightlift

```js
import { ModelManager } from "weightlift";
import { glinerModel } from "@gliner25/adapter/weightlift";

const models = new ModelManager({
  models: { gliner: glinerModel({ size: "small", ort, tokenizerFromPretrained }) },
});
const gliner = await models.load("gliner");
```

## Direct

```js
import { Gliner25 } from "@gliner25/adapter";

const gliner = new Gliner25({ ort, session, tokenize });
await gliner.extract_entities(text, ["person", "organization"], { include_spans: true });
await gliner.classify_text(text, "sentiment", ["positive", "negative", "neutral"]);
await gliner.extract_json(text, { product: ["name::str", "price", "features"] });
await gliner.extract_relations(text, { works_for: { head: ["person"], tail: ["organization"] } });
```

## API surface

| Method | Graph |
|---|---|
| `extract_entities` / `extract_entities_long` | v2+ |
| `extract_json` (field-as-label) | v2+ |
| `classify_text` | v3 |
| `extract_relations` (JointIE beam in JS) | v4 `heads.onnx` |
| `extract_json(..., { records: true })` | host zip of list fields, not RecordHead |

Host-side decode, overlap resolution, long-document chunking, and JSON
field typing are JavaScript here; the ONNX graph does neural scoring only.
Beam search and schema constraints stay on the host by design.

Protocol clean-room reimplemented from
[fastino-ai/GLiNER2](https://github.com/fastino-ai/GLiNER2) (Apache-2.0).
Nothing vendored. MIT. © Pastel-Cloud OÜ.
