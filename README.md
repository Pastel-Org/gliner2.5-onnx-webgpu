# gliner2.5-onnx-webgpu

**Live demo: https://gliner25-onnx-webgpu.pages.dev**

Zero-shot named-entity recognition with [GLiNER2.5](https://huggingface.co/collections/nicolasembleton/gliner25) (boundary architecture), fully in the browser via ONNX Runtime Web — WebGPU when available, WASM fallback. No server, no keys, no build step.

To run locally instead, serve this directory and load `index.html`. Nothing is bundled:

```bash
cd gliner-2.5
python3 -m http.server 8080   # or: npx serve .
# → http://localhost:8080
```

Requires a browser with `navigator.gpu` (Chrome/Edge 113+, Safari 26+ tech preview, etc.) for the WebGPU path; otherwise the demo transparently falls back to WASM. The model (one of three sizes) downloads from Hugging Face on first use and is cached by the browser.

## What this is

The `gliner2.5-{small,base,multi}-v1-onnx` exports are **revision 5** boundary
graphs: DeBERTa encoder, sparse proposer, pair reranker, classification MLP,
cached `text_states`, `candidate_states`, plus tiny `heads.onnx` / `records.onnx`.
One encoder pass. Host packing and decode live here
(clean-room from [fastino-ai/GLiNER2](https://github.com/fastino-ai/GLiNER2), Apache-2.0).

```text
words ──▶ schema pack: ( [P] parent ( [E]|[C] label … ) ) [SEP_TEXT] words
      ──▶ ONNX
      ──▶ entities: sigmoid(pair_logits) → threshold → spans
      ──▶ classify: sigmoid(cls_logits) → argmax / threshold
```

Load via [weightlift](https://weightlift.dev) `ModelManager` + `glinerModel()`
in this package. Demo pins `weightlift@0.2.1` on jsDelivr (same version as
`package.json`) because Cloudflare Pages direct-upload does not honor
`.gitignore`; do not put `node_modules` in this directory.

**Still not in the graph:** JointIE / `extract_relations` (`[R]`), record-mode
JSON, constrained `implies`/`excludes` (those last are JS when we add them),
span-attribute rescoring. `text_states` is exported so those heads do not need
a second encoder later.

## Files

- `index.html` — demo (weightlift ModelManager + Fastino templates)
- `src/gliner-boundary.mjs` — packing, v2/v3 decode, classify, long-doc
- `src/api.mjs` — AutoExtractor-shaped JS API
- `src/weightlift.mjs` — `glinerModel()` adapter
- `src/demo-presets.mjs` — Fastino README/tutorial/blog examples
- `PROGRAM.md` — remaining heads (relations, records)
- `export_v3_cls.py` / `upload_v3_models.py`

## Models

| Model | Params | ONNX | Languages |
|---|---|---|---|
| [gliner2.5-small-v1-onnx](https://huggingface.co/nicolasembleton/gliner2.5-small-v1-onnx) | 74M | 288 MB | English |
| [gliner2.5-base-v1-onnx](https://huggingface.co/nicolasembleton/gliner2.5-base-v1-onnx) | 194M | 746 MB | English |
| [gliner2.5-multi-v1-onnx](https://huggingface.co/nicolasembleton/gliner2.5-multi-v1-onnx) | 287M | 1.12 GB | Multilingual |

## Using the runtime in your own page

```js
import { GLINER_MODELS, hfFileUrl, GlinerBoundaryRuntime, downloadModel } from "./src/gliner-boundary.mjs";
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/ort.webgpu.bundle.min.mjs";

const repo = GLINER_MODELS.small.repo;
const bytes = await downloadModel(hfFileUrl(repo, "onnx/model.onnx"));
const session = await ort.InferenceSession.create(bytes.buffer, { executionProviders: ["webgpu"] });

const { AutoTokenizer, env } = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0");
env.allowLocalModels = false;
const tokenizer = await AutoTokenizer.from_pretrained(repo);
const tokenize = (t) => Array.from(tokenizer.encode(t, { add_special_tokens: false }).ids ?? []);

const rt = new GlinerBoundaryRuntime({ ort, session, tokenize });
const entities = await rt.extract(
  "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday.",
  ["company", "person", "product", "location"],
  { threshold: 0.5 },
);
```

## Notes and honest caveats

- **Decode path**: revision 2 of the exports contains the pair reranker, so span scores are the same reranked logits the Python `AutoExtractor` pipeline produces (4-decimal parity verified). The first revision exposed boundary marginals only and required a `min(sigmoid(start), sigmoid(end))` proxy; that cost precision at low thresholds. With v2, the threshold curve is flat: 0.3–0.7 all within ~0.02 F1 of peak.
- **Eval numbers** (synthetic 26-sample suite, 8 labels, EN/FR/DE, 103 gold spans, CPU): peak partial span-F1 0.86 (small), 0.92 (base), 0.91 (multi). Base is the strongest overall: 0.92 F1 at recall 0.97. Precision at threshold 0.3 improved from 0.45 to 0.77 (small) going from the marginal proxy to the reranker.
- **Candidate pool recall trade-off**: spans must survive a fixed 192-candidate pool before reranking, instead of exhaustive marginal enumeration. Recall at aggressive thresholds dips 3–6 points vs the v1 exhaustive decode; peak-F1 recall is unchanged.
- **WebGPU length**: Python `classify_text` is one pass (max_len 4096 words).
  On this host, one-shot **small** finished 4096 words; **base** and **multi**
  died between 3500 and 3600 words. Long docs: `classify_text_long` /
  `extract_entities_long` (384/64 windows). A 4096-word file is 13 runs of
  384 words, not one 4096-word encode.
- **WebGPU int64**: some browsers lack int64 in WebGPU kernels; the demo
  falls back to WASM when no adapter is present.
- Tokenizer files come from the model repos via transformers.js 4.2.0 (pinned), ONNX Runtime Web 1.26.0 (pinned).

## Credits & license

- GLiNER2/GLiNER2.5 by Fastino (fastino.ai) — models and Python reference under Apache-2.0.
- This repo: MIT. Protocol reimplementation written from scratch; nothing vendored.
