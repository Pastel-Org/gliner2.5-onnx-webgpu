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

The `gliner2.5-{small,base,multi}-v1-onnx` exports are **boundary-architecture** ONNX graphs that include the full candidate path: the DeBERTa encoder, the sparse proposer, the shared document candidate pool, and the **pair reranker**. The graph emits boundary marginals `start_logits` / `end_logits` `[B, Q, L+1]` plus reranked candidates `pair_indices [B,Q,C,2]`, `pair_logits [B,Q,C]`, `pair_valid [B,Q,C]` (C = 192). Everything else — word splitting, entity-schema packing, subword routing, span decode — is host-side and implemented in this repo (~500 lines of dependency-free JavaScript, clean-room reimplemented from the [fastino-ai/GLiNER2](https://github.com/fastino-ai/GLiNER2) Apache-2.0 reference):

```text
words ──▶ schema pack: ( [P] parent ( [E] label … ) ) [SEP_TEXT] words
      ──▶ ONNX (int64 feeds, first-subword routing, [E]-marker queries)
      ──▶ decode: sigmoid(pair_logits / pair_temperature) → threshold
              → dedupe (start,end) → interval-scheduling spans
      ──▶ char-offset entities {label, text, start, end, score}
```

`parent` defaults to `entities`. Structure-field extraction uses the same `[E]` queries with a different parent (`product`, `contact`, …). Optional `[DESCRIPTION]` strings fold into the parent token the way Python `_transform_schema` does.

Verified: JavaScript decode of these graphs matches the Python `AutoExtractor` pipeline's confidences to 4 decimals on all three checkpoints (recorded per model in each HF model card).

**Not in this graph:** document classification (`[C]`), constrained classification, JointIE / `extract_relations` (`[R]`), record-mode JSON instance formation, span-attribute heads. Those live in the Python library. The demo page still loads every README example; blocked tasks fall back to the entity layer and say so.

## Files

- `index.html` — the runnable demo (vanilla JS, CDN imports, Fastino README/tutorial/blog examples)
- `src/gliner-boundary.mjs` — packing, routing, v2 decode, long-document chunking
- `src/demo-presets.mjs` — example texts and labels

## Models

| Model | Params | Size | Languages |
|---|---|---|---|
| [gliner2.5-small-v1-onnx](https://huggingface.co/nicolasembleton/gliner2.5-small-v1-onnx) | 74M | 272 MB | English |
| [gliner2.5-base-v1-onnx](https://huggingface.co/nicolasembleton/gliner2.5-base-v1-onnx) | 194M | 705 MB | English |
| [gliner2.5-multi-v1-onnx](https://huggingface.co/nicolasembleton/gliner2.5-multi-v1-onnx) | 287M | 1.1 GB | Multilingual |

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
- **WebGPU caveat**: int64 input tensors are required by the graph. Some browsers lack int64 support in WebGPU kernels; the demo falls back to WASM when no WebGPU adapter is present.
- Tokenizer files come from the model repos via transformers.js 4.2.0 (pinned), ONNX Runtime Web 1.26.0 (pinned).

## Credits & license

- GLiNER2/GLiNER2.5 by Fastino (fastino.ai) — models and Python reference under Apache-2.0.
- This repo: MIT. Protocol reimplementation written from scratch; nothing vendored.
