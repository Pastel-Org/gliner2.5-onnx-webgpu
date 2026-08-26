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

The `gliner2.5-{small,base,multi}-v1-onnx` exports are **boundary-architecture** ONNX graphs: the graph runs the DeBERTa encoder over packed `input_ids` and emits per-query boundary marginals `start_logits` / `end_logits` `[B, Q, L+1]`. Everything else — word splitting, entity-schema packing, subword routing, span decode — is host-side and implemented in this repo (~330 lines of dependency-free JavaScript, clean-room reimplemented from the [fastino-ai/GLiNER2](https://github.com/fastino-ai/GLiNER2) Apache-2.0 reference):

```text
words ──▶ schema pack: ( [P] entities ( [E] label … ) ) [SEP_TEXT] words
      ──▶ ONNX (int64 feeds, first-subword routing, [E]-marker queries)
      ──▶ decode: sigmoid marginals → threshold → interval-scheduling spans
      ──▶ char-offset entities {label, text, start, end, score}
```

Verified: reproduces the upstream model-card example with exact character offsets (Apple [0,5), "Tim Cook" [10,18), "iPhone 15" [29,38), Cupertino [42,51)).

## Files

- `index.html` — the runnable demo (vanilla JS, CDN imports)
- `src/gliner-boundary.mjs` — the runtime: packing, routing, decode, model registry, streaming downloader

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

- **Decode path**: these exports expose boundary marginals only — the upstream pair reranker is not part of the graph. Span score here is `min(sigmoid(start), sigmoid(end))`, a marginal proxy. Raise the threshold (0.6–0.7) for better precision; that is what our CPU baseline measurements recommend.
- **Baseline numbers** (synthetic 26-sample suite, 8 labels, EN/FR/DE): span-F1 0.86 (small), 0.92 (base), 0.92 (multi) at tuned thresholds, recall 0.94–0.98. Measured with the Node/CPU path over the same `src` protocol code.
- **WebGPU caveat**: int64 input tensors are required by the graph. Some browsers lack int64 support in WebGPU kernels; the demo falls back to WASM when no WebGPU adapter is present, matching the model card's guidance.
- Tokenizer files come from the model repos via transformers.js 4.2.0 (pinned), ONNX Runtime Web 1.26.0 (pinned).

## Credits & license

- GLiNER2/GLiNER2.5 by Fastino (fastino.ai) — models and Python reference under Apache-2.0.
- This repo: MIT. Protocol reimplementation written from scratch; nothing vendored.
