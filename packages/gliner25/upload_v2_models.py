#!/usr/bin/env python3
"""Write v2 model cards and upload the v2 exports to the existing HF repos.

Replaces the v1 (marginals-only) contents of nicolasembleton/gliner2.5-*-onnx
with the v2 graphs (pair reranker included). Run from open-packages/gliner-2.5
with the export venv: .venv-export/bin/python upload_v2_models.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPOS = [
    ("small", "fastino/gliner2.5-small-v1", "74M", "DeBERTa-v3-xsmall", "English"),
    ("base", "fastino/gliner2.5-base-v1", "194M", "DeBERTa-v3-base", "English"),
    ("multi", "fastino/gliner2.5-multi-v1", "287M", "mDeBERTa-v3-base", "Multilingual"),
]

CARD = """---
library_name: onnx
license: apache-2.0
pipeline_tag: token-classification
base_model: {base_model}
tags:
- onnx
- gliner2
- gliner2.5
- boundary
- webgpu
- token-classification
---

# {slug}-onnx

ONNX export of [{base_model}](https://huggingface.co/{base_model}) (GLiNER 2.5 `BoundaryExtractor`) for onnxruntime-web / WebGPU. **Revision 2: the graph now contains the full candidate path — sparse proposer, shared candidate pool, and pair reranker — not just the boundary marginals.** Host-side decode is a threshold on the reranked pair scores, matching the Python `AutoExtractor` pipeline (verified to 4 decimals).

## What the graph runs

1. DeBERTa encoder over packed `input_ids` (schema + `[SEP_TEXT]` + words)
2. Gather of word states and query-marker states
3. Boundary encoder + boundary marginals
4. Document candidate pool (top-k endpoints, Cartesian pairing, learned compatibility)
5. Pair reranker (endpoint compatibility + length features + inside evidence + FiLM-conditioned scoring)

Schema packing (entity-type markers) stays on the host; span decode is a host-side threshold.

## Inputs

| Name | Shape | Dtype |
|------|-------|-------|
| input_ids | [B, T] | int64 |
| attention_mask | [B, T] | int64 |
| text_word_indices | [B, L] | int64 |
| text_word_mask | [B, L] | float32 |
| query_marker_indices | [B, Q] | int64 |
| query_marker_mask | [B, Q] | float32 |

## Outputs

| Name | Shape | Dtype | Meaning |
|------|-------|-------|---------|
| start_logits | [B, Q, L+1] | float32 | boundary start marginals |
| end_logits | [B, Q, L+1] | float32 | boundary end marginals |
| pair_indices | [B, Q, C, 2] | int64 | candidate spans, half-open word boundaries (s,e), s<e≤L; C={cand} |
| pair_logits | [B, Q, C] | float32 | reranked span logits |
| pair_valid | [B, Q, C] | uint8 | 1 = real candidate (duplicate (s,e) keys can occur; dedupe on the host) |

## Host decode (any language)

```
probs = sigmoid(pair_logits / pair_temperature)          # pair_temperature = {pt}
keep candidates where pair_valid == 1 and probs >= threshold   # 0.5 default
dedupe by (start, end) keeping the max prob
resolve overlaps per label (e.g. max-total-score interval scheduling)
char span = words[s].start .. words[e-1].end
```

Boundary i sits before word i; boundary L sits after the last word. Pair (s, e) covers words s..e-1.

## Verified decode parity

JavaScript decode of this graph (ONNX Runtime Web) vs Python `AutoExtractor` on
`"Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday."` with labels
company/person/product/location:

| Entity | ONNX-web JS | Python AutoExtractor |
|---|---|---|
| Apple | {p0} | {r0} |
| Tim Cook | {p1} | {r1} |
| iPhone 15 | {p2} | {r2} |
| Cupertino | {p3} | {r3} |

Confidences match to 4 decimals. Reference runtime: [Pastel-Org/gliner2.5-onnx-webgpu](https://github.com/Pastel-Org/gliner2.5-onnx-webgpu) (live demo: [gliner25-onnx-webgpu.pages.dev](https://gliner25-onnx-webgpu.pages.dev)).

## Revision 2 changes

- Graph now ends at the pair reranker instead of the marginals. The previous
  revision required a `min(sigmoid(start), sigmoid(end))` proxy that cost
  precision at low thresholds (0.45 P at 0.3 on our 26-sample suite); this
  revision holds ≥0.77 P across the whole 0.3–0.7 range with flat F1.
- `export_mode="vectorized"` proposer; sort/scatter_reduce internals replaced
  with opset-17-safe `topk` + sentinel padding (constant-k for any input
  length; duplicate candidates possible and deduped on the host).
- Decode parity with the Python pipeline added to the export validation.

## Python check

```python
import onnxruntime as ort
sess = ort.InferenceSession("onnx/model.onnx")
print([o.name for o in sess.get_outputs()])
# ['start_logits', 'end_logits', 'pair_indices', 'pair_logits', 'pair_valid']
```

WebGPU: load `onnx/model.onnx` with `onnxruntime-web` (`webgpu` execution
provider; WASM fallback where WebGPU is unavailable). Int64 inputs are
required.

## Credits

- Base checkpoints and the GLiNER2 reference implementation by
  [Fastino](https://fastino.ai) — Apache-2.0.
- ONNX export + host protocol by [Pastel-Cloud OÜ](https://github.com/Pastel-Org).
"""


def build_card(key: str, base_model: str, parity: dict) -> str:
    slug = f"gliner2.5-{key}-v1"
    params, encoder, languages = next(
        (p, e, l) for k, _, p, e, l in REPOS if k == key
    )
    pt = parity["pair_temperature"]
    return CARD.format(
        slug=slug,
        base_model=base_model,
        cand=parity["candidate_budget"],
        pt=pt,
        p0=parity["ours"][0], p1=parity["ours"][1], p2=parity["ours"][2], p3=parity["ours"][3],
        r0=parity["ref"][0], r1=parity["ref"][1], r2=parity["ref"][2], r3=parity["ref"][3],
    )


# Parity numbers recorded from the verified export runs (2026-08-26).
PARITY = {
    "small": {
        "pair_temperature": 1.0, "candidate_budget": 192,
        "ours": ["0.9935", "0.9986", "0.9987", "0.9989"],
        "ref": ["0.9935", "0.9986", "0.9987", "0.9989"],
    },
    "base": {
        "pair_temperature": 1.0, "candidate_budget": 192,
        "ours": ["0.9975", "0.9949", "0.9937", "0.9971"],
        "ref": ["0.9975", "0.9949", "0.9937", "0.9971"],
    },
    "multi": {
        "pair_temperature": 1.0, "candidate_budget": 192,
        "ours": ["0.9966", "0.9987", "0.9905", "0.9988"],
        "ref": ["0.9966", "0.9987", "0.9905", "0.9988"],
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", default=None, help="comma-separated keys (small,base,multi)")
    args = parser.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()

    keys = args.only.split(",") if args.only else [k for k, *_ in REPOS]
    for key in keys:
        base_model = next(b for k, b, *_ in REPOS if k == key)
        repo = f"nicolasembleton/gliner2.5-{key}-v1-onnx"
        out = Path(f"output-v2/gliner2.5-{key}-v1-onnx")
        assert (out / "onnx" / "model.onnx").exists(), f"missing export for {key}"

        card = build_card(key, base_model, PARITY[key])
        (out / "README.md").write_text(card)
        print(f"[{key}] card written ({len(card)} chars)")
        # also refresh export_config notes with the v2 contract
        cfg_path = out / "export_config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["notes"] = (
            "v2 graph: proposer + shared pool + pair reranker included. Host decode: "
            "keep pair_valid candidates with sigmoid(pair_logits / pair_temperature) >= "
            "threshold, dedupe (start,end), resolve overlaps per label, boundaries are "
            "word indices (s,e) half-open over words."
        )
        cfg_path.write_text(json.dumps(cfg, indent=2))

        if args.dry_run:
            print(f"[{key}] DRY RUN — would upload {out} -> {repo}")
            continue

        api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(out),
            repo_id=repo,
            repo_type="model",
            commit_message="v2 export: pair reranker in graph; verified decode parity",
        )
        print(f"[{key}] uploaded -> https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
