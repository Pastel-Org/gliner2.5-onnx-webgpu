#!/usr/bin/env python3
"""Patch Hub model cards to match the shipped graphs (README only)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path("/Users/nemb/projects/pastel-org/pastel-evals/models/gliner25-v5-export")
REPOS = [
    ("small", "fastino/gliner2.5-small-v1"),
    ("base", "fastino/gliner2.5-base-v1"),
    ("multi", "fastino/gliner2.5-multi-v1"),
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
- text-classification
- relation-extraction
---

# {slug}-onnx

ONNX export of [{base_model}](https://huggingface.co/{base_model}) (GLiNER 2.5
`BoundaryExtractor`) for onnxruntime-web / WebGPU. Revision 5 graphs plus
`attrs.onnx` (`score_explicit_spans`).

One encoder pass. JointIE beam, classification `implies`/`excludes`, record
assignment, and long-document chunking stay in JavaScript.

Host: [Pastel-Org/gliner2.5-onnx-webgpu](https://github.com/Pastel-Org/gliner2.5-onnx-webgpu)
(live: [gliner25-onnx-webgpu.pages.dev](https://gliner25-onnx-webgpu.pages.dev)).

## Files

| File | Role | Size |
|------|------|------|
| `onnx/model.onnx` | Encoder + entity pair path + classifier + `text_states` + `candidate_states` | {model_mb:.1f} MB |
| `onnx/heads.onnx` | `SparseRelationScorer` only (no encoder) | {heads_mb:.1f} MB |
| `onnx/records.onnx` | RecordHead assignment (inst/field/cand projections + null column) | {records_mb:.2f} MB |
| `onnx/attrs.onnx` | `score_explicit_spans` (dynamo; pad 512/8/16) | {attrs_mb:.2f} MB |

## attrs.onnx

Dynamo export of `head.score_explicit_spans`. Traced at 512 words, 8 attribute
queries, 16 spans. The JS host crops a 512-word window around the mention; that
pad is not a document-length cap. ORT vs torch RMSE ~1e-6 (small) / ~2e-6
(base, multi). Overlay lookup is the fallback if this graph fails to load.

## Decode notes (JS host, not this graph)

Attribute labels are packed with entity labels in one `[E]` block (sorted).
Single-label attributes use softmax. ONNX `pair_valid` is all-true; the host
drops `start >= end` slots before overlap.

Python `max_len` is 4096 words. On WebGPU, one-shot classify/NER on **small**
handled 4096 words; **base** and **multi** died between 3500 and 3600 words
(`createCommandEncoder` / `std::bad_alloc`). Use `classify_text_long` /
`extract_entities_long` (384/64) past that.

## Still not in ONNX

Full Kleene classification AST is not ported; README `implies` / `excludes`
is a JS beam. Latent / anchorless records are not exported.

## Credits

- Base checkpoints: [Fastino](https://fastino.ai), Apache-2.0.
- ONNX export + JS host: [Pastel-Cloud OÜ](https://github.com/Pastel-Org).
"""


def main() -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    for key, base_model in REPOS:
        out = ROOT / f"gliner2.5-{key}-v1-onnx-v5"
        onnx = out / "onnx" / "model.onnx"
        heads = out / "onnx" / "heads.onnx"
        records = out / "onnx" / "records.onnx"
        attrs = out / "onnx" / "attrs.onnx"
        card = CARD.format(
            slug=f"gliner2.5-{key}-v1",
            base_model=base_model,
            model_mb=onnx.stat().st_size / 1e6,
            heads_mb=heads.stat().st_size / 1e6,
            records_mb=records.stat().st_size / 1e6,
            attrs_mb=attrs.stat().st_size / 1e6,
        )
        (out / "README.md").write_text(card)
        repo = f"nicolasembleton/gliner2.5-{key}-v1-onnx"
        api.upload_file(
            path_or_fileobj=card.encode(),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="model",
            commit_message="docs: attrs.onnx sizes, 512-word crop, WebGPU length cliff",
        )
        print(f"[{key}] https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
