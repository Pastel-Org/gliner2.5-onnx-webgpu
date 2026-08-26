#!/usr/bin/env python3
"""Upload v3 ONNX graphs (classifier + text_states) over the existing HF repos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("/Users/nemb/projects/pastel-org/pastel-evals/models/gliner25-v3-export")

REPOS = [
    ("small", "fastino/gliner2.5-small-v1", "74M", "DeBERTa-v3-xsmall", "English", "288.1"),
    ("base", "fastino/gliner2.5-base-v1", "194M", "DeBERTa-v3-base", "English", "745.6"),
    ("multi", "fastino/gliner2.5-multi-v1", "287M", "mDeBERTa-v3-base", "Multilingual", None),
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
---

# {slug}-onnx

ONNX export of [{base_model}](https://huggingface.co/{base_model}) (GLiNER 2.5 `BoundaryExtractor`)
for onnxruntime-web / WebGPU. **Revision 3:** same fused encoder as revision 2,
plus the classification MLP and cached word states.

One encoder pass. Do not download a second graph for classification.

## Graph

1. DeBERTa over packed `input_ids` (schema + `[SEP_TEXT]` + words)
2. Gather word states, `[E]` query markers, `[C]` choice markers
3. Boundary head + pair reranker (entity path, C=192)
4. Classifier MLP on `[C]` states → `cls_logits`
5. `text_states` emitted so later heads (attributes, relations) can score
   without running the encoder again

Host packing and decode: [Pastel-Org/gliner2.5-onnx-webgpu](https://github.com/Pastel-Org/gliner2.5-onnx-webgpu)
(live: [gliner25-onnx-webgpu.pages.dev](https://gliner25-onnx-webgpu.pages.dev)).

## Inputs

| Name | Shape | Dtype |
|------|-------|-------|
| input_ids | [B, T] | int64 |
| attention_mask | [B, T] | int64 |
| text_word_indices | [B, L] | int64 |
| text_word_mask | [B, L] | float32 |
| query_marker_indices | [B, Q] | int64 |
| query_marker_mask | [B, Q] | float32 |
| cls_marker_indices | [B, K] | int64 |
| cls_marker_mask | [B, K] | float32 |

Entity-only calls still must pass `cls_marker_*`. Use K=1 with mask 0.
Classification-only calls still must pass `query_marker_*`. Use Q=1 with mask 0.

## Outputs

| Name | Shape | Dtype |
|------|-------|-------|
| start_logits | [B, Q, L+1] | float32 |
| end_logits | [B, Q, L+1] | float32 |
| pair_indices | [B, Q, 192, 2] | int64 |
| pair_logits | [B, Q, 192] | float32 |
| pair_valid | [B, Q, 192] | uint8 |
| cls_logits | [B, K] | float32 |
| text_states | [B, L, H] | float32 |

Entity decode is unchanged from revision 2:
`sigmoid(pair_logits / pair_temperature)` with pair_temperature=1.0.

Classification: `sigmoid(cls_logits)` then argmax (single-label) or threshold
(multi-label). Constrained `implies`/`excludes` stays on the host.

## Revision 3 changes

- Added `cls_marker_indices` / `cls_marker_mask` and `cls_logits`.
- Added `text_states` (word-pooled encoder states) for a later attribute /
  relation pass without a second encoder.
- File size grows by the classifier MLP only (small: 286.9 → 288.1 MB).

JointIE beam search and record-mode JSON are still not in this graph.

## Credits

- Base checkpoints: [Fastino](https://fastino.ai), Apache-2.0.
- ONNX export + JS host: [Pastel-Cloud OÜ](https://github.com/Pastel-Org).
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", default=None)
    args = parser.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()

    keys = args.only.split(",") if args.only else [k for k, *_ in REPOS]
    for key in keys:
        base_model = next(b for k, b, *_ in REPOS if k == key)
        repo = f"nicolasembleton/gliner2.5-{key}-v1-onnx"
        out = ROOT / f"gliner2.5-{key}-v1-onnx-v3"
        onnx = out / "onnx" / "model.onnx"
        if not onnx.exists():
            raise SystemExit(f"missing {onnx}")
        card = CARD.format(slug=f"gliner2.5-{key}-v1", base_model=base_model)
        (out / "README.md").write_text(card)
        cfg_path = out / "export_config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        cfg["export_version"] = 3
        cfg["notes"] = (
            "v3: v2 entity path + classifier on [C] markers + cached text_states. "
            "Dummy K=1/Q=1 with mask 0 when unused."
        )
        cfg_path.write_text(json.dumps(cfg, indent=2))
        size_mb = onnx.stat().st_size / 1e6
        print(f"[{key}] {size_mb:.1f} MB -> {repo}")
        if args.dry_run:
            continue
        api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(out),
            repo_id=repo,
            repo_type="model",
            commit_message="v3 export: classifier + cached text_states on the same encoder",
        )
        print(f"[{key}] uploaded https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
