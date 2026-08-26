#!/usr/bin/env python3
"""Upload v4 ONNX graphs (rel gather + heads.onnx) over the existing HF repos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("/Users/nemb/projects/pastel-org/pastel-evals/models/gliner25-v4-export")

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
`BoundaryExtractor`) for onnxruntime-web / WebGPU. **Revision 4:** same fused
encoder as revision 3, plus `[R]` role-state gather and a separate relation
scorer graph.

One encoder pass. JointIE beam search stays in JavaScript.

Host packing and decode: [Pastel-Org/gliner2.5-onnx-webgpu](https://github.com/Pastel-Org/gliner2.5-onnx-webgpu)
(live: [gliner25-onnx-webgpu.pages.dev](https://gliner25-onnx-webgpu.pages.dev)).

## Files

| File | Role |
|------|------|
| `onnx/model.onnx` | Encoder + entity pair path + classifier + cached states |
| `onnx/heads.onnx` | `SparseRelationScorer` only (no encoder). Directional 2H + biaffine. |

## model.onnx inputs

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
| rel_marker_indices | [B, R] | int64 |
| rel_marker_mask | [B, R] | float32 |

Unused heads: pass length-1 indices with mask 0.

## model.onnx outputs

| Name | Shape |
|------|-------|
| start_logits / end_logits | [B, Q, L+1] |
| pair_indices / pair_logits / pair_valid | C=192 |
| cls_logits | [B, K] |
| text_states | [B, L, H] |
| query_states | [B, Q, H] |
| rel_role_states | [B, R, H] |

Host concatenates each `[R] head` + `[R] tail` pair into a 2H relation state.

## heads.onnx

Inputs: `text_states`, `rel_states [B, Rel, 2H]`, `head_start/end`, `tail_start/end`,
`rel_index`, `pair_mask` (all pair tensors `[B, P]`). Output: `rel_logits [B, P]`.
`sigmoid(rel_logits)` then JS beam (width 16). Schema constraints stay on the host.

## Still not in ONNX

Neural RecordHead instance formation. Repeated JSON objects on the demo are
host assignment (i-th mention of each field). Constrained `implies`/`excludes`
for classification is JS when used.

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

    keys = args.only.split(",") if args.only else [k for k, _ in REPOS]
    for key in keys:
        base_model = next(b for k, b in REPOS if k == key)
        repo = f"nicolasembleton/gliner2.5-{key}-v1-onnx"
        out = ROOT / f"gliner2.5-{key}-v1-onnx-v4"
        onnx = out / "onnx" / "model.onnx"
        heads = out / "onnx" / "heads.onnx"
        if not onnx.exists() or not heads.exists():
            raise SystemExit(f"missing {onnx} or {heads}")
        card = CARD.format(slug=f"gliner2.5-{key}-v1", base_model=base_model)
        (out / "README.md").write_text(card)
        cfg_path = out / "export_config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        cfg["export_version"] = 4
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print(f"[{key}] model {onnx.stat().st_size/1e6:.1f} MB  heads {heads.stat().st_size/1e6:.1f} MB -> {repo}")
        if args.dry_run:
            continue
        api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(out),
            repo_id=repo,
            repo_type="model",
            commit_message="v4 export: [R] role gather + heads.onnx relation scorer",
        )
        print(f"[{key}] uploaded https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
