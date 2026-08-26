#!/usr/bin/env python3
"""Upload v5 ONNX graphs (candidate_states + records.onnx) over the existing HF repos."""
from __future__ import annotations

import argparse
import json
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
`BoundaryExtractor`) for onnxruntime-web / WebGPU. **Revision 5:** revision 4
plus `candidate_states` on the main graph and a tiny `records.onnx` assignment
head.

One encoder pass. JointIE beam, classification `implies`/`excludes`, and record
assignment stay in JavaScript.

Host packing and decode: [Pastel-Org/gliner2.5-onnx-webgpu](https://github.com/Pastel-Org/gliner2.5-onnx-webgpu)
(live: [gliner25-onnx-webgpu.pages.dev](https://gliner25-onnx-webgpu.pages.dev)).

## Files

| File | Role | Size |
|------|------|------|
| `onnx/model.onnx` | Encoder + entity pair path + classifier + `text_states` + `candidate_states` | {model_mb:.1f} MB |
| `onnx/heads.onnx` | `SparseRelationScorer` only (no encoder) | {heads_mb:.1f} MB |
| `onnx/records.onnx` | RecordHead assignment (inst/field/cand projections + null column) | {records_mb:.2f} MB |

## model.onnx outputs (new in v5)

`candidate_states` `[B, Q, C, H]` — C=192. Needed by `records.onnx`. Unused heads:
pass length-1 indices with mask 0.

## records.onnx

Inputs: `inst_states [B,N,H]`, `inst_mask [B,N]`, `field_query_states [B,F,H]`,
`field_cand_states [B,F,C,H]`, `field_cand_mask [B,F,C]`.
Outputs: `assign_logits [B,N,F,1+C]` (col 0 is ABSENT), `object_logits`,
`latent_logits`. Natural mode seeds instances from the first `::str` field's
candidates above threshold.

## Still not in ONNX

`score_explicit_spans` (pair-reranker broadcast does not export). The demo
attaches attributes by looking up the same `(start,end)` on attribute `[E]`
queries. Full Kleene classification AST is not ported; README `implies` /
`excludes` is a JS beam.

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
        out = ROOT / f"gliner2.5-{key}-v1-onnx-v5"
        onnx = out / "onnx" / "model.onnx"
        heads = out / "onnx" / "heads.onnx"
        records = out / "onnx" / "records.onnx"
        if not onnx.exists() or not heads.exists() or not records.exists():
            raise SystemExit(f"missing files under {out}")
        card = CARD.format(
            slug=f"gliner2.5-{key}-v1",
            base_model=base_model,
            model_mb=onnx.stat().st_size / 1e6,
            heads_mb=heads.stat().st_size / 1e6,
            records_mb=records.stat().st_size / 1e6,
        )
        (out / "README.md").write_text(card)
        cfg_path = out / "export_config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        cfg["export_version"] = 5
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print(
            f"[{key}] model {onnx.stat().st_size/1e6:.1f} MB  "
            f"heads {heads.stat().st_size/1e6:.1f} MB  "
            f"records {records.stat().st_size/1e6:.2f} MB -> {repo}"
        )
        if args.dry_run:
            continue
        api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(out),
            repo_id=repo,
            repo_type="model",
            commit_message="v5 export: candidate_states + records.onnx assignment head",
            allow_patterns=[
                "onnx/model.onnx",
                "onnx/heads.onnx",
                "onnx/records.onnx",
                "README.md",
                "export_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.txt",
                "config.json",
            ],
        )
        print(f"[{key}] uploaded https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
