#!/usr/bin/env python3
"""Upload only onnx/attrs.onnx (score_explicit_spans) + patch the model card."""
from __future__ import annotations

from pathlib import Path

ROOT = Path("/Users/nemb/projects/pastel-org/pastel-evals/models/gliner25-v5-export")
KEYS = ("small", "base", "multi")

STILL_OLD = """## Still not in ONNX

`score_explicit_spans` (pair-reranker broadcast does not export). The demo
attaches attributes by looking up the same `(start,end)` on attribute `[E]`
queries. Full Kleene classification AST is not ported; README `implies` /
`excludes` is a JS beam."""

STILL_NEW = """## attrs.onnx

`score_explicit_spans` via the torch dynamo exporter (legacy tracer rewrote
`unsqueeze(-1)` onto the wrong axis). Traced at a fixed pad: 512 words, 8
attribute queries, 16 spans. The JS host pads/truncates to those caps.
ORT RMSE vs torch is ~1e-6. File is ~2.56 MB.

## Still not in ONNX

Full Kleene classification AST is not ported; README `implies` / `excludes`
is a JS beam. Latent / anchorless records are not exported."""


def main():
    from huggingface_hub import HfApi

    api = HfApi()
    for key in KEYS:
        repo = f"nicolasembleton/gliner2.5-{key}-v1-onnx"
        src = ROOT / f"gliner2.5-{key}-v1-onnx-v5" / "onnx" / "attrs.onnx"
        if not src.exists():
            raise SystemExit(f"missing {src}")
        print(f"[{key}] {src.stat().st_size/1e6:.2f} MB -> {repo}/onnx/attrs.onnx")
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo="onnx/attrs.onnx",
            repo_id=repo,
            repo_type="model",
            commit_message="add attrs.onnx: dynamo score_explicit_spans (pad 512/8/16)",
        )
        try:
            card = api.hf_hub_download(repo_id=repo, filename="README.md")
            text = Path(card).read_text()
        except Exception:
            text = ""
        if STILL_OLD in text:
            text = text.replace(STILL_OLD, STILL_NEW)
            api.upload_file(
                path_or_fileobj=text.encode(),
                path_in_repo="README.md",
                repo_id=repo,
                repo_type="model",
                commit_message="docs: attrs.onnx is shipped; drop the old export-failure note",
            )
        print(f"[{key}] https://huggingface.co/{repo}/blob/main/onnx/attrs.onnx")


if __name__ == "__main__":
    main()
