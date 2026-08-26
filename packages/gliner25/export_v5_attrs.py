#!/usr/bin/env python3
"""Export score_explicit_spans via the torch dynamo ONNX exporter.

The legacy tracer rewrites unsqueeze(-1) onto the wrong axis (Q vs C, 3 vs 4).
Dynamo keeps the stock pair-reranker. The graph is traced at a fixed pad
(L=512 words, Q=8 attribute labels, C=16 spans). The JS host pads/truncates
to those caps — demo texts are far smaller.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_v2_pairs import _patch_proposer_for_export  # noqa: E402
from export_v3_cls import _exportable_attn_forward  # noqa: E402

PAD_WORDS = 512
PAD_QUERIES = 8
PAD_SPANS = 16


class ExplicitSpanWrapper(nn.Module):
    def __init__(self, head: nn.Module):
        super().__init__()
        self.head = head

    def forward(self, text_states, text_word_mask, query_states, query_marker_mask, span_indices):
        return self.head.score_explicit_spans(
            text_states,
            text_word_mask > 0.5,
            query_states,
            query_marker_mask > 0.5,
            span_indices,
        )


def _dummy(hidden: int):
    spans = [[i % 8, (i % 8) + 2] for i in range(PAD_SPANS)]
    idx = torch.tensor(spans, dtype=torch.long).view(1, 1, PAD_SPANS, 2)
    idx = idx.expand(1, PAD_QUERIES, PAD_SPANS, 2).contiguous()
    return (
        torch.randn(1, PAD_WORDS, hidden),
        torch.ones(1, PAD_WORDS),
        torch.randn(1, PAD_QUERIES, hidden),
        torch.ones(1, PAD_QUERIES),
        idx,
    )


def export_attrs(model, out_onnx: Path):
    head = model.boundary_head
    for block in head.boundary_encoder.attention_blocks:
        block.forward = lambda states, mask, _b=block: _exportable_attn_forward(_b, states, mask)
    wrap = ExplicitSpanWrapper(head).eval()
    args = _dummy(model.hidden_size)
    with torch.no_grad():
        ref = wrap(*args)
    print("  torch", tuple(ref.shape))
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrap,
        args,
        str(out_onnx),
        dynamo=True,
        input_names=["text_states", "text_word_mask", "query_states", "query_marker_mask", "span_indices"],
        output_names=["attr_logits"],
    )
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_onnx), providers=["CPUExecutionProvider"])
    names = ["text_states", "text_word_mask", "query_states", "query_marker_mask", "span_indices"]
    out = sess.run(None, {n: t.detach().numpy() for n, t in zip(names, args)})[0]
    rmse = float(((out - ref.detach().numpy()) ** 2).mean() ** 0.5)
    print(f"  wrote {out_onnx} ({out_onnx.stat().st_size / 1e6:.2f} MB) RMSE {rmse:.6f}")
    if rmse > 1e-4:
        raise SystemExit(f"RMSE too high: {rmse}")
    return rmse


def export_one(model_id: str, out_onnx: Path):
    from gliner2 import AutoExtractor
    print(f"Loading {model_id} ...")
    model = AutoExtractor.from_pretrained(model_id, map_location="cpu").eval()
    _patch_proposer_for_export()
    return export_attrs(model, out_onnx)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="fastino/gliner2.5-small-v1")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    export_one(args.model_id, Path(args.out))
