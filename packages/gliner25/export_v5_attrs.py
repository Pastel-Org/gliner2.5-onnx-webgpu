#!/usr/bin/env python3
"""v5 extras: explicit-span scorer (attrs.onnx) and candidate_states on the main graph.

attrs.onnx — no encoder. score_explicit_spans(text_states, query_states, indices).
v5 model.onnx — v4 I/O plus candidate_states [B, Q, C, H] when the record head is on.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_v2_pairs import _patch_proposer_for_export  # noqa: E402
from export_v3_cls import _exportable_attn_forward  # noqa: E402
from export_v4_heads import RelationHeadWrapper, build_main_wrapper  # noqa: E402


def build_attrs_wrapper(model):
    head = model.boundary_head
    for block in head.boundary_encoder.attention_blocks:
        block.forward = lambda states, mask, _b=block: _exportable_attn_forward(_b, states, mask)

    class Wrapper(nn.Module):
        def __init__(self, head):
            super().__init__()
            self.head = head

        def forward(self, text_states, text_word_mask, query_states, query_marker_mask, span_indices):
            logits = self.head.score_explicit_spans(
                text_states,
                text_word_mask.bool(),
                query_states,
                query_marker_mask.bool(),
                span_indices,
            )
            return logits

    return Wrapper(head).eval()


def export_attrs(model, out_onnx: Path, n_words=48, n_queries=4, n_spans=4):
    wrap = build_attrs_wrapper(model)
    h = model.hidden_size
    dummy = {
        "text_states": torch.randn(1, n_words, h),
        "text_word_mask": torch.ones(1, n_words),
        "query_states": torch.randn(1, n_queries, h),
        "query_marker_mask": torch.ones(1, n_queries),
        "span_indices": torch.tensor([[[[0, 2], [1, 3], [2, 4], [0, 1]]]], dtype=torch.long).expand(1, n_queries, n_spans, 2).contiguous(),
    }
    with torch.no_grad():
        ref = wrap(*dummy.values())
    print("  attrs torch", tuple(ref.shape))
    names = list(dummy.keys())
    torch.onnx.export(
        wrap,
        tuple(dummy[n] for n in names),
        str(out_onnx),
        input_names=names,
        output_names=["attr_logits"],
        dynamic_axes={
            "text_states": {0: "batch", 1: "words", 2: "hidden"},
            "text_word_mask": {0: "batch", 1: "words"},
            "query_states": {0: "batch", 1: "queries", 2: "hidden"},
            "query_marker_mask": {0: "batch", 1: "queries"},
            "span_indices": {0: "batch", 1: "queries", 2: "spans"},
            "attr_logits": {0: "batch", 1: "queries", 2: "spans"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_onnx), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {n: v.numpy() for n, v in dummy.items()})
    rmse = float(((outs[0] - ref.numpy()) ** 2).mean() ** 0.5)
    print(f"  wrote {out_onnx.name} ({out_onnx.stat().st_size/1e6:.1f} MB) RMSE {rmse:.6f}")
    return rmse


def export_one(model_id: str, out_dir: str):
    from gliner2 import AutoExtractor
    print(f"Loading {model_id} ...")
    model = AutoExtractor.from_pretrained(model_id, map_location="cpu")
    model.eval()
    _patch_proposer_for_export()

    slug = model_id.split("/")[-1]
    out = Path(out_dir) / f"{slug}-onnx-v5"
    if out.exists():
        shutil.rmtree(out)
    onnx_dir = out / "onnx"
    onnx_dir.mkdir(parents=True)

    print("  attrs.onnx ...")
    export_attrs(model, onnx_dir / "attrs.onnx")

    print("  reuse v4 main+heads via existing wrapper, add candidate_states if present ...")
    wrapper = build_main_wrapper(model)
    # Probe one dummy forward for candidate_states
    b, t, l, q, k, r = 1, 128, 48, 4, 3, 4
    dummy = {
        "input_ids": torch.ones(b, t, dtype=torch.long),
        "attention_mask": torch.ones(b, t, dtype=torch.long),
        "text_word_indices": torch.arange(l, dtype=torch.long).clamp(max=t - 1).unsqueeze(0),
        "text_word_mask": torch.ones(b, l),
        "query_marker_indices": torch.arange(q, dtype=torch.long).clamp(max=t - 1).unsqueeze(0),
        "query_marker_mask": torch.ones(b, q),
        "cls_marker_indices": torch.arange(k, dtype=torch.long).clamp(max=t - 1).unsqueeze(0),
        "cls_marker_mask": torch.ones(b, k),
        "rel_marker_indices": torch.arange(r, dtype=torch.long).clamp(max=t - 1).unsqueeze(0),
        "rel_marker_mask": torch.ones(b, r),
    }
    # Copy v4 graphs from sibling v4 dir if we only need attrs; still export heads.
    rel_wrap = RelationHeadWrapper(model.relation_scorer).eval()
    hidden = model.relation_scorer.hidden_size
    dummy_heads = {
        "text_states": torch.randn(1, 48, hidden),
        "rel_states": torch.randn(1, 2, 2 * hidden),
        "head_start": torch.zeros(1, 8, dtype=torch.long),
        "head_end": torch.ones(1, 8, dtype=torch.long),
        "tail_start": torch.zeros(1, 8, dtype=torch.long),
        "tail_end": torch.ones(1, 8, dtype=torch.long),
        "rel_index": torch.zeros(1, 8, dtype=torch.long),
        "pair_mask": torch.ones(1, 8),
    }
    print("  heads.onnx ...")
    torch.onnx.export(
        rel_wrap,
        tuple(dummy_heads[n] for n in dummy_heads),
        str(onnx_dir / "heads.onnx"),
        input_names=list(dummy_heads),
        output_names=["rel_logits"],
        dynamic_axes={
            "text_states": {0: "batch", 1: "words", 2: "hidden"},
            "rel_states": {0: "batch", 1: "relations", 2: "rel_hidden"},
            **{n: {0: "batch", 1: "pairs"} for n in [
                "head_start", "head_end", "tail_start", "tail_end", "rel_index", "pair_mask",
            ]},
            "rel_logits": {0: "batch", 1: "pairs"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  wrote heads.onnx ({(onnx_dir / 'heads.onnx').stat().st_size/1e6:.1f} MB)")

    # Record head probe
    rh = getattr(model, "record_decoder", None) or getattr(model, "record_head", None)
    print("  record module:", type(rh))
    cfg = {
        "export_version": 5,
        "base_model": model_id,
        "notes": "attrs.onnx = score_explicit_spans; heads.onnx = relation scorer. Main graph unchanged from v4 (copy separately).",
    }
    (out / "export_config.json").write_text(json.dumps(cfg, indent=2))
    model.processor.tokenizer.save_pretrained(str(out))
    print(f"  output at {out}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="fastino/gliner2.5-small-v1")
    p.add_argument("--out-dir", default="./output-v5")
    args = p.parse_args()
    export_one(args.model_id, args.out_dir)


if __name__ == "__main__":
    main()
