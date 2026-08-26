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
def _web_attn_forward(block, states, mask):
    """Same math as _exportable_attn_forward, no tensor-iteration / unbind."""
    b, n, d = states.shape
    heads = block.num_heads
    head_dim = block.head_dim
    qkv = block.qkv_projection(block.norm(states)).reshape(b, n, 3, heads, head_dim)
    query = qkv[:, :, 0].permute(0, 2, 1, 3)
    key = qkv[:, :, 1].permute(0, 2, 1, 3)
    value = qkv[:, :, 2].permute(0, 2, 1, 3)
    scale = head_dim ** -0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    allowed = mask.reshape(b, 1, 1, n)
    if block.window > 0:
        positions = torch.arange(n, device=states.device)
        local = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs() <= block.window
        allowed = allowed & local.reshape(1, 1, n, n)
    idx = torch.arange(n, device=states.device)
    diag = idx.unsqueeze(0) == idx.unsqueeze(1)
    allowed = allowed | diag.reshape(1, 1, n, n)
    scores = scores.masked_fill(~allowed, -1.0e4)
    attn = torch.softmax(scores, dim=-1)
    attended = torch.matmul(attn, value).transpose(1, 2).reshape(b, n, d)
    update = block.dropout(block.output_projection(attended))
    return (states + update) * mask.unsqueeze(-1).to(states.dtype)

PAD_WORDS = 512
PAD_QUERIES = 8
PAD_SPANS = 16

BANNED_OPS = {
    "SplitToSequence",
    "SequenceAt",
    "SequenceEmpty",
    "SequenceInsert",
    "SequenceErase",
    "Optional",
    "OptionalGetElement",
}


def _assert_web_ops(path: Path) -> None:
    import onnx
    from onnx import helper, numpy_helper

    m = onnx.load(str(path))
    del m.functions[:]
    keep = [o for o in m.opset_import if o.domain in ("", "ai.onnx", "ai.onnx.ml")]
    del m.opset_import[:]
    m.opset_import.extend(keep)
    types = {}
    for vi in list(m.graph.input) + list(m.graph.output) + list(m.graph.value_info):
        types[vi.name] = vi.type.tensor_type.elem_type
    for init in m.graph.initializer:
        types[init.name] = init.data_type
    for n in m.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    types[n.output[0]] = a.t.data_type
    new_nodes = []
    for n in m.graph.node:
        if n.op_type == "CastLike" and len(n.input) == 2:
            to = types.get(n.input[1], 1) or 1
            new_nodes.append(helper.make_node("Cast", [n.input[0]], list(n.output), name=n.name, to=int(to)))
            continue
        if n.domain and n.domain not in ("", "ai.onnx", "ai.onnx.ml"):
            raise SystemExit(f"ORT-web-unsafe op {n.op_type}:{n.domain}")
        if n.op_type in BANNED_OPS:
            raise SystemExit(f"ORT-web-unsafe op {n.op_type}")
        new_nodes.append(n)
    del m.graph.node[:]
    m.graph.node.extend(new_nodes)
    onnx.save(m, str(path))
    print(f"  web-ops ok ({len(m.graph.node)} nodes, functions stripped, CastLike->Cast)")


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
        block.forward = lambda states, mask, _b=block: _web_attn_forward(_b, states, mask)
    for block in getattr(head.boundary_encoder, "refinement_blocks", []):
        def _swiglu_forward(states, _b=block):
            proj = _b.input_projection(_b.norm(states))
            half = proj.shape[-1] // 2
            value = proj[..., :half]
            gate = proj[..., half:]
            update = value * torch.nn.functional.silu(gate)
            update = _b.dropout(update)
            update = _b.output_projection(update)
            return states + _b.dropout(update)
        block.forward = _swiglu_forward
    wrap = ExplicitSpanWrapper(head).eval()
    args = _dummy(model.hidden_size)
    orig_log1p = torch.log1p
    torch.log1p = lambda x, _o=orig_log1p: torch.log(x + 1)
    try:
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
    finally:
        torch.log1p = orig_log1p
    _assert_web_ops(out_onnx)
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
