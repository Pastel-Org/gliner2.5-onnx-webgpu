#!/usr/bin/env python3
"""Export score_explicit_spans as onnx/attrs.onnx (no encoder)."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_v2_pairs import _patch_proposer_for_export  # noqa: E402
from export_v3_cls import _exportable_attn_forward  # noqa: E402


def _exportable_pair_forward(
    self,
    boundary_states,
    query_states,
    proposals,
    start_logits,
    end_logits,
    inside_prefix,
    text_lengths,
    text_states=None,
    text_mask=None,
    inside_prefix_mean=None,
):
    from gliner2.models.boundary.scoring import (
        gather_boundary_states,
        interval_prefix_score,
        continuous_length_features,
        mask_invalid_candidate_logits,
    )

    starts = proposals.indices[..., 0]
    ends = proposals.indices[..., 1]
    valid = proposals.valid_mask
    scale = 1.0 / math.sqrt(self.pair_dim)
    cand_count = starts.shape[2]

    if proposals.score_start_states is not None and proposals.score_end_states is not None:
        s_proj = self.dropout(proposals.score_start_states)
        e_proj = self.dropout(proposals.score_end_states)
    else:
        start_all, end_all = self.project_endpoints(boundary_states)
        s_proj = self.dropout(gather_boundary_states(start_all, starts))
        e_proj = self.dropout(gather_boundary_states(end_all, ends))
    gate = torch.sigmoid(self.query_gate(query_states))
    if self.enable_rotary_endpoints:
        gate = gate.repeat_interleave(2, dim=-1)
    gate = gate.unsqueeze(2).expand(-1, -1, cand_count, -1)
    if self.reranker_endpoint_compat:
        per_head = (s_proj * gate * e_proj).reshape(
            *s_proj.shape[:-1], self.multihead_pair_compat_heads, -1
        ).sum(-1)
        compat = self.compat_mix(per_head).squeeze(-1) * scale
    else:
        compat = torch.zeros_like(starts, dtype=s_proj.dtype)
    if self.endpoint_difference_projection is not None:
        difference = torch.cat((s_proj - e_proj, (s_proj - e_proj).abs()), dim=-1)
        compat = compat + self.endpoint_difference_projection(difference).squeeze(-1)

    max_s = start_logits.shape[2] - 1
    max_e = end_logits.shape[2] - 1
    a = torch.gather(start_logits, 2, starts.clamp(0, max_s))
    bmarg = torch.gather(end_logits, 2, ends.clamp(0, max_e))
    prior_source = proposals.compat_logits if proposals.compat_logits is not None else proposals.logits
    if prior_source is None:
        raise ValueError("boundary proposals must provide compat_logits or logits")
    prior = torch.where(valid, prior_source, torch.zeros_like(prior_source))
    score = compat + a + bmarg + prior

    if self.content_pooler is not None:
        if text_states is None or text_mask is None:
            raise ValueError("span content scoring requires text_states and text_mask")
        mean_prefix, lse_prefix = self.content_pooler.build_prefix(text_states, text_mask)
        span_content = self.content_pooler.pool(mean_prefix, lse_prefix, starts, ends, score.dtype)
        coefficient = self.content_query_projection(query_states).unsqueeze(2).expand_as(span_content)
        content_scale = 1.0 / math.sqrt(span_content.shape[-1])
        score = score + (span_content * coefficient).sum(-1) * content_scale
        score = score + self.content_bias(span_content).squeeze(-1)

    if self.use_inside_evidence and inside_prefix is not None:
        interval = interval_prefix_score(inside_prefix, starts, ends, inside_prefix_mean).to(score.dtype)
        denom = torch.sqrt((ends - starts).clamp(min=1).to(score.dtype))
        inside_weight = (
            self.inside_weight(query_states).squeeze(-1).unsqueeze(-1)
            if self.query_conditioned_inside_weight
            else self.inside_weight
        )
        score = score + inside_weight * (interval / denom)

    feats = continuous_length_features(starts, ends, text_lengths)
    length_coeff = self.length_query_projection(query_states).unsqueeze(2).expand_as(feats)
    length_score = (feats.to(length_coeff.dtype) * length_coeff).sum(-1)
    score = score + length_score
    return mask_invalid_candidate_logits(score, valid)


def build_attrs_wrapper(model):
    head = model.boundary_head
    for block in head.boundary_encoder.attention_blocks:
        block.forward = lambda states, mask, _b=block: _exportable_attn_forward(_b, states, mask)
    head.pair_scorer.forward = _exportable_pair_forward.__get__(head.pair_scorer, type(head.pair_scorer))

    class Wrapper(nn.Module):
        def __init__(self, head):
            super().__init__()
            self.head = head

        def forward(self, text_states, text_word_mask, query_states, query_marker_mask, span_indices):
            return self.head.score_explicit_spans(
                text_states.clone(),
                text_word_mask.bool(),
                query_states,
                query_marker_mask.bool(),
                span_indices,
            )

    return Wrapper(head).eval()


def export_attrs(model, out_onnx: Path, n_words=48, n_queries=4, n_spans=4):
    wrap = build_attrs_wrapper(model)
    h = model.hidden_size
    dummy = {
        "text_states": torch.randn(1, n_words, h),
        "text_word_mask": torch.ones(1, n_words),
        "query_states": torch.randn(1, n_queries, h),
        "query_marker_mask": torch.ones(1, n_queries),
        "span_indices": torch.tensor([[[[0, 2], [1, 3], [2, 4], [0, 1]]]], dtype=torch.long)
        .expand(1, n_queries, n_spans, 2)
        .contiguous(),
    }
    with torch.no_grad():
        ref = wrap(*dummy.values())
    print("  attrs torch", tuple(ref.shape))
    names = list(dummy)
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
    rmse = float(((outs[0] - ref.detach().numpy()) ** 2).mean() ** 0.5)
    print(f"  wrote {out_onnx.name} ({out_onnx.stat().st_size / 1e6:.1f} MB) RMSE {rmse:.6f}")
    return rmse


def export_one(model_id: str, out_onnx: Path):
    from gliner2 import AutoExtractor
    print(f"Loading {model_id} ...")
    model = AutoExtractor.from_pretrained(model_id, map_location="cpu").eval()
    _patch_proposer_for_export()
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    export_attrs(model, out_onnx)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="fastino/gliner2.5-small-v1")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    export_one(args.model_id, Path(args.out))
