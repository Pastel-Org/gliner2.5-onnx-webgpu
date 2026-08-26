#!/usr/bin/env python3
"""v4 ONNX: v3 entity+cls graph plus relation-role gather, and a tiny heads graph.

Main graph (onnx/model.onnx)
    v3 feeds plus:
        rel_marker_indices [B, R]
        rel_marker_mask    [B, R]
    v3 outputs plus:
        query_states    [B, Q, H]   [E] marker states
        rel_role_states [B, R, H]   [R] marker states (host concats head||tail)

Heads graph (onnx/heads.onnx) — no encoder
        text_states  [B, L, H]
        rel_states   [B, Rel, 2H]   directional concat
        head_start, head_end, tail_start, tail_end, rel_index  [B, P] int64
        pair_mask    [B, P] float32
    → rel_logits [B, P]

R=1 / P=1 with mask 0 when unused. Beam search stays in JS.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_v2_pairs import _patch_proposer_for_export  # noqa: E402
from export_v3_cls import _exportable_attn_forward  # noqa: E402


class RelationHeadWrapper(nn.Module):
    """Vectorized SparseRelationScorer over padded [B, P] pairs."""

    def __init__(self, scorer: nn.Module):
        super().__init__()
        self.scorer = scorer
        self.hidden_size = scorer.hidden_size
        self.use_biaffine = bool(scorer.use_biaffine_content)

    def forward(
        self,
        text_states,
        rel_states,
        head_start,
        head_end,
        tail_start,
        tail_end,
        rel_index,
        pair_mask,
    ):
        bsz, length, hidden = text_states.shape
        pair_count = head_start.shape[1]
        rel_count = rel_states.shape[1]
        batch_ix = torch.arange(bsz, device=text_states.device).unsqueeze(1).expand(-1, pair_count)

        def gather(pos):
            pos = pos.clamp(0, max(length - 1, 0))
            return text_states[batch_ix, pos]

        h_start = gather(head_start)
        h_end = gather((head_end - 1).clamp(min=0))
        t_start = gather(tail_start)
        t_end = gather((tail_end - 1).clamp(min=0))
        rel_ix = rel_index.clamp(0, max(rel_count - 1, 0))
        rel = rel_states[batch_ix, rel_ix]

        delta = (tail_start - head_start).to(text_states.dtype)
        order = torch.sign(delta).unsqueeze(-1)
        dist = (delta.abs() / float(max(length, 1))).unsqueeze(-1)
        feats = torch.cat([h_start, h_end, t_start, t_end, rel, order, dist], dim=-1)
        score = self.scorer.mlp(feats).squeeze(-1)

        if self.use_biaffine:
            zeros = text_states.new_zeros(bsz, 1, hidden)
            prefix = torch.cat((zeros, text_states.float().cumsum(1).to(text_states.dtype)), dim=1)

            def pool(start, end):
                start_c = start.clamp(0, length)
                end_c = end.clamp(0, length)
                span_sum = prefix[batch_ix, end_c] - prefix[batch_ix, start_c]
                width = (end - start).clamp_min(1).unsqueeze(-1).to(span_sum.dtype)
                return span_sum / width

            head_content = self.scorer.head_content_projection(pool(head_start, head_end))
            tail_content = self.scorer.tail_content_projection(pool(tail_start, tail_end))
            gate = torch.sigmoid(self.scorer.relation_content_gate(rel))
            biaffine = (head_content * gate * tail_content).sum(-1) / (self.hidden_size ** 0.5)
            linear = self.scorer.content_linear(
                torch.cat((head_content, tail_content, rel), dim=-1)
            ).squeeze(-1)
            score = score + biaffine + linear

        return score * pair_mask.to(score.dtype)


def build_main_wrapper(model):
    extractor = model
    try:
        extractor.boundary_head.boundary_proposer.settings = (
            extractor.boundary_head.boundary_proposer.settings.__class__(
                **{**extractor.boundary_head.boundary_proposer.settings.__dict__,
                   "export_mode": "vectorized"}
            )
        )
    except Exception as e:
        print(f"  [warn] could not set vectorized export_mode: {e}")

    class Wrapper(nn.Module):
        def __init__(self, extractor):
            super().__init__()
            self.encoder = extractor.encoder
            self.boundary_head = extractor.boundary_head
            self.classifier = extractor.classifier
            for block in self.boundary_head.boundary_encoder.attention_blocks:
                block.forward = lambda states, mask, _b=block: _exportable_attn_forward(_b, states, mask)

        def _gather(self, hidden_states, indices, mask):
            h = hidden_states.shape[-1]
            safe = indices.clamp(0, hidden_states.shape[1] - 1)
            states = hidden_states.gather(1, safe.unsqueeze(-1).expand(-1, -1, h))
            return states * mask.unsqueeze(-1).to(states.dtype)

        def forward(
            self,
            input_ids,
            attention_mask,
            text_word_indices,
            text_word_mask,
            query_marker_indices,
            query_marker_mask,
            cls_marker_indices,
            cls_marker_mask,
            rel_marker_indices,
            rel_marker_mask,
        ):
            hidden_states = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            text_states = self._gather(hidden_states, text_word_indices, text_word_mask)
            query_states = self._gather(hidden_states, query_marker_indices, query_marker_mask)
            cls_states = self._gather(hidden_states, cls_marker_indices, cls_marker_mask)
            rel_role_states = self._gather(hidden_states, rel_marker_indices, rel_marker_mask)
            out = self.boundary_head(
                text_states, text_word_mask.bool(),
                query_states, query_marker_mask.bool(),
                return_candidates=True,
            )
            cands = out.candidates
            cls_logits = self.classifier(cls_states).squeeze(-1)
            cls_logits = cls_logits * cls_marker_mask.to(cls_logits.dtype)
            return (
                out.start_logits,
                out.end_logits,
                cands.indices.to(torch.int64),
                cands.pair_logits,
                cands.valid_mask.to(torch.uint8),
                cls_logits,
                text_states,
                query_states,
                rel_role_states,
            )

    return Wrapper(extractor).eval()


def export_one(model_id: str, out_dir: str, seq_len: int = 128, n_queries: int = 4,
               n_words: int = 48, n_cls: int = 3, n_rel_roles: int = 4, n_pairs: int = 8):
    from gliner2 import AutoExtractor

    print(f"Loading {model_id} ...")
    model = AutoExtractor.from_pretrained(model_id, map_location="cpu")
    model.eval()
    _patch_proposer_for_export()
    wrapper = build_main_wrapper(model)

    b, t, l, q, k, r = 1, seq_len, n_words, n_queries, n_cls, n_rel_roles
    dummy = {
        "input_ids": torch.ones(b, t, dtype=torch.long),
        "attention_mask": torch.ones(b, t, dtype=torch.long),
        "text_word_indices": torch.arange(l, dtype=torch.long).clamp(max=t - 1).unsqueeze(0),
        "text_word_mask": torch.ones(b, l, dtype=torch.float32),
        "query_marker_indices": torch.arange(q, dtype=torch.long).clamp(max=t - 1).unsqueeze(0),
        "query_marker_mask": torch.ones(b, q, dtype=torch.float32),
        "cls_marker_indices": torch.arange(k, dtype=torch.long).clamp(max=t - 1).unsqueeze(0),
        "cls_marker_mask": torch.ones(b, k, dtype=torch.float32),
        "rel_marker_indices": torch.arange(r, dtype=torch.long).clamp(max=t - 1).unsqueeze(0),
        "rel_marker_mask": torch.ones(b, r, dtype=torch.float32),
    }
    with torch.no_grad():
        ref = wrapper(*dummy.values())
    print("  torch main shapes:", [tuple(x.shape) for x in ref])

    slug = model_id.split("/")[-1]
    out = Path(out_dir) / f"{slug}-onnx-v4"
    if out.exists():
        shutil.rmtree(out)
    onnx_dir = out / "onnx"
    onnx_dir.mkdir(parents=True)

    input_names = list(dummy.keys())
    output_names = [
        "start_logits", "end_logits", "pair_indices", "pair_logits", "pair_valid",
        "cls_logits", "text_states", "query_states", "rel_role_states",
    ]
    dynamic_axes = {
        **{name: {0: "batch", 1: ax} for name, ax in [
            ("input_ids", "tokens"), ("attention_mask", "tokens"),
            ("text_word_indices", "words"), ("text_word_mask", "words"),
            ("query_marker_indices", "queries"), ("query_marker_mask", "queries"),
            ("cls_marker_indices", "choices"), ("cls_marker_mask", "choices"),
            ("rel_marker_indices", "rel_roles"), ("rel_marker_mask", "rel_roles"),
        ]},
        "start_logits": {0: "batch", 1: "queries", 2: "boundaries"},
        "end_logits": {0: "batch", 1: "queries", 2: "boundaries"},
        "pair_indices": {0: "batch", 1: "queries", 2: "candidates"},
        "pair_logits": {0: "batch", 1: "queries", 2: "candidates"},
        "pair_valid": {0: "batch", 1: "queries", 2: "candidates"},
        "cls_logits": {0: "batch", 1: "choices"},
        "text_states": {0: "batch", 1: "words", 2: "hidden"},
        "query_states": {0: "batch", 1: "queries", 2: "hidden"},
        "rel_role_states": {0: "batch", 1: "rel_roles", 2: "hidden"},
    }

    print("  torch.onnx.export model.onnx ...")
    torch.onnx.export(
        wrapper,
        tuple(dummy[n] for n in input_names),
        str(onnx_dir / "model.onnx"),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    size_mb = (onnx_dir / "model.onnx").stat().st_size / 1e6
    print(f"  wrote model.onnx ({size_mb:.1f} MB)")

    scorer = model.relation_scorer
    rel_wrap = RelationHeadWrapper(scorer).eval()
    hidden = scorer.hidden_size
    rel_types = 2
    dummy_heads = {
        "text_states": torch.randn(1, n_words, hidden),
        "rel_states": torch.randn(1, rel_types, 2 * hidden),
        "head_start": torch.zeros(1, n_pairs, dtype=torch.long),
        "head_end": torch.ones(1, n_pairs, dtype=torch.long),
        "tail_start": torch.zeros(1, n_pairs, dtype=torch.long),
        "tail_end": torch.ones(1, n_pairs, dtype=torch.long),
        "rel_index": torch.zeros(1, n_pairs, dtype=torch.long),
        "pair_mask": torch.ones(1, n_pairs, dtype=torch.float32),
    }
    with torch.no_grad():
        rel_ref = rel_wrap(*dummy_heads.values())
    print("  torch heads shape:", tuple(rel_ref.shape))

    head_inputs = list(dummy_heads.keys())
    head_axes = {
        "text_states": {0: "batch", 1: "words", 2: "hidden"},
        "rel_states": {0: "batch", 1: "relations", 2: "rel_hidden"},
        **{n: {0: "batch", 1: "pairs"} for n in [
            "head_start", "head_end", "tail_start", "tail_end", "rel_index", "pair_mask",
        ]},
        "rel_logits": {0: "batch", 1: "pairs"},
    }
    print("  torch.onnx.export heads.onnx ...")
    torch.onnx.export(
        rel_wrap,
        tuple(dummy_heads[n] for n in head_inputs),
        str(onnx_dir / "heads.onnx"),
        input_names=head_inputs,
        output_names=["rel_logits"],
        dynamic_axes=head_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    heads_mb = (onnx_dir / "heads.onnx").stat().st_size / 1e6
    print(f"  wrote heads.onnx ({heads_mb:.1f} MB)")

    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(str(onnx_dir / "model.onnx"))
    onnx.checker.check_model(str(onnx_dir / "heads.onnx"))
    sess = ort.InferenceSession(str(onnx_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    feeds = {n: v.numpy() for n, v in dummy.items()}
    outs = sess.run(None, feeds)
    print("  ort main:", [(o.name, tuple(v.shape)) for o, v in zip(sess.get_outputs(), outs)])
    cls_rmse = float(((outs[5] - ref[5].numpy()) ** 2).mean() ** 0.5)
    print(f"  RMSE cls_logits: {cls_rmse:.6f}")

    hsess = ort.InferenceSession(str(onnx_dir / "heads.onnx"), providers=["CPUExecutionProvider"])
    hfeeds = {n: v.numpy() for n, v in dummy_heads.items()}
    houts = hsess.run(None, hfeeds)
    rel_rmse = float(((houts[0] - rel_ref.numpy()) ** 2).mean() ** 0.5)
    print(f"  RMSE rel_logits: {rel_rmse:.6f}")

    cfg = {
        "architecture": "boundary",
        "export_version": 4,
        "base_model": model_id,
        "opset": 17,
        "inputs": input_names,
        "outputs": output_names,
        "heads_inputs": head_inputs,
        "heads_outputs": ["rel_logits"],
        "hidden_size": hidden,
        "directional_relation_states": True,
        "relation_temperature": 1.0,
        "notes": (
            "v4: v3 plus rel_marker gather and query_states. "
            "heads.onnx is SparseRelationScorer (directional 2H, biaffine on). "
            "Host concats [R] head||tail role states. Beam stays in JS. "
            "Dummy R=1/P=1 with mask 0 when unused."
        ),
    }
    (out / "export_config.json").write_text(json.dumps(cfg, indent=2))
    model.processor.tokenizer.save_pretrained(str(out))
    print(f"  output at {out}")
    return {"onnx_mb": size_mb, "heads_mb": heads_mb, "cls_rmse": cls_rmse, "rel_rmse": rel_rmse}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="fastino/gliner2.5-small-v1")
    p.add_argument("--out-dir", default="./output-v4")
    args = p.parse_args()
    export_one(args.model_id, args.out_dir)


if __name__ == "__main__":
    main()
