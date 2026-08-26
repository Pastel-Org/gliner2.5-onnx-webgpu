#!/usr/bin/env python3
"""Export RecordHead assignment as a tiny onnx/records.onnx (no encoder).

Inputs are padded instance/field/candidate states from the v5 main graph.
Assignment: inst_proj + field_proj vs cand_proj, plus a null column.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn


class RecordAssign(nn.Module):
    def __init__(self, head: nn.Module):
        super().__init__()
        self.inst_proj = head.inst_proj
        self.field_proj = head.field_proj
        self.cand_proj = head.cand_proj
        self.null_embed = head.null_embed
        self.object_head = head.object_head
        self.latent_seed_head = head.latent_seed_head

    def forward(self, inst_states, inst_mask, field_query_states, field_cand_states, field_cand_mask):
        inst_q = self.inst_proj(inst_states)
        field_q = self.field_proj(field_query_states)
        query = inst_q[:, :, None, :] + field_q[:, None, :, :]
        null = torch.einsum("bnfd,d->bnf", query, self.null_embed)
        cand_p = self.cand_proj(field_cand_states)
        scores = torch.einsum("bnfd,bfcd->bnfc", query, cand_p)
        scores = scores.masked_fill(field_cand_mask[:, None, :, :] < 0.5, -1.0e4)
        assign = torch.cat([null.unsqueeze(-1), scores], dim=-1)
        object_logits = self.object_head(inst_states).squeeze(-1)
        object_logits = object_logits.masked_fill(inst_mask < 0.5, -1.0e4)
        latent_logits = self.latent_seed_head(inst_states).squeeze(-1)
        latent_logits = latent_logits.masked_fill(inst_mask < 0.5, -1.0e4)
        return assign, object_logits, latent_logits


def export_one(model_id: str, out_onnx: Path):
    from gliner2 import AutoExtractor

    print(f"Loading {model_id} ...")
    model = AutoExtractor.from_pretrained(model_id, map_location="cpu").eval()
    rh = model.record_decoder
    wrap = RecordAssign(rh).eval()
    h = model.hidden_size
    b, n, f, c = 1, 4, 3, 8
    dummy = {
        "inst_states": torch.randn(b, n, h),
        "inst_mask": torch.ones(b, n),
        "field_query_states": torch.randn(b, f, h),
        "field_cand_states": torch.randn(b, f, c, h),
        "field_cand_mask": torch.ones(b, f, c),
    }
    with torch.no_grad():
        ref = wrap(*dummy.values())
    print("  torch", [tuple(t.shape) for t in ref])
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    names = list(dummy)
    torch.onnx.export(
        wrap,
        tuple(dummy[k] for k in names),
        str(out_onnx),
        input_names=names,
        output_names=["assign_logits", "object_logits", "latent_logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "inst_states": {0: "batch", 1: "instances", 2: "hidden"},
            "inst_mask": {0: "batch", 1: "instances"},
            "field_query_states": {0: "batch", 1: "fields", 2: "hidden"},
            "field_cand_states": {0: "batch", 1: "fields", 2: "cands", 3: "hidden"},
            "field_cand_mask": {0: "batch", 1: "fields", 2: "cands"},
            "assign_logits": {0: "batch", 1: "instances", 2: "fields", 3: "cands_plus_null"},
            "object_logits": {0: "batch", 1: "instances"},
            "latent_logits": {0: "batch", 1: "instances"},
        },
    )
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_onnx), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {k: v.numpy() for k, v in dummy.items()})
    rmses = [float(((o - r.detach().numpy()) ** 2).mean() ** 0.5) for o, r in zip(outs, ref)]
    print(f"  wrote {out_onnx} ({out_onnx.stat().st_size/1e6:.2f} MB) RMSE {rmses}")
    return rmses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="fastino/gliner2.5-small-v1")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    export_one(args.model_id, Path(args.out))


if __name__ == "__main__":
    main()
