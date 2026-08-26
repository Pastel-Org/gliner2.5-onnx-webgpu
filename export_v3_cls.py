#!/usr/bin/env python3
"""v3 ONNX: v2 entity graph + classifier + cached text_states.

Same six v2 feeds plus:
    cls_marker_indices  [B, K]
    cls_marker_mask     [B, K]

Outputs: v2 five tensors plus
    cls_logits    [B, K]     classifier MLP on [C] choice states
    text_states   [B, L, H]  word-pooled encoder states (cache for later heads)

K=1 dummy (mask=0) when the call has no classification schema.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Reuse the v2 proposer patches (topk instead of sort/scatter_reduce).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_v2_pairs import _patch_proposer_for_export  # noqa: E402


def _exportable_attn_forward(block, states, mask):
    b, n, d = states.shape
    qkv = block.qkv_projection(block.norm(states)).view(b, n, 3, block.num_heads, block.head_dim)
    query, key, value = qkv.permute(2, 0, 3, 1, 4)
    scale = block.head_dim ** -0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    allowed = mask.view(b, 1, 1, n)
    if block.window > 0:
        positions = torch.arange(n, device=states.device)
        local = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs() <= block.window
        allowed = allowed & local.view(1, 1, n, n)
    idx = torch.arange(n, device=states.device)
    diag = idx.unsqueeze(0) == idx.unsqueeze(1)
    allowed = allowed | diag.view(1, 1, n, n)
    scores = scores.masked_fill(~allowed, -1.0e4)
    attn = torch.softmax(scores, dim=-1)
    attended = torch.matmul(attn, value).transpose(1, 2).reshape(b, n, d)
    update = block.dropout(block.output_projection(attended))
    return (states + update) * mask.unsqueeze(-1).to(states.dtype)


def build_wrapper(model):
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
        ):
            hidden_states = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            text_states = self._gather(hidden_states, text_word_indices, text_word_mask)
            query_states = self._gather(hidden_states, query_marker_indices, query_marker_mask)
            cls_states = self._gather(hidden_states, cls_marker_indices, cls_marker_mask)
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
            )

    return Wrapper(extractor).eval()


def export_one(model_id: str, out_dir: str, seq_len: int = 128, n_queries: int = 4,
               n_words: int = 48, n_cls: int = 3):
    from gliner2 import AutoExtractor

    print(f"Loading {model_id} ...")
    model = AutoExtractor.from_pretrained(model_id, map_location="cpu")
    model.eval()
    _patch_proposer_for_export()
    wrapper = build_wrapper(model)

    b, t, l, q, k = 1, seq_len, n_words, n_queries, n_cls
    dummy = {
        "input_ids": torch.ones(b, t, dtype=torch.long),
        "attention_mask": torch.ones(b, t, dtype=torch.long),
        "text_word_indices": torch.arange(l, dtype=torch.long).clamp(max=t - 1).unsqueeze(0),
        "text_word_mask": torch.ones(b, l, dtype=torch.float32),
        "query_marker_indices": torch.arange(q, dtype=torch.long).clamp(max=t - 1).unsqueeze(0),
        "query_marker_mask": torch.ones(b, q, dtype=torch.float32),
        "cls_marker_indices": torch.arange(k, dtype=torch.long).clamp(max=t - 1).unsqueeze(0),
        "cls_marker_mask": torch.ones(b, k, dtype=torch.float32),
    }
    with torch.no_grad():
        ref = wrapper(*dummy.values())
    print("  torch shapes:", [tuple(r.shape) for r in ref])

    slug = model_id.split("/")[-1]
    out = Path(out_dir) / f"{slug}-onnx-v3"
    if out.exists():
        shutil.rmtree(out)
    onnx_dir = out / "onnx"
    onnx_dir.mkdir(parents=True)

    input_names = list(dummy.keys())
    output_names = [
        "start_logits", "end_logits", "pair_indices", "pair_logits", "pair_valid",
        "cls_logits", "text_states",
    ]
    dynamic_axes = {
        **{name: {0: "batch", 1: ax} for name, ax in [
            ("input_ids", "tokens"), ("attention_mask", "tokens"),
            ("text_word_indices", "words"), ("text_word_mask", "words"),
            ("query_marker_indices", "queries"), ("query_marker_mask", "queries"),
            ("cls_marker_indices", "choices"), ("cls_marker_mask", "choices"),
        ]},
        "start_logits": {0: "batch", 1: "queries", 2: "boundaries"},
        "end_logits": {0: "batch", 1: "queries", 2: "boundaries"},
        "pair_indices": {0: "batch", 1: "queries", 2: "candidates"},
        "pair_logits": {0: "batch", 1: "queries", 2: "candidates"},
        "pair_valid": {0: "batch", 1: "queries", 2: "candidates"},
        "cls_logits": {0: "batch", 1: "choices"},
        "text_states": {0: "batch", 1: "words", 2: "hidden"},
    }

    print("  torch.onnx.export ...")
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

    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(str(onnx_dir / "model.onnx"))
    sess = ort.InferenceSession(str(onnx_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    feeds = {n: v.numpy() for n, v in dummy.items()}
    outs = sess.run(None, feeds)
    print("  ort:", [(o.name, tuple(v.shape)) for o, v in zip(sess.get_outputs(), outs)])
    cls_rmse = float(((outs[5] - ref[5].numpy()) ** 2).mean() ** 0.5)
    print(f"  RMSE cls_logits: {cls_rmse:.6f}")

    cfg = {
        "architecture": "boundary",
        "export_version": 3,
        "base_model": model_id,
        "opset": 17,
        "inputs": input_names,
        "outputs": output_names,
        "notes": "v2 entity path plus classifier on [C] markers and cached text_states. Host packs cls_marker_* from schema_special_positions of classification groups.",
    }
    (out / "export_config.json").write_text(json.dumps(cfg, indent=2))
    model.processor.tokenizer.save_pretrained(str(out))
    print(f"  output at {out}")
    return {"onnx_mb": size_mb, "cls_rmse": cls_rmse}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="fastino/gliner2.5-small-v1")
    p.add_argument("--out-dir", default="./output-v3")
    args = p.parse_args()
    export_one(args.model_id, args.out_dir)


if __name__ == "__main__":
    main()
