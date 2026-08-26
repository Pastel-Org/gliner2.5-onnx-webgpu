#!/usr/bin/env python3
"""v5 main graph: v4 I/O plus candidate_states [B, Q, C, H] for RecordHead."""
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
        print(f"  [warn] {e}")

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
            self, input_ids, attention_mask, text_word_indices, text_word_mask,
            query_marker_indices, query_marker_mask, cls_marker_indices, cls_marker_mask,
            rel_marker_indices, rel_marker_mask,
        ):
            hidden_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
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
            cls_logits = self.classifier(cls_states).squeeze(-1) * cls_marker_mask.to(text_states.dtype)
            cs = cands.candidate_states
            if cs is None:
                b, q, c, _ = cands.indices.shape
                cs = text_states.new_zeros(b, q, c, text_states.shape[-1])
            return (
                out.start_logits, out.end_logits,
                cands.indices.to(torch.int64), cands.pair_logits, cands.valid_mask.to(torch.uint8),
                cls_logits, text_states, query_states, rel_role_states, cs,
            )

    return Wrapper(extractor).eval()


def export_one(model_id, out_dir):
    from gliner2 import AutoExtractor
    print(f"Loading {model_id} ...")
    model = AutoExtractor.from_pretrained(model_id, map_location="cpu")
    model.eval()
    _patch_proposer_for_export()
    wrapper = build_wrapper(model)
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
    with torch.no_grad():
        ref = wrapper(*dummy.values())
    print("  shapes", [tuple(x.shape) for x in ref])
    slug = model_id.split("/")[-1]
    out = Path(out_dir) / f"{slug}-onnx-v5"
    onnx_dir = out / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    names = list(dummy)
    outs = [
        "start_logits", "end_logits", "pair_indices", "pair_logits", "pair_valid",
        "cls_logits", "text_states", "query_states", "rel_role_states", "candidate_states",
    ]
    print("  torch.onnx.export model.onnx ...")
    torch.onnx.export(
        wrapper, tuple(dummy[n] for n in names), str(onnx_dir / "model.onnx"),
        input_names=names, output_names=outs, opset_version=17, do_constant_folding=True,
        dynamic_axes={
            **{n: {0: "batch", 1: ax} for n, ax in [
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
            "candidate_states": {0: "batch", 1: "queries", 2: "candidates", 3: "hidden"},
        },
    )
    size = (onnx_dir / "model.onnx").stat().st_size / 1e6
    print(f"  wrote model.onnx ({size:.1f} MB)")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    outs_np = sess.run(None, {n: v.numpy() for n, v in dummy.items()})
    print("  ort", [(o.name, tuple(v.shape)) for o, v in zip(sess.get_outputs(), outs_np)])
    (out / "export_config.json").write_text(json.dumps({
        "export_version": 5, "base_model": model_id,
        "notes": "v4 plus candidate_states for RecordHead",
    }, indent=2))
    model.processor.tokenizer.save_pretrained(str(out))
    print("  at", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="fastino/gliner2.5-small-v1")
    p.add_argument("--out-dir", default="./output-v5")
    args = p.parse_args()
    export_one(args.model_id, args.out_dir)
