#!/usr/bin/env python3
"""Export GLiNER 2.5 BoundaryExtractor to ONNX WITH the pair reranker (v2).

Same six inputs as the v1 export. The graph now includes the sparse
proposer + pair scorer (vectorized mode), so outputs are:

    start_logits    [B, Q, L+1]     boundary marginals (same as v1)
    end_logits      [B, Q, L+1]
    pair_indices    [B, Q, C, 2]    half-open word-boundary candidate spans
    pair_logits     [B, Q, C]       reranked span scores (apply sigmoid +
                                    pair_temperature in the host)
    pair_valid      [B, Q, C]       bool (exported as uint8 for ONNX)

C = candidate budget from the checkpoint's BoundaryHeadSettings
(default 64; fixed at export time, padded dynamically at runtime).

Requires local venv: .venv-export (torch 2.5.1, gliner2[local], onnx).

Usage:
    .venv-export/bin/python export_v2_pairs.py --model-id fastino/gliner2.5-small-v1 \
        --out-dir ./output-v2 [--upload --upload-prefix nicolasembleton --repo-suffix "-onnx-v2"]

Validation baked in: ORT outputs vs torch wrapper outputs (RMSE per tensor),
plus a decode-parity check against AutoExtractor.extract_entities on the
model-card sentence when --parity is passed (requires the packed inputs to be
reproduced exactly; we reuse the model's own processor for that).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn


def _patch_proposer_for_export():
    """Replace sort/argsort/scatter_reduce proposer internals with ONNX-safe topk versions.

    torch.sort(stable=True) has no ONNX symbolic ("Sort, Out parameter is not
    supported") and assemble_candidates uses scatter_reduce (opset>=18). We
    substitute topk everywhere. Consequences, both benign for inference:
      - tie order may differ from the Python path (affects only which duplicate
        copy survives), and
      - duplicate (start,end) pairs can occupy multiple candidate slots; the
        host dedupes by (start,end) when iterating candidates.
    """
    from gliner2.models.boundary import proposal as P

    def select_top_boundaries(logits, valid_mask, k):
        # Pad with k invalid sentinel slots before topk: the traced graph has
        # a CONSTANT k, but runtime inputs can have fewer boundaries than k
        # (ORT TopK errors when k > axis dim). Fake slots select last and are
        # zeroed via valid=False — same semantics as upstream invalid slots.
        pad_shape = list(logits.shape)
        pad_shape[-1] = k
        pad_logits = logits.new_full(pad_shape, -1.0e4)
        pad_valid = torch.zeros(pad_shape, dtype=valid_mask.dtype, device=valid_mask.device)
        masked = logits.masked_fill(~valid_mask, -1.0e4)
        padded = torch.cat([masked, pad_logits], dim=-1)
        padded_valid = torch.cat([valid_mask, pad_valid], dim=-1)
        scores, idx = torch.topk(padded, k, dim=-1)
        valid = torch.gather(padded_valid, -1, idx)
        scores = torch.where(valid, scores, torch.zeros_like(scores))
        idx = torch.where(valid, idx, torch.zeros_like(idx))
        return scores, idx, valid

    def merge_running_topk(current_scores, current_indices, block_scores, block_indices, k):
        scores = torch.cat([current_scores, block_scores], dim=-1)
        indices = torch.cat([current_indices, block_indices], dim=-1)
        take = min(k, scores.shape[-1])
        top_scores, order = torch.topk(scores, take, dim=-1)
        top_indices = torch.gather(indices, -1, order)
        return top_scores, top_indices

    def assemble_candidates(pair_starts, pair_ends, pair_scores, pair_valid, query_mask, *,
                            capacity, n_boundaries, gold_pairs=None, gold_mask=None,
                            gold_injection_prob=1.0, generator=None):
        pre_valid = pair_valid & query_mask.unsqueeze(-1)
        floor = -1.0e4
        scores = torch.where(pre_valid, pair_scores, torch.full_like(pair_scores, floor))
        take = min(capacity, scores.shape[-1])
        _, order = torch.topk(scores, take, dim=-1)
        starts = torch.gather(pair_starts, -1, order)
        ends = torch.gather(pair_ends, -1, order)
        selected_valid = torch.gather(pre_valid, -1, order)
        indices = torch.stack((starts, ends), dim=-1)
        indices = torch.where(selected_valid.unsqueeze(-1), indices, torch.zeros_like(indices))
        if take < capacity:
            pad = capacity - take
            indices = torch.nn.functional.pad(indices, (0, 0, 0, pad))
            selected_valid = torch.nn.functional.pad(selected_valid, (0, pad), value=False)
        pre_keys = pair_starts * n_boundaries + pair_ends
        return indices, selected_valid, torch.zeros_like(selected_valid), pre_keys, pre_valid

    P.select_top_boundaries = select_top_boundaries
    P.merge_running_topk = merge_running_topk
    P.assemble_candidates = assemble_candidates
    # pool.py binds select_top_boundaries at import time (from ... import),
    # so it needs the patched name in its own namespace as well.
    from gliner2.models.boundary import pool as Pool

    Pool.select_top_boundaries = select_top_boundaries
    Pool.merge_running_topk = merge_running_topk

    # _deduplicate_pool: replace stable-sort dedup with topk selection.
    # Duplicates may occupy extra slots; identical (start,end) keys produce
    # identical pair_logits, so the decoded span set is unchanged (the host
    # dedupes by (start,end) when consuming candidates).
    def _deduplicate_pool_export(keys, scores, valid, capacity, n_boundaries):
        # Same constant-k padding trick: pad scores/keys/valid by `capacity`
        # sentinel slots so topk(capacity) never exceeds the axis dim and the
        # output is always exactly `capacity` wide (fixed C for the host).
        floor = -1.0e4
        scores = torch.where(valid, scores, torch.full_like(scores, floor))
        pad_shape = list(scores.shape)
        pad_shape[-1] = capacity
        pad_scores = scores.new_full(pad_shape, floor)
        pad_keys = keys.new_zeros(pad_shape)
        pad_valid = torch.zeros(pad_shape, dtype=valid.dtype, device=valid.device)
        scores_p = torch.cat([scores, pad_scores], -1)
        keys_p = torch.cat([keys, pad_keys], -1)
        valid_p = torch.cat([valid, pad_valid], -1)
        _, order = torch.topk(scores_p, capacity, dim=-1)
        selected_keys = keys_p.gather(-1, order)
        selected_valid = valid_p.gather(-1, order)
        return selected_keys, selected_valid

    Pool._deduplicate_pool = _deduplicate_pool_export

    # DocumentCandidatePool.forward: the per-query quota ranking uses
    # torch.argsort inline. Replace forward with the inference-only copy
    # that uses topk (identical selection up to exact ties).
    import math as _math
    from gliner2.models.boundary.indexing import gather_rows as _gather_rows
    from gliner2.models.boundary.constants import MASK_LOGIT as _MASK
    from gliner2.models.boundary.proposal import (  # patched topk versions
        select_top_boundaries as _select_top,
    )

    def _pool_forward_export(
        self,
        boundary_states,  # [B,N,D]
        boundary_mask,    # [B,N]
        query_mask,       # [B,Q]
        start_logits,     # [B,Q,N]
        end_logits,       # [B,Q,N]
        *,
        gold_pairs=None,
        gold_mask=None,
        gold_injection_prob=1.0,
        return_stats=False,
        generator=None,
    ):
        if gold_pairs is not None or return_stats:
            raise RuntimeError("export pool forward supports inference only")
        from gliner2.models.boundary.pool import PooledCandidates

        b, n, d = boundary_states.shape
        q = query_mask.shape[1]
        floor = torch.full_like(start_logits, _MASK)
        q_boundary = boundary_mask.unsqueeze(1) & query_mask.unsqueeze(-1)
        union_start = torch.where(q_boundary, start_logits, floor).amax(1)
        union_end = torch.where(q_boundary, end_logits, floor).amax(1)
        union_valid = boundary_mask & query_mask.any(-1, keepdim=True)

        _, starts, starts_valid = _select_top(
            union_start.unsqueeze(1), union_valid.unsqueeze(1), self.pool_boundary_top_k,
        )
        _, ends, ends_valid = _select_top(
            union_end.unsqueeze(1), union_valid.unsqueeze(1), self.pool_boundary_top_k,
        )
        starts = starts[:, 0]
        ends = ends[:, 0]
        starts_valid = starts_valid[:, 0]
        ends_valid = ends_valid[:, 0]
        ks, ke = starts.shape[1], ends.shape[1]
        pair_s = starts.unsqueeze(-1).expand(b, ks, ke).reshape(b, -1)
        pair_e = ends.unsqueeze(1).expand(b, ks, ke).reshape(b, -1)
        pair_valid = (
            starts_valid.unsqueeze(-1)
            & ends_valid.unsqueeze(1)
            & (ends.unsqueeze(1) > starts.unsqueeze(-1))
        ).reshape(b, -1)

        start_all = self.start_projection(boundary_states)
        end_all = self.end_projection(boundary_states)
        selected_start = _gather_rows(start_all, pair_s)
        selected_end = _gather_rows(end_all, pair_e)
        compat = (selected_start * selected_end).sum(-1) / _math.sqrt(d)
        union_pair_score = (
            compat
            + union_start.gather(1, pair_s.clamp(0, n - 1))
            + union_end.gather(1, pair_e.clamp(0, n - 1))
        )

        quota = min(self.min_pool_per_query, pair_s.shape[-1])
        if quota:
            s_idx = pair_s.clamp(0, start_logits.shape[2] - 1).unsqueeze(1).expand(b, q, -1)
            e_idx = pair_e.clamp(0, end_logits.shape[2] - 1).unsqueeze(1).expand(b, q, -1)
            per_query = (
                start_logits.gather(2, s_idx)
                + end_logits.gather(2, e_idx)
                + compat.unsqueeze(1)
            )
            per_query_valid = pair_valid.unsqueeze(1) & query_mask.unsqueeze(-1)
            # topk instead of argsort (export-safe; same selection up to ties).
            # Pad with quota invalid sentinels first: quota may exceed the
            # number of pairs at runtime (constant-k graph).
            pq = per_query.masked_fill(~per_query_valid, _MASK)
            pad_pq = pq.new_full(list(pq.shape)[:-1] + [quota], _MASK)
            pad_v = torch.zeros_like(pad_pq, dtype=per_query_valid.dtype)
            ranked = torch.topk(torch.cat([pq, pad_pq], -1), quota, dim=-1).indices.clamp(max=per_query_valid.shape[-1] - 1)
            quota_valid_pre = torch.cat([per_query_valid, pad_v], -1).gather(-1, ranked)
            quota_s = s_idx.gather(-1, ranked)
            quota_e = e_idx.gather(-1, ranked)
            quota_valid = quota_valid_pre.reshape(b, -1)
            quota_keys = (quota_s * n + quota_e).reshape(b, -1)
            rank_bonus = torch.arange(
                quota, 0, -1, device=boundary_states.device,
                dtype=union_pair_score.dtype,
            )
            quota_scores = (
                union_pair_score.new_full((b, q, quota), -_MASK * 0.5)
                + rank_bonus.view(1, 1, quota)
            ).reshape(b, -1)
        else:
            quota_keys = pair_s.new_zeros((b, 0))
            quota_scores = union_pair_score.new_zeros((b, 0))
            quota_valid = pair_valid.new_zeros((b, 0))

        global_keys = pair_s * n + pair_e
        all_keys = torch.cat((quota_keys, global_keys), -1)
        all_scores = torch.cat((quota_scores, union_pair_score.detach()), -1)
        all_valid = torch.cat((quota_valid, pair_valid), -1)

        with torch.no_grad():
            selected_keys, selected_valid = Pool._deduplicate_pool(
                all_keys, all_scores, all_valid, self.pool_size, n
            )
        selected_keys = torch.where(
            selected_valid, selected_keys, torch.zeros_like(selected_keys)
        )
        selected_s = torch.div(selected_keys, n, rounding_mode="floor")
        selected_e = selected_keys - selected_s * n
        indices = torch.stack((selected_s, selected_e), -1)
        indices = torch.where(
            selected_valid.unsqueeze(-1), indices, torch.zeros_like(indices)
        )
        gs = _gather_rows(start_all, selected_s)
        ge = _gather_rows(end_all, selected_e)
        selected_compat = (gs * ge).sum(-1) / _math.sqrt(d)
        selected_score = (
            selected_compat
            + union_start.gather(1, selected_s.clamp(0, n - 1))
            + union_end.gather(1, selected_e.clamp(0, n - 1))
        )
        selected_score = selected_score.masked_fill(~selected_valid, _MASK)
        selected_compat = torch.where(
            selected_valid, selected_compat, torch.zeros_like(selected_compat)
        )
        return PooledCandidates(
            indices=indices,
            mask=selected_valid,
            proposal_logits=selected_score,
            gold_mask=None,
            compat_logits=selected_compat,
            stats=None,
        )

    Pool.DocumentCandidatePool.forward = _pool_forward_export
    print("  [patch] pool internals replaced with topk versions (ONNX-safe)")


def build_wrapper(model):
    """Wrap encoder + full boundary head (proposer + pair scorer included)."""
    from gliner2 import AutoExtractor  # noqa: F401  (type hint only)

    # Vectorized proposer: no Python block loop, graph-exportable, and
    # documented upstream as producing identical results.
    try:
        model.boundary_head.boundary_proposer.settings = (
            model.boundary_head.boundary_proposer.settings.__class__(
                **{**model.boundary_head.boundary_proposer.settings.__dict__,
                   "export_mode": "vectorized"}
            )
        )
    except Exception as e:  # pragma: no cover
        print(f"  [warn] could not set vectorized export_mode: {e}")

    encoder = model.encoder
    encoder.eval()
    head = model.boundary_head
    head.eval()

    # EyeLike fix (same as v1): matmul/softmax attention, no torch.eye/SDPA.
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

    class Wrapper(nn.Module):
        """Encoder + gathers + full boundary head with candidate outputs."""

        def __init__(self, extractor):
            super().__init__()
            self.encoder = extractor.encoder
            self.boundary_head = extractor.boundary_head
            for block in self.boundary_head.boundary_encoder.attention_blocks:
                block.forward = lambda states, mask, _b=block: _exportable_attn_forward(_b, states, mask)

        def _gather(self, hidden_states, indices, mask):
            h = hidden_states.shape[-1]
            safe = indices.clamp(0, hidden_states.shape[1] - 1)
            states = hidden_states.gather(1, safe.unsqueeze(-1).expand(-1, -1, h))
            return states * mask.unsqueeze(-1).to(states.dtype)

        def forward(self, input_ids, attention_mask, text_word_indices, text_word_mask,
                    query_marker_indices, query_marker_mask):
            hidden_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            text_states = self._gather(hidden_states, text_word_indices, text_word_mask)
            query_states = self._gather(hidden_states, query_marker_indices, query_marker_mask)
            out = self.boundary_head(
                text_states, text_word_mask.bool(),
                query_states, query_marker_mask.bool(),
                return_candidates=True,
            )
            cands = out.candidates
            # pair_valid as uint8 for ONNX friendliness
            valid_u8 = cands.valid_mask.to(torch.uint8)
            return (
                out.start_logits,      # [B, Q, L+1]
                out.end_logits,        # [B, Q, L+1]
                cands.indices.to(torch.int64),   # [B, Q, C, 2]
                cands.pair_logits,     # [B, Q, C]
                valid_u8,              # [B, Q, C]
            )

    return Wrapper(model).eval()


def export_one(model_id: str, out_dir: str, seq_len: int = 128, n_queries: int = 4,
               n_words: int = 48, parity: bool = True):
    from gliner2 import AutoExtractor

    print(f"Loading {model_id} ...")
    model = AutoExtractor.from_pretrained(model_id, map_location="cpu")
    model.eval()

    _patch_proposer_for_export()
    wrapper = build_wrapper(model)

    candidate_budget = (
        model.boundary_head.boundary_proposer.settings.candidate_budget
    )
    print(f"  candidate budget C = {candidate_budget}")

    b, t, l, q = 1, seq_len, n_words, n_queries
    # Pair temperature lives on the checkpoint settings; try the head first.
    try:
        pair_temperature = float(model.boundary_head.settings.pair_temperature)
    except AttributeError:
        pair_temperature = float(model.boundary_settings.pair_temperature)
    print(f"  pair_temperature = {pair_temperature}")

    dummy = {
        "input_ids": torch.ones(b, t, dtype=torch.long),
        "attention_mask": torch.ones(b, t, dtype=torch.long),
        "text_word_indices": torch.arange(l, dtype=torch.long).clamp(max=t - 1).unsqueeze(0),
        "text_word_mask": torch.ones(b, l, dtype=torch.float32),
        "query_marker_indices": torch.arange(q, dtype=torch.long).clamp(max=t - 1).unsqueeze(0),
        "query_marker_mask": torch.ones(b, q, dtype=torch.float32),
    }

    with torch.no_grad():
        ref = wrapper(*dummy.values())
    print("  torch shapes:", [tuple(r.shape) for r in ref])

    slug = model_id.split("/")[-1]
    out = Path(out_dir) / f"{slug}-onnx"
    if out.exists():
        shutil.rmtree(out)
    onnx_dir = out / "onnx"
    onnx_dir.mkdir(parents=True)

    input_names = list(dummy.keys())
    output_names = ["start_logits", "end_logits", "pair_indices", "pair_logits", "pair_valid"]
    dynamic_axes = {
        **{k: {0: "batch", 1: ax} for k, ax in [
            ("input_ids", "tokens"), ("attention_mask", "tokens"),
            ("text_word_indices", "words"), ("text_word_mask", "words"),
            ("query_marker_indices", "queries"), ("query_marker_mask", "queries"),
        ]},
        "start_logits": {0: "batch", 1: "queries", 2: "boundaries"},
        "end_logits": {0: "batch", 1: "queries", 2: "boundaries"},
        "pair_indices": {0: "batch", 1: "queries", 2: "candidates"},
        "pair_logits": {0: "batch", 1: "queries", 2: "candidates"},
        "pair_valid": {0: "batch", 1: "queries", 2: "candidates"},
    }

    print("  torch.onnx.export ...")
    torch.onnx.export(
        wrapper,
        tuple(dummy[k] for k in input_names),
        str(onnx_dir / "model.onnx"),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    size_mb = (onnx_dir / "model.onnx").stat().st_size / 1e6
    print(f"  wrote model.onnx ({size_mb:.1f} MB)")

    # ── Validate: ORT vs torch ─────────────────────────────────────────
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(str(onnx_dir / "model.onnx"))
    sess = ort.InferenceSession(str(onnx_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    feeds = {k: v.numpy() for k, v in dummy.items()}
    outs = sess.run(None, feeds)
    names = [o.name for o in sess.get_outputs()]
    print("  ort outputs:", list(zip(names, [o.shape for o in outs])))
    for i, name in enumerate(names):
        if name in ("start_logits", "end_logits"):
            err = float(((outs[i] - ref[i].numpy()) ** 2).mean() ** 0.5)
            print(f"  RMSE {name}: {err:.6f}")
    # Candidate slots are score-ordered; topk tie order can differ between
    # eager torch and the traced graph. Compare as SETS keyed by (query,
    # start, end): that is what decode consumes.
    ort_idx, ort_logit, ort_valid = outs[2], outs[3], outs[4]
    t_idx, t_logit, t_valid = ref[2].numpy(), ref[3].numpy(), ref[4].numpy()
    max_diff, matched, unmatched = 0.0, 0, 0
    for b in range(ort_idx.shape[0]):
        for q in range(ort_idx.shape[1]):
            t_map = {}
            for c in range(t_idx.shape[2]):
                if t_valid[b, q, c]:
                    key = (int(t_idx[b, q, c, 0]), int(t_idx[b, q, c, 1]))
                    t_map[key] = float(t_logit[b, q, c])
            for c in range(ort_idx.shape[2]):
                if ort_valid[b, q, c]:
                    key = (int(ort_idx[b, q, c, 0]), int(ort_idx[b, q, c, 1]))
                    if key in t_map:
                        matched += 1
                        max_diff = max(max_diff, abs(t_map[key] - float(ort_logit[b, q, c])))
                    else:
                        unmatched += 1
    print(f"  candidate set check: matched={matched} unmatched(ORT-only)={unmatched} "
          f"max|Δlogit|={max_diff:.6f}")

    # ── Decode parity vs AutoExtractor (optional, strongest check) ─────
    if parity:
        print("  decode parity vs AutoExtractor ...")
        text = "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday."
        labels = ["company", "person", "product", "location"]
        # Reference: the full pipeline
        ref_result = model.extract_entities(text, labels, include_confidence=True, include_spans=True)
        # Our path: pack with the model's own processor, run wrapper, decode pairs
        batch = model.processor.collate_fn_inference([(text, {"entities": {k: [] for k in labels}})], architecture="boundary")
        with torch.no_grad():
            pout = wrapper(
                batch.input_ids, batch.attention_mask,
                batch.text_word_indices, batch.text_word_mask,
                batch.query_marker_indices, batch.query_marker_mask,
            )
        start_logits, end_logits, pair_indices, pair_logits, pair_valid = pout
        probs = torch.sigmoid(pair_logits / pair_temperature)
        # candidates above 0.5 for query 0 (company) — print top spans per query
        q_names = [spec for spec in labels]
        n_q = pair_indices.shape[1]
        print(f"    pair_temperature = {pair_temperature}")
        for qi in range(min(n_q, len(q_names))):
            valid = pair_valid[0, qi].bool()
            top = probs[0, qi][valid].topk(min(3, int(valid.sum())))
            for score, ci in zip(top.values.tolist(), top.indices.tolist()):
                s, e = pair_indices[0, qi, ci].tolist()
                print(f"    q={q_names[qi]!r} span=({s},{e}) p={score:.3f}")
        print(f"    AutoExtractor reference: {json.dumps(ref_result)[:400]}")

    # ── Save tokenizer + configs ────────────────────────────────────────
    model.processor.tokenizer.save_pretrained(str(out))
    cfg = {
        "architecture": "boundary",
        "export_version": 2,
        "base_model": model_id,
        "candidate_budget": int(candidate_budget),
        "pair_temperature": float(pair_temperature),
        "opset": 17,
        "inputs": input_names,
        "outputs": output_names,
        "notes": "Graph includes proposer + pair reranker (vectorized). Host: keep spans with sigmoid(pair_logits / pair_temperature) >= threshold, resolve overlaps per label, map boundaries to chars (boundary i = before word i; pair (s,e) covers words s..e-1).",
    }
    (out / "export_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"  output at {out}")
    return {"onnx_mb": size_mb, "candidate_budget": candidate_budget}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    processor = parser.add_argument("--out-dir", default="./output-v2")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--n-queries", type=int, default=4)
    parser.add_argument("--n-words", type=int, default=48)
    parser.add_argument("--no-parity", action="store_true")
    args = parser.parse_args()
    export_one(
        model_id=args.model_id,
        out_dir=args.out_dir,
        seq_len=args.seq_len,
        n_queries=args.n_queries,
        n_words=args.n_words,
        parity=not args.no_parity,
    )


if __name__ == "__main__":
    main()
