"""
lr_governor_sweep.py
--------------------
End-to-end cache hit-rate sweep with the LR-past predictor wired into a
governor-style simulator.

Inputs:
    - Routing trace (parquet) — full dataset on disk
    - Per-prompt 80/20 token split (matches markov_recall.py / lr_pastexperts_recall.py)

Method:
    1. Reorganize events into per-prompt arrays.
    2. Train one Linear(128, 64) per layer using BCE on the 128-dim input
       [multi-hot of (t-1, L) experts ⊕ multi-hot of (t, L-1) experts].
       For layer L == first_layer, Feature B falls back to (t-1, last_layer).
    3. For each (cache_capacity, k_prefetch) configuration, run a sequential
       simulator that:
         - Processes events in (prompt, token, layer) order.
         - Cache is OrderedDict keyed by (layer, expert_id) — the corrected
           keying from the recently-fixed governor.
         - Counts hits/misses on EVAL tokens only (train tokens warm the cache).
         - When a predictor is configured, prefetches top-k_pref predicted
           experts for layer L+1 right after each layer-L event.

Output: a table of the form
    cap   % total   baseline   k_pref=6   k_pref=8   k_pref=12

Usage:
    python experiments/lr_governor_sweep.py \\
        --trace /Users/shreyajain/llama.cpp/expert_traces.parquet \\
        --caps 30,100,200,500,1000 \\
        --k-prefs 6,8,12 \\
        --output results/lr_governor_sweep.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Data loading & reorganization
# ---------------------------------------------------------------------------

def load_packed(path: Path, with_tids: bool = False) -> dict:
    """
    Read parquet (p, t, l, e [, tid]) and return per-prompt structured data:
        prompts[p] = {
            "tokens":  int32 array of token positions, sorted
            "experts": int8 array of shape (num_tokens, num_layers, top_k)
                       where layer-axis is 0-indexed (layer 1 -> idx 0)
            "tids":    int32 array of vocab token IDs, length num_tokens
                       (only present when with_tids=True and the parquet has a `tid` column)
        }
    """
    print(f"reading parquet ...", flush=True)
    t0 = time.time()
    cols = ["p", "t", "l", "e"] + (["tid"] if with_tids else [])
    try:
        df = pd.read_parquet(path, columns=cols)
    except Exception:
        if with_tids:
            print("  parquet has no 'tid' column — falling back to no-tid load", flush=True)
            df = pd.read_parquet(path, columns=["p", "t", "l", "e"])
            with_tids = False
        else:
            raise
    print(f"  loaded {len(df):,} rows in {time.time()-t0:.1f}s", flush=True)

    first_layer = int(df["l"].min())
    last_layer = int(df["l"].max())
    n_layers = last_layer - first_layer + 1
    top_k = len(df["e"].iloc[0])
    print(f"  layers {first_layer}..{last_layer} ({n_layers}), top_k={top_k}", flush=True)

    print("  reorganizing per-prompt ...", flush=True)
    t0 = time.time()
    df_sorted = df.sort_values(["p", "t", "l"]).reset_index(drop=True)
    e_arr = np.stack(df_sorted["e"].to_numpy()).astype(np.int8)
    p_arr = df_sorted["p"].to_numpy().astype(np.int32)
    t_arr = df_sorted["t"].to_numpy().astype(np.int32)
    l_arr = df_sorted["l"].to_numpy().astype(np.int8)
    tid_arr = df_sorted["tid"].to_numpy().astype(np.int32) if with_tids else None

    prompts: dict[int, dict] = {}
    n_total = len(df_sorted)
    i = 0
    while i < n_total:
        p = int(p_arr[i])
        j = i
        while j < n_total and p_arr[j] == p:
            j += 1
        # rows [i:j) all belong to prompt p
        block_t = t_arr[i:j]
        block_l = l_arr[i:j]
        block_e = e_arr[i:j]
        uniq_tokens, first_occurrence = np.unique(block_t, return_index=True)
        ntok = len(uniq_tokens)
        tok_to_idx = {int(t): k for k, t in enumerate(uniq_tokens)}

        experts = np.full((ntok, n_layers, top_k), -1, dtype=np.int8)
        for r in range(j - i):
            ti = tok_to_idx[int(block_t[r])]
            li = int(block_l[r]) - first_layer
            experts[ti, li] = block_e[r]

        entry = {
            "tokens": uniq_tokens.astype(np.int32),
            "experts": experts,
        }
        if with_tids and tid_arr is not None:
            # token_id is the same for all layers of one token — take the first occurrence.
            entry["tids"] = tid_arr[i + first_occurrence].astype(np.int32)
        prompts[p] = entry
        i = j

    print(f"  reorganized {len(prompts):,} prompts in {time.time()-t0:.1f}s", flush=True)
    return {
        "prompts": prompts,
        "first_layer": first_layer,
        "last_layer": last_layer,
        "n_layers": n_layers,
        "top_k": top_k,
        "n_experts": int(np.max([np.max(d["experts"]) for d in prompts.values()])) + 1,
    }


def split_tokens(
    prompts: dict[int, dict], train_frac: float, mode: str = "token",
) -> dict[int, dict]:
    """
    Returns {prompt: {'train_mask': bool[ntok], 'eval_mask': bool[ntok]}}.

    mode = "token":  per-prompt token split — first train_frac of tokens in
                     every prompt are train, the rest are eval (current default).

    mode = "prompt": prompt-level split — first train_frac of prompts (sorted by
                     prompt id) have train_mask = all True / eval_mask = all False,
                     and vice versa for the remaining prompts. This matches the
                     Markov sweep's 1593-train / 399-held-out methodology and is
                     a strict generalization test.
    """
    if mode not in ("token", "prompt"):
        raise ValueError(f"unknown split mode: {mode}")

    if mode == "token":
        out = {}
        for p, d in prompts.items():
            n = len(d["tokens"])
            cut = int(train_frac * n)
            train_mask = np.zeros(n, dtype=bool)
            eval_mask = np.zeros(n, dtype=bool)
            train_mask[:cut] = True
            eval_mask[cut:] = True
            out[p] = {"train_mask": train_mask, "eval_mask": eval_mask}
        return out

    # mode == "prompt": split entire prompts.
    pids = sorted(prompts.keys())
    cut = int(train_frac * len(pids))
    train_pids = set(pids[:cut])
    out = {}
    for p, d in prompts.items():
        n = len(d["tokens"])
        if p in train_pids:
            out[p] = {
                "train_mask": np.ones(n, dtype=bool),
                "eval_mask":  np.zeros(n, dtype=bool),
            }
        else:
            out[p] = {
                "train_mask": np.zeros(n, dtype=bool),
                "eval_mask":  np.ones(n, dtype=bool),
            }
    return out


# ---------------------------------------------------------------------------
# LR training per layer
# ---------------------------------------------------------------------------

def _multihot_batch(expert_arr: np.ndarray, num_experts: int) -> np.ndarray:
    """expert_arr shape (B, top_k) int -> multi-hot (B, num_experts) float32."""
    B = expert_arr.shape[0]
    out = np.zeros((B, num_experts), dtype=np.float32)
    valid = expert_arr >= 0
    rows = np.repeat(np.arange(B), expert_arr.shape[1])
    cols = expert_arr.reshape(-1).astype(np.int64)
    mask = valid.reshape(-1)
    out[rows[mask], cols[mask]] = 1.0
    return out


def build_lr_examples(
    prompts: dict[int, dict],
    splits: dict[int, dict],
    target_layer_idx: int,
    first_layer_idx: int,
    last_layer_idx: int,
    num_experts: int,
    split_key: str = "train_mask",
    max_examples: int | None = None,
    embed_table: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, Y) for predicting layer at target_layer_idx (0-indexed in layer axis).

    If embed_table is provided, each sample's feature vector is concatenated
    with the current token's embedding (looked up by tids[i] from each prompt's
    `tids` array, which must be populated). Feature dim becomes
    2 * num_experts + embed_table.shape[1] when embeddings are used.
    """
    feat_A_rows = []
    feat_B_rows = []
    feat_embed_rows: list[np.ndarray] = []
    y_rows = []
    use_embed = embed_table is not None

    for p, d in prompts.items():
        experts = d["experts"]               # (ntok, n_layers, top_k)
        tokens = d["tokens"]                 # (ntok,)
        mask = splits[p][split_key]
        ntok = len(tokens)
        if ntok < 2:
            continue
        tids = d.get("tids") if use_embed else None
        if use_embed and tids is None:
            # No tids in this prompt's data — can't use embeddings for it.
            continue
        for i in range(ntok):
            if not mask[i]:
                continue
            tgt = experts[i, target_layer_idx]
            if tgt[0] < 0:
                continue  # no event at this layer for this token

            # Feature A: (t-1, target_layer)
            feat_A = np.zeros(num_experts, dtype=np.float32)
            if i > 0 and tokens[i] == tokens[i - 1] + 1:
                prev = experts[i - 1, target_layer_idx]
                if prev[0] >= 0:
                    for e in prev:
                        if e >= 0:
                            feat_A[e] = 1.0

            # Feature B: (t, target_layer-1) or fallback to (t-1, last_layer)
            feat_B = np.zeros(num_experts, dtype=np.float32)
            if target_layer_idx > first_layer_idx:
                prev_l = experts[i, target_layer_idx - 1]
                if prev_l[0] >= 0:
                    for e in prev_l:
                        if e >= 0:
                            feat_B[e] = 1.0
            else:
                if i > 0 and tokens[i] == tokens[i - 1] + 1:
                    prev_last = experts[i - 1, last_layer_idx]
                    if prev_last[0] >= 0:
                        for e in prev_last:
                            if e >= 0:
                                feat_B[e] = 1.0

            # Target
            y = np.zeros(num_experts, dtype=np.float32)
            for e in tgt:
                if e >= 0:
                    y[e] = 1.0

            feat_A_rows.append(feat_A)
            feat_B_rows.append(feat_B)
            if use_embed:
                tid = int(tids[i])
                if 0 <= tid < embed_table.shape[0]:
                    feat_embed_rows.append(embed_table[tid].astype(np.float32))
                else:
                    feat_embed_rows.append(np.zeros(embed_table.shape[1], dtype=np.float32))
            y_rows.append(y)

            if max_examples is not None and len(y_rows) >= max_examples:
                break
        if max_examples is not None and len(y_rows) >= max_examples:
            break

    if not y_rows:
        return (
            np.zeros((0, 2 * num_experts), dtype=np.float32),
            np.zeros((0, num_experts), dtype=np.float32),
        )

    if use_embed and feat_embed_rows:
        X = np.concatenate(
            [np.stack(feat_A_rows), np.stack(feat_B_rows), np.stack(feat_embed_rows)],
            axis=1,
        )
    else:
        X = np.concatenate([np.stack(feat_A_rows), np.stack(feat_B_rows)], axis=1)
    Y = np.stack(y_rows)
    return X, Y


def train_lr(
    X: np.ndarray, Y: np.ndarray, num_experts: int,
    epochs: int = 6, batch_size: int = 4096, lr: float = 5e-2,
    device: str = "cpu",
) -> nn.Linear:
    """Mini-batch LR with BCEWithLogitsLoss."""
    Xt = torch.from_numpy(X).to(device)
    Yt = torch.from_numpy(Y).to(device)
    model = nn.Linear(X.shape[1], num_experts).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    N = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        for s in range(0, N, batch_size):
            idx = perm[s:s + batch_size]
            opt.zero_grad()
            logits = model(Xt[idx])
            loss = loss_fn(logits, Yt[idx])
            loss.backward()
            opt.step()
    return model


# ---------------------------------------------------------------------------
# Governor-style simulation
# ---------------------------------------------------------------------------

def precompute_prefetches(
    prompts: dict[int, dict],
    predictors: dict[int, nn.Linear],
    num_experts: int,
    n_layers: int,
    max_k: int,
    horizon: int = 1,
) -> dict[int, np.ndarray]:
    """
    Multi-horizon chained prefetch precompute.

    At simulation step (token i, current layer L_cur), prefetches[i, L_cur, h-1, :k]
    is the top-k expert IDs predicted to fire at layer L_cur + h, for h in 1..horizon.

    Chain semantics for h >= 2:
        Feature A is always experts@(t-1, L_cur + h) — known from history.
        Feature B is:
          - h == 1: multi-hot of ACTUAL experts@(t, L_cur).
          - h >  1: multi-hot of the predicted top-`top_k_actual` experts
                    from the chain's previous horizon (i.e., L_cur + h - 1).
    Errors compound across h — that's the realistic runtime model where
    ground-truth future experts aren't available.

    Returns {p: int16 array of shape (ntok, n_layers - 1, horizon, max_k)}.
    For horizon=1 this matches the previous single-step layout (with an
    extra trailing size-1 dim).
    """
    # Stack predictor weights/bias for fast vectorized matmul.
    pred_W = np.zeros((n_layers, num_experts, 2 * num_experts), dtype=np.float32)
    pred_b = np.zeros((n_layers, num_experts), dtype=np.float32)
    for L_idx, model in predictors.items():
        pred_W[L_idx] = model.weight.detach().cpu().numpy()
        pred_b[L_idx] = model.bias.detach().cpu().numpy()

    out: dict[int, np.ndarray] = {}
    for p, d in prompts.items():
        experts = d["experts"]                      # (ntok, n_layers, top_k) int8
        tokens = d["tokens"]                        # (ntok,) int32
        ntok = len(tokens)
        top_k_actual = experts.shape[2]

        # Per-(token, layer) multi-hot of actual experts.
        multihot = np.zeros((ntok, n_layers, num_experts), dtype=np.float32)
        for k in range(top_k_actual):
            slot = experts[:, :, k]
            valid = slot >= 0
            ri, ci = np.where(valid)
            multihot[ri, ci, slot[ri, ci]] = 1.0

        adj = np.zeros(ntok, dtype=bool)
        if ntok > 1:
            adj[1:] = (tokens[1:] == tokens[:-1] + 1)

        prefetches = np.full((ntok, n_layers - 1, horizon, max_k), -1, dtype=np.int16)

        for L_cur in range(n_layers - 1):
            # Initialize chain Feature B with actuals at L_cur.
            chain_feat_B = multihot[:, L_cur, :].copy()             # (ntok, E)

            for h in range(1, horizon + 1):
                L_target = L_cur + h
                if L_target >= n_layers:
                    break

                # Feature A: previous adjacent token's experts at L_target.
                feat_A = np.zeros((ntok, num_experts), dtype=np.float32)
                if ntok > 1:
                    feat_A[1:][adj[1:]] = multihot[:-1, L_target, :][adj[1:]]

                X = np.concatenate([feat_A, chain_feat_B], axis=1)  # (ntok, 2E)
                logits = X @ pred_W[L_target].T + pred_b[L_target]  # (ntok, E)

                top_unsorted = np.argpartition(-logits, max_k, axis=1)[:, :max_k]
                top_vals = np.take_along_axis(logits, top_unsorted, axis=1)
                order = np.argsort(-top_vals, axis=1)
                top_sorted = np.take_along_axis(top_unsorted, order, axis=1)
                prefetches[:, L_cur, h - 1, :] = top_sorted.astype(np.int16)

                # Update chain Feature B with predicted top-`top_k_actual` for next iteration.
                if h < horizon and L_target + 1 < n_layers:
                    k_for_chain = min(top_k_actual, max_k)
                    chain_top = top_sorted[:, :k_for_chain]
                    chain_feat_B = np.zeros((ntok, num_experts), dtype=np.float32)
                    rows = np.repeat(np.arange(ntok), k_for_chain)
                    cols = chain_top.reshape(-1).astype(np.int64)
                    chain_feat_B[rows, cols] = 1.0

        out[p] = prefetches
    return out


def simulate(
    prompts: dict[int, dict],
    splits: dict[int, dict],
    cap: int,
    k_pref: int,
    num_experts: int,
    n_layers: int,
    prefetches: dict[int, np.ndarray] | None = None,
    horizon: int = 1,
    track_per_layer: bool = False,
) -> tuple[int, int] | tuple[int, int, dict, dict]:
    """
    Multi-horizon streaming simulator. Cache key = layer_idx * num_experts + expert_id.

    At each step (token i, current layer L_cur), in addition to processing the
    actual experts, we prefetch top-`k_pref` experts for EACH of layers
    L_cur+1, L_cur+2, ..., L_cur+horizon (skipping any that go past n_layers).

    Prefetches array layout is (ntok, n_layers - 1, horizon, max_k); we slice
    the first k_pref of the last dim. The prefetches dict comes from
    precompute_prefetches(..., horizon=H).
    """
    hits = 0
    misses = 0
    E = num_experts
    per_layer_hits: dict[int, int] = {}
    per_layer_misses: dict[int, int] = {}

    for p, d in prompts.items():
        cache: OrderedDict[int, int] = OrderedDict()
        experts = d["experts"]
        emask = splits[p]["eval_mask"]
        ntok = len(emask)
        pre = prefetches[p] if prefetches is not None else None

        for i in range(ntok):
            count = bool(emask[i])
            for L_idx in range(n_layers):
                tgt = experts[i, L_idx]
                if tgt[0] < 0:
                    continue
                # Touch the actuals (single-int key).
                for e in tgt:
                    if e < 0:
                        continue
                    key = L_idx * E + int(e)
                    if key in cache:
                        cache.move_to_end(key)
                        if count:
                            hits += 1
                            if track_per_layer:
                                per_layer_hits[L_idx] = per_layer_hits.get(L_idx, 0) + 1
                    else:
                        cache[key] = 0
                        if count:
                            misses += 1
                            if track_per_layer:
                                per_layer_misses[L_idx] = per_layer_misses.get(L_idx, 0) + 1
                        while len(cache) > cap:
                            cache.popitem(last=False)

                # Multi-horizon prefetch: layers L_idx+1, L_idx+2, ..., L_idx+horizon.
                if pre is not None and k_pref > 0 and L_idx + 1 < n_layers:
                    for h_idx in range(horizon):
                        L_target = L_idx + 1 + h_idx
                        if L_target >= n_layers:
                            break
                        row = pre[i, L_idx, h_idx]  # (max_k,) sorted descending
                        for jj in range(k_pref):
                            eid = int(row[jj])
                            if eid < 0:
                                continue
                            key = L_target * E + eid
                            if key in cache:
                                cache.move_to_end(key)
                            else:
                                cache[key] = 0
                                while len(cache) > cap:
                                    cache.popitem(last=False)

    if track_per_layer:
        return hits, misses, per_layer_hits, per_layer_misses
    return hits, misses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--caps", default="30,100,200,500,1000")
    ap.add_argument("--k-prefs", default="6,8,12")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--max-examples-per-layer", type=int, default=300_000,
                    help="Cap training examples per layer (for speed). 0 = no cap.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="Path to save trained per-layer LR weights (.pt)")
    ap.add_argument("--per-layer-breakdown", action="store_true",
                    help="Also report per-layer hit rates in the table.")
    ap.add_argument("--horizon", type=int, default=1,
                    help="Multi-layer lookahead horizon. At each step, prefetch top-k for layers L+1..L+H. "
                         "H>=2 uses chained predictions (Feature B = previous horizon's top-6 predictions).")
    ap.add_argument("--split-mode", default="token", choices=["token", "prompt"],
                    help="token: per-prompt 80/20 token split (training data within each prompt). "
                         "prompt: prompt-level split (held-out prompts entirely unseen by LR). "
                         "Use 'prompt' for apples-to-apples with the Markov 1593/399 sweep.")
    ap.add_argument("--use-embeddings", action="store_true",
                    help="Add a frozen token-embedding feature to the LR input. Requires "
                         "--embed-table and a trace with the `tid` column.")
    ap.add_argument("--embed-table", default=None,
                    help="Path to .npy embedding table (shape: vocab × dim, float16). "
                         "Required when --use-embeddings is set.")
    ap.add_argument("--skip-sweep", action="store_true",
                    help="Train LR predictors only, skip the cache-hit-rate sweep. "
                         "Use this when training a model for separate downstream eval.")
    args = ap.parse_args()

    caps = [int(x) for x in args.caps.split(",")]
    k_prefs = [int(x) for x in args.k_prefs.split(",")]

    # If we're using token embeddings, we need the `tid` column from the parquet.
    use_embed = bool(args.use_embeddings)
    if use_embed and not args.embed_table:
        sys.exit("--use-embeddings requires --embed-table <path-to.npy>")
    data = load_packed(args.trace, with_tids=use_embed)
    prompts = data["prompts"]
    first_layer = data["first_layer"]
    last_layer = data["last_layer"]
    n_layers = data["n_layers"]
    num_experts = data["n_experts"]
    print(f"  n_experts={num_experts}, total experts in pool={n_layers * num_experts}", flush=True)

    embed_table = None
    if use_embed:
        if "tids" not in next(iter(prompts.values())):
            sys.exit("--use-embeddings set but trace has no `tid` column")
        print(f"loading embedding table: {args.embed_table}", flush=True)
        embed_table = np.load(args.embed_table)
        if embed_table.ndim != 2:
            sys.exit(f"embed table must be 2D, got shape {embed_table.shape}")
        print(f"  shape={embed_table.shape}, dtype={embed_table.dtype} "
              f"({embed_table.nbytes/1024/1024:.0f} MB)", flush=True)

    splits = split_tokens(prompts, args.train_frac, mode=args.split_mode)

    # In prompt-mode, only prompts with any eval tokens need precompute / sim work.
    if args.split_mode == "prompt":
        active_prompts = {p: d for p, d in prompts.items() if splits[p]["eval_mask"].any()}
        n_train_prompts = len(prompts) - len(active_prompts)
        print(f"  split-mode=prompt: {n_train_prompts} train prompts (LR fit), "
              f"{len(active_prompts)} held-out prompts (eval)", flush=True)
    else:
        active_prompts = prompts
        print(f"  split-mode=token: per-prompt 80/20 token split over all "
              f"{len(prompts)} prompts", flush=True)

    print(f"\nTraining {n_layers} per-layer LR predictors (max {args.max_examples_per_layer:,} examples each) ...", flush=True)
    predictors: dict[int, nn.Linear] = {}
    max_ex = args.max_examples_per_layer or None
    for L_idx in range(n_layers):
        t0 = time.time()
        X, Y = build_lr_examples(
            prompts, splits, L_idx,
            first_layer_idx=0, last_layer_idx=n_layers - 1,
            num_experts=num_experts,
            split_key="train_mask",
            max_examples=max_ex,
            embed_table=embed_table,
        )
        if X.shape[0] == 0:
            print(f"  layer idx {L_idx} (real L={L_idx + first_layer}): no examples — skipping",
                  flush=True)
            continue
        model = train_lr(X, Y, num_experts, epochs=args.epochs, device=args.device)
        predictors[L_idx] = model
        n_examples = X.shape[0]
        # Explicit cleanup — large X (17 GB at full data + embeddings) must be freed
        # before allocating the next layer's matrix, else CPython's lazy GC will OOM us.
        del X, Y
        import gc as _gc
        _gc.collect()
        print(f"  layer idx {L_idx} (real L={L_idx + first_layer}): "
              f"trained on {n_examples:,} examples in {time.time()-t0:.1f}s",
              flush=True)

    # ── Persist all 26 per-layer LR predictors so we can re-use them ───────
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "predictors": {
                int(L_idx): {
                    "weight": predictors[L_idx].weight.detach().cpu(),
                    "bias":   predictors[L_idx].bias.detach().cpu(),
                }
                for L_idx in predictors
            },
            "config": {
                "num_experts": num_experts,
                "n_layers":    n_layers,
                "first_layer": first_layer,
                "last_layer":  last_layer,
                "feature_dim": (2 * num_experts) + (embed_table.shape[1] if use_embed else 0),
                "feature_spec": (
                    "concat(multihot(experts@(t-1,L)), multihot(experts@(t,L-1) or (t-1,last_layer) if L==first)"
                    + (f", token_embed[tid_{{t}}])" if use_embed else ")")
                ),
                "use_embeddings": use_embed,
                "token_embed_dim": int(embed_table.shape[1]) if use_embed else 0,
                "embed_table_path": str(args.embed_table) if use_embed else None,
                "train_max_examples_per_layer": args.max_examples_per_layer,
                "epochs":      args.epochs,
            },
        }
        import torch as _torch
        _torch.save(state, ckpt_path)
        print(f"\nsaved {len(state['predictors'])} per-layer LR predictors → {ckpt_path} "
              f"({ckpt_path.stat().st_size/1024:.1f} KB)", flush=True)

    # ── Optionally skip the sweep when we only want a checkpoint for downstream eval ──
    if args.skip_sweep:
        print("\n--skip-sweep set: training only, exiting before cache-hit sweep.", flush=True)
        return

    if use_embed:
        # precompute_prefetches doesn't yet thread the embed_table through. For
        # a sweep with embeddings we'd need to add that; skip for now.
        print("\nWARNING: --use-embeddings with --skip-sweep recommended. "
              "precompute_prefetches doesn't yet wire embeddings — sweep would crash. "
              "Exiting.", flush=True)
        return

    # ── Precompute prefetches once (max_k = max of k_prefs) ────────────────
    max_k = max(k_prefs)
    print(f"\nPrecomputing top-{max_k} chained prefetches "
          f"(horizon={args.horizon}) per (prompt, token, layer) ...", flush=True)
    t0 = time.time()
    all_prefetches = precompute_prefetches(
        active_prompts, predictors, num_experts=num_experts,
        n_layers=n_layers, max_k=max_k, horizon=args.horizon,
    )
    print(f"  done in {time.time()-t0:.1f}s "
          f"({sum(v.nbytes for v in all_prefetches.values())/1024/1024:.1f} MB)",
          flush=True)

    # ── Sweep ──────────────────────────────────────────────────────────────
    total_experts_in_pool = n_layers * num_experts
    rows = []
    track = args.per_layer_breakdown
    print(f"\nSweeping (cap × k_pref) over {len(caps)*(1+len(k_prefs))} runs"
          f"{' [per-layer tracking on]' if track else ''} ...", flush=True)
    for cap in caps:
        pct = 100.0 * cap / total_experts_in_pool
        row = {"cap": cap, "pct": pct}

        # Baseline (no predictor / no prefetch).
        t0 = time.time()
        sim_out = simulate(
            active_prompts, splits, cap=cap, k_pref=0,
            num_experts=num_experts, n_layers=n_layers,
            prefetches=None, track_per_layer=track,
        )
        if track:
            h, m, pl_h, pl_m = sim_out
            row["per_layer_baseline"] = {
                int(L): pl_h.get(L, 0) / max(1, pl_h.get(L, 0) + pl_m.get(L, 0))
                for L in range(n_layers)
            }
        else:
            h, m = sim_out
        hr = h / (h + m) if (h + m) else 0.0
        row["baseline"] = hr
        print(f"  cap={cap:>5}  baseline  hr={100*hr:5.2f}%  ({time.time()-t0:.1f}s)", flush=True)

        for kp in k_prefs:
            t0 = time.time()
            sim_out = simulate(
                active_prompts, splits, cap=cap, k_pref=kp,
                num_experts=num_experts, n_layers=n_layers,
                prefetches=all_prefetches, horizon=args.horizon,
                track_per_layer=track,
            )
            if track:
                h, m, pl_h, pl_m = sim_out
                row[f"per_layer_k_pref={kp}"] = {
                    int(L): pl_h.get(L, 0) / max(1, pl_h.get(L, 0) + pl_m.get(L, 0))
                    for L in range(n_layers)
                }
            else:
                h, m = sim_out
            hr = h / (h + m) if (h + m) else 0.0
            row[f"k_pref={kp}"] = hr
            print(f"  cap={cap:>5}  k_pref={kp:>2}   hr={100*hr:5.2f}%  ({time.time()-t0:.1f}s)", flush=True)
        rows.append(row)

    # Per-layer breakdown table — printed only if tracking was on.
    if track:
        print(f"\n{'='*92}")
        print("Per-layer hit rate (real layer L = idx + first_layer)")
        print("="*92)
        for row in rows:
            cap = row["cap"]
            print(f"\ncap={cap}")
            hdr = "  L_real   " + "baseline".rjust(10) + "  " + "  ".join(f"k_pref={kp}".rjust(10) for kp in k_prefs)
            print(hdr)
            for L_idx in range(n_layers):
                cells = [f"{100*row['per_layer_baseline'].get(L_idx, 0.0):>9.2f}%"]
                for kp in k_prefs:
                    cells.append(f"{100*row[f'per_layer_k_pref={kp}'].get(L_idx, 0.0):>9.2f}%")
                print(f"  L={L_idx + first_layer:>3}    " + "  ".join(cells))

    # ── Print final table ──────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("Cache hit rate (eval tokens, train tokens warm the cache)")
    print("="*72)
    hdr = f"{'cap':>5}\t{'% total':>7}\t{'baseline':>9}"
    for kp in k_prefs:
        hdr += f"\t{'k_pref='+str(kp):>10}"
    print(hdr)
    for r in rows:
        line = f"{r['cap']:>5}\t{r['pct']:>6.1f}%\t{100*r['baseline']:>8.1f}%"
        for kp in k_prefs:
            line += f"\t{100*r['k_pref='+str(kp)]:>9.1f}%"
        print(line)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "trace": str(args.trace),
                "caps": caps,
                "k_prefs": k_prefs,
                "n_experts_per_layer": num_experts,
                "n_layers": n_layers,
                "total_experts_in_pool": total_experts_in_pool,
                "rows": rows,
            }, f, indent=2)
        print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
