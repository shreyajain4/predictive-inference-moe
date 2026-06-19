#!/bin/bash
# train_mixtral_lr_from_trace.sh
# Given a routing CSV from llama-moe-trace on Mixtral 8x22B, train a per-layer
# LR-no-hidden predictor (multihot of prev routing → next-layer top-k logits).
# Output a Mixtral-shaped .bin compatible with the bench's predictor loader.
#
# Usage:
#   bash train_mixtral_lr_from_trace.sh <routing.csv> <out.bin>
#
# Default: looks for the most recent /tmp/mixtral_history_*/mixtral_routing.csv
# and writes to ~/mixtral_lr_predictor.bin

set -u

ROUTING_CSV=${1:-$(ls -t /tmp/mixtral_history_*/mixtral_routing.csv 2>/dev/null | head -1)}
OUT_BIN=${2:-$HOME/mixtral_lr_predictor.bin}

if [ -z "$ROUTING_CSV" ] || [ ! -f "$ROUTING_CSV" ]; then
  echo "ERROR: no routing CSV found. Pass path explicitly or run the prewarm bench first."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK=/tmp/mixtral_lr_train_$$
mkdir -p "$WORK"
echo "routing CSV: $ROUTING_CSV"
echo "output bin:  $OUT_BIN"
echo "work dir:    $WORK"
echo ""

# Step 1: convert routing CSV → parquet (p, t, l, e columns)
echo "=== Step 1: CSV → parquet ==="
PARQUET=$WORK/routing.parquet
python3 "$SCRIPT_DIR/moe_trace_csv_to_parquet.py" \
  --csv "$ROUTING_CSV" --out "$PARQUET" 2>&1 | tail -5

# Step 2: train per-layer LR via lr_governor_sweep.py (skip-sweep = train only)
echo ""
echo "=== Step 2: train per-layer LR-no-hidden ==="
CKPT=$WORK/mixtral_lr.pt
python3 "$REPO_ROOT/experiments/lr_governor_sweep.py" \
  --trace "$PARQUET" \
  --epochs 6 \
  --device cpu \
  --train-frac 0.85 \
  --split-mode token \
  --skip-sweep \
  --checkpoint "$CKPT" 2>&1 | tail -25

if [ ! -f "$CKPT" ]; then
  echo "ERROR: training failed; .pt not produced"
  exit 1
fi

# Step 3: convert .pt → .bin (zero-pad hidden region since LR-no-hidden has no hidden weights)
echo ""
echo "=== Step 3: convert .pt → .bin (zero-pad hidden region) ==="
python3 - "$CKPT" "$OUT_BIN" << 'PY'
import sys, struct, torch, numpy as np

CKPT, OUT = sys.argv[1], sys.argv[2]
state = torch.load(CKPT, map_location="cpu", weights_only=False)
print(f"loaded checkpoint: keys={list(state.keys())[:5]}...")

# lr_governor_sweep saves per-layer LR weights. Schema typically:
#   state["weights"][layer_idx] = (W, b)
#   W shape: (num_experts, in_features), b shape: (num_experts,)
# where in_features = 2 * num_experts (multihot prev routing features).
#
# The bench's predictor expects feature_dim = 2 * num_experts + hidden_dim.
# For Mixtral with hidden=6144, that's 16 + 6144 = 6160. We zero-pad the
# hidden component since LR-no-hidden was trained without it.

# Try multiple checkpoint layouts
W_per_layer = None
if "weights" in state and isinstance(state["weights"], dict):
    W_per_layer = state["weights"]
elif "per_layer_weights" in state:
    W_per_layer = state["per_layer_weights"]
elif "state_dict" in state:
    # PyTorch Module checkpoint — extract by key
    sd = state["state_dict"]
    W_per_layer = {}
    for k, v in sd.items():
        if "layer" in k:
            print(f"  found {k}: {v.shape}")
# Fallback: dump all keys for diagnosis
if W_per_layer is None:
    print(f"unknown checkpoint layout. Keys: {list(state.keys())}")
    if hasattr(state, "items"):
        for k, v in state.items():
            print(f"  {k}: {type(v).__name__}")
    sys.exit(2)

# Pull out config
n_experts = state.get("num_experts", state.get("n_experts", 8))
n_expert_used = state.get("n_expert_used", state.get("top_k", 2))
n_layers = state.get("n_layers", 56)
hidden = 6144   # Mixtral 8x22B
feature_dim = 2 * n_experts + hidden  # 6160
first_layer = state.get("first_layer", 0)
print(f"config: n_layers={n_layers}, num_experts={n_experts}, hidden={hidden}, "
      f"feature_dim={feature_dim}, n_expert_used={n_expert_used}, first_L={first_layer}")

# Header
MAGIC = 0x4D4F4550
VERSION = 2
with open(OUT, "wb") as f:
    f.write(struct.pack("<8I", MAGIC, VERSION, n_layers, feature_dim,
                         n_experts, hidden, first_layer, n_expert_used))
    # Body: per-layer, per-expert: feature_dim weights + 1 bias = (feature_dim + 1) floats
    # Layout: [num_experts, feature_dim + 1] per layer
    # Multihot region: positions [0, 2*n_experts). Hidden region: [2*n_experts, feature_dim).
    n_padded = 0
    for L in range(n_layers):
        if L in W_per_layer:
            W, b = W_per_layer[L]
            W_np = W.detach().cpu().numpy() if hasattr(W, "detach") else np.asarray(W)
            b_np = b.detach().cpu().numpy() if hasattr(b, "detach") else np.asarray(b)
            assert W_np.shape == (n_experts, 2 * n_experts), \
                f"W shape {W_np.shape} != ({n_experts}, {2*n_experts})"
            for e in range(n_experts):
                row = np.zeros(feature_dim + 1, dtype=np.float32)
                row[:2 * n_experts] = W_np[e]   # routing weights
                # hidden region [2*n_experts:feature_dim] stays zero
                row[feature_dim] = float(b_np[e])
                f.write(row.tobytes())
        else:
            n_padded += 1
            # Zero-pad missing layer
            f.write(b"\x00" * (n_experts * (feature_dim + 1) * 4))

import os
sz = os.path.getsize(OUT)
print(f"wrote {OUT}: {sz/1e6:.1f} MB ({n_padded} layers zero-padded for missing trainings)")
PY

if [ ! -f "$OUT_BIN" ]; then
  echo "ERROR: bin conversion failed"
  exit 1
fi
ls -lh "$OUT_BIN"

echo ""
echo "=== Step 4: verify the bench loads it ==="
$HOME/llama.cpp/build/bin/llama-moe-predictor-bench --help > /dev/null 2>&1
$HOME/llama.cpp/build/bin/llama-moe-predictor-bench \
  -m $HOME/models/Mixtral-8x22B-Instruct-v0.1.IQ4_XS-00001-of-00002.gguf \
  -c 512 -n 1 \
  --predictor-weights "$OUT_BIN" \
  --prefetch-k 0 \
  -p "test" 2>&1 | grep -E "loaded|predictor:|failed" | head -5

echo ""
echo "Done. Predictor saved to: $OUT_BIN"
echo "To use in bench: --predictor-weights $OUT_BIN --prefetch-k 12"
