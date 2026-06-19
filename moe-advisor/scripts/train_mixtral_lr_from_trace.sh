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
print(f"loaded checkpoint: keys={list(state.keys())}")

# lr_governor_sweep saves: {"predictors": {layer_idx: model_or_tensors}, "config": {...}}
# Each per-layer predictor is typically a nn.Linear with .weight (out, in) and .bias (out)
# OR a tuple (W, b).
predictors = state.get("predictors", {})
config = state.get("config", {})
print(f"config: {config}")
print(f"predictors: {len(predictors)} layers, sample key={next(iter(predictors)) if predictors else 'NONE'}")
if predictors:
    sample = next(iter(predictors.values()))
    print(f"sample predictor type={type(sample).__name__}")
    if hasattr(sample, "weight"):
        print(f"  .weight shape: {sample.weight.shape}, dtype: {sample.weight.dtype}")
        print(f"  .bias shape: {sample.bias.shape if sample.bias is not None else None}")
    elif isinstance(sample, dict):
        print(f"  dict keys: {list(sample.keys())}")
    elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
        print(f"  tuple len={len(sample)}, first={type(sample[0]).__name__}")

# Extract W (num_experts, in_features), b (num_experts) per layer
def extract(pred):
    if hasattr(pred, "weight"):
        W = pred.weight.detach().cpu().numpy()
        b = pred.bias.detach().cpu().numpy() if pred.bias is not None else np.zeros(W.shape[0], dtype=np.float32)
        return W, b
    if isinstance(pred, dict):
        # State dict-like
        W = None; b = None
        for k, v in pred.items():
            if "weight" in k.lower(): W = v
            if "bias" in k.lower(): b = v
        if W is None: return None
        W = W.detach().cpu().numpy() if hasattr(W, "detach") else np.asarray(W)
        b = b.detach().cpu().numpy() if (b is not None and hasattr(b, "detach")) else (np.asarray(b) if b is not None else np.zeros(W.shape[0]))
        return W, b
    if isinstance(pred, (tuple, list)) and len(pred) >= 2:
        W = pred[0].detach().cpu().numpy() if hasattr(pred[0], "detach") else np.asarray(pred[0])
        b = pred[1].detach().cpu().numpy() if hasattr(pred[1], "detach") else np.asarray(pred[1])
        return W, b
    return None

W_per_layer = {}
for layer_key, pred in predictors.items():
    extracted = extract(pred)
    if extracted is not None:
        # layer_key might be int or str
        try:
            L = int(layer_key)
        except (ValueError, TypeError):
            L = layer_key
        W_per_layer[L] = extracted

if not W_per_layer:
    sys.exit(f"couldn't extract weights from predictors; layout unknown")
print(f"extracted weights for {len(W_per_layer)} layers")

# Config values with Mixtral defaults
n_experts = config.get("num_experts", config.get("n_experts", 8))
n_expert_used = config.get("n_expert_used", config.get("top_k", 2))
n_layers = config.get("n_layers", 56)
hidden = 6144   # Mixtral 8x22B
feature_dim = 2 * n_experts + hidden  # 6160
first_layer = config.get("first_layer", 0)
print(f"config: n_layers={n_layers}, num_experts={n_experts}, hidden={hidden}, "
      f"feature_dim={feature_dim}, n_expert_used={n_expert_used}, first_L={first_layer}")

# Print one sample's actual shape to verify our assumptions
sample_L = next(iter(W_per_layer))
W_s, b_s = W_per_layer[sample_L]
print(f"sample layer {sample_L}: W shape {W_s.shape}, b shape {b_s.shape}")

# Header
MAGIC = 0x4D4F4550
VERSION = 2
in_dim_trained = W_s.shape[1]   # actual input dim of trained LR (likely 2*n_experts)

with open(OUT, "wb") as f:
    f.write(struct.pack("<8I", MAGIC, VERSION, n_layers, feature_dim,
                         n_experts, hidden, first_layer, n_expert_used))
    n_padded = 0
    n_written = 0
    for L in range(n_layers):
        if L in W_per_layer:
            W, b = W_per_layer[L]
            if W.shape != (n_experts, in_dim_trained):
                # If shape doesn't match, zero-pad and warn
                print(f"  WARN layer {L}: W shape {W.shape} unexpected; zero-padding")
                f.write(b"\x00" * (n_experts * (feature_dim + 1) * 4))
                n_padded += 1
                continue
            for e in range(n_experts):
                row = np.zeros(feature_dim + 1, dtype=np.float32)
                # Copy trained weights into first in_dim_trained positions (multihot region)
                copy_n = min(in_dim_trained, feature_dim)
                row[:copy_n] = W[e][:copy_n]
                # Hidden region stays zero
                row[feature_dim] = float(b[e]) if e < len(b) else 0.0
                f.write(row.tobytes())
            n_written += 1
        else:
            n_padded += 1
            f.write(b"\x00" * (n_experts * (feature_dim + 1) * 4))

import os
sz = os.path.getsize(OUT)
print(f"wrote {OUT}: {sz/1e6:.1f} MB ({n_written} trained, {n_padded} zero-padded)")
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
