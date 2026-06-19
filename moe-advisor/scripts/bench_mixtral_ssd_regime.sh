#!/bin/bash
# bench_mixtral_ssd_regime.sh
# Run snapshot + cache vs ngl_auto vs forced-offload-no-cache on Mixtral
# 8x22B IQ4_XS (76 GB total, doesn't fit in Legion's 45+8=53 GB fast memory).
# This is the SSD-regime test the negative result needs to be tested against.
#
# Generates a stub Mixtral predictor (won't actually predict since
# --prefetch-k=0 in snap tests), generates expert offsets, then runs each
# config sequentially.
#
# Usage: bash bench_mixtral_ssd_regime.sh [N_TOKENS]
# Default N_TOKENS=32.

set -e

N_TOKENS=${1:-32}
MODEL=$HOME/models/Mixtral-8x22B-Instruct-v0.1.IQ4_XS-00001-of-00002.gguf
PROMPT="[INST] explain prefill and decode in llama.cpp [/INST]"
STUB_PREDICTOR=/tmp/mixtral_stub_predictor.bin
OFFSETS_JSON=/tmp/mixtral_iq4xs_offsets.json
LOG_DIR=/tmp/mixtral_ssd_$(date +%s)
mkdir -p "$LOG_DIR"

echo "outdir: $LOG_DIR"
echo "model:  $MODEL"
echo "N_TOKENS=$N_TOKENS"
echo ""

# Step 1: generate stub Mixtral predictor (.bin file the bench will load
# but never actually use since --prefetch-k 0).
# Format (from moe_predictor.cpp):
#   header[8] uint32: magic=0x4D4F4550, version=2, n_layers, feature_dim,
#                     num_experts, hidden_dim, first_layer, n_expert_used
#   body:    n_layers × num_experts × feature_dim float32 zeros
echo "=== generating stub predictor ==="
python3 - << 'PY'
import struct, os
MAGIC = 0x4D4F4550
VERSION = 2
N_LAYERS = 56
NUM_EXPERTS = 8
HIDDEN = 6144
FEATURE_DIM = 2 * NUM_EXPERTS + HIDDEN   # 6160
FIRST_LAYER = 0
N_EXPERT_USED = 2  # top-2

with open('/tmp/mixtral_stub_predictor.bin', 'wb') as f:
    f.write(struct.pack('<8I', MAGIC, VERSION, N_LAYERS, FEATURE_DIM,
                         NUM_EXPERTS, HIDDEN, FIRST_LAYER, N_EXPERT_USED))
    n_floats = N_LAYERS * NUM_EXPERTS * FEATURE_DIM
    # Zero weights — predictor would output 0 for everything, but we won't
    # exercise it since prefetch-k=0 for all snap tests.
    zero_chunk = b'\x00' * (4 * 65536)
    written = 0
    while written < n_floats * 4:
        rem = n_floats * 4 - written
        f.write(zero_chunk[:min(len(zero_chunk), rem)])
        written += min(len(zero_chunk), rem)
print(f"wrote stub: 56 layers × 8 experts × 6160 feat_dim = "
      f"{os.path.getsize('/tmp/mixtral_stub_predictor.bin')/1e6:.1f} MB")
PY

# Step 2: generate Mixtral expert offsets (needed by --expert-cache-mb)
echo ""
echo "=== generating expert offsets ==="
if [ ! -f "$OFFSETS_JSON" ]; then
  python3 $HOME/predictive-inference-moe/bench-bundle/predictor-weights/extract_expert_offsets.py \
    --gguf "$MODEL" \
    --output "$OFFSETS_JSON" 2>&1 | tail -10
else
  echo "already exists: $OFFSETS_JSON"
fi

# Step 3: run the four configs
echo ""
echo "=== running benches ==="

# Cache size: 5 GB available on 8 GB 3070 after KV + non-expert + driver
# Each Mixtral expert at IQ4_XS is ~150 MB → ~30 slots = ~7% of experts
CACHE_MB=4500
BASE=(-m "$MODEL"
      -c 2048 -b 2048 -ub 1024 -n "$N_TOKENS"
      -ngl 99 --override-tensor exps=CPU
      --predictor-weights "$STUB_PREDICTOR"
      --expert-offsets "$OFFSETS_JSON"
      --expert-cache-mb "$CACHE_MB")

run() {
  local label="$1"; shift
  local log="$LOG_DIR/${label// /_}.log"
  echo "--- $label ---"
  GGML_OP_OFFLOAD_MIN_BATCH=1 \
    $HOME/llama.cpp/build/bin/llama-moe-predictor-bench \
    "${BASE[@]}" "$@" -p "$PROMPT" > "$log" 2>&1 || echo "  (failed; see $log)"
  grep -E "decode:|expert_cache final|MOE-COPY-STATS" "$log" | sed 's/^/  /'
  echo ""
}

# Also need a ngl_auto baseline via llama-cli (no bench, no override)
echo "--- E: ngl_auto (via llama-cli, no cache machinery) ---"
$HOME/llama.cpp/build/bin/llama-cli \
  -m "$MODEL" -c 2048 -n "$N_TOKENS" \
  -p "$PROMPT" > "$LOG_DIR/E_ngl_auto.log" 2>&1 || echo "  (failed)"
grep -oE "Generation: [0-9.]+ t/s" "$LOG_DIR/E_ngl_auto.log" | sed 's/^/  /'
echo ""

run "A snap_only"      --prefetch-k 0
# These three need a working predictor — stub has zero weights so they'd
# just prefetch expert 0 over and over. Useful as a sanity check that
# the path runs, but real recall would require a trained Mixtral predictor.
# Skipping for now; the snap_only vs ngl_auto comparison is the key one.

echo ""
echo "raw logs under: $LOG_DIR"
echo "tail of each:"
for f in "$LOG_DIR"/*.log; do
  echo "=== $f ==="
  tail -5 "$f"
done
