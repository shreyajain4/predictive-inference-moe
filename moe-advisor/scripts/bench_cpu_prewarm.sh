#!/bin/bash
# bench_cpu_prewarm.sh
# Test the CPU-side warm-from-history hypothesis: pre-load expert pages
# into OS page cache so ngl_auto's CPU MoE compute doesn't fault from SSD.
#
# This is a fundamentally different mechanism than the GPU-cache snap/warm
# we've tested. No PCIe involved — just keeping the right pages hot in RAM
# so that ngl_auto's CPU path reads from DRAM at ~50 GB/s instead of SSD
# at ~500 MB/s for random page faults.
#
# Sequence:
#   1. Cold ngl_auto baseline — whatever pages are in cache now
#   2. Run prewarm to touch top-K experts per layer (forces them into cache)
#   3. Warm ngl_auto — should serve more from RAM
#
# Usage: bash bench_cpu_prewarm.sh [K] [N_TOKENS]
#   K=4, N_TOKENS=32 by default

set -u

K=${1:-4}
N_TOKENS=${2:-32}
MODEL_PART1=$HOME/models/Mixtral-8x22B-Instruct-v0.1.IQ4_XS-00001-of-00002.gguf
MODEL_PART2=$HOME/models/Mixtral-8x22B-Instruct-v0.1.IQ4_XS-00002-of-00002.gguf
OFFSETS=/tmp/mixtral_iq4xs_offsets.json
PROMPT="[INST] explain prefill and decode in llama.cpp [/INST]"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "K=$K experts/layer  N_TOKENS=$N_TOKENS"
echo "model: $MODEL_PART1 (+part2)"
echo ""

run_bench_ngl_auto() {
  local label="$1"
  echo "=== $label ==="
  # Use the bench binary (exits cleanly after -n tokens). Configure as
  # ngl_auto: no -ngl flag, no override-tensor, no expert-cache-mb.
  # Still needs --predictor-weights for the bench to initialize but
  # --prefetch-k 0 means predictor doesn't fire.
  local logf=/tmp/bench_run_$$.log
  timeout 600 $HOME/llama.cpp/build/bin/llama-moe-predictor-bench \
    -m "$MODEL_PART1" -c 512 -n "$N_TOKENS" \
    --predictor-weights /tmp/mixtral_stub_predictor.bin \
    --prefetch-k 0 \
    -p "$PROMPT" > "$logf" 2>&1 || echo "  (timed out or errored)"
  grep -E "decode:.*tok/s|offloaded|n_gpu_layers" "$logf" | head -5
  rm -f "$logf"
  echo ""
}

# Lightweight page-cache state probe via /proc/meminfo Cached: line.
# Avoids the brittle ctypes mincore call that was segfaulting on munmap.
report_cache() {
  echo "--- page-cache state ---"
  awk '/^Cached:|^Buffers:|^MemAvailable:/ {printf "  %-15s %s %s\n", $1, $2, $3}' /proc/meminfo
  if command -v vmtouch &>/dev/null; then
    echo "  (per-file via vmtouch)"
    vmtouch -q "$MODEL_PART1" 2>&1 | sed "s|^|    |"
    vmtouch -q "$MODEL_PART2" 2>&1 | sed "s|^|    |"
  fi
  echo ""
}

report_cache
run_bench_ngl_auto "A: cold ngl_auto (whatever's in cache)"
report_cache

echo "=== prewarming top-$K experts per layer ==="
python3 "$SCRIPT_DIR/prewarm_experts.py" \
  --offsets "$OFFSETS" --k "$K" \
  --parts "$MODEL_PART2"
echo ""

report_cache
run_bench_ngl_auto "B: ngl_auto after prewarm"
report_cache

echo ""
echo "=== summary ==="
echo "  A (cold): see line above"
echo "  B (warm): see line above"
echo "If B > A in gen t/s, CPU-side prewarming works on this hardware/model."
