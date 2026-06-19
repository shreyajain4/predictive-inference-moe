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

set -e

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

run_llama_cli() {
  local label="$1"
  echo "=== $label ==="
  # Close stdin so llama-cli exits after -n tokens instead of going interactive
  $HOME/llama.cpp/build/bin/llama-cli \
    -m "$MODEL_PART1" -c 512 -n "$N_TOKENS" -p "$PROMPT" < /dev/null 2>&1 \
    | grep -oE "Prompt:[^|]+\|.*tok/s" \
    | tail -1
  echo ""
}

# Quick "what's in page cache for the model right now?" probe
report_cache() {
  echo "--- page-cache state ---"
  for f in "$MODEL_PART1" "$MODEL_PART2"; do
    if command -v vmtouch &>/dev/null; then
      vmtouch -q "$f" | sed "s|^|  |"
    else
      # mincore-based fallback via python
      python3 - "$f" << 'PY'
import sys, mmap, os
path = sys.argv[1]
sz = os.path.getsize(path)
# Use mincore via os
try:
    import ctypes
    libc = ctypes.CDLL("libc.so.6")
    fd = os.open(path, os.O_RDONLY)
    addr = libc.mmap(0, sz, 1, 1, fd, 0)  # PROT_READ, MAP_SHARED
    n_pages = (sz + 4095) // 4096
    vec = (ctypes.c_ubyte * n_pages)()
    libc.mincore(ctypes.c_void_p(addr), sz, vec)
    resident = sum(1 for b in vec if b & 1)
    print(f"  {path}: {resident}/{n_pages} pages resident ({100*resident/n_pages:.1f}%, {resident*4/1024:.0f} MB)")
    libc.munmap(ctypes.c_void_p(addr), sz)
    os.close(fd)
except Exception as e:
    print(f"  {path}: mincore failed: {e}")
PY
    fi
  done
  echo ""
}

report_cache
run_llama_cli "A: cold ngl_auto (whatever's in cache)"
report_cache

echo "=== prewarming top-$K experts per layer ==="
python3 "$SCRIPT_DIR/prewarm_experts.py" \
  --offsets "$OFFSETS" --k "$K" \
  --parts "$MODEL_PART2"
echo ""

report_cache
run_llama_cli "B: ngl_auto after prewarm"
report_cache

echo ""
echo "=== summary ==="
echo "  A (cold): see line above"
echo "  B (warm): see line above"
echo "If B > A in gen t/s, CPU-side prewarming works on this hardware/model."
