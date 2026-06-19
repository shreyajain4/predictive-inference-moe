#!/bin/bash
# bench_mixtral_user_history_prewarm.sh
# End-to-end test: does CPU-side warm-from-history (using YOUR prompt
# history as the routing oracle) beat the OS LRU on Mixtral 8x22B SSD-regime?
#
# Pipeline:
#   1. Collect Mixtral routing on user-history train prompts via llama-moe-trace
#      (~30-60 min for ~50 prompts at 0.4 t/s)
#   2. Build per-(layer, expert) frequency profile from the routing CSV
#   3. Convert to flat warm profile (top-K experts per layer)
#   4. Cold ngl_auto baseline (no prewarm)
#   5. Prewarm those user-history experts into OS page cache
#   6. Warm ngl_auto (after prewarm)
#
# Total: ~1.5-2 hours.
#
# Usage:
#   bash bench_mixtral_user_history_prewarm.sh [K] [N_TRACE_PROMPTS] [N_DECODE_TOKENS]
#   Defaults: K=4 experts/layer, N_TRACE=30 prompts (sampled from start of train),
#             N_DECODE=32 tokens for the cold/warm comparison

set -u

K=${1:-4}
N_TRACE=${2:-30}
N_DECODE=${3:-32}

MODEL_PART1=$HOME/models/Mixtral-8x22B-Instruct-v0.1.IQ4_XS-00001-of-00002.gguf
MODEL_PART2=$HOME/models/Mixtral-8x22B-Instruct-v0.1.IQ4_XS-00002-of-00002.gguf
OFFSETS=/tmp/mixtral_iq4xs_offsets.json
TRAIN_TSV=$HOME/predictive-inference-moe/moe-advisor/data/user_history/shreya_prompts_train.tsv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK=/tmp/mixtral_history_$$
mkdir -p "$WORK"

TRACE_PROMPTS=$WORK/trace_prompts.tsv
TRACE_CSV=$WORK/mixtral_routing.csv
PROFILE_NPY=$WORK/mixtral_profile.npy
PROFILE_TXT=$WORK/mixtral_warm.txt
TEST_PROMPT="[INST] explain prefill and decode in llama.cpp [/INST]"

echo "K=$K experts/layer, N_TRACE=$N_TRACE trace prompts, N_DECODE=$N_DECODE tokens"
echo "working dir: $WORK"
echo ""

# Step 1: Sample first N_TRACE prompts from train TSV
head -n "$N_TRACE" "$TRAIN_TSV" > "$TRACE_PROMPTS"
echo "=== Step 1: sampling $N_TRACE train prompts ==="
echo "  $(wc -l < "$TRACE_PROMPTS") lines selected"
echo ""

# Step 2: Run llama-moe-trace on Mixtral to collect routing
echo "=== Step 2: collecting Mixtral routing (this will be slow ~30-60 min) ==="
if [ ! -f "$TRACE_CSV" ]; then
  $HOME/llama.cpp/build/bin/llama-moe-trace \
    -m "$MODEL_PART1" \
    -f "$TRACE_PROMPTS" \
    -c 4096 -b 4096 -ub 128 \
    -n 16 \
    > "$TRACE_CSV" 2> "$WORK/trace.log" || { echo "  trace failed; see $WORK/trace.log"; tail -20 "$WORK/trace.log"; exit 1; }
fi
echo "  trace CSV: $(wc -l < "$TRACE_CSV") lines"
echo ""

# Step 3: Build per-(layer, expert) profile from trace CSV
echo "=== Step 3: building user-history profile ==="
python3 "$SCRIPT_DIR/build_user_history_profile.py" \
  --csv "$TRACE_CSV" \
  --out "${PROFILE_NPY%.npy}" \
  --n-experts 8 2>&1 | tail -20
echo ""

# Step 4: Convert profile to flat warm txt (top-K per layer)
echo "=== Step 4: converting to flat warm profile (top-$K per layer) ==="
python3 "$SCRIPT_DIR/npy_to_warm_profile.py" \
  --profile "$PROFILE_NPY" \
  --k "$K" \
  --out "$PROFILE_TXT" 2>&1
echo ""

run_bench() {
  local label="$1"
  echo "=== $label ==="
  local logf=$WORK/bench_run.log
  timeout 600 $HOME/llama.cpp/build/bin/llama-moe-predictor-bench \
    -m "$MODEL_PART1" -c 512 -n "$N_DECODE" \
    --predictor-weights /tmp/mixtral_stub_predictor.bin \
    --prefetch-k 0 \
    -p "$TEST_PROMPT" > "$logf" 2>&1 || echo "  (timed out)"
  grep -E "decode:.*tok/s|offloaded" "$logf" | head -3
  echo ""
}

report_cache() {
  echo "--- page-cache state ---"
  awk '/^Cached:|^MemAvailable:/ {printf "  %-15s %s %s\n", $1, $2, $3}' /proc/meminfo
  echo ""
}

# Step 5: Cold baseline
report_cache
run_bench "A: cold ngl_auto"
report_cache

# Step 6: Prewarm using USER-HISTORY profile
echo "=== Step 6: prewarming top-$K experts per layer FROM USER HISTORY ==="
python3 "$SCRIPT_DIR/prewarm_experts.py" \
  --offsets "$OFFSETS" \
  --warm-profile "$PROFILE_TXT" \
  --k "$K" \
  --parts "$MODEL_PART2" 2>&1
echo ""

report_cache
run_bench "B: ngl_auto after USER-HISTORY prewarm"
report_cache

echo "=== summary ==="
echo "  A (cold, OS LRU only): see line"
echo "  B (after user-history prewarm): see line"
echo ""
echo "If B > A, USER-HISTORY prewarm beats the OS's LRU — a real positive result"
echo "(unlike the arbitrary first-K prewarm earlier which evicted useful pages)."
echo ""
echo "raw artifacts in: $WORK"
