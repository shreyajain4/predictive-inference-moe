#!/bin/bash
# bench_mixtral_drop_caches.sh
# Proper cold-start rerun: drop_caches between every run so each "cold"
# is genuinely cold (page cache empty, model pages must fault from SSD).
#
# Tests 4 variants on the SAME prompt (prompt 0 of train), 3 reps each:
#   A: TRUE cold ngl_auto                  -- baseline, OS LRU + first-touch fault
#   B: drop + user-history prewarm + ngl_auto
#   C: drop + per-prompt oracle prewarm + ngl_auto
#   D: WARM ngl_auto (no drop, repeat run) -- hot reference
#
# Reads sudo pw from $SUDO_PASS env var. Never write the pw to a file.
# Usage on Legion:
#   SUDO_PASS='...' bash bench_mixtral_drop_caches.sh \\
#     --user-profile /tmp/mixtral_history_XXX/mixtral_warm.txt \\
#     --oracle-profile /tmp/mixtral_oracle_prompt0.txt \\
#     --prompt-id 0 --reps 3 --n-decode 32

set -u

USER_PROFILE=""
ORACLE_PROFILE=""
PROMPT_ID=0
REPS=3
N_DECODE=32
TRAIN_TSV=$HOME/predictive-inference-moe/moe-advisor/data/user_history/shreya_prompts_train.tsv

while [ $# -gt 0 ]; do
  case "$1" in
    --user-profile)   USER_PROFILE="$2";   shift 2 ;;
    --oracle-profile) ORACLE_PROFILE="$2"; shift 2 ;;
    --prompt-id)      PROMPT_ID="$2";      shift 2 ;;
    --reps)           REPS="$2";           shift 2 ;;
    --n-decode)       N_DECODE="$2";       shift 2 ;;
    --train-tsv)      TRAIN_TSV="$2";      shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

if [ -z "${SUDO_PASS:-}" ]; then
  echo "ERROR: set SUDO_PASS env var (needed for drop_caches)"
  exit 1
fi

MODEL_PART1=$HOME/models/Mixtral-8x22B-Instruct-v0.1.IQ4_XS-00001-of-00002.gguf
MODEL_PART2=$HOME/models/Mixtral-8x22B-Instruct-v0.1.IQ4_XS-00002-of-00002.gguf
OFFSETS=/tmp/mixtral_iq4xs_offsets.json
STUB_BIN=/tmp/mixtral_stub_predictor.bin
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK=/tmp/mixtral_drop_$$
mkdir -p "$WORK"

# Pull prompt 0 (or whichever PROMPT_ID) from the train TSV verbatim as the test prompt
TEST_PROMPT=$(sed -n "$((PROMPT_ID + 1))p" "$TRAIN_TSV")
if [ -z "$TEST_PROMPT" ]; then
  echo "ERROR: prompt-id $PROMPT_ID not found in $TRAIN_TSV"
  exit 1
fi
echo "test prompt (id=$PROMPT_ID, ${#TEST_PROMPT} chars):"
echo "  ${TEST_PROMPT:0:120}..."
echo ""

drop_caches() {
  sync
  echo "$SUDO_PASS" | sudo -S sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null
  # Tiny delay so the kernel actually finishes the flush before we measure
  sleep 1
}

report_cache() {
  awk '/^Cached:|^MemAvailable:/ {printf "  %-15s %12s %s\n", $1, $2, $3}' /proc/meminfo
}

run_one() {
  local label="$1"
  local logf=$WORK/${label// /_}.log
  timeout 600 $HOME/llama.cpp/build/bin/llama-moe-predictor-bench \
    -m "$MODEL_PART1" -c 512 -n "$N_DECODE" \
    --predictor-weights "$STUB_BIN" \
    --prefetch-k 0 \
    -p "$TEST_PROMPT" > "$logf" 2>&1 || echo "  (timed out)"
  # Extract decode tok/s
  local tps
  tps=$(grep -oE 'decode:.*[0-9.]+\s*tok/s' "$logf" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
  echo "$tps"
}

run_variant() {
  local label="$1"
  local prep="$2"   # function name: drop_caches OR noop
  local prewarm_profile="${3:-}"   # path, or empty for none

  echo "=== $label ==="
  local tps_list=()
  for rep in $(seq 1 "$REPS"); do
    echo "  rep $rep:"
    $prep
    if [ -n "$prewarm_profile" ]; then
      echo "  prewarming from $prewarm_profile..."
      python3 "$SCRIPT_DIR/prewarm_experts.py" \
        --offsets "$OFFSETS" \
        --warm-profile "$prewarm_profile" \
        --k 4 \
        --parts "$MODEL_PART2" 2>&1 | tail -3
    fi
    echo "  page-cache before run:"
    report_cache
    local tps
    tps=$(run_one "${label}_rep${rep}")
    echo "  -> decode: $tps tok/s"
    tps_list+=("$tps")
  done
  echo "  ${label} reps: ${tps_list[*]}"
  # median (sort + take middle)
  local sorted
  sorted=$(printf "%s\n" "${tps_list[@]}" | sort -n)
  local n=${#tps_list[@]}
  local mid=$(( (n + 1) / 2 ))
  local median
  median=$(echo "$sorted" | sed -n "${mid}p")
  echo "  ${label} MEDIAN: $median tok/s"
  echo ""
}

noop() { :; }

echo "=== sudo sanity check ==="
echo "$SUDO_PASS" | sudo -S whoami 2>/dev/null && echo "sudo OK" || { echo "sudo failed"; exit 1; }
echo ""

echo "=== initial page-cache state ==="
report_cache
echo ""

# Variant A: TRUE cold ngl_auto baseline
run_variant "A_cold_baseline" drop_caches ""

# Variant B: cold + user-history prewarm
if [ -n "$USER_PROFILE" ] && [ -f "$USER_PROFILE" ]; then
  run_variant "B_user_history" drop_caches "$USER_PROFILE"
else
  echo "(skipping B: --user-profile not provided or missing)"
  echo ""
fi

# Variant C: cold + per-prompt oracle prewarm (upper bound)
if [ -n "$ORACLE_PROFILE" ] && [ -f "$ORACLE_PROFILE" ]; then
  run_variant "C_oracle" drop_caches "$ORACLE_PROFILE"
else
  echo "(skipping C: --oracle-profile not provided or missing)"
  echo ""
fi

# Variant D: HOT reference (no drop, runs back-to-back-to-back)
echo "=== D_warm_reference (no drop_caches; back-to-back to populate cache) ==="
tps_list=()
for rep in $(seq 1 "$REPS"); do
  echo "  rep $rep (no drop):"
  report_cache
  tps=$(run_one "D_warm_rep${rep}")
  echo "  -> decode: $tps tok/s"
  tps_list+=("$tps")
done
echo "  D reps: ${tps_list[*]}"
echo ""

echo "=== ALL DONE ==="
echo "raw logs: $WORK"
echo ""
echo "Interpretation:"
echo "  If A << B,C : prewarm DOES help against true cold cache (positive result)"
echo "  If A ≈ B,C  : OS first-touch faulting is as fast as preloading (neutral)"
echo "  D is the upper bound (everything already in RAM)"
