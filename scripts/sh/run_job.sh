#!/bin/bash
# run_job.sh — seed_and_mature ジョブ起動 + 定期進捗監視
#
# Usage:
#   ./run_job.sh <provider> <model> [interval_min] [--seed-mature|--mature-only] [extra args...]
#
# Examples:
#   ./run_job.sh openrouter nvidia/nemotron-3-nano-30b-a3b:free 20 --mature-only --limit 400
#   ./run_job.sh openrouter nvidia/nemotron-3-nano-30b-a3b:free 30 --seed-mature --limit 150
#   ./run_job.sh fastflowlm qwen3.5:9b 15 --mature-only --limit 200

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# ── 引数パース ─────────────────────────────────────────────
PROVIDER="${1:-openrouter}"
MODEL="${2:-nvidia/nemotron-3-nano-30b-a3b:free}"
INTERVAL_MIN="${3:-20}"
MODE="--seed-mature"   # デフォルト
EXTRA_ARGS=()

shift 3 2>/dev/null || shift $# 2>/dev/null || true

for arg in "$@"; do
    case "$arg" in
        --seed-mature)  MODE="--seed-mature" ;;
        --mature-only)  MODE="--mature-only" ;;
        *)              EXTRA_ARGS+=("$arg") ;;
    esac
done

# seed-mature は実際には --mature-only を付けないフラグ
if [[ "$MODE" == "--seed-mature" ]]; then
    MODE_FLAG=""
else
    MODE_FLAG="--mature-only"
fi

# ── ログファイル ────────────────────────────────────────────
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
SAFE_MODEL="$(echo "$MODEL" | tr '/:' '__')"
LOG="/tmp/med_job_${PROVIDER}_${SAFE_MODEL}_${TIMESTAMP}.log"
PID_FILE="/tmp/med_job_${PROVIDER}_${SAFE_MODEL}.pid"

echo "========================================"
echo "  MED Job Runner"
echo "  Provider : $PROVIDER"
echo "  Model    : $MODEL"
echo "  Mode     : ${MODE:-seed-mature}"
echo "  Interval : ${INTERVAL_MIN} min"
echo "  Log      : $LOG"
echo "  $(date '+%Y-%m-%d %H:%M:%S JST')"
echo "========================================"

# ── ジョブ起動 ──────────────────────────────────────────────
CMD=(poetry run python scripts/seed_and_mature.py
    --provider "$PROVIDER"
    --model "$MODEL"
    --questions scripts/questions.txt
    --exclude-sources tavily
    --top-k 5
)
[[ -n "$MODE_FLAG" ]] && CMD+=("$MODE_FLAG")
CMD+=("${EXTRA_ARGS[@]}")

echo ""
echo "CMD: ${CMD[*]}"
echo ""

"${CMD[@]}" > "$LOG" 2>&1 &
JOB_PID=$!
echo "$JOB_PID" > "$PID_FILE"
echo "Started PID: $JOB_PID"
echo ""

# ── 定期進捗監視ループ ──────────────────────────────────────
_show_progress() {
    local log="$1"
    local pid="$2"

    echo "────────────────────────────────────────"
    echo "  Progress  $(date '+%H:%M:%S JST')"
    echo "────────────────────────────────────────"

    # 実行確認
    if kill -0 "$pid" 2>/dev/null; then
        echo "  Status : Running (PID $pid)"
    else
        echo "  Status : *** EXITED ***"
    fi

    # 最新の処理件数行
    local latest
    latest=$(grep -oP '\[\d+/\d+\]' "$log" 2>/dev/null | tail -1 || echo "(none)")
    echo "  Progress : $latest"

    # PASS/HOLD/FAIL カウント
    local pass hold fail total
    pass=$(grep -c "Review PASS" "$log" 2>/dev/null || true)
    hold=$(grep -c "Review HOLD" "$log" 2>/dev/null || true)
    fail=$(grep -c "Review FAIL" "$log" 2>/dev/null || true)
    pass=${pass:-0}; hold=${hold:-0}; fail=${fail:-0}
    total=$((pass + hold + fail))
    if [[ $total -gt 0 ]]; then
        local rate=$(( pass * 100 / total ))
        echo "  Reviews  : PASS=$pass  HOLD=$hold  FAIL=$fail  (total=$total  approval=${rate}%)"
    else
        echo "  Reviews  : (not started yet)"
    fi

    # OpenRouter使用量
    if [[ "$PROVIDER" == "openrouter" ]]; then
        local usage
        usage=$(poetry run python -c "
import sqlite3
try:
    conn = sqlite3.connect('data/openrouter_usage.db')
    row = conn.execute(\"SELECT total_requests FROM daily_usage WHERE date=date('now') ORDER BY date DESC LIMIT 1\").fetchone()
    print(f'{row[0]}/950' if row else '0/950')
    conn.close()
except: print('?/950')
" 2>/dev/null || echo "?/950")
        echo "  OpenRouter: $usage requests today (UTC)"
    fi

    # 最新ログ末尾5行
    echo "  Last log :"
    tail -5 "$log" 2>/dev/null | sed 's/^/    /'
    echo ""
}

# 初回確認（30秒後）
sleep 30
_show_progress "$LOG" "$JOB_PID"

# 定期ループ
INTERVAL_SEC=$(( INTERVAL_MIN * 60 ))
while kill -0 "$JOB_PID" 2>/dev/null; do
    sleep "$INTERVAL_SEC"
    _show_progress "$LOG" "$JOB_PID"
done

# ── 完了処理 ────────────────────────────────────────────────
echo "========================================"
echo "  Job completed  $(date '+%Y-%m-%d %H:%M:%S JST')"
echo "========================================"

# 最終サマリー（SUMMARY セクション抽出）
if grep -q "SUMMARY" "$LOG" 2>/dev/null; then
    echo ""
    sed -n '/SUMMARY/,$ p' "$LOG"
else
    echo ""
    echo "Final log (last 20 lines):"
    tail -20 "$LOG"
fi

rm -f "$PID_FILE"
