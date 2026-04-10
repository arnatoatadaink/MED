#!/bin/bash
# test_models.sh — 複数モデルを少数件数で比較テスト
# Usage: ./test_models.sh [--limit N] [--provider PROVIDER] model1 model2 ...
# Example:
#   ./test_models.sh nvidia/nemotron-3-nano-30b-a3b:free google/gemma-4-31b-it:free
#   ./test_models.sh --limit 5 --provider openrouter nvidia/nemotron-3-nano-30b-a3b:free

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

LIMIT=3
PROVIDER="openrouter"
MODELS=()

# 引数パース
while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit)   LIMIT="$2";    shift 2 ;;
        --provider) PROVIDER="$2"; shift 2 ;;
        --*)       echo "Unknown option: $1"; exit 1 ;;
        *)         MODELS+=("$1"); shift ;;
    esac
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "Usage: $0 [--limit N] [--provider PROVIDER] model1 [model2 ...]"
    echo "Example: $0 nvidia/nemotron-3-nano-30b-a3b:free google/gemma-4-31b-it:free"
    exit 1
fi

LOG_DIR="/tmp/med_model_test"
mkdir -p "$LOG_DIR"
SUMMARY_FILE="$LOG_DIR/summary_$(date '+%Y%m%d_%H%M%S').txt"

echo "========================================"
echo "  Model Comparison Test"
echo "  Provider: $PROVIDER  |  Docs/model: $LIMIT"
echo "  $(date '+%Y-%m-%d %H:%M:%S JST')"
echo "========================================"
echo ""

for MODEL in "${MODELS[@]}"; do
    SAFE_NAME="$(echo "$MODEL" | tr '/:' '__')"
    LOG="$LOG_DIR/test_${SAFE_NAME}.log"

    echo "────────────────────────────────────────"
    echo "  Testing: $MODEL"
    echo "  Log: $LOG"
    echo "────────────────────────────────────────"

    # 実行（逐次。DB競合を避けるためシリアル）
    poetry run python scripts/seed_and_mature.py \
        --mature-only \
        --provider "$PROVIDER" \
        --model "$MODEL" \
        --limit "$LIMIT" \
        > "$LOG" 2>&1

    # 結果をパース・表示
    echo ""
    echo "  [Results]"

    # 各レビュー行を抽出
    PASS_CNT=$(grep -c "Review PASS" "$LOG" 2>/dev/null || echo 0)
    HOLD_CNT=$(grep -c "Review HOLD" "$LOG" 2>/dev/null || echo 0)
    FAIL_CNT=$(grep -c "Review FAIL" "$LOG" 2>/dev/null || echo 0)
    TOTAL=$((PASS_CNT + HOLD_CNT + FAIL_CNT))

    if [[ $TOTAL -gt 0 ]]; then
        # 平均応答時間
        AVG_TIME=$(grep -oP '\(\K[0-9.]+(?=s,)' "$LOG" | awk '{sum+=$1; n++} END {printf "%.1f", (n>0)?sum/n:0}')
        # 平均 quality
        AVG_QUAL=$(grep -oP 'quality=\K[0-9.]+' "$LOG" | awk '{sum+=$1; n++} END {printf "%.2f", (n>0)?sum/n:0}')
        APPROVE_RATE=$(( PASS_CNT * 100 / TOTAL ))

        echo "    PASS: $PASS_CNT  HOLD: $HOLD_CNT  FAIL: $FAIL_CNT  (total: $TOTAL)"
        echo "    Approval rate : ${APPROVE_RATE}%"
        echo "    Avg response  : ${AVG_TIME}s"
        echo "    Avg quality   : ${AVG_QUAL}"

        # 各レビューの詳細（時間・quality・判定・内容冒頭）
        echo ""
        echo "  [Review details]"
        grep -E "Review (PASS|HOLD|FAIL)" "$LOG" | while IFS= read -r line; do
            echo "    $line"
        done
    else
        # エラーの可能性
        echo "    No reviews completed. Last lines:"
        tail -5 "$LOG" | sed 's/^/    /'
    fi

    # サマリー記録
    {
        echo "Model: $MODEL"
        echo "  PASS=$PASS_CNT HOLD=$HOLD_CNT FAIL=$FAIL_CNT  approval=${APPROVE_RATE}%  avg_time=${AVG_TIME}s  avg_quality=${AVG_QUAL}"
    } >> "$SUMMARY_FILE"

    echo ""
done

echo "========================================"
echo "  Summary saved: $SUMMARY_FILE"
echo "========================================"
cat "$SUMMARY_FILE"
