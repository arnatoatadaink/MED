#!/bin/bash
# run_job_fg.sh — seed_and_mature ジョブをフォアグラウンドで起動しログをリアルタイム表示
#
# 別ウィンドウで実行してログを直接コマンドライン上に流す用途向け。
# ログファイルにも同時保存する（tee 使用）。
# Ctrl+C で安全に停止できる。
#
# Usage:
#   ./run_job_fg.sh <provider> <model> [--seed-mature|--mature-only] [extra args...]
#
# Examples:
#   ./run_job_fg.sh openrouter nvidia/nemotron-3-nano-30b-a3b:free --mature-only --limit 400
#   ./run_job_fg.sh fastflowlm qwen3.5:9b --mature-only --limit 200

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# ── 引数パース ─────────────────────────────────────────────
PROVIDER="${1:-openrouter}"
MODEL="${2:-nvidia/nemotron-3-nano-30b-a3b:free}"
MODE_FLAG="--mature-only"
EXTRA_ARGS=()

shift 2 2>/dev/null || shift $# 2>/dev/null || true

for arg in "$@"; do
    case "$arg" in
        --seed-mature)  MODE_FLAG="" ;;
        --mature-only)  MODE_FLAG="--mature-only" ;;
        *)              EXTRA_ARGS+=("$arg") ;;
    esac
done

# ── ログファイル ────────────────────────────────────────────
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
SAFE_MODEL="$(echo "$MODEL" | tr '/:' '__')"
LOG="/tmp/med_job_${PROVIDER}_${SAFE_MODEL}_${TIMESTAMP}.log"

# ── コマンド構築 ─────────────────────────────────────────────
CMD=(poetry run python scripts/seed_and_mature.py
    --provider "$PROVIDER"
    --model "$MODEL"
    --questions scripts/questions.txt
    --exclude-sources tavily
    --top-k 5
)
[[ -n "$MODE_FLAG" ]] && CMD+=("$MODE_FLAG")
CMD+=("${EXTRA_ARGS[@]}")

# ── ヘッダーをログファイルと端末の両方に出力 ────────────────
{
echo "========================================"
echo "  MED Job Runner [FOREGROUND]"
echo "  Provider : $PROVIDER"
echo "  Model    : $MODEL"
echo "  Mode     : ${MODE_FLAG:---seed-mature}"
echo "  Log      : $LOG"
echo "  $(date '+%Y-%m-%d %H:%M:%S JST')"
echo "  Ctrl+C で停止"
echo "  CMD: ${CMD[*]}"
echo "========================================"
echo ""
} | tee "$LOG"

# ── フォアグラウンドで実行（tee でログ保存 + 端末表示）──────
"${CMD[@]}" 2>&1 | tee -a "$LOG"

{
echo ""
echo "========================================"
echo "  Job completed  $(date '+%Y-%m-%d %H:%M:%S JST')"
echo "  Log saved: $LOG"
echo "========================================"
} | tee -a "$LOG"
