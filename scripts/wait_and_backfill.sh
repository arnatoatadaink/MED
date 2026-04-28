#!/bin/bash
# seed ジョブ完了後に backfill_internal_links.py を実行する

LOG=/tmp/wait_and_backfill.log
echo "$(date '+%H:%M:%S') 待機開始: seed ジョブ終了を監視中..." | tee "$LOG"

# 対象PID: 12160, 12673 (または引数で指定)
PIDS="${@:-12160 12673}"

for pid in $PIDS; do
    if kill -0 "$pid" 2>/dev/null; then
        echo "$(date '+%H:%M:%S') PID $pid 完了待ち..." | tee -a "$LOG"
        wait "$pid" 2>/dev/null || true
        echo "$(date '+%H:%M:%S') PID $pid 完了" | tee -a "$LOG"
    fi
done

echo "$(date '+%H:%M:%S') 全ジョブ完了。backfill 開始..." | tee -a "$LOG"
cd /mnt/d/Projects/claude_work/MED
poetry run python scripts/backfill_internal_links.py 2>&1 | tee -a "$LOG"
echo "$(date '+%H:%M:%S') backfill 完了" | tee -a "$LOG"
