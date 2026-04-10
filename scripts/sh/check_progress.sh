#!/bin/bash
# check_progress.sh — DB状態 / FAISS / OpenRouter使用量 の一括確認
# Usage: ./check_progress.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "========================================"
echo "  MED Progress Check  $(date '+%Y-%m-%d %H:%M:%S JST')"
echo "========================================"

poetry run python - <<'EOF'
import sqlite3, os

# DB状態
conn = sqlite3.connect("data/metadata.db")
rows = conn.execute("SELECT review_status, COUNT(*) FROM documents GROUP BY review_status ORDER BY COUNT(*) DESC").fetchall()
total = sum(r[1] for r in rows)
print("\n[DB] Documents")
for status, cnt in rows:
    bar = "█" * (cnt * 30 // max(total, 1))
    print(f"  {status:<15} {cnt:>5}  {bar}")
print(f"  {'TOTAL':<15} {total:>5}")
conn.close()

# FAISS
import glob
print("\n[FAISS] Index vectors")
for idx_path in sorted(glob.glob("data/faiss_indices/*")):
    domain = os.path.basename(idx_path)
    try:
        import faiss
        idx = faiss.read_index(f"{idx_path}/index.faiss")
        print(f"  {domain:<12} {idx.ntotal:>6} vectors")
    except Exception:
        print(f"  {domain:<12}  (read error)")

# OpenRouter使用量
try:
    conn2 = sqlite3.connect("data/openrouter_usage.db")
    rows2 = conn2.execute(
        "SELECT date, total_requests FROM daily_usage ORDER BY date DESC LIMIT 3"
    ).fetchall()
    limit = 950
    print("\n[OpenRouter] Daily usage (limit=950)")
    for date, req in rows2:
        pct = req * 100 // limit
        bar = "█" * (pct * 20 // 100)
        print(f"  {date}  {req:>4}/{limit}  ({pct:>3}%)  {bar}")
    conn2.close()
except Exception as e:
    print(f"\n[OpenRouter] usage DB error: {e}")
EOF

echo ""
