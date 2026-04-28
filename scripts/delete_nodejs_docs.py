#!/usr/bin/env python3
"""Node.js github_docs を DB + FAISS から一括削除するスクリプト。

SQLite は同期 sqlite3 で直接操作（aiosqlite ロック問題を回避）。
"""
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass


def main() -> None:
    # ── Step 1: 削除対象 ID を取得 ────────────────────────────────────
    conn = sqlite3.connect(str(_ROOT / "data" / "metadata.db"), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=15000")

    cur = conn.execute("""
        SELECT id, domain FROM documents
        WHERE source_type = 'github_docs'
          AND source_url LIKE '%nodejs%'
    """)
    rows = [(r[0], r[1] or "general") for r in cur.fetchall()]
    print(f"削除対象: {len(rows)}件")

    if not rows:
        print("対象なし。終了します。")
        conn.close()
        return

    # ── Step 2: SQLite から一括削除（500件ずつ）──────────────────────
    all_ids = [r[0] for r in rows]
    total_deleted = 0
    for i in range(0, len(all_ids), 500):
        chunk = all_ids[i : i + 500]
        ph = ",".join("?" for _ in chunk)
        conn.execute(f"DELETE FROM documents WHERE id IN ({ph})", chunk)
        conn.commit()
        total_deleted += len(chunk)
        print(f"  SQLite: {total_deleted}/{len(all_ids)}件削除済み")

    conn.close()
    print(f"SQLite 削除完了: {total_deleted}件")

    # ── Step 3: FAISS から削除 + 保存 ────────────────────────────────
    print("FAISS 削除中...")
    from src.memory.faiss_index import FAISSIndexManager
    from src.common.config import get_settings

    settings = get_settings()
    faiss_mgr = FAISSIndexManager(config=settings.faiss)
    faiss_mgr.load()

    by_domain: dict[str, list[str]] = defaultdict(list)
    for doc_id, domain in rows:
        by_domain[domain].append(doc_id)

    for domain, ids in by_domain.items():
        try:
            removed = faiss_mgr.remove(domain, ids)
            print(f"  FAISS [{domain}]: {removed}件削除")
        except Exception as e:
            print(f"  FAISS [{domain}] error: {e}")

    faiss_mgr.save()
    print("FAISS 保存完了")
    print(f"\n完了: SQLite {total_deleted}件 + FAISS 削除・保存済み")


if __name__ == "__main__":
    main()
