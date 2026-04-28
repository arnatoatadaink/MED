#!/usr/bin/env python3
"""既存の chunker_type='markdown' ドキュメントから internal_links を一括抽出・更新する。

再seed不要。DB上のcontentに extract_internal_links() を適用してUPDATEするだけ。
"""
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from src.rag.chunker import Chunker

DB_PATH = _ROOT / "data" / "metadata.db"
BATCH_SIZE = 200


def main() -> None:
    conn = sqlite3.connect(str(DB_PATH), timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")

    total = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE chunker_type='markdown'"
    ).fetchone()[0]
    print(f"対象: {total}件 (chunker_type='markdown')")

    updated = 0
    has_links = 0
    offset = 0

    while True:
        rows = conn.execute(
            "SELECT id, content FROM documents WHERE chunker_type='markdown' LIMIT ? OFFSET ?",
            (BATCH_SIZE, offset),
        ).fetchall()
        if not rows:
            break

        updates = []
        for doc_id, content in rows:
            links = Chunker.extract_internal_links(content or "")
            import json
            updates.append((json.dumps(links), doc_id))
            if links:
                has_links += 1

        conn.executemany(
            "UPDATE documents SET internal_links=? WHERE id=?", updates
        )
        conn.commit()

        updated += len(rows)
        offset += BATCH_SIZE
        print(f"  {updated}/{total} 処理済み (リンクあり: {has_links}件)")

    conn.close()
    print(f"\n完了: {updated}件更新, うちリンクあり {has_links}件")


if __name__ == "__main__":
    main()
