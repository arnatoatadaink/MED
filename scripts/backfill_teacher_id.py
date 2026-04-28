#!/usr/bin/env python3
"""既存レビュー済み文書の teacher_id を reviewed_at 日時に基づいて一括設定する。

OpenRouter/LMStudio/FastFlowLM を同時使用しており個別特定不可のため、
導入前後で unknown1 / unknown2 に区別する。

  - reviewed_at < 2026-03-31  : unknown1  (OpenRouter 導入前)
  - reviewed_at >= 2026-03-31 : unknown2  (OpenRouter 導入後)
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "metadata.db"
BATCH_SIZE = 500

# reviewed_at 境界 (UTC)
OR_START = "2026-03-31"

TEACHER_BEFORE = "unknown1"  # OpenRouter 導入前
TEACHER_AFTER  = "unknown2"  # OpenRouter 導入後


def assign_teacher_id(reviewed_at: str | None) -> str:
    if reviewed_at is None or reviewed_at >= OR_START:
        return TEACHER_AFTER
    return TEACHER_BEFORE


def main() -> None:
    conn = sqlite3.connect(str(DB_PATH), timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")

    # 対象: teacher_id が NULL かつ unreviewed 以外
    total = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE teacher_id IS NULL AND review_status != 'unreviewed'"
    ).fetchone()[0]
    print(f"対象: {total}件 (teacher_id=NULL, reviewed)")

    # 事前統計
    before = conn.execute(
        f"SELECT COUNT(*) FROM documents WHERE teacher_id IS NULL AND review_status != 'unreviewed' AND reviewed_at < '{OR_START}'"
    ).fetchone()[0]
    after = total - before
    print(f"\n割り当て予定:")
    print(f"  {TEACHER_BEFORE}: {before}件 (reviewed_at < {OR_START})")
    print(f"  {TEACHER_AFTER}: {after}件 (reviewed_at >= {OR_START} or NULL)")
    print()

    updated = 0
    offset = 0

    while True:
        rows = conn.execute(
            """SELECT id, reviewed_at FROM documents
               WHERE teacher_id IS NULL AND review_status != 'unreviewed'
               LIMIT ? OFFSET ?""",
            (BATCH_SIZE, offset),
        ).fetchall()
        if not rows:
            break

        updates = [(assign_teacher_id(r[1]), r[0]) for r in rows]
        conn.executemany(
            "UPDATE documents SET teacher_id=? WHERE id=?", updates
        )
        conn.commit()

        updated += len(rows)
        offset += BATCH_SIZE
        print(f"  {updated}/{total} 更新済み", flush=True)

    conn.close()
    print(f"\n完了: {updated}件 teacher_id 設定")


if __name__ == "__main__":
    main()
