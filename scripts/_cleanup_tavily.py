"""Tavily needs_update + rejected ドキュメントを削除する一時スクリプト"""
import asyncio
import logging
import sqlite3
import sys

sys.path.insert(0, "/mnt/d/Projects/claude_work/MED")

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.WARNING)

from src.memory.memory_manager import MemoryManager


async def main() -> None:
    mm = MemoryManager()
    await mm.initialize()

    conn = sqlite3.connect("data/metadata.db")
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM documents WHERE source_type='tavily' AND review_status IN ('needs_update', 'rejected')"
    )
    ids = [r[0] for r in cur.fetchall()]
    conn.close()

    print(f"削除対象: {len(ids)} 件 (Tavily needs_update + rejected)", flush=True)

    deleted = 0
    for i, doc_id in enumerate(ids):
        ok = await mm.delete(doc_id)
        if ok:
            deleted += 1
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(ids)} 処理済み...", flush=True)

    print(f"削除完了: {deleted} 件", flush=True)
    await mm.close()


asyncio.run(main())
