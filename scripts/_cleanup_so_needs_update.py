"""SO needs_update 18件を削除する。"""
import asyncio, sys
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()

import sqlite3
from src.memory.embedder import Embedder
from src.memory.memory_manager import MemoryManager

async def main():
    embedder = Embedder()
    mm = MemoryManager(embedder=embedder)
    await mm.initialize()

    conn = sqlite3.connect("data/metadata.db")
    cur = conn.cursor()
    cur.execute("SELECT id FROM documents WHERE source_type='stackoverflow' AND review_status='needs_update'")
    ids = [r[0] for r in cur.fetchall()]
    conn.close()

    print(f"削除対象: {len(ids)} 件 (SO needs_update)", flush=True)
    deleted = 0
    for doc_id in ids:
        ok = await mm.delete(doc_id)
        if ok:
            deleted += 1
    print(f"削除完了: {deleted} 件", flush=True)
    await mm.close()

asyncio.run(main())
