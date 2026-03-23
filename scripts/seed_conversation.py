#!/usr/bin/env python3
"""scripts/seed_conversation.py — 会話JSONファイルをFAISSへ投入

会話から抽出した知識（JSON配列）をFAISSメモリに一括投入する。

JSON形式:
    [
      {
        "content": "知識の内容...",
        "domain": "code" | "academic" | "general",
        "source": "conversation",
        "title": "タイトル"
      },
      ...
    ]

使い方:
    poetry run python scripts/seed_conversation.py data/seed_conversation_2026-03-23.json
    poetry run python scripts/seed_conversation.py data/seed_*.json
    poetry run python scripts/seed_conversation.py data/seed.json --dry-run
    poetry run python scripts/seed_conversation.py data/seed.json --domain academic
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


async def seed_from_file(
    files: list[Path],
    domain_override: str | None,
    dry_run: bool,
) -> None:
    from src.memory.embedder import Embedder
    from src.memory.memory_manager import MemoryManager
    from src.memory.schema import Document, Domain, SourceMeta, SourceType

    # 全ファイルからアイテムを読み込み
    all_items: list[tuple[Path, dict]] = []
    for fp in files:
        if not fp.exists():
            print(f"[seed] ERROR: {fp} not found, skipping")
            continue
        with open(fp, encoding="utf-8") as f:
            items = json.load(f)
        if not isinstance(items, list):
            print(f"[seed] ERROR: {fp} is not a JSON array, skipping")
            continue
        for item in items:
            all_items.append((fp, item))
        print(f"[seed] Loaded {len(items)} items from {fp.name}")

    if not all_items:
        print("[seed] No items to seed.")
        return

    print(f"[seed] Total: {len(all_items)} items from {len(files)} file(s)")

    if dry_run:
        print("[seed] Dry run — listing items without inserting:")
        for i, (fp, item) in enumerate(all_items, 1):
            domain = domain_override or item.get("domain", "general")
            print(f"  [{i}] [{domain}] {item.get('title', '(no title)')}")
        return

    # FAISS 初期化
    embedder = Embedder()
    mm = MemoryManager(embedder=embedder)
    await mm.initialize()

    added = 0
    failed = 0
    for i, (fp, item) in enumerate(all_items, 1):
        try:
            domain = domain_override or item.get("domain", "general")
            doc = Document(
                content=item["content"],
                domain=Domain(domain),
                source=SourceMeta(
                    source_type=SourceType.MANUAL,
                    title=item.get("title", ""),
                ),
            )
            await mm.add(doc)
            added += 1
            print(f"  [{i}/{len(all_items)}] {item.get('title', '(no title)')}")
        except Exception as e:
            failed += 1
            print(f"  [{i}/{len(all_items)}] FAILED: {e}")

    await mm.close()
    print(f"\n[seed] Done: {added} added, {failed} failed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed conversation knowledge JSON into FAISS memory",
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="JSON file(s) to seed (glob patterns supported from shell)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        choices=["code", "academic", "general"],
        help="Override domain for all items (default: use per-item domain)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List items without inserting into FAISS",
    )
    args = parser.parse_args()

    asyncio.run(seed_from_file(
        files=args.files,
        domain_override=args.domain,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
