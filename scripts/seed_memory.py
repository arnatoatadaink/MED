#!/usr/bin/env python3
"""scripts/seed_memory.py — FAISSメモリ初期シードスクリプト

RAG パイプラインで取得したドキュメントを FAISS メモリに投入する。

使い方:
    python scripts/seed_memory.py --query "FAISS vector search" --domain code
    python scripts/seed_memory.py --queries-file queries.txt --domain academic
    python scripts/seed_memory.py --input-file docs.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


async def seed_from_query(query: str, domain: str, top_k: int, dry_run: bool) -> int:
    """1クエリ分のシードを実行し、投入件数を返す。"""
    from src.memory.embedder import Embedder
    from src.memory.memory_manager import MemoryManager
    from src.memory.schema import Document, Domain, SourceMeta, SourceType
    from src.rag.retriever import RetrieverRouter

    retriever = RetrieverRouter()

    print(f"[seed] Retrieving for: {query!r} (domain={domain}, top_k={top_k})")
    results = await retriever.search(query, max_results=top_k)
    print(f"[seed] Retrieved {len(results)} docs")

    if dry_run:
        for i, r in enumerate(results[:5]):
            print(f"  [{i}] ({r.source}) {r.content[:100]}...")
        return len(results)

    embedder = Embedder()
    mm = MemoryManager(embedder=embedder)
    await mm.initialize()

    # source 名 → SourceType マッピング
    _SOURCE_MAP = {s.value: s for s in SourceType}

    count = 0
    for result in results:
        try:
            doc = Document(
                content=result.content,
                domain=Domain(domain),
                source=SourceMeta(
                    source_type=_SOURCE_MAP.get(result.source, SourceType.MANUAL),
                    url=result.url,
                    title=result.title,
                ),
            )
            await mm.add(doc)
            count += 1
        except Exception as e:
            print(f"  [warn] Failed to add doc: {e}", file=sys.stderr)

    await mm.close()
    print(f"[seed] Added {count}/{len(results)} docs to FAISS memory")
    return count


async def seed_from_file(input_file: str, domain: str, dry_run: bool) -> int:
    """JSON ファイルからドキュメントを直接投入する。

    JSON 形式: [{"content": "...", "source": "...", "domain": "..."}, ...]
    """
    from src.memory.embedder import Embedder
    from src.memory.memory_manager import MemoryManager
    from src.memory.schema import Document, Domain, SourceMeta, SourceType

    path = Path(input_file)
    if not path.exists():
        print(f"[error] File not found: {input_file}", file=sys.stderr)
        return 0

    with open(path, encoding="utf-8") as f:
        docs = json.load(f)

    print(f"[seed] Loaded {len(docs)} docs from {input_file}")
    if dry_run:
        for i, d in enumerate(docs[:5]):
            print(f"  [{i}] {d.get('content', '')[:100]}...")
        return len(docs)

    embedder = Embedder()
    mm = MemoryManager(embedder=embedder)
    await mm.initialize()

    _SOURCE_MAP = {s.value: s for s in SourceType}

    count = 0
    for d in docs:
        try:
            doc = Document(
                content=d["content"],
                domain=Domain(d.get("domain", domain)),
                source=SourceMeta(
                    source_type=_SOURCE_MAP.get(d.get("source", "manual"), SourceType.MANUAL),
                    url=d.get("url", ""),
                    title=d.get("title", ""),
                ),
            )
            await mm.add(doc)
            count += 1
        except Exception as e:
            print(f"  [warn] {e}", file=sys.stderr)

    await mm.close()
    print(f"[seed] Seeded {count}/{len(docs)} docs")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed FAISS memory with documents")
    parser.add_argument("--query", type=str, help="Single query to retrieve and seed")
    parser.add_argument("--queries-file", type=str, help="File with one query per line")
    parser.add_argument("--input-file", type=str, help="JSON file with pre-fetched docs")
    parser.add_argument("--domain", default="general", choices=["code", "academic", "general"])
    parser.add_argument("--top-k", type=int, default=10, help="Docs per query")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    if not any([args.query, args.queries_file, args.input_file]):
        parser.error("Specify --query, --queries-file, or --input-file")

    total = 0

    if args.input_file:
        total += asyncio.run(seed_from_file(args.input_file, args.domain, args.dry_run))

    queries: list[str] = []
    if args.query:
        queries.append(args.query)
    if args.queries_file:
        queries.extend(Path(args.queries_file).read_text().splitlines())

    for q in queries:
        q = q.strip()
        if q:
            total += asyncio.run(seed_from_query(q, args.domain, args.top_k, args.dry_run))

    print(f"\n[seed] Total docs processed: {total}")


if __name__ == "__main__":
    main()
