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
    from src.common.config import get_settings
    from src.llm.gateway import LLMGateway
    from src.memory.embedder import Embedder
    from src.memory.faiss_index import FAISSIndexManager
    from src.memory.memory_manager import MemoryManager
    from src.memory.metadata_store import MetadataStore
    from src.rag.retriever import RetrieverRouter

    settings = get_settings()
    gateway = LLMGateway(settings)
    embedder = Embedder()
    index_manager = FAISSIndexManager()
    metadata_store = MetadataStore(settings.memory.metadata_db_path)
    await metadata_store.initialize()
    memory_manager = MemoryManager(index_manager, metadata_store, embedder)
    retriever = RetrieverRouter(settings, gateway)

    print(f"[seed] Retrieving for: {query!r} (domain={domain}, top_k={top_k})")
    results = await retriever.retrieve(query, domain=domain, top_k=top_k)
    print(f"[seed] Retrieved {len(results)} docs")

    if dry_run:
        for i, r in enumerate(results[:3]):
            print(f"  [{i}] {r.content[:80]}...")
        return len(results)

    count = 0
    for result in results:
        try:
            await memory_manager.add(
                content=result.content,
                domain=domain,
                source=result.source,
                source_url=getattr(result, "url", ""),
            )
            count += 1
        except Exception as e:
            print(f"  [warn] Failed to add doc: {e}", file=sys.stderr)

    print(f"[seed] Added {count}/{len(results)} docs to FAISS memory")
    return count


async def seed_from_file(input_file: str, domain: str, dry_run: bool) -> int:
    """JSON ファイルからドキュメントを直接投入する。

    JSON 形式: [{"content": "...", "source": "...", "domain": "..."}, ...]
    """
    from src.common.config import get_settings
    from src.memory.embedder import Embedder
    from src.memory.faiss_index import FAISSIndexManager
    from src.memory.memory_manager import MemoryManager
    from src.memory.metadata_store import MetadataStore

    path = Path(input_file)
    if not path.exists():
        print(f"[error] File not found: {input_file}", file=sys.stderr)
        return 0

    with open(path, encoding="utf-8") as f:
        docs = json.load(f)

    print(f"[seed] Loaded {len(docs)} docs from {input_file}")
    if dry_run:
        print(f"[seed] Dry run: would add {len(docs)} docs")
        return len(docs)

    settings = get_settings()
    embedder = Embedder()
    index_manager = FAISSIndexManager()
    metadata_store = MetadataStore(settings.memory.metadata_db_path)
    await metadata_store.initialize()
    memory_manager = MemoryManager(index_manager, metadata_store, embedder)

    count = 0
    for doc in docs:
        try:
            await memory_manager.add(
                content=doc["content"],
                domain=doc.get("domain", domain),
                source=doc.get("source", "file"),
                source_url=doc.get("url", ""),
            )
            count += 1
        except Exception as e:
            print(f"  [warn] {e}", file=sys.stderr)

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
