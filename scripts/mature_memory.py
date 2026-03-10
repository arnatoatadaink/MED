#!/usr/bin/env python3
"""scripts/mature_memory.py — FAISSメモリ成熟スクリプト

Teacher Model による品質審査・難易度タグ付けを実行し、
メモリの品質目標（Phase 2: 10,000 docs, confidence>0.7, exec>80%）を確認する。

使い方:
    python scripts/mature_memory.py --check          # 品質チェックのみ
    python scripts/mature_memory.py --review         # 未審査ドキュメントを一括審査
    python scripts/mature_memory.py --tag-difficulty # 難易度タグ付け
    python scripts/mature_memory.py --all            # 全ステップ実行
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


async def check_quality(domain: str | None) -> None:
    """品質メトリクスを表示する。"""
    from src.common.config import get_settings
    from src.memory.maturation.quality_metrics import QualityMetrics
    from src.memory.metadata_store import MetadataStore

    settings = get_settings()
    store = MetadataStore(settings.memory.metadata_db_path)
    await store.initialize()

    metrics = QualityMetrics(store)
    report = await metrics.compute(domain=domain)
    print(report.summary())

    ready, missing = await metrics.check_phase2_readiness()
    if ready:
        print("\n[mature] ✓ Phase 2 quality goal MET!")
    else:
        print(f"\n[mature] ✗ Phase 2 goal NOT MET. Missing: {missing}")


async def review_docs(limit: int, domain: str | None) -> None:
    """未審査ドキュメントを Teacher で審査する。"""
    from src.common.config import get_settings
    from src.llm.gateway import LLMGateway
    from src.memory.maturation.reviewer import MemoryReviewer
    from src.memory.metadata_store import MetadataStore

    settings = get_settings()
    store = MetadataStore(settings.memory.metadata_db_path)
    await store.initialize()
    gateway = LLMGateway(settings)

    reviewer = MemoryReviewer(gateway, store)
    results = await reviewer.review_unreviewed(limit=limit)

    approved = sum(1 for r in results if r.approved)
    print(f"[mature] Reviewed {len(results)} docs: {approved} approved, {len(results)-approved} rejected")


async def tag_difficulty(limit: int, domain: str | None) -> None:
    """未タグドキュメントに難易度を付与する。"""
    from src.common.config import get_settings
    from src.llm.gateway import LLMGateway
    from src.memory.maturation.difficulty_tagger import DifficultyTagger
    from src.memory.metadata_store import MetadataStore

    settings = get_settings()
    store = MetadataStore(settings.memory.metadata_db_path)
    await store.initialize()
    gateway = LLMGateway(settings)

    tagger = DifficultyTagger(gateway)

    docs = await store.get_untagged(limit=limit)
    if not docs:
        print("[mature] No untagged documents found")
        return

    print(f"[mature] Tagging {len(docs)} documents...")
    count = 0
    for doc in docs:
        result = await tagger.tag(doc)
        await store.update_difficulty(doc.id, result.difficulty.value)
        count += 1

    print(f"[mature] Tagged {count} documents")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mature FAISS memory with Teacher Model")
    parser.add_argument("--check", action="store_true", help="Show quality metrics")
    parser.add_argument("--review", action="store_true", help="Review unreviewed docs")
    parser.add_argument("--tag-difficulty", action="store_true", help="Tag difficulty for untagged docs")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--limit", type=int, default=100, help="Max docs to process")
    parser.add_argument("--domain", type=str, default=None, help="Filter by domain")
    args = parser.parse_args()

    if not any([args.check, args.review, args.tag_difficulty, args.all]):
        parser.error("Specify at least one action: --check, --review, --tag-difficulty, --all")

    if args.all or args.review:
        print("\n=== Reviewing docs ===")
        asyncio.run(review_docs(args.limit, args.domain))

    if args.all or args.tag_difficulty:
        print("\n=== Tagging difficulty ===")
        asyncio.run(tag_difficulty(args.limit, args.domain))

    if args.all or args.check:
        print("\n=== Quality check ===")
        asyncio.run(check_quality(args.domain))


if __name__ == "__main__":
    main()
