#!/usr/bin/env python3
"""scripts/mature_memory.py — FAISSメモリ成熟スクリプト

Teacher Model による品質審査・難易度タグ付けを実行し、
メモリの品質目標（Phase 2: 10,000 docs, confidence>0.7, exec>80%）を確認する。

使い方:
    # 品質チェックのみ
    poetry run python scripts/mature_memory.py --check

    # LM Studio Teacher で未審査ドキュメントを一括審査
    poetry run python scripts/mature_memory.py --review --provider lmstudio --limit 50

    # 難易度タグ付け
    poetry run python scripts/mature_memory.py --tag-difficulty --provider lmstudio

    # 全ステップ実行
    poetry run python scripts/mature_memory.py --all --provider lmstudio
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
    store = MetadataStore(db_path=str(settings.metadata.db_path))
    await store.initialize()

    metrics = QualityMetrics(store)
    report = await metrics.compute(domain=domain)
    print(report.summary())

    ready, missing = await metrics.check_phase2_readiness()
    if ready:
        print("\n[mature] Phase 2 quality goal MET!")
    else:
        print(f"\n[mature] Phase 2 goal NOT MET. Missing: {missing}")


async def review_docs(
    limit: int, domain: str | None,
    provider: str | None = None,
    concurrency: int = 1,
) -> None:
    """未審査ドキュメントを Teacher で審査する。"""
    from src.common.config import get_settings
    from src.llm.gateway import LLMGateway
    from src.memory.maturation.reviewer import MemoryReviewer
    from src.memory.metadata_store import MetadataStore

    settings = get_settings()
    store = MetadataStore(db_path=str(settings.metadata.db_path))
    await store.initialize()
    gateway = LLMGateway(settings)

    reviewer = MemoryReviewer(gateway, store, provider=provider)

    docs = await store.get_unreviewed(domain=domain, limit=limit)
    if not docs:
        print("[mature] No unreviewed documents found")
        return

    print(f"[mature] Reviewing {len(docs)} docs (provider={provider or 'default'}, concurrency={concurrency})")

    results = []
    for i, doc in enumerate(docs):
        try:
            result = await reviewer.review(doc)
            results.append(result)
            status = "PASS" if result.approved else "FAIL"
            print(
                f"  [{i+1}/{len(docs)}] {status} "
                f"(quality={result.quality_score:.2f} confidence={result.confidence:.2f}) "
                f"{doc.content[:60]}..."
            )
        except Exception as e:
            print(f"  [{i+1}/{len(docs)}] ERROR: {e}", file=sys.stderr)

    approved = sum(1 for r in results if r.approved)
    print(f"\n[mature] Reviewed {len(results)} docs: {approved} approved, {len(results)-approved} rejected")


async def tag_difficulty(
    limit: int, domain: str | None,
    provider: str | None = None,
) -> None:
    """未タグドキュメントに難易度を付与する。"""
    from src.common.config import get_settings
    from src.llm.gateway import LLMGateway
    from src.memory.maturation.difficulty_tagger import DifficultyTagger
    from src.memory.metadata_store import MetadataStore

    settings = get_settings()
    store = MetadataStore(db_path=str(settings.metadata.db_path))
    await store.initialize()
    gateway = LLMGateway(settings)

    tagger = DifficultyTagger(gateway, provider=provider)

    # difficulty が NULL のドキュメントを取得
    docs = await store.get_unreviewed(domain=domain, limit=limit)
    untagged = [d for d in docs if d.difficulty is None]
    if not untagged:
        print("[mature] No untagged documents found")
        return

    print(f"[mature] Tagging {len(untagged)} documents (provider={provider or 'default'})...")
    count = 0
    for i, doc in enumerate(untagged):
        try:
            level = await tagger.tag(doc)
            await store.update_quality(doc.id, difficulty=level.value)
            count += 1
            print(f"  [{i+1}/{len(untagged)}] {level.value}: {doc.content[:60]}...")
        except Exception as e:
            print(f"  [{i+1}/{len(untagged)}] ERROR: {e}", file=sys.stderr)

    print(f"[mature] Tagged {count} documents")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mature FAISS memory with Teacher Model")
    parser.add_argument("--check", action="store_true", help="Show quality metrics")
    parser.add_argument("--review", action="store_true", help="Review unreviewed docs")
    parser.add_argument("--tag-difficulty", action="store_true", help="Tag difficulty for untagged docs")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--limit", type=int, default=100, help="Max docs to process")
    parser.add_argument("--domain", type=str, default=None, help="Filter by domain")
    parser.add_argument("--provider", type=str, default=None, help="LLM provider (e.g. lmstudio)")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent review requests (local models: 1)")
    args = parser.parse_args()

    if not any([args.check, args.review, args.tag_difficulty, args.all]):
        parser.error("Specify at least one action: --check, --review, --tag-difficulty, --all")

    if args.all or args.review:
        print("\n=== Reviewing docs ===")
        asyncio.run(review_docs(args.limit, args.domain, provider=args.provider, concurrency=args.concurrency))

    if args.all or args.tag_difficulty:
        print("\n=== Tagging difficulty ===")
        asyncio.run(tag_difficulty(args.limit, args.domain, provider=args.provider))

    if args.all or args.check:
        print("\n=== Quality check ===")
        asyncio.run(check_quality(args.domain))


if __name__ == "__main__":
    main()
