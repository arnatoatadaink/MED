"""needs_update ドキュメントを再 mature するスクリプト。

Usage:
    poetry run python scripts/remature_needs_update.py \
        --source arxiv \
        --provider openrouter \
        --model nvidia/nemotron-nano-12b-v2-vl:free \
        --limit 500
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time

sys.path.insert(0, ".")
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def remature(
    source: str | None,
    provider: str,
    model: str | None,
    limit: int,
) -> None:
    from src.llm.gateway import LLMGateway
    from src.memory.embedder import Embedder
    from src.memory.maturation.difficulty_tagger import DifficultyTagger
    from src.memory.maturation.reviewer import MemoryReviewer
    from src.memory.memory_manager import MemoryManager

    embedder = Embedder()
    mm = MemoryManager(embedder=embedder)
    await mm.initialize()
    gateway = LLMGateway()

    reviewer = MemoryReviewer(gateway, mm.store, provider=provider, model=model)
    tagger = DifficultyTagger(gateway, provider=provider, model=model)

    # needs_update ドキュメントを取得
    if source:
        cursor = await mm.store._db.execute(
            "SELECT * FROM documents WHERE review_status = 'needs_update' AND source_type = ? "
            "ORDER BY created_at ASC LIMIT ?",
            (source, limit),
        )
    else:
        cursor = await mm.store._db.execute(
            "SELECT * FROM documents WHERE review_status = 'needs_update' "
            "ORDER BY created_at ASC LIMIT ?",
            (limit,),
        )

    from src.memory.metadata_store import _row_to_doc  # type: ignore[attr-defined]
    docs = [_row_to_doc(row) for row in await cursor.fetchall()]

    if not docs:
        logger.info("needs_update ドキュメントが見つかりませんでした")
        await mm.close()
        return

    logger.info(
        "=== Re-mature %d needs_update docs (source=%s, provider=%s, model=%s) ===",
        len(docs), source or "all", provider, model,
    )

    reviewed = approved = rejected = hold = tagged = errors = 0

    for i, doc in enumerate(docs):
        # まず unreviewed に戻してから審査
        try:
            result = await reviewer.review(doc)
            reviewed += 1
        except Exception as e:
            logger.warning("  [%d/%d] Review error: %s", i + 1, len(docs), e)
            errors += 1
            continue

        elapsed_label = ""
        if result.approved:
            approved += 1
            status = "PASS"
        elif result.needs_supplement:
            hold += 1
            status = "HOLD"
        else:
            rejected += 1
            status = "FAIL"

        logger.info(
            "  [%d/%d] %s (quality=%.2f): %s",
            i + 1, len(docs), status, result.quality_score, doc.content[:70],
        )

        # 難易度タグ (PASS のみ)
        if result.approved:
            try:
                tag_result = await tagger.tag(doc)
                tagged += 1
            except Exception as e:
                logger.warning("  Tagging error: %s", e)

    await mm.close()

    # サマリー
    print("\n" + "=" * 50)
    print("  RE-MATURE SUMMARY")
    print("=" * 50)
    print(f"       source: {source or 'all'}")
    print(f"     reviewed: {reviewed}")
    print(f"     approved: {approved}  ({approved/reviewed*100:.1f}%)" if reviewed else "     approved: 0")
    print(f"         hold: {hold}")
    print(f"     rejected: {rejected}")
    print(f"       tagged: {tagged}")
    print(f"       errors: {errors}")


def main() -> None:
    parser = argparse.ArgumentParser(description="needs_update ドキュメントを再 mature する")
    parser.add_argument("--source", default=None, help="ソース種別 (arxiv/tavily/stackoverflow/all)")
    parser.add_argument("--provider", default="openrouter", help="LLM プロバイダー")
    parser.add_argument("--model", default=None, help="モデル名")
    parser.add_argument("--limit", type=int, default=500, help="最大処理件数")
    args = parser.parse_args()

    source = None if args.source == "all" else args.source
    asyncio.run(remature(source, args.provider, args.model, args.limit))


if __name__ == "__main__":
    main()
