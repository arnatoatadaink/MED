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
import sys

sys.path.insert(0, ".")
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _claim_docs_sync(db_path: str, source: str | None, limit: int) -> list:
    """needs_update docs を排他的に取得し unreviewed にマーク（同期・autocommit）。

    isolation_level=None (autocommit) の独立接続で UPDATE...RETURNING を実行する。
    SQLite の busy_timeout + Python レベルのリトライで DB ロック競合に対処する。
    """
    import sqlite3
    import time

    if source:
        sql = (
            "UPDATE documents SET review_status = 'unreviewed', updated_at = datetime('now') "
            "WHERE id IN ("
            "  SELECT id FROM documents WHERE review_status = 'needs_update' AND source_type = ?"
            "  ORDER BY created_at ASC LIMIT ?"
            ") RETURNING *"
        )
        params: tuple = (source, limit)
    else:
        sql = (
            "UPDATE documents SET review_status = 'unreviewed', updated_at = datetime('now') "
            "WHERE id IN ("
            "  SELECT id FROM documents WHERE review_status = 'needs_update'"
            "  ORDER BY created_at ASC LIMIT ?"
            ") RETURNING *"
        )
        params = (limit,)

    for attempt in range(15):  # 最大 ~60秒待機 (2+4+6+...秒)
        conn = sqlite3.connect(db_path, timeout=5, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        try:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
            return rows
        except sqlite3.OperationalError as e:
            if "locked" not in str(e) or attempt == 14:
                raise
            time.sleep(2 + attempt * 2)
        finally:
            conn.close()
    return []  # unreachable


async def _claim_docs(db_path: str, source: str | None, limit: int) -> list:
    """_claim_docs_sync をスレッドで非同期実行する。"""
    return await asyncio.to_thread(_claim_docs_sync, db_path, source, limit)


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
    from src.memory.metadata_store import _row_to_doc  # type: ignore[attr-defined]

    embedder = Embedder()
    mm = MemoryManager(embedder=embedder)
    await mm.initialize()
    gateway = LLMGateway()

    reviewer = MemoryReviewer(gateway, mm.store, provider=provider, model=model)
    tagger = DifficultyTagger(gateway, provider=provider, model=model)

    # needs_update を排他クレーム（並列起動時の重複処理を防ぐ）
    rows = await _claim_docs(mm.store._db_path, source, limit)
    docs = [_row_to_doc(row) for row in rows]

    if not docs:
        logger.info("needs_update ドキュメントが見つかりませんでした")
        await mm.close()
        return

    logger.info(
        "=== Re-mature %d needs_update docs (source=%s, provider=%s, model=%s) ===",
        len(docs), source or "all", provider, model,
    )

    reviewed = approved = needs_update = hold = tagged = errors = 0

    for i, doc in enumerate(docs):
        try:
            result = await reviewer.review(doc)
            reviewed += 1
        except Exception as e:
            logger.warning("  [%d/%d] Review error: %s", i + 1, len(docs), e)
            errors += 1
            continue

        if result.approved:
            approved += 1
            status = "PASS"
        elif result.needs_supplement:
            needs_update += 1
            status = "NEEDS_UPDATE"
        else:
            hold += 1
            status = "HOLD"

        logger.info(
            "  [%d/%d] %s (quality=%.2f): %s",
            i + 1, len(docs), status, result.quality_score, doc.content[:70],
        )

        # 難易度タグ (PASS のみ)
        if result.approved:
            try:
                await tagger.tag(doc)
                tagged += 1
            except Exception as e:
                logger.warning("  Tagging error: %s", e)

    await mm.close()

    # サマリー
    approval_pct = f"{approved / reviewed * 100:.1f}%" if reviewed else "N/A"
    print("\n" + "=" * 50)
    print("  RE-MATURE SUMMARY")
    print("=" * 50)
    print(f"       source: {source or 'all'}")
    print(f"     reviewed: {reviewed}")
    print(f"     approved: {approved}  ({approval_pct})")
    print(f"  needs_update: {needs_update}")
    print(f"         hold: {hold}")
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
