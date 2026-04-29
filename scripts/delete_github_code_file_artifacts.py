#!/usr/bin/env python3
"""scripts/delete_github_code_file_artifacts.py — Pre-P2 github code_file アーティファクト削除

旧 GitHub Code Search retriever がファイルパスをそのまま content に格納していた
ドキュメント (source_type=github, content_type=code_file) を FAISS + SQLite から削除する。

P2 修正後の新 retriever は Contents API で実際のファイル内容を取得するため、
これらの古いパスのみのドキュメントは不要。

使い方:
    poetry run python scripts/delete_github_code_file_artifacts.py
    poetry run python scripts/delete_github_code_file_artifacts.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _get_target_ids(db_path: str) -> list[str]:
    """削除対象の doc ID を取得する。"""
    conn = sqlite3.connect(db_path, timeout=30)
    cur = conn.execute(
        """
        SELECT id FROM documents
        WHERE source_type = 'github'
          AND json_extract(source_extra, '$.content_type') = 'code_file'
        """
    )
    ids = [row[0] for row in cur.fetchall()]
    conn.close()
    return ids


async def _delete_all(ids: list[str], dry_run: bool) -> tuple[int, int]:
    """FAISS + SQLite から削除する。

    Returns:
        (success_count, fail_count)
    """
    from src.memory.memory_manager import MemoryManager

    mm = MemoryManager()
    await mm.initialize()

    success = 0
    fail = 0

    try:
        for doc_id in ids:
            if dry_run:
                logger.info("[DRY-RUN] would delete: %s", doc_id)
                success += 1
                continue

            ok = await mm.delete(doc_id)
            if ok:
                success += 1
                if success % 50 == 0:
                    logger.info("Deleted %d / %d ...", success, len(ids))
            else:
                logger.warning("Delete returned False for doc=%s", doc_id)
                fail += 1

        if not dry_run:
            # FAISS を保存
            await asyncio.get_event_loop().run_in_executor(None, mm.faiss.save)
            logger.info("FAISS saved.")
    finally:
        await mm.close()

    return success, fail


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-P2 github code_file アーティファクト削除")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="実際には削除せず対象を表示するだけ",
    )
    parser.add_argument(
        "--db-path", type=str, default=str(_ROOT / "data" / "metadata.db"),
        help="metadata.db パス",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="確認プロンプトをスキップして即時削除",
    )
    args = parser.parse_args()

    ids = _get_target_ids(args.db_path)
    print(f"削除対象: {len(ids)} 件 (source_type=github, content_type=code_file)")

    if not ids:
        print("削除対象なし。終了します。")
        return

    if args.dry_run:
        print(f"[DRY-RUN モード] 実際には削除しません\n最初の5件: {ids[:5]}")
        return
    elif not args.force:
        answer = input(f"{len(ids)} 件を削除します。続行しますか？ [y/N] ").strip().lower()
        if answer != "y":
            print("キャンセルしました。")
            return

    success, fail = asyncio.run(_delete_all(ids, args.dry_run))

    print(f"\n完了: 成功={success}, 失敗={fail}")
    if not args.dry_run:
        print("FAISS インデックスを保存しました。")


if __name__ == "__main__":
    main()
