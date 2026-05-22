"""scripts/seed_from_awep.py — AWEP 会話サマリーを episodic FAISS に投入する。

AWEP の /context/recent と /search/conversations エンドポイントから会話サマリーを取得し、
MED のエピソード記憶ゾーン（Domain.EPISODIC / memory_zone="episodic"）に投入する。

カーソル管理: data/awep_cursor.db に取込済み conversation_id を保存し差分取込を行う。

使い方:
    # 直近 20 件の会話サマリーを投入
    poetry run python scripts/seed_from_awep.py

    # 最大投入件数を指定
    poetry run python scripts/seed_from_awep.py --limit 50

    # 特定の検索語でまとめて取込（過去の会話を掘り起こす）
    poetry run python scripts/seed_from_awep.py --search-query "FAISS" --search-query "MED"

    # 取込済みカーソルをリセットして全件再投入
    poetry run python scripts/seed_from_awep.py --reset-cursor

    # dry-run（DB を変更せず投入予定を表示）
    poetry run python scripts/seed_from_awep.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

AWEP_BASE = os.environ.get("AWEP_API_URL", "http://127.0.0.1:8001")
CURSOR_DB = Path("data/awep_cursor.db")
AWEP_TIMEOUT = 5  # 秒

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# AWEP API クライアント
# ============================================================


def _awep_get(path: str, **params: object) -> list | dict:
    """AWEP API に GET リクエストを送りデコードした JSON を返す。"""
    qs = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    url = f"{AWEP_BASE}{path}?{qs}" if qs else f"{AWEP_BASE}{path}"
    try:
        with urllib.request.urlopen(url, timeout=AWEP_TIMEOUT) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        logger.warning("AWEP request failed (%s): %s", url, exc)
        return []


def _fetch_recent(n: int = 20) -> list[dict]:
    """最近の会話サマリーを返す（/context/recent）。"""
    result = _awep_get("/context/recent", n=n)
    if isinstance(result, dict):
        return result.get("conversations", [])
    return []


def _fetch_by_search(query: str, limit: int = 20) -> list[dict]:
    """/search/conversations でヒットした会話サマリーを返す。"""
    result = _awep_get("/search/conversations", q=query, limit=limit)
    if isinstance(result, dict):
        return result.get("results", [])
    return []


# ============================================================
# カーソル DB
# ============================================================


def _init_cursor_db(db_path: Path) -> sqlite3.Connection:
    """カーソル DB を初期化して接続を返す。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingested_conversations (
            conversation_id TEXT PRIMARY KEY,
            ingested_at     TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _is_ingested(conn: sqlite3.Connection, conv_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM ingested_conversations WHERE conversation_id = ?", (conv_id,)
    ).fetchone()
    return row is not None


def _mark_ingested(conn: sqlite3.Connection, conv_id: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO ingested_conversations VALUES (?, ?)",
        (conv_id, datetime.utcnow().isoformat()),
    )
    conn.commit()


def _reset_cursor(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM ingested_conversations")
    conn.commit()
    logger.info("Cursor reset: 取込履歴を全削除しました")


# ============================================================
# ドキュメント変換
# ============================================================


def _conv_to_content(conv: dict) -> str:
    """会話アイテムをエピソード記憶の content テキストに変換する。"""
    parts: list[str] = []

    summary = conv.get("summary_short", "").strip()
    if summary:
        parts.append(summary)

    topics: list[str] = conv.get("topics") or []
    if topics:
        parts.append("Topics: " + ", ".join(topics))

    created_at = conv.get("created_at", "")
    if created_at:
        parts.append(f"Date: {created_at[:10]}")

    return "\n".join(parts)


# ============================================================
# メイン処理
# ============================================================


async def _run(
    limit: int,
    search_queries: list[str],
    reset_cursor: bool,
    dry_run: bool,
) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.memory.memory_manager import MemoryManager
    from src.memory.schema import Document, Domain, SourceMeta, SourceType

    cursor_conn = _init_cursor_db(CURSOR_DB)
    if reset_cursor:
        _reset_cursor(cursor_conn)

    # 会話収集（重複除去しながら蓄積）
    seen: dict[str, dict] = {}
    for conv in _fetch_recent(n=20):
        cid = conv.get("conversation_id")
        if cid and cid not in seen:
            seen[cid] = conv

    for q in search_queries:
        for conv in _fetch_by_search(q, limit=20):
            cid = conv.get("conversation_id")
            if cid and cid not in seen:
                seen[cid] = conv

    logger.info("AWEP から %d 件の会話を取得", len(seen))

    # 未投入のみ絞り込み
    new_convs = [
        c for cid, c in seen.items()
        if not _is_ingested(cursor_conn, cid)
    ]
    logger.info("未投入: %d 件", len(new_convs))

    if not new_convs:
        logger.info("新規会話なし。終了します。")
        return

    # limit 適用（created_at 昇順で古い順から取込）
    new_convs.sort(key=lambda c: c.get("created_at", ""))
    new_convs = new_convs[:limit]
    logger.info("今回投入対象: %d 件（limit=%d）", len(new_convs), limit)

    if dry_run:
        for c in new_convs:
            logger.info("[dry-run] %s %s", c.get("conversation_id"), c.get("summary_short", "")[:60])
        return

    mm = MemoryManager()
    await mm.initialize()
    try:
        ingested = 0
        skipped = 0
        for conv in new_convs:
            conv_id: str = conv.get("conversation_id", "")
            content = _conv_to_content(conv)
            if not content.strip():
                logger.debug("skip empty content: %s", conv_id)
                skipped += 1
                continue

            # created_at パース
            created_at_str: str = conv.get("created_at", "")
            try:
                created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                created_at = datetime.utcnow()

            doc = Document(
                content=content,
                domain=Domain.EPISODIC,
                source=SourceMeta(
                    source_type=SourceType.AWEP,
                    url=f"{AWEP_BASE}/sessions/{conv.get('session_id','')}/conversations",
                    title=conv.get("summary_short", "")[:120],
                    extra={"conversation_id": conv_id, "topics": conv.get("topics", [])},
                ),
                created_at=created_at,
            )
            # memory_zone は model_validator で "episodic" に自動設定される

            try:
                await mm.add(doc)
                _mark_ingested(cursor_conn, conv_id)
                ingested += 1
                logger.debug("投入: %s %s", conv_id, content[:50])
            except Exception as exc:
                logger.warning("投入失敗 %s: %s", conv_id, exc)
                skipped += 1

        logger.info("完了: 投入=%d 件、スキップ=%d 件", ingested, skipped)
    finally:
        await mm.close()
        cursor_conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="AWEP 会話サマリーを episodic FAISS に投入する")
    parser.add_argument("--limit", type=int, default=50, help="最大投入件数 (default: 50)")
    parser.add_argument(
        "--search-query",
        dest="search_queries",
        action="append",
        default=[],
        metavar="QUERY",
        help="追加で /search/conversations を叩くクエリ（複数指定可）",
    )
    parser.add_argument("--reset-cursor", action="store_true", help="カーソルをリセットして全件再取込")
    parser.add_argument("--dry-run", action="store_true", help="DB を変更せず投入予定を表示")
    args = parser.parse_args()

    asyncio.run(
        _run(
            limit=args.limit,
            search_queries=args.search_queries,
            reset_cursor=args.reset_cursor,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
