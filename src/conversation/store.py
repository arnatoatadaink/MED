"""src/conversation/store.py — 会話履歴の永続化ストア（aiosqlite）。"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import aiosqlite

from src.conversation.schema import Session, Turn

logger = logging.getLogger(__name__)

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    title       TEXT NOT NULL,
    domain      TEXT NOT NULL DEFAULT 'general',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    turn_count  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sessions_user
    ON sessions(user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS turns (
    turn_id       TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    role          TEXT NOT NULL,
    content       TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    token_count   INTEGER NOT NULL DEFAULT 0,
    provider      TEXT NOT NULL DEFAULT '',
    model         TEXT NOT NULL DEFAULT '',
    faiss_doc_id  TEXT,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_turns_session
    ON turns(session_id, timestamp DESC);
"""


class ConversationStore:
    """会話セッション・ターンの SQLite CRUD。"""

    def __init__(self, db_path: str = "data/conversations.db") -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL;")
        await self._db.execute("PRAGMA foreign_keys=ON;")
        await self._db.executescript(_CREATE_SQL)
        await self._db.commit()
        logger.info("ConversationStore initialized: %s", self._db_path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    # ── セッション操作 ────────────────────────────────────────

    async def save_session(self, session: Session) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO sessions
              (session_id, user_id, title, domain, created_at, updated_at, turn_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
              title      = excluded.title,
              domain     = excluded.domain,
              updated_at = excluded.updated_at,
              turn_count = excluded.turn_count
            """,
            (
                session.session_id,
                session.user_id,
                session.title,
                session.domain,
                session.created_at.isoformat(),
                session.updated_at.isoformat(),
                session.turn_count,
            ),
        )
        await self._db.commit()

    async def get_session(self, session_id: str) -> Session | None:
        assert self._db is not None
        cur = await self._db.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        )
        row = await cur.fetchone()
        return _row_to_session(row) if row else None

    async def list_sessions(
        self, user_id: str, limit: int = 30
    ) -> list[Session]:
        assert self._db is not None
        cur = await self._db.execute(
            "SELECT * FROM sessions WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit),
        )
        rows = await cur.fetchall()
        return [_row_to_session(r) for r in rows]

    async def delete_session(self, session_id: str) -> bool:
        assert self._db is not None
        cur = await self._db.execute(
            "DELETE FROM sessions WHERE session_id = ?", (session_id,)
        )
        await self._db.commit()
        return cur.rowcount > 0

    async def touch_session(self, session_id: str) -> None:
        """updated_at を現在時刻に更新し turn_count を +1 する。"""
        assert self._db is not None
        await self._db.execute(
            """
            UPDATE sessions
            SET updated_at = ?, turn_count = turn_count + 1
            WHERE session_id = ?
            """,
            (datetime.utcnow().isoformat(), session_id),
        )
        await self._db.commit()

    async def count_sessions(self, user_id: str) -> int:
        assert self._db is not None
        cur = await self._db.execute(
            "SELECT COUNT(*) FROM sessions WHERE user_id = ?", (user_id,)
        )
        row = await cur.fetchone()
        return row[0] if row else 0

    async def delete_oldest_sessions(self, user_id: str, keep: int) -> int:
        """古いセッションを削除して keep 件だけ残す。返値は削除件数。"""
        assert self._db is not None
        cur = await self._db.execute(
            """
            DELETE FROM sessions
            WHERE user_id = ? AND session_id NOT IN (
                SELECT session_id FROM sessions
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
            )
            """,
            (user_id, user_id, keep),
        )
        await self._db.commit()
        return cur.rowcount

    # ── ターン操作 ────────────────────────────────────────────

    async def save_turn(self, turn: Turn) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT OR REPLACE INTO turns
              (turn_id, session_id, role, content, timestamp, token_count,
               provider, model, faiss_doc_id, input_tokens, output_tokens)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                turn.turn_id,
                turn.session_id,
                turn.role,
                turn.content,
                turn.timestamp.isoformat(),
                turn.token_count,
                turn.provider,
                turn.model,
                turn.faiss_doc_id,
                turn.input_tokens,
                turn.output_tokens,
            ),
        )
        await self._db.commit()

    async def get_turns(
        self, session_id: str, limit: int = 200
    ) -> list[Turn]:
        """時系列順（古い順）でターンを返す。"""
        assert self._db is not None
        cur = await self._db.execute(
            """
            SELECT * FROM turns
            WHERE session_id = ?
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (session_id, limit),
        )
        rows = await cur.fetchall()
        return [_row_to_turn(r) for r in rows]

    async def get_recent_turns_within_tokens(
        self, session_id: str, max_tokens: int
    ) -> list[Turn]:
        """新しい順にターンを取得し、累積トークン数が max_tokens 以内の分を返す（時系列順）。"""
        assert self._db is not None
        cur = await self._db.execute(
            """
            SELECT * FROM turns
            WHERE session_id = ?
            ORDER BY timestamp DESC
            """,
            (session_id,),
        )
        rows = await cur.fetchall()
        selected: list[Turn] = []
        total = 0
        for row in rows:
            turn = _row_to_turn(row)
            total += max(turn.token_count, 1)
            if total > max_tokens:
                break
            selected.append(turn)
        # 時系列順（古い順）に並び替えて返す
        selected.reverse()
        return selected

    async def update_turn_faiss_doc_id(
        self, turn_id: str, faiss_doc_id: str
    ) -> None:
        assert self._db is not None
        await self._db.execute(
            "UPDATE turns SET faiss_doc_id = ? WHERE turn_id = ?",
            (faiss_doc_id, turn_id),
        )
        await self._db.commit()


# ── 行 → データクラス変換 ──────────────────────────────────────


def _row_to_session(row: aiosqlite.Row) -> Session:
    return Session(
        session_id=row["session_id"],
        user_id=row["user_id"],
        title=row["title"],
        domain=row["domain"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        turn_count=row["turn_count"],
    )


def _row_to_turn(row: aiosqlite.Row) -> Turn:
    return Turn(
        turn_id=row["turn_id"],
        session_id=row["session_id"],
        role=row["role"],
        content=row["content"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
        token_count=row["token_count"],
        provider=row["provider"],
        model=row["model"],
        faiss_doc_id=row["faiss_doc_id"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
    )
