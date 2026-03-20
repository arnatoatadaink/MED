"""src/auth/store.py — ユーザー永続化ストア（aiosqlite）。"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import aiosqlite

from src.auth.schema import User

logger = logging.getLogger(__name__)

_CREATE_USERS_SQL = """
CREATE TABLE IF NOT EXISTS users (
    user_id          TEXT PRIMARY KEY,
    username         TEXT UNIQUE NOT NULL,
    hashed_password  TEXT,
    is_test          INTEGER NOT NULL DEFAULT 0,
    is_admin         INTEGER NOT NULL DEFAULT 0,
    is_active        INTEGER NOT NULL DEFAULT 1,
    created_at       TEXT NOT NULL,
    last_login       TEXT
);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
"""


class UserStore:
    """ユーザー情報を SQLite に保存する低レベルストア。"""

    def __init__(self, db_path: str = "data/users.db") -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_CREATE_USERS_SQL)
        await self._db.commit()
        logger.info("UserStore initialized: %s", self._db_path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    # ── 書き込み ──────────────────────────────────────────────

    async def save(self, user: User) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO users
              (user_id, username, hashed_password, is_test, is_admin,
               is_active, created_at, last_login)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
              username         = excluded.username,
              hashed_password  = excluded.hashed_password,
              is_test          = excluded.is_test,
              is_admin         = excluded.is_admin,
              is_active        = excluded.is_active,
              last_login       = excluded.last_login
            """,
            (
                user.user_id,
                user.username,
                user.hashed_password,
                int(user.is_test),
                int(user.is_admin),
                int(user.is_active),
                user.created_at.isoformat(),
                user.last_login.isoformat() if user.last_login else None,
            ),
        )
        await self._db.commit()

    async def update_last_login(self, user_id: str, ts: datetime) -> None:
        assert self._db is not None
        await self._db.execute(
            "UPDATE users SET last_login = ? WHERE user_id = ?",
            (ts.isoformat(), user_id),
        )
        await self._db.commit()

    async def delete(self, user_id: str) -> bool:
        assert self._db is not None
        cur = await self._db.execute(
            "DELETE FROM users WHERE user_id = ?", (user_id,)
        )
        await self._db.commit()
        return cur.rowcount > 0

    async def set_active(self, user_id: str, active: bool) -> None:
        assert self._db is not None
        await self._db.execute(
            "UPDATE users SET is_active = ? WHERE user_id = ?",
            (int(active), user_id),
        )
        await self._db.commit()

    # ── 読み取り ──────────────────────────────────────────────

    async def get_by_id(self, user_id: str) -> User | None:
        assert self._db is not None
        cur = await self._db.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        )
        row = await cur.fetchone()
        return _row_to_user(row) if row else None

    async def get_by_username(self, username: str) -> User | None:
        assert self._db is not None
        cur = await self._db.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        )
        row = await cur.fetchone()
        return _row_to_user(row) if row else None

    async def list_all(self, include_inactive: bool = False) -> list[User]:
        assert self._db is not None
        sql = "SELECT * FROM users"
        if not include_inactive:
            sql += " WHERE is_active = 1"
        sql += " ORDER BY created_at ASC"
        cur = await self._db.execute(sql)
        rows = await cur.fetchall()
        return [_row_to_user(r) for r in rows]

    async def exists_username(self, username: str) -> bool:
        assert self._db is not None
        cur = await self._db.execute(
            "SELECT 1 FROM users WHERE username = ?", (username,)
        )
        return await cur.fetchone() is not None


def _row_to_user(row: aiosqlite.Row) -> User:
    return User(
        user_id=row["user_id"],
        username=row["username"],
        hashed_password=row["hashed_password"],
        is_test=bool(row["is_test"]),
        is_admin=bool(row["is_admin"]),
        is_active=bool(row["is_active"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        last_login=(
            datetime.fromisoformat(row["last_login"])
            if row["last_login"]
            else None
        ),
    )
