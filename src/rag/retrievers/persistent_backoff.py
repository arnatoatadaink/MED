"""src/rag/retrievers/persistent_backoff.py — 汎用永続バックオフ

429 発生ごとにリクエスト間隔を指数的に延長し、SQLite へ保存する。

minutes_backoff: Level 0 = base_secs, Level N = multiplier * 2^N (N>=1)
days_backoff   : 1分超バックオフ → アクセス禁止。連続禁止で指数延長。
                 Level N → 2^(N-1) 日禁止 (N>=1)

緩和スケジュール:
  minutes_level: 1日1段階減少
  days_level   : 1週1段階減少

テーブル名は "{source_name}_backoff" — ArXiv なら既存の arxiv_backoff テーブルを使用。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import aiosqlite

logger = logging.getLogger(__name__)


@dataclass
class BackoffState:
    """DB へ永続化するバックオフ状態。"""

    minutes_level: int = 0  # 0-N; wait = wait_secs(level)
    minutes_date: str = ""  # minutes_level が最後に更新された日 (YYYY-MM-DD)
    days_level: int = 0     # 0=禁止なし; N → 2^(N-1) 日禁止
    ban_until: str = ""     # 禁止期限当日含む (YYYY-MM-DD)、"" = 禁止なし
    days_date: str = ""     # days_level が最後に更新された日 (YYYY-MM-DD)


# ---------------------------------------------------------------------------
# Pure functions (no I/O)
# ---------------------------------------------------------------------------


def wait_secs(level: int, multiplier: float = 10.0) -> float:
    """minutes_level から待機秒数を返す。全 level で multiplier * 2^level を使用。"""
    return multiplier * float(2 ** max(level, 0))


def ban_days(days_level: int) -> int:
    """days_level から禁止日数を返す (0 = 禁止なし)。"""
    if days_level <= 0:
        return 0
    return 2 ** (days_level - 1)


def apply_relaxation(
    state: BackoffState,
    today: date | None = None,
) -> BackoffState:
    """日次・週次の緩和を適用した新しい状態を返す (副作用なし)。

    minutes_level: 経過日数分だけ 1 段階ずつ減少 (日次)。
    days_level   : 経過週数分だけ 1 段階ずつ減少 (週次)。
    ban_until    : 期限切れなら空文字にリセット。
    """
    today = today or date.today()
    s = BackoffState(
        minutes_level=state.minutes_level,
        minutes_date=state.minutes_date,
        days_level=state.days_level,
        ban_until=state.ban_until,
        days_date=state.days_date,
    )

    if s.minutes_level > 0 and s.minutes_date:
        elapsed_days = (today - date.fromisoformat(s.minutes_date)).days
        reduction = min(elapsed_days, s.minutes_level)
        if reduction > 0:
            s.minutes_level -= reduction
            s.minutes_date = today.isoformat()

    if s.days_level > 0 and s.days_date:
        elapsed_weeks = (today - date.fromisoformat(s.days_date)).days // 7
        reduction = min(elapsed_weeks, s.days_level)
        if reduction > 0:
            s.days_level -= reduction
            s.days_date = today.isoformat()

    if s.ban_until and date.fromisoformat(s.ban_until) < today:
        s.ban_until = ""

    return s


def is_banned(state: BackoffState, today: date | None = None) -> bool:
    """今日がアクセス禁止期間内かを返す (ban_until 当日を含む)。"""
    today = today or date.today()
    if not state.ban_until:
        return False
    return date.fromisoformat(state.ban_until) >= today


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class PersistentBackoffStore:
    """SQLite でバックオフ状態を永続化する汎用ストア。

    テーブル {source_name}_backoff (key TEXT PK, value TEXT) に
    key-value 形式で保存する。

    source_name="arxiv" なら既存の arxiv_backoff テーブルをそのまま使用する。
    """

    _FIELDS = ("minutes_level", "minutes_date", "days_level", "ban_until", "days_date")

    def __init__(self, source_name: str, db_path: str) -> None:
        self._table = f"{source_name}_backoff"
        self._db_path = db_path
        self._ddl = f"""
        CREATE TABLE IF NOT EXISTS {self._table} (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """

    async def _ensure(self, db: aiosqlite.Connection) -> None:
        await db.execute(self._ddl)
        await db.commit()

    async def load(self) -> BackoffState:
        """DB から状態を読み込む。テーブルが存在しない場合は初期状態を返す。"""
        async with aiosqlite.connect(self._db_path) as db:
            await self._ensure(db)
            async with db.execute(f"SELECT key, value FROM {self._table}") as cur:
                rows = await cur.fetchall()
        kv = {k: v for k, v in rows}
        return BackoffState(
            minutes_level=int(kv.get("minutes_level", "0")),
            minutes_date=kv.get("minutes_date", ""),
            days_level=int(kv.get("days_level", "0")),
            ban_until=kv.get("ban_until", ""),
            days_date=kv.get("days_date", ""),
        )

    async def save(self, state: BackoffState) -> None:
        """状態を DB へ書き込む。"""
        values = {
            "minutes_level": str(state.minutes_level),
            "minutes_date": state.minutes_date,
            "days_level": str(state.days_level),
            "ban_until": state.ban_until,
            "days_date": state.days_date,
        }
        async with aiosqlite.connect(self._db_path) as db:
            await self._ensure(db)
            for key, value in values.items():
                await db.execute(
                    f"INSERT OR REPLACE INTO {self._table} (key, value) VALUES (?, ?)",
                    (key, value),
                )
            await db.commit()
