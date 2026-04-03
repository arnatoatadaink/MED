"""src/llm/daily_usage_tracker.py — OpenRouter 等の日別・ジョブ別リクエスト使用量管理

機能:
- 日別リクエスト数をSQLiteに永続化（日付変更で自動リセット）
- ジョブ（スクリプト実行単位）ごとの使用数を記録
- 日次上限に近づいたら警告、上限到達で DailyLimitExceeded を raise

スキーマ:
  daily_usage  : date × provider → total_requests
  job_usage    : job_id × provider → request_count + タイムスタンプ

使用例:
    tracker = DailyUsageTracker("data/openrouter_usage.db")
    await tracker.initialize()
    await tracker.check_and_increment("openrouter", daily_limit=1000)
    await tracker.close()
"""
from __future__ import annotations

import asyncio
import datetime
import logging
import os
import socket
import sys
import uuid

import aiosqlite

logger = logging.getLogger(__name__)

_WARN_THRESHOLD = 0.90  # 90% 到達で警告


class DailyLimitExceeded(RuntimeError):
    """1日のリクエスト上限を超えた場合に raise される例外。"""

    def __init__(self, provider: str, limit: int, current: int) -> None:
        self.provider = provider
        self.limit = limit
        self.current = current
        super().__init__(
            f"Provider '{provider}' daily request limit reached: {current}/{limit}. "
            "Stopping to avoid paid overage."
        )


class DailyUsageTracker:
    """日別・ジョブ別リクエスト使用量を SQLite で管理するトラッカー。"""

    _DDL = [
        """
        CREATE TABLE IF NOT EXISTS daily_usage (
            date     TEXT NOT NULL,
            provider TEXT NOT NULL,
            total_requests INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (date, provider)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS job_usage (
            job_id     TEXT NOT NULL,
            provider   TEXT NOT NULL,
            started_at TEXT NOT NULL,
            last_request_at TEXT,
            request_count INTEGER NOT NULL DEFAULT 0,
            script_name   TEXT,
            PRIMARY KEY (job_id, provider)
        )
        """,
    ]

    def __init__(self, db_path: str = "data/openrouter_usage.db") -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        # このプロセス内で一意なジョブ ID
        self._job_id = _make_job_id()
        self._warned: dict[str, bool] = {}  # provider → 警告済みフラグ

    @property
    def job_id(self) -> str:
        return self._job_id

    async def initialize(self) -> None:
        """DB を開き、テーブルを作成する。"""
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        for ddl in self._DDL:
            await self._db.execute(ddl)
        await self._db.commit()
        logger.debug("DailyUsageTracker initialized: %s (job_id=%s)", self._db_path, self._job_id)

    async def check_and_increment(
        self,
        provider: str,
        daily_limit: int,
    ) -> int:
        """リクエスト前にカウントを確認・加算する。

        Returns:
            加算後の本日の累計リクエスト数

        Raises:
            DailyLimitExceeded: 上限に達していた場合
            RuntimeError: initialize() が未呼び出しの場合
        """
        if self._db is None:
            raise RuntimeError("DailyUsageTracker.initialize() が未呼び出しです")

        today = _today()
        now = _now_iso()

        async with self._lock:
            # ── daily_usage を upsert ──
            await self._db.execute(
                """
                INSERT INTO daily_usage (date, provider, total_requests, updated_at)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(date, provider) DO UPDATE SET
                    total_requests = total_requests + 1,
                    updated_at = excluded.updated_at
                """,
                (today, provider, now),
            )

            # ── job_usage を upsert ──
            script = _script_name()
            await self._db.execute(
                """
                INSERT INTO job_usage (job_id, provider, started_at, last_request_at, request_count, script_name)
                VALUES (?, ?, ?, ?, 1, ?)
                ON CONFLICT(job_id, provider) DO UPDATE SET
                    last_request_at = excluded.last_request_at,
                    request_count   = request_count + 1
                """,
                (self._job_id, provider, now, now, script),
            )

            await self._db.commit()

            # 最新値を取得
            cur = await self._db.execute(
                "SELECT total_requests FROM daily_usage WHERE date=? AND provider=?",
                (today, provider),
            )
            row = await cur.fetchone()
            total = row["total_requests"] if row else 1

        # 上限チェック
        if total > daily_limit:
            raise DailyLimitExceeded(provider, daily_limit, total)

        # 警告（90%）
        if total >= daily_limit * _WARN_THRESHOLD and not self._warned.get(provider):
            self._warned[provider] = True
            logger.warning(
                "[DailyUsageTracker] Provider '%s': %d/%d requests today (%.0f%%). "
                "Approaching daily limit!",
                provider, total, daily_limit, total / daily_limit * 100,
            )
        elif total % 100 == 0:
            logger.info(
                "[DailyUsageTracker] Provider '%s': %d/%d requests today",
                provider, total, daily_limit,
            )

        return total

    async def get_today_summary(self, provider: str | None = None) -> list[dict]:
        """本日の使用サマリーを返す。"""
        if self._db is None:
            return []
        today = _today()
        if provider:
            cur = await self._db.execute(
                "SELECT * FROM daily_usage WHERE date=? AND provider=?",
                (today, provider),
            )
        else:
            cur = await self._db.execute(
                "SELECT * FROM daily_usage WHERE date=? ORDER BY provider",
                (today,),
            )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def get_job_summary(self, job_id: str | None = None) -> list[dict]:
        """ジョブ別使用サマリーを返す。"""
        if self._db is None:
            return []
        target_id = job_id or self._job_id
        cur = await self._db.execute(
            "SELECT * FROM job_usage WHERE job_id=? ORDER BY provider",
            (target_id,),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def get_recent_jobs(self, limit: int = 10) -> list[dict]:
        """最近のジョブ一覧を返す。"""
        if self._db is None:
            return []
        cur = await self._db.execute(
            """
            SELECT job_id, provider, started_at, last_request_at, request_count, script_name
            FROM job_usage
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None


# ──────────────────────────────────────────
# ヘルパー関数
# ──────────────────────────────────────────

def _today() -> str:
    return datetime.date.today().isoformat()


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _script_name() -> str:
    """呼び出し元スクリプト名を返す（不明な場合は 'unknown'）。"""
    try:
        return os.path.basename(sys.argv[0]) if sys.argv else "unknown"
    except Exception:
        return "unknown"


def _make_job_id() -> str:
    """スクリプト名 + 日時 + ホスト短縮 の一意 ID を生成する。"""
    script = os.path.splitext(_script_name())[0][:20]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    host = socket.gethostname()[:8]
    return f"{script}_{ts}_{host}"
