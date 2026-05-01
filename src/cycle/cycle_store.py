"""src/cycle/cycle_store.py — サイクル状態の永続化 (SQLite)

CollectionTask を metadata.db に保存・読み込みし、
Orchestrator の再起動後にも状態を復元できるようにする。

テーブル:
  cycle_runs  — 1回のサイクル実行（gap_detect → query → collect → mature）
  cycle_tasks — CollectionTask の永続化
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from src.cycle.schema import CollectionTask, GapType

logger = logging.getLogger(__name__)

_DB_PATH = "data/metadata.db"

_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS cycle_runs (
    run_id       TEXT PRIMARY KEY,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    status       TEXT NOT NULL DEFAULT 'running',
    summary      TEXT
)
"""

_CREATE_TASKS = """
CREATE TABLE IF NOT EXISTS cycle_tasks (
    task_id      TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL,
    gap_type     TEXT NOT NULL,
    priority     REAL NOT NULL,
    reason       TEXT NOT NULL,
    signals      TEXT NOT NULL,   -- JSON
    keywords     TEXT NOT NULL DEFAULT '[]',  -- JSON
    queries      TEXT NOT NULL DEFAULT '[]',  -- JSON
    status       TEXT NOT NULL DEFAULT 'pending',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES cycle_runs(run_id)
)
"""

_TASK_STATUS = ("pending", "enriched", "collecting", "done", "error")


class CycleStore:
    """CollectionTask / サイクル実行状態を SQLite に保存する。

    Args:
        db_path: metadata.db のパス。
    """

    def __init__(self, db_path: str = _DB_PATH) -> None:
        self._db_path = db_path

    # ---- 初期化 ----------------------------------------------------

    async def initialize(self) -> None:
        """テーブルを作成する（べき等）。"""
        await asyncio.to_thread(self._create_tables)

    def _create_tables(self) -> None:
        conn = sqlite3.connect(self._db_path)
        conn.executescript(_CREATE_RUNS + ";\n" + _CREATE_TASKS)
        conn.commit()
        conn.close()
        logger.debug("CycleStore tables ensured")

    # ---- cycle_runs ------------------------------------------------

    async def create_run(self, run_id: str) -> None:
        """新しいサイクル実行レコードを作成する。"""
        now = _now()

        def _write() -> None:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                "INSERT INTO cycle_runs (run_id, started_at, status) VALUES (?, ?, 'running')",
                (run_id, now),
            )
            conn.commit()
            conn.close()

        await asyncio.to_thread(_write)
        logger.info("Cycle run created: %s", run_id)

    async def finish_run(
        self,
        run_id: str,
        status: str = "done",
        summary: Optional[str] = None,
    ) -> None:
        """サイクル実行を完了状態にする。"""
        now = _now()

        def _write() -> None:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                "UPDATE cycle_runs SET finished_at=?, status=?, summary=? WHERE run_id=?",
                (now, status, summary, run_id),
            )
            conn.commit()
            conn.close()

        await asyncio.to_thread(_write)

    async def list_runs(self, limit: int = 20) -> list[dict]:
        """最近のサイクル実行リストを返す。"""
        def _query() -> list[dict]:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM cycle_runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]

        return await asyncio.to_thread(_query)

    # ---- cycle_tasks -----------------------------------------------

    async def save_tasks(self, run_id: str, tasks: list[CollectionTask]) -> None:
        """CollectionTask リストを DB に保存する。"""
        now = _now()

        def _write() -> None:
            conn = sqlite3.connect(self._db_path)
            for t in tasks:
                conn.execute(
                    """INSERT OR REPLACE INTO cycle_tasks
                       (task_id, run_id, gap_type, priority, reason,
                        signals, keywords, queries, status, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        t.task_id, run_id, t.gap_type.value, t.priority, t.reason,
                        json.dumps(t.signals, ensure_ascii=False),
                        json.dumps(t.keywords, ensure_ascii=False),
                        json.dumps(t.queries, ensure_ascii=False),
                        "enriched" if (t.keywords or t.queries) else "pending",
                        t.created_at, now,
                    ),
                )
            conn.commit()
            conn.close()

        await asyncio.to_thread(_write)
        logger.info("Saved %d tasks for run %s", len(tasks), run_id)

    async def load_tasks(
        self,
        run_id: str,
        status: Optional[str] = None,
    ) -> list[CollectionTask]:
        """run_id に属するタスクを CollectionTask として返す。"""
        def _query() -> list[CollectionTask]:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            if status:
                rows = conn.execute(
                    "SELECT * FROM cycle_tasks WHERE run_id=? AND status=? ORDER BY priority DESC",
                    (run_id, status),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM cycle_tasks WHERE run_id=? ORDER BY priority DESC",
                    (run_id,),
                ).fetchall()
            conn.close()
            return [_row_to_task(r) for r in rows]

        return await asyncio.to_thread(_query)

    async def update_task_status(self, task_id: str, status: str) -> None:
        """タスクのステータスを更新する。"""
        assert status in _TASK_STATUS, f"Invalid status: {status}"
        now = _now()

        def _write() -> None:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                "UPDATE cycle_tasks SET status=?, updated_at=? WHERE task_id=?",
                (status, now, task_id),
            )
            conn.commit()
            conn.close()

        await asyncio.to_thread(_write)

    async def get_latest_run_id(self) -> Optional[str]:
        """最新の完了済みサイクル run_id を返す。なければ None。"""
        def _query() -> Optional[str]:
            conn = sqlite3.connect(self._db_path)
            row = conn.execute(
                "SELECT run_id FROM cycle_runs WHERE status='done' ORDER BY finished_at DESC LIMIT 1"
            ).fetchone()
            conn.close()
            return row[0] if row else None

        return await asyncio.to_thread(_query)

    async def count_tasks_by_status(self, run_id: str) -> dict[str, int]:
        """run_id のタスク数をステータス別にカウントする。"""
        def _query() -> dict[str, int]:
            conn = sqlite3.connect(self._db_path)
            rows = conn.execute(
                "SELECT status, COUNT(*) FROM cycle_tasks WHERE run_id=? GROUP BY status",
                (run_id,),
            ).fetchall()
            conn.close()
            return {r[0]: r[1] for r in rows}

        return await asyncio.to_thread(_query)


# ---- ヘルパー -------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_task(row: sqlite3.Row) -> CollectionTask:
    return CollectionTask(
        task_id    = row["task_id"],
        created_at = row["created_at"],
        gap_type   = GapType(row["gap_type"]),
        priority   = float(row["priority"]),
        reason     = row["reason"],
        signals    = json.loads(row["signals"]),
        keywords   = json.loads(row["keywords"]),
        queries    = json.loads(row["queries"]),
    )
