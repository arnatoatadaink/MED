"""src/cycle/reviewer_worker.py — マルチスレッド Reviewer ワーカー

複数モデルスロットで unreviewed / low_quality 文書を並列審査する。
タスクリストはメモリ上のみ管理（DBには完了時のみ書き込む）。
停止フラグ + タイムアウトによる安全な終了機構を持つ。
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.memory.maturation.personas import list_personas

log = logging.getLogger(__name__)


@dataclass
class ReviewTask:
    doc_id: str
    source_type: str
    domain_flag: str
    status: str = "pending"       # pending | in_progress | done | error
    assigned_to: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


@dataclass
class SlotConfig:
    provider: str
    model: str
    personas: list[str]           # このスロットが処理できるペルソナ


@dataclass
class ReviewerConfig:
    slots: list[SlotConfig] = field(default_factory=list)
    limit: int = 200
    timeout_sec: int = 60
    db_path: str = "data/metadata.db"
    include_low_quality: bool = True


def build_task_list(cfg: ReviewerConfig) -> list[ReviewTask]:
    """metadata.db から未レビュー・needs_update 文書のタスクリストを構築する。"""
    db_path = Path(cfg.db_path)
    if not db_path.exists():
        return []
    tasks: list[ReviewTask] = []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, source_type, extra FROM documents "
            "WHERE review_status = 'unreviewed' ORDER BY created_at ASC LIMIT ?",
            (cfg.limit,),
        ).fetchall()
        for r in rows:
            extra = {}
            try:
                extra = json.loads(r["extra"] or "{}")
            except Exception:
                pass
            tasks.append(ReviewTask(
                doc_id=r["id"],
                source_type=r["source_type"] or "unknown",
                domain_flag=extra.get("domain_flag", "on_domain"),
            ))
        if cfg.include_low_quality and len(tasks) < cfg.limit:
            remain = cfg.limit - len(tasks)
            seen = {t.doc_id for t in tasks}
            rows2 = conn.execute(
                "SELECT id, source_type, extra FROM documents "
                "WHERE review_status = 'needs_update' ORDER BY updated_at ASC LIMIT ?",
                (remain,),
            ).fetchall()
            for r in rows2:
                if r["id"] in seen:
                    continue
                extra = {}
                try:
                    extra = json.loads(r["extra"] or "{}")
                except Exception:
                    pass
                tasks.append(ReviewTask(
                    doc_id=r["id"],
                    source_type=r["source_type"] or "unknown",
                    domain_flag=extra.get("domain_flag", "on_domain"),
                ))
        conn.close()
    except Exception:
        log.exception("Failed to build task list")
    return tasks


def _resolve_persona(domain_flag: str, supported: list[str]) -> Optional[str]:
    """domain_flag に対応するペルソナを supported から決定する。"""
    if "auto" in supported:
        return "auto"
    if domain_flag in supported:
        return domain_flag
    return None


def _get_next_task(
    tasks: list[ReviewTask],
    lock: threading.Lock,
    supported: list[str],
    thread_name: str,
) -> Optional[ReviewTask]:
    """タスクリストをロックして次の処理可能タスクを取得する（ランダムスリープ後）。"""
    time.sleep(random.randint(100, 1000) / 1000.0)
    with lock:
        for task in tasks:
            if task.status != "pending":
                continue
            if _resolve_persona(task.domain_flag, supported) is None:
                continue
            task.status = "in_progress"
            task.assigned_to = thread_name
            task.started_at = time.time()
            return task
    return None


def _finish_task(tasks: list[ReviewTask], lock: threading.Lock, task: ReviewTask, status: str) -> None:
    time.sleep(random.randint(100, 1000) / 1000.0)
    with lock:
        task.status = status
        task.finished_at = time.time()


def _worker_thread(
    tasks: list[ReviewTask],
    lock: threading.Lock,
    stop_event: threading.Event,
    slot: SlotConfig,
    db_path: str,
) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_worker_async(tasks, lock, stop_event, slot, db_path))
    except Exception:
        log.exception("Worker thread error: %s/%s", slot.provider, slot.model)
    finally:
        loop.close()


async def _worker_async(
    tasks: list[ReviewTask],
    lock: threading.Lock,
    stop_event: threading.Event,
    slot: SlotConfig,
    db_path: str,
) -> None:
    from src.llm.gateway import LLMGateway
    from src.memory.maturation.reviewer import MemoryReviewer
    from src.memory.metadata_store import MetadataStore

    thread_name = threading.current_thread().name
    store = MetadataStore(db_path=db_path)
    await store.initialize()
    gateway = LLMGateway()
    try:
        while not stop_event.is_set():
            task = _get_next_task(tasks, lock, slot.personas, thread_name)
            if task is None:
                break
            persona = _resolve_persona(task.domain_flag, slot.personas) or "auto"
            reviewer = MemoryReviewer(
                gateway=gateway, store=store,
                provider=slot.provider, model=slot.model or None,
                persona=persona,
            )
            try:
                doc = await store.get(task.doc_id)
                if doc is None:
                    _finish_task(tasks, lock, task, "error")
                    continue
                await reviewer.review(doc)
                _finish_task(tasks, lock, task, "done")
                log.info("[%s] Reviewed %s (persona=%s)", thread_name, task.doc_id[:12], persona)
            except Exception:
                log.exception("[%s] Review failed: %s", thread_name, task.doc_id[:12])
                _finish_task(tasks, lock, task, "error")
    finally:
        await store.close()


class ReviewerSession:
    """マルチスレッド Reviewer セッションを管理する。"""

    def __init__(self, cfg: ReviewerConfig) -> None:
        self._cfg = cfg
        self._tasks: list[ReviewTask] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._started_at: Optional[float] = None

    def build(self) -> int:
        """タスクリストを構築して件数を返す。start() 前に呼ぶ。"""
        self._tasks = build_task_list(self._cfg)
        return len(self._tasks)

    def start(self) -> None:
        """スロット毎にワーカースレッドを起動する。"""
        self._stop_event.clear()
        self._started_at = time.time()
        for slot in self._cfg.slots:
            t = threading.Thread(
                target=_worker_thread,
                args=(self._tasks, self._lock, self._stop_event, slot, self._cfg.db_path),
                daemon=True,
                name=f"reviewer-{slot.provider}-{slot.model or 'default'}",
            )
            t.start()
            self._threads.append(t)
        log.info("ReviewerSession started: %d slots, %d tasks", len(self._threads), len(self._tasks))

    def stop(self) -> None:
        """停止フラグを立て、タイムアウト後にデーモンスレッドを放棄する。"""
        self._stop_event.set()
        deadline = time.time() + self._cfg.timeout_sec
        for t in self._threads:
            remain = max(0.0, deadline - time.time())
            t.join(timeout=remain)
            if t.is_alive():
                log.warning("Thread %s did not stop within timeout — abandoning", t.name)
        self._threads.clear()

    def get_stats(self) -> dict:
        """現在の進捗統計を返す。"""
        counts: dict[str, int] = {"pending": 0, "in_progress": 0, "done": 0, "error": 0}
        with self._lock:
            for t in self._tasks:
                counts[t.status] = counts.get(t.status, 0) + 1
        total = len(self._tasks)
        done = counts["done"] + counts["error"]
        elapsed = time.time() - (self._started_at or time.time())
        rate = done / elapsed if elapsed > 1 else 0.0
        remain = counts["pending"] + counts["in_progress"]
        eta_sec = round(remain / rate) if rate > 0 else None
        return {
            "total": total,
            "counts": counts,
            "elapsed_sec": round(elapsed),
            "eta_sec": eta_sec,
            "is_running": any(t.is_alive() for t in self._threads),
        }

    def get_task_rows(self) -> list[dict]:
        """UI テーブル用の行データを返す（最大 200 件）。"""
        with self._lock:
            return [
                {
                    "doc_id": t.doc_id[:16],
                    "source": t.source_type,
                    "persona": t.domain_flag,
                    "status": t.status,
                    "assigned": t.assigned_to or "—",
                }
                for t in self._tasks[:200]
            ]


def get_persona_choices() -> list[str]:
    """UI で選択可能なペルソナ一覧を返す。"""
    return list_personas()
