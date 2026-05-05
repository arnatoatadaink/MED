"""src/cycle/orchestrator.py — サイクル実行ディスパッチャ

gap_detect → enrich → dispatch の 3 フェーズを順に実行する。

UNREVIEWED_BACKLOG / LOW_QUALITY → cycle_tasks に記録のみ（Reviewer タブが担当）
SMALL_CLUSTER / SOURCE_IMBALANCE  → QueryRunner で外部収集して FAISS に投入
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Optional

from src.cycle.cycle_store import CycleStore
from src.cycle.gap_detector import GapDetector, GapDetectorConfig
from src.cycle.query_generator import QueryGenerator
from src.cycle.schema import CollectionTask, GapType

logger = logging.getLogger(__name__)

_MATURE_GAP_TYPES = {GapType.UNREVIEWED_BACKLOG, GapType.LOW_QUALITY}
_COLLECTOR_GAP_TYPES = {GapType.SMALL_CLUSTER, GapType.SOURCE_IMBALANCE}


class OrchestratorConfig:
    """Orchestrator のランタイム設定。

    provider / model: QueryGenerator (enrich フェーズ) で使用。
    persona / mature_interval: P-R1 分離後は未使用（後方互換のため残存）。
    """

    def __init__(
        self,
        provider: str = "fastflowlm",
        model: Optional[str] = None,
        persona: str = "auto",        # 未使用（Reviewer タブが管理）
        enrich_concurrency: int = 1,
        mature_interval: float = 0.0,  # 未使用（Reviewer タブが管理）
        db_path: str = "data/metadata.db",
        cache_path: str = "data/umap_cache.npz",
        detector_config: Optional[GapDetectorConfig] = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.persona = persona
        self.enrich_concurrency = enrich_concurrency
        self.mature_interval = mature_interval
        self.db_path = db_path
        self.cache_path = cache_path
        self.detector_config = detector_config or GapDetectorConfig()


class Orchestrator:
    """1 サイクル（gap_detect → enrich → dispatch）を実行する。

    Args:
        config: ランタイム設定。
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None) -> None:
        self._cfg = config or OrchestratorConfig()
        self._store = CycleStore(db_path=self._cfg.db_path)

    async def run_cycle(self) -> str:
        """1 サイクルを実行して run_id を返す。

        Returns:
            完了した run_id。
        """
        run_id = str(uuid.uuid4())
        await self._store.initialize()
        await self._store.create_run(run_id)
        logger.info("=== Cycle run started: %s ===", run_id)

        try:
            tasks = await self._phase_detect()
            if not tasks:
                logger.info("No gaps detected — cycle complete with no tasks")
                await self._store.finish_run(run_id, status="done", summary="no_gaps")
                return run_id

            tasks = await self._phase_enrich(tasks)
            await self._store.save_tasks(run_id, tasks)

            await self._phase_dispatch(run_id, tasks)

            counts = await self._store.count_tasks_by_status(run_id)
            summary = str(counts)
            await self._store.finish_run(run_id, status="done", summary=summary)
            logger.info("=== Cycle run done: %s  %s ===", run_id, summary)

        except Exception as exc:
            logger.exception("Cycle run failed: %s", exc)
            await self._store.finish_run(run_id, status="error", summary=str(exc))
            raise

        return run_id

    # ---- フェーズ -------------------------------------------------------

    async def _phase_detect(self) -> list[CollectionTask]:
        """GapDetector でタスクリストを生成する（同期処理を thread で実行）。"""
        cfg = self._cfg
        detector = GapDetector(
            cache_path=cfg.cache_path,
            db_path=cfg.db_path,
            config=cfg.detector_config,
        )
        tasks: list[CollectionTask] = await asyncio.to_thread(detector.detect)
        logger.info("Phase 1 (detect): %d tasks", len(tasks))
        return tasks

    async def _phase_enrich(self, tasks: list[CollectionTask]) -> list[CollectionTask]:
        """QueryGenerator で keywords / queries を補完する。"""
        gen = QueryGenerator(
            provider=self._cfg.provider,
            model=self._cfg.model or None,
            db_path=self._cfg.db_path,
        )
        enriched = await gen.enrich_batch(tasks, concurrency=self._cfg.enrich_concurrency)
        logger.info("Phase 2 (enrich): %d tasks enriched", len(enriched))
        return enriched

    async def _phase_dispatch(self, run_id: str, tasks: list[CollectionTask]) -> None:
        """gap_type に応じてタスクをディスパッチする。"""
        from src.cycle.query_runner import QueryRunner, QueryRunnerConfig

        runner: QueryRunner | None = None
        collector_tasks = [t for t in tasks if t.gap_type in _COLLECTOR_GAP_TYPES]
        if collector_tasks:
            runner = QueryRunner(QueryRunnerConfig())
            await runner.initialize()

        try:
            for task in tasks:
                if task.gap_type in _MATURE_GAP_TYPES:
                    await self._dispatch_mature(run_id, task)
                elif task.gap_type in _COLLECTOR_GAP_TYPES:
                    await self._dispatch_collector(run_id, task, runner)
                else:
                    logger.warning("Unknown gap_type %s — skipping task %s", task.gap_type, task.task_id)
        finally:
            if runner is not None:
                await runner.close()

    async def _dispatch_mature(self, run_id: str, task: CollectionTask) -> None:
        """UNREVIEWED_BACKLOG / LOW_QUALITY タスクを Reviewer タブへ委譲する。

        Orchestrator はギャップを記録するのみ。実際のレビューは GUI の
        Reviewer タブ（ReviewerSession）がマルチスレッドで実行する。
        """
        size = task.signals.get("size", "?")
        logger.info(
            "Queued for Reviewer: task %s (gap=%s, island_size=%s) "
            "— run Reviewer tab to process unreviewed/low_quality docs",
            task.task_id[:8], task.gap_type.value, size,
        )
        await self._store.update_task_status(task.task_id, "done")

    async def _dispatch_collector(
        self,
        run_id: str,
        task: CollectionTask,
        runner: object,
    ) -> None:
        """QueryRunner で外部ソースを検索して FAISS に投入する。"""
        from src.cycle.query_runner import QueryRunner

        await self._store.update_task_status(task.task_id, "collecting")
        logger.info(
            "Dispatching collector for task %s (gap=%s, %d queries)",
            task.task_id[:8], task.gap_type.value, len(task.queries),
        )
        try:
            stats = await runner.run_task(task)
            summary = f"added={stats['added']} retrieved={stats['retrieved']}"
            logger.info("Collector done for task %s: %s", task.task_id[:8], summary)
            await self._store.update_task_status(task.task_id, "done")
        except Exception as exc:
            logger.warning("Collector failed for task %s: %s", task.task_id[:8], exc)
            await self._store.update_task_status(task.task_id, "error")
