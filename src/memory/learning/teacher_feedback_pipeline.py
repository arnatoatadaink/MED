"""src/memory/learning/teacher_feedback_pipeline.py — Teacher フィードバックパイプライン

FeedbackCollector が蓄積したフィードバックイベントを TeacherRegistry に転送し、
Teacher の trust_score を自動更新するパイプライン。

フロー:
  1. FeedbackCollector.drain() でイベントを取り出す
  2. MetadataStore.get_batch() で各ドキュメントの teacher_id を解決する
  3. TeacherRegistry.record_feedback() を呼んで EWMA で trust_score を更新する
  4. （オプション）CompositeScorer.update_store() で composite_score を再計算する

使い方:
    from src.memory.learning.teacher_feedback_pipeline import TeacherFeedbackPipeline

    pipeline = TeacherFeedbackPipeline(
        collector=feedback_collector,
        store=metadata_store,
        registry=teacher_registry,
        scorer=composite_scorer,   # 省略可
    )

    # フィードバックを溜めておいて定期的に flush
    result = await pipeline.flush()
    print(result.summary())

    # 周期実行
    await pipeline.start_background_loop(interval_seconds=60)
    ...
    await pipeline.stop_background_loop()
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FlushResult:
    """flush() の結果サマリー。"""

    total_events: int = 0
    teachers_updated: int = 0
    docs_without_teacher: int = 0
    docs_not_found: int = 0
    trust_updates: dict[str, float] = field(default_factory=dict)
    composite_updated: int = 0

    def summary(self) -> str:
        return (
            f"FlushResult(events={self.total_events} "
            f"teachers={self.teachers_updated} "
            f"no_teacher={self.docs_without_teacher} "
            f"not_found={self.docs_not_found} "
            f"composite_updated={self.composite_updated})"
        )


class TeacherFeedbackPipeline:
    """FeedbackCollector → TeacherRegistry 更新パイプライン。

    Args:
        collector:  FeedbackCollector インスタンス。
        store:      MetadataStore インスタンス（doc_id → teacher_id 解決用）。
        registry:   TeacherRegistry インスタンス（trust_score 更新先）。
        scorer:     CompositeScorer インスタンス（省略可）。
            指定した場合、flush() 後に trust_map を反映した composite_score 更新を行う。
        update_all_docs: True の場合 flush() 後に全ドキュメントの composite_score を更新する。
            False の場合は teacher_id が更新されたドキュメントのみを対象とする。
            デフォルト False（効率優先）。
    """

    def __init__(
        self,
        collector,
        store,
        registry,
        scorer=None,
        update_all_docs: bool = False,
    ) -> None:
        self._collector = collector
        self._store = store
        self._registry = registry
        self._scorer = scorer
        self._update_all_docs = update_all_docs
        self._loop_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # メイン処理
    # ------------------------------------------------------------------

    async def flush(self) -> FlushResult:
        """バッファのフィードバックを TeacherRegistry に転送する。

        Returns:
            FlushResult — 処理件数の概要。
        """
        result = FlushResult()

        # 1. イベントを取り出す
        events = self._collector.drain()
        result.total_events = len(events)
        if not events:
            return result

        # 2. doc_id → Document を一括解決
        doc_ids = list({e.doc_id for e in events})
        docs = await self._store.get_batch(doc_ids)
        doc_map: dict[str, object] = {d.id: d for d in docs}

        not_found = set(doc_ids) - set(doc_map.keys())
        result.docs_not_found = len(not_found)

        # 3. イベントを teacher_id ごとにまとめる
        teacher_rewards: dict[str, list[float]] = {}
        for ev in events:
            doc = doc_map.get(ev.doc_id)
            if doc is None:
                continue
            teacher_id = doc.source.teacher_id if doc.source else None
            if not teacher_id:
                result.docs_without_teacher += 1
                continue
            teacher_rewards.setdefault(teacher_id, []).append(ev.reward)

        # 4. TeacherRegistry を更新
        for teacher_id, rewards in teacher_rewards.items():
            profile = None
            for reward in rewards:
                try:
                    profile = await self._registry.record_feedback(teacher_id, reward)
                except Exception:
                    logger.exception(
                        "TeacherFeedbackPipeline: record_feedback failed for %s", teacher_id
                    )
            if profile is not None:
                result.trust_updates[teacher_id] = profile.trust_score

        result.teachers_updated = len(result.trust_updates)

        # 5. CompositeScorer で composite_score を再計算（オプション）
        if self._scorer is not None and result.teachers_updated > 0:
            try:
                profiles = await self._registry.list_all()
                trust_map = self._scorer.build_trust_map(profiles)

                if self._update_all_docs:
                    result.composite_updated = await self._scorer.update_store(
                        self._store, trust_map=trust_map
                    )
                else:
                    # trust が変わった Teacher のドキュメントのみ更新
                    affected_doc_ids = [
                        d.id
                        for d in doc_map.values()
                        if d.source.teacher_id in result.trust_updates
                    ]
                    if affected_doc_ids:
                        result.composite_updated = await self._scorer.update_store(
                            self._store,
                            doc_ids=affected_doc_ids,
                            trust_map=trust_map,
                        )
            except Exception:
                logger.exception("TeacherFeedbackPipeline: composite_score update failed")

        logger.info("TeacherFeedbackPipeline.flush: %s", result.summary())
        return result

    # ------------------------------------------------------------------
    # バックグラウンドループ
    # ------------------------------------------------------------------

    async def start_background_loop(self, interval_seconds: float = 60.0) -> None:
        """指定間隔で flush() を繰り返すバックグラウンドタスクを起動する。

        Args:
            interval_seconds: flush 間隔（秒）。デフォルト 60 秒。
        """
        if self._loop_task is not None and not self._loop_task.done():
            logger.warning("TeacherFeedbackPipeline: background loop already running")
            return
        self._loop_task = asyncio.create_task(
            self._background_loop(interval_seconds), name="teacher_feedback_loop"
        )
        logger.info(
            "TeacherFeedbackPipeline: background loop started (interval=%.1fs)", interval_seconds
        )

    async def stop_background_loop(self) -> None:
        """バックグラウンドタスクを停止する。"""
        if self._loop_task is not None and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        self._loop_task = None
        logger.info("TeacherFeedbackPipeline: background loop stopped")

    @property
    def is_running(self) -> bool:
        """バックグラウンドループが動作中かどうか。"""
        return self._loop_task is not None and not self._loop_task.done()

    async def _background_loop(self, interval_seconds: float) -> None:
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                await self.flush()
            except Exception:
                logger.exception("TeacherFeedbackPipeline: unhandled error in background loop")
