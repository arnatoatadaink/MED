"""tests/unit/test_teacher_provenance_step5.py — TeacherFeedbackPipeline テスト (Step 5)

テスト対象:
- src/memory/learning/teacher_feedback_pipeline.py
  - flush(): FeedbackCollector → TeacherRegistry 更新パイプライン
  - flush() + CompositeScorer 統合
  - バックグラウンドループ
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from src.memory.schema import Document, SourceMeta, SourceType

# ===========================================================================
# Helpers / Stubs
# ===========================================================================

def _make_doc(content: str, teacher_id: str | None = None, domain: str = "general") -> Document:
    source = SourceMeta(source_type=SourceType.TEACHER if teacher_id else SourceType.MANUAL)
    if teacher_id:
        source.set_teacher(teacher_id)
    return Document(content=content, domain=domain, source=source)


async def _make_store(tmp_path: Path, docs: list[Document] | None = None):
    from src.memory.metadata_store import MetadataStore
    store = MetadataStore(db_path=str(tmp_path / "store.db"))
    await store.initialize()
    for doc in (docs or []):
        await store.save(doc)
    return store


async def _make_registry(tmp_path: Path):
    from src.memory.teacher_registry import TeacherRegistry
    reg = TeacherRegistry(tmp_path / "reg.db")
    await reg.initialize()
    return reg


def _make_collector():
    from src.memory.learning.feedback_collector import FeedbackCollector
    return FeedbackCollector(max_buffer=1000)


# ===========================================================================
# FlushResult
# ===========================================================================

class TestFlushResult:
    def test_summary_contains_counts(self):
        from src.memory.learning.teacher_feedback_pipeline import FlushResult
        r = FlushResult(total_events=10, teachers_updated=2, docs_without_teacher=1)
        s = r.summary()
        assert "events=10" in s
        assert "teachers=2" in s


# ===========================================================================
# flush() — 基本動作
# ===========================================================================

class TestFlushBasic:
    def test_flush_empty_collector(self, tmp_path: Path):
        """イベントが無い場合は何もせず FlushResult を返す。"""
        async def run():
            store = await _make_store(tmp_path)
            registry = await _make_registry(tmp_path)
            collector = _make_collector()
            from src.memory.learning.teacher_feedback_pipeline import TeacherFeedbackPipeline
            pipeline = TeacherFeedbackPipeline(collector, store, registry)
            result = await pipeline.flush()
            await store.close()
            return result

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.total_events == 0
        assert result.teachers_updated == 0

    def test_flush_updates_teacher_registry(self, tmp_path: Path):
        """フィードバックが TeacherRegistry の trust_score を更新すること。"""
        async def run():
            doc = _make_doc("content", teacher_id="claude-opus-4-6")
            store = await _make_store(tmp_path, [doc])
            registry = await _make_registry(tmp_path)
            collector = _make_collector()

            # 低い reward を大量投入して trust を下げる
            for _ in range(15):
                collector.record_explicit(doc.id, thumbs_up=False)

            from src.memory.learning.teacher_feedback_pipeline import TeacherFeedbackPipeline
            pipeline = TeacherFeedbackPipeline(collector, store, registry)
            result = await pipeline.flush()

            profile = await registry.get("claude-opus-4-6")
            await store.close()
            return result, profile

        result, profile = asyncio.get_event_loop().run_until_complete(run())
        assert result.total_events == 15
        assert result.teachers_updated == 1
        assert "claude-opus-4-6" in result.trust_updates
        assert profile is not None
        assert profile.trust_score < 1.0

    def test_flush_high_reward_maintains_trust(self, tmp_path: Path):
        """高い reward が続くと trust_score が高く維持される。"""
        async def run():
            doc = _make_doc("good content", teacher_id="good-model")
            store = await _make_store(tmp_path, [doc])
            registry = await _make_registry(tmp_path)
            collector = _make_collector()

            for _ in range(10):
                collector.record_explicit(doc.id, thumbs_up=True)

            from src.memory.learning.teacher_feedback_pipeline import TeacherFeedbackPipeline
            pipeline = TeacherFeedbackPipeline(collector, store, registry)
            await pipeline.flush()

            profile = await registry.get("good-model")
            await store.close()
            return profile

        profile = asyncio.get_event_loop().run_until_complete(run())
        assert profile.trust_score >= 0.7

    def test_flush_multiple_teachers(self, tmp_path: Path):
        """複数 Teacher のフィードバックを同時に処理できること。"""
        async def run():
            doc_a = _make_doc("doc a", teacher_id="model-a")
            doc_b = _make_doc("doc b", teacher_id="model-b")
            store = await _make_store(tmp_path, [doc_a, doc_b])
            registry = await _make_registry(tmp_path)
            collector = _make_collector()

            collector.record_explicit(doc_a.id, thumbs_up=True)
            collector.record_explicit(doc_b.id, thumbs_up=False)

            from src.memory.learning.teacher_feedback_pipeline import TeacherFeedbackPipeline
            pipeline = TeacherFeedbackPipeline(collector, store, registry)
            result = await pipeline.flush()
            await store.close()
            return result

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.teachers_updated == 2
        assert "model-a" in result.trust_updates
        assert "model-b" in result.trust_updates

    def test_flush_doc_without_teacher_counted(self, tmp_path: Path):
        """teacher_id なしドキュメントは docs_without_teacher に計上される。"""
        async def run():
            doc = _make_doc("no teacher")
            store = await _make_store(tmp_path, [doc])
            registry = await _make_registry(tmp_path)
            collector = _make_collector()
            collector.record_click(doc.id)

            from src.memory.learning.teacher_feedback_pipeline import TeacherFeedbackPipeline
            pipeline = TeacherFeedbackPipeline(collector, store, registry)
            result = await pipeline.flush()
            await store.close()
            return result

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.docs_without_teacher == 1
        assert result.teachers_updated == 0

    def test_flush_doc_not_found_counted(self, tmp_path: Path):
        """store に存在しない doc_id は docs_not_found に計上される。"""
        async def run():
            store = await _make_store(tmp_path)
            registry = await _make_registry(tmp_path)
            collector = _make_collector()
            collector.record_click("nonexistent-doc-id")

            from src.memory.learning.teacher_feedback_pipeline import TeacherFeedbackPipeline
            pipeline = TeacherFeedbackPipeline(collector, store, registry)
            result = await pipeline.flush()
            await store.close()
            return result

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.docs_not_found == 1
        assert result.teachers_updated == 0

    def test_flush_drains_collector(self, tmp_path: Path):
        """flush() 後にコレクターバッファが空になること。"""
        async def run():
            doc = _make_doc("content", teacher_id="model-x")
            store = await _make_store(tmp_path, [doc])
            registry = await _make_registry(tmp_path)
            collector = _make_collector()
            collector.record_click(doc.id)
            assert len(collector) == 1

            from src.memory.learning.teacher_feedback_pipeline import TeacherFeedbackPipeline
            pipeline = TeacherFeedbackPipeline(collector, store, registry)
            await pipeline.flush()
            await store.close()
            return len(collector)

        assert asyncio.get_event_loop().run_until_complete(run()) == 0

    def test_flush_multiple_events_same_teacher(self, tmp_path: Path):
        """同一 Teacher への複数イベントがすべて registry に転送される。"""
        async def run():
            doc = _make_doc("content", teacher_id="model-y")
            store = await _make_store(tmp_path, [doc])
            registry = await _make_registry(tmp_path)
            collector = _make_collector()

            for _ in range(5):
                collector.record_click(doc.id, rank=0)

            from src.memory.learning.teacher_feedback_pipeline import TeacherFeedbackPipeline
            pipeline = TeacherFeedbackPipeline(collector, store, registry)
            result = await pipeline.flush()

            profile = await registry.get("model-y")
            await store.close()
            return result, profile

        result, profile = asyncio.get_event_loop().run_until_complete(run())
        assert result.total_events == 5
        assert profile.n_feedback == 5


# ===========================================================================
# flush() + CompositeScorer 統合
# ===========================================================================

class TestFlushWithScorer:
    def test_flush_updates_composite_score(self, tmp_path: Path):
        """flush() 後に composite_score が低信頼 Teacher のドキュメントで下がること。"""
        from src.memory.scoring.composite_scorer import CompositeScorer

        async def run():
            doc = _make_doc("content", teacher_id="bad-model")
            store = await _make_store(tmp_path, [doc])
            registry = await _make_registry(tmp_path)
            scorer = CompositeScorer()
            collector = _make_collector()

            # 初期 composite_score を計算して保存
            await scorer.update_store(store)
            initial = (await store.get(doc.id)).usefulness.composite

            # 低い reward を大量投入
            for _ in range(20):
                collector.record_explicit(doc.id, thumbs_up=False)

            from src.memory.learning.teacher_feedback_pipeline import TeacherFeedbackPipeline
            pipeline = TeacherFeedbackPipeline(collector, store, registry, scorer=scorer)
            result = await pipeline.flush()

            updated = (await store.get(doc.id)).usefulness.composite
            await store.close()
            return initial, updated, result

        initial, updated, result = asyncio.get_event_loop().run_until_complete(run())
        assert result.composite_updated > 0
        assert updated < initial  # 低信頼 Teacher のスコアが下がった

    def test_flush_with_scorer_update_all_docs(self, tmp_path: Path):
        """update_all_docs=True のとき全ドキュメントを更新すること。"""
        from src.memory.scoring.composite_scorer import CompositeScorer

        async def run():
            doc_a = _make_doc("doc a", teacher_id="model-a")
            doc_b = _make_doc("doc b", teacher_id="model-b")
            store = await _make_store(tmp_path, [doc_a, doc_b])
            registry = await _make_registry(tmp_path)
            scorer = CompositeScorer()
            collector = _make_collector()

            collector.record_explicit(doc_a.id, thumbs_up=False)

            from src.memory.learning.teacher_feedback_pipeline import TeacherFeedbackPipeline
            pipeline = TeacherFeedbackPipeline(
                collector, store, registry, scorer=scorer, update_all_docs=True
            )
            result = await pipeline.flush()
            await store.close()
            return result

        result = asyncio.get_event_loop().run_until_complete(run())
        # update_all_docs=True → 2件（doc_a + doc_b）が更新される
        assert result.composite_updated == 2

    def test_flush_no_scorer_does_not_update_composite(self, tmp_path: Path):
        """scorer=None のとき composite_score は更新されない。"""
        async def run():
            doc = _make_doc("content", teacher_id="model-x")
            store = await _make_store(tmp_path, [doc])
            registry = await _make_registry(tmp_path)
            collector = _make_collector()
            collector.record_explicit(doc.id, thumbs_up=False)

            from src.memory.learning.teacher_feedback_pipeline import TeacherFeedbackPipeline
            pipeline = TeacherFeedbackPipeline(collector, store, registry, scorer=None)
            result = await pipeline.flush()
            await store.close()
            return result

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.composite_updated == 0


# ===========================================================================
# バックグラウンドループ
# ===========================================================================

class TestBackgroundLoop:
    def test_is_running_false_initially(self, tmp_path: Path):
        async def run():
            store = await _make_store(tmp_path)
            registry = await _make_registry(tmp_path)
            collector = _make_collector()
            from src.memory.learning.teacher_feedback_pipeline import TeacherFeedbackPipeline
            pipeline = TeacherFeedbackPipeline(collector, store, registry)
            result = pipeline.is_running
            await store.close()
            return result

        assert asyncio.get_event_loop().run_until_complete(run()) is False

    def test_start_and_stop_loop(self, tmp_path: Path):
        async def run():
            store = await _make_store(tmp_path)
            registry = await _make_registry(tmp_path)
            collector = _make_collector()
            from src.memory.learning.teacher_feedback_pipeline import TeacherFeedbackPipeline
            pipeline = TeacherFeedbackPipeline(collector, store, registry)

            await pipeline.start_background_loop(interval_seconds=100)
            running = pipeline.is_running
            await pipeline.stop_background_loop()
            stopped = pipeline.is_running
            await store.close()
            return running, stopped

        running, stopped = asyncio.get_event_loop().run_until_complete(run())
        assert running is True
        assert stopped is False

    def test_start_twice_is_idempotent(self, tmp_path: Path):
        """二重起動してもエラーにならない。"""
        async def run():
            store = await _make_store(tmp_path)
            registry = await _make_registry(tmp_path)
            collector = _make_collector()
            from src.memory.learning.teacher_feedback_pipeline import TeacherFeedbackPipeline
            pipeline = TeacherFeedbackPipeline(collector, store, registry)

            await pipeline.start_background_loop(interval_seconds=100)
            await pipeline.start_background_loop(interval_seconds=100)  # 2回目
            await pipeline.stop_background_loop()
            await store.close()

        asyncio.get_event_loop().run_until_complete(run())  # no error

    def test_loop_executes_flush(self, tmp_path: Path):
        """バックグラウンドループが実際に flush() を呼ぶこと。"""
        async def run():
            doc = _make_doc("content", teacher_id="model-loop")
            store = await _make_store(tmp_path, [doc])
            registry = await _make_registry(tmp_path)
            collector = _make_collector()
            collector.record_explicit(doc.id, thumbs_up=True)

            from src.memory.learning.teacher_feedback_pipeline import TeacherFeedbackPipeline
            pipeline = TeacherFeedbackPipeline(collector, store, registry)

            # 短い interval で1回分だけ動かす
            await pipeline.start_background_loop(interval_seconds=0.05)
            await asyncio.sleep(0.15)  # ループが最低1回回るのを待つ
            await pipeline.stop_background_loop()

            profile = await registry.get("model-loop")
            await store.close()
            return profile

        profile = asyncio.get_event_loop().run_until_complete(run())
        # ループが flush を呼び registry が更新されているはず
        assert profile is not None
        assert profile.n_feedback >= 1
