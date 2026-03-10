"""tests/unit/test_metadata_store.py — src/memory/metadata_store.py の単体テスト

全テストはインメモリ SQLite (:memory:) で実行。
"""

from __future__ import annotations

import pytest

from src.common.config import get_settings
from src.memory.metadata_store import MetadataStore
from src.memory.schema import (
    DifficultyLevel,
    Document,
    Domain,
    ReviewStatus,
    SourceMeta,
    SourceType,
    UsefulnessScore,
)


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    yield
    get_settings.cache_clear()


@pytest.fixture()
async def store():
    """インメモリ MetadataStore。"""
    s = MetadataStore(db_path=":memory:")
    await s.initialize()
    yield s
    await s.close()


def _make_doc(
    doc_id: str = "test_doc",
    content: str = "test content",
    domain: Domain = Domain.CODE,
    **kwargs,
) -> Document:
    return Document(id=doc_id, content=content, domain=domain, **kwargs)


# ============================================================================
# 基本 CRUD
# ============================================================================


class TestCRUD:
    async def test_save_and_get(self, store: MetadataStore):
        doc = _make_doc()
        await store.save(doc)
        result = await store.get("test_doc")
        assert result is not None
        assert result.id == "test_doc"
        assert result.content == "test content"
        assert result.domain == Domain.CODE

    async def test_get_nonexistent_returns_none(self, store: MetadataStore):
        result = await store.get("nonexistent")
        assert result is None

    async def test_save_upsert(self, store: MetadataStore):
        doc = _make_doc(content="version 1")
        await store.save(doc)
        doc2 = _make_doc(content="version 2")
        await store.save(doc2)

        result = await store.get("test_doc")
        assert result.content == "version 2"

    async def test_delete(self, store: MetadataStore):
        doc = _make_doc()
        await store.save(doc)
        deleted = await store.delete("test_doc")
        assert deleted is True
        assert await store.get("test_doc") is None

    async def test_delete_nonexistent(self, store: MetadataStore):
        deleted = await store.delete("nonexistent")
        assert deleted is False

    async def test_exists(self, store: MetadataStore):
        doc = _make_doc()
        await store.save(doc)
        assert await store.exists("test_doc") is True
        assert await store.exists("nonexistent") is False


# ============================================================================
# バッチ操作
# ============================================================================


class TestBatch:
    async def test_save_batch(self, store: MetadataStore):
        docs = [_make_doc(doc_id=f"doc_{i}", content=f"content {i}") for i in range(5)]
        await store.save_batch(docs)
        assert await store.count() == 5

    async def test_save_batch_empty(self, store: MetadataStore):
        await store.save_batch([])
        assert await store.count() == 0

    async def test_get_batch(self, store: MetadataStore):
        docs = [_make_doc(doc_id=f"doc_{i}") for i in range(3)]
        await store.save_batch(docs)
        results = await store.get_batch(["doc_0", "doc_2"])
        assert len(results) == 2
        ids = {r.id for r in results}
        assert ids == {"doc_0", "doc_2"}

    async def test_get_batch_empty(self, store: MetadataStore):
        results = await store.get_batch([])
        assert results == []

    async def test_delete_batch(self, store: MetadataStore):
        docs = [_make_doc(doc_id=f"doc_{i}") for i in range(5)]
        await store.save_batch(docs)
        deleted = await store.delete_batch(["doc_1", "doc_3"])
        assert deleted == 2
        assert await store.count() == 3


# ============================================================================
# メタデータのシリアライズ往復
# ============================================================================


class TestSerialization:
    async def test_source_meta_roundtrip(self, store: MetadataStore):
        doc = _make_doc(
            source=SourceMeta(
                source_type=SourceType.GITHUB,
                url="https://github.com/test",
                title="Test Repo",
                author="testuser",
                language="python",
                tags=["fastapi", "async"],
                extra={"stars": 100},
            )
        )
        await store.save(doc)
        result = await store.get("test_doc")
        assert result.source.source_type == SourceType.GITHUB
        assert result.source.url == "https://github.com/test"
        assert result.source.language == "python"
        assert "fastapi" in result.source.tags
        assert result.source.extra["stars"] == 100

    async def test_usefulness_roundtrip(self, store: MetadataStore):
        doc = _make_doc(
            usefulness=UsefulnessScore(
                retrieval_count=10,
                selection_count=5,
                positive_feedback=3,
                negative_feedback=1,
                teacher_quality=0.8,
                execution_success_rate=0.9,
                freshness=0.7,
                composite=0.75,
            )
        )
        await store.save(doc)
        result = await store.get("test_doc")
        u = result.usefulness
        assert u.retrieval_count == 10
        assert u.selection_count == 5
        assert u.positive_feedback == 3
        assert u.teacher_quality == pytest.approx(0.8)
        assert u.composite == pytest.approx(0.75)

    async def test_difficulty_roundtrip(self, store: MetadataStore):
        doc = _make_doc(difficulty=DifficultyLevel.ADVANCED)
        await store.save(doc)
        result = await store.get("test_doc")
        assert result.difficulty == DifficultyLevel.ADVANCED

    async def test_difficulty_none_roundtrip(self, store: MetadataStore):
        doc = _make_doc(difficulty=None)
        await store.save(doc)
        result = await store.get("test_doc")
        assert result.difficulty is None

    async def test_execution_fields_roundtrip(self, store: MetadataStore):
        doc = _make_doc(
            is_executable=True,
            execution_verified=True,
            last_execution_success=True,
        )
        await store.save(doc)
        result = await store.get("test_doc")
        assert result.is_executable is True
        assert result.execution_verified is True
        assert result.last_execution_success is True

    async def test_embedding_not_stored(self, store: MetadataStore):
        """MetadataStore は embedding を保持しないこと。"""
        doc = _make_doc()
        await store.save(doc)
        result = await store.get("test_doc")
        assert result.embedding is None


# ============================================================================
# クエリ系
# ============================================================================


class TestQueries:
    async def test_list_by_domain(self, store: MetadataStore):
        docs = [
            _make_doc(doc_id="c1", domain=Domain.CODE),
            _make_doc(doc_id="c2", domain=Domain.CODE),
            _make_doc(doc_id="g1", domain=Domain.GENERAL),
        ]
        await store.save_batch(docs)
        code_docs = await store.list_by_domain("code")
        assert len(code_docs) == 2
        assert all(d.domain == Domain.CODE for d in code_docs)

    async def test_list_by_domain_with_limit(self, store: MetadataStore):
        docs = [_make_doc(doc_id=f"d{i}", domain=Domain.CODE) for i in range(10)]
        await store.save_batch(docs)
        results = await store.list_by_domain("code", limit=3)
        assert len(results) == 3

    async def test_get_unreviewed(self, store: MetadataStore):
        docs = [
            _make_doc(doc_id="u1", review_status=ReviewStatus.UNREVIEWED),
            _make_doc(doc_id="u2", review_status=ReviewStatus.UNREVIEWED),
            _make_doc(doc_id="r1", review_status=ReviewStatus.APPROVED),
        ]
        await store.save_batch(docs)
        unreviewed = await store.get_unreviewed()
        assert len(unreviewed) == 2

    async def test_get_unreviewed_by_domain(self, store: MetadataStore):
        docs = [
            _make_doc(doc_id="c1", domain=Domain.CODE, review_status=ReviewStatus.UNREVIEWED),
            _make_doc(doc_id="g1", domain=Domain.GENERAL, review_status=ReviewStatus.UNREVIEWED),
        ]
        await store.save_batch(docs)
        code_unreviewed = await store.get_unreviewed(domain="code")
        assert len(code_unreviewed) == 1
        assert code_unreviewed[0].domain == Domain.CODE

    async def test_get_low_confidence(self, store: MetadataStore):
        docs = [
            _make_doc(doc_id="low", confidence=0.2),
            _make_doc(doc_id="mid", confidence=0.5),
            _make_doc(doc_id="high", confidence=0.9),
        ]
        await store.save_batch(docs)
        low = await store.get_low_confidence(threshold=0.3)
        assert len(low) == 1
        assert low[0].id == "low"


# ============================================================================
# 有用性スコア更新
# ============================================================================


class TestUsefulnessUpdates:
    async def test_increment_retrieval(self, store: MetadataStore):
        doc = _make_doc()
        await store.save(doc)
        await store.increment_retrieval("test_doc")
        await store.increment_retrieval("test_doc")
        result = await store.get("test_doc")
        assert result.usefulness.retrieval_count == 2

    async def test_increment_selection(self, store: MetadataStore):
        doc = _make_doc()
        await store.save(doc)
        await store.increment_selection("test_doc")
        result = await store.get("test_doc")
        assert result.usefulness.selection_count == 1

    async def test_add_positive_feedback(self, store: MetadataStore):
        doc = _make_doc()
        await store.save(doc)
        await store.add_feedback("test_doc", positive=True)
        result = await store.get("test_doc")
        assert result.usefulness.positive_feedback == 1

    async def test_add_negative_feedback(self, store: MetadataStore):
        doc = _make_doc()
        await store.save(doc)
        await store.add_feedback("test_doc", positive=False)
        result = await store.get("test_doc")
        assert result.usefulness.negative_feedback == 1


# ============================================================================
# 品質更新
# ============================================================================


class TestQualityUpdates:
    async def test_update_quality(self, store: MetadataStore):
        doc = _make_doc()
        await store.save(doc)
        await store.update_quality(
            "test_doc",
            teacher_quality=0.9,
            difficulty="advanced",
            review_status="approved",
            confidence=0.85,
        )
        result = await store.get("test_doc")
        assert result.usefulness.teacher_quality == pytest.approx(0.9)
        assert result.difficulty == DifficultyLevel.ADVANCED
        assert result.review_status == ReviewStatus.APPROVED
        assert result.confidence == pytest.approx(0.85)
        assert result.reviewed_at is not None

    async def test_update_quality_partial(self, store: MetadataStore):
        doc = _make_doc(confidence=0.5)
        await store.save(doc)
        await store.update_quality("test_doc", confidence=0.9)
        result = await store.get("test_doc")
        assert result.confidence == pytest.approx(0.9)
        assert result.difficulty is None  # 変更されていない


# ============================================================================
# 統計
# ============================================================================


class TestStatistics:
    async def test_count_all(self, store: MetadataStore):
        docs = [_make_doc(doc_id=f"d{i}") for i in range(5)]
        await store.save_batch(docs)
        assert await store.count() == 5

    async def test_count_by_domain(self, store: MetadataStore):
        docs = [
            _make_doc(doc_id="c1", domain=Domain.CODE),
            _make_doc(doc_id="c2", domain=Domain.CODE),
            _make_doc(doc_id="g1", domain=Domain.GENERAL),
        ]
        await store.save_batch(docs)
        assert await store.count(domain="code") == 2
        assert await store.count(domain="general") == 1

    async def test_avg_confidence(self, store: MetadataStore):
        docs = [
            _make_doc(doc_id="d1", confidence=0.6),
            _make_doc(doc_id="d2", confidence=0.8),
        ]
        await store.save_batch(docs)
        avg = await store.avg_confidence()
        assert avg == pytest.approx(0.7)

    async def test_avg_confidence_empty(self, store: MetadataStore):
        avg = await store.avg_confidence()
        assert avg == 0.0
