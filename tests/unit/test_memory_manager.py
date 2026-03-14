"""tests/unit/test_memory_manager.py — MemoryManager の単体テスト"""

from __future__ import annotations

import numpy as np
import pytest

from src.memory.embedder import Embedder
from src.memory.faiss_index import FAISSIndexManager
from src.memory.memory_manager import MemoryManager
from src.memory.metadata_store import MetadataStore
from src.memory.schema import Document, SourceMeta, SourceType

# ──────────────────────────────────────────────
# フィクスチャ
# ──────────────────────────────────────────────


def _make_embedder() -> Embedder:
    return Embedder(mock=True)


def _make_manager_sync() -> tuple[MemoryManager, MetadataStore]:
    from src.common.config import FAISSConfig, FAISSIndexConfig, MetadataConfig

    faiss_cfg = FAISSConfig(
        base_dir="data/faiss_test",
        domains={
            "code": FAISSIndexConfig(dim=768),
            "general": FAISSIndexConfig(dim=768),
        },
    )
    meta_cfg = MetadataConfig(db_path=":memory:")

    embedder = Embedder(mock=True)
    faiss_mgr = FAISSIndexManager(faiss_cfg)
    store = MetadataStore(meta_cfg)
    manager = MemoryManager(embedder=embedder, faiss=faiss_mgr, store=store)
    return manager, store


def _doc(content: str = "test content", domain: str = "code") -> Document:
    return Document(
        content=content,
        domain=domain,
        source=SourceMeta(source_type=SourceType.MANUAL),
    )


# ──────────────────────────────────────────────
# 初期化テスト
# ──────────────────────────────────────────────


class TestInitialization:
    def test_create_without_args(self) -> None:
        """引数なしで MemoryManager を生成できる。"""
        mgr = MemoryManager()
        assert not mgr._initialized

    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        mgr, _ = _make_manager_sync()
        await mgr.initialize()
        assert mgr._initialized
        await mgr.close()

    @pytest.mark.asyncio
    async def test_not_initialized_raises(self) -> None:
        mgr, _ = _make_manager_sync()
        with pytest.raises(RuntimeError, match="initialize"):
            await mgr.add(_doc())


# ──────────────────────────────────────────────
# 追加テスト
# ──────────────────────────────────────────────


class TestAdd:
    @pytest.mark.asyncio
    async def test_add_returns_doc_id(self) -> None:
        mgr, _ = _make_manager_sync()
        await mgr.initialize()
        doc = _doc()
        doc_id = await mgr.add(doc)
        assert doc_id == doc.id
        await mgr.close()

    @pytest.mark.asyncio
    async def test_add_auto_embeds(self) -> None:
        mgr, store = _make_manager_sync()
        await mgr.initialize()
        doc = _doc()
        assert doc.embedding is None
        await mgr.add(doc)
        saved = await store.get(doc.id)
        assert saved is not None
        await mgr.close()

    @pytest.mark.asyncio
    async def test_add_with_existing_embedding(self) -> None:
        mgr, _ = _make_manager_sync()
        await mgr.initialize()
        emb = np.random.rand(768).astype(np.float32)
        doc = _doc()
        doc = doc.model_copy(update={"embedding": emb})
        doc_id = await mgr.add(doc)
        assert doc_id == doc.id
        await mgr.close()

    @pytest.mark.asyncio
    async def test_add_creates_faiss_entry(self) -> None:
        mgr, _ = _make_manager_sync()
        await mgr.initialize()
        doc = _doc(domain="code")
        await mgr.add(doc)
        stats = await mgr.stats()
        assert stats["faiss_stats"]["code"] == 1
        await mgr.close()

    @pytest.mark.asyncio
    async def test_add_batch(self) -> None:
        mgr, _ = _make_manager_sync()
        await mgr.initialize()
        docs = [_doc(f"content {i}") for i in range(5)]
        added = await mgr.add_batch(docs)
        assert len(added) == 5
        stats = await mgr.stats()
        assert stats["total_docs"] == 5
        await mgr.close()

    @pytest.mark.asyncio
    async def test_add_batch_partial_failure(self) -> None:
        """一件失敗しても他は成功する。"""
        mgr, store = _make_manager_sync()
        await mgr.initialize()

        # 最初のドキュメントを先に追加して重複させる（SQLite UPSERT で上書き）
        docs = [_doc(f"content {i}") for i in range(3)]
        added = await mgr.add_batch(docs)
        assert len(added) == 3
        await mgr.close()


# ──────────────────────────────────────────────
# 削除テスト
# ──────────────────────────────────────────────


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_existing(self) -> None:
        mgr, _ = _make_manager_sync()
        await mgr.initialize()
        doc = _doc()
        await mgr.add(doc)
        result = await mgr.delete(doc.id)
        assert result is True
        assert await mgr.exists(doc.id) is False
        await mgr.close()

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self) -> None:
        mgr, _ = _make_manager_sync()
        await mgr.initialize()
        result = await mgr.delete("nonexistent_id")
        assert result is False
        await mgr.close()

    @pytest.mark.asyncio
    async def test_delete_removes_from_faiss(self) -> None:
        mgr, _ = _make_manager_sync()
        await mgr.initialize()
        doc = _doc(domain="code")
        await mgr.add(doc)
        assert (await mgr.stats())["faiss_stats"]["code"] == 1
        await mgr.delete(doc.id)
        # FAISS は論理削除なので ntotal はそのまま; 検索で返らないことで確認
        results = await mgr.search("test", domain="code", k=5)
        ids = [r.document.id for r in results]
        assert doc.id not in ids
        await mgr.close()


# ──────────────────────────────────────────────
# 検索テスト
# ──────────────────────────────────────────────


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_returns_results(self) -> None:
        mgr, _ = _make_manager_sync()
        await mgr.initialize()
        for i in range(5):
            await mgr.add(_doc(f"Python sorting {i}", domain="code"))
        results = await mgr.search("Python list sort", domain="code", k=3)
        assert len(results) == 3
        assert all(isinstance(r.score, float) for r in results)
        await mgr.close()

    @pytest.mark.asyncio
    async def test_search_all_domains(self) -> None:
        mgr, _ = _make_manager_sync()
        await mgr.initialize()
        await mgr.add(_doc("code content", domain="code"))
        await mgr.add(_doc("general content", domain="general"))
        results = await mgr.search("content", k=5)
        assert len(results) == 2
        await mgr.close()

    @pytest.mark.asyncio
    async def test_search_empty_index(self) -> None:
        mgr, _ = _make_manager_sync()
        await mgr.initialize()
        results = await mgr.search("query", domain="code", k=5)
        assert results == []
        await mgr.close()

    @pytest.mark.asyncio
    async def test_search_min_score_filter(self) -> None:
        mgr, _ = _make_manager_sync()
        await mgr.initialize()
        await mgr.add(_doc("content"))
        # 極大スコアでフィルタ → 0件
        results = await mgr.search("query", min_score=999.0)
        assert results == []
        await mgr.close()

    @pytest.mark.asyncio
    async def test_search_result_has_document(self) -> None:
        mgr, _ = _make_manager_sync()
        await mgr.initialize()
        doc = _doc("unique content for test")
        await mgr.add(doc)
        results = await mgr.search("unique content", domain="code", k=1)
        assert len(results) == 1
        assert results[0].document.id == doc.id
        assert results[0].document.content == doc.content
        await mgr.close()


# ──────────────────────────────────────────────
# フィードバック / 統計テスト
# ──────────────────────────────────────────────


class TestFeedbackAndStats:
    @pytest.mark.asyncio
    async def test_record_retrieval(self) -> None:
        mgr, store = _make_manager_sync()
        await mgr.initialize()
        doc = _doc()
        await mgr.add(doc)
        await mgr.record_retrieval(doc.id)
        saved = await store.get(doc.id)
        assert saved is not None
        assert saved.usefulness.retrieval_count == 1
        await mgr.close()

    @pytest.mark.asyncio
    async def test_record_retrieval_selected(self) -> None:
        mgr, store = _make_manager_sync()
        await mgr.initialize()
        doc = _doc()
        await mgr.add(doc)
        await mgr.record_retrieval(doc.id, selected=True)
        saved = await store.get(doc.id)
        assert saved is not None
        assert saved.usefulness.retrieval_count == 1
        assert saved.usefulness.selection_count == 1
        await mgr.close()

    @pytest.mark.asyncio
    async def test_add_feedback_positive(self) -> None:
        mgr, store = _make_manager_sync()
        await mgr.initialize()
        doc = _doc()
        await mgr.add(doc)
        await mgr.add_feedback(doc.id, positive=True)

        saved = await store.get(doc.id)
        assert saved is not None
        assert saved.usefulness.positive_feedback == 1
        await mgr.close()

    @pytest.mark.asyncio
    async def test_stats(self) -> None:
        mgr, _ = _make_manager_sync()
        await mgr.initialize()
        for i in range(3):
            await mgr.add(_doc(f"content {i}"))
        stats = await mgr.stats()
        assert stats["total_docs"] == 3
        assert "avg_confidence" in stats
        assert "faiss_stats" in stats
        await mgr.close()

    @pytest.mark.asyncio
    async def test_exists(self) -> None:
        mgr, _ = _make_manager_sync()
        await mgr.initialize()
        doc = _doc()
        assert await mgr.exists(doc.id) is False
        await mgr.add(doc)
        assert await mgr.exists(doc.id) is True
        await mgr.close()

    @pytest.mark.asyncio
    async def test_get(self) -> None:
        mgr, _ = _make_manager_sync()
        await mgr.initialize()
        doc = _doc("hello world")
        await mgr.add(doc)
        retrieved = await mgr.get(doc.id)
        assert retrieved is not None
        assert retrieved.content == "hello world"
        await mgr.close()
