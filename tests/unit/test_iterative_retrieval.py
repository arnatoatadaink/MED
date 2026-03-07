"""tests/unit/test_iterative_retrieval.py — IterativeRetriever の単体テスト"""

from __future__ import annotations

import pytest

from src.common.config import FAISSConfig, FAISSIndexConfig, MetadataConfig
from src.memory.embedder import Embedder
from src.memory.faiss_index import FAISSIndexManager
from src.memory.iterative_retrieval import IterativeRetriever
from src.memory.memory_manager import MemoryManager
from src.memory.metadata_store import MetadataStore
from src.memory.schema import Document, SourceMeta, SourceType


# ──────────────────────────────────────────────
# モック LLM
# ──────────────────────────────────────────────


class MockLLM:
    """テスト用の同期 LLM モック。"""

    def __init__(self, response: str = "rewritten query") -> None:
        self.response = response
        self.calls: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response


class FailingLLM:
    async def complete(self, prompt: str) -> str:
        raise RuntimeError("LLM failed")


# ──────────────────────────────────────────────
# フィクスチャ
# ──────────────────────────────────────────────


def _make_manager() -> MemoryManager:
    faiss_cfg = FAISSConfig(
        base_dir="data/faiss_test",
        domains={"code": FAISSIndexConfig(dim=768), "general": FAISSIndexConfig(dim=768)},
    )
    meta_cfg = MetadataConfig(db_path=":memory:")
    embedder = Embedder(mock=True)
    faiss_mgr = FAISSIndexManager(faiss_cfg)
    store = MetadataStore(meta_cfg)
    return MemoryManager(embedder=embedder, faiss=faiss_mgr, store=store)


def _doc(content: str, domain: str = "code") -> Document:
    return Document(
        content=content,
        domain=domain,
        source=SourceMeta(source_type=SourceType.MANUAL),
    )


# ──────────────────────────────────────────────
# 初期化テスト
# ──────────────────────────────────────────────


class TestInitialization:
    def test_create_without_llm(self) -> None:
        mm = _make_manager()
        retriever = IterativeRetriever(mm, mm.embedder)
        assert retriever._llm is None

    def test_create_with_llm(self) -> None:
        mm = _make_manager()
        llm = MockLLM()
        retriever = IterativeRetriever(mm, mm.embedder, llm=llm)
        assert retriever._llm is llm

    @pytest.mark.asyncio
    async def test_unknown_strategy_raises(self) -> None:
        mm = _make_manager()
        await mm.initialize()
        retriever = IterativeRetriever(mm, mm.embedder)
        with pytest.raises(ValueError, match="Unknown strategy"):
            await retriever.retrieve("query", strategy="invalid")
        await mm.close()


# ──────────────────────────────────────────────
# vector_add 戦略
# ──────────────────────────────────────────────


class TestVectorAdd:
    @pytest.mark.asyncio
    async def test_empty_index_returns_empty(self) -> None:
        mm = _make_manager()
        await mm.initialize()
        retriever = IterativeRetriever(mm, mm.embedder)
        results = await retriever.retrieve("query", strategy="vector_add")
        assert results == []
        await mm.close()

    @pytest.mark.asyncio
    async def test_returns_results(self) -> None:
        mm = _make_manager()
        await mm.initialize()
        for i in range(5):
            await mm.add(_doc(f"Python sorting algorithm {i}"))
        retriever = IterativeRetriever(mm, mm.embedder)
        results = await retriever.retrieve(
            "Python sort", domain="code", max_rounds=2, k_per_round=3, strategy="vector_add"
        )
        assert len(results) > 0
        await mm.close()

    @pytest.mark.asyncio
    async def test_dedup_removes_duplicates(self) -> None:
        mm = _make_manager()
        await mm.initialize()
        doc = _doc("unique content")
        await mm.add(doc)
        retriever = IterativeRetriever(mm, mm.embedder, dedup=True)
        results = await retriever.retrieve(
            "unique", domain="code", max_rounds=3, k_per_round=5, strategy="vector_add"
        )
        ids = [r.document.id for r in results]
        assert len(ids) == len(set(ids))
        await mm.close()

    @pytest.mark.asyncio
    async def test_no_dedup_may_have_duplicates(self) -> None:
        """dedup=False では重複が出る可能性があるが、エラーは出ない。"""
        mm = _make_manager()
        await mm.initialize()
        for i in range(3):
            await mm.add(_doc(f"content {i}"))
        retriever = IterativeRetriever(mm, mm.embedder, dedup=False)
        results = await retriever.retrieve(
            "content", domain="code", max_rounds=2, k_per_round=3, strategy="vector_add"
        )
        assert isinstance(results, list)
        await mm.close()

    @pytest.mark.asyncio
    async def test_multi_round_collects_more(self) -> None:
        mm = _make_manager()
        await mm.initialize()
        for i in range(10):
            await mm.add(_doc(f"document about topic {i}"))
        retriever = IterativeRetriever(mm, mm.embedder)
        results_1round = await retriever.retrieve(
            "topic", domain="code", max_rounds=1, k_per_round=3, strategy="vector_add"
        )
        results_3rounds = await retriever.retrieve(
            "topic", domain="code", max_rounds=3, k_per_round=3, strategy="vector_add"
        )
        assert len(results_3rounds) >= len(results_1round)
        await mm.close()


# ──────────────────────────────────────────────
# llm_rewrite 戦略
# ──────────────────────────────────────────────


class TestLLMRewrite:
    @pytest.mark.asyncio
    async def test_fallback_without_llm(self) -> None:
        mm = _make_manager()
        await mm.initialize()
        for i in range(3):
            await mm.add(_doc(f"content {i}"))
        retriever = IterativeRetriever(mm, mm.embedder, llm=None)
        results = await retriever.retrieve(
            "content", strategy="llm_rewrite", max_rounds=2, k_per_round=3
        )
        assert isinstance(results, list)
        await mm.close()

    @pytest.mark.asyncio
    async def test_llm_is_called(self) -> None:
        mm = _make_manager()
        await mm.initialize()
        for i in range(5):
            await mm.add(_doc(f"content {i}"))
        llm = MockLLM("improved query for content")
        retriever = IterativeRetriever(mm, mm.embedder, llm=llm)
        await retriever.retrieve(
            "content", domain="code", strategy="llm_rewrite", max_rounds=2, k_per_round=3
        )
        assert len(llm.calls) > 0
        await mm.close()

    @pytest.mark.asyncio
    async def test_failing_llm_returns_partial(self) -> None:
        mm = _make_manager()
        await mm.initialize()
        for i in range(3):
            await mm.add(_doc(f"content {i}"))
        retriever = IterativeRetriever(mm, mm.embedder, llm=FailingLLM())
        # エラーを出さずに部分結果を返す
        results = await retriever.retrieve(
            "content", domain="code", strategy="llm_rewrite", max_rounds=2, k_per_round=3
        )
        assert isinstance(results, list)
        await mm.close()


# ──────────────────────────────────────────────
# HyDE 戦略
# ──────────────────────────────────────────────


class TestHyDE:
    @pytest.mark.asyncio
    async def test_fallback_without_llm(self) -> None:
        mm = _make_manager()
        await mm.initialize()
        for i in range(3):
            await mm.add(_doc(f"content {i}"))
        retriever = IterativeRetriever(mm, mm.embedder, llm=None)
        results = await retriever.retrieve("content", strategy="hyde", k_per_round=3)
        assert isinstance(results, list)
        await mm.close()

    @pytest.mark.asyncio
    async def test_hyde_with_llm(self) -> None:
        mm = _make_manager()
        await mm.initialize()
        for i in range(5):
            await mm.add(_doc(f"Python list sort algorithm {i}"))
        llm = MockLLM("Python uses Timsort, a hybrid sorting algorithm...")
        retriever = IterativeRetriever(mm, mm.embedder, llm=llm)
        results = await retriever.retrieve(
            "How does Python sort lists?", domain="code", strategy="hyde", k_per_round=3
        )
        assert isinstance(results, list)
        assert len(llm.calls) == 1  # HyDE は 1 回のみ呼び出す
        await mm.close()

    @pytest.mark.asyncio
    async def test_failing_llm_fallback(self) -> None:
        mm = _make_manager()
        await mm.initialize()
        for i in range(3):
            await mm.add(_doc(f"content {i}"))
        retriever = IterativeRetriever(mm, mm.embedder, llm=FailingLLM())
        results = await retriever.retrieve("content", strategy="hyde", k_per_round=3)
        assert isinstance(results, list)
        await mm.close()
