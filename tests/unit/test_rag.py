"""tests/unit/test_rag.py — RAG モジュールの単体テスト"""

from __future__ import annotations

import pytest

from src.rag.chunker import Chunker
from src.rag.retriever import BaseRetriever, RawResult, RetrieverRouter
from src.rag.verifier import ResultVerifier

# ──────────────────────────────────────────────
# モックレトリーバー
# ──────────────────────────────────────────────


class MockRetriever(BaseRetriever):
    def __init__(self, name: str, available: bool = True, results: list[RawResult] = None) -> None:
        self._name = name
        self._available = available
        self._results = results or []
        self.calls: list[str] = []

    @property
    def source_name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return self._available

    async def search(self, query: str, max_results: int = 5) -> list[RawResult]:
        self.calls.append(query)
        return self._results[:max_results]


class FailingRetriever(BaseRetriever):
    @property
    def source_name(self) -> str:
        return "failing"

    def is_available(self) -> bool:
        return True

    async def search(self, query: str, max_results: int = 5) -> list[RawResult]:
        raise RuntimeError("Search failed")


def _make_result(title: str = "Test", content: str = "Content", score: float = 1.0, source: str = "mock") -> RawResult:
    return RawResult(title=title, content=content, url="https://example.com", source=source, score=score)


# ──────────────────────────────────────────────
# RetrieverRouter
# ──────────────────────────────────────────────


class TestRetrieverRouter:
    def _make_router(self) -> RetrieverRouter:
        router = RetrieverRouter.__new__(RetrieverRouter)
        router._retrievers = {}
        router._timeout = 30.0
        router._max_results = 5
        return router

    def test_register_retriever(self) -> None:
        router = self._make_router()
        mock = MockRetriever("test")
        router.register(mock)
        assert "test" in router._retrievers

    @pytest.mark.asyncio
    async def test_search_returns_results(self) -> None:
        router = self._make_router()
        mock = MockRetriever("mock", results=[
            _make_result("Result 1", score=0.9),
            _make_result("Result 2", score=0.7),
        ])
        router.register(mock)
        results = await router.search("test query")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_sorted_by_score(self) -> None:
        router = self._make_router()
        mock = MockRetriever("mock", results=[
            _make_result("Low", score=0.3),
            _make_result("High", score=0.9),
            _make_result("Mid", score=0.6),
        ])
        router.register(mock)
        results = await router.search("test")
        assert results[0].score >= results[1].score >= results[2].score

    @pytest.mark.asyncio
    async def test_search_respects_max_results(self) -> None:
        router = self._make_router()
        mock = MockRetriever("mock", results=[_make_result(f"R{i}") for i in range(10)])
        router.register(mock)
        results = await router.search("test", max_results=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_no_available_retrievers(self) -> None:
        router = self._make_router()
        router.register(MockRetriever("mock", available=False))
        results = await router.search("test")
        assert results == []

    @pytest.mark.asyncio
    async def test_failing_retriever_doesnt_crash(self) -> None:
        router = self._make_router()
        router.register(FailingRetriever())
        router.register(MockRetriever("good", results=[_make_result()]))
        results = await router.search("test")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_sources_filter(self) -> None:
        router = self._make_router()
        router.register(MockRetriever("a", results=[_make_result()]))
        router.register(MockRetriever("b", results=[_make_result()]))
        results = await router.search("test", sources=["a"])
        assert all(r.source == "mock" for r in results)

    @pytest.mark.asyncio
    async def test_parallel_search(self) -> None:
        """複数ソースが並列で呼ばれる。"""
        router = self._make_router()
        router.register(MockRetriever("a", results=[_make_result("A", score=0.8)]))
        router.register(MockRetriever("b", results=[_make_result("B", score=0.9)]))
        results = await router.search("test")
        assert len(results) == 2

    def test_available_sources(self) -> None:
        router = self._make_router()
        router.register(MockRetriever("available", available=True))
        router.register(MockRetriever("unavailable", available=False))
        available = router.available_sources()
        assert "available" in available
        assert "unavailable" not in available


# ──────────────────────────────────────────────
# Chunker
# ──────────────────────────────────────────────


class TestChunker:
    def test_chunk_text_short(self) -> None:
        chunker = Chunker(chunk_size=512)
        chunks = chunker.chunk_text("short text")
        assert len(chunks) == 1
        assert chunks[0] == "short text"

    def test_chunk_text_long(self) -> None:
        chunker = Chunker(chunk_size=100, chunk_overlap=10)
        text = "a" * 250
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1

    def test_chunk_text_empty(self) -> None:
        chunker = Chunker()
        assert chunker.chunk_text("") == []

    def test_chunk_size_respected(self) -> None:
        chunker = Chunker(chunk_size=50, chunk_overlap=5)
        text = "x" * 200
        chunks = chunker.chunk_text(text)
        for chunk in chunks:
            assert len(chunk) <= 50

    def test_overlap_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            Chunker(chunk_size=10, chunk_overlap=10)

    def test_chunk_result_creates_documents(self) -> None:
        chunker = Chunker(chunk_size=100, chunk_overlap=10)
        result = _make_result(
            title="Test Article",
            content="A" * 300,
            source="stackoverflow",
        )
        docs = chunker.chunk_result(result, domain="code")
        assert len(docs) > 0
        for doc in docs:
            assert doc.domain == "code"
            assert doc.source is not None

    def test_chunk_result_sets_chunk_index(self) -> None:
        chunker = Chunker(chunk_size=50, chunk_overlap=5)
        result = _make_result(content="B" * 200)
        docs = chunker.chunk_result(result)
        indices = [d.chunk_index for d in docs]
        assert indices == list(range(len(docs)))

    def test_chunk_result_parent_id(self) -> None:
        chunker = Chunker(chunk_size=50, chunk_overlap=5)
        result = _make_result(content="C" * 200)
        docs = chunker.chunk_result(result)
        if len(docs) > 1:
            # 2つ目以降は最初のドキュメントを parent_id として参照
            assert docs[1].parent_id == docs[0].id

    def test_chunk_results_batch(self) -> None:
        chunker = Chunker()
        results = [_make_result(f"Result {i}") for i in range(3)]
        docs = chunker.chunk_results(results, domain="general")
        assert len(docs) >= 3

    def test_source_type_mapping(self) -> None:
        from src.memory.schema import SourceType
        chunker = Chunker()
        for source, expected_type in [
            ("github", SourceType.GITHUB),
            ("stackoverflow", SourceType.STACKOVERFLOW),
            ("tavily", SourceType.TAVILY),
            ("arxiv", SourceType.ARXIV),
        ]:
            result = _make_result(source=source)
            docs = chunker.chunk_result(result)
            assert docs[0].source.source_type == expected_type


# ──────────────────────────────────────────────
# ResultVerifier
# ──────────────────────────────────────────────


class TestResultVerifier:
    @pytest.mark.asyncio
    async def test_no_gateway_passes_all(self) -> None:
        verifier = ResultVerifier(gateway=None)
        results = [_make_result() for _ in range(3)]
        verified = await verifier.verify(results, query="test")
        assert len(verified) == 3

    @pytest.mark.asyncio
    async def test_empty_results(self) -> None:
        verifier = ResultVerifier(gateway=None)
        verified = await verifier.verify([], query="test")
        assert verified == []

    @pytest.mark.asyncio
    async def test_with_mock_gateway_yes(self) -> None:
        """YES を返す LLM → 全件通過。"""
        from src.llm.gateway import LLMGateway, LLMResponse

        class YesGateway(LLMGateway):
            def __init__(self):
                self._providers = {}
                self._total_input_tokens = 0
                self._total_output_tokens = 0
                self._call_count = 0

            async def complete(self, prompt, **kwargs) -> LLMResponse:
                return LLMResponse(content="YES", provider="mock", model="mock")

        verifier = ResultVerifier(gateway=YesGateway())
        results = [_make_result() for _ in range(3)]
        verified = await verifier.verify(results, query="relevant query")
        assert len(verified) == 3

    @pytest.mark.asyncio
    async def test_with_mock_gateway_no(self) -> None:
        """NO を返す LLM → 0件。"""
        from src.llm.gateway import LLMGateway, LLMResponse

        class NoGateway(LLMGateway):
            def __init__(self):
                self._providers = {}
                self._total_input_tokens = 0
                self._total_output_tokens = 0
                self._call_count = 0

            async def complete(self, prompt, **kwargs) -> LLMResponse:
                return LLMResponse(content="NO", provider="mock", model="mock")

        verifier = ResultVerifier(gateway=NoGateway())
        results = [_make_result() for _ in range(3)]
        verified = await verifier.verify(results, query="irrelevant")
        assert len(verified) == 0
