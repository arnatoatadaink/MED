"""tests/unit/test_maturation.py — Phase 2 メモリ成熟モジュールの単体テスト"""

from __future__ import annotations

import json

import pytest

from src.common.config import MetadataConfig
from src.llm.gateway import LLMGateway, LLMResponse
from src.memory.maturation.difficulty_tagger import DifficultyTagger
from src.memory.maturation.reviewer import MemoryReviewer, ReviewResult
from src.memory.maturation.seed_builder import SeedBuilder, SeedResult
from src.memory.metadata_store import MetadataStore
from src.memory.schema import (
    DifficultyLevel,
    Document,
    ReviewStatus,
    SourceMeta,
    SourceType,
)

# ──────────────────────────────────────────────
# モック
# ──────────────────────────────────────────────


class MockGateway(LLMGateway):
    def __init__(self, response: str = "intermediate") -> None:
        self._response = response
        self._providers = {}
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0

    async def complete(self, prompt, **kwargs) -> LLMResponse:
        self._call_count += 1
        return LLMResponse(
            content=self._response,
            provider="mock",
            model="mock-model",
            input_tokens=10,
            output_tokens=20,
        )

    async def complete_messages(self, messages, **kwargs) -> LLMResponse:
        self._call_count += 1
        return LLMResponse(
            content=self._response,
            provider="mock",
            model="mock-model",
            input_tokens=10,
            output_tokens=20,
        )


def _make_doc(content: str = "Python is a programming language.", domain: str = "general") -> Document:
    return Document(
        content=content,
        domain=domain,
        source=SourceMeta(source_type=SourceType.MANUAL),
    )


async def _make_store() -> MetadataStore:
    store = MetadataStore(MetadataConfig(db_path=":memory:"))
    await store.initialize()
    return store


# ──────────────────────────────────────────────
# DifficultyTagger テスト
# ──────────────────────────────────────────────


class TestDifficultyTagger:
    @pytest.mark.asyncio
    async def test_tag_beginner(self) -> None:
        gateway = MockGateway("beginner")
        tagger = DifficultyTagger(gateway)
        doc = _make_doc("Hello world tutorial for beginners.")
        level = await tagger.tag(doc)
        assert level == DifficultyLevel.BEGINNER

    @pytest.mark.asyncio
    async def test_tag_intermediate(self) -> None:
        gateway = MockGateway("intermediate")
        tagger = DifficultyTagger(gateway)
        doc = _make_doc("Decorators and context managers in Python.")
        level = await tagger.tag(doc)
        assert level == DifficultyLevel.INTERMEDIATE

    @pytest.mark.asyncio
    async def test_tag_advanced(self) -> None:
        gateway = MockGateway("advanced")
        tagger = DifficultyTagger(gateway)
        doc = _make_doc("C++ template metaprogramming with SFINAE.")
        level = await tagger.tag(doc)
        assert level == DifficultyLevel.ADVANCED

    @pytest.mark.asyncio
    async def test_tag_expert(self) -> None:
        gateway = MockGateway("expert")
        tagger = DifficultyTagger(gateway)
        doc = _make_doc("Cutting-edge CUDA kernel optimization.")
        level = await tagger.tag(doc)
        assert level == DifficultyLevel.EXPERT

    @pytest.mark.asyncio
    async def test_tag_with_extra_whitespace(self) -> None:
        """先頭の単語だけ使うので余分なテキストは無視する。"""
        gateway = MockGateway("  intermediate  (moderate complexity)")
        tagger = DifficultyTagger(gateway)
        doc = _make_doc("Some text.")
        level = await tagger.tag(doc)
        assert level == DifficultyLevel.INTERMEDIATE

    @pytest.mark.asyncio
    async def test_tag_unknown_falls_back_to_default(self) -> None:
        gateway = MockGateway("unknown_level")
        tagger = DifficultyTagger(gateway, default_level=DifficultyLevel.BEGINNER)
        doc = _make_doc("Something.")
        level = await tagger.tag(doc)
        assert level == DifficultyLevel.BEGINNER

    @pytest.mark.asyncio
    async def test_tag_failure_returns_default(self) -> None:
        class FailingGateway(MockGateway):
            async def complete(self, prompt, **kwargs):
                raise RuntimeError("LLM offline")

        tagger = DifficultyTagger(FailingGateway(), default_level=DifficultyLevel.ADVANCED)
        doc = _make_doc("test")
        level = await tagger.tag(doc)
        assert level == DifficultyLevel.ADVANCED

    @pytest.mark.asyncio
    async def test_tag_batch(self) -> None:
        gateway = MockGateway("beginner")
        tagger = DifficultyTagger(gateway)
        docs = [_make_doc(f"doc {i}") for i in range(4)]
        result = await tagger.tag_batch(docs)
        assert len(result) == 4
        for doc in docs:
            assert doc.id in result
            assert result[doc.id] == DifficultyLevel.BEGINNER

    @pytest.mark.asyncio
    async def test_tag_batch_empty(self) -> None:
        tagger = DifficultyTagger(MockGateway())
        result = await tagger.tag_batch([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_gateway_called_once_per_doc(self) -> None:
        gateway = MockGateway("advanced")
        tagger = DifficultyTagger(gateway)
        docs = [_make_doc(f"doc {i}") for i in range(3)]
        await tagger.tag_batch(docs)
        assert gateway._call_count == 3

    @pytest.mark.asyncio
    async def test_text_truncated(self) -> None:
        """max_text_length を超えるテキストは切り捨てられる（エラーにならない）。"""
        gateway = MockGateway("intermediate")
        tagger = DifficultyTagger(gateway, max_text_length=10)
        doc = _make_doc("A" * 1000)
        level = await tagger.tag(doc)
        assert level == DifficultyLevel.INTERMEDIATE


# ──────────────────────────────────────────────
# MemoryReviewer テスト
# ──────────────────────────────────────────────


class TestMemoryReviewer:
    def _good_response(self) -> str:
        return json.dumps({
            "quality_score": 0.85,
            "confidence": 0.9,
            "approved": True,
            "reason": "High quality technical document.",
        })

    def _bad_response(self) -> str:
        return json.dumps({
            "quality_score": 0.3,
            "confidence": 0.8,
            "approved": False,
            "reason": "Low quality, lacks depth.",
        })

    @pytest.mark.asyncio
    async def test_review_approved(self) -> None:
        store = await _make_store()
        gateway = MockGateway(self._good_response())
        reviewer = MemoryReviewer(gateway, store)
        doc = _make_doc("Detailed Python tutorial with examples.")
        await store.save(doc)
        result = await reviewer.review(doc)
        assert result.approved is True
        assert result.quality_score == pytest.approx(0.85)
        assert result.review_status == ReviewStatus.APPROVED

    @pytest.mark.asyncio
    async def test_review_rejected(self) -> None:
        store = await _make_store()
        gateway = MockGateway(self._bad_response())
        reviewer = MemoryReviewer(gateway, store)
        doc = _make_doc("Low quality content.")
        await store.save(doc)
        result = await reviewer.review(doc)
        assert result.approved is False
        assert result.review_status == ReviewStatus.REJECTED
        assert result.quality_score == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_review_updates_store(self) -> None:
        store = await _make_store()
        gateway = MockGateway(self._good_response())
        reviewer = MemoryReviewer(gateway, store)
        doc = _make_doc("Good content.")
        await store.save(doc)
        await reviewer.review(doc)
        saved = await store.get(doc.id)
        assert saved is not None
        assert saved.usefulness.teacher_quality == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_review_result_fields(self) -> None:
        store = await _make_store()
        gateway = MockGateway(self._good_response())
        reviewer = MemoryReviewer(gateway, store)
        doc = _make_doc("Test document.")
        await store.save(doc)
        result = await reviewer.review(doc)
        assert result.doc_id == doc.id
        assert isinstance(result.reason, str)
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_review_with_code_block_json(self) -> None:
        """LLM が ```json ... ``` で包んで返した場合も正しくパースできる。"""
        store = await _make_store()
        wrapped = "```json\n" + self._good_response() + "\n```"
        gateway = MockGateway(wrapped)
        reviewer = MemoryReviewer(gateway, store)
        doc = _make_doc("test")
        await store.save(doc)
        result = await reviewer.review(doc)
        assert result.approved is True

    @pytest.mark.asyncio
    async def test_review_llm_failure_returns_rejected(self) -> None:
        class FailingGateway(MockGateway):
            async def complete(self, prompt, **kwargs):
                raise RuntimeError("API error")

        store = await _make_store()
        reviewer = MemoryReviewer(FailingGateway(), store)
        doc = _make_doc("Something.")
        await store.save(doc)
        result = await reviewer.review(doc)
        assert result.approved is False
        assert result.quality_score == 0.0

    @pytest.mark.asyncio
    async def test_review_invalid_json_fallback(self) -> None:
        store = await _make_store()
        gateway = MockGateway("This is not JSON at all.")
        reviewer = MemoryReviewer(gateway, store)
        doc = _make_doc("Something.")
        await store.save(doc)
        result = await reviewer.review(doc)
        # parse error fallback: approved=False, quality=0.5
        assert result.approved is False

    @pytest.mark.asyncio
    async def test_review_batch(self) -> None:
        store = await _make_store()
        gateway = MockGateway(self._good_response())
        reviewer = MemoryReviewer(gateway, store)
        docs = [_make_doc(f"doc {i}") for i in range(3)]
        for d in docs:
            await store.save(d)
        results = await reviewer.review_batch(docs)
        assert len(results) == 3
        assert all(isinstance(r, ReviewResult) for r in results)

    @pytest.mark.asyncio
    async def test_review_unreviewed(self) -> None:
        store = await _make_store()
        gateway = MockGateway(self._good_response())
        reviewer = MemoryReviewer(gateway, store)
        docs = [_make_doc(f"doc {i}") for i in range(2)]
        for d in docs:
            await store.save(d)
        results = await reviewer.review_unreviewed(limit=10)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_review_unreviewed_empty(self) -> None:
        store = await _make_store()
        reviewer = MemoryReviewer(MockGateway(), store)
        results = await reviewer.review_unreviewed(limit=10)
        assert results == []


# ──────────────────────────────────────────────
# SeedBuilder テスト
# ──────────────────────────────────────────────


class MockMemoryManager:
    """MemoryManager の最小モック。"""

    def __init__(self) -> None:
        self._docs: list[Document] = []
        self._id_counter = 0

    async def add(self, doc: Document) -> str:
        self._docs.append(doc)
        self._id_counter += 1
        return f"seed-doc-{self._id_counter}"


class TestSeedBuilder:
    def _make_builder(self, response: str = "Generated seed content.") -> tuple[SeedBuilder, MockMemoryManager]:
        gateway = MockGateway(response)
        mm = MockMemoryManager()
        builder = SeedBuilder(gateway, mm)  # type: ignore[arg-type]
        return builder, mm

    @pytest.mark.asyncio
    async def test_build_returns_seed_result(self) -> None:
        builder, mm = self._make_builder()
        result = await builder.build(topic="Python basics", domain="code", n_samples=4)
        assert isinstance(result, SeedResult)
        assert result.docs_created == 4
        assert result.docs_failed == 0
        assert len(result.doc_ids) == 4

    @pytest.mark.asyncio
    async def test_build_creates_correct_count(self) -> None:
        builder, mm = self._make_builder()
        result = await builder.build(topic="FAISS", domain="code", n_samples=8)
        assert result.docs_created == 8
        assert len(mm._docs) == 8

    @pytest.mark.asyncio
    async def test_build_qa_type(self) -> None:
        builder, mm = self._make_builder("Q: What is FAISS?\nA: A vector search library.")
        result = await builder.build(
            topic="FAISS", domain="general",
            difficulty_distribution={"beginner": 1, "intermediate": 1},
            seed_type="qa",
        )
        assert result.docs_created == 2

    @pytest.mark.asyncio
    async def test_build_custom_distribution(self) -> None:
        builder, mm = self._make_builder()
        dist = {"beginner": 2, "intermediate": 1}
        result = await builder.build(
            topic="Algorithms", domain="code", n_samples=3,
            difficulty_distribution=dist,
        )
        assert result.docs_created == 3
        difficulties = [d.difficulty for d in mm._docs]
        assert difficulties.count(DifficultyLevel.BEGINNER) == 2
        assert difficulties.count(DifficultyLevel.INTERMEDIATE) == 1

    @pytest.mark.asyncio
    async def test_build_default_distribution(self) -> None:
        builder, mm = self._make_builder()
        result = await builder.build(topic="topic", domain="general", n_samples=8)
        assert result.docs_created == 8
        difficulties = [d.difficulty for d in mm._docs]
        assert DifficultyLevel.BEGINNER in difficulties
        assert DifficultyLevel.INTERMEDIATE in difficulties
        assert DifficultyLevel.ADVANCED in difficulties
        assert DifficultyLevel.EXPERT in difficulties

    @pytest.mark.asyncio
    async def test_build_docs_approved_by_default(self) -> None:
        """Teacher 生成ドキュメントは APPROVED として保存される。"""
        builder, mm = self._make_builder()
        await builder.build(topic="topic", domain="code", n_samples=2)
        for doc in mm._docs:
            assert doc.review_status == ReviewStatus.APPROVED

    @pytest.mark.asyncio
    async def test_build_source_type_teacher(self) -> None:
        builder, mm = self._make_builder()
        await builder.build(topic="topic", domain="code", n_samples=2)
        for doc in mm._docs:
            assert doc.source.source_type == SourceType.TEACHER

    @pytest.mark.asyncio
    async def test_build_handles_empty_content(self) -> None:
        """LLM が空文字を返した場合は failed にカウントされる。"""
        builder, mm = self._make_builder("   ")  # whitespace only
        result = await builder.build(topic="topic", domain="code", n_samples=4)
        assert result.docs_failed == 4
        assert result.docs_created == 0

    @pytest.mark.asyncio
    async def test_build_handles_partial_failure(self) -> None:
        """一部が空でも残りは成功する。"""
        call_count = 0

        class AlternatingGateway(MockGateway):
            async def complete(self, prompt, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count % 2 == 0:
                    return LLMResponse(
                        content="   ",
                        provider="mock", model="mock-model",
                        input_tokens=0, output_tokens=0,
                    )
                return LLMResponse(
                    content="Valid content.",
                    provider="mock", model="mock-model",
                    input_tokens=10, output_tokens=20,
                )

        mm = MockMemoryManager()
        builder = SeedBuilder(AlternatingGateway(), mm)  # type: ignore[arg-type]
        result = await builder.build(topic="topic", domain="code", n_samples=4)
        assert result.docs_created + result.docs_failed == 4

    @pytest.mark.asyncio
    async def test_default_distribution_sums_to_n(self) -> None:
        builder, _ = self._make_builder()
        for n in [1, 4, 7, 10, 20]:
            dist = builder._default_distribution(n)
            assert sum(dist.values()) == n

    @pytest.mark.asyncio
    async def test_seed_result_doc_ids_non_empty(self) -> None:
        builder, _ = self._make_builder()
        result = await builder.build(topic="X", domain="general", n_samples=3)
        assert all(isinstance(id_, str) and len(id_) > 0 for id_ in result.doc_ids)

    @pytest.mark.asyncio
    async def test_gateway_called_once_per_sample(self) -> None:
        gateway = MockGateway("content")
        mm = MockMemoryManager()
        builder = SeedBuilder(gateway, mm)  # type: ignore[arg-type]
        await builder.build(topic="T", domain="code", n_samples=5)
        assert gateway._call_count == 5
