"""tests/unit/test_schema.py — src/memory/schema.py の単体テスト"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from src.memory.schema import (
    DifficultyLevel,
    Document,
    Domain,
    ExecutionResult,
    ReviewStatus,
    RewardSignal,
    SearchQuery,
    SearchResult,
    SourceMeta,
    SourceType,
    TrainingBatch,
    TrainStepResult,
    UsefulnessScore,
)

# ============================================================================
# 列挙型テスト
# ============================================================================


class TestEnums:
    def test_source_type_values(self):
        assert SourceType.GITHUB == "github"
        assert SourceType.STACKOVERFLOW == "stackoverflow"
        assert SourceType.TAVILY == "tavily"
        assert SourceType.ARXIV == "arxiv"
        assert SourceType.MANUAL == "manual"
        assert SourceType.TEACHER == "teacher"
        assert SourceType.SEED == "seed"

    def test_difficulty_level_values(self):
        assert DifficultyLevel.BEGINNER == "beginner"
        assert DifficultyLevel.INTERMEDIATE == "intermediate"
        assert DifficultyLevel.ADVANCED == "advanced"
        assert DifficultyLevel.EXPERT == "expert"

    def test_domain_values(self):
        assert Domain.CODE == "code"
        assert Domain.ACADEMIC == "academic"
        assert Domain.GENERAL == "general"

    def test_review_status_values(self):
        assert ReviewStatus.UNREVIEWED == "unreviewed"
        assert ReviewStatus.APPROVED == "approved"
        assert ReviewStatus.REJECTED == "rejected"
        assert ReviewStatus.NEEDS_UPDATE == "needs_update"

    def test_source_type_is_str(self):
        """str Enum なので文字列比較・.value アクセスが可能であること。"""
        assert SourceType.GITHUB == "github"
        assert SourceType.GITHUB.value == "github"
        assert str(SourceType.GITHUB.value) == "github"


# ============================================================================
# SourceMeta テスト
# ============================================================================


class TestSourceMeta:
    def test_default_values(self):
        meta = SourceMeta()
        assert meta.source_type == SourceType.MANUAL
        assert meta.url is None
        assert meta.title is None
        assert meta.tags == []
        assert isinstance(meta.retrieved_at, datetime)

    def test_full_construction(self):
        meta = SourceMeta(
            source_type=SourceType.GITHUB,
            url="https://github.com/example/repo",
            title="Example Repository",
            author="user123",
            language="python",
            tags=["fastapi", "async"],
            extra={"stars": 1500},
        )
        assert meta.source_type == SourceType.GITHUB
        assert meta.url == "https://github.com/example/repo"
        assert meta.language == "python"
        assert "fastapi" in meta.tags
        assert meta.extra["stars"] == 1500

    def test_serialization_roundtrip(self):
        meta = SourceMeta(source_type=SourceType.ARXIV, title="Test Paper")
        data = meta.model_dump()
        restored = SourceMeta(**data)
        assert restored.source_type == SourceType.ARXIV
        assert restored.title == "Test Paper"


# ============================================================================
# UsefulnessScore テスト
# ============================================================================


class TestUsefulnessScore:
    def test_default_values(self):
        score = UsefulnessScore()
        assert score.retrieval_count == 0
        assert score.selection_count == 0
        assert score.positive_feedback == 0
        assert score.negative_feedback == 0
        assert score.teacher_quality == 0.0
        assert score.execution_success_rate == 0.0
        assert score.freshness == 1.0
        assert score.composite == 0.0

    def test_feedback_ratio_no_feedback(self):
        score = UsefulnessScore()
        assert score.feedback_ratio == 0.0

    def test_feedback_ratio_with_feedback(self):
        score = UsefulnessScore(positive_feedback=7, negative_feedback=3)
        assert score.feedback_ratio == pytest.approx(0.7)

    def test_selection_rate_no_retrieval(self):
        score = UsefulnessScore()
        assert score.selection_rate == 0.0

    def test_selection_rate_with_retrieval(self):
        score = UsefulnessScore(retrieval_count=10, selection_count=3)
        assert score.selection_rate == pytest.approx(0.3)

    def test_score_clamping_upper(self):
        score = UsefulnessScore(teacher_quality=1.5)
        assert score.teacher_quality == 1.0

    def test_score_clamping_lower(self):
        score = UsefulnessScore(execution_success_rate=-0.5)
        assert score.execution_success_rate == 0.0

    def test_freshness_clamping(self):
        score = UsefulnessScore(freshness=2.0)
        assert score.freshness == 1.0

    def test_composite_clamping(self):
        score = UsefulnessScore(composite=-1.0)
        assert score.composite == 0.0


# ============================================================================
# Document テスト
# ============================================================================


class TestDocument:
    def test_minimal_construction(self):
        doc = Document(content="Hello, world!")
        assert doc.content == "Hello, world!"
        assert len(doc.id) == 32  # uuid4 hex
        assert doc.domain == Domain.GENERAL
        assert doc.embedding is None
        assert doc.difficulty is None
        assert doc.review_status == ReviewStatus.UNREVIEWED
        assert doc.confidence == 0.5
        assert isinstance(doc.created_at, datetime)

    def test_full_construction(self):
        embedding = np.random.randn(768).astype(np.float32)
        doc = Document(
            content="FastAPIでWebSocket認証を実装する方法",
            domain=Domain.CODE,
            embedding=embedding,
            source=SourceMeta(
                source_type=SourceType.STACKOVERFLOW,
                url="https://stackoverflow.com/q/12345",
                language="python",
            ),
            difficulty=DifficultyLevel.INTERMEDIATE,
            confidence=0.85,
            is_executable=True,
        )
        assert doc.domain == Domain.CODE
        assert doc.embedding.shape == (768,)
        assert doc.source.source_type == SourceType.STACKOVERFLOW
        assert doc.difficulty == DifficultyLevel.INTERMEDIATE
        assert doc.confidence == pytest.approx(0.85)
        assert doc.is_executable is True

    def test_unique_ids(self):
        doc1 = Document(content="a")
        doc2 = Document(content="b")
        assert doc1.id != doc2.id

    def test_confidence_clamping(self):
        doc = Document(content="test", confidence=1.5)
        assert doc.confidence == 1.0

        doc2 = Document(content="test", confidence=-0.3)
        assert doc2.confidence == 0.0

    def test_embedding_validation_1d(self):
        embedding = np.random.randn(768).astype(np.float32)
        doc = Document(content="test", embedding=embedding)
        assert doc.embedding is not None
        assert doc.embedding.ndim == 1

    def test_embedding_validation_rejects_2d(self):
        embedding_2d = np.random.randn(2, 768).astype(np.float32)
        with pytest.raises(Exception):
            Document(content="test", embedding=embedding_2d)

    def test_embedding_none_allowed(self):
        doc = Document(content="test", embedding=None)
        assert doc.embedding is None

    def test_usefulness_default(self):
        doc = Document(content="test")
        assert isinstance(doc.usefulness, UsefulnessScore)
        assert doc.usefulness.retrieval_count == 0

    def test_chunk_fields(self):
        parent = Document(content="full document")
        chunk = Document(
            content="chunk 0 of full document",
            parent_id=parent.id,
            chunk_index=0,
        )
        assert chunk.parent_id == parent.id
        assert chunk.chunk_index == 0

    def test_serialization_without_embedding(self):
        doc = Document(content="test", domain=Domain.CODE)
        data = doc.model_dump()
        assert data["content"] == "test"
        assert data["domain"] == "code"

    def test_serialization_with_embedding(self):
        embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        doc = Document(content="test", embedding=embedding)
        data = doc.model_dump()
        # numpy array は list に変換される
        assert len(data["embedding"]) == 3


# ============================================================================
# SearchResult / SearchQuery テスト
# ============================================================================


class TestSearchResult:
    def test_construction(self):
        doc = Document(content="result doc")
        result = SearchResult(document=doc, score=0.92, rank=0)
        assert result.score == pytest.approx(0.92)
        assert result.rank == 0
        assert result.rerank_score is None

    def test_with_rerank(self):
        doc = Document(content="result doc")
        result = SearchResult(
            document=doc,
            score=0.85,
            rank=2,
            rerank_score=0.95,
        )
        assert result.rerank_score == pytest.approx(0.95)


class TestSearchQuery:
    def test_defaults(self):
        q = SearchQuery(text="Python sort list")
        assert q.text == "Python sort list"
        assert q.domain is None
        assert q.top_k == 5
        assert q.min_score == 0.0
        assert q.filters == {}

    def test_with_domain(self):
        q = SearchQuery(text="sort", domain=Domain.CODE, top_k=10, min_score=0.5)
        assert q.domain == Domain.CODE
        assert q.top_k == 10
        assert q.min_score == pytest.approx(0.5)


# ============================================================================
# ExecutionResult テスト
# ============================================================================


class TestExecutionResult:
    def test_success(self):
        result = ExecutionResult(
            success=True,
            exit_code=0,
            stdout="Hello, world!\n",
            execution_time_ms=150.5,
        )
        assert result.success is True
        assert result.stdout == "Hello, world!\n"
        assert result.error_type is None
        assert result.timed_out is False

    def test_failure(self):
        result = ExecutionResult(
            success=False,
            exit_code=1,
            stderr="NameError: name 'x' is not defined",
            error_type="NameError",
        )
        assert result.success is False
        assert result.error_type == "NameError"

    def test_timeout(self):
        result = ExecutionResult(
            success=False,
            exit_code=-1,
            timed_out=True,
            execution_time_ms=30000.0,
        )
        assert result.timed_out is True


# ============================================================================
# RewardSignal テスト
# ============================================================================


class TestRewardSignal:
    def test_basic(self):
        signal = RewardSignal(
            total=0.75,
            components={
                "correctness": 0.9,
                "retrieval_quality": 0.7,
                "exec_success": 1.0,
                "efficiency": 0.5,
                "memory_utilization": 0.3,
            },
        )
        assert signal.total == pytest.approx(0.75)
        assert len(signal.components) == 5
        assert signal.metadata == {}

    def test_with_metadata(self):
        signal = RewardSignal(
            total=0.5,
            metadata={"teacher_model": "claude-haiku", "latency_ms": 200},
        )
        assert signal.metadata["teacher_model"] == "claude-haiku"


# ============================================================================
# TrainingBatch / TrainStepResult テスト
# ============================================================================


class TestTrainingBatch:
    def test_minimal(self):
        batch = TrainingBatch(queries=["q1", "q2", "q3"])
        assert len(batch.queries) == 3
        assert batch.reference_responses is None
        assert batch.difficulty_levels is None

    def test_with_curriculum(self):
        batch = TrainingBatch(
            queries=["easy question", "hard question"],
            difficulty_levels=[DifficultyLevel.BEGINNER, DifficultyLevel.ADVANCED],
        )
        assert batch.difficulty_levels[0] == DifficultyLevel.BEGINNER
        assert batch.difficulty_levels[1] == DifficultyLevel.ADVANCED


class TestTrainStepResult:
    def test_minimal(self):
        result = TrainStepResult(loss=0.42)
        assert result.loss == pytest.approx(0.42)
        assert result.mean_reward == 0.0
        assert result.grad_norm is None

    def test_full(self):
        result = TrainStepResult(
            loss=0.15,
            mean_reward=0.78,
            grad_norm=1.2,
            learning_rate=1e-4,
            extra_metrics={"kl_divergence": 0.003},
        )
        assert result.mean_reward == pytest.approx(0.78)
        assert result.extra_metrics["kl_divergence"] == pytest.approx(0.003)
