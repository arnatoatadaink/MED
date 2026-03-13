"""tests/unit/test_scoring.py — scoring モジュールの単体テスト"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.memory.schema import Document, SourceMeta, SourceType
from src.memory.scoring.composite_scorer import CompositeScorer
from src.memory.scoring.freshness import FreshnessScorer
from src.memory.scoring.usefulness import UsefulnessScorer, UsefulnessWeights

# ──────────────────────────────────────────────
# FreshnessScorer
# ──────────────────────────────────────────────


class TestFreshnessScorer:
    def test_new_document_score_near_one(self) -> None:
        scorer = FreshnessScorer()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        retrieved_at = now - timedelta(hours=1)
        score = scorer.score("code", retrieved_at=retrieved_at, now=now)
        assert score > 0.999

    def test_old_document_score_near_zero(self) -> None:
        scorer = FreshnessScorer()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        retrieved_at = now - timedelta(days=3650)  # 10年前
        score = scorer.score("code", retrieved_at=retrieved_at, now=now)
        assert score < 0.1

    def test_half_life_at_half_life_days(self) -> None:
        scorer = FreshnessScorer(half_life_days={"code": 180.0})
        now = datetime(2026, 1, 1, tzinfo=UTC)
        retrieved_at = now - timedelta(days=180)
        score = scorer.score("code", retrieved_at=retrieved_at, now=now)
        assert abs(score - 0.5) < 0.01

    def test_unknown_domain_uses_fallback(self) -> None:
        scorer = FreshnessScorer()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        retrieved_at = now - timedelta(days=10)
        score = scorer.score("unknown_domain", retrieved_at=retrieved_at, now=now)
        assert 0.0 <= score <= 1.0

    def test_none_retrieved_at_returns_half(self) -> None:
        scorer = FreshnessScorer()
        score = scorer.score("code", retrieved_at=None)
        assert score == 0.5

    def test_score_range(self) -> None:
        scorer = FreshnessScorer()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        for days in [0, 30, 180, 365, 730]:
            rt = now - timedelta(days=days)
            s = scorer.score("code", retrieved_at=rt, now=now)
            assert 0.0 <= s <= 1.0, f"days={days} score={s}"

    def test_academic_slower_decay_than_code(self) -> None:
        scorer = FreshnessScorer()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        rt = now - timedelta(days=365)
        code_score = scorer.score("code", retrieved_at=rt, now=now)
        academic_score = scorer.score("academic", retrieved_at=rt, now=now)
        assert academic_score > code_score

    def test_score_batch(self) -> None:
        scorer = FreshnessScorer()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        rts = [now - timedelta(days=d) for d in [0, 90, 365]]
        scores = scorer.score_batch("code", rts, now=now)
        assert len(scores) == 3
        assert scores[0] > scores[1] > scores[2]

    def test_set_half_life(self) -> None:
        scorer = FreshnessScorer()
        scorer.set_half_life("custom", 100.0)
        now = datetime(2026, 1, 1, tzinfo=UTC)
        rt = now - timedelta(days=100)
        score = scorer.score("custom", retrieved_at=rt, now=now)
        assert abs(score - 0.5) < 0.01

    def test_set_half_life_invalid(self) -> None:
        scorer = FreshnessScorer()
        with pytest.raises(ValueError):
            scorer.set_half_life("code", -1.0)

    def test_naive_datetime_handled(self) -> None:
        """タイムゾーンなしの datetime も正常処理される。"""
        scorer = FreshnessScorer()
        now = datetime(2026, 1, 1)  # naive
        rt = now - timedelta(days=30)
        score = scorer.score("code", retrieved_at=rt, now=now)
        assert 0.0 <= score <= 1.0


# ──────────────────────────────────────────────
# UsefulnessScorer
# ──────────────────────────────────────────────


class TestUsefulnessWeights:
    def test_default_weights_sum_to_one(self) -> None:
        w = UsefulnessWeights()
        total = w.feedback + w.selection + w.teacher_quality + w.execution + w.freshness
        assert abs(total - 1.0) < 1e-6

    def test_invalid_weights_raise(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            UsefulnessWeights(feedback=0.5, selection=0.5, teacher_quality=0.5, execution=0.5, freshness=0.5)


class TestUsefulnessScorer:
    def test_all_zero_returns_mid_range(self) -> None:
        scorer = UsefulnessScorer()
        score = scorer.compute(
            retrieval_count=0, selection_count=0,
            positive_feedback=0, negative_feedback=0,
            teacher_quality=0.0, execution_success_rate=0.0, freshness=0.0,
        )
        # feedback=0.5(中間), selection=0.5(中間), quality=0, exec=0, fresh=0
        assert 0.0 <= score <= 1.0

    def test_perfect_scores(self) -> None:
        scorer = UsefulnessScorer()
        score = scorer.compute(
            retrieval_count=100, selection_count=100,
            positive_feedback=100, negative_feedback=0,
            teacher_quality=1.0, execution_success_rate=1.0, freshness=1.0,
        )
        assert score > 0.9

    def test_poor_scores(self) -> None:
        scorer = UsefulnessScorer()
        score = scorer.compute(
            retrieval_count=100, selection_count=0,
            positive_feedback=0, negative_feedback=100,
            teacher_quality=0.0, execution_success_rate=0.0, freshness=0.0,
        )
        assert score < 0.3

    def test_code_domain_uses_exec_rate(self) -> None:
        scorer = UsefulnessScorer()
        score_high_exec = scorer.compute(
            execution_success_rate=1.0, domain="code"
        )
        score_low_exec = scorer.compute(
            execution_success_rate=0.0, domain="code"
        )
        assert score_high_exec > score_low_exec

    def test_non_code_exec_rate_ignored(self) -> None:
        scorer = UsefulnessScorer()
        # general ドメインでは exec_score = teacher_quality
        s_high = scorer.compute(execution_success_rate=1.0, teacher_quality=0.5, domain="general")
        s_low = scorer.compute(execution_success_rate=0.0, teacher_quality=0.5, domain="general")
        # exec が異なっても teacher_quality が同じなら exec_score も同じ
        assert abs(s_high - s_low) < 1e-6

    def test_output_range(self) -> None:
        scorer = UsefulnessScorer()
        for _ in range(10):
            import random
            score = scorer.compute(
                retrieval_count=random.randint(0, 100),
                selection_count=random.randint(0, 50),
                positive_feedback=random.randint(0, 50),
                negative_feedback=random.randint(0, 50),
                teacher_quality=random.random(),
                execution_success_rate=random.random(),
                freshness=random.random(),
            )
            assert 0.0 <= score <= 1.0

    def test_compute_from_dict(self) -> None:
        scorer = UsefulnessScorer()
        row = {
            "retrieval_count": 10,
            "selection_count": 5,
            "positive_feedback": 8,
            "negative_feedback": 2,
            "teacher_quality": 0.7,
            "execution_success_rate": 0.8,
            "freshness": 0.9,
            "domain": "code",
        }
        score = scorer.compute_from_dict(row)
        assert 0.0 <= score <= 1.0


# ──────────────────────────────────────────────
# CompositeScorer
# ──────────────────────────────────────────────


def _make_doc(domain: str = "code", days_old: int = 30) -> Document:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    retrieved_at = now - timedelta(days=days_old)
    return Document(
        content="test content",
        domain=domain,
        source=SourceMeta(source_type=SourceType.GITHUB, retrieved_at=retrieved_at),
    )


class TestCompositeScorer:
    def test_score_in_range(self) -> None:
        scorer = CompositeScorer()
        doc = _make_doc()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        score = scorer.compute_for_document(doc, now=now)
        assert 0.0 <= score <= 1.0

    def test_newer_doc_scores_higher(self) -> None:
        scorer = CompositeScorer()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        new_doc = _make_doc(days_old=1)
        old_doc = _make_doc(days_old=500)
        score_new = scorer.compute_for_document(new_doc, now=now)
        score_old = scorer.compute_for_document(old_doc, now=now)
        assert score_new > score_old

    def test_compute_from_row(self) -> None:
        scorer = CompositeScorer()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        row = {
            "domain": "code",
            "retrieval_count": 10,
            "selection_count": 5,
            "positive_feedback": 8,
            "negative_feedback": 2,
            "teacher_quality": 0.7,
            "execution_success_rate": 0.8,
            "source_retrieved_at": (now - timedelta(days=30)).isoformat(),
        }
        score = scorer.compute_from_row(row, now=now)
        assert 0.0 <= score <= 1.0

    def test_compute_from_row_invalid_date(self) -> None:
        scorer = CompositeScorer()
        row = {
            "domain": "general",
            "source_retrieved_at": "not-a-date",
        }
        score = scorer.compute_from_row(row)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_update_store(self) -> None:
        """update_store が MetadataStore を正しく更新する。"""
        from src.common.config import MetadataConfig
        from src.memory.metadata_store import MetadataStore

        store = MetadataStore(MetadataConfig(db_path=":memory:"))
        await store.initialize()

        docs = [_make_doc(days_old=d) for d in [10, 100, 300]]
        for doc in docs:
            await store.save(doc)

        scorer = CompositeScorer()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        updated = await scorer.update_store(store, doc_ids=[d.id for d in docs], now=now)
        assert updated == 3

        # スコアが書き込まれていることを確認
        for doc in docs:
            saved = await store.get(doc.id)
            assert saved is not None
            assert saved.usefulness.composite > 0.0

        await store.close()
