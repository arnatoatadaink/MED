"""tests/unit/test_teacher_provenance_step4.py — CompositeScorer Teacher 信頼度反映テスト (Step 4)

テスト対象:
- src/memory/scoring/composite_scorer.py
  - _apply_trust()
  - compute_for_document() の trust_score 引数
  - compute_from_row() の trust_score 引数
  - update_store() の trust_map 引数
  - build_trust_map() ヘルパー
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.memory.schema import Document, SourceMeta, SourceType

# ===========================================================================
# Helpers
# ===========================================================================

def _make_doc(
    content: str = "test",
    teacher_id: str | None = None,
    domain: str = "general",
) -> Document:
    source = SourceMeta(source_type=SourceType.TEACHER if teacher_id else SourceType.MANUAL)
    if teacher_id:
        source.set_teacher(teacher_id)
    return Document(content=content, domain=domain, source=source)


def _scorer(trust_weight: float = 1.0):
    from src.memory.scoring.composite_scorer import CompositeScorer
    return CompositeScorer(teacher_trust_weight=trust_weight)


# ===========================================================================
# _apply_trust
# ===========================================================================

class TestApplyTrust:
    def test_trust_1_no_change(self):
        scorer = _scorer(trust_weight=1.0)
        assert scorer._apply_trust(0.8, 1.0) == pytest.approx(0.8)

    def test_trust_0_zeroes_score(self):
        scorer = _scorer(trust_weight=1.0)
        assert scorer._apply_trust(0.8, 0.0) == pytest.approx(0.0)

    def test_trust_half(self):
        scorer = _scorer(trust_weight=1.0)
        # base=0.8, trust=0.5 → 0.8 * 0.5 = 0.4
        assert scorer._apply_trust(0.8, 0.5) == pytest.approx(0.4)

    def test_trust_weight_zero_ignores_trust(self):
        scorer = _scorer(trust_weight=0.0)
        # trust_weight=0 → multiplier=1.0 always
        assert scorer._apply_trust(0.8, 0.0) == pytest.approx(0.8)

    def test_trust_weight_half(self):
        scorer = _scorer(trust_weight=0.5)
        # multiplier = 0.5 + 0.5 * 0.2 = 0.6 → 0.8 * 0.6 = 0.48
        assert scorer._apply_trust(0.8, 0.2) == pytest.approx(0.48)

    def test_result_clamped_upper(self):
        scorer = _scorer()
        # base=1.0, trust=1.5（クランプされる）
        assert scorer._apply_trust(1.0, 1.5) <= 1.0

    def test_result_clamped_lower(self):
        scorer = _scorer()
        assert scorer._apply_trust(0.0, 0.0) >= 0.0


# ===========================================================================
# compute_for_document with trust_score
# ===========================================================================

class TestComputeForDocumentTrust:
    _now = datetime(2026, 1, 1, tzinfo=UTC)

    def test_default_trust_equals_base(self):
        scorer = _scorer()
        doc = _make_doc()
        score_default = scorer.compute_for_document(doc, now=self._now)
        score_trust1 = scorer.compute_for_document(doc, now=self._now, trust_score=1.0)
        assert score_default == pytest.approx(score_trust1)

    def test_low_trust_reduces_score(self):
        scorer = _scorer()
        doc = _make_doc(teacher_id="bad-model")
        score_high = scorer.compute_for_document(doc, now=self._now, trust_score=1.0)
        score_low = scorer.compute_for_document(doc, now=self._now, trust_score=0.1)
        assert score_low < score_high

    def test_min_trust_score_reflects_min(self):
        """_MIN_TRUST (0.05) を渡したとき、スコアが元の 5% になること。"""
        from src.memory.teacher_registry import _MIN_TRUST
        scorer = _scorer(trust_weight=1.0)
        doc = _make_doc()
        base = scorer.compute_for_document(doc, now=self._now, trust_score=1.0)
        low = scorer.compute_for_document(doc, now=self._now, trust_score=_MIN_TRUST)
        assert low == pytest.approx(base * _MIN_TRUST, abs=1e-6)

    def test_no_teacher_id_uses_default_trust(self):
        scorer = _scorer()
        doc = _make_doc()  # teacher_id なし
        score = scorer.compute_for_document(doc, now=self._now, trust_score=1.0)
        assert 0.0 <= score <= 1.0


# ===========================================================================
# compute_from_row with trust_score
# ===========================================================================

class TestComputeFromRowTrust:
    _now = datetime(2026, 1, 1, tzinfo=UTC)

    def _row(self, teacher_id: str | None = None) -> dict:
        return {
            "domain": "general",
            "source_retrieved_at": "2026-01-01T00:00:00",
            "retrieval_count": 10,
            "selection_count": 4,
            "positive_feedback": 8,
            "negative_feedback": 2,
            "teacher_quality": 0.8,
            "execution_success_rate": 0.9,
            "teacher_id": teacher_id,
        }

    def test_default_trust(self):
        scorer = _scorer()
        row = self._row()
        score = scorer.compute_from_row(row, now=self._now)
        assert 0.0 <= score <= 1.0

    def test_low_trust_reduces_score(self):
        scorer = _scorer()
        row = self._row(teacher_id="bad-model")
        score_high = scorer.compute_from_row(row, now=self._now, trust_score=1.0)
        score_low = scorer.compute_from_row(row, now=self._now, trust_score=0.1)
        assert score_low < score_high


# ===========================================================================
# build_trust_map
# ===========================================================================

@dataclass
class _FakeProfile:
    teacher_id: str
    trust_score: float


class TestBuildTrustMap:
    def test_basic(self):
        from src.memory.scoring.composite_scorer import CompositeScorer
        profiles = [
            _FakeProfile("model-a", 0.9),
            _FakeProfile("model-b", 0.3),
        ]
        trust_map = CompositeScorer.build_trust_map(profiles)
        assert trust_map["model-a"] == pytest.approx(0.9)
        assert trust_map["model-b"] == pytest.approx(0.3)

    def test_empty(self):
        from src.memory.scoring.composite_scorer import CompositeScorer
        assert CompositeScorer.build_trust_map([]) == {}


# ===========================================================================
# update_store with trust_map (統合テスト)
# ===========================================================================

class TestUpdateStoreTrustMap:
    def _make_store(self, tmp_path: Path):
        from src.memory.metadata_store import MetadataStore
        store = MetadataStore(db_path=str(tmp_path / "test.db"))
        return store

    def test_update_store_no_trust_map(self, tmp_path: Path):
        """trust_map なしでも update_store が正常に動作すること。"""
        from src.memory.scoring.composite_scorer import CompositeScorer

        async def run():
            store = self._make_store(tmp_path)
            await store.initialize()
            doc = _make_doc("test content", teacher_id="claude-opus-4-6")
            await store.save(doc)
            scorer = CompositeScorer()
            count = await scorer.update_store(store)
            result = await store.get(doc.id)
            await store.close()
            return count, result

        count, doc = asyncio.get_event_loop().run_until_complete(run())
        assert count == 1
        assert 0.0 <= doc.usefulness.composite <= 1.0

    def test_low_trust_teacher_gets_lower_score(self, tmp_path: Path):
        """低信頼 Teacher のドキュメントが高信頼より低いスコアになること。"""
        from src.memory.scoring.composite_scorer import CompositeScorer

        async def run():
            store = self._make_store(tmp_path)
            await store.initialize()

            doc_good = _make_doc("good content", teacher_id="trusted-model")
            doc_bad = _make_doc("bad content", teacher_id="bad-model")
            await store.save(doc_good)
            await store.save(doc_bad)

            trust_map = {"trusted-model": 1.0, "bad-model": 0.1}
            scorer = CompositeScorer()
            await scorer.update_store(store, trust_map=trust_map)

            good = await store.get(doc_good.id)
            bad = await store.get(doc_bad.id)
            await store.close()
            return good.usefulness.composite, bad.usefulness.composite

        good_score, bad_score = asyncio.get_event_loop().run_until_complete(run())
        assert good_score > bad_score

    def test_unknown_teacher_gets_full_trust(self, tmp_path: Path):
        """trust_map に存在しない teacher_id は trust_score=1.0 として扱う。"""
        from src.memory.scoring.composite_scorer import CompositeScorer

        async def run():
            store = self._make_store(tmp_path)
            await store.initialize()

            doc_known = _make_doc("known", teacher_id="known-model")
            doc_unknown = _make_doc("unknown", teacher_id="unknown-model")
            await store.save(doc_known)
            await store.save(doc_unknown)

            trust_map = {"known-model": 1.0}  # unknown-model は未登録
            scorer = CompositeScorer()
            await scorer.update_store(store, trust_map=trust_map)

            known = await store.get(doc_known.id)
            unknown = await store.get(doc_unknown.id)
            await store.close()
            return known.usefulness.composite, unknown.usefulness.composite

        known_score, unknown_score = asyncio.get_event_loop().run_until_complete(run())
        # 同じ条件なので同じスコアになるはず
        assert known_score == pytest.approx(unknown_score, abs=1e-6)

    def test_no_teacher_id_unaffected_by_trust_map(self, tmp_path: Path):
        """teacher_id が NULL のドキュメントは trust_map の影響を受けない。"""
        from src.memory.scoring.composite_scorer import CompositeScorer

        async def run():
            store = self._make_store(tmp_path)
            await store.initialize()

            doc_no_teacher = _make_doc("no teacher")
            await store.save(doc_no_teacher)

            trust_map = {"some-model": 0.1}
            scorer = CompositeScorer()
            await scorer.update_store(store, trust_map=trust_map)

            result = await store.get(doc_no_teacher.id)
            # trust_map なしでの計算と同じになるはず
            scorer2 = CompositeScorer()
            expected = scorer2.compute_for_document(result)
            await store.close()
            return result.usefulness.composite, expected

        actual, expected = asyncio.get_event_loop().run_until_complete(run())
        assert actual == pytest.approx(expected, abs=1e-4)

    def test_update_store_specific_doc_ids(self, tmp_path: Path):
        """doc_ids 指定で部分更新できること。"""
        from src.memory.scoring.composite_scorer import CompositeScorer

        async def run():
            store = self._make_store(tmp_path)
            await store.initialize()
            doc1 = _make_doc("doc1", teacher_id="model-a")
            doc2 = _make_doc("doc2", teacher_id="model-b")
            await store.save(doc1)
            await store.save(doc2)

            trust_map = {"model-a": 0.2, "model-b": 1.0}
            scorer = CompositeScorer()
            count = await scorer.update_store(store, doc_ids=[doc1.id], trust_map=trust_map)
            await store.close()
            return count

        assert asyncio.get_event_loop().run_until_complete(run()) == 1


# ===========================================================================
# teacher_trust_weight パラメータ
# ===========================================================================

class TestTeacherTrustWeight:
    _now = datetime(2026, 1, 1, tzinfo=UTC)

    def test_weight_0_trust_has_no_effect(self):
        scorer = _scorer(trust_weight=0.0)
        doc = _make_doc(teacher_id="bad-model")
        score_trust0 = scorer.compute_for_document(doc, now=self._now, trust_score=0.0)
        score_trust1 = scorer.compute_for_document(doc, now=self._now, trust_score=1.0)
        # trust_weight=0 なら trust_score は無視される
        assert score_trust0 == pytest.approx(score_trust1)

    def test_weight_1_full_trust_effect(self):
        scorer = _scorer(trust_weight=1.0)
        doc = _make_doc(teacher_id="model")
        base = scorer.compute_for_document(doc, now=self._now, trust_score=1.0)
        half = scorer.compute_for_document(doc, now=self._now, trust_score=0.5)
        assert half == pytest.approx(base * 0.5, abs=1e-6)

    def test_weight_clamped_to_valid_range(self):
        from src.memory.scoring.composite_scorer import CompositeScorer
        scorer_over = CompositeScorer(teacher_trust_weight=1.5)
        assert scorer_over._trust_weight == pytest.approx(1.0)
        scorer_under = CompositeScorer(teacher_trust_weight=-0.1)
        assert scorer_under._trust_weight == pytest.approx(0.0)
