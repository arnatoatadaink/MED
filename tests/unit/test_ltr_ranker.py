"""tests/unit/test_ltr_ranker.py — LTRRanker の単体テスト"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.memory.learning.ltr_ranker import N_FEATURES, LTRRanker, RankFeatures

# ──────────────────────────────────────────────
# RankFeatures
# ──────────────────────────────────────────────


class TestRankFeatures:
    def test_default_values(self) -> None:
        feat = RankFeatures()
        assert feat.faiss_score == 0.0
        assert feat.freshness == 0.5
        assert feat.confidence == 0.5

    def test_to_vector_shape(self) -> None:
        feat = RankFeatures(faiss_score=0.9, freshness=0.8, usefulness=0.7)
        vec = feat.to_vector()
        assert vec.shape == (N_FEATURES,)
        assert vec.dtype == np.float32

    def test_negative_faiss_score_clamped(self) -> None:
        feat = RankFeatures(faiss_score=-0.5)
        vec = feat.to_vector()
        assert vec[0] == 0.0

    def test_from_dict(self) -> None:
        data = {
            "faiss_score": 0.8,
            "freshness": 0.9,
            "usefulness": 0.7,
            "teacher_quality": 0.6,
            "selection_rate": 0.5,
            "confidence": 0.8,
        }
        feat = RankFeatures.from_dict(data)
        assert feat.faiss_score == 0.8
        assert feat.freshness == 0.9

    def test_from_dict_missing_keys(self) -> None:
        feat = RankFeatures.from_dict({})
        assert feat.faiss_score == 0.0
        assert feat.freshness == 0.5


# ──────────────────────────────────────────────
# LTRRanker 初期化
# ──────────────────────────────────────────────


class TestLTRRankerInit:
    def test_default_weights_sum_to_one(self) -> None:
        ranker = LTRRanker()
        assert abs(ranker.weights.sum() - 1.0) < 1e-6

    def test_custom_initial_weights(self) -> None:
        weights = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05], dtype=np.float32)
        ranker = LTRRanker(initial_weights=weights)
        np.testing.assert_array_almost_equal(ranker.weights, weights)

    def test_invalid_weight_length_raises(self) -> None:
        with pytest.raises(ValueError, match="length must be"):
            LTRRanker(initial_weights=np.array([0.5, 0.5]))

    def test_initial_update_count_zero(self) -> None:
        ranker = LTRRanker()
        assert ranker.update_count == 0


# ──────────────────────────────────────────────
# スコア計算
# ──────────────────────────────────────────────


class TestLTRRankerScore:
    def test_score_range(self) -> None:
        ranker = LTRRanker()
        feat = RankFeatures(faiss_score=0.8, freshness=0.9, usefulness=0.7)
        score = ranker.score(feat)
        assert 0.0 <= score <= 1.0

    def test_higher_features_higher_score(self) -> None:
        ranker = LTRRanker()
        high = RankFeatures(faiss_score=1.0, freshness=1.0, usefulness=1.0,
                            teacher_quality=1.0, selection_rate=1.0, confidence=1.0)
        low = RankFeatures(faiss_score=0.0, freshness=0.0, usefulness=0.0,
                           teacher_quality=0.0, selection_rate=0.0, confidence=0.0)
        assert ranker.score(high) > ranker.score(low)

    def test_score_batch(self) -> None:
        ranker = LTRRanker()
        feats = [RankFeatures(faiss_score=float(i) / 10) for i in range(5)]
        scores = ranker.score_batch(feats)
        assert len(scores) == 5
        # スコアは faiss_score に従って単調増加
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1]


# ──────────────────────────────────────────────
# オンライン学習
# ──────────────────────────────────────────────


class TestLTRRankerUpdate:
    def test_positive_reward_increases_score(self) -> None:
        ranker = LTRRanker(learning_rate=0.1)
        feat = RankFeatures(faiss_score=0.8, freshness=0.9)
        before = ranker.score(feat)
        for _ in range(10):
            ranker.update(feat, reward=1.0)
        after = ranker.score(feat)
        assert after > before

    def test_negative_reward_decreases_score(self) -> None:
        ranker = LTRRanker(learning_rate=0.1)
        feat = RankFeatures(faiss_score=0.8, freshness=0.9)
        before = ranker.score(feat)
        for _ in range(10):
            ranker.update(feat, reward=-1.0)
        after = ranker.score(feat)
        assert after < before

    def test_update_count_increments(self) -> None:
        ranker = LTRRanker()
        feat = RankFeatures()
        ranker.update(feat, reward=1.0)
        ranker.update(feat, reward=1.0)
        assert ranker.update_count == 2

    def test_update_returns_delta(self) -> None:
        ranker = LTRRanker(learning_rate=0.1)
        feat = RankFeatures(faiss_score=0.5)
        delta = ranker.update(feat, reward=1.0)
        assert isinstance(delta, float)


# ──────────────────────────────────────────────
# リランク
# ──────────────────────────────────────────────


class TestLTRRankerRerank:
    def test_rerank_sorted(self) -> None:
        ranker = LTRRanker()
        items = [
            ("doc_low", RankFeatures(faiss_score=0.1, freshness=0.1)),
            ("doc_high", RankFeatures(faiss_score=0.9, freshness=0.9)),
            ("doc_mid", RankFeatures(faiss_score=0.5, freshness=0.5)),
        ]
        result = ranker.rerank(items)
        assert result[0][0] == "doc_high"
        assert result[-1][0] == "doc_low"

    def test_rerank_returns_all(self) -> None:
        ranker = LTRRanker()
        items = [(f"doc_{i}", RankFeatures()) for i in range(5)]
        result = ranker.rerank(items)
        assert len(result) == 5

    def test_rerank_scores_are_floats(self) -> None:
        ranker = LTRRanker()
        items = [("doc", RankFeatures(faiss_score=0.7))]
        result = ranker.rerank(items)
        assert isinstance(result[0][1], float)


# ──────────────────────────────────────────────
# 永続化
# ──────────────────────────────────────────────


class TestLTRRankerPersistence:
    def test_save_and_load(self, tmp_path: Path) -> None:
        ranker = LTRRanker(learning_rate=0.05)
        feat = RankFeatures(faiss_score=0.9)
        ranker.update(feat, reward=1.0)
        ranker.update(feat, reward=0.5)

        path = tmp_path / "ltr.pkl"
        ranker.save(path)

        loaded = LTRRanker.load(path)
        np.testing.assert_array_almost_equal(ranker.weights, loaded.weights)
        assert loaded.update_count == 2

    def test_load_produces_same_scores(self, tmp_path: Path) -> None:
        ranker = LTRRanker()
        ranker.update(RankFeatures(faiss_score=0.7), reward=1.0)

        path = tmp_path / "ltr_score.pkl"
        ranker.save(path)
        loaded = LTRRanker.load(path)

        feat = RankFeatures(faiss_score=0.5, freshness=0.8)
        assert abs(ranker.score(feat) - loaded.score(feat)) < 1e-6

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        ranker = LTRRanker()
        path = tmp_path / "nested" / "dir" / "ltr.pkl"
        ranker.save(path)
        assert path.exists()
