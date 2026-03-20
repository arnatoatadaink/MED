"""tests/unit/test_refuel.py — REFUELAlgorithm 単体テスト"""

from __future__ import annotations

import pytest

from src.training.algorithms.refuel import REFUELAlgorithm
from src.training.base import TrainingBatch
from src.training.registry import TrainingRegistry


class TestREFUELAlgorithmInit:
    def test_name(self):
        algo = REFUELAlgorithm()
        assert algo.name == "refuel"

    def test_registered_in_registry(self):
        cls = TrainingRegistry.get_algorithm("refuel")
        assert cls is REFUELAlgorithm

    def test_default_hyperparams(self):
        algo = REFUELAlgorithm()
        assert algo._margin > 0
        assert algo._temperature > 0
        assert algo._clip_value > 0

    def test_custom_hyperparams(self):
        algo = REFUELAlgorithm(margin=0.1, temperature=2.0, clip_value=5.0)
        assert algo._margin == 0.1
        assert algo._temperature == 2.0
        assert algo._clip_value == 5.0


class TestREFUELComputeLoss:
    """compute_loss のロジックを torch なし・あり両方でテスト。"""

    def _make_batch(self, rewards: list[float]) -> TrainingBatch:
        n = len(rewards)
        return TrainingBatch(
            prompts=[f"prompt_{i}" for i in range(n)],
            responses=[f"response_{i}" for i in range(n)],
            rewards=rewards,
        )

    def test_empty_rewards_raises(self):
        algo = REFUELAlgorithm()
        batch = TrainingBatch(prompts=[], responses=[], rewards=[])
        with pytest.raises((ValueError, Exception)):
            algo.compute_loss(batch, model=None, adapter=None)

    def test_single_sample_returns_zero_loss(self):
        algo = REFUELAlgorithm()
        batch = self._make_batch([1.0])
        result = algo.compute_loss(batch, model=None, adapter=None)
        # n=1 → n-1=0 < 2 → zero loss
        assert float(result.item()) == pytest.approx(0.0)

    def test_even_batch_processes(self):
        algo = REFUELAlgorithm(margin=0.0)
        batch = self._make_batch([1.0, 0.0, 0.8, 0.2])
        result = algo.compute_loss(batch, model=None, adapter=None)
        # Should return a scalar
        val = float(result.item())
        assert isinstance(val, float)

    def test_odd_batch_drops_last(self):
        algo = REFUELAlgorithm(margin=0.0)
        batch = self._make_batch([1.0, 0.0, 0.5])
        # n=3 → odd → last dropped → 2 samples → 1 pair
        result = algo.compute_loss(batch, model=None, adapter=None)
        val = float(result.item())
        assert isinstance(val, float)

    def test_margin_filters_all_small_pairs(self):
        # reward diff = 0.001, margin = 0.5 → all filtered → zero loss
        algo = REFUELAlgorithm(margin=0.5)
        batch = self._make_batch([0.501, 0.500])
        result = algo.compute_loss(batch, model=None, adapter=None)
        assert float(result.item()) == pytest.approx(0.0)

    def test_model_with_log_probs(self):
        import types
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

        algo = REFUELAlgorithm(margin=0.0)
        batch = self._make_batch([1.0, 0.0])

        mock_model = types.SimpleNamespace(
            log_probs=lambda p, r: torch.tensor(0.5, requires_grad=True),
        )
        result = algo.compute_loss(batch, model=mock_model, adapter=None)
        assert hasattr(result, "item")

    def test_all_identical_rewards_zero_loss_with_margin(self):
        algo = REFUELAlgorithm(margin=0.1)
        batch = self._make_batch([0.5, 0.5, 0.5, 0.5])
        result = algo.compute_loss(batch, model=None, adapter=None)
        assert float(result.item()) == pytest.approx(0.0)

    def test_large_reward_diff_clipped(self):
        # reward_diff = 100 / 1.0 = 100, clipped to clip_value=10
        algo = REFUELAlgorithm(margin=0.0, temperature=1.0, clip_value=10.0)
        batch = self._make_batch([100.0, 0.0])
        result = algo.compute_loss(batch, model=None, adapter=None)
        # With stub log_prob_diff=0, loss = mse(0, 10) = 100
        val = float(result.item())
        assert val == pytest.approx(100.0, rel=1e-3)

    def test_loss_is_non_negative(self):
        algo = REFUELAlgorithm(margin=0.0)
        batch = self._make_batch([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
        result = algo.compute_loss(batch, model=None, adapter=None)
        assert float(result.item()) >= 0.0


class TestREFUELPairing:
    """ペアリングロジックの境界値テスト"""

    def _make_batch(self, rewards: list[float]) -> TrainingBatch:
        n = len(rewards)
        return TrainingBatch(
            prompts=[f"p{i}" for i in range(n)],
            responses=[f"r{i}" for i in range(n)],
            rewards=rewards,
        )

    def test_two_samples_one_pair(self):
        algo = REFUELAlgorithm(margin=0.0)
        batch = self._make_batch([1.0, 0.0])
        result = algo.compute_loss(batch, model=None, adapter=None)
        # log_prob_diff=0, reward_diff=1.0/temp → loss = mse(0,1) = 1
        assert float(result.item()) == pytest.approx(1.0, rel=1e-3)

    def test_four_samples_two_pairs(self):
        algo = REFUELAlgorithm(margin=0.0)
        batch = self._make_batch([1.0, 0.0, 1.0, 0.0])
        result = algo.compute_loss(batch, model=None, adapter=None)
        # Both pairs have diff=1.0, loss = mean(mse(0,1), mse(0,1)) = 1.0
        assert float(result.item()) == pytest.approx(1.0, rel=1e-3)
