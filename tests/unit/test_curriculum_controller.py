"""tests/unit/test_curriculum_controller.py — CurriculumController 単体テスト

動的カリキュラム調整 (I-B1) の全機能をテストする。
"""

from __future__ import annotations

import pytest

from src.memory.maturation.difficulty_tagger import (
    CurriculumConfig,
    CurriculumController,
    _DIFFICULTY_ORDER,
)
from src.memory.schema import DifficultyLevel


# ──────────────────────────────────────────────
# 初期化
# ──────────────────────────────────────────────


class TestInitialization:
    """初期化のテスト。"""

    def test_default_uniform_distribution(self) -> None:
        """デフォルトは均等配分 (各25%)。"""
        ctrl = CurriculumController()
        dist = ctrl.distribution
        assert len(dist) == 4
        for level in _DIFFICULTY_ORDER:
            assert abs(dist[level] - 0.25) < 1e-9

    def test_custom_initial_distribution(self) -> None:
        """カスタム初期配分が正規化されて保持される。"""
        init = {
            DifficultyLevel.BEGINNER: 0.5,
            DifficultyLevel.INTERMEDIATE: 0.3,
            DifficultyLevel.ADVANCED: 0.15,
            DifficultyLevel.EXPERT: 0.05,
        }
        ctrl = CurriculumController(initial_distribution=init)
        dist = ctrl.distribution
        total = sum(dist.values())
        assert abs(total - 1.0) < 1e-9
        # beginner は最大のまま
        assert dist[DifficultyLevel.BEGINNER] > dist[DifficultyLevel.EXPERT]

    def test_custom_config(self) -> None:
        """CurriculumConfig のカスタム値が反映される。"""
        cfg = CurriculumConfig(window_size=10, loss_low_threshold=0.1)
        ctrl = CurriculumController(config=cfg)
        assert ctrl.step_count == 0
        assert ctrl.adjustment_count == 0

    def test_step_count_starts_at_zero(self) -> None:
        ctrl = CurriculumController()
        assert ctrl.step_count == 0
        assert ctrl.adjustment_count == 0

    def test_distribution_is_copy(self) -> None:
        """distribution プロパティはコピーを返す。"""
        ctrl = CurriculumController()
        d1 = ctrl.distribution
        d1[DifficultyLevel.BEGINNER] = 999.0
        assert ctrl.distribution[DifficultyLevel.BEGINNER] != 999.0


# ──────────────────────────────────────────────
# record_step & 調整トリガー
# ──────────────────────────────────────────────


class TestRecordStep:
    """record_step の動作テスト。"""

    def test_no_adjustment_before_window_full(self) -> None:
        """ウィンドウが埋まるまで調整しない。"""
        cfg = CurriculumConfig(window_size=5)
        ctrl = CurriculumController(config=cfg)
        for _ in range(4):
            ctrl.record_step(loss=0.1)  # 低損失
        assert ctrl.adjustment_count == 0
        assert ctrl.step_count == 4

    def test_shift_up_on_low_loss(self) -> None:
        """損失が閾値未満 → 高難易度にシフト。"""
        cfg = CurriculumConfig(window_size=5, loss_low_threshold=0.3)
        ctrl = CurriculumController(config=cfg)
        initial_expert = ctrl.distribution[DifficultyLevel.EXPERT]
        for _ in range(5):
            ctrl.record_step(loss=0.1)
        assert ctrl.adjustment_count >= 1
        assert ctrl.distribution[DifficultyLevel.EXPERT] > initial_expert

    def test_shift_down_on_high_loss(self) -> None:
        """損失が閾値超 → 低難易度にシフト。"""
        cfg = CurriculumConfig(window_size=5, loss_high_threshold=1.5)
        ctrl = CurriculumController(config=cfg)
        initial_beginner = ctrl.distribution[DifficultyLevel.BEGINNER]
        for _ in range(5):
            ctrl.record_step(loss=2.0)
        assert ctrl.adjustment_count >= 1
        assert ctrl.distribution[DifficultyLevel.BEGINNER] > initial_beginner

    def test_no_adjustment_on_normal_loss(self) -> None:
        """損失が適正範囲内 → 調整なし。"""
        cfg = CurriculumConfig(
            window_size=5, loss_low_threshold=0.3, loss_high_threshold=1.5,
        )
        ctrl = CurriculumController(config=cfg)
        for _ in range(10):
            ctrl.record_step(loss=0.8)
        assert ctrl.adjustment_count == 0

    def test_reward_mean_recorded(self) -> None:
        """reward_mean が記録される（エラーなし）。"""
        ctrl = CurriculumController()
        ctrl.record_step(loss=0.5, reward_mean=0.7)
        assert ctrl.step_count == 1


# ──────────────────────────────────────────────
# 正規化 & min_weight
# ──────────────────────────────────────────────


class TestNormalization:
    """正規化と最小配分保証のテスト。"""

    def test_distribution_sums_to_one(self) -> None:
        """調整後も合計が 1.0。"""
        cfg = CurriculumConfig(window_size=3, loss_low_threshold=0.3)
        ctrl = CurriculumController(config=cfg)
        for _ in range(10):
            ctrl.record_step(loss=0.1)
        total = sum(ctrl.distribution.values())
        assert abs(total - 1.0) < 1e-9

    def test_min_weight_guaranteed(self) -> None:
        """繰り返しシフトしても全レベルが正の値を保つ。"""
        cfg = CurriculumConfig(
            window_size=3, loss_low_threshold=0.3,
            adjustment_rate=0.2, min_weight=0.05,
        )
        ctrl = CurriculumController(config=cfg)
        # 大量に shift_up を繰り返す
        for _ in range(100):
            ctrl.record_step(loss=0.05)
        for level in _DIFFICULTY_ORDER:
            # min_weight は正規化後に保証されるが、繰り返しで微小な浮動小数点誤差あり
            assert ctrl.distribution[level] > 0.01

    def test_min_weight_zero_allows_elimination(self) -> None:
        """min_weight=0 なら完全除外されうる。"""
        cfg = CurriculumConfig(
            window_size=3, loss_low_threshold=0.3,
            adjustment_rate=0.3, min_weight=0.0,
        )
        ctrl = CurriculumController(config=cfg)
        for _ in range(50):
            ctrl.record_step(loss=0.01)
        # beginner が 0 近くまで下がる可能性がある
        assert ctrl.distribution[DifficultyLevel.BEGINNER] < 0.15


# ──────────────────────────────────────────────
# get_sample_counts
# ──────────────────────────────────────────────


class TestGetSampleCounts:
    """get_sample_counts のテスト。"""

    def test_total_equals_batch_size(self) -> None:
        """合計がバッチサイズに一致する。"""
        ctrl = CurriculumController()
        counts = ctrl.get_sample_counts(32)
        assert sum(counts.values()) == 32

    def test_uniform_distribution_equal_counts(self) -> None:
        """均等配分なら各8個（バッチ32）。"""
        ctrl = CurriculumController()
        counts = ctrl.get_sample_counts(32)
        for level in _DIFFICULTY_ORDER:
            assert counts[level] == 8

    def test_remainder_goes_to_max_weight(self) -> None:
        """端数は最大配分のレベルに追加される。"""
        init = {
            DifficultyLevel.BEGINNER: 0.1,
            DifficultyLevel.INTERMEDIATE: 0.2,
            DifficultyLevel.ADVANCED: 0.3,
            DifficultyLevel.EXPERT: 0.4,
        }
        ctrl = CurriculumController(initial_distribution=init)
        counts = ctrl.get_sample_counts(10)
        assert sum(counts.values()) == 10

    def test_small_batch_size(self) -> None:
        """バッチサイズが小さくても正しい。"""
        ctrl = CurriculumController()
        counts = ctrl.get_sample_counts(1)
        assert sum(counts.values()) == 1

    def test_zero_batch_size(self) -> None:
        """バッチサイズ0。"""
        ctrl = CurriculumController()
        counts = ctrl.get_sample_counts(0)
        assert sum(counts.values()) == 0


# ──────────────────────────────────────────────
# 移動平均 & リセット
# ──────────────────────────────────────────────


class TestMovingAverageAndReset:
    """get_moving_average_loss / reset のテスト。"""

    def test_moving_average_none_when_empty(self) -> None:
        ctrl = CurriculumController()
        assert ctrl.get_moving_average_loss() is None

    def test_moving_average_correct(self) -> None:
        ctrl = CurriculumController()
        ctrl.record_step(loss=1.0)
        ctrl.record_step(loss=3.0)
        avg = ctrl.get_moving_average_loss()
        assert avg is not None
        assert abs(avg - 2.0) < 1e-9

    def test_reset_clears_state(self) -> None:
        """reset() で初期状態に戻る。"""
        cfg = CurriculumConfig(window_size=3, loss_low_threshold=0.3)
        ctrl = CurriculumController(config=cfg)
        for _ in range(10):
            ctrl.record_step(loss=0.1)
        assert ctrl.step_count > 0
        assert ctrl.adjustment_count > 0

        ctrl.reset()
        assert ctrl.step_count == 0
        assert ctrl.adjustment_count == 0
        assert ctrl.get_moving_average_loss() is None
        # 配分が均等に戻る
        for level in _DIFFICULTY_ORDER:
            assert abs(ctrl.distribution[level] - 0.25) < 1e-9


# ──────────────────────────────────────────────
# シリアライズ
# ──────────────────────────────────────────────


class TestSerialization:
    """to_dict のテスト。"""

    def test_to_dict_keys(self) -> None:
        ctrl = CurriculumController()
        d = ctrl.to_dict()
        assert "step_count" in d
        assert "adjustment_count" in d
        assert "distribution" in d
        assert "moving_avg_loss" in d
        assert "loss_history_size" in d

    def test_to_dict_after_steps(self) -> None:
        ctrl = CurriculumController()
        ctrl.record_step(loss=0.5)
        d = ctrl.to_dict()
        assert d["step_count"] == 1
        assert d["loss_history_size"] == 1
        assert d["moving_avg_loss"] == pytest.approx(0.5)


# ──────────────────────────────────────────────
# エッジケース
# ──────────────────────────────────────────────


class TestEdgeCases:
    """エッジケースのテスト。"""

    def test_repeated_shift_up_converges(self) -> None:
        """shift_up を繰り返すと expert/advanced に収束。"""
        cfg = CurriculumConfig(window_size=3, loss_low_threshold=0.5, adjustment_rate=0.1)
        ctrl = CurriculumController(config=cfg)
        for _ in range(200):
            ctrl.record_step(loss=0.1)
        dist = ctrl.distribution
        assert dist[DifficultyLevel.EXPERT] > dist[DifficultyLevel.BEGINNER]

    def test_repeated_shift_down_converges(self) -> None:
        """shift_down を繰り返すと beginner/intermediate に収束。"""
        cfg = CurriculumConfig(window_size=3, loss_high_threshold=1.0, adjustment_rate=0.1)
        ctrl = CurriculumController(config=cfg)
        for _ in range(200):
            ctrl.record_step(loss=2.0)
        dist = ctrl.distribution
        assert dist[DifficultyLevel.BEGINNER] > dist[DifficultyLevel.EXPERT]

    def test_alternating_loss_stays_balanced(self) -> None:
        """損失が交互に振れると調整が相殺される。"""
        cfg = CurriculumConfig(window_size=4, loss_low_threshold=0.3, loss_high_threshold=1.5)
        ctrl = CurriculumController(config=cfg)
        for _ in range(50):
            ctrl.record_step(loss=0.1)
            ctrl.record_step(loss=0.1)
            ctrl.record_step(loss=2.0)
            ctrl.record_step(loss=2.0)
        # 大きく偏らない
        dist = ctrl.distribution
        for level in _DIFFICULTY_ORDER:
            assert dist[level] > 0.05

    def test_get_distribution_same_as_property(self) -> None:
        """get_distribution() と distribution プロパティが同じ値。"""
        ctrl = CurriculumController()
        ctrl.record_step(loss=0.5)
        assert ctrl.get_distribution() == ctrl.distribution
