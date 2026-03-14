"""src/training/registry.py — 学習コンポーネント Registry

デコレータ登録式の Registry パターンで学習アルゴリズム・アダプタ・報酬関数を管理する。

使い方:
    # 登録
    from src.training.registry import TrainingRegistry

    @TrainingRegistry.algorithm("grpo")
    class GRPOAlgorithm(TrainingAlgorithm):
        ...

    @TrainingRegistry.adapter("tinylora")
    class TinyLoRAAdapter(ParameterAdapter):
        ...

    @TrainingRegistry.reward("composite")
    class CompositeReward(RewardFunction):
        ...

    # 取得
    algo_cls = TrainingRegistry.get_algorithm("grpo")
    adapter_cls = TrainingRegistry.get_adapter("tinylora")
    reward_cls = TrainingRegistry.get_reward("composite")
"""

from __future__ import annotations

import logging

from src.training.base import ParameterAdapter, RewardFunction, TrainingAlgorithm

logger = logging.getLogger(__name__)


class TrainingRegistry:
    """学習コンポーネントの Registry。

    クラスメソッドのみで構成されるシングルトン的な Registry。
    """

    _algorithms: dict[str, type[TrainingAlgorithm]] = {}
    _adapters: dict[str, type[ParameterAdapter]] = {}
    _rewards: dict[str, type[RewardFunction]] = {}

    # ── 登録デコレータ ──────────────────────────

    @classmethod
    def algorithm(cls, name: str):
        """TrainingAlgorithm の登録デコレータ。

        Usage:
            @TrainingRegistry.algorithm("grpo")
            class GRPOAlgorithm(TrainingAlgorithm): ...
        """
        def decorator(klass: type[TrainingAlgorithm]) -> type[TrainingAlgorithm]:
            if name in cls._algorithms:
                logger.warning("Algorithm '%s' overwritten in registry", name)
            cls._algorithms[name] = klass
            logger.debug("Registered algorithm: %s → %s", name, klass.__name__)
            return klass
        return decorator

    @classmethod
    def adapter(cls, name: str):
        """ParameterAdapter の登録デコレータ。"""
        def decorator(klass: type[ParameterAdapter]) -> type[ParameterAdapter]:
            if name in cls._adapters:
                logger.warning("Adapter '%s' overwritten in registry", name)
            cls._adapters[name] = klass
            logger.debug("Registered adapter: %s → %s", name, klass.__name__)
            return klass
        return decorator

    @classmethod
    def reward(cls, name: str):
        """RewardFunction の登録デコレータ。"""
        def decorator(klass: type[RewardFunction]) -> type[RewardFunction]:
            if name in cls._rewards:
                logger.warning("Reward '%s' overwritten in registry", name)
            cls._rewards[name] = klass
            logger.debug("Registered reward: %s → %s", name, klass.__name__)
            return klass
        return decorator

    # ── 取得メソッド ──────────────────────────

    @classmethod
    def get_algorithm(cls, name: str) -> type[TrainingAlgorithm]:
        """登録済みアルゴリズムクラスを取得する。

        Raises:
            KeyError: 未登録の名前が指定された場合。
        """
        if name not in cls._algorithms:
            available = sorted(cls._algorithms.keys())
            raise KeyError(
                f"Algorithm '{name}' not found. Available: {available}"
            )
        return cls._algorithms[name]

    @classmethod
    def get_adapter(cls, name: str) -> type[ParameterAdapter]:
        """登録済みアダプタクラスを取得する。"""
        if name not in cls._adapters:
            available = sorted(cls._adapters.keys())
            raise KeyError(
                f"Adapter '{name}' not found. Available: {available}"
            )
        return cls._adapters[name]

    @classmethod
    def get_reward(cls, name: str) -> type[RewardFunction]:
        """登録済み報酬関数クラスを取得する。"""
        if name not in cls._rewards:
            available = sorted(cls._rewards.keys())
            raise KeyError(
                f"Reward '{name}' not found. Available: {available}"
            )
        return cls._rewards[name]

    # ── 一覧メソッド ──────────────────────────

    @classmethod
    def list_algorithms(cls) -> list[str]:
        return sorted(cls._algorithms.keys())

    @classmethod
    def list_adapters(cls) -> list[str]:
        return sorted(cls._adapters.keys())

    @classmethod
    def list_rewards(cls) -> list[str]:
        return sorted(cls._rewards.keys())

    @classmethod
    def summary(cls) -> dict[str, list[str]]:
        """Registry 内容のサマリを返す。"""
        return {
            "algorithms": cls.list_algorithms(),
            "adapters": cls.list_adapters(),
            "rewards": cls.list_rewards(),
        }
