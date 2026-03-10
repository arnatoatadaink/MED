"""src/training/rewards/hybrid.py — Hybrid 報酬関数

複数の RewardFunction を組み合わせて重み付け平均を計算する。
CompositeReward の汎用版として、任意の報酬を合成可能。

使い方:
    from src.training.rewards.hybrid import HybridReward

    hybrid = HybridReward([
        (code_exec_reward, 0.6),
        (teacher_eval_reward, 0.4),
    ])
    score = await hybrid.compute(prompt, response)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.training.base import RewardFunction
from src.training.registry import TrainingRegistry

logger = logging.getLogger(__name__)


@TrainingRegistry.reward("hybrid")
class HybridReward(RewardFunction):
    """複数報酬の重み付け合成。

    Args:
        components: [(reward_function, weight), ...] のリスト。
            weights の合計は 1.0 になること。
    """

    def __init__(
        self,
        components: list[tuple[RewardFunction, float]] | None = None,
    ) -> None:
        self._components = components or []
        if self._components:
            total = sum(w for _, w in self._components)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"HybridReward weights must sum to 1.0, got {total:.4f}")

    @property
    def name(self) -> str:
        return "hybrid"

    def add_component(self, reward_fn: RewardFunction, weight: float) -> None:
        """報酬コンポーネントを追加する。"""
        self._components.append((reward_fn, weight))

    async def compute(
        self,
        prompt: str,
        response: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> float:
        """全コンポーネントの加重平均を計算する。"""
        if not self._components:
            return 0.5

        import asyncio
        scores = await asyncio.gather(*[
            fn.compute(prompt, response, metadata)
            for fn, _ in self._components
        ])

        total = sum(score * weight for score, (_, weight) in zip(scores, self._components))
        logger.debug(
            "HybridReward: %s → %.3f",
            [f"{fn.name}={s:.2f}" for s, (fn, _) in zip(scores, self._components)],
            total,
        )
        return total
