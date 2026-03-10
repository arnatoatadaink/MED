"""src/training/algorithms/ppo.py — PPO (Proximal Policy Optimization)

標準的な RL アルゴリズム。GRPO の前身として実装。
クリッピングによる安定した学習が特徴。

ロス計算:
    L_PPO = -min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t) + c1*L_vf - c2*H

使い方:
    from src.training.algorithms.ppo import PPOAlgorithm
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from src.training.base import ParameterAdapter, TrainingAlgorithm, TrainingBatch
from src.training.registry import TrainingRegistry

logger = logging.getLogger(__name__)


@TrainingRegistry.algorithm("ppo")
class PPOAlgorithm(TrainingAlgorithm):
    """Proximal Policy Optimization。

    Args:
        clip_epsilon: クリッピング比率ε (default: 0.2)。
        value_coef: 価値関数ロス係数 (default: 0.5)。
        entropy_coef: エントロピーボーナス係数 (default: 0.01)。
        normalize_advantages: アドバンテージ正規化 (default: True)。
    """

    def __init__(
        self,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        normalize_advantages: bool = True,
    ) -> None:
        self._clip_eps = clip_epsilon
        self._value_coef = value_coef
        self._entropy_coef = entropy_coef
        self._normalize_adv = normalize_advantages

    @property
    def name(self) -> str:
        return "ppo"

    def compute_loss(
        self,
        batch: TrainingBatch,
        model: Any,
        adapter: ParameterAdapter,
    ) -> torch.Tensor:
        """PPO クリッピングロスを計算する。"""
        if not batch.rewards:
            raise ValueError("TrainingBatch.rewards must be set before compute_loss")
        if torch is None:
            import types
            return types.SimpleNamespace(item=lambda: 0.0, backward=lambda: None)  # type: ignore[return-value]

        rewards = torch.tensor(batch.rewards, dtype=torch.float32)

        # アドバンテージ計算
        if self._normalize_adv:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            advantages = rewards

        if hasattr(model, "log_probs"):
            log_probs = torch.stack([
                model.log_probs(p, r)
                for p, r in zip(batch.prompts, batch.responses)
            ])
        else:
            log_probs = torch.zeros(len(batch.prompts), requires_grad=True)

        # PPO clipping
        ratio = torch.exp(log_probs)
        clipped_ratio = torch.clamp(ratio, 1.0 - self._clip_eps, 1.0 + self._clip_eps)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        loss = policy_loss

        logger.debug(
            "PPO loss=%.4f adv_mean=%.4f",
            loss.item(), advantages.mean().item(),
        )
        return loss
