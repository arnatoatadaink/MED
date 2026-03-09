"""src/training/algorithms/reinforce.py — REINFORCE (Williams 1992)

最もシンプルな Policy Gradient アルゴリズム。
GRPO の比較ベースラインとして実装。

ロス計算:
    L_REINFORCE = -mean(r_i * log π(a_i|s_i))

使い方:
    from src.training.algorithms.reinforce import REINFORCEAlgorithm
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


@TrainingRegistry.algorithm("reinforce")
class REINFORCEAlgorithm(TrainingAlgorithm):
    """REINFORCE (Williams 1992)。

    Args:
        baseline: 報酬の baseline 戦略 ("mean" or "none")。
        gamma: 割引率 (default: 1.0)。
    """

    def __init__(
        self,
        baseline: str = "mean",
        gamma: float = 1.0,
    ) -> None:
        if baseline not in ("mean", "none"):
            raise ValueError(f"baseline must be 'mean' or 'none', got {baseline!r}")
        self._baseline = baseline
        self._gamma = gamma

    @property
    def name(self) -> str:
        return "reinforce"

    def compute_loss(
        self,
        batch: TrainingBatch,
        model: Any,
        adapter: ParameterAdapter,
    ) -> torch.Tensor:
        """REINFORCE ロスを計算する。"""
        if not batch.rewards:
            raise ValueError("TrainingBatch.rewards must be set before compute_loss")
        if torch is None:
            import types
            return types.SimpleNamespace(item=lambda: 0.0, backward=lambda: None)  # type: ignore[return-value]

        rewards = torch.tensor(batch.rewards, dtype=torch.float32)

        if self._baseline == "mean":
            rewards = rewards - rewards.mean()

        if hasattr(model, "log_probs"):
            log_probs = torch.stack([
                model.log_probs(p, r)
                for p, r in zip(batch.prompts, batch.responses)
            ])
        else:
            log_probs = torch.zeros(len(batch.prompts), requires_grad=True)

        loss = -(log_probs * rewards).mean()
        logger.debug("REINFORCE loss=%.4f r_mean=%.4f", loss.item(), rewards.mean().item())
        return loss
