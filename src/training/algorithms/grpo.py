"""src/training/algorithms/grpo.py — GRPO (Group Relative Policy Optimization)

TinyLoRA 論文 (Morris et al., 2026) で採用された RL アルゴリズム。
Group 内の相対報酬でポリシー勾配を安定化し、小モデルでの効率的な学習を実現する。

GRPO の特徴:
- Group 内の平均報酬を baseline として使用（KL ペナルティ不要）
- 小モデル (7B 以下) での安定学習
- TinyLoRA との組み合わせで GSM8K 91% を実現（論文）

ロス計算:
    L_GRPO = -mean( (r_i - r_group_mean) / (r_group_std + eps) * log π(a_i|s_i) )

使い方:
    from src.training.algorithms.grpo import GRPOAlgorithm
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from src.training.base import ParameterAdapter, TrainingAlgorithm, TrainingBatch, TrainingConfig
from src.training.registry import TrainingRegistry

logger = logging.getLogger(__name__)


@TrainingRegistry.algorithm("grpo")
class GRPOAlgorithm(TrainingAlgorithm):
    """Group Relative Policy Optimization。

    Args:
        epsilon: 報酬正規化の安定項 (default: 1e-8)。
        clip_ratio: PPO スタイルの clipping 比率 (None で無効)。
        entropy_coef: エントロピーボーナス係数 (0 で無効)。
    """

    def __init__(
        self,
        epsilon: float = 1e-8,
        clip_ratio: float | None = None,
        entropy_coef: float = 0.01,
    ) -> None:
        self._epsilon = epsilon
        self._clip_ratio = clip_ratio
        self._entropy_coef = entropy_coef

    @property
    def name(self) -> str:
        return "grpo"

    def compute_loss(
        self,
        batch: TrainingBatch,
        model: Any,
        adapter: ParameterAdapter,
    ) -> torch.Tensor:
        """GRPO ロスを計算する。

        バッチ内の報酬を group 平均で正規化してポリシー勾配ロスを計算する。
        モデルが log_probs を返す場合はそれを使用し、
        そうでない場合はダミー Tensor で勾配を保つ。

        Args:
            batch: rewards が設定済みの TrainingBatch。
            model: log_probs(prompt, response) を返すモデル or None。
            adapter: 現在のアダプタ（未使用だが API 統一のため受け取る）。

        Returns:
            スカラー Tensor ロス（autograd グラフ付き）。
        """
        if not batch.rewards:
            raise ValueError("TrainingBatch.rewards must be set before compute_loss")

        if torch is None:
            import types
            stub = types.SimpleNamespace(item=lambda: 0.0, backward=lambda: None)
            return stub  # type: ignore[return-value]

        rewards = torch.tensor(batch.rewards, dtype=torch.float32)

        # Group 内の相対報酬で正規化
        r_mean = rewards.mean()
        r_std = rewards.std() + self._epsilon
        advantages = (rewards - r_mean) / r_std

        # モデルが log_probs を提供する場合は使用する
        if hasattr(model, "log_probs"):
            log_probs = torch.stack([
                model.log_probs(p, r)
                for p, r in zip(batch.prompts, batch.responses)
            ])
        else:
            # テスト・スタブ用: advantages のみで勾配を計算
            log_probs = torch.zeros(len(batch.prompts), requires_grad=True)

        # PPO スタイルのクリッピング（オプション）
        if self._clip_ratio is not None:
            ratio = torch.exp(log_probs)
            clipped = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio)
            policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
        else:
            policy_loss = -(log_probs * advantages).mean()

        # エントロピーボーナス（探索促進）
        if self._entropy_coef > 0 and hasattr(model, "entropy"):
            entropy = model.entropy(batch.prompts).mean()
            loss = policy_loss - self._entropy_coef * entropy
        else:
            loss = policy_loss

        logger.debug(
            "GRPO loss=%.4f r_mean=%.4f r_std=%.4f",
            loss.item(), r_mean.item(), r_std.item(),
        )
        return loss
