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

from src.training.base import ParameterAdapter, TrainingAlgorithm, TrainingBatch
from src.training.registry import TrainingRegistry

logger = logging.getLogger(__name__)


@TrainingRegistry.algorithm("grpo")
class GRPOAlgorithm(TrainingAlgorithm):
    """Group Relative Policy Optimization。

    Args:
        epsilon: 報酬正規化の安定項 (default: 1e-8)。
        clip_ratio: PPO スタイルの clipping 比率 (None で無効)。
        entropy_coef: エントロピーボーナス係数 (0 で無効)。
        variance_filter: StarPO-S 分散フィルタ閾値 (None で無効)。
            報酬の標準偏差がこの値未満のバッチはスキップする。
            推奨値: 0.05〜0.1（上位25〜50%の高分散バッチのみ学習）。
        asymmetric_clip: True のとき正の advantage をより強くクリップせずに
            負の advantage を強く抑制する非対称クリッピング (StarPO-S 方式)。
    """

    def __init__(
        self,
        epsilon: float = 1e-8,
        clip_ratio: float | None = None,
        entropy_coef: float = 0.01,
        variance_filter: float | None = None,
        asymmetric_clip: bool = False,
    ) -> None:
        self._epsilon = epsilon
        self._clip_ratio = clip_ratio
        self._entropy_coef = entropy_coef
        self._variance_filter = variance_filter
        self._asymmetric_clip = asymmetric_clip

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

        # StarPO-S: 分散フィルタ — 低分散バッチはスキップ（学習効率が低い）
        r_std_raw = rewards.std().item()
        if self._variance_filter is not None and r_std_raw < self._variance_filter:
            logger.debug(
                "GRPO: batch skipped by variance filter (std=%.4f < %.4f)",
                r_std_raw, self._variance_filter,
            )
            # 勾配なしのゼロロスを返す（optimizerはstepするが更新量はゼロ）
            return torch.tensor(0.0, requires_grad=True)

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

        # クリッピング
        if self._clip_ratio is not None and self._asymmetric_clip:
            # StarPO-S 非対称クリッピング:
            #   正の advantage（良い回答）→ 上限なしで伸ばす
            #   負の advantage（悪い回答）→ clip_ratio で強く抑制
            ratio = torch.exp(log_probs)
            pos_mask = advantages >= 0
            clipped = torch.where(
                pos_mask,
                ratio,  # 正側: クリップなし
                torch.clamp(ratio, min=1 - self._clip_ratio, max=1 + self._clip_ratio),
            )
            policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
        elif self._clip_ratio is not None:
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
