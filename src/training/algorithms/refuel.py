"""src/training/algorithms/refuel.py — REFUEL (Multi-Turn RL)

REgression-based Fine-tuning with Utterance-Level rewards for LLMs。
多ターン会話に特化した軽量RL手法。

特徴:
- Q値の差分 Q(s,a₁) - Q(s,a₂) を1モデルで回帰（Critic不要）
- 同一プレフィックスから2つの完了系列を生成して差分を学習
- コバリエイトシフト対策により長い多ターン会話で DPO/REBEL より優位
- Llama-3-8B を REFUEL 訓練 → Llama-3.1-70B を多ターン会話で上回った実績

参照: arXiv:2410.04612 (REFUEL, 2024)

GRPO との使い分け:
- GRPO/StarPO-S: 単ターン〜中程度の多ターン。並列サンプリングで崩壊リスクあり
- REFUEL:        長い多ターン会話。Criticなし・逐次更新で安定

使い方:
    from src.training.algorithms.refuel import REFUELAlgorithm
    algo = REFUELAlgorithm()
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


@TrainingRegistry.algorithm("refuel")
class REFUELAlgorithm(TrainingAlgorithm):
    """REFUEL — 多ターン会話特化の軽量 RL アルゴリズム。

    1バッチに偶数個の (prompt, response) を期待する。
    ペアを [0,1], [2,3], ... として Q値差分回帰を行う。

    Args:
        margin: ペア間の reward 差が margin 未満の場合はペアをスキップ。
            ノイジーな報酬によるペアを除外するためのしきい値。
        temperature: ソフトマックス温度（差分をスケール）。
        clip_value: Q値差分のクリッピング上限（勾配爆発防止）。
    """

    def __init__(
        self,
        margin: float = 0.05,
        temperature: float = 1.0,
        clip_value: float = 10.0,
    ) -> None:
        self._margin = margin
        self._temperature = temperature
        self._clip_value = clip_value

    @property
    def name(self) -> str:
        return "refuel"

    def compute_loss(
        self,
        batch: TrainingBatch,
        model: Any,
        adapter: ParameterAdapter,
    ) -> Any:
        """REFUEL ロスを計算する。

        バッチ内のサンプルをペアにして Q値差分回帰損失を計算する。
        - 同一プレフィックスから2つの回答を生成した場合に最も効果的。
        - バッチサイズが奇数の場合は最後の1件を除外する。

        Args:
            batch: rewards が設定済みの TrainingBatch。ペア前提。
            model: log_probs(prompt, response) を返すモデル or None。
            adapter: アダプタ（API統一のため受け取る）。

        Returns:
            スカラー Tensor ロス。
        """
        if not batch.rewards:
            raise ValueError("TrainingBatch.rewards must be set before compute_loss")

        if torch is None:
            import types
            stub = types.SimpleNamespace(item=lambda: 0.0, backward=lambda: None)
            return stub  # type: ignore[return-value]

        n = len(batch.rewards)
        # 偶数個に揃える
        if n % 2 != 0:
            n -= 1

        if n < 2:
            logger.warning("REFUEL: batch too small for pairing (n=%d), returning zero loss", len(batch.rewards))
            return torch.tensor(0.0, requires_grad=True)

        rewards = torch.tensor(batch.rewards[:n], dtype=torch.float32)

        # ペアに分割: (r_a, r_b) × (n//2) ペア
        r_a = rewards[0::2]  # 偶数インデックス
        r_b = rewards[1::2]  # 奇数インデックス
        reward_diff = (r_a - r_b) / self._temperature

        # margin フィルタ: 差が小さいペアはスキップ
        valid = reward_diff.abs() >= self._margin
        if valid.sum() == 0:
            logger.debug("REFUEL: all pairs filtered by margin=%.3f", self._margin)
            return torch.tensor(0.0, requires_grad=True)

        reward_diff = reward_diff[valid]
        reward_diff = torch.clamp(reward_diff, -self._clip_value, self._clip_value)

        # モデルの log_probs を取得
        if hasattr(model, "log_probs"):
            prompts_a = [batch.prompts[i] for i in range(0, n, 2)]
            prompts_b = [batch.prompts[i] for i in range(1, n, 2)]
            responses_a = [batch.responses[i] for i in range(0, n, 2)]
            responses_b = [batch.responses[i] for i in range(1, n, 2)]
            log_probs_a = torch.stack([model.log_probs(p, r) for p, r in zip(prompts_a, responses_a)])
            log_probs_b = torch.stack([model.log_probs(p, r) for p, r in zip(prompts_b, responses_b)])
            log_prob_diff = log_probs_a[valid] - log_probs_b[valid]
        else:
            # スタブ用: 差分ゼロで勾配のみ保持
            log_prob_diff = torch.zeros(valid.sum(), requires_grad=True)

        # Q値差分回帰損失: reward_diff に log_prob_diff を近づける MSE
        loss = torch.nn.functional.mse_loss(log_prob_diff, reward_diff.detach())

        logger.debug(
            "REFUEL loss=%.4f pairs=%d/%d margin_filtered=%d",
            loss.item(), int(valid.sum()), n // 2, int((~valid).sum()),
        )
        return loss
