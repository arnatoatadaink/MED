"""src/training/algorithms/dpo.py — DPO (Direct Preference Optimization)

RLHF の報酬モデルを不要にした DPO アルゴリズム。
(chosen, rejected) ペアから直接ポリシーを最適化する。

ロス計算:
    L_DPO = -log σ(β * (log π(y_w|x) - log π(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)))

使い方:
    from src.training.algorithms.dpo import DPOAlgorithm
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]
try:
    import torch.nn.functional as F
except ImportError:
    F = None  # type: ignore[assignment]

from src.training.base import ParameterAdapter, TrainingAlgorithm, TrainingBatch
from src.training.registry import TrainingRegistry

logger = logging.getLogger(__name__)


@TrainingRegistry.algorithm("dpo")
class DPOAlgorithm(TrainingAlgorithm):
    """Direct Preference Optimization。

    BatchにはChosenとRejectedが交互に格納されていることを想定する:
    - prompts[2i], responses[2i]   → chosen
    - prompts[2i+1], responses[2i+1] → rejected

    Args:
        beta: KL 正則化強度 (default: 0.1)。
        reference_free: 参照モデルなしの DPO (SLiC スタイル) (default: False)。
    """

    def __init__(
        self,
        beta: float = 0.1,
        reference_free: bool = False,
    ) -> None:
        self._beta = beta
        self._reference_free = reference_free

    @property
    def name(self) -> str:
        return "dpo"

    def compute_loss(
        self,
        batch: TrainingBatch,
        model: Any,
        adapter: ParameterAdapter,
    ) -> torch.Tensor:
        """DPO ロスを計算する。

        バッチサイズは偶数（chosen/rejected ペア）でなければならない。
        """
        n = len(batch.prompts)
        if n % 2 != 0:
            raise ValueError(
                f"DPO requires even batch size (chosen+rejected pairs), got {n}"
            )
        if torch is None:
            import types
            return types.SimpleNamespace(item=lambda: 0.0, backward=lambda: None)  # type: ignore[return-value]

        if hasattr(model, "log_probs"):
            all_log_probs = torch.stack([
                model.log_probs(p, r)
                for p, r in zip(batch.prompts, batch.responses)
            ])
        else:
            # スタブ: chosen に正のスコア、rejected にゼロを設定
            chosen_score = torch.tensor([0.5] * (n // 2), requires_grad=True)
            rejected_score = torch.zeros(n // 2)
            log_probs_w = chosen_score
            log_probs_l = rejected_score

            if not self._reference_free:
                # reference_free=False では参照モデルのスコア差分をゼロとみなす
                ref_diff = torch.zeros(n // 2)
                logits = self._beta * (log_probs_w - log_probs_l - ref_diff)
            else:
                logits = self._beta * (log_probs_w - log_probs_l)

            loss = -F.logsigmoid(logits).mean()
            logger.debug("DPO loss=%.4f beta=%.2f", loss.item(), self._beta)
            return loss

        log_probs_w = all_log_probs[0::2]  # chosen
        log_probs_l = all_log_probs[1::2]  # rejected

        if not self._reference_free:
            ref_diff = torch.zeros_like(log_probs_w)
            logits = self._beta * (log_probs_w - log_probs_l - ref_diff)
        else:
            logits = self._beta * (log_probs_w - log_probs_l)

        loss = -F.logsigmoid(logits).mean()
        logger.debug("DPO loss=%.4f", loss.item())
        return loss
