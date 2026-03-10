"""src/training/algorithms/sft.py — SFT (Supervised Fine-Tuning)

標準的な教師あり微調整アルゴリズム。
GRPO のウォームアップフェーズ（Phase 1 of 3-stage training）で使用する。

ロス計算:
    L_SFT = CrossEntropy(model(prompt), target_response)

使い方:
    from src.training.algorithms.sft import SFTAlgorithm
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


@TrainingRegistry.algorithm("sft")
class SFTAlgorithm(TrainingAlgorithm):
    """Supervised Fine-Tuning。

    Args:
        label_smoothing: ラベルスムージング係数 (default: 0.0)。
        ignore_index: パディングトークン ID (default: -100)。
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
    ) -> None:
        self._label_smoothing = label_smoothing
        self._ignore_index = ignore_index

    @property
    def name(self) -> str:
        return "sft"

    def compute_loss(
        self,
        batch: TrainingBatch,
        model: Any,
        adapter: ParameterAdapter,
    ) -> torch.Tensor:
        """クロスエントロピーロスを計算する。

        モデルが logits(prompt, response) を返す場合はそれを使用する。
        """
        if hasattr(model, "logits"):
            logits_list = [
                model.logits(p, r)
                for p, r in zip(batch.prompts, batch.responses)
            ]
            logits = torch.stack(logits_list)
            targets = torch.stack([
                model.target_ids(r) for r in batch.responses
            ])
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self._ignore_index,
                label_smoothing=self._label_smoothing,
            )
        else:
            # スタブ: 簡易実装 (torch なし環境では 0 ロスを返す)
            if torch is None:
                import types
                loss = types.SimpleNamespace(
                    item=lambda: 0.0,
                    backward=lambda: None,
                )
                return loss  # type: ignore[return-value]
            n = len(batch.prompts)
            logits = torch.randn(n, 2, requires_grad=True)
            targets = torch.zeros(n, dtype=torch.long)
            loss = F.cross_entropy(
                logits, targets,
                label_smoothing=self._label_smoothing,
            )

        logger.debug("SFT loss=%.4f", loss.item())
        return loss
