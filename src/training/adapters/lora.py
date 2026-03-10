"""src/training/adapters/lora.py — 標準 LoRA アダプタ

Hu et al. 2021 の LoRA (Low-Rank Adaptation) 実装。
TinyLoRA より多くのパラメータを使用するが、より高い表現力を持つ。

使い方:
    from src.training.adapters.lora import LoRAAdapter

    adapter = LoRAAdapter(hidden_dim=4096, rank=8, alpha=16.0)
    adapter.apply_to(model)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]
try:
    import torch.nn as nn
except ImportError:
    nn = None  # type: ignore[assignment]

from src.training.base import ParameterAdapter
from src.training.registry import TrainingRegistry

logger = logging.getLogger(__name__)


@TrainingRegistry.adapter("lora")
class LoRAAdapter(ParameterAdapter):
    """標準 LoRA アダプタ。

    Args:
        hidden_dim: モデルの隠れ次元。
        rank: LoRA ランク。
        alpha: スケーリング係数 (通常 rank の 2 倍程度)。
        dropout: LoRA ドロップアウト率。
        target_modules: 適用対象モジュール名のサブセット (None で全 Linear)。
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        target_modules: list[str] | None = None,
    ) -> None:
        self._hidden_dim = hidden_dim
        self._rank = rank
        self._alpha = alpha
        self._scale = alpha / rank
        self._dropout_rate = dropout
        self._target_modules = target_modules

        self._lora_A = None
        self._lora_B = None
        self._dropout = None

        if torch is not None and nn is not None:
            self._lora_A = nn.Parameter(torch.empty(hidden_dim, rank), requires_grad=True)
            self._lora_B = nn.Parameter(torch.zeros(rank, hidden_dim), requires_grad=True)
            self._dropout = nn.Dropout(p=dropout) if dropout > 0 else None
            nn.init.kaiming_uniform_(self._lora_A, a=5 ** 0.5)

    @property
    def name(self) -> str:
        return "lora"

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def num_trainable_params(self) -> int:
        if self._lora_A is None:
            return self._hidden_dim * self._rank * 2
        return self._lora_A.numel() + self._lora_B.numel()

    def get_trainable_params(self) -> list:
        if self._lora_A is None:
            return []
        return [self._lora_A, self._lora_B]

    def apply_to(self, model: Any) -> None:
        applied = 0
        for name, module in (model.named_modules() if hasattr(model, "named_modules") else []):
            if not isinstance(module, nn.Linear):
                continue
            if self._target_modules and not any(t in name for t in self._target_modules):
                continue
            self._apply_to_linear(module)
            applied += 1

        logger.info(
            "LoRAAdapter applied to %d Linear layers (rank=%d, trainable_params=%d)",
            applied, self._rank, self.num_trainable_params,
        )

    def _apply_to_linear(self, linear: nn.Linear) -> None:
        adapter = self
        original_forward = linear.forward

        def forward_with_lora(x: torch.Tensor) -> torch.Tensor:
            out = original_forward(x)
            if x.shape[-1] == adapter._hidden_dim:
                h = x @ adapter._lora_A
                if adapter._dropout is not None:
                    h = adapter._dropout(h)
                delta = h @ adapter._lora_B * adapter._scale
                if delta.shape == out.shape:
                    out = out + delta
            return out

        linear.forward = forward_with_lora  # type: ignore[method-assign]

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "lora_A": self._lora_A.detach().cpu(),
            "lora_B": self._lora_B.detach().cpu(),
            "hidden_dim": self._hidden_dim,
            "rank": self._rank,
            "alpha": self._alpha,
        }
        with open(p, "wb") as f:
            pickle.dump(data, f)
        logger.info("LoRAAdapter saved to %s", p)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._lora_A.data.copy_(data["lora_A"])
        self._lora_B.data.copy_(data["lora_B"])
        logger.info("LoRAAdapter loaded from %s", path)
