"""src/training/adapters/lora_xs.py — LoRA-XS (Extreme Small) アダプタ

LoRA のさらに小さいバリアント。
B 行列の代わりに共有スカラーを使用してパラメータ数をさらに削減する。

使い方:
    from src.training.adapters.lora_xs import LoRAXSAdapter
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


@TrainingRegistry.adapter("lora_xs")
class LoRAXSAdapter(ParameterAdapter):
    """LoRA-XS — スカラー重み共有による極小アダプタ。

    Args:
        hidden_dim: モデルの隠れ次元。
        rank: LoRA ランク。
        alpha: スケーリング係数。
        n_scalars: 共有スカラーの数（層間で共有）。
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        rank: int = 4,
        alpha: float = 8.0,
        n_scalars: int = 8,
    ) -> None:
        self._hidden_dim = hidden_dim
        self._rank = rank
        self._alpha = alpha
        self._scale = alpha / rank
        self._n_scalars = n_scalars

        self._A = None
        self._scalars = None

        if torch is not None and nn is not None:
            self._A = nn.Parameter(torch.empty(hidden_dim, rank), requires_grad=False)
            self._scalars = nn.Parameter(torch.ones(n_scalars), requires_grad=True)
            nn.init.xavier_uniform_(self._A)

    @property
    def name(self) -> str:
        return "lora_xs"

    @property
    def num_trainable_params(self) -> int:
        if self._scalars is None:
            return self._n_scalars
        return self._scalars.numel()

    def get_trainable_params(self) -> list:
        if self._scalars is None:
            return []
        return [self._scalars]

    def apply_to(self, model: Any) -> None:
        applied = 0
        for _name, module in (model.named_modules() if hasattr(model, "named_modules") else []):
            if isinstance(module, nn.Linear):
                self._apply_to_linear(module, scalar_idx=applied % self._n_scalars)
                applied += 1
        logger.info(
            "LoRAXSAdapter applied to %d layers (scalars=%d, trainable_params=%d)",
            applied, self._n_scalars, self.num_trainable_params,
        )

    def _apply_to_linear(self, linear: nn.Linear, scalar_idx: int) -> None:
        adapter = self
        original_forward = linear.forward

        def forward_with_lora_xs(x: torch.Tensor) -> torch.Tensor:
            out = original_forward(x)
            if x.shape[-1] == adapter._hidden_dim:
                scalar = adapter._scalars[scalar_idx]
                delta = (x @ adapter._A).sum(dim=-1, keepdim=True) * scalar * adapter._scale
                # broadcast 可能な場合のみ加算
                if delta.shape[-1] == 1 or delta.shape == out.shape:
                    try:
                        out = out + delta.expand_as(out)
                    except RuntimeError:
                        pass
            return out

        linear.forward = forward_with_lora_xs  # type: ignore[method-assign]

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "scalars": self._scalars.detach().cpu(),
            "hidden_dim": self._hidden_dim,
            "rank": self._rank,
            "alpha": self._alpha,
            "n_scalars": self._n_scalars,
        }
        with open(p, "wb") as f:
            pickle.dump(data, f)
        logger.info("LoRAXSAdapter saved to %s", p)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._scalars.data.copy_(data["scalars"])
        logger.info("LoRAXSAdapter loaded from %s", path)
