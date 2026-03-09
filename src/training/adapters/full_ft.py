"""src/training/adapters/full_ft.py — Full Fine-Tuning アダプタ

全パラメータを学習可能にするフルファインチューニング。
アダプタ API の比較ベースラインとして実装。
通常は TinyLoRA/LoRA の代わりに使用しない（メモリ・計算コストが大きい）。

使い方:
    from src.training.adapters.full_ft import FullFTAdapter
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


@TrainingRegistry.adapter("full_ft")
class FullFTAdapter(ParameterAdapter):
    """全パラメータ学習アダプタ。

    Args:
        model: 適用先モデル（apply_to で設定してもよい）。
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._trainable_params: list[nn.Parameter] = []

    @property
    def name(self) -> str:
        return "full_ft"

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self._trainable_params)

    def get_trainable_params(self) -> list[nn.Parameter]:
        return self._trainable_params

    def apply_to(self, model: Any) -> None:
        """全パラメータを学習可能にする。"""
        self._model = model
        self._trainable_params = []

        if hasattr(model, "parameters"):
            for p in model.parameters():
                p.requires_grad_(True)
                self._trainable_params.append(p)

        logger.info(
            "FullFTAdapter: %d trainable parameters",
            self.num_trainable_params,
        )

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if self._model is not None and hasattr(self._model, "state_dict"):
            data = {"state_dict": self._model.state_dict()}
        else:
            data = {"params": [param.detach().cpu() for param in self._trainable_params]}
        with open(p, "wb") as f:
            pickle.dump(data, f)
        logger.info("FullFTAdapter saved to %s", p)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if "state_dict" in data and self._model is not None:
            self._model.load_state_dict(data["state_dict"])
        elif "params" in data:
            for param, saved in zip(self._trainable_params, data["params"]):
                param.data.copy_(saved)
        logger.info("FullFTAdapter loaded from %s", path)
