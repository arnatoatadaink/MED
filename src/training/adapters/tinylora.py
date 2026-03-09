"""src/training/adapters/tinylora.py — TinyLoRA アダプタ

TinyLoRA 論文 (Morris et al., 2026) に基づく極少パラメータアダプタ。
13 パラメータで GSM8K 91% を達成した実績を持つ。

設計:
- frozen_rank: 凍結行列のランク (default: 2)
- projection_dim: 射影次元 (default: 4)
- tie_factor: 重み共有係数 (default: 7)
- 全パラメータ数 ≈ frozen_rank * projection_dim * 2 = 極少

使い方:
    from src.training.adapters.tinylora import TinyLoRAAdapter

    adapter = TinyLoRAAdapter(hidden_dim=4096, frozen_rank=2, projection_dim=4)
    adapter.apply_to(model)
    params = adapter.get_trainable_params()
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


@TrainingRegistry.adapter("tinylora")
class TinyLoRAAdapter(ParameterAdapter):
    """TinyLoRA — 極少パラメータ LoRA バリアント。

    Args:
        hidden_dim: モデルの隠れ次元。
        frozen_rank: 凍結行列のランク。
        projection_dim: 射影次元。
        tie_factor: 重み共有のタイ係数（同一重みを複数層で共有）。
        alpha: スケーリング係数 (LoRA の alpha)。
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        frozen_rank: int = 2,
        projection_dim: int = 4,
        tie_factor: int = 7,
        alpha: float = 1.0,
    ) -> None:
        self._hidden_dim = hidden_dim
        self._frozen_rank = frozen_rank
        self._projection_dim = projection_dim
        self._tie_factor = tie_factor
        self._alpha = alpha
        self._scale = alpha / projection_dim

        self._A = None
        self._B = None
        self._applied_modules: list = []

        if torch is not None and nn is not None:
            # TinyLoRA の中核: A と B の小行列 (極少パラメータ)
            # A: (hidden_dim, frozen_rank) → ランク圧縮
            # B: (frozen_rank, projection_dim) → 射影
            # 学習可能なのは B のみ (A は Xavier 初期化後に凍結)
            self._A = nn.Parameter(
                torch.empty(hidden_dim, frozen_rank),
                requires_grad=False,  # 凍結
            )
            self._B = nn.Parameter(
                torch.zeros(frozen_rank, projection_dim),
                requires_grad=True,  # 学習可能
            )
            nn.init.xavier_uniform_(self._A)
            # B はゼロ初期化（適用直後の出力変化をゼロにする標準 LoRA 慣行）

    @property
    def name(self) -> str:
        return "tinylora"

    @property
    def num_trainable_params(self) -> int:
        if self._B is None:
            return self._frozen_rank * self._projection_dim
        return self._B.numel()

    @property
    def frozen_rank(self) -> int:
        return self._frozen_rank

    @property
    def projection_dim(self) -> int:
        return self._projection_dim

    def get_trainable_params(self) -> list:
        if self._B is None:
            return []
        return [self._B]

    def apply_to(self, model: Any) -> None:
        """モデルの Linear 層に TinyLoRA を差し込む。

        tie_factor: 同一の (A, B) を最大 tie_factor 個の層で共有する。
        """
        applied = 0
        for _name, module in (model.named_modules() if hasattr(model, "named_modules") else []):
            if isinstance(module, nn.Linear) and applied < self._tie_factor:
                self._apply_to_linear(module)
                self._applied_modules.append(module)
                applied += 1

        logger.info(
            "TinyLoRAAdapter applied to %d Linear layers (tie_factor=%d, trainable_params=%d)",
            applied, self._tie_factor, self.num_trainable_params,
        )

    def _apply_to_linear(self, linear: nn.Linear) -> None:
        """Linear 層に LoRA デルタを適用する（in-place weight update なし、フォワードフック）。"""
        adapter = self

        original_forward = linear.forward

        def forward_with_lora(x: torch.Tensor) -> torch.Tensor:
            out = original_forward(x)
            if x.shape[-1] == adapter._hidden_dim:
                delta = x @ adapter._A @ adapter._B * adapter._scale
                if delta.shape == out.shape:
                    out = out + delta
            return out

        linear.forward = forward_with_lora  # type: ignore[method-assign]

    def save(self, path: str) -> None:
        """B 行列（学習済み重み）を pickle で保存する。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "B": self._B.detach().cpu(),
            "hidden_dim": self._hidden_dim,
            "frozen_rank": self._frozen_rank,
            "projection_dim": self._projection_dim,
            "tie_factor": self._tie_factor,
            "alpha": self._alpha,
        }
        with open(p, "wb") as f:
            pickle.dump(data, f)
        logger.info("TinyLoRAAdapter saved to %s (%d bytes)", p, p.stat().st_size)

    def load(self, path: str) -> None:
        """pickle から B 行列を復元する。"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._B.data.copy_(data["B"])
        logger.info("TinyLoRAAdapter loaded from %s", path)
