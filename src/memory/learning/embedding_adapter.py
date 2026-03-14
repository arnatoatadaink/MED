"""src/memory/learning/embedding_adapter.py — 埋め込みアダプタ

Sentence-Transformer の埋め込みベクトルをドメイン固有の空間に
線形変換する軽量アダプタ。フィードバック信号でオンライン更新する。

設計方針:
- Phase 1: 線形変換（入力次元 → 出力次元）の重み行列 W
- 出力 = normalize(x @ W)
- SGD でオンライン学習（LTRRanker と同様の設計）
- pickle で重みを保存・復元

使い方:
    from src.memory.learning.embedding_adapter import EmbeddingAdapter

    adapter = EmbeddingAdapter(input_dim=384, output_dim=128)
    adapted = adapter.transform(embedding_vector)

    # フィードバックで更新
    adapter.update(embedding_vector, adapted, reward=1.0)
    adapter.save("data/adapters/embedding_adapter.pkl")
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class AdapterConfig:
    """アダプタ設定。"""

    input_dim: int = 384
    output_dim: int = 128
    learning_rate: float = 1e-3
    regularization: float = 1e-4


class EmbeddingAdapter:
    """線形埋め込みアダプタ（ドメイン適応用）。

    Args:
        input_dim: 入力埋め込み次元数（sentence-transformer の次元）。
        output_dim: 出力次元数（FAISSインデックスの次元）。
        learning_rate: SGD 学習率。
        regularization: L2 正則化係数。
    """

    def __init__(
        self,
        input_dim: int = 384,
        output_dim: int = 128,
        learning_rate: float = 1e-3,
        regularization: float = 1e-4,
    ) -> None:
        self._cfg = AdapterConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            learning_rate=learning_rate,
            regularization=regularization,
        )
        # 恒等写像に近い初期値（正規化後）
        self._W: NDArray[np.float32] = self._init_weights(input_dim, output_dim)
        self._update_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def input_dim(self) -> int:
        return self._cfg.input_dim

    @property
    def output_dim(self) -> int:
        return self._cfg.output_dim

    @property
    def update_count(self) -> int:
        return self._update_count

    def transform(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """埋め込みベクトルを変換する。

        Args:
            x: 形状 (input_dim,) または (n, input_dim) の ndarray。

        Returns:
            形状 (output_dim,) または (n, output_dim) の L2 正規化済み ndarray。
        """
        x = np.asarray(x, dtype=np.float32)
        out = x @ self._W
        return self._l2_normalize(out)

    def update(
        self,
        x: NDArray[np.float32],
        adapted: NDArray[np.float32],
        reward: float,
        baseline: float = 0.5,
    ) -> None:
        """フィードバック信号で重みを SGD 更新する。

        Args:
            x: 元の埋め込みベクトル (input_dim,)。
            adapted: transform(x) の結果 (output_dim,)。
            reward: 報酬スカラー (0.0〜1.0)。
            baseline: 基準報酬（これより高い場合に正の更新）。
        """
        advantage = reward - baseline
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        adapted = np.asarray(adapted, dtype=np.float32).reshape(-1)

        # gradient: (input_dim, 1) x (1, output_dim) = (input_dim, output_dim)
        grad = np.outer(x, adapted) * advantage

        # L2 正則化
        reg_grad = self._cfg.regularization * self._W

        # SGD update
        self._W += self._cfg.learning_rate * (grad - reg_grad)

        self._update_count += 1
        if self._update_count % 100 == 0:
            logger.debug("EmbeddingAdapter: %d updates", self._update_count)

    def save(self, path: str | Path) -> None:
        """重みを pickle で保存する。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"W": self._W, "cfg": self._cfg, "updates": self._update_count}, f)
        logger.info("EmbeddingAdapter saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> EmbeddingAdapter:
        """pickle から重みを復元する。"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        cfg: AdapterConfig = data["cfg"]
        adapter = cls(
            input_dim=cfg.input_dim,
            output_dim=cfg.output_dim,
            learning_rate=cfg.learning_rate,
            regularization=cfg.regularization,
        )
        adapter._W = data["W"]
        adapter._update_count = data.get("updates", 0)
        logger.info("EmbeddingAdapter loaded from %s (updates=%d)", path, adapter._update_count)
        return adapter

    def reset(self) -> None:
        """重みを初期値にリセットする。"""
        self._W = self._init_weights(self._cfg.input_dim, self._cfg.output_dim)
        self._update_count = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _init_weights(input_dim: int, output_dim: int) -> NDArray[np.float32]:
        """Xavier 均一分布で重み初期化。"""
        limit = np.sqrt(6.0 / (input_dim + output_dim))
        return np.random.uniform(-limit, limit, (input_dim, output_dim)).astype(np.float32)

    @staticmethod
    def _l2_normalize(x: NDArray[np.float32]) -> NDArray[np.float32]:
        """L2 正規化（ゼロベクトル対応）。"""
        if x.ndim == 1:
            norm = np.linalg.norm(x)
            return x / max(norm, 1e-12)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norms, 1e-12)
