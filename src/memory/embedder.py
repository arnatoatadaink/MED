"""src/memory/embedder.py — sentence-transformers 埋め込みラッパー

テキストをベクトルに変換する。FAISS 格納・検索時に使用。

使い方:
    from src.memory.embedder import Embedder

    embedder = Embedder()                       # デフォルト: all-MiniLM-L6-v2
    vec = embedder.embed("Hello, world!")        # shape: (384,)
    vecs = embedder.embed_batch(["a", "b"])      # shape: (2, 384)

テスト時はモックモード:
    embedder = Embedder(mock=True)               # ランダムベクトルを返す
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from src.common.config import EmbeddingConfig, get_settings

logger = logging.getLogger(__name__)


class Embedder:
    """sentence-transformers ベースの埋め込みモデルラッパー。

    Args:
        config: EmbeddingConfig。省略時は get_settings().embedding を使用。
        mock: True ならモデルをロードせずランダムベクトルを返す（テスト用）。
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        mock: bool = False,
    ) -> None:
        self._config = config or get_settings().embedding
        self._mock = mock
        self._model = None

        if not mock:
            self._load_model()

    def _load_model(self) -> None:
        """sentence-transformers モデルを遅延ロードする。"""
        try:
            from sentence_transformers import SentenceTransformer

            kwargs: dict = {"device": self._config.device}
            if self._config.cache_dir is not None:
                kwargs["cache_folder"] = str(self._config.cache_dir)

            logger.info(
                "Loading embedding model: %s (device=%s)",
                self._config.model,
                self._config.device,
            )
            self._model = SentenceTransformer(self._config.model, **kwargs)
            logger.info("Embedding model loaded successfully")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Falling back to mock mode. Install with: pip install sentence-transformers"
            )
            self._mock = True

    @property
    def dim(self) -> int:
        """埋め込み次元数。"""
        return self._config.dim

    @property
    def model_name(self) -> str:
        """モデル名。"""
        return self._config.model

    def embed(self, text: str) -> NDArray[np.float32]:
        """テキスト 1 件を埋め込みベクトルに変換する。

        Args:
            text: 入力テキスト。

        Returns:
            shape (dim,) の float32 ベクトル。内積類似度用に L2 正規化済み。
        """
        if self._mock:
            return self._mock_embed(text)
        return self._model_embed([text])[0]

    def embed_batch(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
    ) -> NDArray[np.float32]:
        """テキスト複数件を一括で埋め込みベクトルに変換する。

        Args:
            texts: 入力テキストのリスト。
            batch_size: バッチサイズ。省略時は config の batch_size を使用。

        Returns:
            shape (len(texts), dim) の float32 行列。各行は L2 正規化済み。
        """
        if not texts:
            return np.empty((0, self._config.dim), dtype=np.float32)

        if self._mock:
            return np.vstack([self._mock_embed(t) for t in texts])

        return self._model_embed(list(texts), batch_size=batch_size)

    def _model_embed(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> NDArray[np.float32]:
        """実際のモデルでエンコードする。"""
        bs = batch_size or self._config.batch_size
        embeddings = self._model.encode(
            texts,
            batch_size=bs,
            normalize_embeddings=True,  # 内積 = コサイン類似度にする
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def _mock_embed(self, text: str) -> NDArray[np.float32]:
        """テスト用: テキストのハッシュから決定論的なランダムベクトルを生成する。

        同じテキストには常に同じベクトルを返す（テストの再現性のため）。
        """
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self._config.dim).astype(np.float32)
        # L2 正規化
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec
