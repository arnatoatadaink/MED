"""src/memory/embedder.py — sentence-transformers 埋め込みラッパー

テキストをベクトルに変換する。FAISS 格納・検索時に使用。

使い方:
    from src.memory.embedder import Embedder

    embedder = Embedder()                       # デフォルト: all-MiniLM-L6-v2
    vec = embedder.embed("Hello, world!")        # shape: (384,)
    vecs = embedder.embed_batch(["a", "b"])      # shape: (2, 384)

カスタムプロバイダー (LMStudio 等):
    環境変数 EMBEDDING_PROVIDER_URL を設定すると OpenAI 互換 /v1/embeddings を使用。
    10 秒以内に応答がない場合はローカルモデルへ自動フォールバック。

    EMBEDDING_PROVIDER_URL=http://192.168.1.104:52624 poetry run python ...

テスト時はモックモード:
    embedder = Embedder(mock=True)               # ランダムベクトルを返す
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.request
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from src.common.config import EmbeddingConfig, get_settings

logger = logging.getLogger(__name__)

_PROVIDER_PROBE_TIMEOUT = 10  # 秒: プロバイダー疎通確認のタイムアウト


class Embedder:
    """sentence-transformers ベースの埋め込みモデルラッパー。

    環境変数 EMBEDDING_PROVIDER_URL が設定されている場合は OpenAI 互換
    /v1/embeddings エンドポイントを優先的に使用する。プロバイダーへの接続が
    10 秒以内に確立できない場合はローカルモデルへフォールバックする。

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
        self._provider_url: str | None = None  # 疎通確認済みの外部プロバイダー URL

        if not mock:
            self._load_model()

    def _load_model(self) -> None:
        """プロバイダー確認 → ローカルモデルの順でロードする。"""
        # 環境変数が config より優先
        provider_url = os.environ.get("EMBEDDING_PROVIDER_URL") or self._config.provider_url
        if provider_url:
            if self._probe_provider(provider_url):
                self._provider_url = provider_url.rstrip("/")
                logger.info("Using external embedding provider: %s", self._provider_url)
                return
            logger.warning(
                "External embedding provider %s unavailable — falling back to local model",
                provider_url,
            )

        self._load_local_model()

    def _probe_provider(self, url: str) -> bool:
        """プロバイダーに短いテキストを送って疎通・次元を確認する (10 秒タイムアウト)。"""
        model = self._config.provider_model or self._config.model
        payload = json.dumps({"model": model, "input": ["test"]}).encode()
        req = urllib.request.Request(
            f"{url.rstrip('/')}/v1/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=_PROVIDER_PROBE_TIMEOUT) as resp:
                data = json.loads(resp.read())
                dim = len(data["data"][0]["embedding"])
            if dim != self._config.dim:
                logger.warning(
                    "Provider returned dim=%d but expected %d — skipping provider",
                    dim,
                    self._config.dim,
                )
                return False
            return True
        except Exception as exc:
            logger.warning("Provider probe failed (%s): %s", url, exc)
            return False

    def _load_local_model(self) -> None:
        """sentence-transformers モデルをローカルからロードする。"""
        try:
            from sentence_transformers import SentenceTransformer

            kwargs: dict = {"device": self._config.device}

            if self._config.cache_dir is not None:
                from pathlib import Path
                local_path = Path(self._config.cache_dir) / self._config.model
                model_name_or_path = str(local_path) if local_path.exists() else self._config.model
            else:
                model_name_or_path = self._config.model

            logger.info(
                "Loading embedding model: %s (device=%s)",
                model_name_or_path,
                self._config.device,
            )
            self._model = SentenceTransformer(model_name_or_path, **kwargs)
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
        """プロバイダーまたはローカルモデルでエンコードする。"""
        if self._provider_url:
            return self._http_embed(texts, batch_size)
        return self._local_embed(texts, batch_size)

    def _http_embed(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> NDArray[np.float32]:
        """外部プロバイダー API (OpenAI 互換) で埋め込みを取得する。"""
        model = self._config.provider_model or self._config.model
        bs = batch_size or self._config.batch_size
        results: list[list[float]] = []

        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            payload = json.dumps({"model": model, "input": batch}).encode()
            req = urllib.request.Request(
                f"{self._provider_url}/v1/embeddings",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            vecs = [
                item["embedding"]
                for item in sorted(data["data"], key=lambda x: x["index"])
            ]
            results.extend(vecs)

        arr = np.array(results, dtype=np.float32)
        # プロバイダーが未正規化の場合に対応
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-9)

    def _local_embed(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> NDArray[np.float32]:
        """ローカル sentence-transformers モデルでエンコードする。"""
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
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec
