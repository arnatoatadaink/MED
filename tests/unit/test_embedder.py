"""tests/unit/test_embedder.py — src/memory/embedder.py の単体テスト

sentence-transformers をインストールせずにテスト可能 (mock=True)。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.common.config import EmbeddingConfig, get_settings
from src.memory.embedder import Embedder


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    yield
    get_settings.cache_clear()


@pytest.fixture()
def mock_embedder() -> Embedder:
    """モックモードの Embedder (モデルロード不要)。"""
    config = EmbeddingConfig(model="test-model", dim=128, batch_size=32)
    return Embedder(config=config, mock=True)


# ============================================================================
# 基本プロパティ
# ============================================================================


class TestProperties:
    def test_dim(self, mock_embedder: Embedder):
        assert mock_embedder.dim == 128

    def test_model_name(self, mock_embedder: Embedder):
        assert mock_embedder.model_name == "test-model"


# ============================================================================
# 単一テキスト埋め込み
# ============================================================================


class TestEmbed:
    def test_returns_correct_shape(self, mock_embedder: Embedder):
        vec = mock_embedder.embed("Hello, world!")
        assert vec.shape == (128,)
        assert vec.dtype == np.float32

    def test_deterministic(self, mock_embedder: Embedder):
        """同じテキストは同じベクトルを返す。"""
        vec1 = mock_embedder.embed("test text")
        vec2 = mock_embedder.embed("test text")
        np.testing.assert_array_equal(vec1, vec2)

    def test_different_texts_different_vectors(self, mock_embedder: Embedder):
        """異なるテキストは異なるベクトルを返す。"""
        vec1 = mock_embedder.embed("hello")
        vec2 = mock_embedder.embed("world")
        assert not np.allclose(vec1, vec2)

    def test_l2_normalized(self, mock_embedder: Embedder):
        """ベクトルが L2 正規化されていること。"""
        vec = mock_embedder.embed("normalize me")
        norm = np.linalg.norm(vec)
        assert norm == pytest.approx(1.0, abs=1e-5)


# ============================================================================
# バッチ埋め込み
# ============================================================================


class TestEmbedBatch:
    def test_returns_correct_shape(self, mock_embedder: Embedder):
        texts = ["a", "b", "c"]
        vecs = mock_embedder.embed_batch(texts)
        assert vecs.shape == (3, 128)
        assert vecs.dtype == np.float32

    def test_empty_input(self, mock_embedder: Embedder):
        vecs = mock_embedder.embed_batch([])
        assert vecs.shape == (0, 128)
        assert vecs.dtype == np.float32

    def test_single_input(self, mock_embedder: Embedder):
        vecs = mock_embedder.embed_batch(["single"])
        assert vecs.shape == (1, 128)

    def test_consistent_with_single_embed(self, mock_embedder: Embedder):
        """batch と single が同じ結果を返すこと。"""
        text = "consistency check"
        single = mock_embedder.embed(text)
        batch = mock_embedder.embed_batch([text])
        np.testing.assert_array_almost_equal(single, batch[0])

    def test_all_rows_normalized(self, mock_embedder: Embedder):
        texts = ["alpha", "beta", "gamma", "delta"]
        vecs = mock_embedder.embed_batch(texts)
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_custom_batch_size(self, mock_embedder: Embedder):
        """batch_size 引数が受け付けられること (mock では効果なし)。"""
        texts = ["a", "b", "c", "d", "e"]
        vecs = mock_embedder.embed_batch(texts, batch_size=2)
        assert vecs.shape == (5, 128)


# ============================================================================
# 異なる次元数
# ============================================================================


class TestDifferentDimensions:
    def test_768_dim(self):
        config = EmbeddingConfig(dim=768)
        embedder = Embedder(config=config, mock=True)
        vec = embedder.embed("test")
        assert vec.shape == (768,)

    def test_384_dim(self):
        config = EmbeddingConfig(dim=384)
        embedder = Embedder(config=config, mock=True)
        vec = embedder.embed("test")
        assert vec.shape == (384,)


# ============================================================================
# Inner product が cosine similarity と等価であること
# ============================================================================


class TestInnerProductAsCosine:
    def test_similar_texts_higher_score(self, mock_embedder: Embedder):
        """類似テキストが高いスコアを返すとは限らないが (mock)、
        内積が -1 〜 1 の範囲に収まること。"""
        vec1 = mock_embedder.embed("python sort list")
        vec2 = mock_embedder.embed("javascript array map")
        score = float(np.dot(vec1, vec2))
        assert -1.0 <= score <= 1.0

    def test_self_similarity_is_one(self, mock_embedder: Embedder):
        """自己内積が 1.0 であること (L2 正規化済みなので)。"""
        vec = mock_embedder.embed("identical text")
        score = float(np.dot(vec, vec))
        assert score == pytest.approx(1.0, abs=1e-5)
