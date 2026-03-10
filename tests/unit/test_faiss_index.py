"""tests/unit/test_faiss_index.py — src/memory/faiss_index.py の単体テスト"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.common.config import FAISSConfig, FAISSIndexConfig, FAISSScaleRule, get_settings
from src.memory.faiss_index import DomainIndex, FAISSIndexManager


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    yield
    get_settings.cache_clear()


DIM = 64  # テスト用の小さい次元


def _random_vecs(n: int, dim: int = DIM) -> np.ndarray:
    """L2 正規化済みランダムベクトルを生成する。"""
    vecs = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _make_config(dim: int = DIM, initial_type: str = "Flat") -> FAISSIndexConfig:
    return FAISSIndexConfig(dim=dim, initial_type=initial_type, metric="inner_product", nprobe=4)


# ============================================================================
# DomainIndex テスト
# ============================================================================


class TestDomainIndex:
    def test_empty_index(self):
        idx = DomainIndex(_make_config())
        assert idx.count == 0

    def test_add_and_count(self):
        idx = DomainIndex(_make_config())
        vecs = _random_vecs(10)
        ids = [f"doc_{i}" for i in range(10)]
        idx.add(ids, vecs)
        assert idx.count == 10

    def test_search_basic(self):
        idx = DomainIndex(_make_config())
        vecs = _random_vecs(5)
        ids = [f"doc_{i}" for i in range(5)]
        idx.add(ids, vecs)

        results = idx.search(vecs[0], k=3)
        assert len(results) == 3
        # 自分自身が最も高いスコアになるはず
        assert results[0][0] == "doc_0"
        assert results[0][1] == pytest.approx(1.0, abs=0.01)

    def test_search_empty_index(self):
        idx = DomainIndex(_make_config())
        query = _random_vecs(1)[0]
        results = idx.search(query, k=5)
        assert results == []

    def test_search_k_larger_than_count(self):
        idx = DomainIndex(_make_config())
        vecs = _random_vecs(3)
        idx.add(["a", "b", "c"], vecs)
        results = idx.search(vecs[0], k=10)
        assert len(results) == 3

    def test_duplicate_ids_skipped(self):
        idx = DomainIndex(_make_config())
        vecs = _random_vecs(3)
        idx.add(["a", "b", "c"], vecs)
        assert idx.count == 3

        # 同じ ID を追加しても増えない
        idx.add(["a", "b"], _random_vecs(2))
        assert idx.count == 3

    def test_contains(self):
        idx = DomainIndex(_make_config())
        idx.add(["doc_1"], _random_vecs(1))
        assert idx.contains("doc_1") is True
        assert idx.contains("doc_999") is False

    def test_remove(self):
        idx = DomainIndex(_make_config())
        vecs = _random_vecs(5)
        ids = [f"doc_{i}" for i in range(5)]
        idx.add(ids, vecs)

        removed = idx.remove(["doc_1", "doc_3"])
        assert removed == 2
        assert idx.contains("doc_1") is False
        assert idx.contains("doc_3") is False
        assert idx.contains("doc_0") is True

    def test_remove_nonexistent(self):
        idx = DomainIndex(_make_config())
        idx.add(["a"], _random_vecs(1))
        removed = idx.remove(["nonexistent"])
        assert removed == 0

    def test_remove_excludes_from_search(self):
        idx = DomainIndex(_make_config())
        vecs = _random_vecs(5)
        ids = [f"doc_{i}" for i in range(5)]
        idx.add(ids, vecs)
        idx.remove(["doc_2"])

        results = idx.search(vecs[2], k=5)
        returned_ids = [r[0] for r in results]
        assert "doc_2" not in returned_ids

    def test_rebuild(self):
        idx = DomainIndex(_make_config())
        vecs = _random_vecs(10)
        ids = [f"doc_{i}" for i in range(10)]
        idx.add(ids, vecs)
        idx.remove(["doc_5", "doc_6", "doc_7"])

        # rebuild で物理的にも縮小
        remaining_ids = [f"doc_{i}" for i in range(10) if i not in (5, 6, 7)]
        remaining_vecs = np.vstack([vecs[i] for i in range(10) if i not in (5, 6, 7)])
        idx.rebuild(remaining_ids, remaining_vecs)

        assert idx.count == 7

    def test_dimension_mismatch_raises(self):
        idx = DomainIndex(_make_config(dim=64))
        wrong_dim_vecs = _random_vecs(3, dim=128)
        with pytest.raises(ValueError, match="embedding dim"):
            idx.add(["a", "b", "c"], wrong_dim_vecs)

    def test_length_mismatch_raises(self):
        idx = DomainIndex(_make_config())
        vecs = _random_vecs(3)
        with pytest.raises(ValueError, match="doc_ids length"):
            idx.add(["a", "b"], vecs)  # 2 ids for 3 vecs

    def test_should_migrate_returns_none_when_below_threshold(self):
        config = _make_config()
        config.scale_rules = [FAISSScaleRule(threshold=1000, migrate_to="IVF32,Flat")]
        idx = DomainIndex(config)
        idx.add(["a"], _random_vecs(1))
        assert idx.should_migrate() is None

    def test_should_migrate_returns_type_when_above_threshold(self):
        config = _make_config()
        config.scale_rules = [FAISSScaleRule(threshold=5, migrate_to="IVF32,Flat")]
        idx = DomainIndex(config)
        idx.add([f"doc_{i}" for i in range(10)], _random_vecs(10))
        assert idx.should_migrate() == "IVF32,Flat"

    def test_save_and_load(self, tmp_path: Path):
        idx = DomainIndex(_make_config())
        vecs = _random_vecs(5)
        ids = ["a", "b", "c", "d", "e"]
        idx.add(ids, vecs)

        save_dir = tmp_path / "test_index"
        idx.save(save_dir)

        # 新しいインデックスに復元
        idx2 = DomainIndex(_make_config())
        idx2.load(save_dir)
        assert idx2.count == 5
        assert idx2.contains("c") is True

        # 検索結果が同じ
        results = idx2.search(vecs[0], k=3)
        assert results[0][0] == "a"

    def test_load_nonexistent_raises(self, tmp_path: Path):
        idx = DomainIndex(_make_config())
        with pytest.raises(FileNotFoundError):
            idx.load(tmp_path / "nonexistent")


# ============================================================================
# FAISSIndexManager テスト
# ============================================================================


class TestFAISSIndexManager:
    def _make_manager(self) -> FAISSIndexManager:
        config = FAISSConfig(
            base_dir=Path("data/test_indices"),
            indices={
                "code": FAISSIndexConfig(dim=DIM),
                "general": FAISSIndexConfig(dim=DIM),
            },
        )
        return FAISSIndexManager(config=config)

    def test_initial_domains(self):
        mgr = self._make_manager()
        assert set(mgr.domains) == {"code", "general"}

    def test_add_and_search(self):
        mgr = self._make_manager()
        vecs = _random_vecs(5)
        ids = [f"doc_{i}" for i in range(5)]
        mgr.add("code", ids, vecs)

        results = mgr.search("code", vecs[0], k=3)
        assert len(results) == 3
        assert results[0][0] == "doc_0"

    def test_search_across_domains(self):
        mgr = self._make_manager()
        code_vecs = _random_vecs(3)
        gen_vecs = _random_vecs(3)

        mgr.add("code", ["c0", "c1", "c2"], code_vecs)
        mgr.add("general", ["g0", "g1", "g2"], gen_vecs)

        results = mgr.search_all(code_vecs[0], k=3)
        assert len(results) == 3
        # 結果は (doc_id, domain, score) のタプル
        assert len(results[0]) == 3

    def test_total_count(self):
        mgr = self._make_manager()
        mgr.add("code", ["a", "b"], _random_vecs(2))
        mgr.add("general", ["c"], _random_vecs(1))
        assert mgr.total_count() == 3

    def test_domain_stats(self):
        mgr = self._make_manager()
        mgr.add("code", ["a", "b"], _random_vecs(2))
        mgr.add("general", ["c", "d", "e"], _random_vecs(3))
        stats = mgr.domain_stats()
        assert stats["code"] == 2
        assert stats["general"] == 3

    def test_remove(self):
        mgr = self._make_manager()
        mgr.add("code", ["a", "b", "c"], _random_vecs(3))
        removed = mgr.remove("code", ["b"])
        assert removed == 1
        assert mgr.contains("code", "b") is False
        assert mgr.contains("code", "a") is True

    def test_contains(self):
        mgr = self._make_manager()
        mgr.add("code", ["a"], _random_vecs(1))
        assert mgr.contains("code", "a") is True
        assert mgr.contains("code", "x") is False
        assert mgr.contains("nonexistent_domain", "a") is False

    def test_dynamic_domain_creation(self):
        mgr = self._make_manager()
        # academic ドメインは設定にないが、動的に作成される
        mgr.add("academic", ["a1"], _random_vecs(1))
        assert "academic" in mgr.domains
        assert mgr.total_count() == 1

    def test_save_and_load(self, tmp_path: Path):
        mgr = self._make_manager()
        mgr.add("code", ["a", "b"], _random_vecs(2))
        mgr.add("general", ["c"], _random_vecs(1))

        mgr.save(tmp_path)

        mgr2 = self._make_manager()
        mgr2.load(tmp_path)
        assert mgr2.total_count() == 3
        assert mgr2.contains("code", "a") is True
        assert mgr2.contains("general", "c") is True
