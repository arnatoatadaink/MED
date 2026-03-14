"""src/memory/faiss_index.py — ドメイン別 FAISS インデックス管理

Phase 1: IndexFlatIP (全件探索, 内積)
10万件超: IVF 自動移行
100万件超: PQ 自動移行

使い方:
    from src.memory.faiss_index import FAISSIndexManager

    manager = FAISSIndexManager()
    manager.add("code", doc_ids, embeddings)
    results = manager.search("code", query_vec, k=5)
"""

from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np
from numpy.typing import NDArray

from src.common.config import FAISSConfig, FAISSIndexConfig, get_settings

logger = logging.getLogger(__name__)


class DomainIndex:
    """単一ドメインの FAISS インデックス + ID マッピング。

    FAISS は連番の internal index を使うため、doc_id ↔ internal_idx の対応を管理する。
    """

    def __init__(self, config: FAISSIndexConfig) -> None:
        self._config = config
        self._index: faiss.Index = self._create_index(config.initial_type)
        self._id_to_idx: dict[str, int] = {}  # doc_id → FAISS internal idx
        self._idx_to_id: dict[int, str] = {}  # FAISS internal idx → doc_id

    def _create_index(self, index_type: str) -> faiss.Index:
        """インデックスタイプ文字列から FAISS インデックスを生成する。"""
        dim = self._config.dim

        if index_type == "Flat":
            if self._config.metric == "inner_product":
                return faiss.IndexFlatIP(dim)
            return faiss.IndexFlatL2(dim)

        # IVF 系: "IVF1024,Flat" 等
        if index_type.startswith("IVF"):
            return self._create_ivf_index(index_type, dim)

        # HNSW 系: "HNSW32" 等
        if index_type.startswith("HNSW"):
            m = int(index_type.replace("HNSW", ""))
            index = faiss.IndexHNSWFlat(dim, m)
            return index

        raise ValueError(f"Unsupported index type: {index_type!r}")

    def _create_ivf_index(self, index_type: str, dim: int) -> faiss.Index:
        """IVF 系インデックスを生成する。事前学習が必要なため空で返す。"""
        index = faiss.index_factory(dim, index_type, faiss.METRIC_INNER_PRODUCT)
        return index

    @property
    def count(self) -> int:
        """格納済みベクトル数。"""
        return self._index.ntotal

    def add(self, doc_ids: list[str], embeddings: NDArray[np.float32]) -> None:
        """ベクトルを追加する。

        Args:
            doc_ids: ドキュメント ID のリスト。
            embeddings: shape (n, dim) の float32 行列。
        """
        if len(doc_ids) != embeddings.shape[0]:
            raise ValueError(
                f"doc_ids length ({len(doc_ids)}) != embeddings rows ({embeddings.shape[0]})"
            )
        if embeddings.shape[1] != self._config.dim:
            raise ValueError(
                f"embedding dim ({embeddings.shape[1]}) != config dim ({self._config.dim})"
            )

        # 重複チェック: 既に存在する doc_id はスキップ
        new_ids = []
        new_vecs = []
        for i, doc_id in enumerate(doc_ids):
            if doc_id not in self._id_to_idx:
                new_ids.append(doc_id)
                new_vecs.append(embeddings[i])

        if not new_ids:
            return

        vecs = np.vstack(new_vecs).astype(np.float32)

        # IVF 系で未学習の場合は train してから add
        if hasattr(self._index, "is_trained") and not self._index.is_trained:
            logger.info("Training IVF index with %d vectors", vecs.shape[0])
            self._index.train(vecs)

        start_idx = self._index.ntotal
        self._index.add(vecs)

        for offset, doc_id in enumerate(new_ids):
            idx = start_idx + offset
            self._id_to_idx[doc_id] = idx
            self._idx_to_id[idx] = doc_id

    def search(
        self,
        query: NDArray[np.float32],
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """類似ベクトルを検索する。

        Args:
            query: shape (dim,) の検索クエリベクトル。
            k: 返す結果数。

        Returns:
            (doc_id, score) のリスト。スコア降順。
        """
        if self._index.ntotal == 0:
            return []

        actual_k = min(k, self._index.ntotal)
        query_2d = query.reshape(1, -1).astype(np.float32)

        # IVF 系の場合は nprobe を設定
        if hasattr(self._index, "nprobe"):
            self._index.nprobe = self._config.nprobe

        scores, indices = self._index.search(query_2d, actual_k)

        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS が結果不足のとき -1 を返す
                continue
            doc_id = self._idx_to_id.get(int(idx))
            if doc_id is not None:
                results.append((doc_id, float(score)))

        return results

    def remove(self, doc_ids: list[str]) -> int:
        """ドキュメントを論理削除する。

        IndexFlatIP は直接削除をサポートしないため、
        ID マッピングから除外して検索結果に出ないようにする。
        rebuild() で物理的に再構築可能。

        Returns:
            実際に削除された件数。
        """
        removed = 0
        for doc_id in doc_ids:
            idx = self._id_to_idx.pop(doc_id, None)
            if idx is not None:
                self._idx_to_id.pop(idx, None)
                removed += 1
        return removed

    def contains(self, doc_id: str) -> bool:
        """指定 doc_id が存在するか。"""
        return doc_id in self._id_to_idx

    def rebuild(self, doc_ids: list[str], embeddings: NDArray[np.float32]) -> None:
        """インデックスを全データで再構築する。

        論理削除の蓄積後や、インデックスタイプの移行時に使用。
        """
        self._index = self._create_index(self._config.initial_type)
        self._id_to_idx.clear()
        self._idx_to_id.clear()
        if len(doc_ids) > 0:
            self.add(doc_ids, embeddings)

    def should_migrate(self) -> str | None:
        """scale_rules に基づき、移行が必要かチェックする。

        Returns:
            移行先のインデックスタイプ。移行不要なら None。
        """
        n = self.count
        for rule in self._config.scale_rules:
            if n >= rule.threshold:
                return rule.migrate_to
        return None

    def save(self, path: Path) -> None:
        """インデックスと ID マッピングをファイルに保存する。"""
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))
        np.savez(
            str(path / "id_map.npz"),
            ids=list(self._id_to_idx.keys()),
            indices=list(self._id_to_idx.values()),
        )
        logger.info("Saved index to %s (%d vectors)", path, self.count)

    def load(self, path: Path) -> None:
        """インデックスと ID マッピングをファイルから復元する。"""
        index_path = path / "index.faiss"
        map_path = path / "id_map.npz"

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        self._index = faiss.read_index(str(index_path))

        if map_path.exists():
            data = np.load(str(map_path), allow_pickle=True)
            ids = data["ids"].tolist()
            indices = data["indices"].tolist()
            self._id_to_idx = dict(zip(ids, indices))
            self._idx_to_id = dict(zip(indices, ids))

        logger.info("Loaded index from %s (%d vectors)", path, self.count)


class FAISSIndexManager:
    """ドメイン別 FAISS インデックスの統合管理。

    config の indices 設定に基づき、ドメインごとの DomainIndex を管理する。
    """

    def __init__(self, config: FAISSConfig | None = None) -> None:
        self._config = config or get_settings().faiss
        self._indices: dict[str, DomainIndex] = {}
        self._init_indices()

    def _init_indices(self) -> None:
        """設定に基づきドメインインデックスを初期化する。"""
        for domain, idx_config in self._config.indices.items():
            self._indices[domain] = DomainIndex(idx_config)
            logger.debug("Initialized domain index: %s (type=%s)", domain, idx_config.initial_type)

    def get_domain(self, domain: str) -> DomainIndex:
        """指定ドメインの DomainIndex を取得する。存在しなければ作成。"""
        if domain not in self._indices:
            idx_config = self._config.indices.get(
                domain,
                FAISSIndexConfig(dim=self._config.indices.get("general", FAISSIndexConfig()).dim),
            )
            self._indices[domain] = DomainIndex(idx_config)
        return self._indices[domain]

    @property
    def domains(self) -> list[str]:
        """管理中のドメイン一覧。"""
        return list(self._indices.keys())

    def add(
        self,
        domain: str,
        doc_ids: list[str],
        embeddings: NDArray[np.float32],
    ) -> None:
        """指定ドメインにベクトルを追加する。"""
        self.get_domain(domain).add(doc_ids, embeddings)

    def search(
        self,
        domain: str,
        query: NDArray[np.float32],
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """指定ドメインで類似検索する。"""
        return self.get_domain(domain).search(query, k=k)

    def search_all(
        self,
        query: NDArray[np.float32],
        k: int = 5,
    ) -> list[tuple[str, str, float]]:
        """全ドメイン横断で検索し、スコア上位 k 件を返す。

        Returns:
            (doc_id, domain, score) のリスト。スコア降順。
        """
        all_results: list[tuple[str, str, float]] = []
        for domain_name, index in self._indices.items():
            for doc_id, score in index.search(query, k=k):
                all_results.append((doc_id, domain_name, score))

        all_results.sort(key=lambda x: x[2], reverse=True)
        return all_results[:k]

    def remove(self, domain: str, doc_ids: list[str]) -> int:
        """指定ドメインからドキュメントを削除する。"""
        return self.get_domain(domain).remove(doc_ids)

    def contains(self, domain: str, doc_id: str) -> bool:
        """指定ドメインにドキュメントが存在するか。"""
        if domain not in self._indices:
            return False
        return self._indices[domain].contains(doc_id)

    def total_count(self) -> int:
        """全ドメインの合計ベクトル数。"""
        return sum(idx.count for idx in self._indices.values())

    def domain_stats(self) -> dict[str, int]:
        """ドメインごとのベクトル数。"""
        return {domain: idx.count for domain, idx in self._indices.items()}

    def save(self, base_dir: Path | None = None) -> None:
        """全ドメインのインデックスを保存する。"""
        base = base_dir or self._config.base_dir
        for domain_name, index in self._indices.items():
            domain_dir = Path(base) / domain_name
            index.save(domain_dir)

    def load(self, base_dir: Path | None = None) -> None:
        """全ドメインのインデックスをロードする。"""
        base = base_dir or self._config.base_dir
        for domain_name, index in self._indices.items():
            domain_dir = Path(base) / domain_name
            if domain_dir.exists():
                index.load(domain_dir)
