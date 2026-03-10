"""src/memory/deduplicator.py — ドキュメント重複排除

FAISS に追加しようとするドキュメントの重複チェックを行う。
コンテンツハッシュ (SHA-256) と意味的類似度 (コサイン) の2段階で検出する。

設計:
- Stage 1: コンテンツハッシュによる完全一致検出 (O(1))
- Stage 2: 埋め込みベクトルのコサイン類似度による近似重複検出
- 閾値: hash_match → exact dup、cosine > threshold → near-dup

使い方:
    from src.memory.deduplicator import Deduplicator

    dedup = Deduplicator(threshold=0.95)
    result = dedup.check(doc, existing_hashes, existing_vectors)
    if result.is_duplicate:
        skip this doc
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class DedupResult:
    """重複チェック結果。"""

    is_duplicate: bool
    duplicate_type: str = ""       # "exact" | "near" | ""
    duplicate_doc_id: Optional[str] = None
    similarity: float = 0.0        # 近似重複の場合の類似度

    @property
    def is_exact(self) -> bool:
        return self.duplicate_type == "exact"

    @property
    def is_near(self) -> bool:
        return self.duplicate_type == "near"


class Deduplicator:
    """ドキュメント重複排除クラス。

    Args:
        near_dup_threshold: 近似重複とみなすコサイン類似度閾値 (default: 0.95)。
        check_near_dup: 近似重複チェックを行うか (default: True)。
    """

    def __init__(
        self,
        near_dup_threshold: float = 0.95,
        check_near_dup: bool = True,
    ) -> None:
        self._threshold = near_dup_threshold
        self._check_near = check_near_dup

    @staticmethod
    def content_hash(content: str) -> str:
        """コンテンツの SHA-256 ハッシュを返す。"""
        return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()

    def check(
        self,
        content_hash: str,
        query_vector: Optional[NDArray[np.float32]] = None,
        existing_hashes: Optional[dict[str, str]] = None,
        existing_vectors: Optional[list[tuple[str, NDArray[np.float32]]]] = None,
    ) -> DedupResult:
        """重複チェックを行う。

        Args:
            content_hash: チェック対象のコンテンツハッシュ。
            query_vector: チェック対象の埋め込みベクトル（near-dup 用）。
            existing_hashes: {doc_id: hash} の辞書。
            existing_vectors: [(doc_id, vector), ...] のリスト。

        Returns:
            DedupResult。
        """
        # Stage 1: 完全一致
        if existing_hashes:
            for doc_id, h in existing_hashes.items():
                if h == content_hash:
                    logger.debug("Exact duplicate detected: doc_id=%s", doc_id)
                    return DedupResult(
                        is_duplicate=True,
                        duplicate_type="exact",
                        duplicate_doc_id=doc_id,
                        similarity=1.0,
                    )

        # Stage 2: 近似重複
        if self._check_near and query_vector is not None and existing_vectors:
            result = self._check_near_dup(query_vector, existing_vectors)
            if result.is_duplicate:
                return result

        return DedupResult(is_duplicate=False)

    def _check_near_dup(
        self,
        query_vec: NDArray[np.float32],
        existing_vectors: list[tuple[str, NDArray[np.float32]]],
    ) -> DedupResult:
        """コサイン類似度で近似重複を検出する。"""
        q = query_vec.astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return DedupResult(is_duplicate=False)
        q_normalized = q / q_norm

        best_sim = 0.0
        best_doc_id = ""

        for doc_id, vec in existing_vectors:
            v = vec.astype(np.float32)
            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                continue
            sim = float(np.dot(q_normalized, v / v_norm))
            if sim > best_sim:
                best_sim = sim
                best_doc_id = doc_id

        if best_sim >= self._threshold:
            logger.debug(
                "Near-duplicate detected: doc_id=%s sim=%.4f",
                best_doc_id, best_sim,
            )
            return DedupResult(
                is_duplicate=True,
                duplicate_type="near",
                duplicate_doc_id=best_doc_id,
                similarity=best_sim,
            )

        return DedupResult(is_duplicate=False)

    def filter_batch(
        self,
        content_hashes: list[tuple[str, str]],  # [(content_hash, doc_candidate_id), ...]
        existing_hashes: Optional[dict[str, str]] = None,
    ) -> tuple[list[str], list[str]]:
        """バッチ重複フィルタリング（ハッシュのみ）。

        Returns:
            (unique_ids, duplicate_ids) のタプル。
        """
        existing = existing_hashes or {}
        seen_in_batch: set[str] = set()
        unique_ids = []
        duplicate_ids = []

        for content_hash, candidate_id in content_hashes:
            # 既存との重複
            if content_hash in existing.values():
                duplicate_ids.append(candidate_id)
                continue
            # バッチ内での重複
            if content_hash in seen_in_batch:
                duplicate_ids.append(candidate_id)
                continue
            seen_in_batch.add(content_hash)
            unique_ids.append(candidate_id)

        logger.debug(
            "Deduplicator.filter_batch: %d unique, %d duplicates",
            len(unique_ids), len(duplicate_ids),
        )
        return unique_ids, duplicate_ids
