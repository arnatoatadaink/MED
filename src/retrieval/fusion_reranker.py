"""src/retrieval/fusion_reranker.py — Reciprocal Rank Fusion (RRF) リランカー

FAISS・Knowledge Graph・SQL の複数検索バックエンドの結果を
Reciprocal Rank Fusion (RRF) で統合し、統一ランキングを生成する。

RRF 公式:
    score(d) = Σ 1 / (k + rank_i(d))
    k = 60 (デフォルト, Robertson 2009)

設計:
- 入力: 複数の「(doc_id, score)」リスト（各バックエンドの結果）
- 出力: RRF スコア降順の doc_id リスト
- ソース名ごとに重みを設定可能（デフォルト: 全て均等）

使い方:
    from src.retrieval.fusion_reranker import FusionReranker

    reranker = FusionReranker()
    ranked_ids = reranker.fuse({
        "faiss": [("doc1", 0.9), ("doc2", 0.7)],
        "kg":    [("doc2", 0.8), ("doc3", 0.6)],
        "sql":   [("doc1", 1.0)],
    })
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_K = 60  # RRF 定数 (Robertson 2009)


@dataclass
class FusionResult:
    """Fusion 結果。"""

    doc_id: str
    rrf_score: float
    sources: list[str] = field(default_factory=list)  # このドキュメントが出現したソース名
    source_ranks: dict[str, int] = field(default_factory=dict)  # ソース別ランク


class FusionReranker:
    """RRF ベースの Fusion/Reranker。

    複数検索バックエンドの結果を Reciprocal Rank Fusion で統合する。

    Args:
        k: RRF 定数（大きいほど上位ランクの影響が緩和される）。
        source_weights: ソース別重み {"faiss": 1.0, "kg": 0.8, "sql": 1.2}。
            省略時は全て 1.0。
    """

    def __init__(
        self,
        k: int = _DEFAULT_K,
        source_weights: Optional[dict[str, float]] = None,
    ) -> None:
        self._k = k
        self._weights = source_weights or {}

    def fuse(
        self,
        ranked_lists: dict[str, list[tuple[str, float]]],
        top_k: Optional[int] = None,
    ) -> list[FusionResult]:
        """複数ランクリストを RRF で統合する。

        Args:
            ranked_lists: {"source_name": [(doc_id, score), ...]} の辞書。
                各リストはスコア降順を期待するが、順序で RRF 計算する。
            top_k: 返す上位件数（省略時は全件）。

        Returns:
            RRF スコア降順の FusionResult リスト。
        """
        if not ranked_lists:
            return []

        # doc_id → FusionResult の集計
        fusion_scores: dict[str, FusionResult] = {}

        for source_name, ranked in ranked_lists.items():
            weight = self._weights.get(source_name, 1.0)
            for rank_0based, (doc_id, _score) in enumerate(ranked):
                rank = rank_0based + 1  # 1-indexed
                rrf_contrib = weight / (self._k + rank)

                if doc_id not in fusion_scores:
                    fusion_scores[doc_id] = FusionResult(doc_id=doc_id, rrf_score=0.0)

                fusion_scores[doc_id].rrf_score += rrf_contrib
                if source_name not in fusion_scores[doc_id].sources:
                    fusion_scores[doc_id].sources.append(source_name)
                fusion_scores[doc_id].source_ranks[source_name] = rank

        results = sorted(fusion_scores.values(), key=lambda r: r.rrf_score, reverse=True)

        if top_k is not None:
            results = results[:top_k]

        logger.debug(
            "FusionReranker: %d sources, %d unique docs → top %d",
            len(ranked_lists),
            len(fusion_scores),
            len(results),
        )
        return results

    def fuse_doc_ids(
        self,
        ranked_lists: dict[str, list[tuple[str, float]]],
        top_k: Optional[int] = None,
    ) -> list[str]:
        """fuse() の簡略版: doc_id リストのみ返す。"""
        return [r.doc_id for r in self.fuse(ranked_lists, top_k=top_k)]

    def fuse_with_dedup(
        self,
        ranked_lists: dict[str, list[tuple[str, float]]],
        top_k: Optional[int] = None,
    ) -> list[FusionResult]:
        """重複 doc_id を除去した fuse() （fuse と同一、API 一貫性のため）。

        RRF は自動的に重複を集計するため、dedup は不要。
        このメソッドは fuse() のエイリアス。
        """
        return self.fuse(ranked_lists, top_k=top_k)

    @staticmethod
    def normalize_scores(
        items: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """スコアを [0, 1] に正規化する（前処理用ユーティリティ）。

        Args:
            items: [(doc_id, score), ...] リスト。

        Returns:
            正規化済みリスト（スコア順序は保持）。
        """
        if not items:
            return []
        scores = [s for _, s in items]
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return [(doc_id, 1.0) for doc_id, _ in items]
        return [
            (doc_id, (s - min_s) / (max_s - min_s))
            for doc_id, s in items
        ]
