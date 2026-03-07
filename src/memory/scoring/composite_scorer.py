"""src/memory/scoring/composite_scorer.py — 複合スコアラー

FreshnessScorer と UsefulnessScorer を統合し、Document の composite_score を計算・更新する。

使い方:
    from src.memory.scoring.composite_scorer import CompositeScorer

    scorer = CompositeScorer()
    score = scorer.compute_for_document(doc, now=datetime.utcnow())

    # MetadataStore 経由で一括更新
    await scorer.update_store(store, doc_ids)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from src.memory.schema import Document
from src.memory.scoring.freshness import FreshnessScorer
from src.memory.scoring.usefulness import UsefulnessScorer, UsefulnessWeights

logger = logging.getLogger(__name__)


class CompositeScorer:
    """FreshnessScorer + UsefulnessScorer を統合した複合スコアラー。

    Args:
        freshness_half_life: ドメイン別半減期（日数）マッピング。
        usefulness_weights: 有用性の重みパラメータ。
    """

    def __init__(
        self,
        freshness_half_life: Optional[dict[str, float]] = None,
        usefulness_weights: Optional[UsefulnessWeights] = None,
    ) -> None:
        self.freshness = FreshnessScorer(half_life_days=freshness_half_life)
        self.usefulness = UsefulnessScorer(weights=usefulness_weights)

    def compute_for_document(
        self,
        doc: Document,
        now: Optional[datetime] = None,
    ) -> float:
        """Document オブジェクトから複合スコアを計算する。

        Returns:
            0.0〜1.0 の複合スコア。
        """
        if now is None:
            now = datetime.now(timezone.utc)

        domain = doc.domain if isinstance(doc.domain, str) else doc.domain.value
        retrieved_at = doc.source.retrieved_at if doc.source else None

        freshness_score = self.freshness.score(domain, retrieved_at=retrieved_at, now=now)

        score = self.usefulness.compute(
            retrieval_count=doc.usefulness.retrieval_count,
            selection_count=doc.usefulness.selection_count,
            positive_feedback=doc.usefulness.positive_feedback,
            negative_feedback=doc.usefulness.negative_feedback,
            teacher_quality=doc.usefulness.teacher_quality,
            execution_success_rate=doc.usefulness.execution_success_rate,
            freshness=freshness_score,
            domain=domain,
        )
        return score

    def compute_from_row(self, row: dict, now: Optional[datetime] = None) -> float:
        """SQLite の行辞書から複合スコアを計算する。

        MetadataStore の行データを直接受け取り、スコアを計算する。
        """
        if now is None:
            now = datetime.now(timezone.utc)

        domain = row.get("domain", "general")

        # フレッシュネス計算
        retrieved_at_str = row.get("source_retrieved_at")
        retrieved_at: Optional[datetime] = None
        if retrieved_at_str:
            try:
                retrieved_at = datetime.fromisoformat(retrieved_at_str)
            except ValueError:
                logger.warning("Invalid retrieved_at format: %s", retrieved_at_str)

        freshness_score = self.freshness.score(domain, retrieved_at=retrieved_at, now=now)

        data = dict(row)
        data["freshness"] = freshness_score
        data["domain"] = domain

        return self.usefulness.compute_from_dict(data)

    async def update_store(
        self,
        store,  # MetadataStore (avoid circular import with type hint)
        doc_ids: Optional[list[str]] = None,
        now: Optional[datetime] = None,
    ) -> int:
        """MetadataStore 内のドキュメントの composite_score を更新する。

        Args:
            store: MetadataStore インスタンス。
            doc_ids: 更新対象 ID リスト。None の場合は全件更新。
            now: 現在日時（テスト用）。

        Returns:
            更新件数。
        """
        if now is None:
            now = datetime.now(timezone.utc)

        if doc_ids is None:
            # 全ドメインを取得（ドメイン一覧を動的取得）
            from src.memory.schema import Domain
            docs: list[Document] = []
            for domain in Domain:
                domain_docs = await store.list_by_domain(domain.value, limit=10000)
                docs.extend(domain_docs)
        else:
            docs_or_none = await store.get_batch(doc_ids)
            docs = [d for d in docs_or_none if d is not None]

        updated = 0
        for doc in docs:
            score = self.compute_for_document(doc, now=now)
            await store.update_quality(doc.id, composite_score=score)
            updated += 1

        logger.info("Updated composite_score for %d documents", updated)
        return updated
