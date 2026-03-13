"""src/memory/scoring/composite_scorer.py — 複合スコアラー

FreshnessScorer と UsefulnessScorer を統合し、Document の composite_score を計算・更新する。
TeacherRegistry の trust_score を乗算することで、信頼度の低い Teacher のデータを
検索結果で自動的に後退させる。

使い方:
    from src.memory.scoring.composite_scorer import CompositeScorer

    scorer = CompositeScorer()
    score = scorer.compute_for_document(doc, now=datetime.utcnow())

    # TeacherRegistry の trust_score を乗算する場合
    score = scorer.compute_for_document(doc, trust_score=0.3)

    # MetadataStore 経由で一括更新（trust_map を渡すと Teacher 信頼度を反映）
    trust_map = {"claude-opus-4-6": 1.0, "bad-model": 0.1}
    await scorer.update_store(store, doc_ids, trust_map=trust_map)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from src.memory.schema import Document
from src.memory.scoring.freshness import FreshnessScorer
from src.memory.scoring.usefulness import UsefulnessScorer, UsefulnessWeights

logger = logging.getLogger(__name__)


class CompositeScorer:
    """FreshnessScorer + UsefulnessScorer を統合した複合スコアラー。

    Teacher の trust_score を乗算することで、信頼度の低い Teacher のデータを
    検索結果で自動的に後退させる（trust_score=1.0 なら影響なし）。

    Args:
        freshness_half_life: ドメイン別半減期（日数）マッピング。
        usefulness_weights: 有用性の重みパラメータ。
        teacher_trust_weight: trust_score の影響度 (0.0〜1.0)。
            0.0 = trust を完全無視、1.0 = trust を直接乗算。
            デフォルト 1.0（trust_score そのままを乗算）。
    """

    def __init__(
        self,
        freshness_half_life: dict[str, float] | None = None,
        usefulness_weights: UsefulnessWeights | None = None,
        teacher_trust_weight: float = 1.0,
    ) -> None:
        self.freshness = FreshnessScorer(half_life_days=freshness_half_life)
        self.usefulness = UsefulnessScorer(weights=usefulness_weights)
        self._trust_weight = float(min(1.0, max(0.0, teacher_trust_weight)))

    def _apply_trust(self, base_score: float, trust_score: float) -> float:
        """trust_score をベーススコアに適用する。

        multiplier = (1 - trust_weight) + trust_weight * trust_score
        trust_weight=1.0 のとき: multiplier = trust_score （直接乗算）
        trust_weight=0.0 のとき: multiplier = 1.0 （trust 無視）

        これにより trust_score が 0.05 (_MIN_TRUST) でも
        teacher_trust_weight=1.0 ならスコアは 5% まで抑制される。
        """
        multiplier = (1.0 - self._trust_weight) + self._trust_weight * trust_score
        return float(min(1.0, max(0.0, base_score * multiplier)))

    def compute_for_document(
        self,
        doc: Document,
        now: datetime | None = None,
        trust_score: float = 1.0,
    ) -> float:
        """Document オブジェクトから複合スコアを計算する。

        Args:
            doc: 対象ドキュメント。
            now: 現在日時（テスト用）。
            trust_score: Teacher の信頼スコア (0.0〜1.0)。
                TeacherRegistry から取得した値を渡す。
                デフォルト 1.0（Teacher 不明時は信頼）。

        Returns:
            0.0〜1.0 の複合スコア（trust_score を反映）。
        """
        if now is None:
            now = datetime.now(UTC)

        domain = doc.domain if isinstance(doc.domain, str) else doc.domain.value
        retrieved_at = doc.source.retrieved_at if doc.source else None

        freshness_score = self.freshness.score(domain, retrieved_at=retrieved_at, now=now)

        base_score = self.usefulness.compute(
            retrieval_count=doc.usefulness.retrieval_count,
            selection_count=doc.usefulness.selection_count,
            positive_feedback=doc.usefulness.positive_feedback,
            negative_feedback=doc.usefulness.negative_feedback,
            teacher_quality=doc.usefulness.teacher_quality,
            execution_success_rate=doc.usefulness.execution_success_rate,
            freshness=freshness_score,
            domain=domain,
        )
        return self._apply_trust(base_score, trust_score)

    def compute_from_row(
        self,
        row: dict,
        now: datetime | None = None,
        trust_score: float = 1.0,
    ) -> float:
        """SQLite の行辞書から複合スコアを計算する。

        MetadataStore の行データを直接受け取り、スコアを計算する。

        Args:
            row: MetadataStore の行辞書。
            now: 現在日時（テスト用）。
            trust_score: Teacher の信頼スコア (0.0〜1.0)。
        """
        if now is None:
            now = datetime.now(UTC)

        domain = row.get("domain", "general")

        # フレッシュネス計算
        retrieved_at_str = row.get("source_retrieved_at")
        retrieved_at: datetime | None = None
        if retrieved_at_str:
            try:
                retrieved_at = datetime.fromisoformat(retrieved_at_str)
            except ValueError:
                logger.warning("Invalid retrieved_at format: %s", retrieved_at_str)

        freshness_score = self.freshness.score(domain, retrieved_at=retrieved_at, now=now)

        data = dict(row)
        data["freshness"] = freshness_score
        data["domain"] = domain

        base_score = self.usefulness.compute_from_dict(data)
        return self._apply_trust(base_score, trust_score)

    async def update_store(
        self,
        store,  # MetadataStore (avoid circular import with type hint)
        doc_ids: list[str] | None = None,
        now: datetime | None = None,
        trust_map: dict[str, float] | None = None,
    ) -> int:
        """MetadataStore 内のドキュメントの composite_score を更新する。

        Args:
            store: MetadataStore インスタンス。
            doc_ids: 更新対象 ID リスト。None の場合は全件更新。
            now: 現在日時（テスト用）。
            trust_map: {teacher_id: trust_score} の辞書。
                TeacherRegistry.list_all() の結果から構築して渡す。
                None の場合は trust_score=1.0 として扱う（信頼度反映なし）。

        Returns:
            更新件数。
        """
        if now is None:
            now = datetime.now(UTC)

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
            # teacher_id があれば trust_map から trust_score を引く
            trust: float = 1.0
            if trust_map and doc.source.teacher_id:
                trust = trust_map.get(doc.source.teacher_id, 1.0)
            score = self.compute_for_document(doc, now=now, trust_score=trust)
            await store.update_quality(doc.id, composite_score=score)
            updated += 1

        logger.info("Updated composite_score for %d documents (trust_map=%s)",
                    updated, "provided" if trust_map else "none")
        return updated

    @staticmethod
    def build_trust_map(profiles) -> dict[str, float]:
        """TeacherProfile のリストから trust_map を構築する。

        Args:
            profiles: TeacherRegistry.list_all() が返す TeacherProfile のリスト。

        Returns:
            {teacher_id: trust_score} の辞書。
        """
        return {p.teacher_id: p.trust_score for p in profiles}
