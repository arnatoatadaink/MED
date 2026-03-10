"""src/memory/maturation/quality_metrics.py — メモリ品質メトリクス

FAISSメモリ全体の品質を計測するユーティリティ。
Phase 2 品質目標の達成度を数値で報告する。

品質目標（CLAUDE.md）:
- 10,000 docs
- confidence > 0.7
- 実行成功率 > 80%

使い方:
    from src.memory.maturation.quality_metrics import QualityMetrics

    metrics = QualityMetrics(store)
    report = await metrics.compute()
    print(report.meets_phase2_goal)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Phase 2 品質目標
_PHASE2_DOC_TARGET = 10_000
_PHASE2_CONFIDENCE_TARGET = 0.7
_PHASE2_EXEC_SUCCESS_TARGET = 0.8


@dataclass
class QualityReport:
    """品質メトリクスレポート。"""

    total_docs: int = 0
    approved_docs: int = 0
    rejected_docs: int = 0
    pending_docs: int = 0

    avg_confidence: float = 0.0
    avg_teacher_quality: float = 0.0
    avg_composite_score: float = 0.0

    exec_success_rate: float = 0.0    # コード実行成功率
    avg_retrieval_count: float = 0.0  # 平均検索回数
    avg_selection_count: float = 0.0  # 平均選択回数

    # 難易度分布
    difficulty_distribution: dict[str, int] = field(default_factory=dict)

    # Phase 2 目標達成フラグ
    doc_target: int = _PHASE2_DOC_TARGET
    confidence_target: float = _PHASE2_CONFIDENCE_TARGET
    exec_success_target: float = _PHASE2_EXEC_SUCCESS_TARGET

    @property
    def approval_rate(self) -> float:
        """承認率 (0.0〜1.0)。"""
        if self.total_docs == 0:
            return 0.0
        return self.approved_docs / self.total_docs

    @property
    def meets_doc_target(self) -> bool:
        return self.total_docs >= self.doc_target

    @property
    def meets_confidence_target(self) -> bool:
        return self.avg_confidence >= self.confidence_target

    @property
    def meets_exec_success_target(self) -> bool:
        return self.exec_success_rate >= self.exec_success_target

    @property
    def meets_phase2_goal(self) -> bool:
        """Phase 2 目標を全て満たすか。"""
        return (
            self.meets_doc_target
            and self.meets_confidence_target
            and self.meets_exec_success_target
        )

    @property
    def phase2_progress(self) -> dict[str, float]:
        """Phase 2 目標に対する進捗 (0.0〜1.0)。"""
        return {
            "docs": min(1.0, self.total_docs / self.doc_target) if self.doc_target > 0 else 0.0,
            "confidence": min(1.0, self.avg_confidence / self.confidence_target) if self.confidence_target > 0 else 0.0,
            "exec_success": min(1.0, self.exec_success_rate / self.exec_success_target) if self.exec_success_target > 0 else 0.0,
        }

    def summary(self) -> str:
        """人間可読なサマリー文字列を返す。"""
        lines = [
            f"=== Memory Quality Report ===",
            f"Total docs:         {self.total_docs:,} / {self.doc_target:,} "
            f"({'✓' if self.meets_doc_target else '✗'})",
            f"Avg confidence:     {self.avg_confidence:.3f} / {self.confidence_target:.1f} "
            f"({'✓' if self.meets_confidence_target else '✗'})",
            f"Exec success rate:  {self.exec_success_rate:.1%} / {self.exec_success_target:.0%} "
            f"({'✓' if self.meets_exec_success_target else '✗'})",
            f"Approval rate:      {self.approval_rate:.1%} "
            f"(approved={self.approved_docs}, rejected={self.rejected_docs}, pending={self.pending_docs})",
            f"Avg teacher quality:{self.avg_teacher_quality:.3f}",
            f"Avg composite:      {self.avg_composite_score:.3f}",
            f"Phase 2 goal:       {'MET ✓' if self.meets_phase2_goal else 'NOT MET ✗'}",
        ]
        if self.difficulty_distribution:
            lines.append(f"Difficulty: {self.difficulty_distribution}")
        return "\n".join(lines)


class QualityMetrics:
    """FAISSメモリ品質メトリクス計算クラス。

    Args:
        store: MetadataStore インスタンス。
    """

    def __init__(self, store: object) -> None:
        self._store = store

    async def compute(self, domain: Optional[str] = None) -> QualityReport:
        """メモリ全体の品質メトリクスを計算する。

        Args:
            domain: ドメイン絞り込み（None で全ドメイン）。

        Returns:
            QualityReport。
        """
        try:
            stats = await self._store.get_stats(domain=domain)
        except Exception:
            logger.exception("Failed to get stats from store")
            return QualityReport()

        report = QualityReport(
            total_docs=stats.get("total_docs", 0),
            approved_docs=stats.get("approved_docs", 0),
            rejected_docs=stats.get("rejected_docs", 0),
            pending_docs=stats.get("pending_docs", 0),
            avg_confidence=stats.get("avg_confidence", 0.0),
            avg_teacher_quality=stats.get("avg_teacher_quality", 0.0),
            avg_composite_score=stats.get("avg_composite_score", 0.0),
            exec_success_rate=stats.get("exec_success_rate", 0.0),
            avg_retrieval_count=stats.get("avg_retrieval_count", 0.0),
            avg_selection_count=stats.get("avg_selection_count", 0.0),
            difficulty_distribution=stats.get("difficulty_distribution", {}),
        )

        logger.info(
            "QualityMetrics: total=%d approved=%d confidence=%.3f exec=%.2f",
            report.total_docs, report.approved_docs,
            report.avg_confidence, report.exec_success_rate,
        )
        return report

    async def check_phase2_readiness(self) -> tuple[bool, list[str]]:
        """Phase 2 目標達成状況をチェックする。

        Returns:
            (ready, missing_criteria) タプル。
        """
        report = await self.compute()
        missing = []

        if not report.meets_doc_target:
            missing.append(
                f"docs: {report.total_docs}/{report.doc_target}"
            )
        if not report.meets_confidence_target:
            missing.append(
                f"confidence: {report.avg_confidence:.3f}/{report.confidence_target}"
            )
        if not report.meets_exec_success_target:
            missing.append(
                f"exec_success: {report.exec_success_rate:.1%}/{report.exec_success_target:.0%}"
            )

        return report.meets_phase2_goal, missing

    @staticmethod
    def from_doc_list(docs: list) -> QualityReport:
        """ドキュメントリストから品質レポートを生成する（ストアなし）。

        Args:
            docs: Document オブジェクトのリスト。

        Returns:
            QualityReport。
        """
        if not docs:
            return QualityReport()

        total = len(docs)
        approved = sum(1 for d in docs if getattr(d, "review_status", "") == "approved")
        rejected = sum(1 for d in docs if getattr(d, "review_status", "") == "rejected")
        pending = total - approved - rejected

        confidences = [getattr(d, "confidence", 0.0) for d in docs]
        qualities = [getattr(d, "teacher_quality", 0.0) for d in docs]
        composites = [getattr(d, "composite_score", 0.0) for d in docs]
        exec_rates = [getattr(d, "execution_success_rate", 0.0) for d in docs]
        retrieval_counts = [getattr(d, "retrieval_count", 0) for d in docs]
        selection_counts = [getattr(d, "selection_count", 0) for d in docs]

        difficulty_dist: dict[str, int] = {}
        for d in docs:
            diff = str(getattr(d, "difficulty", "unknown"))
            difficulty_dist[diff] = difficulty_dist.get(diff, 0) + 1

        def _mean(lst: list[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        return QualityReport(
            total_docs=total,
            approved_docs=approved,
            rejected_docs=rejected,
            pending_docs=pending,
            avg_confidence=_mean(confidences),
            avg_teacher_quality=_mean(qualities),
            avg_composite_score=_mean(composites),
            exec_success_rate=_mean(exec_rates),
            avg_retrieval_count=_mean(retrieval_counts),
            avg_selection_count=_mean(selection_counts),
            difficulty_distribution=difficulty_dist,
        )
