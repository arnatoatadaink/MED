"""src/memory/scoring/usefulness.py — 多面的有用性スコア計算

ドキュメントの有用性を多面的指標から計算する。
- フィードバック比率（肯定 / 全フィードバック）
- 選択率（選択回数 / 検索ヒット回数）
- Teacher 品質スコア（外部審査）
- 実行成功率（コードドキュメント）
- フレッシュネス（時間減衰）

使い方:
    from src.memory.scoring.usefulness import UsefulnessScorer

    scorer = UsefulnessScorer()
    score = scorer.compute(
        retrieval_count=10,
        selection_count=4,
        positive_feedback=8,
        negative_feedback=2,
        teacher_quality=0.8,
        execution_success_rate=0.9,
        freshness=0.7,
    )  # → float 0.0〜1.0
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UsefulnessWeights:
    """有用性スコアの重みパラメータ。合計が 1.0 になること。"""

    feedback: float = 0.30
    selection: float = 0.25
    teacher_quality: float = 0.25
    execution: float = 0.10
    freshness: float = 0.10

    def __post_init__(self) -> None:
        total = sum([
            self.feedback,
            self.selection,
            self.teacher_quality,
            self.execution,
            self.freshness,
        ])
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"UsefulnessWeights must sum to 1.0, got {total:.4f}")


class UsefulnessScorer:
    """多面的指標から有用性スコアを計算する。

    各指標は 0.0〜1.0 に正規化してから重み付き合計を行う。

    Args:
        weights: 各指標の重み。省略時はデフォルト値を使用。
    """

    def __init__(self, weights: UsefulnessWeights | None = None) -> None:
        self._weights = weights or UsefulnessWeights()

    def compute(
        self,
        retrieval_count: int = 0,
        selection_count: int = 0,
        positive_feedback: int = 0,
        negative_feedback: int = 0,
        teacher_quality: float = 0.0,
        execution_success_rate: float = 0.0,
        freshness: float = 1.0,
        domain: str = "general",
    ) -> float:
        """有用性スコアを計算する。

        Args:
            retrieval_count: FAISS 検索でヒットした回数。
            selection_count: ユーザーが実際に利用した回数。
            positive_feedback: 肯定的フィードバック数。
            negative_feedback: 否定的フィードバック数。
            teacher_quality: Teacher Model による品質スコア (0.0〜1.0)。
            execution_success_rate: コード実行成功率 (0.0〜1.0)。
            freshness: フレッシュネススコア (0.0〜1.0)。
            domain: ドメイン名（コードドメインは実行成功率を重視）。

        Returns:
            0.0〜1.0 の有用性スコア。
        """
        w = self._weights

        # フィードバック比率
        total_fb = positive_feedback + negative_feedback
        feedback_score = positive_feedback / total_fb if total_fb > 0 else 0.5

        # 選択率
        selection_score = selection_count / retrieval_count if retrieval_count > 0 else 0.5

        # Teacher 品質スコア（そのまま使用）
        quality_score = float(min(1.0, max(0.0, teacher_quality)))

        # 実行成功率（コードドメインのみ有効、他は teacher_quality で代替）
        exec_score: float
        if domain == "code":
            exec_score = float(min(1.0, max(0.0, execution_success_rate)))
        else:
            exec_score = quality_score  # コード以外はteacher品質で代用

        # フレッシュネス
        fresh_score = float(min(1.0, max(0.0, freshness)))

        score = (
            w.feedback * feedback_score
            + w.selection * selection_score
            + w.teacher_quality * quality_score
            + w.execution * exec_score
            + w.freshness * fresh_score
        )
        return float(min(1.0, max(0.0, score)))

    def compute_from_dict(self, data: dict) -> float:
        """辞書から有用性スコアを計算する（MetadataStore の行データ向け）。"""
        return self.compute(
            retrieval_count=data.get("retrieval_count", 0),
            selection_count=data.get("selection_count", 0),
            positive_feedback=data.get("positive_feedback", 0),
            negative_feedback=data.get("negative_feedback", 0),
            teacher_quality=data.get("teacher_quality", 0.0),
            execution_success_rate=data.get("execution_success_rate", 0.0),
            freshness=data.get("freshness", 1.0),
            domain=data.get("domain", "general"),
        )
