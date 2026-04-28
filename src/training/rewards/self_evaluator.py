"""src/training/rewards/self_evaluator.py — 自己評価 → GRPO 報酬変換パイプライン

Teacher LLM が出力した自己評価 (accuracy / relevance / completeness) を
GRPO 報酬スカラー (0.0-1.0) に変換する。

使い方:
    evaluator = SelfEvaluator()

    # Teacher LLM の自己評価 JSON から報酬を計算
    self_eval = {
        "accuracy": 0.85,
        "relevance": 0.9,
        "completeness": 0.75,
        "improvement_notes": "参照ドキュメントが不足していた",
    }
    reward = evaluator.to_reward(self_eval)  # 0.0-1.0

    # ThoughtLog を構築して保存
    log = evaluator.build_log(
        input_text=query,
        output_text=answer,
        reasoning_steps=[{"step": 1, "thought": "...", "confidence": 0.8}],
        self_eval=self_eval,
        pattern_id="retrieval_v1",
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.memory.schema import ThoughtLog

logger = logging.getLogger(__name__)

# GRPO 報酬計算の重み (合計 1.0)
_ACCURACY_WEIGHT = 0.40
_RELEVANCE_WEIGHT = 0.35
_COMPLETENESS_WEIGHT = 0.25


@dataclass
class SelfEval:
    """self_evaluate() の出力構造。

    各スコアは 0.0-1.0 の範囲。Teacher LLM がこの構造を JSON で返す想定。
    """

    accuracy: float = 0.0
    relevance: float = 0.0
    completeness: float = 0.0
    improvement_notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """ThoughtLog.self_eval に保存する dict 形式に変換する。"""
        return {
            "accuracy": self.accuracy,
            "relevance": self.relevance,
            "completeness": self.completeness,
            "improvement_notes": self.improvement_notes,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SelfEval:
        """dict から SelfEval を復元する。"""
        known = {"accuracy", "relevance", "completeness", "improvement_notes"}
        return cls(
            accuracy=float(d.get("accuracy", 0.0)),
            relevance=float(d.get("relevance", 0.0)),
            completeness=float(d.get("completeness", 0.0)),
            improvement_notes=str(d.get("improvement_notes", "")),
            extra={k: v for k, v in d.items() if k not in known},
        )


class SelfEvaluator:
    """自己評価 → GRPO 報酬変換パイプライン。

    Teacher LLM が生成した SelfEval を受け取り、
    加重平均によって GRPO 報酬スカラーを計算する。

    LLM 呼び出しは含まない（N-4 で実装予定）。
    このクラスは変換ロジックと ThoughtLog 構築を担う。
    """

    def __init__(
        self,
        accuracy_weight: float = _ACCURACY_WEIGHT,
        relevance_weight: float = _RELEVANCE_WEIGHT,
        completeness_weight: float = _COMPLETENESS_WEIGHT,
    ) -> None:
        total = accuracy_weight + relevance_weight + completeness_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.4f}"
            )
        self._w_acc = accuracy_weight
        self._w_rel = relevance_weight
        self._w_com = completeness_weight

    def to_reward(self, self_eval: SelfEval | dict[str, Any]) -> float:
        """SelfEval を GRPO 報酬スカラー (0.0-1.0) に変換する。

        Args:
            self_eval: SelfEval オブジェクトまたは dict。

        Returns:
            加重平均スコア (0.0-1.0)。
        """
        if isinstance(self_eval, dict):
            self_eval = SelfEval.from_dict(self_eval)

        reward = (
            self._w_acc * self_eval.accuracy
            + self._w_rel * self_eval.relevance
            + self._w_com * self_eval.completeness
        )
        return max(0.0, min(1.0, reward))

    def build_log(
        self,
        input_text: str,
        output_text: str,
        self_eval: SelfEval | dict[str, Any],
        reasoning_steps: list[dict[str, Any]] | None = None,
        pattern_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> ThoughtLog:
        """SelfEval から ThoughtLog を構築する。

        Args:
            input_text:     クエリ / Student への入力テキスト。
            output_text:    応答 / Student の出力テキスト。
            self_eval:      自己評価 (SelfEval or dict)。
            reasoning_steps: 推論ステップリスト [{step, thought, confidence}]。
            pattern_id:     パターン識別子（PatternExtractor が付与）。
            timestamp:      ログ時刻（省略時は現在時刻）。

        Returns:
            保存可能な ThoughtLog。
        """
        if isinstance(self_eval, dict):
            self_eval = SelfEval.from_dict(self_eval)

        reward = self.to_reward(self_eval)
        return ThoughtLog(
            timestamp=timestamp or datetime.utcnow(),
            input=input_text,
            reasoning=reasoning_steps or [],
            output=output_text,
            reward=reward,
            self_eval=self_eval.to_dict(),
            pattern_id=pattern_id,
        )
