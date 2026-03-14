"""src/training/evaluation/teacher_comparison.py — Teacher vs Student 比較評価

Teacher Model の出力を基準として Student の相対性能を測定する。

使い方:
    from src.training.evaluation.teacher_comparison import TeacherComparison

    comparison = TeacherComparison(gateway)
    result = await comparison.compare(query, student_answer, teacher_answer)
    # → ComparisonResult(winner="teacher", quality_gap=0.15, ...)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.llm.gateway import LLMGateway

logger = logging.getLogger(__name__)

_COMPARE_SYSTEM = """\
You are an expert evaluator comparing two answers to the same question.
Respond with ONLY valid JSON:
{
  "student_score": 0.0-1.0,
  "teacher_score": 0.0-1.0,
  "winner": "student|teacher|tie",
  "quality_gap": 0.0-1.0,
  "feedback": "brief feedback on student answer"
}"""

_COMPARE_PROMPT = """\
Question: {query}

Answer A (Student): {student_answer}

Answer B (Teacher): {teacher_answer}

Compare these answers."""


@dataclass
class ComparisonResult:
    """比較評価結果。"""

    query: str
    student_score: float = 0.0
    teacher_score: float = 0.0
    winner: str = "tie"         # "student" | "teacher" | "tie"
    quality_gap: float = 0.0   # teacher_score - student_score
    feedback: str = ""

    @property
    def student_wins(self) -> bool:
        return self.winner == "student"

    @property
    def relative_performance(self) -> float:
        """Student の Teacher に対する相対性能 (0〜1)。"""
        if self.teacher_score == 0:
            return 1.0
        return min(1.0, self.student_score / self.teacher_score)


@dataclass
class BatchComparisonResult:
    """バッチ比較評価結果。"""

    results: list[ComparisonResult] = field(default_factory=list)

    @property
    def avg_student_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.student_score for r in self.results) / len(self.results)

    @property
    def avg_teacher_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.teacher_score for r in self.results) / len(self.results)

    @property
    def avg_quality_gap(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.quality_gap for r in self.results) / len(self.results)

    @property
    def student_win_rate(self) -> float:
        if not self.results:
            return 0.0
        wins = sum(1 for r in self.results if r.student_wins)
        return wins / len(self.results)

    def summary(self) -> dict:
        return {
            "n_samples": len(self.results),
            "avg_student_score": round(self.avg_student_score, 4),
            "avg_teacher_score": round(self.avg_teacher_score, 4),
            "avg_quality_gap": round(self.avg_quality_gap, 4),
            "student_win_rate": round(self.student_win_rate, 4),
        }


class TeacherComparison:
    """Teacher vs Student 比較評価器。

    Args:
        gateway: Teacher LLM（評価者として使用）。
        provider: 優先プロバイダ。
        concurrency: 並列評価数。
    """

    def __init__(
        self,
        gateway: LLMGateway,
        provider: str | None = None,
        concurrency: int = 4,
    ) -> None:
        self._gateway = gateway
        self._provider = provider
        self._concurrency = concurrency

    async def compare(
        self,
        query: str,
        student_answer: str,
        teacher_answer: str,
    ) -> ComparisonResult:
        """1 クエリの Teacher vs Student 比較を行う。"""
        import json
        try:
            response = await self._gateway.complete(
                _COMPARE_PROMPT.format(
                    query=query[:300],
                    student_answer=student_answer[:500],
                    teacher_answer=teacher_answer[:500],
                ),
                system=_COMPARE_SYSTEM,
                provider=self._provider,
                max_tokens=200,
                temperature=0.0,
            )
            content = re.sub(r"```(?:json)?\s*", "", response.content).strip().rstrip("`")
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                student_score = float(data.get("student_score", 0.5))
                teacher_score = float(data.get("teacher_score", 0.8))
                return ComparisonResult(
                    query=query,
                    student_score=student_score,
                    teacher_score=teacher_score,
                    winner=data.get("winner", "teacher"),
                    quality_gap=float(data.get("quality_gap", teacher_score - student_score)),
                    feedback=str(data.get("feedback", "")),
                )
        except Exception:
            logger.exception("TeacherComparison failed for: %r", query[:50])

        # フォールバック
        return ComparisonResult(
            query=query,
            student_score=0.5,
            teacher_score=0.8,
            winner="teacher",
            quality_gap=0.3,
            feedback="evaluation failed",
        )

    async def compare_batch(
        self,
        queries: list[str],
        student_answers: list[str],
        teacher_answers: list[str],
    ) -> BatchComparisonResult:
        """複数クエリを並列比較する。"""
        import asyncio
        semaphore = asyncio.Semaphore(self._concurrency)

        async def _compare_one(query, s_ans, t_ans):
            async with semaphore:
                return await self.compare(query, s_ans, t_ans)

        results = await asyncio.gather(*[
            _compare_one(q, s, t)
            for q, s, t in zip(queries, student_answers, teacher_answers)
        ])
        return BatchComparisonResult(results=list(results))

    async def generate_teacher_answers(
        self,
        queries: list[str],
    ) -> list[str]:
        """Teacher LLM で参照回答を生成する。"""
        import asyncio
        semaphore = asyncio.Semaphore(self._concurrency)

        async def _gen(query: str) -> str:
            async with semaphore:
                try:
                    resp = await self._gateway.complete(
                        query,
                        provider=self._provider,
                        max_tokens=500,
                        temperature=0.3,
                    )
                    return resp.content
                except Exception:
                    return ""

        return list(await asyncio.gather(*[_gen(q) for q in queries]))
