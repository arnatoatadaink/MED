"""src/training/evaluation/benchmark_suite.py — ベンチマークスイート

複数の評価タスクを一括実行し、Student モデルの総合性能を測定する。

ベンチマークセット:
- code_generation: コード生成タスク (HumanEval スタイル)
- qa_retrieval: Q&A + FAISS 検索タスク
- math_reasoning: 数学的推論タスク (GSM8K スタイル)
- memory_utilization: メモリ活用タスク

使い方:
    from src.training.evaluation.benchmark_suite import BenchmarkSuite

    suite = BenchmarkSuite(gateway, memory_manager)
    report = await suite.run(student_model, benchmark_names=["qa_retrieval"])
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional

from src.llm.gateway import LLMGateway
from src.training.evaluation.student_evaluator import EvalMetrics, EvalSample, StudentEvaluator

logger = logging.getLogger(__name__)

# サンプルベンチマークデータ
_CODE_SAMPLES = [
    EvalSample(
        query="Write a Python function to check if a number is prime.",
        expected_answer="def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True",
        domain="code",
        metadata={"requires_code": True},
    ),
    EvalSample(
        query="Implement binary search in Python.",
        expected_answer="def binary_search(arr, target):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        mid = (lo+hi)//2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: lo = mid+1\n        else: hi = mid-1\n    return -1",
        domain="code",
        metadata={"requires_code": True},
    ),
]

_QA_SAMPLES = [
    EvalSample(
        query="What is FAISS and what is it used for?",
        expected_answer="FAISS (Facebook AI Similarity Search) is a library for efficient similarity search of dense vectors.",
        domain="general",
    ),
    EvalSample(
        query="Explain the difference between TinyLoRA and standard LoRA.",
        expected_answer="TinyLoRA uses only 13 parameters by freezing A and learning only B with tied weights, while LoRA trains both A and B matrices.",
        domain="general",
    ),
    EvalSample(
        query="What is Reciprocal Rank Fusion (RRF)?",
        expected_answer="RRF is a method to combine multiple ranked lists by scoring each item as the sum of 1/(k+rank_i) across lists.",
        domain="academic",
    ),
]

_MATH_SAMPLES = [
    EvalSample(
        query="If I have 12 apples and give away 1/3, how many do I have left?",
        expected_answer="8",
        domain="general",
        metadata={"answer": "8"},
    ),
    EvalSample(
        query="What is the derivative of x^3 + 2x^2 - 5x + 1?",
        expected_answer="3x^2 + 4x - 5",
        domain="academic",
        metadata={"answer": "3x^2 + 4x - 5"},
    ),
]


@dataclass
class BenchmarkReport:
    """ベンチマーク総合レポート。"""

    benchmark_results: dict[str, EvalMetrics] = field(default_factory=dict)
    overall_score: float = 0.0
    elapsed_seconds: float = 0.0

    def summary(self) -> dict[str, Any]:
        per_bench = {
            name: metrics.to_dict()
            for name, metrics in self.benchmark_results.items()
        }
        return {
            "overall_score": round(self.overall_score, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "benchmarks": per_bench,
        }

    def __str__(self) -> str:
        lines = [f"BenchmarkReport (overall={self.overall_score:.4f}):"]
        for name, m in self.benchmark_results.items():
            lines.append(
                f"  {name}: reward={m.avg_reward:.3f} quality={m.answer_quality:.3f} "
                f"exec={m.exec_success_rate:.3f}"
            )
        return "\n".join(lines)


# ベンチマーク登録テーブル
_BENCHMARK_SAMPLES: dict[str, list[EvalSample]] = {
    "code_generation": _CODE_SAMPLES,
    "qa_retrieval": _QA_SAMPLES,
    "math_reasoning": _MATH_SAMPLES,
}


class BenchmarkSuite:
    """ベンチマークスイート実行器。

    Args:
        gateway: Teacher LLM（評価に使用）。
        memory_manager: MemoryManager（省略可）。
        reward_fn: RewardFunction（省略時は品質スコアのみ）。
        custom_benchmarks: 追加ベンチマークデータ {"name": [EvalSample, ...]}。
    """

    def __init__(
        self,
        gateway: LLMGateway,
        memory_manager: Optional[Any] = None,
        reward_fn: Optional[Any] = None,
        custom_benchmarks: Optional[dict[str, list[EvalSample]]] = None,
    ) -> None:
        self._gateway = gateway
        self._memory = memory_manager
        self._reward_fn = reward_fn
        self._evaluator = StudentEvaluator(
            gateway=gateway,
            memory_manager=memory_manager,
            reward_fn=reward_fn,
        )
        self._benchmarks = dict(_BENCHMARK_SAMPLES)
        if custom_benchmarks:
            self._benchmarks.update(custom_benchmarks)

    def list_benchmarks(self) -> list[str]:
        """利用可能なベンチマーク名リストを返す。"""
        return sorted(self._benchmarks.keys())

    def register(self, name: str, samples: list[EvalSample]) -> None:
        """カスタムベンチマークを登録する。"""
        self._benchmarks[name] = samples
        logger.info("BenchmarkSuite: registered '%s' (%d samples)", name, len(samples))

    async def run(
        self,
        student_model: Optional[Any] = None,
        benchmark_names: Optional[list[str]] = None,
    ) -> BenchmarkReport:
        """ベンチマークを実行する。

        Args:
            student_model: 評価対象モデル（callable or None）。
            benchmark_names: 実行するベンチマーク名リスト（None で全て）。

        Returns:
            BenchmarkReport。
        """
        start = time.time()
        names = benchmark_names or self.list_benchmarks()
        results: dict[str, EvalMetrics] = {}

        for name in names:
            if name not in self._benchmarks:
                logger.warning("BenchmarkSuite: unknown benchmark '%s'", name)
                continue
            samples = self._benchmarks[name]
            logger.info("BenchmarkSuite: running '%s' (%d samples)", name, len(samples))
            metrics = await self._evaluator.evaluate(samples, student_model=student_model)
            results[name] = metrics
            logger.info(
                "BenchmarkSuite: '%s' done — reward=%.3f quality=%.3f",
                name, metrics.avg_reward, metrics.answer_quality,
            )

        overall = self._compute_overall(results)
        elapsed = time.time() - start
        report = BenchmarkReport(
            benchmark_results=results,
            overall_score=overall,
            elapsed_seconds=elapsed,
        )
        logger.info("BenchmarkSuite complete: %s", report)
        return report

    def _compute_overall(self, results: dict[str, EvalMetrics]) -> float:
        """全ベンチマークの加重平均スコアを計算する。"""
        if not results:
            return 0.0
        scores = [m.avg_reward * 0.5 + m.answer_quality * 0.5 for m in results.values()]
        return sum(scores) / len(scores)
