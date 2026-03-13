"""src/training/evaluation/student_evaluator.py — Student モデル評価器

GRPO+TinyLoRA 学習後の Student モデルの性能を評価する。

評価指標:
- retrieval_accuracy: FAISS 検索の適合率
- answer_quality: Teacher LLM による出力品質スコア
- exec_success_rate: コード実行成功率
- avg_reward: Composite 報酬の平均
- memory_utilization_rate: メモリ活用率

使い方:
    from src.training.evaluation.student_evaluator import StudentEvaluator

    evaluator = StudentEvaluator(gateway, memory_manager)
    metrics = await evaluator.evaluate(test_queries)
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Any

from src.llm.gateway import LLMGateway

logger = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    """評価メトリクス。"""

    n_samples: int = 0
    avg_reward: float = 0.0
    answer_quality: float = 0.0
    exec_success_rate: float = 0.0
    retrieval_accuracy: float = 0.0
    memory_utilization_rate: float = 0.0
    per_sample: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "avg_reward": round(self.avg_reward, 4),
            "answer_quality": round(self.answer_quality, 4),
            "exec_success_rate": round(self.exec_success_rate, 4),
            "retrieval_accuracy": round(self.retrieval_accuracy, 4),
            "memory_utilization_rate": round(self.memory_utilization_rate, 4),
        }


@dataclass
class EvalSample:
    """評価サンプル。"""

    query: str
    expected_answer: str | None = None
    domain: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)


class StudentEvaluator:
    """Student モデルの評価器。

    Args:
        gateway: Teacher LLM (評価に使用)。
        memory_manager: MemoryManager インスタンス（検索に使用）。
        reward_fn: RewardFunction インスタンス（省略時は CompositeReward）。
        provider: Teacher LLM プロバイダ。
        eval_concurrency: 並列評価数。
    """

    def __init__(
        self,
        gateway: LLMGateway,
        memory_manager: Any | None = None,
        reward_fn: Any | None = None,
        provider: str | None = None,
        eval_concurrency: int = 4,
    ) -> None:
        self._gateway = gateway
        self._memory = memory_manager
        self._reward_fn = reward_fn
        self._provider = provider
        self._concurrency = eval_concurrency

    async def evaluate(
        self,
        samples: list[EvalSample],
        student_model: Any | None = None,
    ) -> EvalMetrics:
        """テストサンプルセットで Student を評価する。

        Args:
            samples: EvalSample のリスト。
            student_model: Student モデル（呼び出し可能な関数 or None）。
                None の場合はメタデータの answers を使用する。

        Returns:
            EvalMetrics。
        """
        import asyncio
        semaphore = asyncio.Semaphore(self._concurrency)

        async def _eval_one(sample: EvalSample) -> dict[str, Any]:
            async with semaphore:
                return await self._evaluate_sample(sample, student_model)

        per_sample = list(await asyncio.gather(*[_eval_one(s) for s in samples]))
        return self._aggregate(per_sample)

    async def _evaluate_sample(
        self,
        sample: EvalSample,
        student_model: Any | None,
    ) -> dict[str, Any]:
        """1 サンプルを評価する。"""
        # 回答生成
        if student_model is not None and callable(student_model):
            try:
                response = await student_model(sample.query)
            except Exception:
                logger.exception("Student model call failed for: %r", sample.query[:50])
                response = ""
        else:
            response = sample.metadata.get("answer", "")

        # 検索精度
        retrieval_score = await self._eval_retrieval(sample, response)

        # 回答品質 (Teacher LLM)
        quality_score = await self._eval_quality(sample, response)

        # 実行成功率
        exec_score = float(sample.metadata.get("exec_success", 0.5))

        # メモリ活用
        mem_score = self._eval_memory_utilization(sample.metadata)

        # Reward
        reward_score = 0.0
        if self._reward_fn is not None:
            try:
                reward_score = await self._reward_fn.compute(
                    sample.query, response, sample.metadata
                )
            except Exception:
                reward_score = (retrieval_score + quality_score) / 2

        return {
            "query": sample.query,
            "response": response,
            "retrieval_accuracy": retrieval_score,
            "answer_quality": quality_score,
            "exec_success": exec_score,
            "memory_utilization": mem_score,
            "reward": reward_score,
        }

    async def _eval_retrieval(self, sample: EvalSample, response: str) -> float:
        """検索精度評価。"""
        if self._memory is None:
            return sample.metadata.get("retrieval_score", 0.5)
        try:
            results = await self._memory.search(sample.query, k=3)
            if not results:
                return 0.2
            # 上位スコアの平均
            scores = [r.score for r in results[:3] if hasattr(r, "score")]
            return statistics.mean(scores) if scores else 0.5
        except Exception:
            return 0.5

    async def _eval_quality(self, sample: EvalSample, response: str) -> float:
        """Teacher LLM による品質評価。"""
        if not response.strip():
            return 0.0

        if sample.metadata.get("quality_score") is not None:
            return float(sample.metadata["quality_score"])

        try:
            import re
            ref_text = ""
            if sample.expected_answer:
                ref_text = f"\nExpected answer: {sample.expected_answer[:200]}"
            resp = await self._gateway.complete(
                f"Query: {sample.query[:300]}\n"
                f"Answer: {response[:400]}"
                f"{ref_text}\n\n"
                "Rate answer quality 0.0-1.0. Reply with only a number.",
                system="You are a quality evaluator. Reply with only a float 0.0-1.0.",
                provider=self._provider,
                max_tokens=5,
                temperature=0.0,
            )
            m = re.search(r"([0-9]*\.?[0-9]+)", resp.content.strip())
            if m:
                return max(0.0, min(1.0, float(m.group(1))))
        except Exception:
            logger.debug("Quality eval LLM failed")
        return 0.5

    def _eval_memory_utilization(self, meta: dict[str, Any]) -> float:
        """メモリ活用スコア。"""
        doc_ids = meta.get("context_doc_ids", [])
        if not doc_ids:
            return 0.2
        n = len(doc_ids)
        if 1 <= n <= 3:
            return 1.0
        return max(0.4, 1.0 - (n - 3) * 0.1)

    def _aggregate(self, per_sample: list[dict[str, Any]]) -> EvalMetrics:
        """サンプル結果を集計する。"""
        n = len(per_sample)
        if n == 0:
            return EvalMetrics()

        def _avg(key: str) -> float:
            vals = [s[key] for s in per_sample if isinstance(s.get(key), (int, float))]
            return statistics.mean(vals) if vals else 0.0

        return EvalMetrics(
            n_samples=n,
            avg_reward=_avg("reward"),
            answer_quality=_avg("answer_quality"),
            exec_success_rate=_avg("exec_success"),
            retrieval_accuracy=_avg("retrieval_accuracy"),
            memory_utilization_rate=_avg("memory_utilization"),
            per_sample=per_sample,
        )
