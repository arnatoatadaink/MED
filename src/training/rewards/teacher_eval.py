"""src/training/rewards/teacher_eval.py — Teacher 評価報酬

Teacher LLM が Student の回答を直接評価する報酬関数。
CompositeReward の correctness コンポーネントの独立版。

使い方:
    from src.training.rewards.teacher_eval import TeacherEvalReward
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.training.base import RewardFunction
from src.training.registry import TrainingRegistry

logger = logging.getLogger(__name__)

_EVAL_SYSTEM = """\
You are a strict teacher evaluating a student's answer.
Rate the answer quality from 0.0 to 1.0 considering:
- Correctness (most important)
- Completeness
- Clarity

Respond with ONLY a float between 0.0 and 1.0."""

_EVAL_PROMPT = "Question: {prompt}\n\nStudent's Answer: {response}\n\nScore (0.0-1.0):"


@TrainingRegistry.reward("teacher_eval")
class TeacherEvalReward(RewardFunction):
    """Teacher LLM による直接評価報酬。

    Args:
        gateway: LLMGateway インスタンス。
        provider: 優先プロバイダ。
        fallback_score: LLM 失敗時のスコア。
    """

    def __init__(
        self,
        gateway: Any = None,
        provider: str | None = None,
        fallback_score: float = 0.5,
    ) -> None:
        self._gateway = gateway
        self._provider = provider
        self._fallback = fallback_score

    @property
    def name(self) -> str:
        return "teacher_eval"

    async def compute(
        self,
        prompt: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        if self._gateway is None:
            return self._fallback

        try:
            from src.llm.gateway import LLMGateway
            assert isinstance(self._gateway, LLMGateway)
            resp = await self._gateway.complete(
                _EVAL_PROMPT.format(
                    prompt=prompt[:400],
                    response=response[:600],
                ),
                system=_EVAL_SYSTEM,
                provider=self._provider,
                max_tokens=5,
                temperature=0.0,
            )
            m = re.search(r"([0-9]*\.?[0-9]+)", resp.content.strip())
            if m:
                return max(0.0, min(1.0, float(m.group(1))))
        except Exception:
            logger.exception("TeacherEvalReward LLM call failed")
        return self._fallback
