"""src/llm/feedback_analyzer.py — ユーザーフィードバック分析器

ユーザーのフィードバック（自然言語）を解析し、
FAISS メモリの有用性スコア更新や訓練シグナルに変換する。

使い方:
    from src.llm.feedback_analyzer import FeedbackAnalyzer, FeedbackSignal

    analyzer = FeedbackAnalyzer(gateway)
    signal = await analyzer.analyze("This answer was very helpful!")
    # → FeedbackSignal(sentiment="positive", reward=0.9, update_memory=True)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.llm.gateway import LLMGateway

logger = logging.getLogger(__name__)

_ANALYZE_SYSTEM = """\
You are a feedback analyzer. Analyze user feedback and respond with ONLY valid JSON:
{
  "sentiment": "positive|negative|neutral",
  "reward": 0.0-1.0,
  "aspects": ["helpful", "accurate", "complete", "clear"],
  "update_memory": true/false,
  "summary": "brief summary"
}

Reward guidelines:
- positive/enthusiastic: 0.8-1.0
- positive/mild: 0.6-0.8
- neutral: 0.4-0.6
- negative/mild: 0.2-0.4
- negative/strong: 0.0-0.2"""

_ANALYZE_PROMPT = "Analyze this user feedback: {feedback}"

# キーワードベースフォールバック
_POS_PATTERN = re.compile(
    r"\b(great|excellent|perfect|helpful|good|nice|thank|thanks|awesome|correct|"
    r"accurate|useful|well done|clear|brilliant)\b",
    re.IGNORECASE,
)
_NEG_PATTERN = re.compile(
    r"\b(wrong|incorrect|bad|unhelpful|useless|terrible|awful|error|mistake|"
    r"confused|misleading|incomplete|inaccurate)\b",
    re.IGNORECASE,
)


@dataclass
class FeedbackSignal:
    """フィードバックシグナル。"""

    raw_feedback: str
    sentiment: str = "neutral"       # "positive" | "negative" | "neutral"
    reward: float = 0.5              # 0.0〜1.0
    aspects: list[str] = None       # type: ignore[assignment]
    update_memory: bool = True       # メモリ更新すべきか
    summary: str = ""

    def __post_init__(self) -> None:
        if self.aspects is None:
            self.aspects = []

    @property
    def is_positive(self) -> bool:
        return self.sentiment == "positive"

    @property
    def is_negative(self) -> bool:
        return self.sentiment == "negative"

    @property
    def memory_delta(self) -> float:
        """FAISS 有用性スコアへの加算値 (-0.5〜+0.5)。"""
        return self.reward - 0.5


class FeedbackAnalyzer:
    """ユーザーフィードバック分析器。

    Args:
        gateway: LLMGateway インスタンス（省略時はキーワード分析のみ）。
        provider: 優先プロバイダ。
        use_llm: True なら LLM 分析、False なら常にキーワードフォールバック。
    """

    def __init__(
        self,
        gateway: LLMGateway | None = None,
        provider: str | None = None,
        use_llm: bool = True,
    ) -> None:
        self._gateway = gateway
        self._provider = provider
        self._use_llm = use_llm and gateway is not None

    async def analyze(self, feedback: str) -> FeedbackSignal:
        """フィードバックを分析する。"""
        if not feedback.strip():
            return FeedbackSignal(
                raw_feedback=feedback,
                sentiment="neutral",
                reward=0.5,
                summary="empty feedback",
            )

        if self._use_llm:
            try:
                return await self._llm_analyze(feedback)
            except Exception:
                logger.exception("FeedbackAnalyzer LLM failed; using keyword fallback")

        return self._keyword_analyze(feedback)

    async def _llm_analyze(self, feedback: str) -> FeedbackSignal:
        import json
        response = await self._gateway.complete(  # type: ignore[union-attr]
            _ANALYZE_PROMPT.format(feedback=feedback[:500]),
            system=_ANALYZE_SYSTEM,
            provider=self._provider,
            max_tokens=150,
            temperature=0.0,
        )
        content = re.sub(r"```(?:json)?\s*", "", response.content).strip().rstrip("`")
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            return FeedbackSignal(
                raw_feedback=feedback,
                sentiment=data.get("sentiment", "neutral"),
                reward=float(data.get("reward", 0.5)),
                aspects=data.get("aspects", []),
                update_memory=bool(data.get("update_memory", True)),
                summary=str(data.get("summary", "")),
            )
        return self._keyword_analyze(feedback)

    def _keyword_analyze(self, feedback: str) -> FeedbackSignal:
        pos = bool(_POS_PATTERN.search(feedback))
        neg = bool(_NEG_PATTERN.search(feedback))

        if pos and not neg:
            sentiment = "positive"
            reward = 0.8
        elif neg and not pos:
            sentiment = "negative"
            reward = 0.2
        elif pos and neg:
            sentiment = "neutral"
            reward = 0.5
        else:
            sentiment = "neutral"
            reward = 0.5

        return FeedbackSignal(
            raw_feedback=feedback,
            sentiment=sentiment,
            reward=reward,
            aspects=[],
            update_memory=True,
            summary=f"keyword-based: {sentiment}",
        )

    async def analyze_batch(self, feedbacks: list[str]) -> list[FeedbackSignal]:
        """複数フィードバックを並列分析する。"""
        import asyncio
        return list(await asyncio.gather(*[self.analyze(f) for f in feedbacks]))
