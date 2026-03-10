"""src/llm/usage_tracker.py — LLM 使用量トラッカー

API コスト・トークン使用量・レイテンシを記録・集計する。

使い方:
    from src.llm.usage_tracker import UsageTracker

    tracker = UsageTracker()
    tracker.record("anthropic", "claude-sonnet-4-6", input_tokens=100, output_tokens=50, latency_ms=1200)
    report = tracker.report()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# モデル別コスト (USD per 1M tokens, 概算)
_COST_TABLE: dict[str, dict[str, float]] = {
    "claude-opus-4-6":     {"input": 15.0,  "output": 75.0},
    "claude-sonnet-4-6":   {"input": 3.0,   "output": 15.0},
    "claude-haiku-4-5":    {"input": 0.25,  "output": 1.25},
    "gpt-4o":              {"input": 5.0,   "output": 15.0},
    "gpt-4o-mini":         {"input": 0.15,  "output": 0.6},
    "gpt-3.5-turbo":       {"input": 0.5,   "output": 1.5},
}
_DEFAULT_COST = {"input": 1.0, "output": 3.0}  # 不明モデルのデフォルト


@dataclass
class UsageRecord:
    """1 回の LLM 呼び出し記録。"""

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    timestamp: float = field(default_factory=time.time)
    cost_usd: float = 0.0


@dataclass
class UsageSummary:
    """使用量サマリ。"""

    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    by_provider: dict[str, dict] = field(default_factory=dict)
    by_model: dict[str, dict] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"UsageSummary(calls={self.total_calls}, "
            f"input={self.total_input_tokens:,}tok, "
            f"output={self.total_output_tokens:,}tok, "
            f"cost=${self.total_cost_usd:.4f}, "
            f"avg_latency={self.avg_latency_ms:.0f}ms)"
        )


class UsageTracker:
    """LLM 使用量追跡クラス。

    スレッドセーフではない（非同期環境を想定）。

    Args:
        cost_table: モデル別コスト辞書（省略時はデフォルト値）。
    """

    def __init__(
        self,
        cost_table: Optional[dict[str, dict[str, float]]] = None,
    ) -> None:
        self._cost_table = cost_table or dict(_COST_TABLE)
        self._records: list[UsageRecord] = []

    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float = 0.0,
    ) -> UsageRecord:
        """1 回の LLM 呼び出しを記録する。"""
        cost = self._calc_cost(model, input_tokens, output_tokens)
        rec = UsageRecord(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
        )
        self._records.append(rec)
        logger.debug(
            "UsageTracker: %s/%s in=%d out=%d cost=$%.5f",
            provider, model, input_tokens, output_tokens, cost,
        )
        return rec

    def _calc_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """コストを計算する (USD)。"""
        costs = self._cost_table.get(model, _DEFAULT_COST)
        return (
            input_tokens * costs["input"] / 1_000_000
            + output_tokens * costs["output"] / 1_000_000
        )

    def report(self) -> UsageSummary:
        """使用量サマリを返す。"""
        if not self._records:
            return UsageSummary()

        by_provider: dict[str, dict] = {}
        by_model: dict[str, dict] = {}

        for rec in self._records:
            for key, bucket in [(rec.provider, by_provider), (rec.model, by_model)]:
                if key not in bucket:
                    bucket[key] = {
                        "calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                    }
                bucket[key]["calls"] += 1
                bucket[key]["input_tokens"] += rec.input_tokens
                bucket[key]["output_tokens"] += rec.output_tokens
                bucket[key]["cost_usd"] += rec.cost_usd

        latencies = [r.latency_ms for r in self._records if r.latency_ms > 0]
        return UsageSummary(
            total_calls=len(self._records),
            total_input_tokens=sum(r.input_tokens for r in self._records),
            total_output_tokens=sum(r.output_tokens for r in self._records),
            total_cost_usd=sum(r.cost_usd for r in self._records),
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
            by_provider=by_provider,
            by_model=by_model,
        )

    def reset(self) -> None:
        """記録をリセットする。"""
        self._records.clear()

    @property
    def record_count(self) -> int:
        return len(self._records)
