"""src/memory/learning/feedback_collector.py — フィードバック収集器

ユーザーフィードバック（クリック・評価・テキスト）を収集・集計し、
LTRRanker と EmbeddingAdapter のオンライン学習に利用する。

フィードバック種別:
- click: 検索結果クリック（暗示的ポジティブ信号）
- thumbs_up / thumbs_down: 明示的フィードバック
- rating: 1〜5 の数値評価
- text: テキストフィードバック（FeedbackAnalyzer で処理）

使い方:
    from src.memory.learning.feedback_collector import FeedbackCollector

    collector = FeedbackCollector()
    collector.record_click("doc_id_123", query="What is FAISS?", rank=0)
    collector.record_explicit("doc_id_123", thumbs_up=True)

    signals = collector.drain()  # → list[FeedbackEvent]
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# クリックの暗示的報酬値
_CLICK_REWARD = 0.7
_THUMBS_UP_REWARD = 1.0
_THUMBS_DOWN_REWARD = 0.0
_RATING_MAX = 5.0


@dataclass
class FeedbackEvent:
    """1件のフィードバックイベント。"""

    doc_id: str
    query: str
    reward: float             # 正規化済み報酬 (0.0〜1.0)
    feedback_type: str        # "click" | "thumbs_up" | "thumbs_down" | "rating" | "text"
    rank: int = 0             # 検索結果でのランク（0-indexed）
    timestamp: float = field(default_factory=time.time)
    raw_value: float | None = None  # 元の評価値（rating=1〜5 等）
    text: str = ""            # テキストフィードバック

    @property
    def is_positive(self) -> bool:
        return self.reward >= 0.6

    @property
    def is_negative(self) -> bool:
        return self.reward <= 0.3


@dataclass
class AggregatedFeedback:
    """ドキュメントごとの集計フィードバック。"""

    doc_id: str
    n_events: int = 0
    avg_reward: float = 0.0
    click_count: int = 0
    thumbs_up: int = 0
    thumbs_down: int = 0
    avg_rank: float = 0.0

    @property
    def click_through_rate(self) -> float:
        """クリック率（全イベント中のクリック比率）。"""
        if self.n_events == 0:
            return 0.0
        return self.click_count / self.n_events

    @property
    def net_positive(self) -> int:
        return self.thumbs_up - self.thumbs_down


class FeedbackCollector:
    """フィードバックイベント収集クラス。

    Args:
        max_buffer: バッファ上限（超えたら古いイベントを捨てる）。
        gateway: LLMGateway（テキストフィードバック解析用、省略可）。
        use_text_analyzer: True の場合 FeedbackAnalyzer でテキスト解析。
    """

    def __init__(
        self,
        max_buffer: int = 10_000,
        gateway: object | None = None,
        use_text_analyzer: bool = False,
    ) -> None:
        self._max_buffer = max_buffer
        self._events: list[FeedbackEvent] = []
        self._lock = asyncio.Lock()

        self._analyzer: object | None = None
        if gateway is not None and use_text_analyzer:
            try:
                from src.llm.feedback_analyzer import FeedbackAnalyzer
                self._analyzer = FeedbackAnalyzer(gateway)
            except Exception:
                logger.warning("FeedbackAnalyzer init failed")

    # ------------------------------------------------------------------
    # Record methods
    # ------------------------------------------------------------------

    def record_click(self, doc_id: str, query: str = "", rank: int = 0) -> None:
        """クリックイベントを記録する。"""
        reward = self._rank_adjusted_reward(_CLICK_REWARD, rank)
        self._append(FeedbackEvent(
            doc_id=doc_id, query=query, reward=reward,
            feedback_type="click", rank=rank,
        ))

    def record_explicit(
        self,
        doc_id: str,
        thumbs_up: bool | None = None,
        rating: float | None = None,
        query: str = "",
    ) -> None:
        """明示的フィードバックを記録する。

        Args:
            doc_id: ドキュメント ID。
            thumbs_up: True=UP / False=DOWN / None=使わない。
            rating: 1〜5 の数値評価（省略可）。
            query: 元のクエリ（任意）。
        """
        if thumbs_up is not None:
            reward = _THUMBS_UP_REWARD if thumbs_up else _THUMBS_DOWN_REWARD
            fb_type = "thumbs_up" if thumbs_up else "thumbs_down"
            self._append(FeedbackEvent(
                doc_id=doc_id, query=query, reward=reward,
                feedback_type=fb_type, raw_value=float(thumbs_up),
            ))
        if rating is not None:
            reward = max(0.0, min(1.0, (rating - 1) / (_RATING_MAX - 1)))
            self._append(FeedbackEvent(
                doc_id=doc_id, query=query, reward=reward,
                feedback_type="rating", raw_value=rating,
            ))

    async def record_text(self, doc_id: str, text: str, query: str = "") -> None:
        """テキストフィードバックを記録する（LLM 解析つき）。"""
        reward = 0.5  # デフォルト中立
        if self._analyzer is not None and text.strip():
            try:
                signal = await self._analyzer.analyze(text)
                reward = signal.reward
            except Exception:
                logger.exception("FeedbackAnalyzer error for doc=%s", doc_id)
        self._append(FeedbackEvent(
            doc_id=doc_id, query=query, reward=reward,
            feedback_type="text", text=text,
        ))

    # ------------------------------------------------------------------
    # Query / aggregate
    # ------------------------------------------------------------------

    def drain(self) -> list[FeedbackEvent]:
        """バッファのイベントを全て取り出してクリアする。"""
        events = self._events[:]
        self._events.clear()
        return events

    def peek(self) -> list[FeedbackEvent]:
        """バッファをクリアせずに全イベントを返す。"""
        return self._events[:]

    def aggregate(self) -> dict[str, AggregatedFeedback]:
        """バッファ内イベントをドキュメント ID ごとに集計する。"""
        agg: dict[str, AggregatedFeedback] = {}
        for ev in self._events:
            if ev.doc_id not in agg:
                agg[ev.doc_id] = AggregatedFeedback(doc_id=ev.doc_id)
            a = agg[ev.doc_id]
            a.n_events += 1
            a.avg_reward = (a.avg_reward * (a.n_events - 1) + ev.reward) / a.n_events
            a.avg_rank = (a.avg_rank * (a.n_events - 1) + ev.rank) / a.n_events
            if ev.feedback_type == "click":
                a.click_count += 1
            elif ev.feedback_type == "thumbs_up":
                a.thumbs_up += 1
            elif ev.feedback_type == "thumbs_down":
                a.thumbs_down += 1
        return agg

    def clear(self) -> None:
        """バッファを空にする。"""
        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _append(self, event: FeedbackEvent) -> None:
        if len(self._events) >= self._max_buffer:
            self._events.pop(0)  # 古いイベントを削除
        self._events.append(event)
        logger.debug(
            "FeedbackCollector: %s doc=%s reward=%.2f",
            event.feedback_type, event.doc_id, event.reward,
        )

    @staticmethod
    def _rank_adjusted_reward(base: float, rank: int) -> float:
        """ランクが高い（0-indexed で小さい）ほど報酬を増加させる。"""
        # rank 0 → base, rank 4 → base * 0.6
        decay = max(0.6, 1.0 - rank * 0.1)
        return base * decay
