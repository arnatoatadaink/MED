"""src/memory/scoring/freshness.py — ドメイン別指数減衰フレッシュネススコア

ドキュメントの経過時間に応じてフレッシュネススコアを計算する。
ドメインごとに半減期（half_life_days）を設定し、指数減衰を適用する。

使い方:
    from src.memory.scoring.freshness import FreshnessScorer

    scorer = FreshnessScorer()
    score = scorer.score("code", retrieved_at=datetime(2025, 1, 1))  # 0.0〜1.0
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from src.common.config import get_settings

# ドメイン別デフォルト半減期（日数）
_DEFAULT_HALF_LIFE: dict[str, float] = {
    "code": 180.0,      # コードは半年で半減（バージョン変化が速い）
    "academic": 730.0,  # 学術論文は2年で半減
    "general": 365.0,   # 一般情報は1年で半減
}

_FALLBACK_HALF_LIFE = 365.0  # 未定義ドメインのデフォルト


class FreshnessScorer:
    """ドメイン別指数減衰でフレッシュネススコアを計算する。

    スコアは 0.0〜1.0 の範囲。retrieved_at が新しいほど高スコア。

    Args:
        half_life_days: ドメインごとの半減期（日数）マッピング。
            省略時はデフォルト値を使用。
    """

    def __init__(
        self,
        half_life_days: Optional[dict[str, float]] = None,
    ) -> None:
        self._half_life = half_life_days or _DEFAULT_HALF_LIFE.copy()

    def score(
        self,
        domain: str,
        retrieved_at: Optional[datetime] = None,
        now: Optional[datetime] = None,
    ) -> float:
        """フレッシュネススコアを計算する。

        Args:
            domain: ドキュメントのドメイン。
            retrieved_at: ドキュメント取得日時。None の場合は 0.5 を返す。
            now: 現在日時（テスト用）。省略時は UTC 現在時刻。

        Returns:
            0.0〜1.0 のフレッシュネススコア。
        """
        if retrieved_at is None:
            return 0.5  # 不明な場合は中間値

        if now is None:
            now = datetime.now(timezone.utc)

        # タイムゾーン揃え
        if retrieved_at.tzinfo is None:
            retrieved_at = retrieved_at.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        elapsed_days = max(0.0, (now - retrieved_at).total_seconds() / 86400.0)
        half_life = self._half_life.get(domain, _FALLBACK_HALF_LIFE)
        score = math.exp(-math.log(2) * elapsed_days / half_life)
        return float(min(1.0, max(0.0, score)))

    def score_batch(
        self,
        domain: str,
        retrieved_ats: list[Optional[datetime]],
        now: Optional[datetime] = None,
    ) -> list[float]:
        """複数ドキュメントのフレッシュネススコアを一括計算する。"""
        return [self.score(domain, rt, now=now) for rt in retrieved_ats]

    def set_half_life(self, domain: str, days: float) -> None:
        """特定ドメインの半減期を設定する。"""
        if days <= 0:
            raise ValueError(f"half_life_days must be positive, got {days}")
        self._half_life[domain] = days
