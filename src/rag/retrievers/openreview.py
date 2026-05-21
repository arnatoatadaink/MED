"""src/rag/retrievers/openreview.py — OpenReview 論文検索レトリーバー

ICLR・NeurIPS などの accepted 論文を OpenReview API v2 で検索する。

API: https://api2.openreview.net/notes
レート制限: 60 req/min (x-ratelimit-limit ヘッダー)

429 対策:
  バックオフ状態 (minutes_level / days_level) を data/openreview_backoff.db に永続化する。
  詳細な緩和ロジックは persistent_backoff.py を参照。
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import date, timedelta
from typing import TYPE_CHECKING, ClassVar

from src.rag.retriever import BaseRetriever, RawResult

if TYPE_CHECKING:
    from src.rag.retrievers.persistent_backoff import PersistentBackoffStore

logger = logging.getLogger(__name__)

# MED_OPENREVIEW_API_URL でスタブサーバーに切り替え可能（テスト用）
_OPENREVIEW_API = os.environ.get("MED_OPENREVIEW_API_URL", "https://api2.openreview.net/notes")

# (conference_prefix, year) → invitation = "{prefix}/{year}/Conference/-/Submission"
_DEFAULT_VENUES: list[tuple[str, int]] = [
    ("ICLR.cc", 2025),
    ("NeurIPS.cc", 2024),
    ("ICLR.cc", 2024),
    ("NeurIPS.cc", 2023),
]

# Accepted 論文の判定: venue / venueid フィールドのキーワード
_ACCEPT_KEYWORDS = frozenset({"poster", "oral", "spotlight", "accept", "notable", "highlight"})
_REJECT_KEYWORDS = frozenset({"reject", "withdraw"})


def _is_accepted(note: dict) -> bool:
    """OpenReview note が accepted 論文かを判定する。"""
    content = note.get("content", {})
    venue = content.get("venue", {}).get("value", "").lower()
    venueid = content.get("venueid", {}).get("value", "").lower()
    if any(k in venueid for k in _REJECT_KEYWORDS):
        return False
    if any(k in venue for k in _ACCEPT_KEYWORDS):
        return True
    if "accept" in venueid:
        return True
    return False


def _score(title: str, abstract: str, query_terms: list[str]) -> float:
    """クエリ語のタイトル/アブストへの出現割合を返す (0.0–1.0)。"""
    if not query_terms:
        return 0.0
    text = (title + " " + abstract).lower()
    hits = sum(1 for t in query_terms if t in text)
    return hits / len(query_terms)


class OpenReviewRetriever(BaseRetriever):
    """OpenReview API v2 を使った学術論文検索。

    accepted 論文のみを対象とし、クエリ語によるスコアリングで関連度順に返す。

    Class-level backoff parameters
    --------------------------------
    BACKOFF_MULTIPLIER       : Level N の待機 = MULTIPLIER * 2^N (全 level 共通)
    BACKOFF_BAN_THRESHOLD_SECS : この秒数を超えたら day ban へ昇格
    BACKOFF_MAX_MINUTES_LEVEL  : minutes_level の上限
    BACKOFF_DB_PATH          : バックオフ状態を保存する SQLite DB パス
    FETCH_LIMIT_PER_VENUE    : 各 venue から取得する最大 note 数
    """

    BACKOFF_BASE_SECS: ClassVar[float] = 1.0
    BACKOFF_MULTIPLIER: ClassVar[float] = 5.0
    BACKOFF_BAN_THRESHOLD_SECS: ClassVar[float] = 120.0
    BACKOFF_MAX_MINUTES_LEVEL: ClassVar[int] = 5   # 5*2^5=160s > 120s → ban
    BACKOFF_DB_PATH: ClassVar[str] = "data/openreview_backoff.db"
    FETCH_LIMIT_PER_VENUE: ClassVar[int] = 50

    def __init__(self, venues: list[tuple[str, int]] | None = None) -> None:
        self._venues = venues if venues is not None else list(_DEFAULT_VENUES)
        self._backoff_store: PersistentBackoffStore | None = None

    # ------------------------------------------------------------------
    # クラスメソッド: 外部から状態参照用
    # ------------------------------------------------------------------

    @classmethod
    async def current_backoff_state(cls) -> "BackoffState":  # type: ignore[name-defined]
        """現在の永続バックオフ状態を返す（デバッグ・GUI 表示用）。"""
        from src.rag.retrievers.persistent_backoff import PersistentBackoffStore, apply_relaxation
        store = PersistentBackoffStore("openreview", cls.BACKOFF_DB_PATH)
        state = await store.load()
        return apply_relaxation(state)

    # ------------------------------------------------------------------
    # BaseRetriever 必須プロパティ / メソッド
    # ------------------------------------------------------------------

    @property
    def source_name(self) -> str:
        return "openreview"

    def is_available(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_store(self) -> "PersistentBackoffStore":
        if self._backoff_store is None:
            from src.rag.retrievers.persistent_backoff import PersistentBackoffStore
            self._backoff_store = PersistentBackoffStore("openreview", self.BACKOFF_DB_PATH)
        return self._backoff_store

    def _wait_secs(self, level: int) -> float:
        from src.rag.retrievers.persistent_backoff import wait_secs
        return wait_secs(level, self.BACKOFF_MULTIPLIER)

    # ------------------------------------------------------------------
    # search() オーバーライド — BaseRetriever._rate_limit_wait を迂回
    # ------------------------------------------------------------------

    async def search(self, query: str, max_results: int = 5) -> list[RawResult]:
        """永続バックオフ状態を適用してから検索する。

        1. DB からバックオフ状態をロードし日次/週次の緩和を適用する。
        2. アクセス禁止中なら即座に [] を返す。
        3. 現在の minutes_level に対応した待機時間だけスリープする。
        4. セマフォを取得して _do_search を呼ぶ。
        """
        from src.rag.retrievers.persistent_backoff import apply_relaxation, is_banned

        store = self._get_store()
        state = await store.load()
        state = apply_relaxation(state)

        if is_banned(state):
            await store.save(state)
            logger.warning(
                "OpenReview access banned until %s (days_level=%d) — skipping: %r",
                state.ban_until, state.days_level, query[:50],
            )
            return []

        interval = self._wait_secs(state.minutes_level)
        if interval > 0:
            logger.debug(
                "OpenReview backoff wait %.1fs (minutes_level=%d)",
                interval, state.minutes_level,
            )
            await asyncio.sleep(interval)

        await store.save(state)

        async with self._get_sem():
            return await self._do_search(query, max_results)

    # ------------------------------------------------------------------
    # _do_search — 各 venue を検索してクライアント側スコアリング
    # ------------------------------------------------------------------

    async def _do_search(self, query: str, max_results: int = 5) -> list[RawResult]:
        raw_terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2]
        query_terms = raw_terms if raw_terms else [query.lower().strip()]

        candidates: list[RawResult] = []
        for i, (conference, year) in enumerate(self._venues):
            if i > 0:
                await asyncio.sleep(self.BACKOFF_BASE_SECS)
            notes = await self._fetch_venue(conference, year)
            for note in notes:
                result = self._note_to_result(note, query_terms)
                if result is not None:
                    candidates.append(result)

        candidates.sort(key=lambda r: r.score, reverse=True)
        seen_urls: set[str] = set()
        results: list[RawResult] = []
        for r in candidates:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                results.append(r)
            if len(results) >= max_results:
                break

        if results:
            logger.info(
                "OpenReview search: query=%r → %d results",
                query[:50], len(results),
            )
        return results

    async def _fetch_venue(self, conference: str, year: int) -> list[dict]:
        """1 venue の accepted 論文 note 一覧を取得する。"""
        import httpx

        from src.rag.retrievers.persistent_backoff import apply_relaxation, ban_days

        invitation = f"{conference}/{year}/Conference/-/Submission"
        params = {
            "invitation": invitation,
            "limit": self.FETCH_LIMIT_PER_VENUE,
            "offset": 0,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(_OPENREVIEW_API, params=params)
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 429:
                    logger.warning(
                        "OpenReview HTTP %d for %s/%d",
                        exc.response.status_code, conference, year,
                    )
                    return []

                # 429 → バックオフレベルを上げて保存
                store = self._get_store()
                state = await store.load()
                state = apply_relaxation(state)

                new_level = min(state.minutes_level + 1, self.BACKOFF_MAX_MINUTES_LEVEL)
                state.minutes_level = new_level
                state.minutes_date = date.today().isoformat()
                new_wait = self._wait_secs(new_level)

                if new_wait > self.BACKOFF_BAN_THRESHOLD_SECS:
                    state.days_level += 1
                    state.days_date = date.today().isoformat()
                    ban_end = date.today() + timedelta(days=ban_days(state.days_level))
                    state.ban_until = ban_end.isoformat()
                    await store.save(state)
                    logger.warning(
                        "OpenReview 429 → minutes_level=%d (%.0fs > %.0fs) "
                        "→ day ban level=%d, banned until %s",
                        new_level, new_wait, self.BACKOFF_BAN_THRESHOLD_SECS,
                        state.days_level, state.ban_until,
                    )
                    return []

                await store.save(state)
                logger.warning(
                    "OpenReview 429 → minutes_level=%d (%.0fs), next request will wait",
                    new_level, new_wait,
                )
                return []
            except httpx.RequestError as exc:
                logger.warning("OpenReview request error for %s/%d: %s", conference, year, exc)
                return []

        data = resp.json()
        notes: list[dict] = data.get("notes", [])
        return [n for n in notes if _is_accepted(n)]

    def _note_to_result(self, note: dict, query_terms: list[str]) -> RawResult | None:
        """OpenReview note を RawResult に変換する。スコアが 0 なら None を返す。"""
        content = note.get("content", {})
        title = content.get("title", {}).get("value", "").strip()
        abstract = content.get("abstract", {}).get("value", "").strip()
        keywords: list[str] = content.get("keywords", {}).get("value", [])
        authors: list[str] = content.get("authors", {}).get("value", [])
        venue = content.get("venue", {}).get("value", "")
        note_id = note.get("id", "")

        if not title or not abstract or not note_id:
            return None

        score = _score(title, abstract, query_terms)
        if score == 0.0:
            return None

        url = f"https://openreview.net/forum?id={note_id}"
        return RawResult(
            title=title,
            content=abstract[:2000],
            url=url,
            source=self.source_name,
            score=score,
            metadata={
                "content_type": "paper_abstract",
                "authors": authors,
                "keywords": keywords,
                "venue": venue,
                "forum_id": note_id,
            },
        )
