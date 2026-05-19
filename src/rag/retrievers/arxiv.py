"""src/rag/retrievers/arxiv.py — ArXiv 論文検索レトリーバー

カテゴリフィルタを適用し、プロジェクト関連分野の論文のみを返す。

429 対策:
  バックオフ状態 (minutes_level / days_level) を data/arxiv_backoff.db に永続化する。
  詳細な緩和ロジックは persistent_backoff.py を参照。
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, ClassVar

from src.rag.retriever import BaseRetriever, RawResult

if TYPE_CHECKING:
    from src.rag.retrievers.persistent_backoff import PersistentBackoffStore

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+\{([^}]*)\}")
_LATEX_MATH_RE = re.compile(r"\$\$?[^$]+\$\$?")


def _clean_arxiv_text(text: str) -> str:
    """ArXiv アブストラクトの HTML タグと LaTeX を除去する。"""
    text = _HTML_TAG_RE.sub(" ", text)
    text = _LATEX_CMD_RE.sub(r"\1", text)
    text = _LATEX_MATH_RE.sub("[MATH]", text)
    return re.sub(r"\s+", " ", text).strip()


logger = logging.getLogger(__name__)

# MED_ARXIV_API_URL でスタブサーバーに切り替え可能（テスト用）
_ARXIV_API = os.environ.get("MED_ARXIV_API_URL", "https://export.arxiv.org/api/query")
_NS = {"atom": "http://www.w3.org/2005/Atom"}

_DEFAULT_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.IR", "cs.DB", "stat.ML"]


class ArXivRetriever(BaseRetriever):
    """ArXiv API を使った学術論文検索。

    バックオフ状態を DB に永続化し、429 が続くと段階的にアクセスを制限する。

    Class-level backoff parameters
    --------------------------------
    BACKOFF_BASE_SECS        : Level 0 (正常時) の待機秒数 [arXiv ToS 最低値]
    BACKOFF_MULTIPLIER       : Level N の待機 = MULTIPLIER * 2^N (N>=1)
    BACKOFF_BAN_THRESHOLD_SECS : この秒数を超えたら day ban へ昇格
    BACKOFF_MAX_MINUTES_LEVEL  : minutes_level の上限 (= ban 閾値を超えるレベル)
    BACKOFF_DB_PATH          : バックオフ状態を保存する SQLite DB パス
    """

    # --- バックオフ静的パラメーター (クラスレベルで変更可) ---
    BACKOFF_BASE_SECS: ClassVar[float] = 3.0
    BACKOFF_MULTIPLIER: ClassVar[float] = 10.0
    BACKOFF_BAN_THRESHOLD_SECS: ClassVar[float] = 60.0
    BACKOFF_MAX_MINUTES_LEVEL: ClassVar[int] = 3   # 10*2^3=80s > 60s → ban
    BACKOFF_DB_PATH: ClassVar[str] = "data/arxiv_backoff.db"

    def __init__(self, categories: list[str] | None = None) -> None:
        if categories is None:
            self._categories = list(_DEFAULT_CATEGORIES)
        else:
            self._categories = categories
        self._backoff_store: PersistentBackoffStore | None = None

    # ------------------------------------------------------------------
    # クラスメソッド: 外部から状態参照用
    # ------------------------------------------------------------------

    @classmethod
    async def current_backoff_state(cls) -> "BackoffState":  # type: ignore[name-defined]
        """現在の永続バックオフ状態を返す（デバッグ・GUI 表示用）。"""
        from src.rag.retrievers.persistent_backoff import PersistentBackoffStore, apply_relaxation
        store = PersistentBackoffStore("arxiv", cls.BACKOFF_DB_PATH)
        state = await store.load()
        return apply_relaxation(state)

    # ------------------------------------------------------------------
    # BaseRetriever 必須プロパティ / メソッド
    # ------------------------------------------------------------------

    @property
    def source_name(self) -> str:
        return "arxiv"

    def is_available(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_store(self) -> "PersistentBackoffStore":
        if self._backoff_store is None:
            from src.rag.retrievers.persistent_backoff import PersistentBackoffStore
            self._backoff_store = PersistentBackoffStore("arxiv", self.BACKOFF_DB_PATH)
        return self._backoff_store

    def _wait_secs(self, level: int) -> float:
        from src.rag.retrievers.persistent_backoff import wait_secs
        return wait_secs(level, self.BACKOFF_BASE_SECS, self.BACKOFF_MULTIPLIER)

    # ------------------------------------------------------------------
    # search() オーバーライド — ban チェック + 永続インターバル待機
    # ------------------------------------------------------------------

    async def search(self, query: str, max_results: int = 5) -> list[RawResult]:
        """永続バックオフ状態を適用してから検索する。

        1. DB からバックオフ状態をロードし日次/週次の緩和を適用する。
        2. アクセス禁止中なら即座に [] を返す。
        3. 現在の minutes_level に対応した待機時間だけスリープする (永続インターバル)。
        4. セマフォを取得して _do_search を呼ぶ (BaseRetriever._rate_limit_wait は使用しない)。
        """
        from src.rag.retrievers.persistent_backoff import apply_relaxation, is_banned

        store = self._get_store()
        state = await store.load()
        state = apply_relaxation(state)

        if is_banned(state):
            await store.save(state)
            logger.warning(
                "ArXiv access banned until %s (days_level=%d) — skipping: %r",
                state.ban_until, state.days_level, query[:50],
            )
            return []

        # 永続インターバル: minutes_level に基づく待機
        interval = self._wait_secs(state.minutes_level)
        if interval > 0:
            logger.debug(
                "ArXiv backoff wait %.1fs (minutes_level=%d)",
                interval, state.minutes_level,
            )
            await asyncio.sleep(interval)

        await store.save(state)

        # BaseRetriever._rate_limit_wait を迂回し、セマフォ + _do_search を直接呼ぶ
        async with self._get_sem():
            return await self._do_search(query, max_results)

    # ------------------------------------------------------------------
    # _do_search — HTTP リクエスト + 429 時のバックオフ更新
    # ------------------------------------------------------------------

    async def _do_search(self, query: str, max_results: int = 5) -> list[RawResult]:
        import httpx

        from src.rag.retrievers.persistent_backoff import apply_relaxation, ban_days

        search_query = self._build_query(query)
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        content = ""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(_ARXIV_API, params=params)
                resp.raise_for_status()
                content = resp.text
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 429:
                    raise

                # Retry-After ヘッダーを確認
                retry_after_secs: float | None = None
                ra_header = exc.response.headers.get("Retry-After", "")
                if ra_header:
                    try:
                        retry_after_secs = float(ra_header)
                        logger.warning("ArXiv 429 Retry-After: %.0fs", retry_after_secs)
                    except ValueError:
                        try:
                            ra_dt = parsedate_to_datetime(ra_header)
                            now = datetime.now(timezone.utc)
                            retry_after_secs = max(0.0, (ra_dt - now).total_seconds())
                            logger.warning(
                                "ArXiv 429 Retry-After: %s (→ %.0fs)", ra_header, retry_after_secs
                            )
                        except Exception:
                            logger.warning("ArXiv 429 Retry-After: %r (unparseable)", ra_header)
                else:
                    logger.warning("ArXiv 429: no Retry-After header")

                # 429 受信 → バックオフレベルを上げて保存
                store = self._get_store()
                state = await store.load()
                state = apply_relaxation(state)

                # Retry-After が ban_threshold を超える場合は直接 ban_until に反映
                if retry_after_secs is not None and retry_after_secs > self.BACKOFF_BAN_THRESHOLD_SECS:
                    ban_end = date.today() + timedelta(seconds=retry_after_secs)
                    state.days_level = max(state.days_level + 1, 1)
                    state.days_date = date.today().isoformat()
                    state.ban_until = ban_end.isoformat()
                    await store.save(state)
                    logger.warning(
                        "ArXiv 429 Retry-After=%.0fs → day ban until %s",
                        retry_after_secs, state.ban_until,
                    )
                    return []

                new_level = min(state.minutes_level + 1, self.BACKOFF_MAX_MINUTES_LEVEL)
                state.minutes_level = new_level
                state.minutes_date = date.today().isoformat()

                new_wait = self._wait_secs(new_level)

                if new_wait > self.BACKOFF_BAN_THRESHOLD_SECS:
                    # 1分超 → day ban へ昇格
                    state.days_level += 1
                    state.days_date = date.today().isoformat()
                    ban_end = date.today() + timedelta(days=ban_days(state.days_level))
                    state.ban_until = ban_end.isoformat()
                    await store.save(state)
                    logger.warning(
                        "ArXiv 429 → minutes_level=%d (%.0fs > %.0fs) "
                        "→ day ban level=%d, banned until %s",
                        new_level, new_wait, self.BACKOFF_BAN_THRESHOLD_SECS,
                        state.days_level, state.ban_until,
                    )
                    return []

                await store.save(state)
                logger.warning(
                    "ArXiv 429 → minutes_level=%d (%.0fs), next request will wait",
                    new_level, new_wait,
                )
                return []

        try:
            root = ET.fromstring(content)
        except ET.ParseError:
            logger.warning("Failed to parse ArXiv response")
            return []

        results: list[RawResult] = []
        for entry in root.findall("atom:entry", _NS)[:max_results]:
            title_el = entry.find("atom:title", _NS)
            summary_el = entry.find("atom:summary", _NS)
            id_el = entry.find("atom:id", _NS)

            title = _clean_arxiv_text(title_el.text or "") if title_el is not None else ""
            summary = _clean_arxiv_text(summary_el.text or "") if summary_el is not None else ""
            arxiv_url = id_el.text.strip() if id_el is not None else ""

            categories = [
                c.get("term", "")
                for c in entry.findall("atom:category", _NS)
            ]
            authors = [
                a.findtext("atom:name", namespaces=_NS) or ""
                for a in entry.findall("atom:author", _NS)
            ]

            results.append(RawResult(
                title=title,
                content=summary[:2000],
                url=arxiv_url,
                source=self.source_name,
                score=1.0,
                metadata={
                    "content_type": "paper_abstract",
                    "authors": authors,
                    "published": entry.findtext("atom:published", namespaces=_NS) or "",
                    "categories": categories,
                },
            ))

        if results:
            logger.info(
                "ArXiv search: query=%r categories=%s → %d results",
                query[:50], self._categories or ["all"], len(results),
            )
        return results

    def _build_query(self, query: str) -> str:
        """カテゴリフィルタ付き ArXiv API クエリを構築する。"""
        base = f"all:{query}"
        if not self._categories:
            return base
        cat_filter = " OR ".join(f"cat:{c}" for c in self._categories)
        return f"{base} AND ({cat_filter})"
