"""scripts/fetchers/note.py — note.com 記事フェッチャー

API エンドポイント:
  一覧: GET https://note.com/api/v2/creators/{account}/contents?kind=note&page={n}
  本文: GET https://note.com/api/v3/notes/{key}   （v1=405, v2=404, v3=200）

有料記事判定:
  price > 0  または  canRead == False
"""

from __future__ import annotations

import logging
import random
import time

import httpx

from scripts.fetchers.base import BaseFetcher, FetchedArticle, strip_html

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "ja,en;q=0.9",
}


class NoteFetcher(BaseFetcher):
    site_name = "note"

    def __init__(self) -> None:
        self._client = httpx.Client()

    def close(self) -> None:
        self._client.close()

    # ── 内部ヘルパー ──────────────────────────────────────────────────────────

    def _get(self, url: str) -> dict | None:
        try:
            resp = self._client.get(url, headers=_HEADERS, timeout=30)
            if resp.status_code == 404:
                logger.warning("404: %s", url)
                return None
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.warning("HTTP %s: %s", e.response.status_code, url)
            return None
        except Exception as e:
            logger.warning("Request failed (%s): %s", type(e).__name__, url)
            return None

    def _sleep(self, delay_range: tuple[float, float]) -> None:
        sec = random.uniform(*delay_range)
        logger.info("  Waiting %.1fs ...", sec)
        time.sleep(sec)

    # ── BaseFetcher 実装 ───────────────────────────────────────────────────────

    def is_paid(self, article_meta: dict) -> bool:
        price = article_meta.get("price", 0) or 0
        can_read = article_meta.get("canRead", True)
        return price > 0 or not can_read

    def get_key(self, article_meta: dict) -> str:
        return article_meta.get("key", "")

    def fetch_article_list(
        self,
        account: str,
        delay_range: tuple[float, float],
        limit: int | None = None,
    ) -> list[dict]:
        """全記事メタデータ一覧を取得（ページネーション対応）。"""
        results: list[dict] = []
        page = 1

        while True:
            url = (
                f"https://note.com/api/v2/creators/{account}/contents"
                f"?kind=note&page={page}"
            )
            logger.info("Fetching list page %d ...", page)
            data = self._get(url)
            if not data:
                logger.error("List fetch failed at page %d", page)
                break

            contents = data.get("data", {}).get("contents", [])
            if not contents:
                break

            results.extend(contents)
            total = data.get("data", {}).get("totalCount", "?")
            logger.info(
                "  page %d: +%d articles  (total so far: %d / %s)",
                page, len(contents), len(results), total,
            )

            if limit and len(results) >= limit:
                results = results[:limit]
                break

            if data.get("data", {}).get("isLastPage", True):
                break

            page += 1
            self._sleep(delay_range)

        return results

    def fetch_article(
        self,
        key: str,
        delay_range: tuple[float, float],
        account: str,
    ) -> FetchedArticle | None:
        """記事詳細を取得。有料記事は is_paid=True で本文空として返す。"""
        self._sleep(delay_range)
        data = self._get(f"https://note.com/api/v3/notes/{key}")
        if not data:
            return None

        d = data.get("data", {})
        price = d.get("price", 0) or 0
        can_read = d.get("canRead", True)
        is_paid = price > 0 or not can_read

        if is_paid:
            logger.info("  [paid/restricted] %s  price=%d  canRead=%s", key, price, can_read)
            return FetchedArticle(
                key=key,
                title=d.get("name", ""),
                url=f"https://note.com/{account}/n/{key}",
                body_html="",
                body_text="",
                published_at=d.get("publishAt", ""),
                is_paid=True,
                site="note",
                account=account,
                metadata={"price": price},
            )

        body_html = d.get("body", "") or ""
        body_text = strip_html(body_html)

        return FetchedArticle(
            key=key,
            title=d.get("name", ""),
            url=f"https://note.com/{account}/n/{key}",
            body_html=body_html,
            body_text=body_text,
            published_at=d.get("publishAt", ""),
            is_paid=False,
            site="note",
            account=account,
            metadata={
                "like_count": d.get("likeCount", 0),
                "price": 0,
                "hashtags": [
                    h.get("hashtag", {}).get("name", "")
                    for h in d.get("hashtags", [])
                    if h.get("hashtag", {}).get("name")
                ],
            },
        )
