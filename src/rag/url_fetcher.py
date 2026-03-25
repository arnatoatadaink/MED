"""src/rag/url_fetcher.py — URL 直接取得

クエリ内で検出された URL からコンテンツを直接取得する。
CRAG ルールベース処理の一部として使用される。

対応 URL:
- arxiv.org: ArXiv API 経由で論文アブストラクトを取得
- 一般 Web: httpx で HTML を取得し、テキストを抽出

使い方:
    from src.rag.url_fetcher import URLFetcher

    fetcher = URLFetcher()
    results = await fetcher.fetch_urls(extracted_urls)
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET

from src.rag.query_expander import ExtractedURL
from src.rag.retriever import RawResult

logger = logging.getLogger(__name__)

_ARXIV_API = "https://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom"}

# HTML タグ除去用
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


class URLFetcher:
    """URL からコンテンツを直接取得する。

    Args:
        timeout: HTTP リクエストのタイムアウト秒数。
        max_content_length: 取得するテキストの最大文字数。
    """

    def __init__(
        self,
        timeout: float = 30.0,
        max_content_length: int = 5000,
    ) -> None:
        self._timeout = timeout
        self._max_content = max_content_length

    async def fetch_urls(self, urls: list[ExtractedURL]) -> list[RawResult]:
        """複数の URL からコンテンツを取得する。

        Args:
            urls: ExtractedURL のリスト。

        Returns:
            取得成功した RawResult のリスト。
        """
        import asyncio

        tasks = [self._fetch_one(u) for u in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        fetched: list[RawResult] = []
        for url_info, result in zip(urls, results):
            if isinstance(result, Exception):
                logger.warning("URL fetch failed for %s: %s", url_info.url, result)
            elif result is not None:
                fetched.append(result)

        logger.info("URL fetcher: %d/%d URLs fetched successfully", len(fetched), len(urls))
        return fetched

    async def _fetch_one(self, url_info: ExtractedURL) -> RawResult | None:
        """単一 URL からコンテンツを取得する。"""
        if url_info.url_type == "arxiv":
            return await self._fetch_arxiv(url_info)
        else:
            return await self._fetch_web(url_info)

    async def _fetch_arxiv(self, url_info: ExtractedURL) -> RawResult | None:
        """ArXiv API 経由で論文情報を取得する。"""
        import httpx

        arxiv_id = url_info.arxiv_id
        if not arxiv_id:
            return None

        params = {
            "id_list": arxiv_id,
            "max_results": 1,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(_ARXIV_API, params=params)
                resp.raise_for_status()
        except Exception:
            logger.warning("ArXiv API request failed for %s", arxiv_id)
            return None

        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError:
            logger.warning("Failed to parse ArXiv response for %s", arxiv_id)
            return None

        entry = root.find("atom:entry", _NS)
        if entry is None:
            return None

        title_el = entry.find("atom:title", _NS)
        summary_el = entry.find("atom:summary", _NS)

        title = title_el.text.strip() if title_el is not None and title_el.text else ""
        summary = summary_el.text.strip() if summary_el is not None and summary_el.text else ""

        if not summary:
            return None

        authors = [
            a.findtext("atom:name", namespaces=_NS) or ""
            for a in entry.findall("atom:author", _NS)
        ]

        logger.info("ArXiv fetched: [%s] %s", arxiv_id, title[:80])

        return RawResult(
            title=title,
            content=summary[:self._max_content],
            url=url_info.url,
            source="arxiv",
            score=1.0,
            metadata={
                "arxiv_id": arxiv_id,
                "authors": authors,
                "published": entry.findtext("atom:published", namespaces=_NS) or "",
                "direct_fetch": True,
            },
        )

    async def _fetch_web(self, url_info: ExtractedURL) -> RawResult | None:
        """一般 Web ページからテキストを取得する。"""
        import httpx

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
                headers={"User-Agent": "MED-RAG/1.0 (Research)"},
            ) as client:
                resp = await client.get(url_info.url)
                resp.raise_for_status()
        except Exception:
            logger.warning("Web fetch failed for %s", url_info.url)
            return None

        content_type = resp.headers.get("content-type", "")

        # PDF やバイナリはスキップ
        if "text/html" not in content_type and "text/plain" not in content_type:
            logger.info("Skipping non-text content: %s (%s)", url_info.url, content_type)
            return None

        text = self._extract_text(resp.text)
        if len(text.strip()) < 50:
            logger.info("Skipping near-empty page: %s", url_info.url)
            return None

        # タイトル抽出
        title = self._extract_title(resp.text) or url_info.url

        logger.info("Web fetched: %s (%d chars)", url_info.url[:80], len(text))

        return RawResult(
            title=title,
            content=text[:self._max_content],
            url=url_info.url,
            source="web",
            score=0.8,
            metadata={
                "direct_fetch": True,
                "url_type": url_info.url_type,
            },
        )

    @staticmethod
    def _extract_text(html: str) -> str:
        """HTML からテキストを抽出する（簡易版）。"""
        # script, style タグを除去
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # HTML タグ除去
        text = _HTML_TAG_RE.sub(" ", text)
        # HTML エンティティ
        text = text.replace("&nbsp;", " ").replace("&amp;", "&")
        text = text.replace("&lt;", "<").replace("&gt;", ">")
        # 連続空白を正規化
        text = _WHITESPACE_RE.sub(" ", text).strip()
        return text

    @staticmethod
    def _extract_title(html: str) -> str:
        """HTML から <title> タグの内容を抽出する。"""
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.DOTALL | re.IGNORECASE)
        if m:
            return _HTML_TAG_RE.sub("", m.group(1)).strip()[:200]
        return ""
