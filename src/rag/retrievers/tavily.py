"""src/rag/retrievers/tavily.py — Tavily Web Search レトリーバー"""

from __future__ import annotations

import logging
import os
import re

from src.rag.retriever import BaseRetriever, RawResult

_HTML_TAG_RE = re.compile(r"<[^>]+>")
# Markdown 画像: ![alt](url)
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# Markdown リンク: [text](url) → text を残す
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
# Markdown 見出し記号（行頭・行中）
_MD_HEADING_RE = re.compile(r"#{1,6}\s+")
# 連続する記号行（ナビゲーション残骸）
_NAV_LINE_RE = re.compile(r"^[\[\]|#*>\-]+\s*$", re.MULTILINE)


def _clean_web_text(text: str) -> str:
    """Web 記事コンテンツを正規化する。

    HTML タグ、Markdown 画像/リンク記法、ナビゲーション残骸を除去する。
    """
    # HTML タグ除去
    text = _HTML_TAG_RE.sub(" ", text)
    # Markdown 画像除去
    text = _MD_IMAGE_RE.sub(" ", text)
    # Markdown リンク → リンクテキストのみ残す
    text = _MD_LINK_RE.sub(r"\1", text)
    # Markdown 見出し記号除去
    text = _MD_HEADING_RE.sub("", text)
    # 記号のみの行（ナビゲーション残骸）除去
    text = _NAV_LINE_RE.sub("", text)
    # 空白正規化
    return re.sub(r"\s+", " ", text).strip()

logger = logging.getLogger(__name__)


class TavilyRetriever(BaseRetriever):
    """Tavily AI Search API を使ったウェブ検索。TAVILY_API_KEY が必要。"""

    @property
    def source_name(self) -> str:
        return "tavily"

    def __init__(self, include_raw_content: bool = True) -> None:
        self._api_key = os.environ.get("TAVILY_API_KEY", "")
        # True にすると全文コンテンツを取得（断片化防止）
        self._include_raw_content = include_raw_content

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def _do_search(self, query: str, max_results: int = 5) -> list[RawResult]:
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx not installed. Run: pip install httpx")

        payload = {
            "api_key": self._api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "advanced",
            "include_raw_content": self._include_raw_content,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        results: list[RawResult] = []
        for item in data.get("results", [])[:max_results]:
            url = item.get("url", "")
            # raw_content があれば優先使用（全文）、なければ snippet にフォールバック
            raw = item.get("raw_content") or ""
            snippet = item.get("content", "")
            content = _clean_web_text(raw) if raw else _clean_web_text(snippet)
            results.append(RawResult(
                title=_clean_web_text(item.get("title", "")),
                content=content,
                url=url,
                source=self.source_name,
                score=float(item.get("score", 0.0)),
                metadata={
                    "published_date": item.get("published_date", ""),
                    "content_type": "web_article",
                    "domain": _extract_domain(url),
                    "has_raw_content": bool(raw),
                },
            ))

        return results


def _extract_domain(url: str) -> str:
    """URL からドメインを抽出する。"""
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc
    except Exception:
        return ""
