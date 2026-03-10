"""src/rag/retrievers/tavily.py — Tavily Web Search レトリーバー"""

from __future__ import annotations

import logging
import os

from src.rag.retriever import BaseRetriever, RawResult

logger = logging.getLogger(__name__)


class TavilyRetriever(BaseRetriever):
    """Tavily AI Search API を使ったウェブ検索。TAVILY_API_KEY が必要。"""

    @property
    def source_name(self) -> str:
        return "tavily"

    def __init__(self) -> None:
        self._api_key = os.environ.get("TAVILY_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def search(self, query: str, max_results: int = 5) -> list[RawResult]:
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx not installed. Run: pip install httpx")

        payload = {
            "api_key": self._api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_raw_content": False,
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
            results.append(RawResult(
                title=item.get("title", ""),
                content=item.get("content", ""),
                url=item.get("url", ""),
                source=self.source_name,
                score=float(item.get("score", 0.0)),
                metadata={
                    "published_date": item.get("published_date", ""),
                },
            ))

        return results
