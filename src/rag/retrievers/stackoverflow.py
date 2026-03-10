"""src/rag/retrievers/stackoverflow.py — Stack Overflow Search レトリーバー"""

from __future__ import annotations

import logging

from src.rag.retriever import BaseRetriever, RawResult

logger = logging.getLogger(__name__)

_API_BASE = "https://api.stackexchange.com/2.3"


class StackOverflowRetriever(BaseRetriever):
    """Stack Exchange API を使った Stack Overflow 検索。API キー不要。"""

    @property
    def source_name(self) -> str:
        return "stackoverflow"

    def is_available(self) -> bool:
        return True  # 公開 API のため常に利用可能

    async def search(self, query: str, max_results: int = 5) -> list[RawResult]:
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx not installed. Run: pip install httpx")

        params = {
            "q": query,
            "site": "stackoverflow",
            "pagesize": max_results,
            "order": "desc",
            "sort": "relevance",
            "filter": "withbody",  # 回答本文も取得
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{_API_BASE}/search/advanced", params=params)
            resp.raise_for_status()
            data = resp.json()

        results: list[RawResult] = []
        for item in data.get("items", [])[:max_results]:
            body = item.get("body", "")
            # HTML タグを除去（簡易）
            import re
            body_text = re.sub(r"<[^>]+>", " ", body).strip()

            results.append(RawResult(
                title=item.get("title", ""),
                content=body_text[:2000],
                url=item.get("link", ""),
                source=self.source_name,
                score=float(item.get("score", 0)),
                metadata={
                    "answer_count": item.get("answer_count", 0),
                    "is_answered": item.get("is_answered", False),
                    "view_count": item.get("view_count", 0),
                    "tags": item.get("tags", []),
                },
            ))

        return results
