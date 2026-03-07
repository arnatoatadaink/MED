"""src/rag/retrievers/arxiv.py — ArXiv 論文検索レトリーバー"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

from src.rag.retriever import BaseRetriever, RawResult

logger = logging.getLogger(__name__)

_ARXIV_API = "https://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom"}


class ArXivRetriever(BaseRetriever):
    """ArXiv API を使った学術論文検索。API キー不要。"""

    @property
    def source_name(self) -> str:
        return "arxiv"

    def is_available(self) -> bool:
        return True  # 公開 API のため常に利用可能

    async def search(self, query: str, max_results: int = 5) -> list[RawResult]:
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx not installed. Run: pip install httpx")

        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(_ARXIV_API, params=params)
            resp.raise_for_status()
            content = resp.text

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

            title = title_el.text.strip() if title_el is not None else ""
            summary = summary_el.text.strip() if summary_el is not None else ""
            arxiv_url = id_el.text.strip() if id_el is not None else ""

            # 著者
            authors = [
                a.findtext("atom:name", namespaces=_NS) or ""
                for a in entry.findall("atom:author", _NS)
            ]

            results.append(RawResult(
                title=title,
                content=summary[:2000],
                url=arxiv_url,
                source=self.source_name,
                score=1.0,  # ArXiv は relevance スコアを提供しないため固定
                metadata={
                    "authors": authors,
                    "published": entry.findtext("atom:published", namespaces=_NS) or "",
                },
            ))

        return results
