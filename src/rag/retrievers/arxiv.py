"""src/rag/retrievers/arxiv.py — ArXiv 論文検索レトリーバー

カテゴリフィルタを適用し、プロジェクト関連分野の論文のみを返す。

ArXiv API 利用規約に基づき、リクエスト間隔を最低3秒空ける。
https://info.arxiv.org/help/api/tou.html
レート制限は BaseRetriever 共通機構で管理（retriever.py の _RATE_LIMIT_INTERVALS）。
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

from src.rag.retriever import BaseRetriever, RawResult

logger = logging.getLogger(__name__)

_ARXIV_API = "https://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom"}

# デフォルトの許可カテゴリ（cs.AI, cs.LG, cs.CL, cs.IR, cs.DB, stat.ML）
_DEFAULT_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.IR", "cs.DB", "stat.ML"]


class ArXivRetriever(BaseRetriever):
    """ArXiv API を使った学術論文検索。

    カテゴリフィルタを API クエリに適用し、関連分野の論文に限定する。
    ArXiv API の利用規約（3秒/1リクエスト）は BaseRetriever のレート制限で遵守。

    Args:
        categories: 許可する ArXiv カテゴリリスト。
            None の場合デフォルトカテゴリ（cs.AI, cs.LG, cs.CL 等）を使用。
            空リスト [] の場合はフィルタなし（全カテゴリ）。
    """

    def __init__(self, categories: list[str] | None = None) -> None:
        if categories is None:
            self._categories = list(_DEFAULT_CATEGORIES)
        else:
            self._categories = categories

    @property
    def source_name(self) -> str:
        return "arxiv"

    def is_available(self) -> bool:
        return True  # 公開 API のため常に利用可能

    async def _do_search(self, query: str, max_results: int = 5) -> list[RawResult]:
        import httpx

        # カテゴリフィルタ付きクエリ構築
        search_query = self._build_query(query)

        params = {
            "search_query": search_query,
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

            # カテゴリ抽出
            categories = [
                c.get("term", "")
                for c in entry.findall("atom:category", _NS)
            ]

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
        """カテゴリフィルタ付き ArXiv API クエリを構築する。

        例: categories=["cs.AI","cs.LG"] の場合
            → "all:FAISS vector search AND (cat:cs.AI OR cat:cs.LG)"
        """
        base = f"all:{query}"
        if not self._categories:
            return base

        cat_filter = " OR ".join(f"cat:{c}" for c in self._categories)
        return f"{base} AND ({cat_filter})"
