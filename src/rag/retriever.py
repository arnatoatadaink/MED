"""src/rag/retriever.py — RAG 基底クラス + Retriever Router

外部ソース（GitHub/SO/Tavily/ArXiv）への検索を統一インターフェースで管理する。
ソース別に検索を並列実行し、結果をマージして返す。

使い方:
    from src.rag.retriever import RetrieverRouter

    router = RetrieverRouter()
    results = await router.search("Python FAISS usage", max_results=10)
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================================
# データクラス
# ============================================================================


@dataclass
class RawResult:
    """外部検索の生結果。chunker に渡す前の形式。"""

    title: str
    content: str
    url: str
    source: str  # "github", "stackoverflow", "tavily", "arxiv"
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


# ============================================================================
# 抽象基底クラス
# ============================================================================


class BaseRetriever(ABC):
    """外部検索ソースの抽象基底クラス。"""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """ソース識別子。"""
        ...

    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> list[RawResult]:
        """検索クエリを実行し、結果を返す。"""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """APIキー等の設定が揃っているか。"""
        ...


# ============================================================================
# Retriever Router
# ============================================================================


class RetrieverRouter:
    """複数の外部検索ソースを並列実行し、結果をまとめて返す。

    Args:
        timeout: 各ソースのタイムアウト秒数。
        max_results_per_source: ソースあたりの最大取得件数。
    """

    def __init__(
        self,
        timeout: float = 30.0,
        max_results_per_source: int = 5,
    ) -> None:
        self._retrievers: dict[str, BaseRetriever] = {}
        self._timeout = timeout
        self._max_results = max_results_per_source
        self._register_defaults()

    def _register_defaults(self) -> None:
        """デフォルトのレトリーバーを登録する。"""
        from src.rag.retrievers.github import GitHubRetriever
        from src.rag.retrievers.stackoverflow import StackOverflowRetriever
        from src.rag.retrievers.tavily import TavilyRetriever
        from src.rag.retrievers.arxiv import ArXivRetriever

        for retriever in [
            GitHubRetriever(),
            StackOverflowRetriever(),
            TavilyRetriever(),
            ArXivRetriever(),
        ]:
            self._retrievers[retriever.source_name] = retriever
            logger.debug(
                "Registered retriever: %s (available=%s)",
                retriever.source_name, retriever.is_available(),
            )

    def register(self, retriever: BaseRetriever) -> None:
        """カスタムレトリーバーを登録する。"""
        self._retrievers[retriever.source_name] = retriever

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        sources: Optional[list[str]] = None,
    ) -> list[RawResult]:
        """全利用可能ソースを並列検索し、結果をまとめて返す。

        Args:
            query: 検索クエリ。
            max_results: 全ソース合計の最大件数。None = 制限なし。
            sources: 使用するソース名のリスト。None = 全利用可能ソース。

        Returns:
            スコア降順の RawResult リスト。
        """
        active_retrievers = [
            r for name, r in self._retrievers.items()
            if r.is_available() and (sources is None or name in sources)
        ]

        if not active_retrievers:
            logger.warning("No available retrievers for query: %r", query[:50])
            return []

        per_source = self._max_results

        async def _fetch(retriever: BaseRetriever) -> list[RawResult]:
            try:
                return await asyncio.wait_for(
                    retriever.search(query, max_results=per_source),
                    timeout=self._timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("Retriever %s timed out", retriever.source_name)
                return []
            except Exception:
                logger.exception("Retriever %s failed", retriever.source_name)
                return []

        tasks = [_fetch(r) for r in active_retrievers]
        results_per_source = await asyncio.gather(*tasks)

        all_results: list[RawResult] = []
        for results in results_per_source:
            all_results.extend(results)

        all_results.sort(key=lambda x: x.score, reverse=True)

        if max_results is not None:
            all_results = all_results[:max_results]

        logger.info(
            "Search query=%r sources=%d total_results=%d",
            query[:50], len(active_retrievers), len(all_results),
        )
        return all_results

    def available_sources(self) -> list[str]:
        """利用可能なソース名のリストを返す。"""
        return [name for name, r in self._retrievers.items() if r.is_available()]
