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
        """デフォルトのレトリーバーを登録する。retrievers.yaml の設定を反映。"""
        from src.rag.retrievers.arxiv import ArXivRetriever
        from src.rag.retrievers.github import GitHubRetriever
        from src.rag.retrievers.stackoverflow import StackOverflowRetriever
        from src.rag.retrievers.tavily import TavilyRetriever

        cfg = self._load_config()
        sources_cfg = cfg.get("sources", {})

        # SO 設定
        so_cfg = sources_cfg.get("stackoverflow", {})
        so_retriever = StackOverflowRetriever(
            min_answer_score=int(so_cfg.get("min_answer_score", 1)),
            prefer_accepted=bool(so_cfg.get("prefer_accepted", True)),
        )

        # ArXiv 設定
        arxiv_cfg = sources_cfg.get("arxiv", {})
        arxiv_categories = arxiv_cfg.get("categories", None)
        arxiv_retriever = ArXivRetriever(categories=arxiv_categories)

        for retriever in [
            GitHubRetriever(),
            so_retriever,
            TavilyRetriever(),
            arxiv_retriever,
        ]:
            self._retrievers[retriever.source_name] = retriever
            logger.debug(
                "Registered retriever: %s (available=%s)",
                retriever.source_name, retriever.is_available(),
            )

    @staticmethod
    def _load_config() -> dict:
        """retrievers.yaml を読み込む。"""
        from pathlib import Path

        import yaml

        cfg_path = Path(__file__).parent.parent.parent / "configs" / "retrievers.yaml"
        try:
            with open(cfg_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            logger.debug("Could not load retrievers.yaml; using defaults")
            return {}

    def register(self, retriever: BaseRetriever) -> None:
        """カスタムレトリーバーを登録する。"""
        self._retrievers[retriever.source_name] = retriever

    async def search(
        self,
        query: str,
        max_results: int | None = None,
        sources: list[str] | None = None,
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
            except TimeoutError:
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
