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
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm.daily_usage_tracker import DailyUsageTracker

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
# レート制限
# ============================================================================

# ソース別のAPI実行回数（日次上限）
_RATE_LIMIT_COUNTS: dict[str, int] = {
    "stackoverflow": 300,  # 429対策、日次制限
}

# 日次使用量トラッカー（遅延初期化）
_daily_tracker: "DailyUsageTracker | None" = None


async def _get_daily_tracker() -> "DailyUsageTracker":
    """DailyUsageTracker をモジュールレベルで遅延初期化して返す。

    asyncio.Lock を使わず SO セマフォ（concurrency=1）に依存して二重初期化を防ぐ。
    """
    global _daily_tracker
    if _daily_tracker is None:
        from src.llm.daily_usage_tracker import DailyUsageTracker as _DUT
        tracker = _DUT("data/openrouter_usage.db")
        await tracker.initialize()
        _daily_tracker = tracker
    return _daily_tracker




# ソース別のリクエスト間隔（秒）。
_RATE_LIMIT_INTERVALS: dict[str, float] = {
    "arxiv":         10.0,  # 公式推奨 3秒 + バースト 429 対策
    "stackoverflow": 12.0,  # 400/429 対策
}
_DEFAULT_RATE_LIMIT = 1.0

# ソース別の最終リクエスト時刻
_last_request_times: dict[str, float] = {}

# 2026/05/04 19:19(JST) arxivで429が出るため追加、同時取得した件数分インターバルを延長する
# ソース毎の前回取得ドキュメント数
_last_request_counts: dict[str, int] = {}

async def _rate_limit_wait(source: str,last_results: int = 0) -> None:
    """ソース別のレート制限待機。各 API に対して最低間隔を保証する。"""
    interval = _RATE_LIMIT_INTERVALS.get(source, _DEFAULT_RATE_LIMIT)*last_results
    last = _last_request_times.get(source, 0.0)
    now = time.monotonic()
    elapsed = now - last
    if elapsed < interval:
        wait = interval - elapsed
        logger.debug("Rate limit [%s]: waiting %.1fs", source, wait)
        await asyncio.sleep(wait)
    _last_request_times[source] = time.monotonic()


# ============================================================================
# ソース別並列数制限
# ============================================================================

# ソース別の最大同時リクエスト数。同一ソースへの並列アクセスを防ぐ。
# 異なるソース同士（例: SO + arXiv）は並列実行を許可する。
_SOURCE_CONCURRENCY: dict[str, int] = {
    "arxiv": 1,
    "stackoverflow": 1,
}
_DEFAULT_CONCURRENCY = 2


# ============================================================================
# 抽象基底クラス
# ============================================================================


class BaseRetriever(ABC):
    """外部検索ソースの抽象基底クラス。

    サブクラスは _do_search を実装する。search() がソース別並列制限 +
    レート制限を適用した上で _do_search を呼び出す。

    並列制限セマフォはインスタンスに保持し、イベントループ跨ぎを防ぐ。
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """ソース識別子。"""
        ...

    @abstractmethod
    async def _do_search(self, query: str, max_results: int = 5) -> list[RawResult]:
        """検索クエリを実行し、結果を返す（サブクラスで実装）。"""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """APIキー等の設定が揃っているか。"""
        ...

    def _get_sem(self) -> asyncio.Semaphore:
        """ソース別並列制限セマフォをインスタンス単位で遅延生成する。"""
        sem: asyncio.Semaphore | None = getattr(self, "_sem", None)
        if sem is None:
            limit = _SOURCE_CONCURRENCY.get(self.source_name, _DEFAULT_CONCURRENCY)
            self._sem: asyncio.Semaphore = asyncio.Semaphore(limit)
        return self._sem

    async def search(self, query: str, max_results: int = 5) -> list[RawResult]:
        """ソース別並列制限 + レート制限付き検索。サブクラスは _do_search を実装する。"""
        # 日次上限チェック（セマフォ取得前にスキップ判定）
        daily_limit = _RATE_LIMIT_COUNTS.get(self.source_name)
        if daily_limit is not None:
            from src.llm.daily_usage_tracker import DailyLimitExceeded
            try:
                tracker = await _get_daily_tracker()
                await tracker.check_and_increment(self.source_name, daily_limit)
            except DailyLimitExceeded as exc:
                logger.warning(
                    "Daily limit reached for %s (%d/%d) — skipping query: %r",
                    self.source_name, exc.current, exc.limit, query[:50],
                )
                return []

        async with self._get_sem():
            last_count = _last_request_counts.get(self.source_name, 0)
            await _rate_limit_wait(self.source_name, last_count)
            ret = await self._do_search(query, max_results)
            _last_request_counts[self.source_name] = len(ret)
            return ret


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
        timeout: float = 120.0,
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
