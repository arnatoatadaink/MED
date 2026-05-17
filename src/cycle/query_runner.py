"""src/cycle/query_runner.py — サイクル用 QueryRunner

CollectionTask.queries を使って外部ソース (arXiv/GitHub/SO/Tavily) を検索し、
重複排除・関連性フィルタを経て FAISS に投入する（mature なし）。

SMALL_CLUSTER    : 全利用可能ソースを検索してドキュメント数を拡充。
SOURCE_IMBALANCE : dominant_source を除外し多様性を確保。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.cycle.schema import CollectionTask, GapType

logger = logging.getLogger(__name__)

# seed_and_mature._DOMAIN_FLAG_MAP と同期
_DOMAIN_FLAG_MAP: dict[str, str] = {
    "arxiv": "strict",
    "github": "on_domain",
    "stackoverflow": "practical_reference",
    "tavily": "practical_reference",
    "web_docs": "practical_reference",
}

_MAX_QUERIES_PER_TASK = 6
_MIN_CONTENT_LEN = 50
_MIN_CONTENT_LEN_TAVILY = 300


@dataclass
class QueryRunnerConfig:
    """QueryRunner の実行パラメータ。"""

    top_k: int = 5
    relevance_threshold: float = 0.25
    domain: str = "code"
    max_queries: int = _MAX_QUERIES_PER_TASK
    disabled_sources: frozenset[str] = field(default_factory=frozenset)


class QueryRunner:
    """CollectionTask の queries を外部検索して FAISS に投入する。

    1つのインスタンスを複数タスクで再利用するために initialize() / close() を使う。
    """

    def __init__(self, config: Optional[QueryRunnerConfig] = None) -> None:
        self._cfg = config or QueryRunnerConfig()
        self._router: Optional[object] = None
        self._embedder: Optional[object] = None
        self._mm: Optional[object] = None
        self._dedup: Optional[object] = None
        self._existing_hashes: dict[str, str] = {}

    async def initialize(self) -> None:
        """共有リソースを初期化する。run_task() の前に呼ぶ。"""
        from src.memory.deduplicator import Deduplicator
        from src.memory.embedder import Embedder
        from src.memory.memory_manager import MemoryManager
        from src.rag.retriever import RetrieverRouter

        self._router = RetrieverRouter()
        self._embedder = Embedder()
        self._mm = MemoryManager(embedder=self._embedder)
        await self._mm.initialize()
        self._dedup = Deduplicator(near_dup_threshold=0.95)

        # 既存ハッシュを取得（完全一致排除）
        try:
            cursor = await self._mm.store._db.execute(
                "SELECT id, content_hash FROM documents WHERE content_hash IS NOT NULL"
            )
            for row in await cursor.fetchall():
                self._existing_hashes[row["id"]] = row["content_hash"]
        except Exception:
            logger.warning("Could not load existing hashes; dedup by near-dup only")

        logger.info(
            "QueryRunner initialized: %d existing hashes, sources=%s",
            len(self._existing_hashes),
            self._router.available_sources(),
        )

    async def close(self) -> None:
        """共有リソースを解放する。"""
        if self._mm is not None:
            await self._mm.close()

    async def run_task(self, task: CollectionTask) -> dict:
        """1タスクを実行して統計 dict を返す。

        Returns:
            {"added": int, "retrieved": int, "duplicates": int,
             "irrelevant": int, "errors": int}
        """
        if self._mm is None:
            raise RuntimeError("QueryRunner.initialize() must be called first")

        sources = self._select_sources(task)
        stats: dict[str, int] = {
            "added": 0, "retrieved": 0,
            "duplicates": 0, "irrelevant": 0, "errors": 0,
        }

        queries = task.queries[: self._cfg.max_queries]
        if not queries:
            logger.warning("Task %s has no queries — skipping", task.task_id[:8])
            return stats

        logger.info(
            "QueryRunner task %s (gap=%s, %d queries, sources=%s)",
            task.task_id[:8], task.gap_type.value, len(queries), sources,
        )

        for qi, query in enumerate(queries):
            logger.info("[%d/%d] %s", qi + 1, len(queries), query[:80])
            await self._run_query(query, sources, stats)

        logger.info(
            "Task %s done: retrieved=%d added=%d dup=%d irrel=%d err=%d",
            task.task_id[:8],
            stats["retrieved"], stats["added"],
            stats["duplicates"], stats["irrelevant"], stats["errors"],
        )
        return stats

    # ── 内部実装 ────────────────────────────────────────────────

    def _select_sources(self, task: CollectionTask) -> Optional[list[str]]:
        """gap_type と signals からソースリストを決定する。

        SOURCE_IMBALANCE: dominant_source を除外して多様性を確保。
        その他: 利用可能な全ソース（None = 制限なし）。
        disabled_sources が設定されている場合は対象外ソースを除外する。
        """
        available = self._router.available_sources()
        if self._cfg.disabled_sources:
            available = [s for s in available if s not in self._cfg.disabled_sources]

        if task.gap_type != GapType.SOURCE_IMBALANCE:
            return available if self._cfg.disabled_sources else None

        dominant = task.signals.get("dominant_source", "")
        filtered = [s for s in available if s != dominant]
        if not filtered:
            logger.warning(
                "SOURCE_IMBALANCE: no sources remain after excluding '%s' — using all",
                dominant,
            )
            return available or None
        logger.info("SOURCE_IMBALANCE: excluding '%s', using %s", dominant, filtered)
        return filtered

    async def _run_query(
        self,
        query: str,
        sources: Optional[list[str]],
        stats: dict,
    ) -> None:
        """1クエリを外部検索して FAISS に投入する。"""
        from src.memory.schema import Document, Domain, SourceMeta, SourceType

        try:
            results = await self._router.search(
                query,
                max_results=self._cfg.top_k,
                sources=sources,
            )
        except Exception as exc:
            logger.warning("RAG search failed: %s", exc)
            stats["errors"] += 1
            return

        stats["retrieved"] += len(results)
        query_vec = self._embedder.embed(query)

        _source_map = {s.value: s for s in SourceType}

        for result in results:
            content = getattr(result, "content", "") or ""
            source = getattr(result, "source", "")
            min_len = _MIN_CONTENT_LEN_TAVILY if source == "tavily" else _MIN_CONTENT_LEN
            if not content or len(content.strip()) < min_len:
                stats["irrelevant"] += 1
                continue

            # 関連性チェック
            content_vec = self._embedder.embed(content[:1000])
            similarity = float(np.dot(query_vec, content_vec))
            if similarity < self._cfg.relevance_threshold:
                stats["irrelevant"] += 1
                continue

            # ブラックリスト
            url = getattr(result, "url", "") or ""
            title = getattr(result, "title", "") or ""
            if await self._mm.store.is_blacklisted(source_url=url, source_title=title):
                stats["duplicates"] += 1
                continue

            # 重複チェック
            content_hash = self._dedup.content_hash(content)
            dup = self._dedup.check(
                content_hash=content_hash,
                existing_hashes=self._existing_hashes,
            )
            if dup.is_duplicate:
                stats["duplicates"] += 1
                continue

            # FAISS 投入
            try:
                extra = dict(getattr(result, "metadata", {}) or {})
                extra.setdefault("domain_flag", _DOMAIN_FLAG_MAP.get(source, "on_domain"))
                doc = Document(
                    content=content,
                    domain=Domain(self._cfg.domain),
                    source=SourceMeta(
                        source_type=_source_map.get(source, SourceType.MANUAL),
                        url=url,
                        title=title,
                        extra=extra,
                    ),
                    content_hash=content_hash,
                )
                doc_id = await self._mm.add(doc)
                self._existing_hashes[doc_id] = content_hash
                stats["added"] += 1
            except Exception as exc:
                logger.warning("Add failed: %s", exc)
                stats["errors"] += 1
