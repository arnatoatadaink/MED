#!/usr/bin/env python3
"""scripts/seed_only.py — Rate limit 準拠の Retriever 毎タスク管理版 Seed スクリプト

実装方針:
  1. Query × Retriever のタスクを Retriever 毎のスタックに積む
  2. 各 Retriever の前回リクエスト時刻をチェック
  3. (現在時刻 - 前回リクエスト) >= 待機時間 なら非同期実行
  4. 待機時間中なら 1 秒スリープ
  5. すべてのスタックが消化できたら完了

使い方:
    # 内蔵質問で seed（率制御付き）
    poetry run python scripts/seed_only.py

    # 質問ファイルから実行
    poetry run python scripts/seed_only.py \\
        --questions-file scripts/questions.txt \\
        --domain code

    # 特定ソースのみ（arXiv + StackOverflow）
    poetry run python scripts/seed_only.py \\
        --sources arxiv,stackoverflow

    # dry-run で確認
    poetry run python scripts/seed_only.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 内蔵質問集 ──
_BUILTIN_QUESTIONS = [
    "How to use FAISS for similarity search in Python?",
    "What is the difference between IndexFlatIP and IndexIVFFlat in FAISS?",
    "How to implement Product Quantization (PQ) with FAISS?",
    "FAISS GPU acceleration with CUDA examples",
    "How to implement RAG (Retrieval Augmented Generation) from scratch?",
    "What is Corrective RAG and how does it improve retrieval quality?",
    "HyDE (Hypothetical Document Embeddings) implementation in Python",
    "Reciprocal Rank Fusion (RRF) algorithm for combining search results",
    "How to use sentence-transformers for text embedding?",
    "Cross-encoder reranker implementation for document reranking",
]

# ── Retriever 毎のレート制限（秒） ──
# github_docs は意図的に除外: seed_from_docs.py 専任（chunk_markdown パイプライン必須）
RATE_LIMITS = {
    "arxiv": 5.0,        # 公式推奨 3秒 + 並列実行対策
    "stackoverflow": 1.0,
    "github": 1.0,
    "tavily": 1.0,
}

DEFAULT_SOURCES = list(RATE_LIMITS.keys())


@dataclass
class TaskItem:
    """Retriever タスク: (query, retriever_name) のペア"""
    query: str
    retriever_name: str
    query_index: int  # ログ用


async def execute_retrieval(
    task: TaskItem,
    router,
    embedder,
    memory_manager,
    relevance_threshold: float = 0.25,
) -> int:
    """1つの Query × Retriever タスク実行。追加ドキュメント数を返す。"""
    query_idx = task.query_index
    retriever_name = task.retriever_name

    try:
        results = await router.search(task.query, max_results=10, sources=[retriever_name])
    except Exception as e:
        logger.warning(f"  [{retriever_name}] Query {query_idx}: search failed: {e}")
        return 0

    if not results:
        logger.debug(f"  [{retriever_name}] Query {query_idx}: no results")
        return 0

    query_vec = embedder.embed(task.query)
    added = 0

    for result in results:
        content = getattr(result, "content", "")
        if not content or len(content.strip()) < 50:
            continue

        content_vec = embedder.embed(content[:1000])
        similarity = float(np.dot(query_vec, content_vec))
        if similarity < relevance_threshold:
            continue

        # FAISS 追加（重複排除は memory_manager.add() で内部的に処理）
        try:
            doc_id = await memory_manager.add(
                content=content,
                source=retriever_name,
                url=getattr(result, "url", ""),
                title=getattr(result, "title", ""),
                metadata=getattr(result, "metadata", {}),
            )
            added += 1
        except Exception as e:
            logger.debug(f"  Failed to add document: {e}")

    if added > 0:
        logger.info(f"  [{retriever_name}] Query {query_idx}: added {added} docs")

    return added


async def seed_with_rate_control(
    queries: list[str],
    sources: list[str],
    dry_run: bool = False,
) -> dict:
    """
    Rate limit を遵守しながら Query × Retriever タスクを実行。

    実装:
      1. 各 Retriever ごとにタスクスタックを作成
      2. メインループで:
         - 各 Retriever について、待機時間が経過したかチェック
         - 経過していたら非同期タスク実行 + タイムスタンプ更新
         - 待機中なら 1 秒スリープして再度チェック
      3. すべてのスタックが空になったら終了
    """
    from src.rag.retriever import RetrieverRouter
    from src.memory.embedder import Embedder
    from src.memory.memory_manager import MemoryManager

    # ── 初期化 ──
    router = RetrieverRouter()
    embedder = Embedder()
    memory_manager = MemoryManager(embedder=embedder)
    await memory_manager.initialize()

    # ── タスクスタック初期化 ──
    tasks_stack: dict[str, list[TaskItem]] = {s: [] for s in sources}

    for qi, query in enumerate(queries):
        for source in sources:
            tasks_stack[source].append(TaskItem(
                query=query,
                retriever_name=source,
                query_index=qi + 1,
            ))

    logger.info(f"Created task stacks: {sum(len(s) for s in tasks_stack.values())} tasks total")
    logger.info(f"Task distribution: {', '.join(f'{s}: {len(tasks_stack[s])}' for s in sources)}")

    if dry_run:
        logger.info("[DRY-RUN] Task stacks created. No execution.")
        await memory_manager.close()
        return {
            "queries": len(queries),
            "sources": len(sources),
            "total_tasks": sum(len(s) for s in tasks_stack.values()),
            "added": 0,
            "errors": 0,
        }

    # ── メインループ ──
    last_request_times: dict[str, float] = {s: 0.0 for s in sources}
    running_tasks: set[asyncio.Task] = set()
    stats = {"added": 0, "errors": 0}
    start_time = time.time()

    logger.info(f"Starting rate-controlled seed (rate limits: {RATE_LIMITS})")
    print(f"\n{'='*70}")
    print(f"Rate-Controlled Seed Pipeline")
    print(f"{'='*70}\n")

    while any(tasks_stack.values()) or running_tasks:
        # ── 実行可能なタスクをチェック ──
        executed = False

        for source in sources:
            if not tasks_stack[source]:
                continue

            now = time.monotonic()
            last_time = last_request_times[source]
            wait_interval = RATE_LIMITS[source]
            elapsed = now - last_time

            if elapsed >= wait_interval:
                # ── タスク実行 ──
                task = tasks_stack[source].pop(0)

                async def run_task(t: TaskItem):
                    try:
                        added = await execute_retrieval(
                            t, router, embedder, memory_manager
                        )
                        stats["added"] += added
                    except Exception as e:
                        logger.exception(f"Task error: {e}")
                        stats["errors"] += 1

                coroutine = run_task(task)
                new_task = asyncio.create_task(coroutine)
                running_tasks.add(new_task)
                new_task.add_done_callback(running_tasks.discard)

                last_request_times[source] = now
                executed = True
                logger.debug(
                    f"[{source}] Launched task {task.query_index}. "
                    f"Remaining: {len(tasks_stack[source])}"
                )

        if not executed and running_tasks:
            # ── 実行中のタスク待機 + スリープ ──
            await asyncio.sleep(1)
        elif not executed and not running_tasks:
            # ── スタック空 + タスク実行なし → 完了 ──
            break
        elif executed:
            # ── タスク実行直後、1秒スリープで次のリクエストを待つ ──
            await asyncio.sleep(1)

    # ── 残りのタスク完了待機 ──
    if running_tasks:
        logger.info(f"Waiting for {len(running_tasks)} remaining tasks...")
        await asyncio.gather(*running_tasks, return_exceptions=True)

    elapsed_time = time.time() - start_time
    await memory_manager.close()

    return {
        "queries": len(queries),
        "sources": len(sources),
        "total_tasks": len(queries) * len(sources),
        "added": stats["added"],
        "errors": stats["errors"],
        "elapsed_seconds": elapsed_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rate-controlled seed pipeline with per-retriever task stacks"
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        help="File with one question per line",
    )
    parser.add_argument(
        "--domain",
        choices=["code", "academic", "general"],
        default="code",
        help="Domain filter for FAISS",
    )
    parser.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCES),
        help=f"Comma-separated retrievers (default: {','.join(DEFAULT_SOURCES)})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview queries and tasks without execution",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Max queries to process",
    )

    args = parser.parse_args()

    # ── クエリロード ──
    if args.questions_file:
        queries = [
            line.strip()
            for line in args.questions_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    else:
        queries = _BUILTIN_QUESTIONS

    if args.limit:
        queries = queries[: args.limit]

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    sources = [s for s in sources if s in RATE_LIMITS]

    if not sources:
        logger.error("No valid sources. Available: %s", ", ".join(RATE_LIMITS.keys()))
        sys.exit(1)

    logger.info(f"Loaded {len(queries)} queries from {args.questions_file or 'built-in'}")
    logger.info(f"Sources: {', '.join(sources)}")
    logger.info(f"Domain: {args.domain}")

    # ── 実行 ──
    stats = asyncio.run(
        seed_with_rate_control(
            queries=queries,
            sources=sources,
            dry_run=args.dry_run,
        )
    )

    # ── 結果表示 ──
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")
    print(f"  Queries: {stats['queries']}")
    print(f"  Sources: {stats['sources']}")
    print(f"  Total Tasks: {stats['total_tasks']}")
    print(f"  Added Documents: {stats['added']}")
    print(f"  Errors: {stats['errors']}")
    if "elapsed_seconds" in stats:
        print(f"  Elapsed Time: {stats['elapsed_seconds']:.1f}s")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
