#!/usr/bin/env python3
"""scripts/seed_and_mature.py — 外部RAG取得 → 重複排除 → FAISS投入 → Teacher成熟 の統合パイプライン

1パスで以下を実行する:
  1. 質問リストから外部RAG検索（GitHub/SO/Tavily/arXiv）
  2. 重複排除（ハッシュ + コサイン類似度）
  3. FAISS + SQLite に投入
  4. Teacher Model で品質審査 + 難易度タグ付け

使い方:
    # 質問ファイルから実行（LM Studio Teacher）
    poetry run python scripts/seed_and_mature.py \\
        --questions-file scripts/questions.txt \\
        --provider lmstudio \\
        --domain code

    # 内蔵質問で dry-run
    poetry run python scripts/seed_and_mature.py --dry-run

    # seed のみ（mature スキップ）
    poetry run python scripts/seed_and_mature.py \\
        --questions-file scripts/questions.txt \\
        --no-mature

    # mature のみ（既存の未審査ドキュメントを審査）
    poetry run python scripts/seed_and_mature.py \\
        --mature-only --provider lmstudio --limit 50

    # 特定クエリ1件
    poetry run python scripts/seed_and_mature.py \\
        --query "How to use FAISS IVF index" \\
        --provider lmstudio --domain code
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 内蔵質問集（--questions-file 省略時に使用）──────────────────
_BUILTIN_QUESTIONS = [
    # FAISS / Vector Search
    "How to use FAISS for similarity search in Python?",
    "What is the difference between IndexFlatIP and IndexIVFFlat in FAISS?",
    "How to implement Product Quantization (PQ) with FAISS?",
    "FAISS GPU acceleration with CUDA examples",
    # RAG
    "How to implement RAG (Retrieval Augmented Generation) from scratch?",
    "What is Corrective RAG and how does it improve retrieval quality?",
    "HyDE (Hypothetical Document Embeddings) implementation in Python",
    "Reciprocal Rank Fusion (RRF) algorithm for combining search results",
    # Embedding / Reranking
    "How to use sentence-transformers for text embedding?",
    "Cross-encoder reranker implementation for document reranking",
    "Learning to Rank (LTR) with linear models for search",
    # Knowledge Graph
    "How to build a knowledge graph with NetworkX in Python?",
    "Entity extraction from text using LLM APIs",
    "Neo4j Cypher query examples for graph traversal",
    # LLM Fine-tuning
    "What is LoRA and how to fine-tune with PEFT?",
    "TinyLoRA: efficient fine-tuning with minimal parameters",
    "GRPO vs PPO for RLHF training comparison",
    "Knowledge distillation for small language models",
    "SFT (Supervised Fine-Tuning) with Hugging Face Transformers",
    # Infrastructure
    "Docker sandbox for safe code execution in Python",
    "FastAPI async endpoint implementation patterns",
    "aiosqlite async database operations in Python",
    "Gradio multi-tab interface with state management",
    # Python / ML
    "Python asyncio patterns for concurrent API calls",
    "PyTorch training loop with mixed precision",
    "Pydantic settings management with YAML config",
    "pytest fixtures and async test patterns",
    # Advanced Topics
    "Iterative retrieval with multi-hop reasoning",
    "Memory consolidation patterns for RAG systems",
    "Query expansion and rewriting for search systems",
    "Curriculum learning for model training difficulty scheduling",
]


async def seed_and_mature(
    queries: list[str],
    domain: str,
    top_k: int,
    provider: str | None,
    dry_run: bool,
    skip_mature: bool,
    dedup_threshold: float,
) -> dict:
    """外部RAG取得 → 重複排除 → FAISS投入 → Teacher成熟。"""
    from src.memory.deduplicator import Deduplicator
    from src.memory.embedder import Embedder
    from src.memory.memory_manager import MemoryManager
    from src.memory.schema import Document, Domain, SourceMeta, SourceType
    from src.rag.retriever import RetrieverRouter

    stats = {
        "queries": len(queries),
        "retrieved": 0,
        "duplicates": 0,
        "added": 0,
        "reviewed": 0,
        "approved": 0,
        "tagged": 0,
        "errors": 0,
    }

    if dry_run:
        logger.info("DRY RUN — %d queries would be processed:", len(queries))
        for i, q in enumerate(queries):
            logger.info("  [%d] %s", i + 1, q)
        return stats

    retriever = RetrieverRouter()
    embedder = Embedder()
    mm = MemoryManager(embedder=embedder)
    await mm.initialize()
    dedup = Deduplicator(near_dup_threshold=dedup_threshold)

    # 既存ハッシュを取得（完全一致排除用）
    existing_hashes: dict[str, str] = {}
    try:
        cursor = await mm.store._db.execute(
            "SELECT id, content_hash FROM documents WHERE content_hash IS NOT NULL"
        )
        for row in await cursor.fetchall():
            existing_hashes[row["id"]] = row["content_hash"]
        logger.info("Loaded %d existing content hashes for dedup", len(existing_hashes))
    except Exception:
        logger.warning("Could not load existing hashes; dedup will rely on near-dup only")

    _SOURCE_MAP = {s.value: s for s in SourceType}
    new_doc_ids: list[str] = []

    # ── Phase 1: 外部RAG → 重複排除 → FAISS投入 ──────────
    logger.info("=== Phase 1: Seed (%d queries, top_k=%d) ===", len(queries), top_k)

    for qi, query in enumerate(queries):
        logger.info("[%d/%d] Searching: %s", qi + 1, len(queries), query[:80])

        try:
            results = await retriever.search(query, max_results=top_k)
        except Exception as e:
            logger.warning("  RAG search failed: %s", e)
            stats["errors"] += 1
            continue

        stats["retrieved"] += len(results)

        for result in results:
            content = getattr(result, "content", "")
            if not content or len(content.strip()) < 50:
                continue

            # 重複チェック（ハッシュ）
            content_hash = dedup.content_hash(content)
            dup_result = dedup.check(
                content_hash=content_hash,
                existing_hashes=existing_hashes,
            )
            if dup_result.is_duplicate:
                stats["duplicates"] += 1
                continue

            # FAISS 投入
            try:
                doc = Document(
                    content=content,
                    domain=Domain(domain),
                    source=SourceMeta(
                        source_type=_SOURCE_MAP.get(
                            getattr(result, "source", "manual"), SourceType.MANUAL
                        ),
                        url=getattr(result, "url", ""),
                        title=getattr(result, "title", ""),
                    ),
                )
                doc_id = await mm.add(doc)
                new_doc_ids.append(doc_id)
                existing_hashes[doc_id] = content_hash
                stats["added"] += 1
            except Exception as e:
                logger.warning("  Add failed: %s", e)
                stats["errors"] += 1

        logger.info(
            "  Retrieved %d, added %d new (total: %d)",
            len(results), stats["added"], len(new_doc_ids),
        )

    logger.info(
        "Phase 1 done: retrieved=%d, duplicates=%d, added=%d",
        stats["retrieved"], stats["duplicates"], stats["added"],
    )

    # ── Phase 2: Teacher 成熟（審査 + 難易度タグ） ────────
    if skip_mature or not new_doc_ids:
        if not new_doc_ids:
            logger.info("No new documents to mature")
        else:
            logger.info("Skipping maturation (--no-mature)")
        await mm.close()
        return stats

    if provider is None:
        logger.info("Skipping maturation (no --provider specified)")
        await mm.close()
        return stats

    logger.info("=== Phase 2: Mature %d new docs (provider=%s) ===", len(new_doc_ids), provider)

    from src.llm.gateway import LLMGateway
    from src.memory.maturation.difficulty_tagger import DifficultyTagger
    from src.memory.maturation.reviewer import MemoryReviewer

    gateway = LLMGateway()
    reviewer = MemoryReviewer(gateway, mm.store, provider=provider)
    tagger = DifficultyTagger(gateway, provider=provider)

    for i, doc_id in enumerate(new_doc_ids):
        doc = await mm.store.get(doc_id)
        if doc is None:
            continue

        # 審査
        try:
            start = time.monotonic()
            result = await reviewer.review(doc)
            elapsed = time.monotonic() - start
            stats["reviewed"] += 1
            if result.approved:
                stats["approved"] += 1
            status = "PASS" if result.approved else "FAIL"
            logger.info(
                "  [%d/%d] Review %s (%.1fs, quality=%.2f): %s",
                i + 1, len(new_doc_ids), status, elapsed,
                result.quality_score, doc.content[:60],
            )
        except Exception as e:
            logger.warning("  [%d/%d] Review error: %s", i + 1, len(new_doc_ids), e)
            stats["errors"] += 1
            continue

        # 難易度タグ
        try:
            level = await tagger.tag(doc)
            await mm.store.update_quality(doc.id, difficulty=level.value)
            stats["tagged"] += 1
            logger.info("    Difficulty: %s", level.value)
        except Exception as e:
            logger.warning("    Tagging error: %s", e)

    logger.info(
        "Phase 2 done: reviewed=%d, approved=%d, tagged=%d",
        stats["reviewed"], stats["approved"], stats["tagged"],
    )

    await mm.close()
    return stats


async def mature_only(
    limit: int, domain: str | None, provider: str,
) -> None:
    """既存の未審査ドキュメントのみ成熟させる。"""
    from src.llm.gateway import LLMGateway
    from src.memory.embedder import Embedder
    from src.memory.maturation.difficulty_tagger import DifficultyTagger
    from src.memory.maturation.reviewer import MemoryReviewer
    from src.memory.memory_manager import MemoryManager

    embedder = Embedder()
    mm = MemoryManager(embedder=embedder)
    await mm.initialize()
    gateway = LLMGateway()

    reviewer = MemoryReviewer(gateway, mm.store, provider=provider)
    tagger = DifficultyTagger(gateway, provider=provider)

    docs = await mm.store.get_unreviewed(domain=domain, limit=limit)
    if not docs:
        logger.info("No unreviewed documents found")
        await mm.close()
        return

    logger.info("=== Mature %d existing docs (provider=%s) ===", len(docs), provider)

    reviewed = 0
    approved = 0
    tagged = 0

    for i, doc in enumerate(docs):
        # 審査
        try:
            start = time.monotonic()
            result = await reviewer.review(doc)
            elapsed = time.monotonic() - start
            reviewed += 1
            if result.approved:
                approved += 1
            status = "PASS" if result.approved else "FAIL"
            logger.info(
                "  [%d/%d] Review %s (%.1fs, quality=%.2f): %s",
                i + 1, len(docs), status, elapsed,
                result.quality_score, doc.content[:60],
            )
        except Exception as e:
            logger.warning("  [%d/%d] Review error: %s", i + 1, len(docs), e)
            continue

        # 難易度タグ
        if doc.difficulty is None:
            try:
                level = await tagger.tag(doc)
                await mm.store.update_quality(doc.id, difficulty=level.value)
                tagged += 1
                logger.info("    Difficulty: %s", level.value)
            except Exception as e:
                logger.warning("    Tagging error: %s", e)

    logger.info("Done: reviewed=%d, approved=%d, tagged=%d", reviewed, approved, tagged)
    await mm.close()


def load_questions(args) -> list[str]:
    """質問リストをロードする。"""
    questions: list[str] = []

    if args.query:
        questions.append(args.query)

    if args.questions_file:
        path = Path(args.questions_file)
        if not path.exists():
            logger.error("Questions file not found: %s", path)
            sys.exit(1)
        lines = path.read_text(encoding="utf-8").splitlines()
        questions.extend(line.strip() for line in lines if line.strip() and not line.startswith("#"))

    if not questions:
        questions = list(_BUILTIN_QUESTIONS)
        logger.info("Using %d built-in questions", len(questions))

    return questions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed + Mature: External RAG → Dedup → FAISS → Teacher Review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 入力
    parser.add_argument("--query", type=str, help="Single query")
    parser.add_argument("--questions-file", type=str, help="File with one query per line (# = comment)")
    parser.add_argument("--domain", default="code", choices=["code", "academic", "general"])
    parser.add_argument("--top-k", type=int, default=10, help="Docs per query from RAG")

    # Teacher
    parser.add_argument("--provider", type=str, default=None, help="Teacher provider (e.g. lmstudio)")

    # 制御
    parser.add_argument("--dry-run", action="store_true", help="Preview queries without execution")
    parser.add_argument("--no-mature", action="store_true", help="Skip Teacher maturation")
    parser.add_argument("--mature-only", action="store_true", help="Only mature existing unreviewed docs")
    parser.add_argument("--limit", type=int, default=100, help="Max docs for --mature-only")
    parser.add_argument("--dedup-threshold", type=float, default=0.95, help="Near-dup cosine threshold")

    args = parser.parse_args()

    # --mature-only モード
    if args.mature_only:
        if not args.provider:
            parser.error("--mature-only requires --provider")
        asyncio.run(mature_only(args.limit, args.domain, args.provider))
        return

    questions = load_questions(args)

    stats = asyncio.run(seed_and_mature(
        queries=questions,
        domain=args.domain,
        top_k=args.top_k,
        provider=args.provider,
        dry_run=args.dry_run,
        skip_mature=args.no_mature,
        dedup_threshold=args.dedup_threshold,
    ))

    # サマリー
    print(f"\n{'='*50}")
    print(f"  SUMMARY")
    print(f"{'='*50}")
    for k, v in stats.items():
        print(f"  {k:>12}: {v}")


if __name__ == "__main__":
    main()
