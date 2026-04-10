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
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# .env から環境変数をロード（TAVILY_API_KEY 等）
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


_OPENROUTER_PROVIDERS = {"openrouter"}


def _utc_reset_action(provider: str) -> str:
    """UTC 日次リセット前の停止チェック（OpenRouter のみ対象）。

    Returns:
        'continue' — 通常処理継続
        'wait'     — UTC 23:50〜23:55: リセット待機中
        'stop'     — UTC 23:55 超: 即時停止
    """
    if provider not in _OPENROUTER_PROVIDERS:
        return "continue"
    now = datetime.now(timezone.utc)
    h, m = now.hour, now.minute
    if h == 23 and m >= 55:
        return "stop"
    if h == 23 and m >= 50:
        return "wait"
    return "continue"


async def _wait_for_utc_reset() -> None:
    """UTC 23:50〜00:00 まで待機してリセットを待つ。"""
    while True:
        now = datetime.now(timezone.utc)
        h, m, s = now.hour, now.minute, now.second
        if not (h == 23 and m >= 50):
            logger.info("UTC リセット完了 — 処理を再開します")
            return
        remaining = (24 * 3600) - (h * 3600 + m * 60 + s)
        logger.info("OpenRouter UTC リセット待機中... 残 %ds (UTC 23:%02d)", remaining, m)
        await asyncio.sleep(30)


def _load_seed_filters() -> dict:
    """retrievers.yaml から seed_filters 設定を読み込む。"""
    import yaml

    cfg_path = _ROOT / "configs" / "retrievers.yaml"
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("seed_filters", {})
    except Exception:
        logger.warning("Could not load seed_filters from retrievers.yaml; using defaults")
        return {}


def _has_code(content: str) -> bool:
    """コンテンツにコードブロックが含まれるか判定する。"""
    return any(marker in content for marker in [
        "```", "\ndef ", "\nimport ", "\nclass ", "\nfunction ",
        "def ", "import ", "class ", "function ",
    ])


async def seed_and_mature(
    queries: list[str],
    domain: str,
    top_k: int,
    provider: str | None,
    dry_run: bool,
    skip_mature: bool,
    dedup_threshold: float,
    model: str | None = None,
    exclude_sources: list[str] | None = None,
) -> dict:
    """外部RAG取得 → 重複排除 → 関連性フィルタ → FAISS投入 → Teacher成熟。"""
    import numpy as np

    from src.memory.deduplicator import Deduplicator
    from src.memory.embedder import Embedder
    from src.memory.memory_manager import MemoryManager
    from src.memory.schema import Document, Domain, SourceMeta, SourceType
    from src.rag.retriever import RetrieverRouter

    stats = {
        "queries": len(queries),
        "retrieved": 0,
        "duplicates": 0,
        "irrelevant": 0,
        "quality_filtered": 0,
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

    # Seed フィルタ設定
    seed_filters = _load_seed_filters()
    relevance_threshold = float(seed_filters.get("relevance_threshold", 0.25))
    quality_cfg = seed_filters.get("min_quality_filter", {})
    quality_filter_enabled = bool(quality_cfg.get("enabled", False))
    quality_min_length = int(quality_cfg.get("min_length", 500))

    logger.info(
        "Seed filters: relevance_threshold=%.2f, quality_filter=%s (min_len=%d)",
        relevance_threshold, quality_filter_enabled, quality_min_length,
    )

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

    # ── Phase 1: 外部RAG → 関連性フィルタ → 重複排除 → FAISS投入 ──────────
    logger.info("=== Phase 1: Seed (%d queries, top_k=%d) ===", len(queries), top_k)

    for qi, query in enumerate(queries):
        logger.info("[%d/%d] Searching: %s", qi + 1, len(queries), query[:80])

        try:
            # 除外ソースを除いた利用可能ソースを計算
            if exclude_sources:
                available = [s for s in retriever.available_sources() if s not in exclude_sources]
                results = await retriever.search(query, max_results=top_k, sources=available)
            else:
                results = await retriever.search(query, max_results=top_k)
        except Exception as e:
            logger.warning("  RAG search failed: %s", e)
            stats["errors"] += 1
            continue

        stats["retrieved"] += len(results)

        # クエリの埋め込みを事前計算（ドメイン関連性チェック用）
        query_vec = embedder.embed(query)

        for result in results:
            content = getattr(result, "content", "")
            source = getattr(result, "source", "")
            min_len = 300 if source == "tavily" else 50
            if not content or len(content.strip()) < min_len:
                stats["quality_filtered"] += 1
                continue

            # ── ドメイン関連性チェック（cosine similarity）──
            content_vec = embedder.embed(content[:1000])  # 先頭1000文字で計算
            similarity = float(np.dot(query_vec, content_vec))
            if similarity < relevance_threshold:
                logger.debug(
                    "  Irrelevant (sim=%.3f < %.2f): %s",
                    similarity, relevance_threshold, content[:60],
                )
                stats["irrelevant"] += 1
                continue

            # ── 最小品質フィルタ（configurable, default OFF）──
            if quality_filter_enabled:
                if not _has_code(content) and len(content.strip()) < quality_min_length:
                    logger.debug(
                        "  Quality filtered (no code, len=%d < %d): %s",
                        len(content.strip()), quality_min_length, content[:60],
                    )
                    stats["quality_filtered"] += 1
                    continue

            # ── ブラックリストチェック ──
            result_url = getattr(result, "url", "") or ""
            result_title = getattr(result, "title", "") or ""
            if await mm.store.is_blacklisted(source_url=result_url, source_title=result_title):
                logger.debug("  Blacklisted: %s", result_url or result_title)
                stats["duplicates"] += 1
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
                        extra=getattr(result, "metadata", {}) or {},
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
            "  Retrieved %d, added %d new (irrelevant=%d, total: %d)",
            len(results), stats["added"], stats["irrelevant"], len(new_doc_ids),
        )

    logger.info(
        "Phase 1 done: retrieved=%d, irrelevant=%d, quality_filtered=%d, duplicates=%d, added=%d",
        stats["retrieved"], stats["irrelevant"], stats["quality_filtered"],
        stats["duplicates"], stats["added"],
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
    reviewer = MemoryReviewer(gateway, mm.store, provider=provider, model=model)
    tagger = DifficultyTagger(gateway, provider=provider, model=model)

    for i, doc_id in enumerate(new_doc_ids):
        # UTC リセット前の停止チェック（OpenRouter のみ）
        action = _utc_reset_action(provider or "")
        if action == "stop":
            logger.warning("OpenRouter UTC リセット直前 (23:55+) — mature を中断します [%d/%d]", i, len(new_doc_ids))
            break
        if action == "wait":
            logger.info("OpenRouter UTC リセット待機 (23:50〜23:55) — 00:00 まで一時停止 [%d/%d]", i, len(new_doc_ids))
            await _wait_for_utc_reset()

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
            elif result.needs_supplement:
                stats.setdefault("needs_supplement", 0)
                stats["needs_supplement"] += 1
            if result.needs_supplement:
                status = "HOLD"
            elif result.approved:
                status = "PASS"
            else:
                status = "FAIL"
            logger.info(
                "  [%d/%d] Review %s (%.1fs, quality=%.2f): %s",
                i + 1, len(new_doc_ids), status, elapsed,
                result.quality_score, doc.content[:60],
            )
        except Exception as e:
            logger.warning("  [%d/%d] Review error: %s", i + 1, len(new_doc_ids), e)
            stats["errors"] += 1
            continue

        # 難易度タグ (HOLD はスキップ)
        if result.approved:
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
    limit: int, domain: str | None, provider: str, model: str | None = None,
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

    reviewer = MemoryReviewer(gateway, mm.store, provider=provider, model=model)
    tagger = DifficultyTagger(gateway, provider=provider, model=model)

    docs = await mm.store.get_unreviewed(domain=domain, limit=limit)
    if not docs:
        logger.info("No unreviewed documents found")
        await mm.close()
        return

    logger.info("=== Mature %d existing docs (provider=%s) ===", len(docs), provider)

    reviewed = 0
    approved = 0
    needs_supplement = 0
    tagged = 0

    for i, doc in enumerate(docs):
        # UTC リセット前の停止チェック（OpenRouter のみ）
        action = _utc_reset_action(provider)
        if action == "stop":
            logger.warning("OpenRouter UTC リセット直前 (23:55+) — mature を中断します [%d/%d]", i, len(docs))
            break
        if action == "wait":
            logger.info("OpenRouter UTC リセット待機 (23:50〜23:55) — 00:00 まで一時停止 [%d/%d]", i, len(docs))
            await _wait_for_utc_reset()

        # 審査
        try:
            start = time.monotonic()
            result = await reviewer.review(doc)
            elapsed = time.monotonic() - start
            reviewed += 1
            if result.approved:
                approved += 1
            elif result.needs_supplement:
                needs_supplement += 1
            if result.needs_supplement:
                status = "HOLD"
            elif result.approved:
                status = "PASS"
            else:
                status = "FAIL"
            logger.info(
                "  [%d/%d] Review %s (%.1fs, quality=%.2f): %s",
                i + 1, len(docs), status, elapsed,
                result.quality_score, doc.content[:60],
            )
        except Exception as e:
            logger.warning("  [%d/%d] Review error: %s", i + 1, len(docs), e)
            continue

        # 難易度タグ (HOLD はスキップ)
        if result.approved and doc.difficulty is None:
            try:
                level = await tagger.tag(doc)
                await mm.store.update_quality(doc.id, difficulty=level.value)
                tagged += 1
                logger.info("    Difficulty: %s", level.value)
            except Exception as e:
                logger.warning("    Tagging error: %s", e)

    logger.info(
        "Done: reviewed=%d, approved=%d, needs_supplement=%d, tagged=%d",
        reviewed, approved, needs_supplement, tagged,
    )
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
    parser.add_argument("--provider", type=str, default=None, help="Teacher provider (e.g. lmstudio, anthropic)")
    parser.add_argument("--model", type=str, default=None, help="Model override (e.g. claude-haiku-4-5-20251001)")

    # 制御
    parser.add_argument("--dry-run", action="store_true", help="Preview queries without execution")
    parser.add_argument("--no-mature", action="store_true", help="Skip Teacher maturation")
    parser.add_argument("--mature-only", action="store_true", help="Only mature existing unreviewed docs")
    parser.add_argument("--limit", type=int, default=100, help="Max docs for --mature-only")
    parser.add_argument("--dedup-threshold", type=float, default=0.95, help="Near-dup cosine threshold")
    parser.add_argument("--exclude-sources", type=str, default="", help="除外するソース (カンマ区切り, e.g. tavily,github)")

    args = parser.parse_args()

    # --mature-only モード
    if args.mature_only:
        if not args.provider:
            parser.error("--mature-only requires --provider")
        asyncio.run(mature_only(args.limit, args.domain, args.provider, model=args.model))
        return

    questions = load_questions(args)

    exclude_sources = [s.strip() for s in args.exclude_sources.split(",") if s.strip()]
    if exclude_sources:
        logger.info("Excluding sources: %s", exclude_sources)

    stats = asyncio.run(seed_and_mature(
        queries=questions,
        domain=args.domain,
        top_k=args.top_k,
        provider=args.provider,
        dry_run=args.dry_run,
        skip_mature=args.no_mature,
        dedup_threshold=args.dedup_threshold,
        model=args.model,
        exclude_sources=exclude_sources or None,
    ))

    # サマリー
    print(f"\n{'='*50}")
    print(f"  SUMMARY")
    print(f"{'='*50}")
    for k, v in stats.items():
        print(f"  {k:>12}: {v}")


if __name__ == "__main__":
    main()
