#!/usr/bin/env python3
"""scripts/seed_from_docs.py — GitHub ドキュメントリポジトリ / URL リストから FAISS に投入

github_docs ソースの唯一の投入経路。chunk_markdown パイプライン（見出し単位分割）を使用する。
seed_and_mature.py / seed_only.py は github_docs を扱わない。

外部 RAG（arXiv/SO）ではカバーできない公式ドキュメント・教科書を取得し、
FAISS メモリに追加する。オプションで Teacher による品質審査も実行する。

使い方:
    # GitHub ドキュメントリポジトリから取得（github_doc_repos.yaml）
    poetry run python scripts/seed_from_docs.py --source github_docs --max-files 100

    # URL リストから取得（data/doc_urls/*.txt）
    poetry run python scripts/seed_from_docs.py --source url_list

    # 特定ファイル指定
    poetry run python scripts/seed_from_docs.py --source url_list \\
        --url-file data/doc_urls/archwiki.txt --limit 20

    # 取得後に Teacher で成熟
    poetry run python scripts/seed_from_docs.py --source github_docs \\
        --mature --provider openrouter --max-files 50

    # dry-run（投入せず件数確認のみ）
    poetry run python scripts/seed_from_docs.py --source github_docs --dry-run
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


async def seed_from_docs(
    source: str,
    url_file: Path | None,
    github_config: Path | None,
    max_files: int,
    limit: int | None,
    domain: str,
    dry_run: bool,
    mature: bool,
    provider: str | None,
    model: str | None,
) -> None:
    from src.memory.embedder import Embedder
    from src.memory.memory_manager import MemoryManager
    from src.rag.chunker import Chunker
    from src.rag.retriever import RawResult

    embedder = Embedder()
    mm = MemoryManager(embedder=embedder)
    await mm.initialize()

    chunker = Chunker(chunk_size=1500, chunk_overlap=100, min_chunk_len=100)

    # ── Phase 1: フェッチ ──────────────────────────────────────────────
    raw_results: list[RawResult] = []

    if source in ("github_docs", "all"):
        from src.rag.github_docs_fetcher import GitHubDocsFetcher
        fetcher = GitHubDocsFetcher()
        if not fetcher.is_available():
            logger.error("GITHUB_TOKEN not set. Cannot fetch GitHub docs.")
        else:
            cfg_path = github_config or (_ROOT / "data" / "doc_urls" / "github_doc_repos.yaml")
            results = await fetcher.fetch_all(cfg_path, max_files_per_repo=max_files)
            raw_results.extend(results)
            logger.info("GitHub docs: %d raw results", len(results))

    if source in ("url_list", "all"):
        from src.rag.url_list_fetcher import UrlListFetcher
        fetcher_url = UrlListFetcher()
        if url_file:
            results = await fetcher_url.fetch_from_file(url_file, limit=limit)
        else:
            results = await fetcher_url.fetch_all_files(
                _ROOT / "data" / "doc_urls",
                limit_per_file=limit,
            )
        raw_results.extend(results)
        logger.info("URL list: %d raw results", len(results))

    if not raw_results:
        logger.warning("No results fetched. Exiting.")
        await mm.close()
        return

    logger.info("Total raw results: %d", len(raw_results))

    # ── Phase 1.5: URL 単位の既取得フィルタ ──────────────────────────
    # DB に登録済みの source_url を一括取得し、該当する RawResult をブロック単位でスキップ。
    # URL が一致すれば全チャンクをスキップできるため、content_hash 比較より大幅に効率的。
    # URL が空の結果はスキップできないため通常処理に回す。
    seeded_urls: set[str] = await mm.store.get_seeded_urls()
    logger.info("Loaded %d seeded URLs for pre-filter", len(seeded_urls))

    total_raw = len(raw_results)
    filtered_results: list[RawResult] = []
    url_skipped = 0
    for r in raw_results:
        r_url = (r.url or "").strip()
        if r_url and r_url in seeded_urls:
            url_skipped += 1
        else:
            filtered_results.append(r)
    if url_skipped:
        logger.info("URL pre-filter: %d / %d results skipped (already seeded)", url_skipped, len(raw_results))
    raw_results = filtered_results

    if not raw_results:
        logger.info("All results already seeded. Exiting.")
        await mm.close()
        return

    # ── Phase 2: チャンク化 ───────────────────────────────────────────
    from src.memory.schema import Document
    docs: list[Document] = chunker.chunk_results(raw_results, domain=domain)
    logger.info("After chunking: %d documents", len(docs))

    if dry_run:
        logger.info("[DRY RUN] Would add %d documents. Exiting.", len(docs))
        await mm.close()
        return

    # ── Phase 3: 重複排除 + FAISS 投入 ───────────────────────────────
    from src.memory.deduplicator import Deduplicator

    dedup = Deduplicator(near_dup_threshold=0.95)

    # 既存 content_hash を一括ロード（チャンク虫食い防止）
    existing_hashes: dict[str, str] = {}
    try:
        async with mm.store._db.execute(
            "SELECT id, content_hash FROM documents WHERE content_hash IS NOT NULL"
        ) as cur:
            async for row in cur:
                existing_hashes[row[0]] = row[1]
        logger.info("Loaded %d existing content hashes for dedup", len(existing_hashes))
    except Exception:
        logger.warning("Could not load existing hashes; dedup will rely on blacklist only")

    added = 0
    duplicates = 0
    errors = 0

    for i, doc in enumerate(docs):
        try:
            content_hash = dedup.content_hash(doc.content)
            dup_result = dedup.check(
                content_hash=content_hash,
                existing_hashes=existing_hashes,
            )
            if dup_result.is_duplicate:
                duplicates += 1
                continue

            doc = doc.model_copy(update={"content_hash": content_hash})
            doc_id = await mm.add(doc)
            if doc_id:
                added += 1
                existing_hashes[doc_id] = content_hash
            else:
                duplicates += 1
        except Exception as e:
            logger.warning("Add failed [%d]: %s", i, e)
            errors += 1

        if (i + 1) % 50 == 0:
            logger.info("  [%d/%d] added=%d dup=%d err=%d",
                        i + 1, len(docs), added, duplicates, errors)

    logger.info("Phase 1 done: added=%d, duplicates=%d, errors=%d", added, duplicates, errors)

    # ── Phase 4: 品質審査（オプション） ──────────────────────────────
    if mature and added > 0:
        logger.info("Starting maturation (provider=%s)...", provider)
        from src.llm.gateway import LLMGateway
        from src.memory.maturation.reviewer import MemoryReviewer
        from src.memory.maturation.difficulty_tagger import DifficultyTagger

        gw = LLMGateway()
        reviewer = MemoryReviewer(gateway=gw, store=mm.store, provider=provider, model=model)
        tagger = DifficultyTagger(gateway=gw, provider=provider, model=model)

        # 未審査ドキュメントを取得
        unreviewed = await mm.store.get_unreviewed(limit=added + 50)
        logger.info("Unreviewed docs to mature: %d", len(unreviewed))

        approved = rejected = tagged = 0
        for i, doc in enumerate(unreviewed):
            try:
                result = await reviewer.review(doc)
                if result.approved:
                    approved += 1
                    await tagger.tag(doc)
                    tagged += 1
                else:
                    rejected += 1
            except Exception as e:
                logger.warning("Review failed [%s]: %s", doc.id[:8], e)

            if (i + 1) % 20 == 0:
                logger.info("  [%d/%d] approved=%d rejected=%d",
                            i + 1, len(unreviewed), approved, rejected)

        logger.info("Maturation done: approved=%d, rejected=%d, tagged=%d",
                    approved, rejected, tagged)

    await mm.close()

    # ── サマリー ─────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  SUMMARY")
    print("=" * 50)
    print(f"    raw_results: {total_raw}")
    print(f"    url_skipped: {url_skipped}  (already seeded)")
    print(f"    to_chunk:    {len(raw_results)}")
    print(f"    chunked:     {len(docs)}")
    print(f"    added:       {added}")
    print(f"    duplicates:  {duplicates}  (same content, different URL)")
    print(f"    errors:      {errors}")
    if mature:
        print(f"    mature:      yes (provider={provider})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed FAISS from GitHub docs or URL lists")
    parser.add_argument(
        "--source",
        choices=["github_docs", "url_list", "all"],
        default="github_docs",
        help="取得ソース (default: github_docs)",
    )
    parser.add_argument(
        "--url-file",
        type=Path,
        default=None,
        help="URL リストファイルのパス（url_list 時に使用）",
    )
    parser.add_argument(
        "--github-config",
        type=Path,
        default=None,
        help="github_doc_repos.yaml のパス",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=100,
        help="リポジトリあたりの最大取得ファイル数 (default: 100)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="URL リストの最大取得件数",
    )
    parser.add_argument(
        "--domain",
        default="code",
        help="FAISS ドメイン (default: code)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="フェッチのみ、FAISS 投入をスキップ",
    )
    parser.add_argument(
        "--mature",
        action="store_true",
        help="投入後に Teacher で品質審査を実行",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="LLM プロバイダー (mature 時に使用)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM モデル名 (mature 時に使用)",
    )
    args = parser.parse_args()

    start = time.time()
    asyncio.run(seed_from_docs(
        source=args.source,
        url_file=args.url_file,
        github_config=args.github_config,
        max_files=args.max_files,
        limit=args.limit,
        domain=args.domain,
        dry_run=args.dry_run,
        mature=args.mature,
        provider=args.provider,
        model=args.model,
    ))
    elapsed = time.time() - start
    print(f"\nElapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
