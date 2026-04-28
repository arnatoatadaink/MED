#!/usr/bin/env python3
"""scripts/rebuild_github_docs.py — needs_update の github_docs を再チャンク化して再投入

処理フロー:
  1. DB から needs_update & github_docs の distinct source_url を収集
  2. URL から repo / ref / path を抽出
  3. GitHub API で再フェッチ（1 req/sec レート制限）
  4. chunk_markdown で再チャンク
  5. 旧チャンク（同 source_url）を DB + FAISS からバッチ削除
  6. 新チャンクを unreviewed で投入

使い方:
    # 全対象を処理
    poetry run python scripts/rebuild_github_docs.py

    # 最初の10 URL のみ（動作確認）
    poetry run python scripts/rebuild_github_docs.py --limit 10

    # dry-run（フェッチのみ、DB/FAISS を変更しない）
    poetry run python scripts/rebuild_github_docs.py --dry-run --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
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

_GITHUB_BLOB_RE = re.compile(
    r"https://github\.com/(?P<repo>[^/]+/[^/]+)/blob/(?P<ref>[^/]+)/(?P<path>.+)"
)


def _parse_github_url(url: str) -> tuple[str, str, str] | None:
    """GitHub blob URL から (repo, ref, path) を抽出する。

    Returns:
        (repo, ref, path) または解析不能な場合 None。
    """
    m = _GITHUB_BLOB_RE.match(url)
    if not m:
        return None
    return m.group("repo"), m.group("ref"), m.group("path")


async def rebuild_github_docs(
    limit: int | None,
    dry_run: bool,
    status_filter: str,
) -> None:
    from src.memory.embedder import Embedder
    from src.memory.memory_manager import MemoryManager
    from src.rag.chunker import Chunker
    from src.rag.github_docs_fetcher import GitHubDocsFetcher
    from src.rag.retriever import RawResult

    embedder = Embedder()
    mm = MemoryManager(embedder=embedder)
    await mm.initialize()

    chunker = Chunker(chunk_size=1500, chunk_overlap=100, min_chunk_len=100)
    fetcher = GitHubDocsFetcher()

    if not fetcher.is_available():
        logger.error("GITHUB_TOKEN not set. Cannot fetch GitHub docs.")
        await mm.close()
        return

    # ── 対象 URL を収集 ─────────────────────────────────────────────────
    query = f"""
        SELECT source_url,
               COUNT(*) as chunk_count,
               MIN(domain) as domain
        FROM documents
        WHERE source_tags LIKE '%github_docs%'
          AND review_status = ?
          AND source_url IS NOT NULL AND source_url != ''
        GROUP BY source_url
        ORDER BY chunk_count DESC
    """
    if limit:
        query += f" LIMIT {limit}"

    cur = await mm.store._db.execute(query, (status_filter,))
    rows = await cur.fetchall()
    logger.info("Target URLs: %d (status=%s)", len(rows), status_filter)

    if not rows:
        logger.info("No target URLs found. Exiting.")
        await mm.close()
        return

    # ── URL ごとに再フェッチ → 削除 → 再投入 ──────────────────────────
    rebuilt = 0
    skipped = 0
    failed = 0
    total_deleted = 0
    total_added = 0

    for i, row in enumerate(rows):
        source_url: str = row[0]
        old_chunk_count: int = row[1]
        domain: str = row[2] or "code"

        parsed = _parse_github_url(source_url)
        if parsed is None:
            logger.warning("[%d/%d] Cannot parse URL: %s", i + 1, len(rows), source_url)
            skipped += 1
            continue

        repo, ref, file_path = parsed
        ext = Path(file_path).suffix.lower()

        # ── GitHub API から再フェッチ ──────────────────────────────────
        content = await fetcher._fetch_file(repo, file_path, ref)
        if content is None:
            logger.warning("[%d/%d] Fetch failed: %s", i + 1, len(rows), source_url)
            failed += 1
            continue

        # ── クリーナー適用（nodejs_api は専用プロファイル）──────────────
        if repo == "nodejs/node" and ext == ".md":
            doc_meta = fetcher._extract_nodejs_meta(content)
            clean = fetcher._clean_nodejs_markdown(content)
        else:
            doc_meta = {}
            clean = fetcher._clean_content(content, ext)

        if len(clean) < 100:
            logger.debug("[%d/%d] Content too short after cleaning: %s", i + 1, len(rows), source_url)
            skipped += 1
            continue

        # ── RawResult → chunk_markdown ─────────────────────────────────
        filename = Path(file_path).stem
        label = repo
        raw = RawResult(
            title=f"{label}: {filename}",
            content=clean,
            url=source_url,
            source="github_docs",
            score=0.9,
            metadata={
                "repo": repo,
                "file_path": file_path,
                "ref": ref,
                "format": "markdown" if ext == ".md" else "rst",
                "label": label,
                "content_type": "documentation",
                **doc_meta,
            },
        )
        new_docs = chunker.chunk_result(raw, domain=domain)

        if not new_docs:
            logger.warning("[%d/%d] No chunks after re-chunking: %s", i + 1, len(rows), source_url)
            skipped += 1
            continue

        if dry_run:
            logger.info(
                "[DRY RUN] %d/%d  old=%d→new=%d  %s",
                i + 1, len(rows), old_chunk_count, len(new_docs), source_url[-70:],
            )
            rebuilt += 1
            continue

        # ── 旧チャンクを DB + FAISS からバッチ削除 ─────────────────────
        cur_old = await mm.store._db.execute(
            "SELECT id, domain FROM documents WHERE source_url = ?",
            (source_url,),
        )
        old_rows = await cur_old.fetchall()
        old_ids = [r[0] for r in old_rows]
        old_domain = old_rows[0][1] if old_rows else domain

        deleted_db = await mm.store.delete_batch(old_ids)
        deleted_faiss = mm.faiss.remove(old_domain, old_ids)
        total_deleted += deleted_db

        # ── 新チャンクをバッチ投入（embed_batch + FAISS 保存なし）──────
        added_ids = await mm.add_batch(new_docs)
        added = len(added_ids)
        total_added += added

        logger.info(
            "[%d/%d] deleted=%d(faiss=%d) added=%d  %s",
            i + 1, len(rows),
            deleted_db, deleted_faiss, added,
            source_url[-70:],
        )
        rebuilt += 1

        # 20 URL ごとに FAISS を中間保存（クラッシュ対策）
        if (i + 1) % 20 == 0:
            await asyncio.get_event_loop().run_in_executor(None, mm.faiss.save)
            logger.info(
                "  Progress: rebuilt=%d skipped=%d failed=%d  "
                "total_deleted=%d total_added=%d  [FAISS saved]",
                rebuilt, skipped, failed, total_deleted, total_added,
            )

    await mm.close()

    print("\n" + "=" * 50)
    print("  REBUILD SUMMARY")
    print("=" * 50)
    print(f"    target URLs:   {len(rows)}")
    print(f"    rebuilt:       {rebuilt}")
    print(f"    skipped:       {skipped}")
    print(f"    failed:        {failed}")
    if not dry_run:
        print(f"    total_deleted: {total_deleted}")
        print(f"    total_added:   {total_added}")
    else:
        print("    (dry-run: no changes made)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild needs_update github_docs with chunk_markdown"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="処理する最大 URL 数（省略時: 全件）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="フェッチのみ実行し、DB/FAISS を変更しない",
    )
    parser.add_argument(
        "--status",
        default="needs_update",
        help="対象 review_status (default: needs_update)",
    )
    args = parser.parse_args()

    start = time.time()
    asyncio.run(rebuild_github_docs(
        limit=args.limit,
        dry_run=args.dry_run,
        status_filter=args.status,
    ))
    elapsed = time.time() - start
    print(f"\nElapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
