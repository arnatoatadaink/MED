#!/usr/bin/env python3
"""scripts/seed_arxiv_ids.py — arXiv ID 指定で論文アブストラクトを FAISS に投入

ArXiv API の id_list パラメータを使い、特定の論文を精確に取得する。
クエリ検索ではなく ID 直接指定のため、関連性の高い論文だけを確実に収録できる。

使い方:
    # デフォルト ID ファイル（data/doc_urls/med_papers.txt）から投入
    poetry run python scripts/seed_arxiv_ids.py

    # 別ファイルを指定
    poetry run python scripts/seed_arxiv_ids.py --ids-file data/doc_urls/med_papers.txt

    # dry-run（取得のみ、FAISS 変更なし）
    poetry run python scripts/seed_arxiv_ids.py --dry-run

    # academic ドメインに投入（デフォルト）
    poetry run python scripts/seed_arxiv_ids.py --domain academic
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

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

_ARXIV_API = "https://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom"}
_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?)")

_DEFAULT_IDS_FILE = _ROOT / "data" / "doc_urls" / "med_papers.txt"
_BATCH_SIZE = 10      # ArXiv API の推奨バッチサイズ
_RATE_INTERVAL = 5.0  # ArXiv 利用規約: 3秒 + 余裕


# ── テキスト正規化 ────────────────────────────────────────────────────────────

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+\{([^}]*)\}")
_LATEX_MATH_RE = re.compile(r"\$\$?[^$]+\$\$?")


def _clean(text: str) -> str:
    text = _HTML_TAG_RE.sub(" ", text)
    text = _LATEX_CMD_RE.sub(r"\1", text)
    text = _LATEX_MATH_RE.sub("[MATH]", text)
    return re.sub(r"\s+", " ", text).strip()


# ── ID ファイル読み込み ───────────────────────────────────────────────────────

def load_arxiv_ids(path: Path) -> list[str]:
    """コメント（#）・空行を除いて arXiv ID を抽出する。"""
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        m = _ARXIV_ID_RE.search(line)
        if m:
            ids.append(m.group(1))
        else:
            logger.warning("Unrecognized line (no arXiv ID found): %r", line)
    return ids


# ── ArXiv API 取得 ───────────────────────────────────────────────────────────

async def fetch_by_ids(
    ids: list[str],
    client: httpx.AsyncClient,
) -> list[dict]:
    """id_list を使って ArXiv から論文メタデータを取得する。"""
    params = {
        "id_list": ",".join(ids),
        "max_results": len(ids),
    }
    resp = await client.get(_ARXIV_API, params=params, timeout=30.0)
    resp.raise_for_status()

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as e:
        logger.error("XML parse error: %s", e)
        return []

    papers: list[dict] = []
    for entry in root.findall("atom:entry", _NS):
        title_el = entry.find("atom:title", _NS)
        summary_el = entry.find("atom:summary", _NS)
        id_el = entry.find("atom:id", _NS)
        authors_els = entry.findall("atom:author", _NS)
        published_el = entry.find("atom:published", _NS)
        categories = [c.get("term", "") for c in entry.findall("atom:category", _NS)]

        url = (id_el.text or "").strip() if id_el is not None else ""
        arxiv_id = _ARXIV_ID_RE.search(url)
        clean_id = re.sub(r"v\d+$", "", arxiv_id.group(1)) if arxiv_id else ""

        authors = [
            (a.find("atom:name", _NS).text or "").strip()
            for a in authors_els
            if a.find("atom:name", _NS) is not None
        ]

        papers.append({
            "id": clean_id,
            "title": _clean(title_el.text or "") if title_el is not None else "",
            "abstract": _clean(summary_el.text or "") if summary_el is not None else "",
            "url": url,
            "authors": authors,
            "published": (published_el.text or "").strip() if published_el is not None else "",
            "categories": categories,
        })

    return papers


# ── メイン ───────────────────────────────────────────────────────────────────

async def seed_arxiv_ids(
    ids: list[str],
    domain: str,
    dry_run: bool,
) -> dict[str, int]:
    from src.memory.embedder import Embedder
    from src.memory.memory_manager import MemoryManager
    from src.memory.schema import Document

    stats = {"fetched": 0, "added": 0, "duplicate": 0, "error": 0}

    async with httpx.AsyncClient() as client:
        # バッチ単位で取得
        all_papers: list[dict] = []
        for i in range(0, len(ids), _BATCH_SIZE):
            batch = ids[i : i + _BATCH_SIZE]
            logger.info(
                "Fetching batch [%d-%d] / %d: %s",
                i + 1, i + len(batch), len(ids), ", ".join(batch),
            )
            try:
                papers = await fetch_by_ids(batch, client)
                all_papers.extend(papers)
                stats["fetched"] += len(papers)
                logger.info("  → %d papers retrieved", len(papers))
            except Exception as e:
                logger.error("Batch fetch failed: %s", e)
                stats["error"] += len(batch)

            if i + _BATCH_SIZE < len(ids):
                logger.info("  Rate limit: sleeping %.1fs ...", _RATE_INTERVAL)
                await asyncio.sleep(_RATE_INTERVAL)

    if dry_run:
        logger.info("[DRY RUN] Would add %d papers. Exiting.", len(all_papers))
        for p in all_papers:
            logger.info("  %s — %s", p["id"], p["title"][:70])
        stats["fetched"] = len(all_papers)
        return stats

    if not all_papers:
        logger.warning("No papers fetched.")
        return stats

    # FAISS 投入
    from src.memory.schema import Domain, SourceMeta, SourceType

    try:
        domain_enum = Domain(domain)
    except ValueError:
        logger.error("Unknown domain '%s'. Choose from: %s", domain, [d.value for d in Domain])
        return stats

    embedder = Embedder()
    mm = MemoryManager(embedder=embedder)
    await mm.initialize()

    try:
        seeded_urls = await mm.store.get_seeded_urls()
        logger.info("Already seeded URLs: %d", len(seeded_urls))

        for paper in all_papers:
            url = paper["url"]
            if url in seeded_urls:
                logger.info("  [skip] already seeded: %s", paper["id"])
                stats["duplicate"] += 1
                continue

            content = (
                f"Title: {paper['title']}\n\n"
                f"Authors: {', '.join(paper['authors'][:5])}\n"
                f"Published: {paper['published'][:10]}\n"
                f"ArXiv: {url}\n"
                f"Categories: {', '.join(paper['categories'])}\n\n"
                f"Abstract:\n{paper['abstract']}"
            )

            doc = Document(
                content=content,
                domain=domain_enum,
                source=SourceMeta(
                    source_type=SourceType.ARXIV,
                    url=url,
                    title=paper["title"],
                    tags=paper["categories"],
                ),
            )

            try:
                doc_id = await mm.add(doc)
                if doc_id:
                    logger.info("  [added] %s — %s", paper["id"], paper["title"][:60])
                    stats["added"] += 1
                else:
                    logger.info("  [dup]   %s — %s", paper["id"], paper["title"][:60])
                    stats["duplicate"] += 1
            except Exception as e:
                logger.warning("  [error] %s: %s", paper["id"], e)
                stats["error"] += 1
    finally:
        await mm.close()

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="arXiv ID 指定で論文アブストラクトを FAISS に投入",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        default=_DEFAULT_IDS_FILE,
        help=f"arXiv ID ファイル (default: {_DEFAULT_IDS_FILE.relative_to(_ROOT)})",
    )
    parser.add_argument(
        "--domain",
        default="academic",
        help="FAISS ドメイン (default: academic)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="取得のみ（FAISS 変更なし）",
    )
    args = parser.parse_args()

    if not args.ids_file.exists():
        logger.error("IDs file not found: %s", args.ids_file)
        sys.exit(1)

    ids = load_arxiv_ids(args.ids_file)
    if not ids:
        logger.error("No arXiv IDs found in %s", args.ids_file)
        sys.exit(1)

    logger.info("Loaded %d arXiv IDs from %s", len(ids), args.ids_file)

    stats = asyncio.run(seed_arxiv_ids(ids, domain=args.domain, dry_run=args.dry_run))

    print(f"\n{'='*52}")
    print(f"  SUMMARY  [arxiv → domain={args.domain}]")
    print(f"{'='*52}")
    print(f"  IDs loaded:  {len(ids)}")
    print(f"  Fetched:     {stats['fetched']}")
    print(f"  Added:       {stats['added']}")
    print(f"  Duplicate:   {stats['duplicate']}")
    print(f"  Errors:      {stats['error']}")
    print(f"{'='*52}")


if __name__ == "__main__":
    main()
