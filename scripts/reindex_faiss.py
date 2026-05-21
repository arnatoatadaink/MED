"""scripts/reindex_faiss.py — approved 文書を SQLite から再読み込みして FAISS を再構築する。

SQLite metadata.db の approved 文書を全件 embed し直し、空になった FAISS インデックスを
再構築する。metadata.db の Document レコードは変更しない（ベクトル追加のみ）。

使い方:
    # 全件再インデックス
    poetry run python scripts/reindex_faiss.py

    # 動作確認 (最初の 50 件のみ、DB/FAISS を変更しない)
    poetry run python scripts/reindex_faiss.py --dry-run --limit 50

    # ドメインを指定して実行
    poetry run python scripts/reindex_faiss.py --domain code

    # バッチサイズを変更 (デフォルト 64)
    poetry run python scripts/reindex_faiss.py --batch-size 32
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


async def _fetch_approved_batch(
    db_path: str,
    domain: str | None,
    limit: int | None,
    offset: int,
    batch_size: int,
) -> list[dict]:
    """approved 文書を 1 バッチ分取得する。"""
    import aiosqlite

    where = "review_status = 'approved'"
    params: list = []
    if domain:
        where += " AND domain = ?"
        params.append(domain)

    sql = f"SELECT id, domain, content FROM documents WHERE {where} ORDER BY id LIMIT ? OFFSET ?"
    effective_limit = min(batch_size, limit - offset) if limit is not None else batch_size
    params.extend([effective_limit, offset])

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(sql, params)
        return [dict(r) for r in await cur.fetchall()]


async def _count_approved(db_path: str, domain: str | None) -> int:
    """対象 approved 件数を返す。"""
    import aiosqlite

    where = "review_status = 'approved'"
    params: list = []
    if domain:
        where += " AND domain = ?"
        params.append(domain)

    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(f"SELECT COUNT(*) FROM documents WHERE {where}", params)
        row = await cur.fetchone()
        return row[0] if row else 0


async def run(
    db_path: str,
    faiss_base_dir: str,
    domain: str | None,
    batch_size: int,
    limit: int | None,
    dry_run: bool,
) -> None:
    from src.memory.embedder import Embedder
    from src.memory.faiss_index import FAISSIndexManager

    total = await _count_approved(db_path, domain)
    effective_total = min(total, limit) if limit is not None else total
    log.info("対象 approved 文書: %d 件 (全体 %d 件)", effective_total, total)

    if dry_run:
        log.info("[DRY-RUN] DB/FAISS を変更しません。")

    log.info("Embedder 初期化中...")
    embedder = Embedder()
    log.info("Embedder 初期化完了 (dim=%d)", embedder.dim)

    from src.common.config import FAISSConfig
    faiss_cfg = FAISSConfig(base_dir=Path(faiss_base_dir))
    faiss_index = FAISSIndexManager(config=faiss_cfg)
    if not dry_run:
        try:
            faiss_index.load()
            log.info("既存 FAISS インデックス読み込み: %s", faiss_index.domain_stats())
        except Exception:
            log.info("既存インデックスなし — 空で開始")

    processed = 0
    added = 0
    skipped = 0
    offset = 0

    while True:
        batch = await _fetch_approved_batch(db_path, domain, limit, offset, batch_size)
        if not batch:
            break

        # content が空のものを除外
        valid = [r for r in batch if r["content"] and len(r["content"].strip()) >= 10]
        if len(valid) < len(batch):
            log.debug("空 content スキップ: %d 件", len(batch) - len(valid))
            skipped += len(batch) - len(valid)

        if valid:
            texts = [r["content"] for r in valid]
            try:
                embeddings = embedder.embed_batch(texts)
            except Exception as exc:
                log.warning("embed_batch 失敗 (offset=%d): %s", offset, exc)
                skipped += len(valid)
                offset += len(batch)
                continue

            # ドメイン別にグループ化して FAISS に追加
            by_domain: dict[str, tuple[list[str], list]] = {}
            for row, vec in zip(valid, embeddings):
                d = row["domain"]
                if d not in by_domain:
                    by_domain[d] = ([], [])
                by_domain[d][0].append(row["id"])
                by_domain[d][1].append(vec)

            if not dry_run:
                for d, (ids, vecs) in by_domain.items():
                    faiss_index.add(d, ids, np.vstack(vecs).astype(np.float32))

            added += len(valid)

        processed += len(batch)
        offset += len(batch)

        pct = processed / effective_total * 100 if effective_total > 0 else 0
        log.info(
            "進捗: %d/%d (%.1f%%)  added=%d  skipped=%d",
            processed, effective_total, pct, added, skipped,
        )

        if limit is not None and processed >= limit:
            break

    if not dry_run:
        log.info("FAISS インデックス保存中...")
        faiss_index.save()
        log.info("保存完了: %s", faiss_index.domain_stats())

    log.info("完了: added=%d  skipped=%d  total_processed=%d", added, skipped, processed)


def main() -> None:
    ap = argparse.ArgumentParser(description="FAISS インデックスを SQLite approved 文書から再構築する")
    ap.add_argument("--db-path", default="data/metadata.db")
    ap.add_argument("--faiss-base-dir", default="data/faiss_indices")
    ap.add_argument("--domain", choices=["code", "academic", "general"], default=None,
                    help="ドメインを絞り込む (省略時は全ドメイン)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="処理件数上限 (動作確認用)")
    ap.add_argument("--dry-run", action="store_true", help="DB/FAISS を変更しない")
    args = ap.parse_args()

    asyncio.run(run(
        db_path=args.db_path,
        faiss_base_dir=args.faiss_base_dir,
        domain=args.domain,
        batch_size=args.batch_size,
        limit=args.limit,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
