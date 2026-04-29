#!/usr/bin/env python3
"""scripts/eval_faiss_k.py — FAISS k値比較実験（MRR / Recall@k）

k=3/5/7/10 それぞれで検索し、MRR と Recall@k を計測する。
クエリは DB の approved ドキュメントからサンプリングし、
「同じ source_url を持つ他チャンク」を正解とする弱い関連性評価を使う。

使い方:
    poetry run python scripts/eval_faiss_k.py
    poetry run python scripts/eval_faiss_k.py --k-values 3 5 10 --n-queries 100
    poetry run python scripts/eval_faiss_k.py --domain code --n-queries 200
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def _load_eval_queries(
    db_path: str,
    domain: str | None,
    n_queries: int,
    seed: int = 42,
) -> list[dict]:
    """DB から評価クエリを構築する。

    approved かつ同一 source_url に複数チャンクが存在するドキュメントを選ぶ。
    クエリ = content の先頭 200 文字、正解 = 同一 source_url の他チャンク ID。

    Returns:
        [{"query": str, "doc_id": str, "relevant_ids": set[str]}, ...]
    """
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row

    domain_filter = "AND domain = ?" if domain else ""
    params: list = [domain] if domain else []

    # 同一 source_url に複数チャンクがある approved 文書を取得
    cur = conn.execute(
        f"""
        SELECT id, content, source_url, domain
        FROM documents
        WHERE review_status = 'approved'
          AND source_url IS NOT NULL
          AND source_url != ''
          {domain_filter}
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (*params, n_queries * 10),
    )
    rows = [dict(r) for r in cur.fetchall()]

    # source_url ごとにグループ化
    url_groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        url_groups[r["source_url"]].append(r)

    # 複数チャンクあるグループのみ使用
    multi_chunk_urls = [url for url, docs in url_groups.items() if len(docs) >= 2]
    random.seed(seed)
    random.shuffle(multi_chunk_urls)

    queries = []
    for url in multi_chunk_urls[:n_queries]:
        docs = url_groups[url]
        random.shuffle(docs)
        anchor = docs[0]
        relevant_ids = {d["id"] for d in docs if d["id"] != anchor["id"]}
        queries.append({
            "query": anchor["content"][:200],
            "doc_id": anchor["id"],
            "relevant_ids": relevant_ids,
            "domain": anchor["domain"],
        })

    conn.close()
    logger.info("Loaded %d eval queries (from %d URLs)", len(queries), len(multi_chunk_urls))
    return queries[:n_queries]


async def _run_eval(
    queries: list[dict],
    k_values: list[int],
    domain: str | None,
) -> dict[int, dict[str, float]]:
    """各 k 値で MRR と Recall@k を計算する。"""
    from src.memory.memory_manager import MemoryManager

    mm = MemoryManager()
    await mm.initialize()

    results: dict[int, dict[str, float]] = {}

    try:
        for k in k_values:
            reciprocal_ranks: list[float] = []
            recalls: list[float] = []

            for q in queries:
                search_domain = domain or q.get("domain")
                hits = await mm.search(q["query"], domain=search_domain, k=k)

                hit_ids = [h.document.id for h in hits]
                relevant = q["relevant_ids"]

                # 自分自身を除く（anchor doc が検索結果に入る場合）
                hit_ids_clean = [hid for hid in hit_ids if hid != q["doc_id"]]

                # MRR: 最初に relevant が現れた順位の逆数
                rr = 0.0
                for rank, hid in enumerate(hit_ids_clean, 1):
                    if hid in relevant:
                        rr = 1.0 / rank
                        break
                reciprocal_ranks.append(rr)

                # Recall@k: relevant の何割が top-k に入ったか
                found = sum(1 for hid in hit_ids_clean if hid in relevant)
                recalls.append(found / max(len(relevant), 1))

            mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
            recall = sum(recalls) / len(recalls) if recalls else 0.0
            results[k] = {"mrr": mrr, f"recall@{k}": recall}

    finally:
        await mm.close()

    return results


def _print_table(results: dict[int, dict[str, float]], n_queries: int) -> None:
    """結果を表形式で出力する。"""
    print(f"\nFAISS k値比較実験 (n_queries={n_queries})")
    print("=" * 50)
    print(f"{'k':>4}  {'MRR':>8}  {'Recall@k':>10}")
    print("-" * 50)
    for k in sorted(results):
        m = results[k]
        recall_key = f"recall@{k}"
        print(f"{k:>4}  {m['mrr']:>8.4f}  {m.get(recall_key, 0.0):>10.4f}")
    print("=" * 50)

    # 最適 k の推奨
    best_k = max(results, key=lambda k: results[k]["mrr"])
    print(f"\n推奨 k: {best_k}  (MRR={results[best_k]['mrr']:.4f})")
    print("設定: configs/default.yaml の rag.faiss_k を更新してください")


def main() -> None:
    parser = argparse.ArgumentParser(description="FAISS k値比較実験")
    parser.add_argument(
        "--k-values", nargs="+", type=int, default=[3, 5, 7, 10],
        help="比較する k 値リスト (default: 3 5 7 10)",
    )
    parser.add_argument(
        "--n-queries", type=int, default=100,
        help="評価クエリ数 (default: 100)",
    )
    parser.add_argument(
        "--domain", type=str, default=None,
        help="対象ドメイン: code / academic / general (default: 全ドメイン)",
    )
    parser.add_argument(
        "--db-path", type=str, default=str(_ROOT / "data" / "metadata.db"),
        help="metadata.db パス",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="乱数シード (default: 42)",
    )
    args = parser.parse_args()

    print(f"クエリ構築中 (n={args.n_queries}, domain={args.domain or 'all'})...")
    queries = _load_eval_queries(args.db_path, args.domain, args.n_queries, args.seed)
    if not queries:
        print("ERROR: 評価クエリが構築できませんでした。DB に approved ドキュメントが必要です。")
        sys.exit(1)
    print(f"クエリ数: {len(queries)}")

    print(f"検索中 (k={args.k_values})...")
    results = asyncio.run(_run_eval(queries, args.k_values, args.domain))

    _print_table(results, len(queries))


if __name__ == "__main__":
    main()
