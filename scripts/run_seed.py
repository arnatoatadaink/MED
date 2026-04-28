#!/usr/bin/env python3
"""scripts/run_seed.py — Seed 管理スクリプト

事前チェック → Seed 実行 → 事後チェック を1コマンドで完結させる。
Claude による実行確認・完了待ち・ログチェックの繰り返しを削減する。

github_docs の取得は seed_from_docs.py 経由（chunk_markdown パイプライン）。

使い方:
    # 状態確認のみ
    poetry run python scripts/run_seed.py --check-only

    # GitHub docs を seed
    poetry run python scripts/run_seed.py --source github_docs

    # URL リストを seed
    poetry run python scripts/run_seed.py --source url_list

    # 全ソースを seed（github_docs + url_list）
    poetry run python scripts/run_seed.py --source all

    # seed 後に mature も実行
    poetry run python scripts/run_seed.py --source github_docs --mature --provider lmstudio

    # dry-run（取得確認のみ、DB/FAISS を変更しない）
    poetry run python scripts/run_seed.py --source github_docs --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import logging
import os
import sqlite3
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

_STATUS_ORDER = ["unreviewed", "approved", "needs_update", "hold", "rejected"]
_OPENROUTER_LIMIT = 950


# ── スナップショット取得 ──────────────────────────────────────────────────────

def _db_snapshot() -> dict[str, int]:
    """DB の review_status ごとの件数を返す。"""
    db_path = _ROOT / "data" / "metadata.db"
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT review_status, COUNT(*) FROM documents GROUP BY review_status"
    ).fetchall()
    conn.close()
    return dict(rows)


def _faiss_snapshot() -> dict[str, int]:
    """FAISS インデックスのベクター数を返す。domain → count"""
    try:
        import faiss
    except ImportError:
        return {}
    result: dict[str, int] = {}
    pattern = str(_ROOT / "data" / "faiss_indices" / "*")
    for idx_path in sorted(glob.glob(pattern)):
        domain = os.path.basename(idx_path)
        faiss_file = os.path.join(idx_path, "index.faiss")
        try:
            idx = faiss.read_index(faiss_file)
            result[domain] = idx.ntotal
        except Exception:
            result[domain] = -1
    return result


def _openrouter_snapshot() -> list[tuple[str, int]]:
    """OpenRouter の直近使用量を返す。[(date, requests), ...]"""
    db_path = _ROOT / "data" / "openrouter_usage.db"
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT date, total_requests FROM daily_usage ORDER BY date DESC LIMIT 3"
        ).fetchall()
        conn.close()
        return rows
    except Exception:
        return []


# ── 表示 ────────────────────────────────────────────────────────────────────

def _print_status(
    db: dict[str, int],
    faiss: dict[str, int],
    label: str = "STATUS",
) -> None:
    total = sum(db.values())
    print(f"\n{'─'*44}")
    print(f"  [{label}]")
    print(f"{'─'*44}")

    print("  DB Documents:")
    for status in _STATUS_ORDER:
        cnt = db.get(status, 0)
        if cnt > 0:
            bar = "█" * (cnt * 20 // max(total, 1))
            print(f"    {status:<15} {cnt:>6}  {bar}")
    other = {k: v for k, v in db.items() if k not in _STATUS_ORDER}
    for status, cnt in other.items():
        print(f"    {status:<15} {cnt:>6}")
    print(f"    {'TOTAL':<15} {total:>6}")

    if faiss:
        print("  FAISS:")
        for domain, cnt in faiss.items():
            if cnt >= 0:
                print(f"    {domain:<12} {cnt:>6} vectors")
            else:
                print(f"    {domain:<12}  (read error)")

    rows = _openrouter_snapshot()
    if rows:
        print(f"  OpenRouter (limit={_OPENROUTER_LIMIT}):")
        for date, req in rows:
            pct = req * 100 // _OPENROUTER_LIMIT
            bar = "█" * (pct * 16 // 100)
            print(f"    {date}  {req:>4}/{_OPENROUTER_LIMIT}  ({pct:>3}%)  {bar}")


def _print_diff(
    db_before: dict[str, int],
    db_after: dict[str, int],
    faiss_before: dict[str, int],
    faiss_after: dict[str, int],
    elapsed: float,
) -> None:
    print(f"\n{'─'*44}")
    print("  [DIFF]")
    print(f"{'─'*44}")

    # DB diff
    all_statuses = sorted(set(list(db_before.keys()) + list(db_after.keys())))
    changed = False
    for status in all_statuses:
        before = db_before.get(status, 0)
        after = db_after.get(status, 0)
        delta = after - before
        if delta != 0:
            sign = "+" if delta > 0 else ""
            print(f"  {status:<15} {before:>6} → {after:>6}  ({sign}{delta})")
            changed = True
    if not changed:
        print("  DB: 変化なし")

    # FAISS diff
    all_domains = sorted(set(list(faiss_before.keys()) + list(faiss_after.keys())))
    for domain in all_domains:
        before = faiss_before.get(domain, 0)
        after = faiss_after.get(domain, 0)
        delta = after - before
        if delta != 0 and before >= 0 and after >= 0:
            sign = "+" if delta > 0 else ""
            print(f"  FAISS [{domain}]  {before:>6} → {after:>6}  ({sign}{delta})")

    print(f"\n  Elapsed: {elapsed:.1f}s")


# ── メイン処理 ───────────────────────────────────────────────────────────────

async def run_seed(
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
    from scripts.seed_from_docs import seed_from_docs

    # ── 事前チェック ──────────────────────────────────────────────────────
    db_before = _db_snapshot()
    faiss_before = _faiss_snapshot()
    _print_status(db_before, faiss_before, label="PRE-CHECK")

    if source == "check-only":
        return

    # ── Seed 実行 ─────────────────────────────────────────────────────────
    print(f"\n{'─'*44}")
    print(f"  [SEED: {source}]  domain={domain}  dry_run={dry_run}")
    print(f"{'─'*44}")
    start = time.monotonic()

    await seed_from_docs(
        source=source,
        url_file=url_file,
        github_config=github_config,
        max_files=max_files,
        limit=limit,
        domain=domain,
        dry_run=dry_run,
        mature=mature,
        provider=provider,
        model=model,
    )

    elapsed = time.monotonic() - start

    # ── 事後チェック + diff ───────────────────────────────────────────────
    if not dry_run:
        db_after = _db_snapshot()
        faiss_after = _faiss_snapshot()
        _print_diff(db_before, db_after, faiss_before, faiss_after, elapsed)
        _print_status(db_after, faiss_after, label="POST-CHECK")
    else:
        print(f"\n  (dry-run: DB/FAISS 変更なし)  elapsed={elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed 管理スクリプト: 事前チェック → Seed → 事後チェック",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["github_docs", "url_list", "all", "check-only"],
        default="check-only",
        help="Seed ソース。check-only は状態確認のみ (default: check-only)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="--source check-only の短縮形",
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
        help="フェッチのみ、DB/FAISS を変更しない",
    )
    parser.add_argument(
        "--mature",
        action="store_true",
        help="Seed 後に Teacher で品質審査を実行",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="LLM プロバイダー（mature 時に使用）",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM モデル名（mature 時に使用）",
    )

    args = parser.parse_args()

    source = "check-only" if args.check_only else args.source

    asyncio.run(run_seed(
        source=source,
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


if __name__ == "__main__":
    main()
