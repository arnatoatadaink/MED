#!/usr/bin/env python3
"""scripts/fetch_articles.py — サイト別記事取得統合スクリプト

対応サイト: note（追加は scripts/fetchers/ にサブクラスを実装して REGISTRY に登録）

出力先: data/articles/{site}/{account}/
  list.json       — 最新メタデータ一覧（毎回上書き）
  articles.jsonl  — 取得済み記事本文（追記・レジューム対応）
  arxiv_ids.txt   — 抽出 arXiv ID（--extract-arxiv 時のみ更新）

有料記事: price > 0 または canRead == False の記事は本文を取得しない
          is_paid=True として JSONL に記録し、以後スキップ対象にする

使い方:
    # テスト（3件のみ、delay 3秒）
    poetry run python scripts/fetch_articles.py \\
        --site note --account tender_peony902 --limit 3 --delay-min 3 --delay-max 5

    # 新着のみ取得（前回以降の新記事だけ）
    poetry run python scripts/fetch_articles.py \\
        --site note --account tender_peony902 --new-only

    # 全件取得（長時間放置向け、20-60秒ランダム間隔）
    poetry run python scripts/fetch_articles.py \\
        --site note --account tender_peony902

    # arXiv ID を抽出して arxiv_ids.txt に保存
    poetry run python scripts/fetch_articles.py \\
        --site note --account tender_peony902 --extract-arxiv

    # 一覧のみ（本文取得なし）
    poetry run python scripts/fetch_articles.py \\
        --site note --account tender_peony902 --list-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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

_DEFAULT_DELAY_MIN = 20.0
_DEFAULT_DELAY_MAX = 60.0


# ── 保存済みキー読み込み（レジューム・新着検出共用） ───────────────────────────

def _load_processed_keys(jsonl_path: Path) -> set[str]:
    """取得済み（有料スキップ含む）記事キーを返す。"""
    if not jsonl_path.exists():
        return set()
    keys: set[str] = set()
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            key = obj.get("key", "")
            if key:
                keys.add(key)
        except Exception:
            pass
    return keys


# ── arXiv ID 抽出・保存 ───────────────────────────────────────────────────────

def _update_arxiv_ids(
    arxiv_path: Path,
    new_ids: list[str],
) -> int:
    """既存ファイルに新規 arXiv ID を追記。追加件数を返す。"""
    existing: set[str] = set()
    if arxiv_path.exists():
        for line in arxiv_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                existing.add(line)

    to_add = [i for i in new_ids if i not in existing]
    if to_add:
        with arxiv_path.open("a", encoding="utf-8") as f:
            for arxiv_id in to_add:
                f.write(arxiv_id + "\n")
    return len(to_add)


# ── メイン ───────────────────────────────────────────────────────────────────

def main() -> None:
    from scripts.fetchers import REGISTRY, extract_arxiv_ids, get_fetcher

    parser = argparse.ArgumentParser(
        description="サイト別記事取得スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--site",
        choices=list(REGISTRY.keys()),
        default="note",
        help=f"取得対象サイト (default: note)  available: {', '.join(REGISTRY.keys())}",
    )
    parser.add_argument(
        "--account",
        default="tender_peony902",
        help="アカウント識別子 (default: tender_peony902)",
    )
    parser.add_argument(
        "--delay-min",
        type=float,
        default=_DEFAULT_DELAY_MIN,
        help=f"待機時間の最小値（秒）(default: {_DEFAULT_DELAY_MIN})",
    )
    parser.add_argument(
        "--delay-max",
        type=float,
        default=_DEFAULT_DELAY_MAX,
        help=f"待機時間の最大値（秒）(default: {_DEFAULT_DELAY_MAX})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="取得上限（テスト用）",
    )
    parser.add_argument(
        "--new-only",
        action="store_true",
        help="前回取得以降の新着記事のみ取得（既存キーをスキップ）",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="記事一覧のみ取得（本文取得なし）",
    )
    parser.add_argument(
        "--extract-arxiv",
        action="store_true",
        help="取得済み本文から arXiv ID を抽出して arxiv_ids.txt に保存",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="出力ルートディレクトリ (default: data/articles/)",
    )
    args = parser.parse_args()

    if args.delay_min > args.delay_max:
        parser.error("--delay-min must be <= --delay-max")

    delay_range = (args.delay_min, args.delay_max)

    # ── 出力ディレクトリ ──
    out_root = args.out_dir or (_ROOT / "data" / "articles")
    out_dir = out_root / args.site / args.account
    out_dir.mkdir(parents=True, exist_ok=True)

    list_path = out_dir / "list.json"
    articles_path = out_dir / "articles.jsonl"
    arxiv_path = out_dir / "arxiv_ids.txt"

    with get_fetcher(args.site) as fetcher:
        # ── Phase 1: 記事一覧取得（ページネーション間のみ delay） ──
        logger.info("=== Phase 1: Fetch article list [%s / %s] ===", args.site, args.account)
        articles_meta = fetcher.fetch_article_list(
            account=args.account,
            delay_range=delay_range,
            limit=args.limit,
        )

        if not articles_meta:
            logger.error("No articles found.")
            sys.exit(1)

        list_path.write_text(
            json.dumps(articles_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Saved %d article metadata → %s", len(articles_meta), list_path)

        if args.list_only:
            _print_summary(args, articles_meta, 0, 0, 0, 0, list_path, None)
            return

        # ── Phase 2: 本文取得 ──
        logger.info("=== Phase 2: Fetch article bodies ===")
        processed_keys = _load_processed_keys(articles_path)
        if processed_keys:
            logger.info("Already processed: %d articles", len(processed_keys))

        # --new-only: 既存キーとの差分だけ取得
        if args.new_only and processed_keys:
            targets = [
                m for m in articles_meta
                if fetcher.get_key(m) not in processed_keys
            ]
            logger.info(
                "new-only mode: %d new / %d total",
                len(targets), len(articles_meta),
            )
        else:
            targets = [
                m for m in articles_meta
                if fetcher.get_key(m) not in processed_keys
            ]

        new_count = skip_paid = skip_resume = error_count = 0

        with articles_path.open("a", encoding="utf-8") as f:
            for i, meta in enumerate(articles_meta):
                key = fetcher.get_key(meta)
                title = meta.get("name", meta.get("title", ""))[:60]

                if not key:
                    logger.warning("[%d/%d] No key, skipping.", i + 1, len(articles_meta))
                    error_count += 1
                    continue

                if key in processed_keys:
                    skip_resume += 1
                    continue

                # 一覧メタデータ段階で有料と判定できる場合は先行スキップ
                if fetcher.is_paid(meta):
                    logger.info(
                        "[%d/%d] [paid] %s — %s",
                        i + 1, len(articles_meta), key, title,
                    )
                    paid_stub = {
                        "key": key,
                        "title": title,
                        "url": meta.get("noteUrl", ""),
                        "body_html": "",
                        "body_text": "",
                        "published_at": meta.get("publishAt", ""),
                        "is_paid": True,
                        "site": args.site,
                        "account": args.account,
                        "metadata": {"price": meta.get("price", 0)},
                    }
                    f.write(json.dumps(paid_stub, ensure_ascii=False) + "\n")
                    f.flush()
                    skip_paid += 1
                    processed_keys.add(key)
                    continue

                logger.info(
                    "[%d/%d] %s — %s",
                    i + 1, len(articles_meta), key, title,
                )
                article = fetcher.fetch_article(key, delay_range=delay_range, account=args.account)

                if article:
                    f.write(json.dumps(article.to_dict(), ensure_ascii=False) + "\n")
                    f.flush()
                    new_count += 1
                    processed_keys.add(key)
                    if article.is_paid:
                        skip_paid += 1
                        new_count -= 1
                else:
                    error_count += 1

        # ── Phase 3: arXiv ID 抽出（オプション） ──
        arxiv_added = 0
        if args.extract_arxiv:
            logger.info("=== Phase 3: Extract arXiv IDs ===")
            all_ids: list[str] = []
            for line in articles_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("is_paid"):
                        continue
                    body = obj.get("body_text", "") or obj.get("body_html", "")
                    ids = extract_arxiv_ids(body)
                    all_ids.extend(ids)
                except Exception:
                    pass

            unique_ids = sorted(set(all_ids))
            arxiv_added = _update_arxiv_ids(arxiv_path, unique_ids)
            logger.info(
                "arXiv extraction: %d unique IDs found, %d new → %s",
                len(unique_ids), arxiv_added, arxiv_path,
            )

    _print_summary(
        args, articles_meta, new_count, skip_paid, skip_resume, error_count,
        list_path, articles_path,
        arxiv_added=arxiv_added if args.extract_arxiv else None,
        arxiv_path=arxiv_path if args.extract_arxiv else None,
    )


def _print_summary(
    args: argparse.Namespace,
    articles_meta: list[dict],
    new_count: int,
    skip_paid: int,
    skip_resume: int,
    error_count: int,
    list_path: Path,
    articles_path: Path | None,
    arxiv_added: int | None = None,
    arxiv_path: Path | None = None,
) -> None:
    print(f"\n{'='*56}")
    print(f"  SUMMARY  [{args.site} / {args.account}]")
    print(f"{'='*56}")
    print(f"  Total in list:   {len(articles_meta)}")
    if articles_path:
        print(f"  Newly fetched:   {new_count}")
        print(f"  Paid (skipped):  {skip_paid}")
        print(f"  Already saved:   {skip_resume}")
        print(f"  Errors:          {error_count}")
    print(f"  List output:     {list_path}")
    if articles_path:
        print(f"  Body output:     {articles_path}")
    if arxiv_added is not None:
        print(f"  arXiv new IDs:   {arxiv_added}  →  {arxiv_path}")
    print(f"{'='*56}")


if __name__ == "__main__":
    main()
