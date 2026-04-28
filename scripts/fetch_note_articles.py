#!/usr/bin/env python3
"""scripts/fetch_note_articles.py — [非推奨] fetch_articles.py に移行済み

  新しいスクリプト: scripts/fetch_articles.py
  対応サイトの追加: scripts/fetchers/ にサブクラスを追加
  出力先: data/articles/{site}/{account}/

このスクリプトは後方互換性のため残してありますが、新規作業には fetch_articles.py を使用してください。
---
note.com アカウントの記事を取得して JSONL に保存

サーバー負荷対策: リクエスト間に DELAY 秒待機（デフォルト 3秒）。
レジューム対応: 既取得スラッグはスキップ。
マガジンは取得しない（対象アカウントに存在しないため）。

使い方:
    # テスト（最初の3件のみ）
    poetry run python scripts/fetch_note_articles.py --limit 3

    # 一覧のみ取得（本文なし）
    poetry run python scripts/fetch_note_articles.py --list-only

    # 全記事取得（長時間放置向け）
    poetry run python scripts/fetch_note_articles.py

    # 別アカウント / 間隔変更
    poetry run python scripts/fetch_note_articles.py --user other_user --delay 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import httpx

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_DEFAULT_USER = "tender_peony902"
_DEFAULT_DELAY = 3.0

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "ja,en;q=0.9",
}


# ── API 取得ヘルパー ──────────────────────────────────────────────────────────

def _get(client: httpx.Client, url: str) -> dict | None:
    """GET リクエスト。失敗時は None を返す。"""
    try:
        resp = client.get(url, headers=_HEADERS, timeout=30)
        if resp.status_code == 404:
            logger.warning("404: %s", url)
            return None
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        logger.warning("HTTP error %s: %s", e.response.status_code, url)
        return None
    except Exception as e:
        logger.warning("Request failed (%s): %s", type(e).__name__, url)
        return None


# ── 一覧取得 ─────────────────────────────────────────────────────────────────

def fetch_article_list(
    client: httpx.Client,
    user: str,
    delay: float,
    limit: int | None = None,
) -> list[dict]:
    """全記事メタデータを取得（ページネーション対応）。"""
    results: list[dict] = []
    page = 1

    while True:
        url = (
            f"https://note.com/api/v2/creators/{user}/contents"
            f"?kind=note&page={page}"
        )
        logger.info("Fetching list page %d ...", page)
        data = _get(client, url)
        if not data:
            logger.error("Failed to fetch list page %d. Stopping.", page)
            break

        contents = data.get("data", {}).get("contents", [])
        if not contents:
            break

        results.extend(contents)
        total = data.get("data", {}).get("totalCount", "?")
        logger.info(
            "  page %d: +%d articles (total so far: %d / %s)",
            page, len(contents), len(results), total,
        )

        if limit and len(results) >= limit:
            results = results[:limit]
            break

        is_last = data.get("data", {}).get("isLastPage", True)
        if is_last:
            break

        page += 1
        time.sleep(delay)

    return results


# ── 本文取得 ─────────────────────────────────────────────────────────────────

def fetch_article_detail(
    client: httpx.Client,
    key: str,
    delay: float,
) -> dict | None:
    """記事詳細（本文含む）を取得。key = 'n3abc...' 形式。"""
    time.sleep(delay)
    url = f"https://note.com/api/v3/notes/{key}"
    return _get(client, url)


# ── メイン ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="note.com 記事取得スクリプト（サーバー負荷対策付き）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--user",
        default=_DEFAULT_USER,
        help=f"note.com ユーザー名 (default: {_DEFAULT_USER})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=_DEFAULT_DELAY,
        help=f"リクエスト間隔（秒）(default: {_DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="取得上限（テスト用）",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="記事一覧のみ（本文取得なし）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="出力ディレクトリ (default: data/note_articles/)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or (_ROOT / "data" / "note_articles")
    out_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client() as client:
        # ── Phase 1: 記事一覧取得 ──
        articles = fetch_article_list(
            client, args.user, delay=args.delay, limit=args.limit
        )
        if not articles:
            logger.error("No articles found for user: %s", args.user)
            sys.exit(1)

        list_path = out_dir / f"{args.user}_list.json"
        list_path.write_text(
            json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Saved %d article metadata → %s", len(articles), list_path)

        if args.list_only:
            _print_summary(args.user, articles, 0, 0, 0, list_path, None)
            return

        # ── Phase 2: 本文取得（レジューム対応） ──
        body_path = out_dir / f"{args.user}_articles.jsonl"
        fetched_slugs = _load_fetched_slugs(body_path)
        if fetched_slugs:
            logger.info("Resume: %d articles already fetched, skipping.", len(fetched_slugs))

        new_count = skip_count = error_count = 0

        with body_path.open("a", encoding="utf-8") as f:
            for i, article in enumerate(articles):
                key = article.get("key", "")
                title = article.get("name", "")[:60]

                if not key:
                    logger.warning("[%d/%d] No key, skipping.", i + 1, len(articles))
                    error_count += 1
                    continue

                if key in fetched_slugs:
                    skip_count += 1
                    continue

                logger.info(
                    "[%d/%d] %s — %s",
                    i + 1, len(articles), key, title,
                )
                detail = fetch_article_detail(client, key, delay=args.delay)

                if detail:
                    f.write(json.dumps(detail, ensure_ascii=False) + "\n")
                    f.flush()
                    new_count += 1
                    fetched_slugs.add(key)
                else:
                    error_count += 1

        _print_summary(
            args.user, articles, new_count, skip_count, error_count,
            list_path, body_path,
        )


def _load_fetched_slugs(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            # API v3 レスポンス: {"data": {"key": "n...", "slug": "slug-n..."}}
            key = (
                obj.get("data", {}).get("key")
                or obj.get("key", "")
            )
            if key:
                keys.add(key)
        except Exception:
            pass
    return keys


def _print_summary(
    user: str,
    articles: list[dict],
    new_count: int,
    skip_count: int,
    error_count: int,
    list_path: Path,
    body_path: Path | None,
) -> None:
    print(f"\n{'='*52}")
    print(f"  SUMMARY  [{user}]")
    print(f"{'='*52}")
    print(f"  Total articles:  {len(articles)}")
    if body_path:
        print(f"  Newly fetched:   {new_count}")
        print(f"  Skipped:         {skip_count}  (already saved)")
        print(f"  Errors:          {error_count}")
    print(f"  List output:     {list_path}")
    if body_path:
        print(f"  Body output:     {body_path}")
    print(f"{'='*52}")


if __name__ == "__main__":
    main()
