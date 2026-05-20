"""scripts/test_retriever_stub.py — iptestserver スタブを使ったretriever動作確認

事前準備:
  1. iptestserver を LAN IP で起動:
       cd /mnt/d/Projects/claude_work/iptestserver
       poetry run uvicorn main:app --host 0.0.0.0 --port 8002

  2. iptestserver に retriever スタブエンドポイントを実装済みであること
     (RETRIEVER_STUB_SPEC.md 参照)

使い方:
  # 全ソースをデフォルト設定でテスト（200レスポンス）
  STUB=http://localhost:8002 poetry run python scripts/test_retriever_stub.py

  # 特定ソースのみ
  STUB=http://localhost:8002 poetry run python scripts/test_retriever_stub.py --source arxiv

  # 429ハンドリングをテスト（コントロールAPIで429を設定してから検索）
  STUB=http://localhost:8002 poetry run python scripts/test_retriever_stub.py --source arxiv --test-429

  # 結果確認後にリクエストログを表示（User-Agent / 送信元IPの確認）
  STUB=http://localhost:8002 poetry run python scripts/test_retriever_stub.py --inspect

環境変数:
  STUB          iptestserver のベース URL（必須）
  MED_*_API_URL / MED_*_API_BASE  各retrieverが参照するURL（STUB から自動設定）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.parse
import urllib.request
from typing import Any


# ──────────────────────────────────────────────
# iptestserver クライアントヘルパー
# ──────────────────────────────────────────────

def _stub_get(base: str, path: str, **params: Any) -> dict:
    url = f"{base}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def _stub_put(base: str, path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{base}{path}", data=data, method="PUT",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def _stub_delete(base: str, path: str) -> dict:
    req = urllib.request.Request(f"{base}{path}", method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


# ──────────────────────────────────────────────
# 環境変数セットアップ
# ──────────────────────────────────────────────

def setup_stub_env(stub_base: str) -> None:
    """各retrieverの環境変数をiptestserverに向ける。"""
    os.environ["MED_ARXIV_API_URL"] = f"{stub_base}/arxiv/query"
    os.environ["MED_OPENREVIEW_API_URL"] = f"{stub_base}/openreview/notes"
    os.environ["MED_SO_API_BASE"] = f"{stub_base}/so"
    os.environ["MED_GITHUB_API_BASE"] = f"{stub_base}/github"


# ──────────────────────────────────────────────
# テスト実行
# ──────────────────────────────────────────────

async def test_source(source: str, query: str = "FAISS similarity search") -> None:
    """1ソースのretrieverをスタブに向けて実行し、結果を表示する。"""
    print(f"\n{'='*60}")
    print(f"Source: {source}")
    print(f"{'='*60}")

    if source == "arxiv":
        from src.rag.retrievers.arxiv import ArXivRetriever
        retriever = ArXivRetriever()
    elif source == "openreview":
        from src.rag.retrievers.openreview import OpenReviewRetriever
        retriever = OpenReviewRetriever()
    elif source == "stackoverflow":
        from src.rag.retrievers.stackoverflow import StackOverflowRetriever
        retriever = StackOverflowRetriever()
    elif source == "github":
        from src.rag.retrievers.github import GitHubRetriever
        os.environ.setdefault("GITHUB_TOKEN", "stub-token")
        retriever = GitHubRetriever()
    else:
        print(f"Unknown source: {source}")
        return

    print(f"API URL: {os.environ.get(f'MED_{source.upper()}_API_URL') or os.environ.get(f'MED_{source.upper()}_API_BASE')}")

    results = await retriever.search(query, max_results=3)
    print(f"Results: {len(results)} 件")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r.title[:60]!r}  score={r.score:.2f}  source={r.source}")
        print(f"       url={r.url}")


async def test_429(stub_base: str, source: str) -> None:
    """コントロールAPIで429を設定→retriever実行→バックオフ確認→リセット。"""
    print(f"\n{'='*60}")
    print(f"429 Backoff Test: {source}")
    print(f"{'='*60}")

    # 429を設定
    resp = _stub_put(stub_base, f"/control/{source}", {"status": 429})
    print(f"Control SET 429: {resp}")

    # retrieverを実行（429が返るはず）
    await test_source(source)

    # バックオフDBの状態を確認
    if source == "arxiv":
        from src.rag.retrievers.arxiv import ArXivRetriever
        state = await ArXivRetriever.current_backoff_state()
        print(f"Backoff state after 429: minutes_level={state.minutes_level}, days_level={state.days_level}, ban_until={state.ban_until}")
    elif source == "openreview":
        from src.rag.retrievers.openreview import OpenReviewRetriever
        state = await OpenReviewRetriever.current_backoff_state()
        print(f"Backoff state after 429: minutes_level={state.minutes_level}, days_level={state.days_level}, ban_until={state.ban_until}")

    # 429をリセット
    resp = _stub_delete(stub_base, f"/control/{source}")
    print(f"Control RESET: {resp}")


def show_request_log(stub_base: str, source: str | None = None, n: int = 20) -> None:
    """iptestserverに記録されたリクエストログを表示する。"""
    print(f"\n{'='*60}")
    print("Request Log (iptestserver側の受信記録)")
    print(f"{'='*60}")

    params: dict[str, Any] = {"n": n}
    if source:
        params["source"] = source
    data = _stub_get(stub_base, "/requests/recent", **params)

    requests = data.get("requests", [])
    if not requests:
        print("(ログなし — エンドポイント未実装の可能性)")
        return

    for req in requests:
        print(f"  [{req.get('timestamp', '?')}]")
        print(f"    source     : {req.get('source', '?')}")
        print(f"    path       : {req.get('path', '?')}")
        print(f"    client_ip  : {req.get('client_ip', '?')}")
        print(f"    user_agent : {req.get('user_agent', '?')}")


# ──────────────────────────────────────────────
# seed_only.py クエリキャッシュ テスト
# ──────────────────────────────────────────────

async def test_query_cache(stub_base: str) -> None:
    """seed_with_rate_control のクエリキャッシュ動作を確認する。

    シナリオ:
      1. 1回目: 2クエリ × arxiv → iptestserver に 2 リクエスト送信
      2. 2回目: 同クエリ → キャッシュヒットで 0 リクエスト
      3. cache_ttl_days=0: キャッシュ無効 → 再度 2 リクエスト送信
    """
    import tempfile
    from seed_only import seed_with_rate_control

    QUERIES = [
        "FAISS similarity search Python",
        "sentence-transformers embedding",
    ]
    SOURCES_CACHE = ["arxiv"]
    TEMP_DB = "/tmp/seed_cache_stub_test.db"

    def _count_new_requests(before: int) -> int:
        data = _stub_get(stub_base, "/requests/recent", n=50)
        return len(data.get("requests", [])) - before

    def _request_count() -> int:
        data = _stub_get(stub_base, "/requests/recent", n=50)
        return len(data.get("requests", []))

    print(f"\n{'='*60}")
    print("Query Cache Test (seed_with_rate_control)")
    print(f"{'='*60}")
    print(f"  Queries   : {QUERIES}")
    print(f"  DB        : {TEMP_DB}")

    # テスト用 DB を毎回クリーン
    import os as _os
    if _os.path.exists(TEMP_DB):
        _os.remove(TEMP_DB)

    # ── Run 1: 初回送信 ──
    _stub_delete(stub_base, "/requests")
    print("\n[Run 1] 初回送信 (キャッシュなし)")
    await seed_with_rate_control(
        queries=QUERIES,
        sources=SOURCES_CACHE,
        cache_ttl_days=7,
        metadata_db_path=TEMP_DB,
    )
    count1 = _request_count()
    result1 = "PASS" if count1 >= len(QUERIES) else "FAIL"
    print(f"  iptestserver リクエスト数: {count1}  (期待: >={len(QUERIES)})  [{result1}]")

    # ── Run 2: 2回目（キャッシュヒット）──
    _stub_delete(stub_base, "/requests")
    print("\n[Run 2] 2回目 (キャッシュヒット → 0件送信)")
    await seed_with_rate_control(
        queries=QUERIES,
        sources=SOURCES_CACHE,
        cache_ttl_days=7,
        metadata_db_path=TEMP_DB,
    )
    count2 = _request_count()
    result2 = "PASS" if count2 == 0 else "FAIL"
    print(f"  iptestserver リクエスト数: {count2}  (期待: 0)  [{result2}]")

    # ── Run 3: cache_ttl_days=0（無効）──
    _stub_delete(stub_base, "/requests")
    print("\n[Run 3] cache_ttl_days=0 (キャッシュ無効 → 再送信)")
    await seed_with_rate_control(
        queries=QUERIES,
        sources=SOURCES_CACHE,
        cache_ttl_days=0,
        metadata_db_path=TEMP_DB,
    )
    count3 = _request_count()
    result3 = "PASS" if count3 >= len(QUERIES) else "FAIL"
    print(f"  iptestserver リクエスト数: {count3}  (期待: >={len(QUERIES)})  [{result3}]")

    # ── サマリ ──
    print(f"\n{'─'*60}")
    all_pass = result1 == "PASS" and result2 == "PASS" and result3 == "PASS"
    print(f"  Run1 (初回送信)  : {result1}")
    print(f"  Run2 (キャッシュ): {result2}")
    print(f"  Run3 (ttl=0)     : {result3}")
    print(f"  Overall          : {'ALL PASS' if all_pass else 'FAILED'}")


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────

SOURCES = ["arxiv", "openreview", "stackoverflow", "github"]


async def main() -> None:
    parser = argparse.ArgumentParser(description="retriever stub tester")
    parser.add_argument("--source", choices=SOURCES, help="テスト対象ソース（省略時は全て）")
    parser.add_argument("--test-429", action="store_true", help="429ハンドリングをテスト")
    parser.add_argument("--test-cache", action="store_true", help="seed_only.py クエリキャッシュをテスト")
    parser.add_argument("--inspect", action="store_true", help="iptestserverのリクエストログを表示")
    parser.add_argument("--query", default="FAISS similarity search", help="検索クエリ")
    args = parser.parse_args()

    stub_base = os.environ.get("STUB", "").rstrip("/")
    if not stub_base:
        print("ERROR: STUB 環境変数が未設定です")
        print("  例: STUB=http://localhost:8002 poetry run python scripts/test_retriever_stub.py")
        sys.exit(1)

    # ヘルスチェック
    health = _stub_get(stub_base, "/health")
    if "error" in health:
        print(f"ERROR: iptestserver に接続できません ({stub_base}): {health['error']}")
        sys.exit(1)
    print(f"iptestserver: OK ({stub_base})")

    # 環境変数をスタブに向ける（モジュールインポート前に実行）
    setup_stub_env(stub_base)

    targets = [args.source] if args.source else SOURCES

    if args.inspect:
        show_request_log(stub_base, args.source)
        return

    if args.test_429:
        for source in targets:
            if source in ("arxiv", "openreview"):
                await test_429(stub_base, source)
            else:
                print(f"[skip] 429テストは arxiv/openreview のみ対応 (source={source})")
        return

    if args.test_cache:
        await test_query_cache(stub_base)
        return

    _stub_delete(stub_base, "/requests")
    for source in targets:
        await test_source(source, args.query)

    show_request_log(stub_base)


if __name__ == "__main__":
    asyncio.run(main())
