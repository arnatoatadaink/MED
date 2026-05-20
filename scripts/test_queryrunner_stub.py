"""scripts/test_queryrunner_stub.py — WebGUI 経路（QueryRunner）のスタブテスト

seed_only.py の直接 retriever 呼び出しと異なり、GUI は
  Orchestrator → QueryRunner → RetrieverRouter → 各 retriever
という経路を通る。このスクリプトはその QueryRunner 層の挙動を
iptestserver を使って確認する。

テスト項目:
  1. SMALL_CLUSTER タスク → 全ソースがスタブに到達する
  2. SOURCE_IMBALANCE タスク → dominant_source (arxiv) が除外される
  3. QueryRunner キャッシュ (TTL=7) → 2回目は 0 リクエスト
  4. INTER_ISLAND_BRIDGE タスク → 全ソースがスタブに到達する        [P-QE-1]
  5. 0件ピボット → pivot_threshold 超過でピボットが発火し新クエリ到達 [P-QE-4/5]
     ※ Test 5 は iptestserver の empty モード (EMPTY_MODE_SPEC.md) が必要

FAISS への書き込みは relevance_threshold=2.0 で完全防止。
（cosine similarity の最大値は 1.0 なので閾値超えは不可能）

使い方:
  STUB=http://localhost:8002 PYTHONPATH=/mnt/d/Projects/claude_work/MED \\
    poetry run python scripts/test_queryrunner_stub.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.request

# ── 環境変数を MED モジュールの import より前に設定 ──────────────────
# retriever の API URL はモジュールレベルで os.environ.get() するため
# import 前の設定が必須。

STUB = os.environ.get("STUB", "").rstrip("/")
if not STUB:
    print("ERROR: STUB 環境変数が未設定です")
    print("  例: STUB=http://localhost:8002 poetry run python scripts/test_queryrunner_stub.py")
    sys.exit(1)

os.environ["MED_ARXIV_API_URL"]      = f"{STUB}/arxiv/query"
os.environ["MED_OPENREVIEW_API_URL"] = f"{STUB}/openreview/notes"
os.environ["MED_SO_API_BASE"]        = f"{STUB}/so"
os.environ["MED_GITHUB_API_BASE"]    = f"{STUB}/github"
os.environ.setdefault("GITHUB_TOKEN", "stub-token")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# ── MED モジュール ────────────────────────────────────────────────
from src.cycle.query_runner import QueryRunner, QueryRunnerConfig  # noqa: E402
from src.cycle.schema import CollectionTask, GapType              # noqa: E402


# ── iptestserver ヘルパー ─────────────────────────────────────────

def _get(path: str, **params) -> dict:
    url = f"{STUB}{path}"
    if params:
        url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


def _delete(path: str) -> dict:
    req = urllib.request.Request(f"{STUB}{path}", method="DELETE")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def _put(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{STUB}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def _request_sources(n: int = 100) -> list[str]:
    """iptestserver が受信したリクエストのソース名リストを返す。"""
    data = _get("/requests/recent", n=n)
    return [r["source"] for r in data.get("requests", [])]


def _request_queries(n: int = 200) -> list[str]:
    """iptestserver が受信したリクエストの query 文字列リストを返す（URL デコード済み）。"""
    import urllib.parse
    data = _get("/requests/recent", n=n)
    return [urllib.parse.unquote(r.get("query", "")) for r in data.get("requests", [])]


def _has_empty_mode() -> bool:
    """iptestserver が empty モードをサポートしているか確認する。"""
    try:
        _put("/control/arxiv", {"status": 200, "empty": False})
        _delete("/control/arxiv")
        return True
    except Exception:
        return False


# ── QueryRunner ファクトリ ────────────────────────────────────────

def _make_qr(
    cache_ttl_days: int = 0,
    pivot_threshold: float = 1.1,   # デフォルトは発火しない（1.1 > 100%）
    pivot_enabled: bool = False,
) -> QueryRunner:
    cfg = QueryRunnerConfig(
        relevance_threshold=2.0,   # 閾値超え不可 → FAISS 書き込みなし
        cache_ttl_days=cache_ttl_days,
        top_k=3,
        max_queries=2,
        pivot_threshold=pivot_threshold,
        pivot_enabled=pivot_enabled,
    )
    return QueryRunner(config=cfg)


# ── テスト関数 ────────────────────────────────────────────────────

async def test_small_cluster_all_sources() -> str:
    """SMALL_CLUSTER タスクが全ソース（arxiv/so/github/openreview）に到達する。"""
    print("\n" + "=" * 60)
    print("Test 1: SMALL_CLUSTER → 全ソースがスタブに到達")
    print("=" * 60)

    _delete("/requests")
    task = CollectionTask(
        gap_type=GapType.SMALL_CLUSTER,
        signals={},
        priority=0.8,
        reason="stub test",
        queries=["FAISS vector search Python", "sentence transformers embedding"],
    )

    qr = _make_qr(cache_ttl_days=0)
    try:
        await qr.initialize()
        stats = await qr.run_task(task)
    finally:
        await qr.close()

    sources = sorted(set(_request_sources()))
    print(f"Stats      : {stats}")
    print(f"Sources hit: {sources}")
    print(f"Requests   : {len(sources)} 種別")

    expected = {"arxiv", "openreview", "so", "github"}
    missing = expected - set(sources)
    result = "PASS" if not missing else f"FAIL (missing: {missing})"
    print(f"Result     : {result}")
    return result


async def test_source_imbalance_exclusion() -> str:
    """SOURCE_IMBALANCE タスクが dominant_source=arxiv を除外する。"""
    print("\n" + "=" * 60)
    print("Test 2: SOURCE_IMBALANCE → dominant_source (arxiv) を除外")
    print("=" * 60)

    _delete("/requests")
    task = CollectionTask(
        gap_type=GapType.SOURCE_IMBALANCE,
        signals={"dominant_source": "arxiv"},
        priority=0.5,
        reason="stub test imbalance",
        queries=["FAISS vector search Python"],
    )

    qr = _make_qr(cache_ttl_days=0)
    try:
        await qr.initialize()
        stats = await qr.run_task(task)
    finally:
        await qr.close()

    sources = sorted(set(_request_sources()))
    print(f"Stats      : {stats}")
    print(f"Sources hit: {sources}  (arxiv が含まれないこと)")

    result = "PASS" if "arxiv" not in sources and len(sources) > 0 else f"FAIL (sources={sources})"
    print(f"Result     : {result}")
    return result


async def test_query_cache() -> str:
    """QueryRunner の TTL=7 キャッシュ: 2 回目は 0 リクエスト。"""
    print("\n" + "=" * 60)
    print("Test 3: QueryRunner キャッシュ (TTL=7) → 2回目は 0 リクエスト")
    print("=" * 60)

    # 毎回異なるユニーク文字列を使用（日付固定だと前回キャッシュが残るため UUID を使う）
    import uuid
    unique_query = f"qr_stub_cache_test_unique_{uuid.uuid4().hex[:12]}"
    task = CollectionTask(
        gap_type=GapType.SMALL_CLUSTER,
        signals={},
        priority=0.5,
        reason="cache test",
        queries=[unique_query],
    )

    # Run 1: キャッシュなし → リクエスト発生
    _delete("/requests")
    qr1 = _make_qr(cache_ttl_days=7)
    try:
        await qr1.initialize()
        await qr1.run_task(task)
    finally:
        await qr1.close()
    count1 = len(_request_sources(n=100))

    # Run 2: キャッシュあり → リクエスト 0
    _delete("/requests")
    qr2 = _make_qr(cache_ttl_days=7)
    try:
        await qr2.initialize()
        await qr2.run_task(task)
    finally:
        await qr2.close()
    count2 = len(_request_sources(n=100))

    print(f"Run 1 requests: {count1}  (期待: >0)")
    print(f"Run 2 requests: {count2}  (期待: 0)")

    result = "PASS" if count1 > 0 and count2 == 0 else f"FAIL (run1={count1}, run2={count2})"
    print(f"Result     : {result}")
    return result


async def test_inter_island_bridge_all_sources() -> str:
    """INTER_ISLAND_BRIDGE タスクが全ソース（arxiv/so/github/openreview）に到達する。

    _select_sources() は INTER_ISLAND_BRIDGE を SMALL_CLUSTER と同様に扱い
    全利用可能ソースを返す。[P-QE-1]
    """
    print("\n" + "=" * 60)
    print("Test 4: INTER_ISLAND_BRIDGE → 全ソースがスタブに到達 [P-QE-1]")
    print("=" * 60)

    _delete("/requests")
    task = CollectionTask(
        gap_type=GapType.INTER_ISLAND_BRIDGE,
        signals={
            "island_a": {"cluster_id": 1},
            "island_b": {"cluster_id": 3},
            "bridge_dist": 2.5,
            "sample_doc_ids": [],
            "size": 80,
            "q_avg": 0.65,
            "approved_pct": 0.5,
            "source_dist": {"arxiv": 50, "github": 30},
            "dominant_source": "arxiv",
            "theory_pct": 0.625,
            "impl_pct": 0.375,
        },
        priority=0.75,
        reason="stub test bridge islands #1 and #3",
        queries=[
            "bridging FAISS and knowledge graphs",
            "cross-domain retrieval survey",
        ],
    )

    qr = _make_qr()
    try:
        await qr.initialize()
        stats = await qr.run_task(task)
    finally:
        await qr.close()

    sources = sorted(set(_request_sources()))
    print(f"Stats      : {stats}")
    print(f"Sources hit: {sources}")

    expected = {"arxiv", "openreview", "so", "github"}
    missing = expected - set(sources)
    result = "PASS" if not missing else f"FAIL (missing: {missing})"
    print(f"Result     : {result}")
    return result


async def test_pivot_on_zero_results() -> str:
    """0件クエリが pivot_threshold を超えたとき、ピボットが発火し新クエリがスタブに届く。

    前提: iptestserver が empty モードをサポートしていること (EMPTY_MODE_SPEC.md)。
    QueryGenerator.enrich_pivot は LLM を呼ぶため、テスト内でモックに差し替える。
    [P-QE-4/5]
    """
    print("\n" + "=" * 60)
    print("Test 5: 0件ピボット → 新クエリがスタブに到達 [P-QE-4/5]")
    print("=" * 60)

    if not _has_empty_mode():
        msg = "SKIP (iptestserver が empty モード未対応 — EMPTY_MODE_SPEC.md を参照)"
        print(f"Result     : {msg}")
        return msg

    PIVOT_MARKER = "pivot_stub_marker_xqz"

    # 全ソースを empty モードに設定（0件レスポンス）
    for src in ["arxiv", "openreview", "so", "github"]:
        _put(f"/control/{src}", {"status": 200, "empty": True})

    _delete("/requests")

    from unittest.mock import patch
    from src.cycle.query_generator import QueryGenerator

    async def _fake_enrich_pivot(self: object, task: object, zero_result_queries: list) -> object:
        task.queries = [PIVOT_MARKER]  # type: ignore[attr-defined]
        return task

    task = CollectionTask(
        gap_type=GapType.SMALL_CLUSTER,
        signals={},
        priority=0.5,
        reason="pivot test",
        queries=["zero_result_q1", "zero_result_q2"],
    )

    try:
        with patch.object(QueryGenerator, "enrich_pivot", _fake_enrich_pivot):
            qr = _make_qr(pivot_threshold=0.5, pivot_enabled=True)
            await qr.initialize()
            stats = await qr.run_task(task)
            await qr.close()
    finally:
        # 全ソースを正常に戻す
        for src in ["arxiv", "openreview", "so", "github"]:
            _delete(f"/control/{src}")

    queries_logged = _request_queries(n=200)
    pivot_hit = any(PIVOT_MARKER in q for q in queries_logged)

    print(f"Stats      : {stats}")
    print(f"Pivot marker '{PIVOT_MARKER}' in iptestserver logs: {pivot_hit}")

    result = "PASS" if pivot_hit else f"FAIL (pivot marker not found in request logs)"
    print(f"Result     : {result}")
    return result


# ── メイン ────────────────────────────────────────────────────────

async def main() -> None:
    # ヘルスチェック
    try:
        health = _get("/health")
    except Exception as e:
        print(f"ERROR: iptestserver に接続できません ({STUB}): {e}")
        sys.exit(1)
    print(f"iptestserver: OK ({STUB})  status={health}")
    print(f"Stub env vars:")
    print(f"  MED_ARXIV_API_URL      = {os.environ['MED_ARXIV_API_URL']}")
    print(f"  MED_OPENREVIEW_API_URL = {os.environ['MED_OPENREVIEW_API_URL']}")
    print(f"  MED_SO_API_BASE        = {os.environ['MED_SO_API_BASE']}")
    print(f"  MED_GITHUB_API_BASE    = {os.environ['MED_GITHUB_API_BASE']}")

    r1 = await test_small_cluster_all_sources()
    r2 = await test_source_imbalance_exclusion()
    r3 = await test_query_cache()
    r4 = await test_inter_island_bridge_all_sources()
    r5 = await test_pivot_on_zero_results()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    results = [r1, r2, r3, r4, r5]
    all_pass = all(r.startswith("PASS") or r.startswith("SKIP") for r in results)
    print(f"  Test 1 (SMALL_CLUSTER 全ソース)             : {r1}")
    print(f"  Test 2 (SOURCE_IMBALANCE 除外)               : {r2}")
    print(f"  Test 3 (QueryRunner キャッシュ TTL=7)        : {r3}")
    print(f"  Test 4 (INTER_ISLAND_BRIDGE 全ソース) [P-QE] : {r4}")
    print(f"  Test 5 (0件ピボット発火)             [P-QE] : {r5}")
    print(f"  Overall: {'ALL PASS ✓' if all_pass else 'FAILED ✗'}")


if __name__ == "__main__":
    asyncio.run(main())
