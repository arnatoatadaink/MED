#!/usr/bin/env python3
"""scripts/collect_sandbox.py — DockerSandbox 体験知収集

DB のドキュメントからコードブロックを抽出し、Docker コンテナで実行して
結果を知識ドキュメントとして FAISS メモリに格納する。

Usage:
    # Python + Bash（デフォルト）
    poetry run python scripts/collect_sandbox.py

    # 言語・ソース種別を指定
    poetry run python scripts/collect_sandbox.py --languages python bash --limit 30

    # 全実行可能言語
    poetry run python scripts/collect_sandbox.py --languages python bash javascript

    # ネットワーク無効（pip install 不可、高速）
    poetry run python scripts/collect_sandbox.py --no-network

    # 統計のみ（実行なし）
    poetry run python scripts/collect_sandbox.py --stats-only
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def run_collection(
    languages: list[str],
    source_types: list[str] | None,
    limit: int,
    cmd_timeout: int,
    network_disabled: bool,
) -> None:
    from src.memory.embedder import Embedder
    from src.memory.memory_manager import MemoryManager
    from src.sandbox.sandbox_collector import SandboxCollector

    embedder = Embedder()
    mm = MemoryManager(embedder=embedder)
    await mm.initialize()

    try:
        collector = SandboxCollector(
            mm,
            network_disabled=network_disabled,
        )
        logger.info(
            "Starting sandbox collection: languages=%s limit=%d timeout=%ds network=%s",
            languages, limit, cmd_timeout,
            "disabled" if network_disabled else "enabled",
        )
        stats = await collector.collect(
            languages=languages,
            source_types=source_types or None,
            limit=limit,
            cmd_timeout=cmd_timeout,
        )
        print(f"\n{'='*50}")
        print("  SANDBOX COLLECTION SUMMARY")
        print(f"{'='*50}")
        print(f"  extracted:  {stats.extracted}")
        print(f"  new blocks: {stats.new_blocks}")
        print(f"  executed:   {stats.executed}")
        print(f"  stored:     {stats.stored}")
        print(f"  skipped:    {stats.skipped}")
        print(f"  errors:     {stats.errors}")
        if stats.by_lang:
            print(f"  by lang:    {stats.by_lang}")
    finally:
        await mm.close()


def show_stats(languages: list[str] | None) -> None:
    from src.sandbox.code_extractor import CodeExtractor, _EXECUTABLE

    ext = CodeExtractor()
    print("\n[Sandbox] 実行可能コードブロック統計")
    print(f"  実行可能言語: {list(_EXECUTABLE.keys())}")

    stats = ext.stats(languages=languages)
    print(f"\n  言語分布 (対象: {languages or '全言語'}):")
    for lang, cnt in stats.items():
        exe = " ← 実行可" if lang in _EXECUTABLE else ""
        print(f"    {lang:<20} {cnt:4d}{exe}")

    exe_total = sum(v for k, v in stats.items() if k in _EXECUTABLE)
    print(f"\n  実行可能ブロック合計: {exe_total}")
    print("\n  → 収集開始: poetry run python scripts/collect_sandbox.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="DockerSandbox 体験知収集")
    parser.add_argument(
        "--languages", nargs="+", default=["python", "bash"],
        choices=["python", "bash", "javascript", "typescript"],
        help="対象言語（デフォルト: python bash）",
    )
    parser.add_argument(
        "--source-types", nargs="+", default=None,
        choices=["github_docs", "web_docs", "stackoverflow", "arxiv", "tavily"],
        help="対象ソース種別（省略時: 全種別）",
    )
    parser.add_argument(
        "--limit", type=int, default=50,
        help="格納する最大件数（デフォルト: 50）",
    )
    parser.add_argument(
        "--timeout", type=int, default=30,
        help="1コマンドのタイムアウト秒数（デフォルト: 30）",
    )
    parser.add_argument(
        "--no-network", action="store_true",
        help="Docker ネットワークを無効化（pip install 不可、高速）",
    )
    parser.add_argument(
        "--stats-only", action="store_true",
        help="統計のみ表示して終了（実行なし）",
    )
    args = parser.parse_args()

    if args.stats_only:
        show_stats(args.languages)
        return

    asyncio.run(run_collection(
        languages=args.languages,
        source_types=args.source_types,
        limit=args.limit,
        cmd_timeout=args.timeout,
        network_disabled=args.no_network,
    ))


if __name__ == "__main__":
    main()
