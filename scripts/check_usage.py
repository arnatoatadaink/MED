"""OpenRouter 日別・ジョブ別使用量確認スクリプト。

Usage:
    poetry run python scripts/check_usage.py              # 今日のサマリー
    poetry run python scripts/check_usage.py --jobs 20   # 最近20ジョブ
    poetry run python scripts/check_usage.py --all        # 全日履歴
"""
from __future__ import annotations

import argparse
import asyncio
import sys

sys.path.insert(0, ".")


async def main(args: argparse.Namespace) -> None:
    from src.llm.daily_usage_tracker import DailyUsageTracker

    tracker = DailyUsageTracker()
    await tracker.initialize()

    if args.all:
        import aiosqlite
        db = tracker._db
        cur = await db.execute(
            "SELECT date, provider, total_requests, updated_at FROM daily_usage ORDER BY date DESC, provider"
        )
        rows = await cur.fetchall()
        print(f"{'Date':<12} {'Provider':<20} {'Requests':>10}  {'Updated'}")
        print("-" * 65)
        for r in rows:
            print(f"{r['date']:<12} {r['provider']:<20} {r['total_requests']:>10}  {r['updated_at']}")
    else:
        # 今日のサマリー
        today_rows = await tracker.get_today_summary()
        print("=== Today's Usage ===")
        if today_rows:
            for r in today_rows:
                print(f"  {r['provider']:<20}  {r['total_requests']:>5} requests  (updated {r['updated_at']})")
        else:
            print("  No requests today.")

    # 最近のジョブ
    job_limit = args.jobs
    jobs = await tracker.get_recent_jobs(limit=job_limit)
    print(f"\n=== Recent {job_limit} Jobs ===")
    if jobs:
        print(f"{'Job ID':<40} {'Provider':<15} {'Count':>6}  {'Script':<25}  {'Started'}")
        print("-" * 110)
        for j in jobs:
            print(
                f"{j['job_id']:<40} {j['provider']:<15} {j['request_count']:>6}  "
                f"{(j['script_name'] or ''):<25}  {j['started_at']}"
            )
    else:
        print("  No jobs recorded.")

    await tracker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenRouter 使用量確認")
    parser.add_argument("--jobs", type=int, default=10, help="表示するジョブ数 (default: 10)")
    parser.add_argument("--all", action="store_true", help="全日履歴を表示")
    asyncio.run(main(parser.parse_args()))
