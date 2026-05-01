"""scripts/run_cycle.py — 1 サイクルを CLI から実行する。

Usage:
    poetry run python scripts/run_cycle.py [--provider fastflowlm] [--dry-run]
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one knowledge collection cycle")
    p.add_argument("--provider", default="fastflowlm", help="LLM provider for enrich+mature")
    p.add_argument("--model", default=None, help="Override LLM model")
    p.add_argument("--persona", default="auto", help="Reviewer persona")
    p.add_argument("--enrich-only", action="store_true", help="Detect+enrich but skip dispatch")
    p.add_argument("--detect-only", action="store_true", help="Detect only, no enrich or dispatch")
    return p.parse_args()


async def main() -> None:
    args = _parse_args()

    from src.cycle.orchestrator import Orchestrator, OrchestratorConfig

    cfg = OrchestratorConfig(
        provider=args.provider,
        model=args.model,
        persona=args.persona,
    )

    if args.detect_only:
        from src.cycle.gap_detector import GapDetector
        import asyncio as _aio
        detector = GapDetector()
        tasks = await _aio.to_thread(detector.detect)
        print(f"\n=== Gap Detection Results ({len(tasks)} tasks) ===")
        for t in tasks:
            print(f"  [{t.priority:.2f}] {t.gap_type.value:22s}  {t.reason[:80]}")
        return

    orch = Orchestrator(config=cfg)

    if args.enrich_only:
        tasks = await orch._phase_detect()
        if tasks:
            tasks = await orch._phase_enrich(tasks)
            print(f"\n=== Enriched {len(tasks)} tasks ===")
            for t in tasks:
                print(f"  [{t.gap_type.value}] keywords={t.keywords[:3]}  queries={t.queries[:2]}")
        return

    run_id = await orch.run_cycle()
    print(f"\nCycle complete — run_id: {run_id}")


if __name__ == "__main__":
    asyncio.run(main())
