#!/usr/bin/env python3
"""scripts/train_student.py — Student モデル学習スクリプト

SeedBuilder でトレーニングデータを生成し、GRPO+TinyLoRA で Student を学習する。

使い方:
    python scripts/train_student.py
    python scripts/train_student.py --algorithm grpo --adapter tinylora --steps 200
    python scripts/train_student.py --algorithm dpo --adapter lora --steps 100
    python scripts/train_student.py --dry-run   # データ生成のみ（学習なし）
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


async def run_training(
    algorithm: str,
    adapter: str,
    n_steps: int,
    batch_size: int,
    domain: str | None,
    topic: str,
    dry_run: bool,
) -> None:
    from src.common.config import get_settings
    from src.llm.gateway import LLMGateway
    from src.memory.embedder import Embedder
    from src.memory.maturation.seed_builder import SeedBuilder
    from src.memory.memory_manager import MemoryManager

    settings = get_settings()
    gateway = LLMGateway(settings)

    # メモリマネージャ初期化
    embedder = Embedder()
    mm = MemoryManager(embedder=embedder)
    await mm.initialize()

    # Seed データ生成（Teacher API 呼び出し）
    n_samples = min(n_steps * batch_size, 50)  # コスト制御: 最大50サンプル
    print(f"[train] Building {n_samples} training samples via SeedBuilder...")
    print(f"[train] Topic: {topic!r}, Domain: {domain or 'general'}")

    builder = SeedBuilder(gateway, mm)
    result = await builder.build(
        topic=topic,
        domain=domain or "general",
        n_samples=n_samples,
    )
    print(f"[train] Generated {result.docs_created} docs, {result.docs_failed} failed")

    if dry_run:
        print("[train] Dry run: skipping actual training")
        print(f"[train] Doc IDs: {result.doc_ids[:5]}{'...' if len(result.doc_ids) > 5 else ''}")
        await mm.close()
        return

    # 学習パイプライン（骨格 — 実モデルが必要）
    print(f"[train] Starting {algorithm.upper()}+{adapter} training ({n_steps} steps)...")
    print("[train] NOTE: Full training requires Student model (Qwen2.5-7B) + GPU.")
    print("[train] Training pipeline skeleton is ready but actual model loading is not yet implemented.")

    await mm.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Student model with GRPO+TinyLoRA")
    parser.add_argument("--algorithm", default="grpo",
                        choices=["grpo", "ppo", "dpo", "reinforce", "sft"])
    parser.add_argument("--adapter", default="tinylora",
                        choices=["tinylora", "lora", "lora_xs", "full_ft"])
    parser.add_argument("--steps", type=int, default=100, dest="n_steps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--topic", type=str, default="Python programming",
                        help="Topic for seed data generation")
    parser.add_argument("--dry-run", action="store_true", help="Generate data only, no training")
    args = parser.parse_args()

    asyncio.run(run_training(
        algorithm=args.algorithm,
        adapter=args.adapter,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        domain=args.domain,
        topic=args.topic,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
