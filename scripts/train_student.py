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
    dry_run: bool,
) -> None:
    from src.common.config import get_settings
    from src.llm.gateway import LLMGateway
    from src.memory.embedder import Embedder
    from src.memory.faiss_index import FAISSIndexManager
    from src.memory.maturation.seed_builder import SeedBuilder
    from src.memory.memory_manager import MemoryManager
    from src.memory.metadata_store import MetadataStore
    from src.training.pipeline import TrainingConfig, TrainingPipeline
    from src.training.registry import TrainingRegistry

    settings = get_settings()
    gateway = LLMGateway(settings)

    # Seed データ生成
    print("[train] Building training data from memory...")
    embedder = Embedder()
    index_manager = FAISSIndexManager()
    metadata_store = MetadataStore(settings.memory.metadata_db_path)
    await metadata_store.initialize()
    memory_manager = MemoryManager(index_manager, metadata_store, embedder)

    builder = SeedBuilder(gateway, memory_manager)
    training_data = await builder.build_dataset(n_samples=n_steps * batch_size, domain=domain)
    print(f"[train] Generated {len(training_data)} training examples")

    if dry_run:
        print("[train] Dry run: skipping actual training")
        for i, sample in enumerate(training_data[:3]):
            print(f"  [{i}] {sample.get('prompt', '')[:80]}...")
        return

    # 学習パイプライン
    print(f"[train] Starting {algorithm.upper()}+{adapter} training ({n_steps} steps)...")

    AlgoCls = TrainingRegistry.get_algorithm(algorithm)
    AdaptCls = TrainingRegistry.get_adapter(adapter)

    config = TrainingConfig(
        algorithm=algorithm,
        adapter=adapter,
        n_steps=n_steps,
        batch_size=batch_size,
    )

    pipeline = TrainingPipeline(
        algorithm=AlgoCls(config),
        adapter=AdaptCls(config),
        gateway=gateway,
        config=config,
    )

    result = await pipeline.run(training_data)
    print(f"[train] Training completed: avg_reward={result.avg_reward:.4f}")
    print(f"[train] Adapter saved to: {result.adapter_path or 'N/A'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Student model with GRPO+TinyLoRA")
    parser.add_argument("--algorithm", default="grpo",
                        choices=["grpo", "ppo", "dpo", "reinforce", "sft"])
    parser.add_argument("--adapter", default="tinylora",
                        choices=["tinylora", "lora", "lora_xs", "full_ft"])
    parser.add_argument("--steps", type=int, default=100, dest="n_steps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Generate data only, no training")
    args = parser.parse_args()

    asyncio.run(run_training(
        algorithm=args.algorithm,
        adapter=args.adapter,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        domain=args.domain,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
