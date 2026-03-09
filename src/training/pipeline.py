"""src/training/pipeline.py — 3段階学習パイプライン

CLAUDE.md 設計:
    Stage 1: SFT ウォームアップ (warm_steps)
    Stage 2: GRPO + TinyLoRA 本学習 (main_steps)
    Stage 3: 評価 (eval)

使い方:
    from src.training.pipeline import TrainingPipeline, PipelineConfig

    pipeline = TrainingPipeline.from_config(config)
    result = await pipeline.run(data_loader)
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Optional

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from src.training.base import (
    ParameterAdapter,
    RewardFunction,
    TrainingAlgorithm,
    TrainingBatch,
    TrainingConfig,
    TrainingResult,
    TrainingStep,
)
from src.training.logger import TrainingLogger
from src.training.registry import TrainingRegistry

log = logging.getLogger(__name__)


class _NoOpOptimizer:
    """torch なし環境用ダミーオプティマイザ。"""
    def zero_grad(self) -> None: pass
    def step(self) -> None: pass



@dataclass
class PipelineConfig:
    """3段階学習パイプライン設定。"""

    # Stage 1: SFT ウォームアップ
    sft_steps: int = 100
    sft_lr: float = 5e-5

    # Stage 2: GRPO 本学習
    grpo_steps: int = 900
    grpo_lr: float = 1e-4
    batch_size: int = 8

    # アルゴリズム / アダプタ / 報酬
    algorithm: str = "grpo"
    adapter: str = "tinylora"
    reward: str = "composite"
    algo_params: dict[str, Any] = field(default_factory=dict)
    adapter_params: dict[str, Any] = field(default_factory=dict)
    reward_params: dict[str, Any] = field(default_factory=dict)

    # チェックポイント
    checkpoint_dir: str = "data/adapters"
    save_interval: int = 200
    eval_interval: int = 100

    # ログ
    run_name: str = "med_training"
    use_wandb: bool = False

    @property
    def total_steps(self) -> int:
        return self.sft_steps + self.grpo_steps


class TrainingPipeline:
    """3段階学習パイプライン。

    Stage 1 (SFT): 教師あり微調整でモデルをウォームアップ
    Stage 2 (GRPO+TinyLoRA): RL でメモリ利用スキルを学習
    Stage 3 (Eval): 最終評価

    Args:
        model: Student モデル（PyTorch モジュール or ラッパー）。
        algorithm: TrainingAlgorithm インスタンス。
        adapter: ParameterAdapter インスタンス。
        reward_fn: RewardFunction インスタンス。
        config: PipelineConfig。
        logger: TrainingLogger（省略時は自動作成）。
    """

    def __init__(
        self,
        model: Any,
        algorithm: TrainingAlgorithm,
        adapter: ParameterAdapter,
        reward_fn: RewardFunction,
        config: PipelineConfig,
        logger: Optional[TrainingLogger] = None,
    ) -> None:
        self._model = model
        self._algorithm = algorithm
        self._adapter = adapter
        self._reward_fn = reward_fn
        self._config = config
        self._logger = logger or TrainingLogger(
            run_name=config.run_name,
            use_wandb=config.use_wandb,
        )
        self._sft_algorithm: Optional[TrainingAlgorithm] = None
        self._step = 0

    @classmethod
    def from_config(
        cls,
        config: PipelineConfig,
        model: Any,
        gateway: Any = None,
    ) -> "TrainingPipeline":
        """PipelineConfig からパイプラインを生成する。

        Registry を使ってアルゴリズム・アダプタ・報酬関数をインスタンス化する。
        """
        # Import all modules to trigger registration
        import src.training.algorithms.grpo  # noqa: F401
        import src.training.algorithms.ppo   # noqa: F401
        import src.training.algorithms.dpo   # noqa: F401
        import src.training.algorithms.sft   # noqa: F401
        import src.training.algorithms.reinforce  # noqa: F401
        import src.training.adapters.tinylora  # noqa: F401
        import src.training.adapters.lora      # noqa: F401
        import src.training.adapters.lora_xs   # noqa: F401
        import src.training.adapters.full_ft   # noqa: F401
        import src.training.rewards.composite  # noqa: F401
        import src.training.rewards.code_exec  # noqa: F401
        import src.training.rewards.teacher_eval  # noqa: F401
        import src.training.rewards.hybrid     # noqa: F401

        algo_cls = TrainingRegistry.get_algorithm(config.algorithm)
        adapter_cls = TrainingRegistry.get_adapter(config.adapter)
        reward_cls = TrainingRegistry.get_reward(config.reward)

        algorithm = algo_cls(**config.algo_params)
        adapter = adapter_cls(**config.adapter_params)
        reward_params = dict(config.reward_params)
        if gateway is not None:
            reward_params.setdefault("gateway", gateway)
        reward_fn = reward_cls(**reward_params)

        return cls(
            model=model,
            algorithm=algorithm,
            adapter=adapter,
            reward_fn=reward_fn,
            config=config,
        )

    async def run(
        self,
        data_source: AsyncIterator[TrainingBatch] | list[TrainingBatch],
    ) -> TrainingResult:
        """3段階学習を実行する。

        Args:
            data_source: バッチの非同期イテレータ or リスト。

        Returns:
            TrainingResult。
        """
        # アダプタをモデルに適用
        self._adapter.apply_to(self._model)

        # オプティマイザ
        trainable = self._adapter.get_trainable_params()
        if torch is not None and trainable:
            optimizer = torch.optim.AdamW(trainable, lr=self._config.grpo_lr)
        else:
            optimizer = _NoOpOptimizer()

        # バッチをリストに変換
        if isinstance(data_source, list):
            batches = data_source
        else:
            batches = [b async for b in data_source]

        if not batches:
            log.warning("No training batches provided")
            return self._make_result(0.0, 0.0)

        log.info(
            "TrainingPipeline: starting %d steps (sft=%d, grpo=%d)",
            self._config.total_steps,
            self._config.sft_steps,
            self._config.grpo_steps,
        )

        # Stage 1: SFT ウォームアップ
        if self._config.sft_steps > 0:
            await self._stage_sft(batches, optimizer)

        # Stage 2: GRPO 本学習
        await self._stage_grpo(batches, optimizer)

        # Stage 3: 評価
        eval_reward = await self._stage_eval(batches)

        # チェックポイント保存
        self._save_checkpoint(self._step)
        self._logger.finish()

        result = self._make_result(
            final_loss=self._logger.latest_step.loss if self._logger.latest_step else 0.0,
            best_reward=eval_reward,
        )
        log.info("TrainingPipeline complete: %s", result)
        return result

    # ── ステージ実装 ─────────────────────────────

    async def _stage_sft(
        self,
        batches: list[TrainingBatch],
        optimizer: Any,
    ) -> None:
        """Stage 1: SFT ウォームアップ。"""
        import src.training.algorithms.sft  # noqa: F401
        sft_algo = TrainingRegistry.get_algorithm("sft")()

        log.info("Stage 1: SFT warmup (%d steps)", self._config.sft_steps)
        for i in range(self._config.sft_steps):
            batch = batches[i % len(batches)]
            optimizer.zero_grad()
            loss = sft_algo.compute_loss(batch, self._model, self._adapter)
            loss.backward()
            optimizer.step()
            self._step += 1

            step_result = TrainingStep(
                step=self._step,
                loss=loss.item(),
                reward_mean=0.0,
                reward_std=0.0,
                algorithm="sft",
                adapter=self._adapter.name,
            )
            self._logger.log_step(step_result)

    async def _stage_grpo(
        self,
        batches: list[TrainingBatch],
        optimizer: Any,
    ) -> None:
        """Stage 2: GRPO + TinyLoRA 本学習。"""
        log.info("Stage 2: GRPO (%d steps)", self._config.grpo_steps)
        for i in range(self._config.grpo_steps):
            batch = batches[i % len(batches)]

            # 報酬計算
            batch = await self._reward_fn.compute_batch(batch)

            # ロス計算 + 更新
            optimizer.zero_grad()
            loss = self._algorithm.compute_loss(batch, self._model, self._adapter)
            loss.backward()
            optimizer.step()
            self._step += 1

            rewards = batch.rewards or [0.0]
            step_result = TrainingStep(
                step=self._step,
                loss=loss.item(),
                reward_mean=statistics.mean(rewards),
                reward_std=statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
                algorithm=self._algorithm.name,
                adapter=self._adapter.name,
            )
            self._logger.log_step(step_result)

            # 定期チェックポイント
            if self._step % self._config.save_interval == 0:
                self._save_checkpoint(self._step)

    async def _stage_eval(self, batches: list[TrainingBatch]) -> float:
        """Stage 3: 評価ステップ。"""
        log.info("Stage 3: Evaluation")
        eval_batch = batches[0]
        eval_batch = await self._reward_fn.compute_batch(eval_batch)
        rewards = eval_batch.rewards or [0.0]
        avg = statistics.mean(rewards)
        self._logger.log_metrics({"eval_reward": avg}, step=self._step)
        return avg

    def _save_checkpoint(self, step: int) -> None:
        ckpt_dir = Path(self._config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"{self._config.run_name}_step{step}.pkl"
        try:
            self._adapter.save(str(path))
            log.info("Checkpoint saved: %s", path)
        except Exception:
            log.exception("Checkpoint save failed at step %d", step)

    def _make_result(self, final_loss: float, best_reward: float) -> TrainingResult:
        return TrainingResult(
            algorithm=self._algorithm.name,
            adapter=self._adapter.name,
            total_steps=self._step,
            final_loss=final_loss,
            best_reward=best_reward,
            steps=self._logger.steps,
        )
