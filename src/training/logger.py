"""src/training/logger.py — 学習ログ管理

学習ステップのメトリクスを記録・集計する。
Weights & Biases との統合をサポートする（オプション）。

使い方:
    from src.training.logger import TrainingLogger

    logger = TrainingLogger(run_name="grpo_tinylora_v1", use_wandb=False)
    logger.log_step(step)
    logger.save("data/training_logs/run.jsonl")
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from src.training.base import TrainingStep

log = logging.getLogger(__name__)


class TrainingLogger:
    """学習メトリクスの記録・集計クラス。

    Args:
        run_name: 学習ランの名前（ログファイル名・W&B ランク名に使用）。
        use_wandb: W&B を使用するか (False ならローカルのみ)。
        project: W&B プロジェクト名。
        log_interval: コンソール出力する間隔 (steps)。
    """

    def __init__(
        self,
        run_name: str = "training_run",
        use_wandb: bool = False,
        project: str = "MED",
        log_interval: int = 10,
    ) -> None:
        self._run_name = run_name
        self._use_wandb = use_wandb
        self._project = project
        self._log_interval = log_interval
        self._steps: list[TrainingStep] = []
        self._start_time = time.time()
        self._wandb_run: Any = None

        if use_wandb:
            self._init_wandb()

    def _init_wandb(self) -> None:
        try:
            import wandb  # type: ignore[import]
            self._wandb_run = wandb.init(
                project=self._project,
                name=self._run_name,
                reinit=True,
            )
            log.info("W&B initialized: run=%s", self._run_name)
        except ImportError:
            log.warning("wandb not installed; disabling W&B logging")
            self._use_wandb = False
        except Exception:
            log.exception("W&B init failed; disabling")
            self._use_wandb = False

    def log_step(self, step: TrainingStep) -> None:
        """1 ステップのメトリクスを記録する。"""
        self._steps.append(step)

        if step.step % self._log_interval == 0:
            elapsed = time.time() - self._start_time
            log.info(
                "[%s] step=%d loss=%.4f reward=%.4f±%.4f elapsed=%.1fs",
                self._run_name, step.step, step.loss,
                step.reward_mean, step.reward_std, elapsed,
            )

        if self._use_wandb and self._wandb_run is not None:
            try:
                import wandb  # type: ignore[import]
                wandb.log({
                    "loss": step.loss,
                    "reward_mean": step.reward_mean,
                    "reward_std": step.reward_std,
                    **step.extra,
                }, step=step.step)
            except Exception:
                log.debug("W&B log failed at step %d", step.step)

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        """任意のメトリクスを記録する（評価時など）。"""
        step_label = f"step={step}" if step is not None else "eval"
        log.info("[%s] %s metrics: %s", self._run_name, step_label, metrics)

        if self._use_wandb and self._wandb_run is not None:
            try:
                import wandb  # type: ignore[import]
                kwargs = {"step": step} if step is not None else {}
                wandb.log(metrics, **kwargs)
            except Exception:
                pass

    @property
    def steps(self) -> list[TrainingStep]:
        return list(self._steps)

    @property
    def latest_step(self) -> Optional[TrainingStep]:
        return self._steps[-1] if self._steps else None

    def summary(self) -> dict[str, Any]:
        """学習サマリを返す。"""
        if not self._steps:
            return {}
        losses = [s.loss for s in self._steps]
        rewards = [s.reward_mean for s in self._steps]
        return {
            "run_name": self._run_name,
            "total_steps": len(self._steps),
            "final_loss": losses[-1],
            "best_loss": min(losses),
            "final_reward": rewards[-1],
            "best_reward": max(rewards),
            "avg_reward": sum(rewards) / len(rewards),
            "elapsed_seconds": time.time() - self._start_time,
        }

    def save(self, path: str) -> None:
        """ステップログを JSONL 形式で保存する。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for step in self._steps:
                f.write(json.dumps(asdict(step)) + "\n")
        log.info("TrainingLogger saved %d steps to %s", len(self._steps), p)

    @classmethod
    def load(cls, path: str, run_name: Optional[str] = None) -> "TrainingLogger":
        """JSONL ファイルからログを復元する。"""
        p = Path(path)
        logger = cls(run_name=run_name or p.stem, use_wandb=False)
        with open(p, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                step = TrainingStep(**data)
                logger._steps.append(step)
        log.info("TrainingLogger loaded %d steps from %s", len(logger._steps), p)
        return logger

    def finish(self) -> None:
        """ログセッションを終了する（W&B Finish）。"""
        if self._use_wandb and self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception:
                pass
        log.info("TrainingLogger finished: %s", self.summary())
