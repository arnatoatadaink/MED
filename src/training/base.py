"""src/training/base.py — 学習フレームワーク抽象インターフェース

Strategy パターン + Registry パターンで拡張可能な学習フレームワークを定義する。

設計原則:
- TrainingAlgorithm: RL/SFT アルゴリズムの基底クラス (GRPO/PPO/DPO/SFT)
- ParameterAdapter: モデルアダプタの基底クラス (TinyLoRA/LoRA/LoRA-XS/Full)
- RewardFunction: 報酬関数の基底クラス (Composite/CodeExec/TeacherEval/Hybrid)
- TrainingBatch / TrainingStep / TrainingResult: データ構造
- TrainingConfig: 学習設定 (pydantic)

使い方:
    from src.training.base import TrainingAlgorithm, ParameterAdapter, RewardFunction

    # 実装は registry.py でデコレータ登録
    @register_algorithm("grpo")
    class GRPOAlgorithm(TrainingAlgorithm):
        ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

from pydantic import BaseModel, Field

# ──────────────────────────────────────────────
# データ構造
# ──────────────────────────────────────────────


@dataclass
class TrainingBatch:
    """学習バッチ。

    Attributes:
        prompts: 入力プロンプトのリスト。
        responses: モデル出力（生成済み or 正解）のリスト。
        rewards: 各 (prompt, response) に対する報酬スコア。
        metadata: バッチに付随するメタデータ（doc_id など）。
    """

    prompts: list[str]
    responses: list[str]
    rewards: list[float] = field(default_factory=list)
    metadata: list[dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.prompts)

    def validate(self) -> None:
        """バッチの整合性を検証する。"""
        n = len(self.prompts)
        if len(self.responses) != n:
            raise ValueError(
                f"prompts({n}) and responses({len(self.responses)}) must have the same length"
            )
        if self.rewards and len(self.rewards) != n:
            raise ValueError(
                f"rewards({len(self.rewards)}) must match batch size ({n})"
            )


@dataclass
class TrainingStep:
    """1 ステップの学習結果。"""

    step: int
    loss: float
    reward_mean: float
    reward_std: float
    algorithm: str
    adapter: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """学習セッション全体の結果。"""

    algorithm: str
    adapter: str
    total_steps: int
    final_loss: float
    best_reward: float
    steps: list[TrainingStep] = field(default_factory=list)
    checkpoint_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def avg_reward(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.reward_mean for s in self.steps) / len(self.steps)


# ──────────────────────────────────────────────
# 設定モデル
# ──────────────────────────────────────────────


class TrainingConfig(BaseModel):
    """学習設定。"""

    algorithm: str = Field(default="grpo", description="学習アルゴリズム名")
    adapter: str = Field(default="tinylora", description="パラメータアダプタ名")
    reward: str = Field(default="composite", description="報酬関数名")

    max_steps: int = Field(default=1000, ge=1)
    batch_size: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    warmup_steps: int = Field(default=100, ge=0)
    eval_interval: int = Field(default=100, ge=1)
    save_interval: int = Field(default=500, ge=1)
    checkpoint_dir: str = Field(default="data/adapters")

    # アルゴリズム固有パラメータ
    algo_params: dict[str, Any] = Field(default_factory=dict)
    # アダプタ固有パラメータ
    adapter_params: dict[str, Any] = Field(default_factory=dict)
    # 報酬関数固有パラメータ
    reward_params: dict[str, Any] = Field(default_factory=dict)


# ──────────────────────────────────────────────
# 抽象基底クラス
# ──────────────────────────────────────────────


class TrainingAlgorithm(ABC):
    """学習アルゴリズムの基底クラス。

    GRPO / PPO / DPO / SFT など各種 RL/SFT アルゴリズムはこのクラスを継承する。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """アルゴリズム名（Registry 登録キーと一致すること）。"""
        ...

    @abstractmethod
    def compute_loss(
        self,
        batch: TrainingBatch,
        model: Any,
        adapter: ParameterAdapter,
    ) -> Any:  # torch.Tensor when torch is available
        """バッチからロスを計算する。

        Args:
            batch: 学習バッチ（rewards が設定済みであること）。
            model: ベースモデル（Student など）。
            adapter: パラメータアダプタ。

        Returns:
            スカラー Tensor ロス。
        """
        ...

    def before_step(self, step: int, config: TrainingConfig) -> None:
        """各ステップ前に呼ばれるフック（オプション）。"""

    def after_step(self, step: TrainingStep, config: TrainingConfig) -> None:
        """各ステップ後に呼ばれるフック（オプション）。"""


class ParameterAdapter(ABC):
    """パラメータアダプタの基底クラス。

    TinyLoRA / LoRA / LoRA-XS / Full Fine-Tuning はこのクラスを継承する。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """アダプタ名（Registry 登録キーと一致すること）。"""
        ...

    @property
    @abstractmethod
    def num_trainable_params(self) -> int:
        """学習可能パラメータ数。"""
        ...

    @abstractmethod
    def get_trainable_params(self) -> list[Any]:  # list[torch.nn.Parameter]
        """オプティマイザに渡すパラメータリストを返す。"""
        ...

    @abstractmethod
    def apply_to(self, model: Any) -> None:
        """モデルにアダプタを適用する（重みの差し込み）。"""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """アダプタ重みをファイルに保存する。"""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """ファイルからアダプタ重みを読み込む。"""
        ...


class RewardFunction(ABC):
    """報酬関数の基底クラス。

    Composite / CodeExec / TeacherEval / Hybrid などはこのクラスを継承する。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """報酬関数名（Registry 登録キーと一致すること）。"""
        ...

    @abstractmethod
    async def compute(
        self,
        prompt: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """(prompt, response) ペアの報酬スコアを計算する。

        Args:
            prompt: 入力プロンプト。
            response: モデル出力。
            metadata: ドキュメント情報・実行結果などのメタデータ。

        Returns:
            報酬スコア（通常 0.0〜1.0）。
        """
        ...

    async def compute_batch(
        self,
        batch: TrainingBatch,
    ) -> TrainingBatch:
        """バッチ全体の報酬を計算して batch.rewards に設定する。

        Returns:
            rewards が設定された TrainingBatch。
        """
        import asyncio

        metadata = batch.metadata or [{}] * len(batch)
        rewards = await asyncio.gather(*[
            self.compute(p, r, m)
            for p, r, m in zip(batch.prompts, batch.responses, metadata)
        ])
        batch.rewards = list(rewards)
        return batch
