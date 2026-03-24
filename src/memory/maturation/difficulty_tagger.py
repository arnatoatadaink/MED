"""src/memory/maturation/difficulty_tagger.py — 難易度タガー + 動的カリキュラム

ドキュメントの内容を LLM で分析し、DifficultyLevel を付与する。
Student モデルのカリキュラム学習（beginner → expert 順提示）に使用する。

CurriculumController:
    訓練中の損失推移を監視し、次バッチの難易度配分を動的に調整する。
    - 損失が低すぎる（簡単すぎる） → 難しいデータの割合を増やす
    - 損失が高すぎる（難しすぎる） → 中間難易度の割合を増やす

使い方:
    from src.memory.maturation.difficulty_tagger import DifficultyTagger, CurriculumController

    tagger = DifficultyTagger(gateway)
    level = await tagger.tag(doc)

    controller = CurriculumController()
    controller.record_step(loss=0.5, reward_mean=0.3)
    distribution = controller.get_distribution()
    # → {BEGINNER: 0.2, INTERMEDIATE: 0.4, ADVANCED: 0.3, EXPERT: 0.1}
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

from src.llm.gateway import LLMGateway
from src.memory.schema import DifficultyLevel, Document

logger = logging.getLogger(__name__)

_TAG_SYSTEM = """\
You are an expert at assessing technical difficulty. Classify the difficulty of the given text.
Respond with ONLY one of: beginner, intermediate, advanced, expert

Criteria:
- beginner: Basic concepts, no prior knowledge needed, simple examples
- intermediate: Requires some background, moderate complexity
- advanced: Deep technical knowledge required, complex algorithms/patterns
- expert: Cutting-edge research, specialized expertise needed"""

_TAG_PROMPT = "Classify the difficulty of this text:\n\n{text}"

_LEVEL_MAP: dict[str, DifficultyLevel] = {
    "beginner": DifficultyLevel.BEGINNER,
    "intermediate": DifficultyLevel.INTERMEDIATE,
    "advanced": DifficultyLevel.ADVANCED,
    "expert": DifficultyLevel.EXPERT,
}


class DifficultyTagger:
    """LLM でドキュメントの難易度を分類する。

    Args:
        gateway: LLMGateway インスタンス。
        provider: 優先プロバイダ（省略時はデフォルト）。
        max_text_length: LLM に渡す最大文字数。
        default_level: LLM 失敗時のデフォルト難易度。
    """

    def __init__(
        self,
        gateway: LLMGateway,
        provider: str | None = None,
        max_text_length: int = 800,
        default_level: DifficultyLevel = DifficultyLevel.INTERMEDIATE,
    ) -> None:
        self._gateway = gateway
        self._provider = provider
        self._max_text = max_text_length
        self._default_level = default_level

    async def tag(self, doc: Document) -> DifficultyLevel:
        """ドキュメントの難易度を分類する。

        Returns:
            DifficultyLevel（LLM 失敗時はデフォルト値）。
        """
        text = doc.content[:self._max_text]
        prompt = _TAG_PROMPT.format(text=text)

        try:
            response = await self._gateway.complete(
                prompt,
                system=_TAG_SYSTEM,
                provider=self._provider,
                max_tokens=20,
                temperature=0.0,
            )
            level_str = response.content.strip().lower().split()[0]
            level = _LEVEL_MAP.get(level_str, self._default_level)
            logger.debug("Difficulty for doc=%s: %s → %s", doc.id, level_str, level)
            return level
        except Exception:
            logger.exception("Difficulty tagging failed for doc=%s; using default", doc.id)
            return self._default_level

    async def tag_batch(self, docs: list[Document]) -> dict[str, DifficultyLevel]:
        """複数ドキュメントの難易度を一括分類する。

        Returns:
            {doc_id: DifficultyLevel} の辞書。
        """
        import asyncio
        results = await asyncio.gather(*[self.tag(doc) for doc in docs])
        return {doc.id: level for doc, level in zip(docs, results)}


# ──────────────────────────────────────────────
# Phase B-1: 動的カリキュラム調整
# ──────────────────────────────────────────────

# 難易度レベルの順序（低→高）
_DIFFICULTY_ORDER: list[DifficultyLevel] = [
    DifficultyLevel.BEGINNER,
    DifficultyLevel.INTERMEDIATE,
    DifficultyLevel.ADVANCED,
    DifficultyLevel.EXPERT,
]


@dataclass
class CurriculumConfig:
    """動的カリキュラム調整の設定。

    Attributes:
        window_size: 損失の移動平均を計算するウィンドウサイズ。
        loss_low_threshold: この値未満の損失 = 簡単すぎる → 難易度を上げる。
        loss_high_threshold: この値超の損失 = 難しすぎる → 難易度を下げる。
        adjustment_rate: 1回の調整で配分を動かす量（0.0〜1.0）。
        min_weight: 各難易度の最小配分割合。0 にすると完全に除外されうる。
    """

    window_size: int = 20
    loss_low_threshold: float = 0.3
    loss_high_threshold: float = 1.5
    adjustment_rate: float = 0.05
    min_weight: float = 0.05


class CurriculumController:
    """訓練中の損失推移を監視し、難易度配分を動的に調整する。

    損失が閾値を外れると配分をシフトする:
    - 低損失（簡単すぎ）→ 高難易度の割合を増やす
    - 高損失（難しすぎ）→ 低〜中難易度の割合を増やす
    - 中間（適正）→ 配分を維持

    Args:
        config: CurriculumConfig。
        initial_distribution: 初期の難易度配分。
            None の場合は均等配分 (各25%) を使用。
    """

    def __init__(
        self,
        config: CurriculumConfig | None = None,
        initial_distribution: dict[DifficultyLevel, float] | None = None,
    ) -> None:
        self._config = config or CurriculumConfig()
        self._loss_history: deque[float] = deque(maxlen=self._config.window_size)
        self._reward_history: deque[float] = deque(maxlen=self._config.window_size)
        self._step_count: int = 0
        self._adjustment_count: int = 0

        # 初期配分
        if initial_distribution is not None:
            self._distribution = dict(initial_distribution)
        else:
            n = len(_DIFFICULTY_ORDER)
            self._distribution = {level: 1.0 / n for level in _DIFFICULTY_ORDER}

        self._normalize()

    @property
    def distribution(self) -> dict[DifficultyLevel, float]:
        """現在の難易度配分（読み取り専用コピー）。"""
        return dict(self._distribution)

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def adjustment_count(self) -> int:
        """配分調整が行われた回数。"""
        return self._adjustment_count

    def record_step(self, loss: float, reward_mean: float = 0.0) -> None:
        """訓練ステップの結果を記録し、必要なら配分を調整する。

        Args:
            loss: 当該ステップの損失値。
            reward_mean: 当該ステップの平均報酬。
        """
        self._loss_history.append(loss)
        self._reward_history.append(reward_mean)
        self._step_count += 1

        # ウィンドウが埋まるまでは調整しない
        if len(self._loss_history) < self._config.window_size:
            return

        avg_loss = sum(self._loss_history) / len(self._loss_history)
        self._adjust(avg_loss)

    def get_distribution(self) -> dict[DifficultyLevel, float]:
        """現在の難易度配分を返す。バッチ構成に使用する。"""
        return dict(self._distribution)

    def get_sample_counts(self, batch_size: int) -> dict[DifficultyLevel, int]:
        """配分に基づき、指定バッチサイズでの各難易度のサンプル数を返す。

        端数は最大配分のレベルに加算する。

        Args:
            batch_size: バッチサイズ。

        Returns:
            {DifficultyLevel: count} の辞書。合計 = batch_size。
        """
        counts: dict[DifficultyLevel, int] = {}
        total_assigned = 0

        for level in _DIFFICULTY_ORDER:
            weight = self._distribution.get(level, 0.0)
            n = int(batch_size * weight)
            counts[level] = n
            total_assigned += n

        # 端数処理: 最大配分のレベルに残りを追加
        remainder = batch_size - total_assigned
        if remainder > 0:
            max_level = max(self._distribution, key=self._distribution.get)  # type: ignore[arg-type]
            counts[max_level] += remainder

        return counts

    def get_moving_average_loss(self) -> float | None:
        """現在の損失移動平均を返す。履歴が空なら None。"""
        if not self._loss_history:
            return None
        return sum(self._loss_history) / len(self._loss_history)

    def reset(self) -> None:
        """履歴と配分を初期状態に戻す。"""
        self._loss_history.clear()
        self._reward_history.clear()
        self._step_count = 0
        self._adjustment_count = 0
        n = len(_DIFFICULTY_ORDER)
        self._distribution = {level: 1.0 / n for level in _DIFFICULTY_ORDER}

    def to_dict(self) -> dict:
        """状態をシリアライズ可能な辞書に変換する。"""
        return {
            "step_count": self._step_count,
            "adjustment_count": self._adjustment_count,
            "distribution": {k.value: v for k, v in self._distribution.items()},
            "moving_avg_loss": self.get_moving_average_loss(),
            "loss_history_size": len(self._loss_history),
        }

    # ── 内部ロジック ─────────────────────────────

    def _adjust(self, avg_loss: float) -> None:
        """平均損失に基づいて配分を調整する。"""
        rate = self._config.adjustment_rate

        if avg_loss < self._config.loss_low_threshold:
            # 簡単すぎる → 高難易度を増やし、低難易度を減らす
            self._shift_up(rate)
            logger.info(
                "Curriculum shift UP (avg_loss=%.3f < %.3f): %s",
                avg_loss, self._config.loss_low_threshold,
                self._fmt_distribution(),
            )
            self._adjustment_count += 1

        elif avg_loss > self._config.loss_high_threshold:
            # 難しすぎる → 低〜中難易度を増やし、高難易度を減らす
            self._shift_down(rate)
            logger.info(
                "Curriculum shift DOWN (avg_loss=%.3f > %.3f): %s",
                avg_loss, self._config.loss_high_threshold,
                self._fmt_distribution(),
            )
            self._adjustment_count += 1

    def _shift_up(self, rate: float) -> None:
        """高難易度に配分をシフトする。"""
        # advanced, expert を増やし、beginner を減らす
        self._distribution[DifficultyLevel.BEGINNER] -= rate
        self._distribution[DifficultyLevel.EXPERT] += rate * 0.5
        self._distribution[DifficultyLevel.ADVANCED] += rate * 0.5
        self._normalize()

    def _shift_down(self, rate: float) -> None:
        """低〜中難易度に配分をシフトする。"""
        # beginner, intermediate を増やし、expert を減らす
        self._distribution[DifficultyLevel.EXPERT] -= rate
        self._distribution[DifficultyLevel.BEGINNER] += rate * 0.5
        self._distribution[DifficultyLevel.INTERMEDIATE] += rate * 0.5
        self._normalize()

    def _normalize(self) -> None:
        """配分を正規化する（最小値保証 + 合計1.0）。"""
        min_w = self._config.min_weight
        for level in _DIFFICULTY_ORDER:
            if self._distribution[level] < min_w:
                self._distribution[level] = min_w

        total = sum(self._distribution.values())
        if total > 0:
            for level in _DIFFICULTY_ORDER:
                self._distribution[level] /= total

    def _fmt_distribution(self) -> str:
        """配分を見やすい文字列に変換する。"""
        parts = [f"{lv.value}={self._distribution[lv]:.2f}" for lv in _DIFFICULTY_ORDER]
        return "{" + ", ".join(parts) + "}"
