"""src/memory/learning/ltr_ranker.py — Learning to Rank (線形モデル)

FAISS スコア・メタデータ特徴量を入力に、ランキングスコアを出力する線形モデル。
ユーザーフィードバック（クリック・選択）を使ったオンライン学習に対応する。

設計方針:
- Phase 1: 線形モデル（重みベクトルによる特徴量の内積）
- Phase 2: Cross-Encoder (cross_encoder.py) に移行
- オンライン学習: Stochastic Gradient Descent (SGD) を使ったリアルタイム更新

特徴量:
    0: faiss_score      — FAISS 内積スコア (コサイン類似度)
    1: freshness        — フレッシュネススコア
    2: usefulness       — 有用性スコア
    3: teacher_quality  — Teacher 品質スコア
    4: selection_rate   — 選択率 (selection/retrieval)
    5: confidence       — SQLite confidence カラム

使い方:
    from src.memory.learning.ltr_ranker import LTRRanker, RankFeatures

    ranker = LTRRanker()
    features = RankFeatures(faiss_score=0.8, freshness=0.9, usefulness=0.7, ...)
    score = ranker.score(features)

    # フィードバックで重みを更新
    ranker.update(features, reward=1.0)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# 特徴量インデックス
_FEATURE_NAMES = [
    "faiss_score",
    "freshness",
    "usefulness",
    "teacher_quality",
    "selection_rate",
    "confidence",
]
N_FEATURES = len(_FEATURE_NAMES)


@dataclass
class RankFeatures:
    """LTR モデルへの入力特徴量。

    全特徴量は 0.0〜1.0 の範囲に正規化すること。
    """

    faiss_score: float = 0.0      # FAISS 内積スコア（-1〜1 → クランプして 0〜1）
    freshness: float = 0.5        # フレッシュネススコア
    usefulness: float = 0.5       # 有用性スコア
    teacher_quality: float = 0.0  # Teacher 品質スコア
    selection_rate: float = 0.5   # 選択率（実績なし → 0.5 デフォルト）
    confidence: float = 0.5       # 信頼度スコア

    def to_vector(self) -> NDArray[np.float32]:
        """特徴量を numpy 配列に変換する。"""
        return np.array([
            max(0.0, self.faiss_score),  # 負値はクランプ
            self.freshness,
            self.usefulness,
            self.teacher_quality,
            self.selection_rate,
            self.confidence,
        ], dtype=np.float32)

    @classmethod
    def from_dict(cls, data: dict) -> "RankFeatures":
        """辞書から RankFeatures を生成する。"""
        return cls(
            faiss_score=float(data.get("faiss_score", 0.0)),
            freshness=float(data.get("freshness", 0.5)),
            usefulness=float(data.get("usefulness", 0.5)),
            teacher_quality=float(data.get("teacher_quality", 0.0)),
            selection_rate=float(data.get("selection_rate", 0.5)),
            confidence=float(data.get("confidence", 0.5)),
        )


class LTRRanker:
    """線形 Learning to Rank モデル。

    重みベクトルと特徴量の内積をランキングスコアとする。
    SGD ベースのオンライン学習でリアルタイムに重みを更新する。

    Args:
        learning_rate: SGD の学習率。
        initial_weights: 初期重みベクトル (length = N_FEATURES)。
            省略時は特徴量の平均（一様重み）を使用。
        regularization: L2 正則化係数。過学習防止。
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        initial_weights: Optional[NDArray[np.float32]] = None,
        regularization: float = 0.001,
    ) -> None:
        if initial_weights is not None:
            if len(initial_weights) != N_FEATURES:
                raise ValueError(
                    f"initial_weights length must be {N_FEATURES}, got {len(initial_weights)}"
                )
            self._weights = np.array(initial_weights, dtype=np.float32)
        else:
            # デフォルト: 一様重み（正規化済み）
            self._weights = np.ones(N_FEATURES, dtype=np.float32) / N_FEATURES

        self._lr = learning_rate
        self._reg = regularization
        self._update_count = 0

    @property
    def weights(self) -> NDArray[np.float32]:
        """現在の重みベクトル（読み取り専用コピー）。"""
        return self._weights.copy()

    @property
    def update_count(self) -> int:
        """累積更新回数。"""
        return self._update_count

    def score(self, features: RankFeatures) -> float:
        """特徴量からランキングスコアを計算する。

        Returns:
            ランキングスコア（値域は制限なし、通常 0.0〜1.0 付近）。
        """
        vec = features.to_vector()
        raw = float(np.dot(self._weights, vec))
        # シグモイドで 0〜1 に変換
        return float(1.0 / (1.0 + np.exp(-raw * 4)))  # 4 倍でシグモイドの傾きを急にする

    def score_batch(self, features_list: list[RankFeatures]) -> list[float]:
        """複数特徴量のスコアを一括計算する。"""
        return [self.score(f) for f in features_list]

    def update(self, features: RankFeatures, reward: float) -> float:
        """SGD でモデルを更新する。

        Args:
            features: 入力特徴量。
            reward: 報酬シグナル。正値 → 重みを強化、負値 → 弱化。
                通常は 0.0（否定）または 1.0（肯定）を使用する。

        Returns:
            更新前後のスコア差分（デバッグ用）。
        """
        vec = features.to_vector()
        before = self.score(features)

        # SGD 更新: w += lr * reward * x - lr * reg * w
        gradient = reward * vec - self._reg * self._weights
        self._weights += self._lr * gradient

        self._update_count += 1
        after = self.score(features)

        logger.debug(
            "LTR update #%d: reward=%.3f score %.4f → %.4f",
            self._update_count, reward, before, after,
        )
        return after - before

    def rerank(
        self,
        items: list[tuple[str, RankFeatures]],
    ) -> list[tuple[str, float]]:
        """doc_id + 特徴量のリストをリランクする。

        Args:
            items: [(doc_id, features), ...] のリスト。

        Returns:
            スコア降順の [(doc_id, score), ...] リスト。
        """
        scored = [(doc_id, self.score(feat)) for doc_id, feat in items]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def save(self, path: str | Path) -> None:
        """モデルをファイルに保存する。

        Args:
            path: 保存先パス（pickle 形式）。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "weights": self._weights,
            "learning_rate": self._lr,
            "regularization": self._reg,
            "update_count": self._update_count,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("LTRRanker saved to %s (updates=%d)", path, self._update_count)

    @classmethod
    def load(cls, path: str | Path) -> "LTRRanker":
        """ファイルからモデルを復元する。

        Args:
            path: 保存済みファイルのパス。

        Returns:
            復元された LTRRanker インスタンス。
        """
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)

        ranker = cls(
            learning_rate=data["learning_rate"],
            initial_weights=data["weights"],
            regularization=data.get("regularization", 0.001),
        )
        ranker._update_count = data.get("update_count", 0)
        logger.info("LTRRanker loaded from %s (updates=%d)", path, ranker._update_count)
        return ranker
