"""src/memory/schema.py — FAISS メモリシステムのコアデータモデル

全モジュール（faiss_index, metadata_store, memory_manager, scoring, maturation,
training rewards）が共有するスキーマを定義する。

使い方:
    from src.memory.schema import Document, SearchResult, SourceType

    doc = Document(
        content="Pythonでリストをソートするには sorted() を使う",
        domain="code",
        source=SourceMeta(source_type=SourceType.STACKOVERFLOW, url="..."),
    )
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator

# ============================================================================
# 列挙型
# ============================================================================


class SourceType(str, Enum):
    """ドキュメントの取得元。"""

    GITHUB = "github"
    STACKOVERFLOW = "stackoverflow"
    TAVILY = "tavily"
    ARXIV = "arxiv"
    MANUAL = "manual"  # 手動投入
    TEACHER = "teacher"  # Teacher Model 生成
    SEED = "seed"  # シードデータ


class DifficultyLevel(str, Enum):
    """ドキュメントの難易度。Student 学習のカリキュラムに利用。

    beginner → intermediate → advanced → expert の順に提示する。
    """

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class Domain(str, Enum):
    """FAISS インデックスのドメイン分類。

    faiss_config.yaml の indices キーと対応する。
    """

    CODE = "code"
    ACADEMIC = "academic"
    GENERAL = "general"


class ReviewStatus(str, Enum):
    """Teacher によるメモリレビューの状態。"""

    UNREVIEWED = "unreviewed"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_UPDATE = "needs_update"


# ============================================================================
# ソースメタデータ
# ============================================================================


# ============================================================================
# Teacher 素性ユーティリティ
# ============================================================================

# ── Teacher 素性キー（extra dict の規約） ──────────────────────────────────
# SourceMeta.extra に格納するキー名を定数で統一する。
# 将来 TeacherRegistry（Step 2）が参照するキーと対応する。
_TEACHER_ID_KEY = "teacher_id"        # 例: "claude-opus-4-6", "gpt-4o", "human"
_TEACHER_PROVIDER_KEY = "teacher_provider"  # 例: "anthropic", "openai", "ollama"
_TEACHER_MODEL_KEY = "teacher_model"   # provider と同一でも可; gateway が使う model 文字列

# モデル名プレフィックス → プロバイダのマッピング
_MODEL_PROVIDER_MAP: dict[str, str] = {
    "claude": "anthropic",
    "gpt": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "gemini": "google",
    "llama": "ollama",
    "qwen": "ollama",
    "mistral": "ollama",
}


def _infer_provider(teacher_id: str) -> str | None:
    """teacher_id のプレフィックスからプロバイダ名を推定する。

    例: ``"claude-opus-4-6"`` → ``"anthropic"``
    """
    lower = teacher_id.lower()
    for prefix, provider in _MODEL_PROVIDER_MAP.items():
        if lower.startswith(prefix):
            return provider
    return None


class SourceMeta(BaseModel):
    """ドキュメントの取得元情報。

    Teacher 素性は ``extra`` dict に格納する（スキーマ変更なし）。
    規約化されたキーは ``_TEACHER_*_KEY`` 定数を使い、
    ``set_teacher`` / ``get_teacher_id`` ヘルパーで読み書きする。

    例::

        meta = SourceMeta(source_type=SourceType.TEACHER)
        meta.set_teacher("claude-opus-4-6", provider="anthropic")
        print(meta.teacher_id)  # "claude-opus-4-6"
    """

    source_type: SourceType = SourceType.MANUAL
    url: str | None = None
    title: str | None = None
    author: str | None = None
    language: str | None = None  # プログラミング言語 (code ドメイン向け)
    tags: list[str] = Field(default_factory=list)
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)
    extra: dict[str, Any] = Field(default_factory=dict)

    # ── Teacher 素性ヘルパー ────────────────────────────────────────────────

    def set_teacher(
        self,
        teacher_id: str,
        provider: str | None = None,
        model: str | None = None,
    ) -> SourceMeta:
        """Teacher 素性を extra dict に書き込む（メソッドチェーン可）。

        Args:
            teacher_id: モデル識別子。例: ``"claude-opus-4-6"``。
            provider:   プロバイダ名。例: ``"anthropic"``。省略時は teacher_id から推定。
            model:      gateway が使う model 文字列（teacher_id と異なる場合のみ指定）。
        """
        self.extra[_TEACHER_ID_KEY] = teacher_id
        resolved_provider = provider or _infer_provider(teacher_id)
        if resolved_provider:
            self.extra[_TEACHER_PROVIDER_KEY] = resolved_provider
        if model and model != teacher_id:
            self.extra[_TEACHER_MODEL_KEY] = model
        return self

    @property
    def teacher_id(self) -> str | None:
        """Teacher モデル識別子。未設定なら ``None``。"""
        return self.extra.get(_TEACHER_ID_KEY)

    @property
    def teacher_provider(self) -> str | None:
        """Teacher プロバイダ名。未設定なら ``None``。"""
        return self.extra.get(_TEACHER_PROVIDER_KEY)

    @property
    def is_teacher_generated(self) -> bool:
        """Teacher が生成したドキュメントか（source_type または extra で判定）。"""
        return (
            self.source_type in (SourceType.TEACHER, SourceType.SEED)
            or _TEACHER_ID_KEY in self.extra
        )


# ============================================================================
# 有用性スコア
# ============================================================================


class UsefulnessScore(BaseModel):
    """多面的な有用性スコア。scoring/ モジュールが計算・更新する。

    各スコアは 0.0 〜 1.0 の範囲。
    """

    retrieval_count: int = 0  # 検索でヒットした回数
    selection_count: int = 0  # 実際に回答生成に使用された回数
    positive_feedback: int = 0  # ユーザーからの正フィードバック
    negative_feedback: int = 0  # ユーザーからの負フィードバック
    teacher_quality: float = 0.0  # Teacher Model による品質評価 (0.0-1.0)
    execution_success_rate: float = 0.0  # コード実行成功率 (0.0-1.0)
    freshness: float = 1.0  # 鮮度スコア (0.0-1.0、時間減衰)
    composite: float = 0.0  # composite_scorer が計算する総合スコア (0.0-1.0)

    @field_validator("teacher_quality", "execution_success_rate", "freshness", "composite")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        """スコアを 0.0〜1.0 にクランプする。"""
        return max(0.0, min(1.0, v))

    @property
    def feedback_ratio(self) -> float:
        """正フィードバック率。フィードバックがなければ 0.0。"""
        total = self.positive_feedback + self.negative_feedback
        if total == 0:
            return 0.0
        return self.positive_feedback / total

    @property
    def selection_rate(self) -> float:
        """検索ヒット時に実際に選択された率。"""
        if self.retrieval_count == 0:
            return 0.0
        return self.selection_count / self.retrieval_count


# ============================================================================
# ドキュメント (コアエンティティ)
# ============================================================================


def _generate_doc_id() -> str:
    return uuid.uuid4().hex


class Document(BaseModel):
    """FAISS + SQLite に格納するドキュメント。

    - content: テキスト本体
    - embedding: FAISS に格納するベクトル (embedder.py が生成)
    - domain: FAISS インデックスの振り分け先
    - source: 取得元情報
    - usefulness: 有用性スコア (scoring/ が更新)
    - difficulty: 難易度タグ (maturation/ が付与)
    """

    model_config = {"arbitrary_types_allowed": True}

    # ── 識別子 ──
    id: str = Field(default_factory=_generate_doc_id)

    # ── コンテンツ ──
    content: str
    content_hash: str | None = None  # deduplicator.py が設定
    chunk_index: int = 0  # チャンク分割時のインデックス
    parent_id: str | None = None  # チャンク元ドキュメントの ID

    # ── 分類 ──
    domain: Domain = Domain.GENERAL

    # ── 埋め込みベクトル ──
    embedding: NDArray[np.float32] | None = None  # shape: (dim,)

    # ── メタデータ ──
    source: SourceMeta = Field(default_factory=SourceMeta)
    usefulness: UsefulnessScore = Field(default_factory=UsefulnessScore)

    # ── 品質管理 (maturation/) ──
    difficulty: DifficultyLevel | None = None
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED
    confidence: float = 0.5  # 総合信頼度 (0.0-1.0)

    # ── コード実行情報 (sandbox/) ──
    is_executable: bool = False  # コードブロックを含むか
    execution_verified: bool = False  # Sandbox で検証済みか
    last_execution_success: bool | None = None

    # ── タイムスタンプ ──
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    reviewed_at: datetime | None = None

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: NDArray[np.float32] | None) -> NDArray[np.float32] | None:
        """埋め込みベクトルが 1 次元であることを検証する。"""
        if v is not None and v.ndim != 1:
            raise ValueError(f"embedding must be 1-dimensional, got ndim={v.ndim}")
        return v


# ============================================================================
# 検索結果
# ============================================================================


class SearchResult(BaseModel):
    """FAISS 検索の結果 1 件。"""

    model_config = {"arbitrary_types_allowed": True}

    document: Document
    score: float  # FAISS の類似度スコア (inner product or L2)
    rank: int = 0  # 検索結果内での順位 (0-indexed)

    # Rerank 後のスコア (LTR / Cross-Encoder が設定)
    rerank_score: float | None = None


class SearchQuery(BaseModel):
    """検索リクエスト。iterative_retrieval.py 等が利用する。"""

    text: str
    domain: Domain | None = None  # None なら全ドメイン横断
    top_k: int = 5
    min_score: float = 0.0
    filters: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# コード実行結果
# ============================================================================


class ExecutionResult(BaseModel):
    """Docker Sandbox でのコード実行結果。"""

    success: bool
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    execution_time_ms: float = 0.0
    language: str = "python"
    error_type: str | None = None  # "SyntaxError", "RuntimeError" 等
    timed_out: bool = False


# ============================================================================
# Reward シグナル (Training Framework)
# ============================================================================


class RewardSignal(BaseModel):
    """学習フレームワーク用の報酬シグナル。

    composite.py 等の RewardFunction が生成する。
    """

    total: float  # 加重合計スカラー
    components: dict[str, float] = Field(default_factory=dict)  # 各要素の内訳
    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# 学習用バッチデータ
# ============================================================================


class TrainingBatch(BaseModel):
    """学習アルゴリズムに渡すバッチ。"""

    queries: list[str]
    reference_responses: list[str] | None = None  # SFT / DPO 用
    difficulty_levels: list[DifficultyLevel] | None = None  # カリキュラム学習用


class TrainStepResult(BaseModel):
    """1 ステップの学習結果。"""

    loss: float
    mean_reward: float = 0.0
    grad_norm: float | None = None
    learning_rate: float | None = None
    extra_metrics: dict[str, float] = Field(default_factory=dict)
