"""src/common/models.py — 共通 Pydantic モデル / レスポンス型

HTTP API / 内部バス共通の型定義を提供する。

使い方:
    from src.common.models import QueryRequest, QueryResponse, HealthResponse
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TargetModel(str, Enum):
    STUDENT = "student"
    TEACHER = "teacher"
    AUTO = "auto"


class ResponseFormat(str, Enum):
    TEXT = "text"
    CODE = "code"
    JSON = "json"


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """チャットクエリリクエスト。"""

    query: str = Field(..., min_length=1, description="ユーザークエリ")
    domain: Optional[str] = Field(None, description="ドメイン指定 (code/academic/general)")
    target: TargetModel = Field(TargetModel.AUTO, description="利用モデル指定")
    use_memory: bool = Field(True, description="FAISS メモリを使用するか")
    use_rag: bool = Field(True, description="外部 RAG を使用するか")
    use_sandbox: bool = Field(False, description="コード実行を有効にするか")
    max_tokens: int = Field(1024, ge=1, le=8192, description="最大出力トークン数")
    session_id: Optional[str] = Field(None, description="セッション ID")

    model_config = {"json_schema_extra": {"example": {"query": "Implement binary search in Python.", "use_sandbox": True}}}


class QueryResponse(BaseModel):
    """チャットクエリレスポンス。"""

    query: str
    answer: str
    model_used: str = ""
    domain: str = "general"
    retrieval_sources: list[str] = Field(default_factory=list)
    execution_result: Optional[str] = None
    tokens_used: int = 0
    latency_ms: float = 0.0
    session_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemorySearchRequest(BaseModel):
    """FAISSメモリ検索リクエスト。"""

    query: str = Field(..., min_length=1)
    domain: Optional[str] = None
    top_k: int = Field(5, ge=1, le=50)
    min_score: float = Field(0.0, ge=0.0, le=1.0)


class MemorySearchResult(BaseModel):
    """FAISSメモリ検索結果。"""

    doc_id: str
    content: str
    score: float
    domain: str
    difficulty: str = "unknown"
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemorySearchResponse(BaseModel):
    """FAISSメモリ検索レスポンス。"""

    query: str
    results: list[MemorySearchResult]
    total_found: int
    latency_ms: float = 0.0


class TrainingRequest(BaseModel):
    """学習開始リクエスト。"""

    algorithm: str = Field("grpo", description="学習アルゴリズム")
    adapter: str = Field("tinylora", description="パラメータアダプタ")
    n_steps: int = Field(100, ge=1, le=10000)
    batch_size: int = Field(8, ge=1, le=256)
    domain: Optional[str] = None
    notes: str = ""


class TrainingStatusResponse(BaseModel):
    """学習ステータスレスポンス。"""

    status: str  # "idle" | "running" | "completed" | "failed"
    algorithm: str = ""
    adapter: str = ""
    current_step: int = 0
    total_steps: int = 0
    avg_reward: float = 0.0
    elapsed_seconds: float = 0.0
    message: str = ""


class SandboxRequest(BaseModel):
    """サンドボックス実行リクエスト。"""

    code: str = Field(..., min_length=1, description="実行コード")
    language: str = Field("python", description="プログラミング言語")
    timeout_seconds: int = Field(10, ge=1, le=60)
    allow_network: bool = Field(False)


class SandboxResponse(BaseModel):
    """サンドボックス実行レスポンス。"""

    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    execution_time_ms: float = 0.0
    error_message: Optional[str] = None


class HealthResponse(BaseModel):
    """ヘルスチェックレスポンス。"""

    status: str = "ok"
    version: str = "0.1.0"
    components: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def healthy(cls, components: Optional[dict[str, str]] = None) -> "HealthResponse":
        return cls(status="ok", components=components or {})

    @classmethod
    def degraded(cls, reason: str) -> "HealthResponse":
        return cls(status="degraded", components={"error": reason})


class ErrorResponse(BaseModel):
    """エラーレスポンス。"""

    error: str
    detail: Optional[str] = None
    code: int = 500
