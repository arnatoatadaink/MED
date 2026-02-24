"""src/common/config.py — pydantic-settings + YAML 統合設定管理

優先順位（高→低）:
  1. 環境変数 (ANTHROPIC_API_KEY 等)
  2. .env ファイル
  3. configs/*.yaml ファイル
  4. デフォルト値

使い方:
    from src.common.config import get_settings

    settings = get_settings()
    api_key = settings.anthropic_api_key
    model   = settings.llm.anthropic.default_model
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Tuple, Type

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# プロジェクトルート (src/common/config.py の 3 つ上)
_PROJECT_ROOT = Path(__file__).parent.parent.parent


# ============================================================================
# アプリケーション設定
# ============================================================================


class AppConfig(BaseModel):
    """FastAPI サーバー基本設定。"""

    name: str = "rag-faiss-llm"
    version: str = "0.4.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"


# ============================================================================
# 埋め込みモデル設定
# ============================================================================


class EmbeddingConfig(BaseModel):
    """sentence-transformers 埋め込みモデル設定。"""

    model: str = "all-MiniLM-L6-v2"
    dim: int = 768
    batch_size: int = 64
    device: str = "cpu"
    cache_dir: Optional[Path] = None


# ============================================================================
# FAISS メモリ設定
# ============================================================================


class FAISSScaleRule(BaseModel):
    """ドキュメント数に応じたインデックス自動移行ルール。"""

    threshold: int
    migrate_to: str


class FAISSIndexConfig(BaseModel):
    """ドメイン別 FAISS インデックス設定。

    Phase 1: IndexFlatIP (Flat) → 10万件超で IVF → 100万件超で PQ 自動移行。
    """

    dim: int = 768
    initial_type: str = "Flat"  # "Flat" | "IVF1024,Flat" | "HNSW32" etc.
    metric: str = "inner_product"  # "inner_product" | "l2"
    nprobe: int = 32
    scale_rules: list[FAISSScaleRule] = []


class FAISSConfig(BaseModel):
    """FAISS 全体設定。"""

    base_dir: Path = Path("data/faiss_indices")
    indices: dict[str, FAISSIndexConfig] = {
        "code": FAISSIndexConfig(),
        "academic": FAISSIndexConfig(),
        "general": FAISSIndexConfig(),
    }


# ============================================================================
# SQLite メタデータ設定
# ============================================================================


class MetadataConfig(BaseModel):
    """SQLite メタデータストア設定。"""

    db_path: Path = Path("data/metadata.db")


# ============================================================================
# LLM プロバイダ設定
# ============================================================================


class AnthropicConfig(BaseModel):
    """Anthropic Claude 設定。API キーは Settings.anthropic_api_key で管理。"""

    default_model: str = "claude-sonnet-4-20250514"
    haiku_model: str = "claude-haiku-4-5-20251001"
    max_retries: int = 3
    timeout_seconds: float = 60.0


class OpenAIConfig(BaseModel):
    """OpenAI GPT 設定。API キーは Settings.openai_api_key で管理。"""

    default_model: str = "gpt-4o"
    mini_model: str = "gpt-4o-mini"
    base_url: Optional[str] = None
    max_retries: int = 3
    timeout_seconds: float = 60.0


class OllamaConfig(BaseModel):
    """Ollama ローカル LLM 設定。"""

    base_url: str = "http://localhost:11434"
    default_model: str = "llama3.1:8b"
    timeout_seconds: float = 120.0


class BudgetConfig(BaseModel):
    """LLM API コスト予算設定。"""

    daily_limit_usd: float = 10.0
    alert_threshold: float = 0.8  # daily_limit の 80% で警告
    fallback_to_local: bool = True  # 予算超過時にローカルモデルへフォールバック


# タスク種別とモデル階層のマッピング
_DEFAULT_TASK_ROUTING: dict[str, str] = {
    "query_parsing": "haiku",
    "query_rewrite": "haiku",
    "hyde": "haiku",
    "verification": "sonnet",
    "response_generation": "sonnet",
    "code_generation": "sonnet",
    "error_analysis": "sonnet",
    "feedback_analysis": "haiku",
}


class LLMConfig(BaseModel):
    """LLM プロバイダ統合設定。"""

    anthropic: AnthropicConfig = AnthropicConfig()
    openai: OpenAIConfig = OpenAIConfig()
    ollama: OllamaConfig = OllamaConfig()
    budget: BudgetConfig = BudgetConfig()
    task_routing: dict[str, str] = Field(default_factory=lambda: dict(_DEFAULT_TASK_ROUTING))


# ============================================================================
# RAG パイプライン設定
# ============================================================================


class RAGSourceConfig(BaseModel):
    """個別 RAG ソース設定。"""

    enabled: bool = True
    max_results: int = 5


class RAGConfig(BaseModel):
    """外部検索 RAG パイプライン設定。"""

    max_results_per_source: int = 5
    timeout_seconds: int = 30
    verify_results: bool = True
    chunk_size: int = 512
    chunk_overlap: int = 50


# ============================================================================
# Docker Sandbox 設定
# ============================================================================


class SandboxResourceLimits(BaseModel):
    """コンテナリソース制限。"""

    memory: str = "256m"
    cpu: str = "0.5"
    pids: int = 100


class SandboxConfig(BaseModel):
    """Docker Sandbox 実行環境設定。"""

    max_containers: int = 5
    timeout_seconds: int = 30
    resource_limits: SandboxResourceLimits = SandboxResourceLimits()
    network_enabled: bool = False
    allowed_domains: list[str] = []
    read_only_root: bool = True
    writable_tmp: bool = True
    max_file_size_mb: int = 10
    blocked_syscalls: list[str] = ["mount", "ptrace", "reboot"]


# ============================================================================
# Model Router 設定
# ============================================================================


class SimpleRouteThreshold(BaseModel):
    """simple ルーティングの閾値。FAISSスコアが高く結果数が十分なら Student へ。"""

    min_faiss_score: float = 0.85
    min_results: int = 3


class ModerateRouteThreshold(BaseModel):
    """moderate ルーティングの閾値。Student + 外部 RAG へ。"""

    min_faiss_score: float = 0.60


class RouterThresholds(BaseModel):
    simple: SimpleRouteThreshold = SimpleRouteThreshold()
    moderate: ModerateRouteThreshold = ModerateRouteThreshold()


class RouterConfig(BaseModel):
    """Model Router 設定。クエリ複雑度に応じた Teacher/Student 振り分け。"""

    thresholds: RouterThresholds = RouterThresholds()
    prefer_student_weight: float = 0.7
    prefer_teacher_weight: float = 0.3


# ============================================================================
# Student 学習フレームワーク設定
# ============================================================================


class GRPOConfig(BaseModel):
    """GRPO (Group Relative Policy Optimization) ハイパーパラメータ。"""

    group_size: int = 8
    kl_coeff: float = 0.001
    clip_ratio: float = 0.2
    temperature: float = 1.0


class TinyLoRAConfig(BaseModel):
    """TinyLoRA アダプタ設定 (Morris et al., 2026)。

    frozen_rank=2, projection_dim=4, tie_factor=7 が論文推奨値。
    """

    frozen_rank: int = 2
    projection_dim: int = 4  # u: 各モジュールの学習次元
    tie_factor: int = 7  # n_tie: 重み共有モジュール数
    target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    ]


class RewardWeights(BaseModel):
    """複合 Reward 関数の重み。合計が 1.0 になること。

    correctness(0.35) + retrieval_quality(0.20) + exec_success(0.20)
    + efficiency(0.10) + memory_utilization(0.15) = 1.00
    """

    correctness: float = 0.35
    retrieval_quality: float = 0.20
    exec_success: float = 0.20
    efficiency: float = 0.10
    memory_utilization: float = 0.15

    @field_validator("memory_utilization")
    @classmethod
    def weights_sum_to_one(cls, v: float, info: Any) -> float:
        """全重みの合計が 1.0 であることを検証する。"""
        data = info.data
        total = (
            data.get("correctness", 0.35)
            + data.get("retrieval_quality", 0.20)
            + data.get("exec_success", 0.20)
            + data.get("efficiency", 0.10)
            + v
        )
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Reward weights must sum to 1.0, got {total:.4f}")
        return v


class TrainingRunConfig(BaseModel):
    """学習実行パラメータ。"""

    epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 1e-4
    max_generation_length: int = 4096
    seeds: list[int] = [42, 123, 456]


class StudentModelConfig(BaseModel):
    """Student ベースモデル設定。"""

    name: str = "Qwen/Qwen2.5-7B-Instruct"
    inference_engine: str = "vllm"  # "vllm" | "ollama" | "transformers"


class TrainingConfig(BaseModel):
    """学習フレームワーク設定。algorithm/adapter/reward は Registry 経由で差し替え可能。"""

    algorithm: str = "grpo"  # "grpo" | "ppo" | "dpo" | "reinforce" | "sft"
    adapter: str = "tinylora"  # "tinylora" | "lora" | "lora_xs" | "full_ft"
    reward: str = "composite"  # "composite" | "code_exec" | "teacher_eval" | "hybrid"
    algorithm_kwargs: GRPOConfig = GRPOConfig()
    adapter_kwargs: TinyLoRAConfig = TinyLoRAConfig()
    reward_weights: RewardWeights = RewardWeights()
    training: TrainingRunConfig = TrainingRunConfig()
    student_model: StudentModelConfig = StudentModelConfig()
    output_dir: Path = Path("data/adapters")
    logs_dir: Path = Path("data/training_logs")
    wandb_project: str = "rag-faiss-llm"


# ============================================================================
# メイン Settings — pydantic-settings で env var / .env を統合
# ============================================================================


class Settings(BaseSettings):
    """全設定の統合クラス。

    env var / .env ファイル → YAML → デフォルト値 の優先順位で設定を解決する。
    get_settings() 経由でシングルトンとして利用すること。

    Examples:
        >>> from src.common.config import get_settings
        >>> s = get_settings()
        >>> s.llm.anthropic.default_model
        'claude-sonnet-4-20250514'
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # ANTHROPIC_API_KEY → anthropic_api_key に自動マッチ
        extra="ignore",
        populate_by_name=True,  # alias と field name の両方で初期化を許可
    )

    # ── API キー (環境変数 / .env から読む) ──────────────────────────────
    # pydantic-settings は case_sensitive=False のため:
    #   anthropic_api_key フィールド ↔ ANTHROPIC_API_KEY 環境変数 が自動マッチ
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    github_token: str = ""
    tavily_api_key: str = ""
    stackoverflow_api_key: str = ""

    # ── 構造化設定 (YAML から load_settings() 経由で注入される) ─────────
    app: AppConfig = Field(default_factory=AppConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    faiss: FAISSConfig = Field(default_factory=FAISSConfig)
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """env var / .env を YAML (init_settings) より高優先にする。

        優先順位 (高 → 低):
            1. env_settings          — 環境変数
            2. dotenv_settings       — .env ファイル
            3. init_settings         — load_settings() が渡す YAML 由来の kwargs
            4. file_secret_settings  — /run/secrets 等
        """
        return (env_settings, dotenv_settings, init_settings, file_secret_settings)

    # ── ヘルパーメソッド ─────────────────────────────────────────────────

    def get_llm_model(self, task_type: str, provider: str = "anthropic") -> str:
        """タスク種別とプロバイダからモデル名を解決する。

        Args:
            task_type: "query_parsing" | "verification" | "response_generation" 等
            provider: "anthropic" | "openai"

        Returns:
            モデル名 (例: "claude-sonnet-4-20250514")
        """
        tier = self.llm.task_routing.get(task_type, "sonnet")
        if provider == "anthropic":
            if tier == "haiku":
                return self.llm.anthropic.haiku_model
            return self.llm.anthropic.default_model
        if provider == "openai":
            if tier in ("haiku", "mini"):
                return self.llm.openai.mini_model
            return self.llm.openai.default_model
        raise ValueError(f"Unknown provider: {provider!r}")

    def has_anthropic_key(self) -> bool:
        return bool(self.anthropic_api_key)

    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key)

    def has_github_token(self) -> bool:
        return bool(self.github_token)

    def has_tavily_key(self) -> bool:
        return bool(self.tavily_api_key)


# ============================================================================
# YAML ロードユーティリティ
# ============================================================================


def _load_yaml(path: Path) -> dict[str, Any]:
    """YAML ファイルを読み込む。ファイルが存在しない場合は空 dict を返す。"""
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """2つの dict を深くマージする (override が優先)。"""
    result: dict[str, Any] = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _build_init_kwargs(configs_dir: Path) -> dict[str, Any]:
    """configs/ ディレクトリ内の YAML ファイルを読み込み、

    Settings の constructor kwargs として使える dict を構築する。
    """
    default_raw = _load_yaml(configs_dir / "default.yaml")
    faiss_raw = _load_yaml(configs_dir / "faiss_config.yaml")
    llm_raw = _load_yaml(configs_dir / "llm_config.yaml")
    sandbox_raw = _load_yaml(configs_dir / "sandbox_policy.yaml")
    training_raw = _load_yaml(configs_dir / "training.yaml")
    router_raw = _load_yaml(configs_dir / "model_router.yaml")
    rag_raw = _load_yaml(configs_dir / "retrievers.yaml")

    kwargs: dict[str, Any] = {}

    # ── default.yaml → app, embedding ────────────────────────────────────
    if "app" in default_raw:
        kwargs["app"] = default_raw["app"]
    if "embedding" in default_raw:
        kwargs["embedding"] = default_raw["embedding"]

    # ── faiss_config.yaml → faiss ─────────────────────────────────────────
    if faiss_raw:
        kwargs["faiss"] = faiss_raw

    # ── llm_config.yaml → llm ────────────────────────────────────────────
    if llm_raw:
        llm_data: dict[str, Any] = {}
        if "providers" in llm_raw:
            providers = llm_raw["providers"]
            if "anthropic" in providers:
                llm_data["anthropic"] = providers["anthropic"]
            if "openai" in providers:
                llm_data["openai"] = providers["openai"]
            if "ollama" in providers:
                llm_data["ollama"] = providers["ollama"]
        if "task_routing" in llm_raw:
            llm_data["task_routing"] = llm_raw["task_routing"]
        if "budget" in llm_raw:
            llm_data["budget"] = llm_raw["budget"]
        if llm_data:
            kwargs["llm"] = llm_data

    # ── retrievers.yaml → rag ─────────────────────────────────────────────
    if rag_raw:
        rag_data: dict[str, Any] = {}
        for key in ("max_results_per_source", "timeout_seconds", "verify_results",
                    "chunk_size", "chunk_overlap"):
            if key in rag_raw:
                rag_data[key] = rag_raw[key]
        if rag_data:
            kwargs["rag"] = rag_data

    # ── sandbox_policy.yaml → sandbox ────────────────────────────────────
    if sandbox_raw:
        sandbox_data: dict[str, Any] = {}
        for key in ("max_containers", "timeout_seconds", "resource_limits"):
            if key in sandbox_raw:
                sandbox_data[key] = sandbox_raw[key]
        if "network" in sandbox_raw:
            net = sandbox_raw["network"]
            sandbox_data["network_enabled"] = net.get("enabled", False)
            sandbox_data["allowed_domains"] = net.get("allowed_domains", [])
        if "filesystem" in sandbox_raw:
            fs = sandbox_raw["filesystem"]
            sandbox_data["read_only_root"] = fs.get("read_only_root", True)
            sandbox_data["writable_tmp"] = fs.get("writable_tmp", True)
            sandbox_data["max_file_size_mb"] = fs.get("max_file_size_mb", 10)
        if "blocked" in sandbox_raw:
            sandbox_data["blocked_syscalls"] = sandbox_raw["blocked"].get("syscalls", [])
        if sandbox_data:
            kwargs["sandbox"] = sandbox_data

    # ── training.yaml → training (default セクション) ────────────────────
    if "default" in training_raw:
        dt = training_raw["default"]
        training_data: dict[str, Any] = {}
        for key in ("algorithm", "adapter", "reward", "output_dir", "logs_dir",
                    "wandb_project"):
            if key in dt:
                training_data[key] = dt[key]
        if "algorithm_kwargs" in dt:
            training_data["algorithm_kwargs"] = dt["algorithm_kwargs"]
        if "adapter_kwargs" in dt:
            training_data["adapter_kwargs"] = dt["adapter_kwargs"]
        if "reward_kwargs" in dt and "weights" in dt["reward_kwargs"]:
            training_data["reward_weights"] = dt["reward_kwargs"]["weights"]
        if "training" in dt:
            training_data["training"] = dt["training"]
        if "student_model" in dt:
            training_data["student_model"] = dt["student_model"]
        if training_data:
            kwargs["training"] = training_data

    # ── model_router.yaml → router ────────────────────────────────────────
    if router_raw:
        router_data: dict[str, Any] = {}
        if "thresholds" in router_raw:
            thresholds = router_raw["thresholds"]
            router_data["thresholds"] = {}
            if "simple" in thresholds:
                router_data["thresholds"]["simple"] = thresholds["simple"]
            if "moderate" in thresholds:
                router_data["thresholds"]["moderate"] = thresholds["moderate"]
        if "cost_weights" in router_raw:
            cw = router_raw["cost_weights"]
            router_data["prefer_student_weight"] = cw.get("prefer_student", 0.7)
            router_data["prefer_teacher_weight"] = cw.get("prefer_teacher", 0.3)
        if router_data:
            kwargs["router"] = router_data

    return kwargs


# ============================================================================
# 公開 API
# ============================================================================


def load_settings(configs_dir: Optional[Path] = None) -> Settings:
    """YAML ファイルと環境変数から Settings を構築して返す。

    Args:
        configs_dir: configs/ ディレクトリのパス。省略時はプロジェクトルートの configs/ を使用。

    Returns:
        構築済みの Settings インスタンス。
    """
    if configs_dir is None:
        configs_dir = _PROJECT_ROOT / "configs"

    init_kwargs = _build_init_kwargs(configs_dir)
    return Settings(**init_kwargs)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """シングルトン Settings インスタンスを返す。

    アプリケーション全体でこの関数を通じて設定を取得すること。
    テスト時は get_settings.cache_clear() でキャッシュをリセット可能。

    Returns:
        グローバルな Settings インスタンス。
    """
    return load_settings()
