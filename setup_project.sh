#!/bin/bash
# setup_project.sh — Claude Code引き継ぎ用プロジェクト初期化
# 
# 使い方:
#   1. このスクリプトをダウンロード
#   2. 計画書ファイル群を同じディレクトリに配置
#   3. bash setup_project.sh を実行
#   4. cd rag-faiss-llm && claude で Claude Code を起動

set -e

PROJECT_DIR="rag-faiss-llm"

echo "=== RAG × FAISS × LLM プロジェクト初期化 ==="

# プロジェクトディレクトリ作成
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# CLAUDE.md配置（Claude Codeが自動読み込み）
if [ -f "../CLAUDE.md" ]; then
    cp ../CLAUDE.md ./CLAUDE.md
    echo "✓ CLAUDE.md 配置完了"
fi

# ドキュメント配置
mkdir -p docs
for doc in project_plan_v4.md learning_patterns.md v4_idea_proposal.md rag_search_apis.md; do
    if [ -f "../$doc" ]; then
        cp "../$doc" "docs/$doc"
        echo "✓ docs/$doc 配置完了"
    fi
done

# ソースディレクトリ構造作成
mkdir -p src/{orchestrator,llm/providers,llm/prompt_templates,rag/retrievers,memory/{learning,scoring,maturation},training/{algorithms,adapters,rewards,evaluation},sandbox/templates,common}
mkdir -p tests/{unit,integration,fixtures}
mkdir -p data/{faiss_indices,adapters,training_logs}
mkdir -p scripts
mkdir -p configs

# __init__.py 配置
find src -type d -exec touch {}/__init__.py \;

# pyproject.toml 作成
cat > pyproject.toml << 'TOML'
[project]
name = "rag-faiss-llm"
version = "0.4.0"
description = "RAG × FAISS × LLM × Memory Environment Distillation System"
requires-python = ">=3.11"
dependencies = [
    # API
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic>=2.6.0",
    "pydantic-settings>=2.1.0",
    # LLM
    "anthropic>=0.40.0",
    "openai>=1.12.0",
    # Embedding + Vector Search
    "sentence-transformers>=2.3.0",
    "faiss-cpu>=1.7.4",
    "numpy>=1.26.0",
    # Database
    "aiosqlite>=0.19.0",
    # Container
    "docker>=7.0.0",
    # HTTP
    "httpx>=0.27.0",
    # Config
    "pyyaml>=6.0.1",
]

[project.optional-dependencies]
training = [
    # Student学習 (Phase 3)
    "torch>=2.2.0",
    "transformers>=4.38.0",
    "peft>=0.9.0",
    "trl>=0.7.0",
    "vllm>=0.3.0",
    "wandb>=0.16.0",
]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "testcontainers>=3.7.0",
    "ruff>=0.3.0",
    "mypy>=1.8.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
TOML

# デフォルト設定ファイル作成
cat > configs/default.yaml << 'YAML'
app:
  name: "rag-faiss-llm"
  version: "0.4.0"
  debug: false
  host: "0.0.0.0"
  port: 8000

embedding:
  model: "all-MiniLM-L6-v2"
  dim: 768
  batch_size: 64
YAML

cat > configs/faiss_config.yaml << 'YAML'
indices:
  code:
    dim: 768
    initial_type: "Flat"
    metric: "inner_product"
    scale_rules:
      - threshold: 100000
        migrate_to: "IVF1024,Flat"
      - threshold: 1000000
        migrate_to: "IVF4096,PQ48"
    nprobe: 32

  academic:
    dim: 768
    initial_type: "Flat"
    metric: "inner_product"
    scale_rules:
      - threshold: 100000
        migrate_to: "HNSW32"

  general:
    dim: 768
    initial_type: "Flat"
    metric: "inner_product"
    scale_rules:
      - threshold: 100000
        migrate_to: "IVF1024,Flat"
YAML

cat > configs/llm_config.yaml << 'YAML'
providers:
  anthropic:
    default_model: "claude-sonnet-4-20250514"
    haiku_model: "claude-haiku-4-5-20251001"
    api_key_env: "ANTHROPIC_API_KEY"

  openai:
    default_model: "gpt-4o"
    mini_model: "gpt-4o-mini"
    api_key_env: "OPENAI_API_KEY"

  ollama:
    base_url: "http://localhost:11434"
    default_model: "llama3.1:8b"

task_routing:
  query_parsing: "haiku"
  query_rewrite: "haiku"
  hyde: "haiku"
  verification: "sonnet"
  response_generation: "sonnet"
  code_generation: "sonnet"
  error_analysis: "sonnet"
  feedback_analysis: "haiku"

budget:
  daily_limit_usd: 10.0
  alert_threshold: 0.8
  fallback_to_local: true
YAML

cat > configs/sandbox_policy.yaml << 'YAML'
sandbox:
  max_containers: 5
  timeout_seconds: 30
  resource_limits:
    memory: "256m"
    cpu: "0.5"
    pids: 100
  network:
    enabled: false
    allowed_domains: []
  filesystem:
    read_only_root: true
    writable_tmp: true
    max_file_size_mb: 10
  blocked:
    syscalls: [mount, ptrace, reboot]
YAML

cat > configs/training.yaml << 'YAML'
default:
  algorithm: "grpo"
  algorithm_kwargs:
    group_size: 8
    kl_coeff: 0.001
    clip_ratio: 0.2
    temperature: 1.0

  adapter: "tinylora"
  adapter_kwargs:
    frozen_rank: 2
    projection_dim: 4
    tie_factor: 7
    target_modules:
      - "q_proj"
      - "k_proj"
      - "v_proj"
      - "o_proj"
      - "up_proj"
      - "down_proj"
      - "gate_proj"

  reward: "composite"
  reward_kwargs:
    weights:
      correctness: 0.35
      retrieval_quality: 0.20
      exec_success: 0.20
      efficiency: 0.10
      memory_utilization: 0.15

  training:
    epochs: 3
    batch_size: 64
    learning_rate: 1.0e-4
    max_generation_length: 4096
    seeds: [42, 123, 456]

  student_model:
    name: "Qwen/Qwen2.5-7B-Instruct"
    inference_engine: "vllm"

alternatives:
  ppo_lora:
    algorithm: "ppo"
    adapter: "lora"
    adapter_kwargs:
      rank: 8

  dpo_loraxs:
    algorithm: "dpo"
    adapter: "lora_xs"
YAML

cat > configs/model_router.yaml << 'YAML'
router:
  thresholds:
    simple:
      min_faiss_score: 0.85
      min_results: 3
    moderate:
      min_faiss_score: 0.60
    complex:
      fallback: true

  cost_weights:
    prefer_student: 0.7
    prefer_teacher: 0.3
YAML

# .gitignore
cat > .gitignore << 'GITIGNORE'
__pycache__/
*.pyc
.env
data/faiss_indices/*.index
data/metadata.db
data/adapters/*.bin
data/training_logs/
.venv/
dist/
*.egg-info/
.mypy_cache/
.pytest_cache/
wandb/
GITIGNORE

# .env.example
cat > .env.example << 'ENV'
ANTHROPIC_API_KEY=sk-ant-xxxxx
OPENAI_API_KEY=sk-xxxxx
GITHUB_TOKEN=ghp_xxxxx
TAVILY_API_KEY=tvly-xxxxx
ENV

echo ""
echo "=== 初期化完了 ==="
echo ""
echo "次のステップ:"
echo "  1. cd $PROJECT_DIR"
echo "  2. cp .env.example .env  # APIキーを設定"
echo "  3. claude                 # Claude Codeを起動"
echo ""
echo "Claude Codeへの最初の指示例:"
echo '  "CLAUDE.mdを読んでプロジェクト構造を理解し、Phase 1のStep 1から実装を開始してください"'
echo ""
