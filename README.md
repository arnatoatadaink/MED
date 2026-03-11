# MED — RAG × FAISS × LLM × Memory Environment Distillation

> **Teacher Model（大モデルAPI）が FAISS メモリを成熟させ、Student Model（ローカル小モデル）に GRPO+TinyLoRA で「メモリの使い方」を極少パラメータで教えるシステム。**

Knowledge Graph (KG) を FAISS（連想記憶）と SQL/BI（宣言的記憶）の橋渡し層として導入し、エンティティ間の関係性（因果・階層・時系列）を活用して推論品質を向上させる。

---

## 着想元

| 論文 / プロジェクト | ポイント |
|---|---|
| [TinyLoRA](https://arxiv.org/abs/2xxx) (Morris et al., 2026) | 13 パラメータで GSM8K 91% 達成。「知識は既にある、使い方だけ RL で教える」 |
| [GraphRAG](https://arxiv.org/abs/2404.16130) (Microsoft, 2024) | Vector + KG 統合の基本設計 |
| [HippoRAG](https://arxiv.org/abs/2405.14831) (2024) | 海馬モデルの RAG 実装、誤想起軽減 |
| [Self-RAG](https://arxiv.org/abs/2310.11511) (2023) | 検索ルーティングの判断ロジック |
| RRF | Reciprocal Rank Fusion ベースの Fusion/Rerank |

記憶モデルの対応:
- **FAISS** = 海馬（エピソード記憶・連想）
- **Knowledge Graph** = 概念地図（意味記憶・関係性）
- **SQL/BI** = ノート（宣言的記憶・正確検索）

---

## アーキテクチャ

```
Client ──→ Gradio WebGUI / CLI
               │
               ▼
       Orchestrator (FastAPI)  ←→  QueryParser (LLM intent)
               │
       ModelRouter (Graph-aware)
               │
   ┌───────────┼───────────┐
   ▼           ▼           ▼
simple      moderate    complex
   │           │           │
Student     Student +   Teacher
(Qwen-7B    FAISS+RAG  (Claude/GPT)
+TinyLoRA)     │           │
   │           └─────┬─────┘
   ▼                 ▼
            ┌─────────────────────┐
            │     MED RAG Layer    │
            │                     │
            │  FAISS ←──┐          │
            │  (意味検索)  │  Fusion  │
            │       KG Bridge      │
            │  Knowledge Graph     │
            │  (関係・構造) ←──┐    │
            │       SQL / BI       │
            │  (正確検索)          │
            └────────┬────────────┘
                     ▼
             Docker Sandbox (コード実行)
```

---

## 実装状況

### 全体進捗

| フェーズ | 内容 | 状態 |
|---------|------|------|
| **Phase 1** | コアインフラ (Config → FastAPI) | ✅ **完了** |
| **Phase 1.5** | Knowledge Graph プロトタイプ | ✅ **完了** |
| **Phase 2** | メモリ成熟 + MCP Tools + Fusion | ✅ **完了** |
| **Phase 3** | 学習フレームワーク (GRPO+TinyLoRA) | ✅ **完了** |
| **Phase 4** | Orchestrator Routing + 評価 + LLM ユーティリティ | ✅ **完了** |
| **Phase 5** | 共通ユーティリティ + CLI スクリプト | ✅ **完了** |
| **Phase 2+** | メモリ品質目標達成 (10k docs) | ⬜ 運用フェーズ |
| **Phase 4** | Neo4j 移行・本番最適化 | ⬜ 未着手 |

### モジュール一覧

#### Phase 1 — コアインフラ

| ファイル | 概要 |
|---------|------|
| `src/common/config.py` | pydantic-settings + YAML 統合設定管理 |
| `src/memory/schema.py` | `Document`, `UsefulnessScore`, `SourceMeta` データクラス |
| `src/memory/embedder.py` | sentence-transformers ラッパー (all-MiniLM-L6-v2) |
| `src/memory/faiss_index.py` | ドメイン別 FAISS 管理 (Flat → IVF → PQ 自動切替) |
| `src/memory/metadata_store.py` | SQLite CRUD + 有用性カラム (aiosqlite) |
| `src/memory/memory_manager.py` | FAISS-SQLite 原子的操作 + KG 登録フック |
| `src/memory/scoring/freshness.py` | ドメイン別指数減衰フレッシュネス |
| `src/memory/scoring/usefulness.py` | 多面的有用性スコア |
| `src/memory/scoring/composite_scorer.py` | 複合スコア合成 |
| `src/memory/learning/ltr_ranker.py` | 線形 LTR (SGD オンライン学習) |
| `src/memory/iterative_retrieval.py` | マルチホップ (ベクトル加算 → LLM リライト + HyDE) |
| `src/llm/gateway.py` | LLM プロバイダ抽象化 |
| `src/llm/providers/` | Anthropic / OpenAI / Ollama / vLLM |
| `src/llm/response_generator.py` | RAG 付き応答生成 |
| `src/llm/code_generator.py` | コード生成 + サンドボックス連携 |
| `src/rag/retriever.py` | Retriever Router (基底クラス) |
| `src/rag/retrievers/` | GitHub / StackOverflow / Tavily / arXiv |
| `src/rag/verifier.py` | LLM ベース裏どり |
| `src/rag/chunker.py` | テキストチャンク分割 |
| `src/sandbox/manager.py` | Docker コンテナ管理 |
| `src/sandbox/executor.py` | コード実行 |
| `src/sandbox/security.py` | セキュリティ制限 |
| `src/orchestrator/server.py` | FastAPI エントリーポイント |
| `src/orchestrator/pipeline.py` | MEDPipeline (全モジュール統合) |

#### Phase 1.5 — Knowledge Graph

| ファイル | 概要 |
|---------|------|
| `src/knowledge_graph/store.py` | KnowledgeGraphStore (NetworkX バックエンド) |
| `src/knowledge_graph/extractor.py` | EntityExtractor (Teacher API 呼び出し) |
| `src/knowledge_graph/router_bridge.py` | ModelRouter ↔ KG 接続 |

#### Phase 2 — メモリ成熟 + MCP + Fusion

| ファイル | 概要 |
|---------|------|
| `src/memory/maturation/difficulty_tagger.py` | LLM 難易度分類 (beginner/intermediate/advanced/expert) |
| `src/memory/maturation/reviewer.py` | Teacher 品質審査 + MetadataStore 更新 |
| `src/memory/maturation/seed_builder.py` | Teacher 生成トレーニングデータ構築 |
| `src/memory/learning/cross_encoder.py` | LLM-as-reranker Cross-Encoder |
| `src/mcp_tools/sql_query_tool.py` | 自然言語 → SQL → 実行 (SELECT only) |
| `src/mcp_tools/bi_aggregation_tool.py` | COUNT/SUM/AVG/MIN/MAX 集計クエリ |
| `src/retrieval/query_classifier.py` | SEMANTIC/FACTUAL/RELATIONAL/HYBRID 分類 |
| `src/retrieval/fusion_reranker.py` | RRF ベース Fusion/Rerank |

#### Phase 3 — 学習フレームワーク

| ファイル | 概要 |
|---------|------|
| `src/training/base.py` | `TrainingAlgorithm` / `ParameterAdapter` / `RewardFunction` 抽象 IF |
| `src/training/registry.py` | デコレータ登録式 `TrainingRegistry` |
| `src/training/pipeline.py` | 3 段階学習 (SFT warmup → GRPO+TinyLoRA → Eval) |
| `src/training/algorithms/grpo.py` | GRPO (Group-Relative Policy Optimization) |
| `src/training/algorithms/ppo.py` | PPO |
| `src/training/algorithms/dpo.py` | DPO |
| `src/training/algorithms/sft.py` | SFT |
| `src/training/adapters/tinylora.py` | TinyLoRA (frozen_rank=2, projection_dim=4, 13 params) |
| `src/training/adapters/lora.py` | 標準 LoRA |
| `src/training/adapters/lora_xs.py` | LoRA-XS |
| `src/training/adapters/full_ft.py` | フルファインチューニング |
| `src/training/rewards/composite.py` | 複合 Reward (correctness 0.35 + retrieval 0.20 + exec 0.20 + efficiency 0.10 + memory 0.15) |
| `src/training/rewards/code_exec.py` | コード実行成功報酬 |
| `src/training/rewards/teacher_eval.py` | Teacher 評価報酬 |
| `src/training/logger.py` | 学習ログ (W&B 対応) |

#### Phase 4 — Orchestrator + 評価 + LLM ユーティリティ

| ファイル | 概要 |
|---------|------|
| `src/orchestrator/query_parser.py` | LLM クエリ解析 (intent/domain/complexity/entities) |
| `src/orchestrator/model_router.py` | Graph-aware ルーター (simple→Student, complex→Teacher) |
| `src/training/evaluation/student_evaluator.py` | Student 評価 (answer_quality/exec/retrieval_accuracy) |
| `src/training/evaluation/teacher_comparison.py` | Teacher vs Student 比較・win_rate |
| `src/training/evaluation/benchmark_suite.py` | ベンチマーク (code_generation/qa_retrieval/math_reasoning) |
| `src/llm/usage_tracker.py` | トークン + USD コスト追跡 |
| `src/llm/prompt_cache.py` | TTL + LRU プロンプトキャッシュ |
| `src/llm/error_analyzer.py` | regex + LLM エラー解析 + 自動修正 |
| `src/llm/feedback_analyzer.py` | LLM センチメント解析 + `memory_delta` |
| `src/memory/deduplicator.py` | 2 段階重複除去 (SHA-256 exact + cosine near-dup) |
| `src/memory/maturation/quality_metrics.py` | Phase 2 品質メトリクス + 目標達成チェック |

#### Phase 5 — 共通ユーティリティ + CLI

| ファイル | 概要 |
|---------|------|
| `src/common/logger.py` | 構造化ロギング (text/JSON 形式切替) |
| `src/common/models.py` | Pydantic API モデル (QueryRequest/Response, HealthResponse 等) |
| `src/sandbox/retry_handler.py` | 実行リトライ + LLM 自動修正ループ |
| `src/memory/learning/embedding_adapter.py` | 線形埋め込みアダプタ (SGD オンライン更新) |
| `src/memory/learning/feedback_collector.py` | フィードバック収集 (click/thumbs/rating/text) |
| `scripts/seed_memory.py` | RAG 検索 → FAISS シード投入 CLI |
| `scripts/mature_memory.py` | Teacher 審査 + 難易度タグ + 品質チェック CLI |
| `scripts/train_student.py` | GRPO+TinyLoRA 学習パイプライン CLI |
| `scripts/evaluate_student.py` | ベンチマーク評価 + Teacher 比較 CLI |

#### GUI (別セッション実装済み)

| ファイル | 概要 |
|---------|------|
| `src/gui/app.py` | Gradio Blocks アセンブリ |
| `src/gui/tabs/chat.py` | RAG + LLM クエリタブ |
| `src/gui/tabs/memory.py` | FAISS メモリ管理タブ |
| `src/gui/tabs/sandbox.py` | コードエディタ + 実行タブ |
| `src/gui/tabs/training.py` | 学習制御 + 可視化タブ |
| `src/gui/tabs/settings.py` | API キー・YAML 設定タブ |

---

## テスト

| テストファイル | 対象 | テスト数 |
|---|---|---|
| `test_config.py` | `src/common/config.py` | 53 |
| `test_schema.py` | `src/memory/schema.py` | — |
| `test_embedder.py` | `src/memory/embedder.py` | — |
| `test_faiss_index.py` | `src/memory/faiss_index.py` | — |
| `test_metadata_store.py` | `src/memory/metadata_store.py` | — |
| `test_memory_manager.py` | `src/memory/memory_manager.py` | — |
| `test_scoring.py` | `src/memory/scoring/` | — |
| `test_ltr_ranker.py` | `src/memory/learning/ltr_ranker.py` | — |
| `test_iterative_retrieval.py` | `src/memory/iterative_retrieval.py` | — |
| `test_llm_gateway.py` | `src/llm/gateway.py` + providers | — |
| `test_llm_generators.py` | response_generator / code_generator | — |
| `test_rag.py` | `src/rag/` 全体 | — |
| `test_sandbox.py` | `src/sandbox/` 全体 | — |
| `test_orchestrator.py` | server / pipeline | — |
| `test_knowledge_graph.py` | `src/knowledge_graph/` | — |
| `test_maturation.py` | difficulty_tagger / reviewer / seed_builder | 33 |
| `test_phase2.py` | cross_encoder / MCP tools / fusion / classifier | 56 |
| `test_training.py` | base / registry / algorithms / adapters / rewards / pipeline | 64 (+22 skip) |
| `test_phase4.py` | query_parser / model_router / evaluators / LLM utils / deduplicator | 103 |
| `test_phase5.py` | logger / models / retry_handler / embedding_adapter / feedback_collector | 48 |

```
合計: 357 passed, 22 skipped (torch 未インストール環境でのスキップ)
```

---

## 技術スタック

| レイヤー | 技術 |
|---------|------|
| API | FastAPI + uvicorn (非同期) |
| Web GUI | Gradio 5 タブ構成 |
| Teacher LLM | Anthropic SDK / OpenAI SDK |
| Student 推論 | vLLM |
| Student 学習 | VERL (GRPO) / trl (PPO/DPO) |
| アダプタ | TinyLoRA (自作) + peft (LoRA) |
| 埋め込み | sentence-transformers (all-MiniLM-L6-v2) |
| ベクトル検索 | faiss-cpu |
| Knowledge Graph | NetworkX (Phase 1.5) → Neo4j (Phase 2+) |
| メタデータ | SQLite (aiosqlite) |
| SQL/BI | SQLite → PostgreSQL (将来) |
| Fusion/Rerank | Reciprocal Rank Fusion (RRF) |
| コンテナ | docker-py |
| 設定 | pydantic-settings + YAML |
| テスト | pytest |
| 学習ログ | Weights & Biases |

---

## セットアップ

### 必要条件

- Python 3.11+
- Docker (サンドボックス実行用)

### インストール

```bash
# 基本依存
pip install -e .

# 学習フレームワーク (PyTorch 必要)
pip install -e ".[training]"
```

### 環境変数

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export TAVILY_API_KEY=tvly-...
```

または `.env` ファイルに記載:

```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

---

## 使い方

### Web GUI 起動

```bash
python scripts/launch_gui.py
# → http://localhost:7860
```

### API サーバー起動

```bash
uvicorn src.orchestrator.server:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Compose (全サービス)

```bash
docker compose up
```

### CLI スクリプト

```bash
# FAISSメモリにドキュメントを投入
python scripts/seed_memory.py --query "FAISS vector search" --domain code
python scripts/seed_memory.py --input-file docs.json

# Teacher審査 + 品質チェック
python scripts/mature_memory.py --all
python scripts/mature_memory.py --check

# Student学習
python scripts/train_student.py --algorithm grpo --adapter tinylora --steps 200

# 評価
python scripts/evaluate_student.py --benchmarks qa_retrieval code_generation
python scripts/evaluate_student.py --compare-teacher --output results.json
```

### API リクエスト例

```bash
# クエリ
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I implement FAISS index?", "use_sandbox": true}'

# メモリ検索
curl -X POST http://localhost:8000/memory/search \
  -d '{"query": "FAISS IVF", "top_k": 5}'

# ヘルスチェック
curl http://localhost:8000/health
```

---

## ディレクトリ構造

```
MED/
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.sandbox
├── Dockerfile.student
├── pyproject.toml
├── CLAUDE.md
├── plan.md
│
├── src/
│   ├── common/          # config, logger, models
│   ├── llm/             # gateway, providers, generators, utilities
│   ├── rag/             # retriever router + 4 sources + verifier + chunker
│   ├── memory/          # FAISS, SQLite, scoring, learning, maturation, deduplicator
│   ├── knowledge_graph/ # NetworkX KG store + entity extractor + router bridge
│   ├── retrieval/       # query classifier + RRF fusion reranker
│   ├── mcp_tools/       # SQL query tool + BI aggregation tool
│   ├── orchestrator/    # FastAPI server + query parser + model router + pipeline
│   ├── sandbox/         # Docker manager + executor + security + retry handler
│   ├── training/        # algorithms, adapters, rewards, pipeline, evaluation
│   └── gui/             # Gradio 5-tab WebGUI
│
├── tests/
│   └── unit/            # 20 テストファイル、357 passed / 22 skipped
│
├── scripts/             # seed_memory / mature_memory / train_student / evaluate_student / launch_gui
│
└── configs/             # default.yaml / llm_config.yaml / faiss_config.yaml 等
```

---

## 設計原則

- **抽象 IF ファースト**: 全モジュールはインターフェース → 実装の順で作成
- **Strategy + Registry**: 学習アルゴリズム/アダプタ/Reward はデコレータ登録で拡張 (`@TrainingRegistry.algorithm("grpo")`)
- **FAISS-SQLite 同期**: `memory_manager.py` で原子的操作を保証
- **段階的導入**: Phase ごとに動く MVP を維持しながら機能追加
- **テスト駆動**: 各モジュールに単体テストを必ず用意
- **KG は橋渡しに徹する**: KG 単独で答えを出さず、ルーティングと融合を担う
- **コスト意識**: GraphDB は最小構成 (NetworkX) から開始し Phase 2+ で Neo4j 移行

---

## 今後の予定

| 項目 | 内容 |
|------|------|
| Neo4j 移行 | NetworkX → Neo4j (Phase 2+ KG 永続化) |
| メモリ品質目標 | 10,000 docs, confidence > 0.7, exec > 80% の達成 |
| vLLM Student 推論 | Qwen2.5-7B-Instruct + TinyLoRA 実機検証 |
| PPO / DPO 実験 | GRPO との比較評価 |
| Cross-Encoder 強化 | Phase 2 本格 reranker 導入 |
| PostgreSQL 移行 | SQLite → PostgreSQL (本番 SQL/BI) |
| CI/CD | GitHub Actions テスト自動化 |
