# CLAUDE.md — RAG × FAISS × LLM × Memory Environment Distillation

## プロジェクト概要

Teacher Model（大モデルAPI）がFAISSメモリを成熟させ、Student Model（ローカル小モデル）にGRPO+TinyLoRAで「メモリの使い方」を極少パラメータで教えるシステム。Docker Sandboxでコードを安全に実行・検証する。

Knowledge Graph（KG）をFAISS（連想記憶）とSQL/BI（宣言的記憶）の橋渡し層として導入し、Entity間の関係性（因果・階層・時系列）を活用して推論品質を向上させる。

### 着想元
- TinyLoRA論文 (Morris et al., 2026): 13パラメータでGSM8K 91%達成。「知識は既にある、使い方だけRLで教える」
- 本プロジェクト: 知識はFAISS外部メモリに蓄積。小モデルに検索・利用スキルだけRLで教える
- 記憶モデル: FAISS=海馬（エピソード記憶）、KG=概念地図（意味記憶）、SQL/BI=ノート（宣言的記憶）

## アーキテクチャ

```
Client ─→ Gradio WebGUI / CLI
              │
              ▼
      Orchestrator(FastAPI) → Graph-aware Model Router
                                  │
                  ┌───────────────┼───────────────┐
                  ▼               ▼               ▼
              simple          moderate          complex
                  │               │               │
      Student(Qwen-7B       Student +         Teacher(Claude/GPT)
       +TinyLoRA)           FAISS + 外部RAG       │
                  │               │               │
                  └───────┬───────┘               │
                          ▼                       ▼
                  ┌──────────────────────────────────┐
                  │         MED RAG Layer              │
                  │                                    │
                  │  FAISS ←──┐                        │
                  │  (意味検索)     │  Fusion / Rerank   │
                  │       │    KG Bridge               │
                  │       ▼        │                   │
                  │  Knowledge ────┘                   │
                  │  Graph                             │
                  │  (関係・構造) ←──┐                  │
                  │       │         │                  │
                  │       ▼         │                  │
                  │  SQL / BI    Structured Filter     │
                  │  (正確検索)                         │
                  └────────────┬───────────────────────┘
                               ▼
                       Docker Sandbox(コード実行)
```

### モジュール構成
1. **Orchestrator** — FastAPI、Graph-aware Model Router、Query Parser(LLM)
2. **RAG Pipeline** — GitHub/SO/Tavily検索、LLM裏どり、チャンク化
3. **FAISS Memory** — ベクトル検索、SQLiteメタデータ、Iterative Retrieval、LTR/Cross-Encoder、有用性管理
4. **Knowledge Graph** — Entity・関係性管理、検索ルーティング補助、Fusion/Rerank（★ 新規）
5. **MCP Tools** — SQL/BI 構造化クエリ（★ 新規）
6. **Docker Sandbox** — コード実行、セキュリティ制限、自動リトライ
7. **Training Framework** — GRPO+TinyLoRA(デフォルト)、PPO/DPO/SFT/LoRA/LoRA-XS(拡張可能)
8. **Web GUI** — Gradio 5タブ構成（Chat, FAISSメモリ, Sandbox, 学習, 設定）

## ディレクトリ構造

```
MED/
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.sandbox
├── Dockerfile.student
├── pyproject.toml
├── CLAUDE.md                       # ← このファイル
├── plan.md                         # Knowledge Graph統合計画
│
├── src/
│   ├── orchestrator/               # FastAPI + Model Router
│   │   ├── server.py
│   │   ├── query_parser.py         # LLMベース意図分類
│   │   ├── model_router.py         # Teacher/Student振り分け（KG参照ロジック含む）
│   │   └── pipeline.py
│   │
│   ├── llm/                        # LLM統合
│   │   ├── gateway.py              # プロバイダ抽象化(Claude/GPT/Ollama/vLLM)
│   │   ├── providers/
│   │   │   ├── anthropic.py
│   │   │   ├── openai.py
│   │   │   ├── ollama.py
│   │   │   └── vllm_student.py
│   │   ├── response_generator.py
│   │   ├── code_generator.py
│   │   ├── error_analyzer.py
│   │   ├── feedback_analyzer.py
│   │   ├── prompt_templates/       # YAMLプロンプト管理
│   │   ├── usage_tracker.py
│   │   └── prompt_cache.py
│   │
│   ├── rag/                        # 外部検索パイプライン
│   │   ├── retriever.py            # 基底クラス + Router
│   │   ├── retrievers/
│   │   │   ├── github.py
│   │   │   ├── stackoverflow.py
│   │   │   ├── tavily.py
│   │   │   └── arxiv.py
│   │   ├── verifier.py             # LLMベース裏どり
│   │   ├── query_rewriter.py       # CRAG用 Query Rewriter (FLAN-T5/Qwen/LLM)
│   │   └── chunker.py
│   │
│   ├── memory/                     # FAISSメモリ
│   │   ├── faiss_index.py          # ドメイン別インデックス管理
│   │   ├── metadata_store.py       # SQLite(有用性+難易度カラム)
│   │   ├── memory_manager.py       # FAISS-SQLite原子的操作（KG登録フック含む）
│   │   ├── iterative_retrieval.py  # LLMリライト+HyDEマルチホップ
│   │   ├── embedder.py             # sentence-transformers
│   │   ├── deduplicator.py
│   │   ├── schema.py               # Document, UsefulnessScore, SourceMeta
│   │   ├── learning/
│   │   │   ├── ltr_ranker.py       # Learning to Rank
│   │   │   ├── cross_encoder.py    # Cross-Encoder(Phase 2)
│   │   │   ├── embedding_adapter.py
│   │   │   └── feedback_collector.py
│   │   ├── scoring/
│   │   │   ├── freshness.py        # ドメイン別指数減衰
│   │   │   ├── usefulness.py       # 多面的有用性スコア
│   │   │   └── composite_scorer.py
│   │   └── maturation/
│   │       ├── reviewer.py         # Teacher品質審査
│   │       ├── difficulty_tagger.py
│   │       ├── seed_builder.py
│   │       └── quality_metrics.py
│   │
│   ├── knowledge_graph/            # ★ KG統合（plan.md由来）
│   │   ├── store.py                # KnowledgeGraphStore ABC + データクラス + ファクトリ
│   │   ├── networkx_store.py       # NetworkXKnowledgeGraphStore (Phase 1.5)
│   │   ├── neo4j_store.py          # Neo4jKnowledgeGraphStore (Phase 2+)
│   │   ├── extractor.py            # EntityExtractor (Teacher API使用)
│   │   └── router_bridge.py        # ModelRouter連携
│   │
│   ├── retrieval/                  # ★ Fusion/Rerank層（plan.md由来）
│   │   ├── query_classifier.py     # SEMANTIC/FACTUAL/RELATIONAL/HYBRID分類
│   │   └── fusion_reranker.py      # RRFベース Fusion/Rerank
│   │
│   ├── mcp_tools/                  # ★ SQL/BI MCP（plan.md由来）
│   │   ├── sql_query_tool.py       # テキスト → SQL変換 → 実行
│   │   └── bi_aggregation_tool.py  # 集計クエリ(COUNT/SUM/AVG)
│   │
│   ├── training/                   # 拡張可能な学習フレームワーク
│   │   ├── base.py                 # TrainingAlgorithm / ParameterAdapter / RewardFunction 抽象IF
│   │   ├── registry.py             # TrainingRegistry(デコレータ登録式)
│   │   ├── pipeline.py             # 3段階学習(SFTウォームアップ→GRPO+TinyLoRA→評価)
│   │   ├── logger.py
│   │   ├── algorithms/
│   │   │   ├── grpo.py             # デフォルト
│   │   │   ├── ppo.py
│   │   │   ├── dpo.py
│   │   │   ├── reinforce.py
│   │   │   └── sft.py
│   │   ├── adapters/
│   │   │   ├── tinylora.py         # デフォルト: 極少パラメータ
│   │   │   ├── lora.py
│   │   │   ├── lora_xs.py
│   │   │   └── full_ft.py
│   │   ├── rewards/
│   │   │   ├── composite.py        # デフォルト: 多信号加重合計
│   │   │   ├── code_exec.py
│   │   │   ├── teacher_eval.py
│   │   │   └── hybrid.py
│   │   └── evaluation/
│   │       ├── student_evaluator.py
│   │       ├── teacher_comparison.py
│   │       └── benchmark_suite.py
│   │
│   ├── sandbox/
│   │   ├── manager.py
│   │   ├── executor.py
│   │   ├── security.py
│   │   ├── retry_handler.py
│   │   └── templates/
│   │
│   ├── gui/                        # Gradio WebGUI（別セッションで実装済み）
│   │   ├── app.py                  # Blocks assembly
│   │   ├── components/
│   │   │   └── status_bar.py       # API + Docker接続インジケータ
│   │   └── tabs/
│   │       ├── chat.py             # RAG + LLMクエリ
│   │       ├── memory.py           # FAISSメモリ管理
│   │       ├── sandbox.py          # コードエディタ + 実行
│   │       ├── training.py         # 学習制御 + 可視化
│   │       └── settings.py         # APIキー・YAML設定
│   │
│   └── common/
│       ├── config.py               # pydantic-settings + YAML ★実装済み
│       ├── logger.py
│       └── models.py
│
├── tests/
│   ├── unit/
│   │   └── test_config.py          # ★実装済み (53 tests)
│   ├── integration/
│   └── fixtures/
├── data/
│   ├── faiss_indices/
│   ├── metadata.db
│   ├── adapters/                   # 学習済みTinyLoRA等(~1KB each)
│   └── training_logs/
├── scripts/
│   ├── seed_memory.py
│   ├── mature_memory.py
│   ├── train_student.py
│   ├── evaluate_student.py
│   └── launch_gui.py              # Gradio起動スクリプト
├── configs/
│   ├── default.yaml
│   ├── retrievers.yaml
│   ├── faiss_config.yaml
│   ├── sandbox_policy.yaml
│   ├── llm_config.yaml
│   ├── training.yaml               # アルゴリズム/アダプタ/Rewardの組み合わせ設定
│   └── model_router.yaml
└── docs/
    └── project_plan_v4.md
```

## 技術スタック

| レイヤー | 技術 |
|---------|------|
| API | FastAPI (非同期) |
| Web GUI | Gradio (5タブ構成) |
| Teacher LLM | anthropic / openai SDK |
| Student推論 | vLLM |
| Student学習 | VERL (GRPO) / trl (PPO/DPO) |
| アダプタ | TinyLoRA(自作) + peft(LoRA) |
| 埋め込み | sentence-transformers (all-MiniLM-L6-v2 → UniXcoder) |
| ベクトル検索 | faiss-cpu |
| Knowledge Graph | NetworkX (Phase 1.5) → Neo4j (Phase 2+) |
| メタデータ | SQLite (aiosqlite) |
| SQL/BI | SQLite → PostgreSQL (将来) |
| CRAG Query Rewrite | FLAN-T5-small / Qwen2.5-0.5B-Instruct |
| Fusion/Rerank | RRF (Reciprocal Rank Fusion) |
| コンテナ | docker-py |
| 設定 | pydantic-settings + YAML |
| テスト | pytest + testcontainers |
| 学習ログ | Weights & Biases |

## 開発フェーズと優先順位

### Phase 1 (Week 1-3): v3 MVP — ✅ **完了**
1. `src/common/config.py` — pydantic-settings ✅ **完了** (53 tests)
2. `src/memory/schema.py` — Document, UsefulnessScore, SourceMeta dataclass ✅ **完了**
3. `src/memory/embedder.py` — sentence-transformers wrapper ✅ **完了**
4. `src/memory/faiss_index.py` — ドメイン別FAISS管理(Flat→IVF→PQ自動切替) ✅ **完了**
5. `src/memory/metadata_store.py` — SQLite CRUD + 有用性カラム ✅ **完了**
6. `src/memory/memory_manager.py` — FAISS-SQLite原子的操作 ✅ **完了**
7. `src/memory/scoring/` — freshness, usefulness, composite_scorer ✅ **完了**
8. `src/memory/learning/ltr_ranker.py` — 線形LTR(オンライン学習) ✅ **完了**
9. `src/memory/iterative_retrieval.py` — マルチホップ(ベクトル加算→LLMリライト) ✅ **完了**
10. `src/llm/gateway.py` + `src/llm/providers/` — LLMプロバイダ抽象化 ✅ **完了**
11. `src/rag/` — Retriever Router + GitHub/SO/Tavily/arXiv + Verifier ✅ **完了**
12. `src/llm/response_generator.py` + `src/llm/code_generator.py` ✅ **完了**
13. `src/sandbox/` — Docker管理 + 実行 + セキュリティ ✅ **完了**
14. `src/orchestrator/server.py` + `pipeline.py` — FastAPI統合 ✅ **完了**

### Phase 1.5 (3-5日): Knowledge Graph プロトタイプ — ✅ **完了**
Phase 1完了後、メモリ成熟前に KG の基盤を構築する。

- `src/knowledge_graph/store.py` — KnowledgeGraphStore (NetworkX backend) ✅ **完了**
  - add_entity / add_relation / query_neighbors / find_path
- `src/knowledge_graph/extractor.py` — EntityExtractor (Teacher API呼び出し) ✅ **完了**
- `src/knowledge_graph/router_bridge.py` — ModelRouterとKGの接続 ✅ **完了**
- `src/retrieval/query_classifier.py` — SEMANTIC/FACTUAL/RELATIONAL/HYBRID分類 ✅ **完了**
- `src/retrieval/fusion_reranker.py` — RRF Fusion/Rerank ✅ **完了**
- 単体テスト (`tests/unit/test_knowledge_graph.py`) ✅ **完了**

技術選定:
- Phase 1.5: **NetworkX** (インメモリ、依存ゼロ、プロトタイプ) ✅ **採用**
- Phase 2+: **Neo4j** (永続化、本格クエリ) ⬜ 将来対応

未決事項 → **解決済み**:
- Entity抽出: **Teacher API採用**（精度優先）
- KG永続化: Phase 1.5は NetworkX + pickle、Phase 2 で Neo4j 移行予定

### Phase 2 (Week 4-5): メモリ成熟 + SQL/BI MCP + Fusion — ✅ **完了**
メモリ成熟タスク:
- `src/memory/maturation/seed_builder.py` ✅ **完了**
- `src/memory/maturation/reviewer.py` (MemoryReviewer) ✅ **完了**
- `src/memory/maturation/difficulty_tagger.py` ✅ **完了**
- `src/memory/maturation/quality_metrics.py` ✅ **完了**
- `src/memory/learning/cross_encoder.py` ✅ **完了**
- `src/memory/teacher_registry.py` (TeacherRegistry + EWMA信頼度) ✅ **完了**
- `src/memory/learning/teacher_feedback_pipeline.py` ✅ **完了**
- GUI Phase 2 成熟管理タブ ✅ **完了**

KG統合タスク（plan.md由来）:
- `src/mcp_tools/sql_query_tool.py` ✅ **完了**
- `src/mcp_tools/bi_aggregation_tool.py` ✅ **完了**
- メモリ品質目標: 10,000docs, confidence>0.7, 実行成功率>80% ⬜ **運用時に達成**

### Phase 3 (Week 6-7): 学習フレームワーク — ✅ **骨格完了**（本番学習は運用フェーズ）
- `src/training/base.py` + `registry.py` — 抽象IF+Registry ✅ **完了**
- `src/training/algorithms/` — grpo, ppo, dpo, sft, reinforce ✅ **完了** (骨格)
- `src/training/adapters/` — tinylora, lora, lora_xs, full_ft ✅ **完了** (骨格)
- `src/training/rewards/` — composite, code_exec, teacher_eval, hybrid ✅ **完了** (骨格)
- `src/training/pipeline.py` — 3段階学習制御 ✅ **完了** (骨格)

KG訓練統合タスク（将来）:
- KGパスをTeacherプロンプトに含める（CoT強化） ⬜ Phase 3+ で実装
- GRPO報酬関数にKG整合性スコアを追加 ⬜ Phase 3+ で実装
- 評価指標にEntity精度・関係再現率を追加 ⬜ Phase 3+ で実装

### Phase 4 (Week 8-9): 運用最適化 — ✅ **完了**
- `src/orchestrator/model_router.py` — KG参照ルーティング本格実装 ✅ **完了**
  - RoutingDecision (target / use_kg / use_faiss / expanded_doc_ids)
  - ParsedQuery の complexity / intent / entities によるモデル選択
  - KGRouterBridge 経由でエンティティ → doc_id 拡張
- `src/orchestrator/query_parser.py` — LLMベース意図分類 ✅ **完了**
- `src/llm/error_analyzer.py` / `feedback_analyzer.py` / `usage_tracker.py` / `prompt_cache.py` ✅ **完了**
- `src/memory/deduplicator.py` — 重複排除 ✅ **完了**
- 拡張アルゴリズム (PPO, DPO) 骨格実装 ✅ **完了**（本番チューニングは将来）
- ベンチマーク骨格 (`tests/unit/test_phase4.py`, `test_phase5.py`) ✅ **完了**

### Phase 5: NEAT Context-Sensitive Search — ⬜ **計画策定済み**
`plan_neat_hyp_e.md` に基づき、FAISS検索結果をコンテキスト依存で再スコアリングする。

- Phase 5-1: `AssociationFn` — numpy版 MLP（3項関数: query, candidate, context）
  - `cosine(q,c)`, `cosine(q,ctx)`, `cosine(c,ctx)`, `cosine(q-ctx,c)` の加重合計
  - 重みは学習可能（JSON保存・ロード対応）
- Phase 5-2: `ContextSensitiveSearch` — FAISS k*3 候補取得 → association_fn リランク → top-k
  - faiss 未インストール時は numpy ブルートフォースにフォールバック
- Phase 5-3: MED 統合 — 既存 FAISS モジュールへの差し込み、context_emb 生成元の決定
- 将来: `AssociationFn` のアーキテクチャを NEAT (CPPN) で進化させる

## 設計原則

- **抽象IFファースト**: 全モジュールはインターフェース→実装の順で作成
- **Strategy + Registry**: 学習アルゴリズム/アダプタ/Rewardはデコレータ登録で拡張
- **FAISS-SQLite同期**: memory_manager.pyで原子的操作を保証
- **段階的導入**: Phase 1で動くMVP → Phase毎に機能追加
- **テスト駆動**: 各モジュールに単体テストを必ず用意
- **既存非破壊**: KG導入はFAISS・SQL/BIを壊さず橋渡し層として追加（plan.md原則）
- **KGは橋渡しに徹する**: KG単独で答えを出さず、ルーティングと融合を担う（plan.md原則）
- **コスト意識**: GraphDB新規導入は最小構成(NetworkX)から開始（plan.md原則）

## 重要な設計判断

- FAISSインデックス: Phase 1は`IndexFlatIP`、10万件超でIVF自動移行
- 埋め込み次元: 768 (all-MiniLM-L6-v2)
- Studentベースモデル: Qwen2.5-7B-Instruct推奨(TinyLoRA論文で最も効率的)
- TinyLoRA設定: frozen_rank=2, projection_dim=4, tie_factor=7
- Reward: correctness(0.35) + retrieval_quality(0.20) + exec_success(0.20) + efficiency(0.10) + memory_utilization(0.15)
- Knowledge Graph: Phase 1.5は NetworkX → Phase 2以降で Neo4j 移行
- Entity抽出: Teacher API (精度優先) を推奨
- SQL対象DB: Phase 2は SQLite から開始
- Fusion: Reciprocal Rank Fusion (RRF) ベース
- 記憶モデル: FAISS=海馬(連想), KG=概念地図(関係性), SQL/BI=ノート(宣言的)

## 参照ドキュメント

プロジェクトルートの `docs/` に詳細計画書を配置:
- `docs/project_plan_v4.md` — 正式計画書(本版)
- `plan.md` — Knowledge Graph統合計画
- `plan_data.md` — データ世代管理計画（restic + NAS）
- `plan_neat_hyp_e.md` — NEAT Context-Sensitive Search 計画（association_fn によるコンテキスト依存リランキング）

参照アーキテクチャ・論文:
- GraphRAG (Microsoft, 2024) — Vector + KG統合の基本設計
- HippoRAG (2024) — 海馬モデルのRAG実装、誤想起軽減
- Self-RAG (2023) — 検索ルーティングの判断ロジック
- REALM (2020) — 外部知識統合の基礎研究
- TinyLoRA (Morris et al., 2026) — 極少パラメータRL
- RRF (Reciprocal Rank Fusion) — Fusionアルゴリズム

## 実装進捗

| フェーズ | ファイル / モジュール | 状態 |
|---------|---------------------|------|
| **Phase 1** | `src/common/config.py` | ✅ 完了 (53 tests) |
| **Phase 1** | `src/memory/schema.py` | ✅ 完了 |
| **Phase 1** | `src/memory/embedder.py` | ✅ 完了 |
| **Phase 1** | `src/memory/faiss_index.py` | ✅ 完了 |
| **Phase 1** | `src/memory/metadata_store.py` | ✅ 完了 |
| **Phase 1** | `src/memory/memory_manager.py` | ✅ 完了 |
| **Phase 1** | `src/memory/scoring/` (freshness / usefulness / composite) | ✅ 完了 |
| **Phase 1** | `src/memory/learning/ltr_ranker.py` | ✅ 完了 |
| **Phase 1** | `src/memory/iterative_retrieval.py` | ✅ 完了 |
| **Phase 1** | `src/llm/gateway.py` + `src/llm/providers/` (4プロバイダー) | ✅ 完了 |
| **Phase 1** | `src/rag/` (retriever / chunker / verifier / 4検索源) | ✅ 完了 |
| **Phase 1** | `src/llm/response_generator.py` + `code_generator.py` | ✅ 完了 |
| **Phase 1** | `src/sandbox/` (manager / executor / security / retry) | ✅ 完了 |
| **Phase 1** | `src/orchestrator/` (server / pipeline / query_parser) | ✅ 完了 |
| **Phase 1.5** | `src/knowledge_graph/store.py` (ABC) + `networkx_store.py` + `neo4j_store.py` | ✅ 完了 |
| **Phase 1.5** | `src/knowledge_graph/extractor.py` (Teacher API) | ✅ 完了 |
| **Phase 1.5** | `src/knowledge_graph/router_bridge.py` | ✅ 完了 |
| **Phase 1.5** | `src/retrieval/query_classifier.py` | ✅ 完了 |
| **Phase 1.5** | `src/retrieval/fusion_reranker.py` (RRF) | ✅ 完了 |
| **Phase 2** | `src/memory/maturation/` (reviewer / tagger / metrics / seed) | ✅ 完了 |
| **Phase 2** | `src/memory/learning/cross_encoder.py` | ✅ 完了 |
| **Phase 2** | `src/memory/teacher_registry.py` (EWMA 信頼度) | ✅ 完了 |
| **Phase 2** | `src/memory/learning/teacher_feedback_pipeline.py` | ✅ 完了 |
| **Phase 2** | `src/mcp_tools/sql_query_tool.py` | ✅ 完了 |
| **Phase 2** | `src/mcp_tools/bi_aggregation_tool.py` | ✅ 完了 |
| **Phase 3** | `src/training/` (base / registry / algorithms×5 / adapters×4 / rewards×4 / pipeline) | ✅ 完了 (骨格) |
| **GUI** | `src/gui/` — Gradio 6タブ (chat / memory / sandbox / training / settings / guide) | ✅ 完了 |
| **GUI** | `src/gui/docs_chat.py` — ドキュメント Q&A チャットBot (案C) | ✅ 完了 |
| **ドキュメント** | `docs/site/` — MkDocs 24ページ (案B) + `mkdocs.yml` | ✅ 完了 |
| **CI / テスト** | `.github/workflows/ci.yml` + `Dockerfile.test` + `tests/conftest.py` | ✅ 完了 |
| **CI / テスト** | `tests/unit/` (25ファイル) + `tests/integration/` (1ファイル) | ✅ 完了 |
| **Phase 4** | `src/orchestrator/model_router.py` — Graph-aware KG ルーティング | ✅ 完了 |
| **Phase 4** | `src/llm/error_analyzer.py` / `feedback_analyzer.py` / `usage_tracker.py` / `prompt_cache.py` | ✅ 完了 |
| **Phase 4** | `src/memory/deduplicator.py` | ✅ 完了 |
| **Phase 4** | `src/rag/url_fetcher.py` — CRAG URL 直接取得 (arxiv/web) | ✅ 完了 |
| **Phase 4** | `src/rag/query_expander.py` — URL 検出 (`extract_urls`) | ✅ 完了 |
| **Phase 4** | `scripts/seed_and_mature.py` — 外部RAG→重複排除→FAISS→Teacher成熟 統合パイプライン | ✅ 完了 |
| **Phase 4** | `scripts/test_teacher.py` — Teacher テスト CLI (ping/benchmark/ingest) | ✅ 完了 |
| **Phase 4** | `scripts/questions.txt` — シード質問集 (125問/11カテゴリ) | ✅ 完了 |
| **Phase 4** | LLM プロバイダー max_tokens=4096 デフォルト + yaml per-provider 設定 | ✅ 完了 |
| **Phase 4** | OpenAI-compatible: Qwen3.5 thinking model (`reasoning_content`) 対応 | ✅ 完了 |
| **Phase 4** | CRAG: use_memory/use_rag パラメータ伝播修正 + provider/timeout 伝播 | ✅ 完了 |
| **Phase 5** | NEAT Context-Sensitive Search — `association_fn(q, c, ctx)` リランキング | ⬜ 計画策定済み (`plan_neat_hyp_e.md`) |
| **Phase 3+** | 学習フレームワーク 本番稼働 (KG CoT / GRPO本番) | ⬜ 将来対応 |
| **将来** | Neo4j 移行 / PostgreSQL 移行 / vLLM Student 本番 | ⬜ 将来対応 |

### 次セッションへの引き継ぎ事項

**作業ブランチ**: `main`

**完了済み（直近セッション — 2026-03-31）**
- **SOデータ再seed**: question_body → answer-first 取得に変更、`answers>=1` フィルタ
- **データ属性ラベリング**: `content_type` / `categories` / `domain_flag` を全APIソースに追加
- **プロンプトインジェクション対策**: SO retriever に `_sanitize()` 追加
- **reviewer.py 強化**: domain_flag aware + `needs_supplement` フラグ、Bオプション、`model=` param
- **difficulty_tagger.py 強化**: `model=` パラメータ追加
- **seed_and_mature.py 拡張**: `--model` CLI フラグ、HOLD/PASS/FAIL 表示
- **questions.txt 拡張**: 125問/11カテゴリ → **150問/19カテゴリ** (+25問)
  - 追加カテゴリ: Context Engineering, ICL Theory, Self-Play, LLM Uncertainty, Training Stability, Hyperbolic Embeddings, Authorship Style
  - arXiv URL 埋め込み（URL フェッチャー直接取得対応）
- **クリーンアップ**: Tavily断片(733) + SO不完全(57) + arXiv重複(472) = **1,262件削除**
- **seed_and_mature 150問実行（途中）**: 1,156/1,230件 mature 完了時点で予算切れ停止
  - FAISS code: 1,844 → **3,074 vectors**
  - approved: 704 → **1,043**、unreviewed 残: **74件**（mature 未完）
- **neat extras 追加**: pyproject.toml に jax/tensorneat/qdax/evosax を optional deps として登録
- **NEAT実装完了**: `claude_work/neat_trident` — ES-HyperNEAT / HybridIndexer / MAP-Elites (WSL動作未確認)

**完了済み（2026-03-26セッション）**
- **CRAG バグ修正**: FAISS 使用が常に ON → `use_memory`/`use_rag` パラメータを CRAG/Agentic リトライに伝播
- **provider/timeout 伝播**: pipeline → rewriter → gateway の全経路で provider/timeout を伝播
- **Qwen3.5 thinking model 対応**: `reasoning_content` フィールド抽出、content 空時の警告
- **max_tokens=4096 デフォルト化**: 全プロバイダー (anthropic/openai/ollama/vllm/openai_compatible)
- **yaml per-provider 設定**: `llm_config.local.yaml` で max_tokens/timeout/temperature/extra_params を個別指定可能
- **CRAG URL 直接取得**: クエリ内の URL (arxiv/GitHub/web) を検出し直接コンテンツ取得 → FAISS 保存
  - `src/rag/query_expander.py` — `extract_urls()` メソッド追加
  - `src/rag/url_fetcher.py` — 新規（arxiv API / httpx web fetch）
  - `src/orchestrator/pipeline.py` — Step 1.5 として URL 取得を挿入
- **Teacher テストスクリプト**: `scripts/test_teacher.py` — 接続テスト/品質ベンチマーク/FAISS投入
- **統合パイプライン**: `scripts/seed_and_mature.py` — 外部RAG→重複排除→FAISS→Teacher成熟を1パスで実行
- **シード質問集**: `scripts/questions.txt` — 47問/7カテゴリ

**.env に設定するキー（ローカルでの実行に必要）**
```
ANTHROPIC_API_KEY=...   # seed_builder / mature / KG抽出 に必要
OPENAI_API_KEY=...      # 代替 Teacher として使用可
TAVILY_API_KEY=...      # 外部RAG（任意）
GITHUB_TOKEN=...        # 外部RAG（任意・レート制限緩和）
```

**残作業 (優先度: 高)**
- **API 残高補充後: mature 残り 74件**
  ```bash
  poetry run python scripts/seed_and_mature.py \
    --mature-only --domain code \
    --provider anthropic --model claude-haiku-4-5-20251001 --limit 100
  ```
- **needs_update 838件（code）の対処方針決定**
  - Tavily 断片が主因 → Chunker 改善後に再 seed が推奨
  - arXiv 175件は保持中（needs_update）
- **med_hyp_style_g.md の内容確認・方針決定**（まだ未確認）
- オーケストレーター起動 + E2E 動作確認
  ```bash
  python -m uvicorn src.orchestrator.server:app --port 8000 --reload
  python scripts/launch_gui.py
  ```

**残作業 (優先度: 中)**
- `data/faiss_indices/` へのシードデータ投入継続（目標: 10,000 docs）
  - 現状: approved **1,043件** / FAISS code **3,074 vectors**
- **Tavily Chunker 改善 + 再 seed**
  - 問題: 記事途中チャンクで断片化 → 大量 HOLD になる根本原因
  - 対策: `src/rag/retrievers/tavily.py` のチャンクサイズ拡大 or 記事全文取得に変更
  - 改善後に seed_and_mature を再実行
- **NEAT 環境検証** (WSL2): `claude_work/neat_trident` の動作確認
  ```bash
  cd /mnt/d/Projects/claude_work/neat_trident
  python scripts/phase0_verify.py
  python scripts/faiss_hybrid_verify.py   # HybridIndexer (9/9)
  python scripts/es_hyperneat_verify.py   # ES-HyperNEAT (13/13)
  python scripts/long_term_loop.py        # 長期進化ループ (5/5)
  ```
- **NEAT × MED 統合**: `neat_trident` の `HybridIndexer` を MED の FAISS インデクサに接続
  - MED FAISS I/F 確認: `src/memory/faiss_index.py`, `src/memory/memory_manager.py`
  - アダプタ層設計: `neat_trident/src/med_integration/` (interfaces.py / trident_adapter.py / stub_med.py)
  - 詳細: `claude_work/neat_trident/handover_next_session.md` 参照
- Phase 3+: GRPO + TinyLoRA 本番学習パイプラインの実稼働
- Phase 5: NEAT Context-Sensitive Search 本格実装（`plan_neat_hyp_e.md` 参照）
- `tests/integration/` の E2E テスト（Docker 必要）

**技術的負債**
- `src/memory/maturation/seed_builder.py` — Teacher API 呼び出し部分はスタブ。実プロバイダー接続時に完成
- `src/training/algorithms/` — 骨格実装のみ。VERL/trl との実際の統合が必要
- KG 永続化: NetworkX + pickle → Neo4j 移行スクリプト未実装
- Docker E2E テスト: Docker 環境が必要
- `tests/unit/test_alias_extractor.py` — pytest-asyncio 設定問題で1件失敗（既知）

**ローカルモデルを Teacher にする場合の設定例**
```yaml
# configs/llm_config.local.yaml
providers:
  lmstudio:
    type: openai_compatible
    base_url: http://192.168.2.104:52624/v1
    default_model: qwen3.5-122b-a10b
    timeout: 3600
    max_tokens: 32767
```
- `teacher_registry.py` の EWMA が信頼度を自動追跡・調整する
