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
│   │   ├── store.py                # KnowledgeGraphStore (NetworkX → Neo4j)
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
| Fusion/Rerank | RRF (Reciprocal Rank Fusion) |
| コンテナ | docker-py |
| 設定 | pydantic-settings + YAML |
| テスト | pytest + testcontainers |
| 学習ログ | Weights & Biases |

## 開発フェーズと優先順位

### Phase 1 (Week 1-3): v3 MVP — ★着手中★
1. `src/common/config.py` — pydantic-settings ✅ **完了**
2. `src/memory/schema.py` — Document, UsefulnessScore, SourceMeta dataclass
3. `src/memory/embedder.py` — sentence-transformers wrapper
4. `src/memory/faiss_index.py` — ドメイン別FAISS管理(Flat→IVF→PQ自動切替)
5. `src/memory/metadata_store.py` — SQLite CRUD + 有用性カラム
6. `src/memory/memory_manager.py` — FAISS-SQLite原子的操作
7. `src/memory/scoring/` — freshness, usefulness, composite_scorer
8. `src/memory/learning/ltr_ranker.py` — 線形LTR(オンライン学習)
9. `src/memory/iterative_retrieval.py` — マルチホップ(ベクトル加算→LLMリライト)
10. `src/llm/gateway.py` + `src/llm/providers/` — LLMプロバイダ抽象化
11. `src/rag/` — Retriever Router + GitHub/SO/Tavily + Verifier
12. `src/llm/response_generator.py` + `src/llm/code_generator.py`
13. `src/sandbox/` — Docker管理 + 実行 + セキュリティ
14. `src/orchestrator/server.py` + `pipeline.py` — FastAPI統合

### Phase 1.5 (3-5日): Knowledge Graph プロトタイプ — ★plan.md由来★
Phase 1完了後、メモリ成熟前に KG の基盤を構築する。

- `src/knowledge_graph/store.py` — KnowledgeGraphStore (NetworkX backend)
  - add_entity / add_relation / query_neighbors / find_path
- `src/knowledge_graph/extractor.py` — EntityExtractor (Teacher API呼び出し)
- `src/knowledge_graph/router_bridge.py` — ModelRouterとKGの接続
- FAISS格納時にKGへ自動登録するパイプライン
- 単体テスト

技術選定:
- Phase 1.5: **NetworkX** (インメモリ、依存ゼロ、プロトタイプ)
- Phase 2+: **Neo4j** (永続化、本格クエリ)

未決事項:
- Entity抽出: Teacher API（精度優先）vs spaCy小モデル（コスト優先）→ **Teacher API推奨**
- KGスキーマ: 汎用 vs MED特化 → 要検討
- KG永続化: Phase 1.5は NetworkX + pickle、Phase 2 で Neo4j 移行

### Phase 2 (Week 4-5): メモリ成熟 + SQL/BI MCP + Fusion
既存メモリ成熟タスク:
- `src/memory/maturation/` — seed_builder, reviewer, difficulty_tagger
- `src/memory/learning/cross_encoder.py`
- メモリ品質目標: 10,000docs, confidence>0.7, 実行成功率>80%

KG統合タスク（plan.md由来）:
- `src/mcp_tools/sql_query_tool.py` — テキスト → SQL変換 → 実行
- `src/mcp_tools/bi_aggregation_tool.py` — 集計クエリ(COUNT/SUM/AVG)
- `src/retrieval/query_classifier.py` — SEMANTIC / FACTUAL / RELATIONAL / HYBRID 分類
- `src/retrieval/fusion_reranker.py` — FAISS + KG + SQL の RRF Fusion
- Neo4j 移行スクリプト（NetworkX → Neo4j）
- 統合テスト

### Phase 3 (Week 6-7): 学習フレームワーク + KG訓練品質向上
既存学習タスク:
- `src/training/base.py` + `registry.py` — 抽象IF+Registry
- `src/training/algorithms/grpo.py` — デフォルトRL
- `src/training/adapters/tinylora.py` — 極少パラメータアダプタ
- `src/training/rewards/composite.py` — Reward関数
- `src/training/pipeline.py` — 3段階学習制御

KG訓練統合タスク（plan.md由来）:
- KGパスをTeacherプロンプトに含める（CoT強化）
- 訓練データ生成時にKG根拠をアノテーション
- GRPO報酬関数にKG整合性スコアを追加検討
- 評価指標にEntity精度・関係再現率を追加

### Phase 4 (Week 8-9): 運用最適化
- `src/orchestrator/model_router.py` — KG参照ルーティング含む
- 拡張アルゴリズム(PPO, DPO)
- ベンチマーク

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
- `plan.md` — Knowledge Graph統合計画（★ 新規）

参照アーキテクチャ・論文:
- GraphRAG (Microsoft, 2024) — Vector + KG統合の基本設計
- HippoRAG (2024) — 海馬モデルのRAG実装、誤想起軽減
- Self-RAG (2023) — 検索ルーティングの判断ロジック
- REALM (2020) — 外部知識統合の基礎研究
- TinyLoRA (Morris et al., 2026) — 極少パラメータRL
- RRF (Reciprocal Rank Fusion) — Fusionアルゴリズム

## 実装進捗

| ステップ | ファイル | 状態 |
|---------|---------|------|
| Phase 1-1 | `src/common/config.py` | ✅ 完了 (53 tests) |
| Phase 1-2 | `src/memory/schema.py` | ⬜ 未着手 ← 次 |
| Phase 1-3〜14 | (上記参照) | ⬜ 未着手 |
| Phase 1.5 | `src/knowledge_graph/` | ⬜ 未着手 |
| GUI | `src/gui/` | ✅ 別セッションで実装済み（5タブ） |
