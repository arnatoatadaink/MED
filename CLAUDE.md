# CLAUDE.md — RAG × FAISS × LLM × Memory Environment Distillation

## プロジェクト概要

Teacher Model（大モデルAPI）がFAISSメモリを成熟させ、Student Model（ローカル小モデル）にGRPO+TinyLoRAで「メモリの使い方」を極少パラメータで教えるシステム。Docker Sandboxでコードを安全に実行・検証する。

### 着想元
- TinyLoRA論文 (Morris et al., 2026): 13パラメータでGSM8K 91%達成。「知識は既にある、使い方だけRLで教える」
- 本プロジェクト: 知識はFAISS外部メモリに蓄積。小モデルに検索・利用スキルだけRLで教える

## アーキテクチャ

```
Client → Orchestrator(FastAPI) → Model Router
                                    ├─ simple → Student(Qwen-7B+TinyLoRA) + FAISSメモリ
                                    ├─ moderate → Student + FAISS + 外部RAG
                                    └─ complex → Teacher(Claude/GPT API)
                                                    ↓
                                              FAISSメモリ成熟
                                                    ↓
                                              Docker Sandbox(コード実行)
```

### 4モジュール構成
1. **Orchestrator** — FastAPI、Model Router、Query Parser(LLM)
2. **RAG Pipeline** — GitHub/SO/Tavily検索、LLM裏どり、チャンク化
3. **FAISS Memory** — ベクトル検索、SQLiteメタデータ、Iterative Retrieval、LTR/Cross-Encoder、有用性管理
4. **Docker Sandbox** — コード実行、セキュリティ制限、自動リトライ
5. **Training Framework** — GRPO+TinyLoRA(デフォルト)、PPO/DPO/SFT/LoRA/LoRA-XS(拡張可能)

## ディレクトリ構造

```
rag-faiss-llm/
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.sandbox
├── Dockerfile.student
├── pyproject.toml
├── CLAUDE.md                       # ← このファイル
│
├── src/
│   ├── orchestrator/               # FastAPI + Model Router
│   │   ├── server.py
│   │   ├── query_parser.py         # LLMベース意図分類
│   │   ├── model_router.py         # Teacher/Student振り分け
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
│   │   ├── memory_manager.py       # FAISS-SQLite原子的操作
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
│   └── common/
│       ├── config.py               # pydantic-settings + YAML
│       ├── logger.py
│       └── models.py
│
├── tests/
├── data/
│   ├── faiss_indices/
│   ├── metadata.db
│   ├── adapters/                   # 学習済みTinyLoRA等(~1KB each)
│   └── training_logs/
├── scripts/
│   ├── seed_memory.py
│   ├── mature_memory.py
│   ├── train_student.py
│   └── evaluate_student.py
└── configs/
    ├── default.yaml
    ├── retrievers.yaml
    ├── faiss_config.yaml
    ├── sandbox_policy.yaml
    ├── llm_config.yaml
    └── training.yaml               # アルゴリズム/アダプタ/Rewardの組み合わせ設定
```

## 技術スタック

| レイヤー | 技術 |
|---------|------|
| API | FastAPI (非同期) |
| Teacher LLM | anthropic / openai SDK |
| Student推論 | vLLM |
| Student学習 | VERL (GRPO) / trl (PPO/DPO) |
| アダプタ | TinyLoRA(自作) + peft(LoRA) |
| 埋め込み | sentence-transformers (all-MiniLM-L6-v2 → UniXcoder) |
| ベクトル検索 | faiss-cpu |
| メタデータ | SQLite (aiosqlite) |
| コンテナ | docker-py |
| 設定 | pydantic-settings + YAML |
| テスト | pytest + testcontainers |
| 学習ログ | Weights & Biases |

## 開発フェーズと優先順位

### Phase 1 (Week 1-3): v3 MVP — ★最初に着手★
1. `src/common/config.py` — pydantic-settings
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

### Phase 2 (Week 4-5): メモリ成熟
- `src/memory/maturation/` — seed_builder, reviewer, difficulty_tagger
- `src/memory/learning/cross_encoder.py`
- メモリ品質目標: 10,000docs, confidence>0.7, 実行成功率>80%

### Phase 3 (Week 6-7): 学習フレームワーク
- `src/training/base.py` + `registry.py` — 抽象IF+Registry
- `src/training/algorithms/grpo.py` — デフォルトRL
- `src/training/adapters/tinylora.py` — 極少パラメータアダプタ
- `src/training/rewards/composite.py` — Reward関数
- `src/training/pipeline.py` — 3段階学習制御

### Phase 4 (Week 8-9): 運用最適化
- `src/orchestrator/model_router.py`
- 拡張アルゴリズム(PPO, DPO)
- ベンチマーク

## 設計原則

- **抽象IFファースト**: 全モジュールはインターフェース→実装の順で作成
- **Strategy + Registry**: 学習アルゴリズム/アダプタ/Rewardはデコレータ登録で拡張
- **FAISS-SQLite同期**: memory_manager.pyで原子的操作を保証
- **段階的導入**: Phase 1で動くMVP → Phase毎に機能追加
- **テスト駆動**: 各モジュールに単体テストを必ず用意

## 重要な設計判断

- FAISSインデックス: Phase 1は`IndexFlatIP`、10万件超でIVF自動移行
- 埋め込み次元: 768 (all-MiniLM-L6-v2)
- Studentベースモデル: Qwen2.5-7B-Instruct推奨(TinyLoRA論文で最も効率的)
- TinyLoRA設定: frozen_rank=2, projection_dim=4, tie_factor=7
- Reward: correctness(0.35) + retrieval_quality(0.20) + exec_success(0.20) + efficiency(0.10) + memory_utilization(0.15)

## 参照ドキュメント

プロジェクトルートの `docs/` に詳細計画書を配置:
- `docs/project_plan_v4.md` — 正式計画書(本版)
- `docs/learning_patterns.md` — 学習可能検索+有用性管理の設計
- `docs/v4_idea_proposal.md` — Memory Environment Distillationのアイデア検討
- `docs/rag_search_apis.md` — 検索API一覧(9ドメイン60+API)
