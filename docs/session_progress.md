# MED セッション進捗まとめ

> 作成日: 2026-03-16
> 対象ブランチ: `claude/implement-phase1-config-iIPTF`
> 総コミット数: 76 commits
> テスト数: 862 tests collected

---

## 実装完了モジュール一覧

### Phase 1 — MVP コア (Step 1〜14) ✅

| Step | ファイル | 内容 |
|------|---------|------|
| 1 | `src/common/config.py` | pydantic-settings + YAML 設定管理 (53 tests) |
| 2 | `src/memory/schema.py` | Document / UsefulnessScore / SourceMeta データモデル |
| 3 | `src/memory/embedder.py` | sentence-transformers ラッパー |
| 4 | `src/memory/faiss_index.py` | ドメイン別 FAISS 管理 (Flat→IVF→PQ 自動切替) |
| 5 | `src/memory/metadata_store.py` | SQLite CRUD + 有用性カラム |
| 6 | `src/memory/memory_manager.py` | FAISS-SQLite 原子的操作 |
| 7 | `src/memory/scoring/` | freshness / usefulness / composite_scorer |
| 8 | `src/memory/learning/ltr_ranker.py` | 線形 LTR + オンライン SGD |
| 9 | `src/memory/iterative_retrieval.py` | マルチホップ (vector_add / llm_rewrite / HyDE) |
| 10 | `src/llm/gateway.py` + `src/llm/providers/` | LLM プロバイダ抽象化 (Claude / GPT / Ollama / vLLM) |
| 11 | `src/rag/` | Retriever Router + GitHub / SO / Tavily / arXiv + Verifier + Chunker |
| 12 | `src/llm/response_generator.py` + `code_generator.py` | 応答生成・コード生成 |
| 13 | `src/sandbox/` | Docker サンドボックス + Executor + Security + RetryHandler |
| 14 | `src/orchestrator/server.py` + `pipeline.py` | FastAPI 統合 + MEDPipeline |

---

### Phase 1.5 — Knowledge Graph プロトタイプ ✅

| ファイル | 内容 |
|---------|------|
| `src/knowledge_graph/store.py` | KnowledgeGraphStore (NetworkX backend) |
| `src/knowledge_graph/extractor.py` | EntityExtractor (Teacher API 使用) |
| `src/knowledge_graph/router_bridge.py` | ModelRouter との接続 |

---

### Phase 2 — メモリ成熟 + SQL/BI MCP + Fusion ✅

| ファイル | 内容 |
|---------|------|
| `src/memory/maturation/reviewer.py` | Teacher 品質審査 |
| `src/memory/maturation/difficulty_tagger.py` | 難易度タグ付け |
| `src/memory/maturation/seed_builder.py` | シードデータ構築 |
| `src/memory/maturation/quality_metrics.py` | 品質指標 |
| `src/memory/learning/cross_encoder.py` | Cross-Encoder リランク |
| `src/mcp_tools/sql_query_tool.py` | テキスト → SQL 変換 → 実行 |
| `src/mcp_tools/bi_aggregation_tool.py` | 集計クエリ (COUNT/SUM/AVG) |
| `src/retrieval/query_classifier.py` | SEMANTIC / FACTUAL / RELATIONAL / HYBRID 分類 |
| `src/retrieval/fusion_reranker.py` | RRF ベース Fusion/Rerank |

---

### Phase 2 追加 — Teacher Provenance (Step 1〜5) ✅

| Step | ファイル | 内容 |
|------|---------|------|
| 1 | `src/memory/schema.py` (拡張) | `SourceMeta.extra` に `teacher_id` 標準化 |
| 2 | `src/memory/teacher_registry.py` | SQLite-backed Teacher Trust 管理 |
| 3 | `src/memory/metadata_store.py` (拡張) | `teacher_id` カラム追加 + Teacher 素性クエリ |
| 4 | `src/memory/scoring/composite_scorer.py` (拡張) | `teacher_trust` 乗算スコアリング |
| 5 | `src/memory/learning/teacher_feedback_pipeline.py` | FeedbackCollector → TeacherRegistry 更新パイプライン |

---

### Phase 3 — 学習フレームワーク ✅

| ファイル | 内容 |
|---------|------|
| `src/training/base.py` | TrainingAlgorithm / ParameterAdapter / RewardFunction 抽象 IF |
| `src/training/registry.py` | TrainingRegistry (デコレータ登録式) |
| `src/training/algorithms/grpo.py` | デフォルト RL (GRPO) |
| `src/training/algorithms/ppo.py` | PPO 実装 |
| `src/training/algorithms/dpo.py` | DPO 実装 |
| `src/training/algorithms/sft.py` | SFT ウォームアップ |
| `src/training/adapters/tinylora.py` | TinyLoRA (frozen_rank=2, proj=4, tie=7) |
| `src/training/adapters/lora.py` | LoRA アダプタ |
| `src/training/adapters/lora_xs.py` | LoRA-XS アダプタ |
| `src/training/rewards/composite.py` | 多信号加重合計 Reward |
| `src/training/pipeline.py` | 3 段階学習制御 |
| `src/training/evaluation/` | student_evaluator / teacher_comparison / benchmark_suite |

---

### Phase 4 — オーケストレーター + ユーティリティ ✅

| ファイル | 内容 |
|---------|------|
| `src/orchestrator/model_router.py` | KG 参照ルーティング含む Graph-aware Model Router |
| `src/orchestrator/query_parser.py` | LLM ベース意図分類 |
| `src/llm/error_analyzer.py` | エラー解析 |
| `src/llm/feedback_analyzer.py` | フィードバック解析 |
| `src/llm/usage_tracker.py` | API 使用量追跡 |
| `src/llm/prompt_cache.py` | プロンプトキャッシュ |
| `src/memory/deduplicator.py` | 重複排除 |

---

### Web GUI — Gradio 5 タブ ✅

| ファイル | 内容 |
|---------|------|
| `src/gui/app.py` | Blocks アセンブリ |
| `src/gui/tabs/chat.py` | RAG + LLM クエリ |
| `src/gui/tabs/memory.py` | FAISS メモリ管理 |
| `src/gui/tabs/sandbox.py` | コードエディタ + 実行 |
| `src/gui/tabs/training.py` | 学習制御 + 可視化 |
| `src/gui/tabs/settings.py` | API キー・YAML 設定 |
| `src/gui/tabs/guide.py` | セットアップウィザード + ガイド |
| `src/gui/components/status_bar.py` | API + Docker 接続インジケータ |
| `src/gui/docs_chat.py` | ドキュメント Q&A チャットBot |

---

### CI/CD + テスト基盤 ✅

| 内容 | 詳細 |
|------|------|
| GitHub Actions CI | pytest + ruff チェック自動実行 |
| 共有フィクスチャ | `tests/conftest.py` |
| Docker 統合テスト | `tests/integration/test_docker_sandbox.py` |
| 単体テスト | 862 テスト (27 ファイル) |
| リンター | ruff (E/F/I/UP ルール、全エラー解消済み) |

---

## テストファイル一覧

```
tests/
├── conftest.py
├── unit/
│   ├── test_config.py                   (53 tests)
│   ├── test_schema.py
│   ├── test_embedder.py
│   ├── test_faiss_index.py
│   ├── test_metadata_store.py
│   ├── test_memory_manager.py
│   ├── test_scoring.py
│   ├── test_ltr_ranker.py
│   ├── test_iterative_retrieval.py
│   ├── test_llm_gateway.py
│   ├── test_llm_generators.py
│   ├── test_rag.py
│   ├── test_sandbox.py
│   ├── test_orchestrator.py
│   ├── test_knowledge_graph.py
│   ├── test_maturation.py
│   ├── test_training.py
│   ├── test_phase2.py
│   ├── test_phase4.py
│   ├── test_phase5.py
│   ├── test_teacher_provenance_step1.py
│   ├── test_teacher_provenance_step2.py
│   ├── test_teacher_provenance_step3.py
│   ├── test_teacher_provenance_step4.py
│   └── test_teacher_provenance_step5.py
└── integration/
    └── test_docker_sandbox.py
```

---

## 直近の作業履歴 (このセッション)

1. **pip 依存パッケージインストール確認** — faiss-cpu / aiosqlite / pyyaml / httpx / networkx / docker
2. **ruff リンター自動修正** — `ruff check . --fix` で 17 件自動修正 + 1 件手動修正
   - `src/gui/app.py`: 重複 import 削除 (F811)、import ソート (I001)
   - `src/gui/docs_chat.py`: 未使用 import 削除 (F401)
   - `src/gui/tabs/chat.py`: 未使用 import 削除 (F401)
   - `src/gui/tabs/guide.py`: import ソート (I001)
   - `src/gui/tabs/memory.py`: import ソート (I001)
   - `src/gui/utils.py`: 未使用 import 削除 (F401)
   - `src/orchestrator/server.py`: `Optional[str]` → `str | None` (F821・手動)
3. **コミット & プッシュ** — `fix: ruff linter errors auto-fixed (17 auto + 1 manual)`

---

## 次フェーズの未着手タスク

### Phase 2 残タスク
- メモリ品質目標達成: 10,000 docs / confidence > 0.7 / 実行成功率 > 80%
- Neo4j 移行スクリプト (NetworkX → Neo4j)
- 統合テスト拡充

### Phase 3 残タスク
- KG パスを Teacher プロンプトに含める (CoT 強化)
- 訓練データ生成時に KG 根拠をアノテーション
- GRPO 報酬関数に KG 整合性スコア追加
- 評価指標に Entity 精度・関係再現率追加

### Phase 4 残タスク
- 拡張アルゴリズム (PPO, DPO) のチューニング
- ベンチマーク実行

---

## 技術的注意事項

- **sentence-transformers の CI 除外**: インストール時間がかかるため CI タイムアウト対策で除外済み (`plan_test.md` 参照)
- **FAISS インデックス**: Phase 1 は `IndexFlatIP`、10 万件超で IVF 自動移行
- **KG バックエンド**: Phase 1.5 は NetworkX (インメモリ) → Phase 2 以降で Neo4j 移行予定
- **TinyLoRA 設定**: `frozen_rank=2, projection_dim=4, tie_factor=7`
- **Reward 重み**: correctness(0.35) + retrieval_quality(0.20) + exec_success(0.20) + efficiency(0.10) + memory_utilization(0.15)
