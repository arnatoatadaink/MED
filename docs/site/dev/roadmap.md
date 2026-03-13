# ロードマップ

## 現在の実装状況

| フェーズ | 項目 | 状態 |
|---------|------|------|
| **Phase 1** | `src/common/config.py` | ✅ 完了 (53 tests) |
| **Phase 1** | `src/memory/` — FAISS, SQLite, Scoring, LTR | ✅ 完了 |
| **Phase 1** | `src/llm/` — Gateway, Providers | ✅ 完了 |
| **Phase 1** | `src/rag/` — Retriever Router, GitHub/SO/Tavily | ✅ 完了 |
| **Phase 1** | `src/sandbox/` — Docker, Security, Retry | ✅ 完了 |
| **Phase 1** | `src/orchestrator/` — FastAPI, Pipeline | ✅ 完了 |
| **Phase 1.5** | `src/knowledge_graph/` — NetworkX KG | ✅ 完了 |
| **Phase 1.5** | `src/retrieval/` — QueryClassifier, RRF Fusion | ✅ 完了 |
| **Phase 2** | Teacher 信頼度評価 (TeacherRegistry + EWMA) | ✅ 完了 |
| **Phase 2** | MemoryReviewer, DifficultyTagger, QualityMetrics | ✅ 完了 |
| **Phase 2** | CrossEncoder 再ランキング | ✅ 完了 |
| **Phase 2** | GUI 成熟管理タブ | ✅ 完了 |
| **Phase 2** | `src/mcp_tools/` — SQL/BI MCP | ✅ 完了 |
| **GUI** | 5タブ Gradio WebGUI | ✅ 完了 |
| **GUI** | セットアップウィザード + ガイドタブ | ✅ 完了 |
| **GUI** | プロバイダープリセット + カスタム設定 | ✅ 完了 |
| **ドキュメント** | MkDocs サイト (案B) | ✅ 完了 |
| **Phase 3** | GRPO + TinyLoRA 学習パイプライン（骨格） | ✅ 骨格完了 |

## 残作業

### 優先度: 高

| 項目 | 説明 |
|------|------|
| **統合テスト** | `tests/integration/` に Docker ベースの E2E テストを追加 |
| **動作確認** | 実際にシステムを起動してエンドツーエンドの動作を検証 |

### 優先度: 中

| 項目 | 説明 |
|------|------|
| **ドキュメント案C** | `guide.py` タブの FAISS ガイドチャットBot |
| **スクリプト整備** | `scripts/` の各起動スクリプトを確認・完備 |

### 優先度: 低（将来フェーズ）

| フェーズ | 項目 |
|---------|------|
| **Phase 3** | KG パスを Teacher プロンプトに含めて CoT 強化 |
| **Phase 3** | GRPO 報酬関数に KG 整合性スコアを追加 |
| **Phase 3** | 評価指標に Entity 精度・関係再現率を追加 |
| **Phase 4** | ModelRouter の KG 参照ロジックを完備 |
| **Phase 4** | 拡張アルゴリズム (PPO, DPO) の本番対応 |
| **Phase 4** | ベンチマーク整備と最適化 |
| **Neo4j 移行** | NetworkX → Neo4j 移行スクリプト |
| **PostgreSQL 移行** | SQLite → PostgreSQL |

## バージョン計画

| バージョン | 内容 | 目標時期 |
|----------|------|---------|
| v0.1 | Phase 1 MVP | Week 3 |
| v0.2 | Phase 1.5 + KG | Week 5 |
| **v0.3（現在）** | Phase 2 成熟管理 + GUI 完備 | Week 7 |
| v0.4 | 統合テスト + 動作確認 | Week 8 |
| v1.0 | Phase 3 学習フレームワーク本番稼働 | Week 12 |
