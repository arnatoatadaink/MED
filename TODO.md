# TODO.md — MED フレームワーク 残作業一覧

> 最終更新: 2026-03-19
> 参照元: `CLAUDE.md` / `plan.md` / `plan_think.md` / `plan_test.md` / `docs/session_progress.md` / `docs/site/dev/roadmap.md`

---

## 凡例

| 記号 | 意味 |
|------|------|
| 🔴 | 優先度: 高（次セッションで着手） |
| 🟡 | 優先度: 中（近いうちに） |
| 🟢 | 優先度: 低（将来フェーズ） |
| 📄 | 詳細計画書あり |

---

## A. 新機能 — 今セッションで設計したもの

### A-1. 会話履歴の永続化（ローカルストレージ）
> Web利用時のユーザー利便性向上。タブを閉じても履歴が残る。

- 🔴 `src/gui/tabs/chat.py` — セッション開始時にローカルストレージから履歴を復元する JS を注入
- 🔴 `src/gui/tabs/chat.py` — 各ターン後にローカルストレージへ履歴を書き込む JS を注入
- 🟡 GradioのJavaScript APIを使った実装方式の選定（`gr.HTML` + JS or Gradio JS event）
- 🟡 保存形式の設計（JSON: `[{role, content, timestamp, sources}]`）
- 🟡 最大保持件数・TTLの設定（例: 最新100ターン / 30日）

---

### A-2. Teacher思考過程の抽出・保存（ReasoningTrace）
> 📄 詳細: `plan_think.md`

**Step 1: スキーマ追加**
- 🔴 `src/memory/schema.py` — `KnowledgeType` / `TraceMethod` 列挙型を追加
- 🔴 `src/memory/schema.py` — `ReasoningTrace` Pydantic モデルを追加

**Step 2: LLMレイヤー拡張**
- 🔴 `src/llm/gateway.py` — `LLMResponse` に `thinking_text: str | None` / `thinking_tokens: int` を追加
- 🔴 `src/llm/gateway.py` — `BaseLLMProvider.complete()` に `enable_thinking: bool = False` を追加
- 🔴 `src/llm/providers/anthropic.py` — Extended Thinking API（`thinking={"type":"enabled","budget_tokens":N}`）対応
  - 注意: Extended Thinking 有効時は `temperature=1.0` 固定（API要件）
  - 注意: `claude-opus-4-6` 以上が必要（Haiku非対応）

**Step 3: ストレージ**
- 🔴 `src/memory/metadata_store.py` — `reasoning_traces` テーブル追加
- 🔴 `src/memory/metadata_store.py` — `trace_documents`（多対多）テーブル追加
- 🔴 `src/memory/memory_manager.py` — `save_reasoning_trace()` メソッド追加

**Step 4: 抽出ロジック**
- 🔴 `src/llm/prompt_templates/reasoning_extraction.yaml` — CoT抽出プロンプト（非Anthropicプロバイダ用）
- 🔴 `src/llm/thinking_extractor.py` — `ThinkingExtractor` クラス新規作成（プロバイダ別に抽出方式を切替）

**Step 5: テスト**
- 🟡 `tests/unit/test_thinking_extractor.py` — Extended Thinking / CoTパース / ReasoningTrace検証
- 🟡 `tests/unit/test_reasoning_store.py` — SQLite保存・取得・JSONシリアライズ

**後続フェーズ（任意）**
- 🟢 `src/orchestrator/pipeline.py` — Teacher呼び出し後に思考過程を自動保存するフック
- 🟢 `src/gui/tabs/chat.py` — デバッグパネルに thinking_text を表示
- 🟢 FAISSへの `reasoning` ドメイン新設（judgment_criteriaを検索可能に）

---

## B. CI/CD 改善
> 📄 詳細: `plan_test.md`

### B-0. 即効修正（CI 6h → 20分）
- 🔴 `.github/workflows/ci.yml` — 各ジョブに `timeout-minutes` 追加（lint:5 / unit-tests:20 / docker-tests:30）
- 🔴 `.github/workflows/ci.yml` — `unit-tests` ジョブの `pip install` から `sentence-transformers` を削除
- 🔴 `.github/workflows/ci.yml` — `docker-tests` ジョブの重複 pytest ステップを削除（"Run unit tests in Docker" / "Extract coverage report"）

### B-1〜4. testmon + xdist 移行
- 🟡 `requirements-dev.txt` — `pytest-testmon>=2.1` / `pytest-xdist>=3.5` を追記
- 🟡 `Dockerfile.test` — sentence-transformers 除外 + testmon/xdist 入り軽量イメージに修正
- 🟡 `docker-compose.test.yml` — `.testmondata` ボリュームマウント設定
- 🟡 `.github/workflows/test.yml` — testmon差分収集 → xdist並列実行ワークフロー作成
- 🟡 `.github/workflows/test-full.yml` — 週次フルラン + `.testmondata` 再生成ワークフロー作成

---

## C. 動作確認・統合テスト

- 🔴 オーケストレーターを実際に起動してエンドツーエンドの動作確認
  ```bash
  cd MED && uvicorn src.orchestrator.server:app --reload
  ```
- 🔴 `tests/integration/` に Docker ベースの E2E テストを追加・実行
- 🟡 `scripts/seed_memory.py` — 動作確認（初期ドキュメントのFAISS投入）
- 🟡 `scripts/mature_memory.py` — 動作確認（Teacher品質審査パイプライン）
- 🟡 `scripts/train_student.py` — 動作確認（GRPO + TinyLoRA骨格）

---

## D. Knowledge Graph フェーズ2（永続化）
> 📄 詳細: `plan.md` Phase 2

- 🟡 `src/knowledge_graph/` — NetworkX → Neo4j 移行スクリプト作成
- 🟡 `pyproject.toml` — `neo4j>=5.0` を optional dependency として追加
- 🟢 KGスキーマの MED 特化設計（汎用 vs MED特化 Entity型の決定）
- 🟢 Neo4j 統合テスト追加

---

## E. 学習フレームワーク 本番稼働（Phase 3+）
> 現状: 骨格実装のみ。VERL/trl との実際の統合が未完

- 🟢 `src/training/algorithms/grpo.py` — VERL/trl との実際の統合
- 🟢 `src/training/adapters/tinylora.py` — `frozen_rank=2, proj=4, tie=7` の本番チューニング
- 🟢 KG パスを Teacher へのプロンプトに含めて CoT 強化
- 🟢 GRPO 報酬関数に KG 整合性スコアを追加
- 🟢 評価指標に Entity 精度・関係再現率を追加
- 🟢 拡張アルゴリズム（PPO, DPO）の本番チューニング

---

## F. メモリ品質目標（運用フェーズ）

- 🟢 `data/faiss_indices/` へのシードデータ投入（目標: 10,000 docs）
- 🟢 confidence > 0.7 の達成
- 🟢 コード実行成功率 > 80% の達成
- 🟢 `src/memory/maturation/seed_builder.py` — Teacher API 呼び出し部分（現状スタブ）を実接続

---

## G. インフラ移行（将来フェーズ）

- 🟢 SQLite → PostgreSQL 移行スクリプト
- 🟢 KG: NetworkX + pickle → Neo4j 移行（plan.md Phase 2）
- 🟢 埋め込みモデル: `all-MiniLM-L6-v2` → `UniXcoder` 移行評価
- 🟢 将来評価: Cognee / Weaviate（FAISS + KG 統合候補）

---

## H. ドキュメント・スクリプト整備

- 🟡 `scripts/` 各スクリプトのヘルプ文・引数整備（`argparse`）
- 🟡 `docs/site/dev/roadmap.md` — 本 TODO に合わせて更新
- 🟢 OpenAI o1/o3 系の `reasoning_content` フィールド対応調査（A-2 の拡張）

---

## 完了済み参照（変更不要）

以下は実装済み。混同しないよう記載。

| モジュール | 状態 |
|-----------|------|
| Phase 1: config / memory / llm / rag / sandbox / orchestrator | ✅ |
| Phase 1.5: knowledge_graph / retrieval (KG prototype) | ✅ |
| Phase 2: maturation / cross_encoder / teacher_registry / mcp_tools | ✅ |
| Phase 3: training 骨格（base / algorithms / adapters / rewards） | ✅ 骨格 |
| Phase 4: model_router / query_parser / error_analyzer / deduplicator | ✅ |
| GUI: Gradio 6タブ + docs_chat | ✅ |
| CI: GitHub Actions + ruff + pytest 862テスト | ✅ |
