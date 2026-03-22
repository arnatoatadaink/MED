# TODO.md — MED フレームワーク 残作業一覧

> 最終更新: 2026-03-22（C. 動作確認完了 — オーケストレーター+3スクリプト修正・E2E通過）
> 参照元: `CLAUDE.md` / `plan.md` / `plan_think.md` / `plan_test.md` / `plan_training_a.md` / `plan_training_b.md` / `docs/session_progress.md`

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

### A-1. 会話履歴の永続化 + ユーザー管理 ✅ **完了（サーバーサイド）**
> サーバーサイド SQLite + JWT 認証による完全実装。

**認証モジュール（`src/auth/`）**
- ✅ `src/auth/schema.py` — User / TokenPayload / LoginRequest / RegisterRequest / TestTokenRequest / TokenResponse
- ✅ `src/auth/store.py` — UserStore (aiosqlite CRUD)
- ✅ `src/auth/service.py` — AuthService (bcrypt + python-jose JWT)
- ✅ `src/auth/deps.py` — FastAPI 依存注入（get_current_user / get_optional_user / require_localhost）

**会話履歴モジュール（`src/conversation/`）**
- ✅ `src/conversation/schema.py` — Session / Turn データクラス
- ✅ `src/conversation/store.py` — ConversationStore (aiosqlite, CASCADE DELETE, WAL)
- ✅ `src/conversation/manager.py` — ConversationManager (セッション上限, トークンウィンドウ, FAISS自動登録)

**統合・エンドポイント**
- ✅ `src/common/config.py` — AuthConfig / ConversationConfig 追加
- ✅ `configs/default.yaml` — auth / conversation セクション追加
- ✅ `src/orchestrator/pipeline.py` — user_id / session_id 統合、会話履歴コンテキスト注入
- ✅ `src/orchestrator/server.py` — /auth/* / /sessions/* / /admin/* エンドポイント追加
- ✅ `src/gui/tabs/chat.py` — セッション選択ドロップダウン・履歴復元 UI 追加
- ✅ `scripts/seed_test_users.py` — テストユーザー登録スクリプト
- ✅ `tests/unit/test_auth.py` — 認証テスト（TestRegister/Login/TestToken/JWT/UserStore）
- ✅ `tests/unit/test_conversation.py` — 会話履歴テスト（CASCADE/トークン窓/時系列順）

**残作業（ブラウザローカルストレージ、オプション）**
- 🟡 `src/gui/tabs/chat.py` — セッション開始時にローカルストレージから履歴を復元する JS を注入（サーバーAPIで代替済みのため低優先度）
- 🟡 GradioのJavaScript APIを使った実装方式の選定（`gr.HTML` + JS）

---

### A-2. Teacher思考過程の抽出・保存（ReasoningTrace）
> 📄 詳細: `plan_think.md`

**Step 1: スキーマ追加** ✅ **完了**
- ✅ `src/memory/schema.py` — `KnowledgeType` / `TraceMethod` 列挙型を追加
- ✅ `src/memory/schema.py` — `ReasoningTrace` Pydantic モデルを追加

**Step 2: LLMレイヤー拡張** ✅ **完了**
- ✅ `src/llm/gateway.py` — `LLMResponse` に `thinking_text: str | None` / `thinking_tokens: int` を追加
- ✅ `src/llm/gateway.py` — `BaseLLMProvider.complete()` に `enable_thinking: bool = False` を追加
- ✅ `src/llm/providers/anthropic.py` — Extended Thinking API（`thinking={"type":"enabled","budget_tokens":N}`）対応

**Step 3: ストレージ** ✅ **完了**
- ✅ `src/memory/metadata_store.py` — `reasoning_traces` テーブル追加
- ✅ `src/memory/metadata_store.py` — `trace_documents`（多対多）テーブル追加
- ✅ `src/memory/memory_manager.py` — `save_reasoning_trace()` メソッド追加

**Step 4: 抽出ロジック** ✅ **完了**
- ✅ `src/llm/prompt_templates/reasoning_extraction.yaml` — CoT抽出プロンプト（非Anthropicプロバイダ用）
- ✅ `src/llm/thinking_extractor.py` — `ThinkingExtractor` クラス新規作成（プロバイダ別に抽出方式を切替）

**Step 5: テスト** ✅ **完了**
- ✅ `tests/unit/test_thinking_extractor.py` — 30テスト全通過（Extended Thinking / CoTパース / SQLite CRUD）

**後続フェーズ（任意）**
- 🟢 `src/orchestrator/pipeline.py` — Teacher呼び出し後に思考過程を自動保存するフック
- 🟢 `src/gui/tabs/chat.py` — デバッグパネルに thinking_text を表示
- 🟢 FAISSへの `reasoning` ドメイン新設（judgment_criteriaを検索可能に）

---

## B. CI/CD 改善
> 📄 詳細: `plan_test.md`

### B-0. 即効修正（CI 6h → 20分） ✅ **完了**
- ✅ `.github/workflows/ci.yml` — 各ジョブに `timeout-minutes` 追加（lint:5 / unit-tests:20 / docker-tests:30）
- ✅ `.github/workflows/ci.yml` — `unit-tests` ジョブの `pip install` から `sentence-transformers` を削除
- ✅ `.github/workflows/ci.yml` — `docker-tests` ジョブの重複 pytest ステップ削除済み

### B-1〜4. testmon + xdist 移行
- ✅ `pyproject.toml` — `pytest-testmon>=2.2` を dev dependency に追加（poetry add 済み）
- ✅ `pyproject.toml` — `asyncio_default_fixture_loop_scope` / `filterwarnings` 追加（Event loop is closed 警告修正）
- ✅ testmon ローカル動作確認済み（変更なし→0件/0.13s、embedder変更→68件/12s）
- 🟡 `Dockerfile.test` — testmon/xdist 入り軽量イメージに修正
- 🟡 `docker-compose.test.yml` — `.testmondata` ボリュームマウント設定
- 🟡 `.github/workflows/test.yml` — testmon差分収集 → xdist並列実行ワークフロー作成
- 🟡 `.github/workflows/test-full.yml` — 週次フルラン + `.testmondata` 再生成ワークフロー作成

---

## C. 動作確認・統合テスト

- ✅ オーケストレーター起動 E2E 動作確認完了
  - `/health` / `/stats` / `/auth/*` / `/sessions/*` / `/add` / `/query` 全正常
  - Haiku でのクエリ応答・FAISSコンテキスト付き回答生成を確認
- ✅ `tests/integration/` Docker ベースの E2E テスト全通過（Docker内: 1096件 / ローカル: 1096件）
  - `test_docker_sandbox.py` — 17 passed（コンテナ実行・セキュリティ・タイムアウト・並行実行）
  - `test_e2e_pipeline.py` — 49 passed（CRUD / FastAPI / 認証 / セッション / 管理者）
- ✅ `scripts/seed_memory.py` — 修正・動作確認完了（RetrieverRouter/Document/SourceMeta API修正）
- ✅ `scripts/mature_memory.py` — 修正・動作確認完了（--check/--review/--tag-difficulty 全正常）
- ✅ `scripts/train_student.py` — 修正・動作確認完了（SeedBuilder.build() API修正、dry-run成功）

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

## I. 面接形式テスト・多ターン訓練拡張
> 📄 詳細: `plan_training_b.md`

### Phase B-1: データ品質層の基盤 — ✅ **完了**

- ✅ `src/training/pipeline.py` — `TrainingDataGate` + `GateConfig` 追加
- ✅ `src/training/algorithms/grpo.py` — StarPO-S 分散フィルタ + 非対称クリッピング追加
- 🟡 `src/memory/maturation/difficulty_tagger.py` — 動的カーリキュラム調整
  - 損失推移を監視して難易度配分をリアルタイム変更

### Phase B-2: 評価フレームワーク拡張 — ✅ **完了**

- ✅ `src/training/evaluation/interview_evaluator.py` — `InterviewEvaluator` (圧迫深掘りテスト)
- ✅ `src/training/evaluation/multi_challenge_evaluator.py` — `MultiChallengeEvaluator` (長期指示維持4カテゴリ)
- ✅ `src/training/evaluation/assumption_correction_evaluator.py` — `AssumptionCorrectionEvaluator` (MEDオリジナル)
- ✅ `tests/unit/test_interview_evaluators.py` — 23テスト全通過
- ✅ `src/training/evaluation/benchmark_suite.py` — mtRAG ベンチマーク統合

### Phase B-3: 訓練アルゴリズム拡張 — ✅ **完了**

- ✅ `src/training/algorithms/refuel.py` — REFUEL アルゴリズム (Q値差分回帰)
- ✅ `src/training/rewards/composite.py` — CURIO 情報利得報酬 (`curio_coef` パラメータ)
- ✅ `configs/training.yaml` — `starpo_s` / `refuel_tinylora` / `grpo_curio` プロファイル追加

### Phase B-4: 統合・モニタリング

- 🟢 `src/knowledge_graph/router_bridge.py` — KGカバレッジ監視フック追加
  - エンティティ経路の多様性スコアを計算し Echo Trap 早期検出シグナルに
  - `多様性スコア = ユニークエンティティ数 / 総エンティティ参照数 < θ` で警告
- 🟢 `src/memory/learning/cross_encoder.py` — Cross-Encoder疑似報酬モードを追加
  - Teacher API を毎ステップ呼ばずに Cross-Encoder でコスト削減
  - N ステップに1回 Teacher API で品質を補正するハイブリッド運用
- 🟢 `src/training/evaluation/` — IQA-EVAL ペルソナ別評価自動化
  - `src/memory/maturation/seed_builder.py` と連携してペルソナ付きテストデータを生成

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
| CI: GitHub Actions + ruff + pytest 1096テスト (unit 1030 + integration 66) | ✅ |
| A-1: src/auth/ + src/conversation/ + JWT + セッション管理 | ✅ |
| Docker統合テスト: sandbox 17件 + E2E pipeline 49件 全通過 | ✅ |
| testmon: pytest-testmon 2.2.0 導入・ベースライン記録済み | ✅ |
| pytest警告修正: asyncio loop_scope + filterwarnings 設定 | ✅ |
| C. 動作確認: オーケストレーター + seed/mature/train スクリプト修正・E2E通過 | ✅ |
