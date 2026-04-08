# TODO.md — MED フレームワーク 残作業一覧

> 最終更新: 2026-04-09
> 参照元: `CLAUDE.md` / `plan.md` / `plan_translate.md` / `plan_version_aware.md` / `plan_neat_hyp_e.md` / `plan_programming_seed.md`

---

## 凡例

| 記号 | 意味 |
|------|------|
| 🔴 | 優先度: 高（次セッションで着手） |
| 🟡 | 優先度: 中（近いうちに） |
| 🟢 | 優先度: 低（将来フェーズ） |
| 📄 | 詳細計画書あり |

---

## A. 新機能 — 実装済み

### A-1. 会話履歴の永続化 + ユーザー管理 ✅ **完了**
- ✅ `src/auth/` — User / JWT 認証（bcrypt + python-jose）
- ✅ `src/conversation/` — Session / Turn / ConversationManager
- ✅ `src/orchestrator/server.py` — /auth/* / /sessions/* / /admin/* エンドポイント
- ✅ `src/gui/tabs/chat.py` — セッション選択・履歴復元 UI
- 🟢 ブラウザローカルストレージからの履歴復元（サーバーAPIで代替済みのため低優先度）

### A-2. Teacher思考過程の抽出・保存（ReasoningTrace）✅ **完了**
- ✅ `src/memory/schema.py` — KnowledgeType / TraceMethod / ReasoningTrace
- ✅ `src/llm/gateway.py` — thinking_text / enable_thinking 対応
- ✅ `src/llm/providers/anthropic.py` — Extended Thinking API 対応
- ✅ `src/memory/metadata_store.py` — reasoning_traces / trace_documents テーブル
- ✅ `src/llm/thinking_extractor.py` — ThinkingExtractor（プロバイダ別抽出）
- 🟢 pipeline.py に思考過程自動保存フック追加
- 🟢 GUI デバッグパネルに thinking_text 表示

---

## B. CI/CD

### B-0. CI 高速化 ✅ **完了**
- ✅ `timeout-minutes` 追加 / sentence-transformers 除去 / 重複 pytest 削除

### B-1〜4. testmon + xdist 移行
- ✅ `pytest-testmon>=2.2` 導入・ローカル動作確認済み
- 🟡 `Dockerfile.test` — testmon/xdist 入り軽量イメージ
- 🟡 `.github/workflows/test.yml` — testmon差分 → xdist並列実行ワークフロー
- 🟡 `.github/workflows/test-full.yml` — 週次フルラン + `.testmondata` 再生成

---

## C. 動作確認・統合テスト ✅ **完了**
- ✅ オーケストレーター E2E 動作確認（/health / /query / /auth/* 全正常）
- ✅ `tests/integration/` Docker E2E 全通過（unit 1030件 + integration 66件）
- ✅ seed / mature / train スクリプト動作確認済み

---

## D. Knowledge Graph ✅ **完了（Phase 1.5）**
- ✅ `src/knowledge_graph/store.py` — ABC + NetworkX + Neo4j バックエンド
- ✅ `src/knowledge_graph/extractor.py` / `router_bridge.py`
- ✅ `src/knowledge_graph/migration.py` — NetworkX↔Neo4j↔JSON 双方向移行
- 🟢 KGスキーマの MED 特化設計（汎用 vs MED特化 Entity型の決定）
- 🟢 Neo4j 永続化本番移行（現状: NetworkX + pickle）

---

## E. 学習フレームワーク（Phase 3+）
> 現状: 骨格実装のみ。VERL/trl との実際の統合が未完

- 🟢 `src/training/algorithms/grpo.py` — VERL/trl 実統合
- 🟢 `src/training/adapters/tinylora.py` — 本番チューニング（frozen_rank=2, proj=4, tie=7）
- 🟢 KG パスを Teacher プロンプトに含めて CoT 強化
- 🟢 GRPO 報酬関数に KG 整合性スコアを追加
- 🟢 拡張アルゴリズム（PPO, DPO）本番チューニング

### Phase B（訓練拡張）✅ **骨格完了**
- ✅ `TrainingDataGate` / StarPO-S / CurriculumController
- ✅ `InterviewEvaluator` / `MultiChallengeEvaluator` / `AssumptionCorrectionEvaluator`
- ✅ REFUEL アルゴリズム / CURIO 情報利得報酬
- 🟢 KGカバレッジ監視フック（Echo Trap 早期検出）
- 🟢 Cross-Encoder 疑似報酬モード（Teacher API コスト削減）
- 🟢 IQA-EVAL ペルソナ別評価自動化

---

## F. メモリ品質目標（シード継続）
> 📄 `plan_programming_seed.md`

**現状: approved 4,834件 / FAISS code 6,813 vectors（2026-04-09）**

### F-1. 日次 seed_and_mature ジョブ 🔴
- 🔴 **Apr 9 ジョブ起動**（UTC 00:00 / JST 09:00以降・gemma-4-31b-it:free）
  ```bash
  poetry run python scripts/seed_and_mature.py \
    --questions scripts/questions.txt \
    --exclude-sources tavily \
    --top-k 5 --limit 150
  ```
- OpenRouter 日次上限: 950件/UTC日。毎日 JST 09:00 以降に起動
- 目標: approved **10,000件**

### F-2. seed_from_docs.py 本番実行 🔴
> 📄 `plan_programming_seed.md` カテゴリ I〜L（見込み 2,150〜4,200件）
- 🔴 GitHub ドキュメントリポジトリ（tldr-pages / Node.js / cpython / MDN）
  ```bash
  poetry run python scripts/seed_from_docs.py --source github_docs --max-files 100 --dry-run
  poetry run python scripts/seed_from_docs.py --source github_docs --max-files 100 --mature --provider openrouter
  ```
- 🔴 URLリスト（Arch Wiki / Python docs / Linux Command Line）
  ```bash
  poetry run python scripts/seed_from_docs.py --source url_list --mature --provider openrouter
  ```

### F-3. needs_update 再mature 🟡
- 現状: needs_update **188件**（arXiv中心）
  ```bash
  poetry run python scripts/remature_needs_update.py --provider fastflowlm --limit 200
  ```

### F-4. seed_blacklist ✅ **完了（2026-04-09）**
- ✅ `src/memory/metadata_store.py` — `seed_blacklist` テーブル追加
- ✅ `reviewer.py` — rejected 判定時に自動登録
- ✅ `seed_and_mature.py` — fetch後・dedup前にblacklistチェック
- ✅ `seed_from_docs.py` — Phase 1.5 にblacklistフィルタ挿入
- 現状: 172件登録済み（既存rejected文書から自動投入）

---

## G. ローカル Teacher 設定 ✅ **完了（2026-04-09）**

### FastFlowLM (NPU)
- ✅ `configs/llm_config.local.yaml` — `fastflowlm` プロバイダー追加
  - `qwen3.5:9b`（Q4_1・NPU）採用
  - IFBench: 9b Q4_1≈57% / 4b Q4_1≈50% / 2b Q4_1≈35%
  - reviewer用途は9b推奨（4bはJSON失敗率増のリスク）
- ✅ 全32モデルベンチマーク完了（decode速度 / JSON出力品質）

### LM Studio
- ✅ `configs/llm_config.local.yaml` — `lmstudio` プロバイダー設定済み
  - `qwen3.5-9b`（BF16 IFBench 64.5%）推奨

---

## H. 多言語対応 🟡
> 📄 `plan_translate.md`

- ✅ 日本語 manual 5件 英訳 + FAISS 再エンベッド済み
- ✅ 多言語対応方針決定（10,000 docs 達成後に移行）
- 🟡 **バックアップ**: `cp -r data/faiss_indices/ data/faiss_indices_minilm_backup/`（移行前に実施）
- 🟡 **`scripts/reindex_faiss.py` 作成**（未実装）
- 🟡 `configs/default.yaml` の embedding model を `paraphrase-multilingual-MiniLM-L12-v2` に変更

---

## I. バージョン対応知識管理 🟡
> 📄 `plan_version_aware.md`

- 🟡 **Step 1**: `src/memory/schema.py` に version フィールド追加
  - `version_status: str = "unknown"` / `tech_name` / `version_introduced` / `version_deprecated` / `version_removed`
  - `src/memory/metadata_store.py` に ALTER TABLE マイグレーション追加
- 🟢 **Step 2**: KG バージョンノード設計（introduced_in / deprecated_in / removed_in / replaced_by）
- 🟢 **Step 3**: バージョン対応検索フロー（クエリからバージョン抽出 → フィルタ）

---

## J. データ世代管理 ✅ **完了**
- ✅ restic + NAS バックアップ基盤
- ✅ `scripts/backup_data.sh` / `poetry_run_backup.bat`
- 🟢 定期バックアップ（cron / タスクスケジューラ）
- 🟢 保持ポリシー（`restic forget --keep-last 10 --keep-daily 7 --keep-weekly 4 --prune`）

---

## K. CRAG Query Rewriter ✅ **完了**
- ✅ QueryRewriter（4戦略: rule_expand / flan_t5 / qwen / llm）
- ✅ FLAN-T5-small / Qwen2.5-0.5B-Instruct DL済み
- ✅ タイムアウト伝播（GUI→FastAPI→Pipeline→Gateway→全5プロバイダー）
- 🟡 訓練データ生成実行 + SFT 実行（Teacher API キー必要）
- 🟡 RL fine-tune（GRPO報酬 = FAISS検索品質スコア）

---

## L. NEAT Context-Sensitive Search 🟢
> 📄 `plan_neat_hyp_e.md`

- 🟢 **Phase 5-1**: `AssociationFn` — numpy版 MLP（3項関数: query, candidate, context）
- 🟢 **Phase 5-2**: `ContextSensitiveSearch` — FAISS k*3 候補 → association_fn リランク
- 🟢 **Phase 5-3**: MED 統合 + StyleVector 連携（`med_hyp_style_g.md`）
- 🟢 NEAT 環境検証（WSL2）: `claude_work/neat_trident`
  ```bash
  cd /mnt/d/Projects/claude_work/neat_trident
  python scripts/phase0_verify.py
  python scripts/faiss_hybrid_verify.py
  python scripts/es_hyperneat_verify.py
  python scripts/long_term_loop.py
  ```
- 🟢 NEAT × MED 統合（`neat_trident/src/med_integration/` アダプタ層設計）

---

## M. インフラ移行（将来フェーズ）🟢

- 🟢 SQLite → PostgreSQL 移行スクリプト
- 🟢 KG: NetworkX + pickle → Neo4j 本番移行
- 🟢 埋め込みモデル: all-MiniLM-L6-v2 → UniXcoder 移行評価
- 🟢 将来評価: Cognee / Weaviate（FAISS + KG 統合候補）

---

## 技術的負債

- `src/memory/maturation/seed_builder.py` — Teacher API 呼び出し部分はスタブ
- `src/training/algorithms/` — 骨格実装のみ、VERL/trl 実統合が必要
- `tests/unit/test_alias_extractor.py` — pytest-asyncio 設定問題で1件失敗（既知）

---

## 完了済みモジュール一覧

| モジュール | 状態 |
|-----------|------|
| Phase 1: config / memory / llm / rag / sandbox / orchestrator | ✅ |
| Phase 1.5: knowledge_graph / retrieval (KG prototype) | ✅ |
| Phase 2: maturation / cross_encoder / teacher_registry / mcp_tools | ✅ |
| Phase 3: training 骨格（base / algorithms / adapters / rewards） | ✅ 骨格 |
| Phase 4: model_router / query_parser / error_analyzer / deduplicator | ✅ |
| Seed拡張: github_docs_fetcher / url_list_fetcher / seed_from_docs.py | ✅ |
| Seed品質管理: seed_blacklist / remature_needs_update.py | ✅ |
| OpenRouter日次管理: daily_usage_tracker / check_usage.py | ✅ |
| GUI: Gradio 6タブ + docs_chat | ✅ |
| CI: GitHub Actions + ruff + pytest 1096テスト | ✅ |
| A-1: src/auth/ + src/conversation/ + JWT + セッション管理 | ✅ |
| A-2: ReasoningTrace / ThinkingExtractor / Extended Thinking | ✅ |
| J: restic + NAS バックアップ基盤 | ✅ |
| K: CRAG Query Rewriter 4戦略 + タイムアウト伝播 | ✅ |
