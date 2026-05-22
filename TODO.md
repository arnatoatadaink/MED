# TODO.md — MED フレームワーク 残作業一覧

> 最終更新: 2026-05-23 (Q-3b/Q-3c/RetrieverRouter episodic 統合 完了)

## 次セッション推奨タスク（優先度順）

> 最終更新: 2026-05-23（Q-3b/Q-3c 完了 → 次優先 I-Step1 または P-R4）

| 優先 | ID | 内容 | 工数感 |
|------|-----|------|--------|
| 1 | ~~**Q-1〜Q-4**~~ ✅ | エピソード記憶ゾーニング 完了 | — |
| 1 | ~~**Q-3b/Q-3c**~~ ✅ | thought_log + 会話ターン → episodic FAISS フック 完了 | — |
| 2 | ~~**P-R10**~~ ✅ | `reviewer_worker.py` 分離 完了 | — |
| 3 | **I (Step1)** | バージョン対応知識管理：`schema.py` に `version_status` 等フィールド + `ALTER TABLE` マイグレーション | 小 |
| 4 | **P-R4** | 文書側ペルソナ指定フィールド追加（`ALTER TABLE documents ADD COLUMN required_persona`） | 小 |
| 5 | **N-5 / N-9** | KG 自動更新トリガー + `SourceTrustScore` データクラス + `geoopt` 追加 | 中 |
| 6 | **B-1〜4** | CI/CD: `Dockerfile.test` + GitHub Actions workflows (testmon/xdist) | 中 |
| 7 | **R-1** | 責務マップと分離方針決定（R-0 完了により着手可能、docs/module_separation_plan.md 作成） | 中 |

> arXiv/Seeder 系タスク（F-1, F-2, N-S-0）は selfban 解除後に再開。

---
> 参照元: `CLAUDE.md` / `plan.md` / `plan_translate.md` / `plan_version_aware.md` / `plan_neat_hyp_e.md` / `plan_programming_seed.md` / `med_enhancement_seed.md` / `med_seed_papers.md`

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

### A-0. GUI 改善 ✅ **完了（2026-05-10）**
- ✅ `src/gui/tabs/reviewer.py` — 各スロットに「有効」チェックボックス追加（設定を保持したまま無効化可能）
  - `_start_review()` のシグネチャを `(en1, p1, m1, ps1, ...)` に変更
  - localStorage キー: `med-rev-slot{i}-enabled`
- ✅ `src/gui/tabs/plan.py` — ソース選択フラグ追加（GitHub/SO/Tavily/arXiv/OpenReview）
  - arXiv はデフォルト OFF（BAN 中）。5/17 以降に解除確認して手動 ON
  - `_trigger_cycle()` が `disabled_sources` を `OrchestratorConfig` に渡す
  - localStorage キー: `med-plan-src-{source}`
- ✅ `src/cycle/orchestrator.py` — `OrchestratorConfig` に `disabled_sources: frozenset[str]` 追加
- ✅ `src/cycle/query_runner.py` — `QueryRunnerConfig` に `disabled_sources` 追加、`_select_sources()` で除外
  - retrievers.yaml を書き換えない（コメント保持）
- ✅ `src/gui/app.py` — reviewer/plan の localStorage restore 更新

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
- ✅ Stop hook に `--testmon` 追加 (`.claude/settings.json`)（2026-05-09）
- ✅ **AWEP curl フックタイムアウト設定（2026-05-23）**: 不通時に最大2分ブロックしていた問題を修正
  - PreToolUse / PostToolUse: `--max-time 2 --connect-timeout 1` 追加
  - Stop: `--max-time 5 --connect-timeout 2` 追加
  - `localhost` → `127.0.0.1` に変更（IPv6先引き問題回避）
  - 正常時レスポンス ~55ms に対して2s/5sのフェイルセーフを設定
- ✅ `poetry install --extras dev` で `pytest-asyncio 1.3.0` インストール済み
  - `[project.optional-dependencies].dev` に定義されているため `--extras dev` が必須
- 🟡 `Dockerfile.test` — testmon/xdist 入り軽量イメージ
- 🟡 `.github/workflows/test.yml` — testmon差分 → xdist並列実行ワークフロー
- 🟡 `.github/workflows/test-full.yml` — 週次フルラン + `.testmondata` 再生成

### B-5. ユニットテスト 残存 15 件 ✅ **完了（2026-05-22）**

`poetry run python -m pytest tests/unit/ --testmon` で検出・全修正済み。**1094 passed**。

| グループ | 件数 | ファイル | 修正内容 |
|---------|------|---------|---------|
| 環境変数リーク | 3 | `test_config.py` | `delenv` → `setenv("")` に変更（env var が .env より優先のため） |
| モック署名不一致 | 6 | `test_llm_gateway.py` | `SuccessProvider.complete()` に `timeout=None, **kwargs` 追加 |
| ロジック変更追従 | 1 | `test_maturation.py` | 期待値 `REJECTED` → `HOLD` に更新 |
| コードパス変更 | 1 | `test_iterative_retrieval.py` | `_doc()` に `review_status=APPROVED` 追加（`mm.search()` フィルタ回避） |
| 外部プロバイダー混入 | 3 | `test_query_rewriter.py` | `monkeypatch.setenv("QWEN_PROVIDER_URL", "")` 追加（LMStudio 稼働中に qwen_available=True になっていた） |
| タイミング依存 | 1 | `test_teacher_provenance_step5.py` | `asyncio.run()` + interval=0.01s / sleep=0.5s に変更 |

**ハング・未調査テスト（除外済み）:**
- `test_memory_manager.py::TestSearch` (3件): 実行すると無限ブロック。testmon deselect 済みで通常実行に影響なし
- `test_orchestrator.py::TestMEDPipeline::test_query_with_memory` (1件): 同様。testmon deselect 済み

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

**現状: approved 11,510件 / needs_update 5,090件 / unreviewed 276件（2026-05-21）**
- ✅ approved 10,000件目標 達成済み（Apr 28確認）、現在 11,510件
- needs_update: 5,090件（F-5クリーナー修正により削減傾向）
- unreviewed: 276件（WebGUI サイクル実行で +71 件追加、mature 待ち）
- mature はローカルモデル（LM Studio / FastFlowLM）が継続稼働中

### F-1. 日次 seed_and_mature ジョブ 🟡 **WebGUI サイクル実行中**
- ✅ Apr 9: approved +302件
- ✅ Apr 10: approved +175件
- ✅ **approved 10,000件 目標達成**（2026-04-28確認）
- ⏸️ **seeder 中断 (2026-05-09)**: arXiv BAN 中。永続バックオフが `data/arxiv_backoff.db` に記録中。
  - 🟡 **2026-05-17 以降: arXiv BAN 解除確認** → 解除後に GUI Plan タブの arXiv チェックを ON にする
  - BAN 確認コマンド: `poetry run python -c "import asyncio; from src.rag.retrievers.arxiv import ArXivRetriever; print(asyncio.run(ArXivRetriever.current_backoff_state()))"`
- ✅ **WebGUI Run Cycle 初回実行（2026-05-21）**: Plan タブ → Run Cycle ボタン経由で Orchestrator サイクル完了
  - run_id: `88769ea5-ef2b-44c9-93a8-c247ab67c01e`（20/20タスク done、error=0）
  - 新規文書 **+71 件** 収集・FAISS 投入（code ドメイン: 11,486 → 11,557 vectors）
  - unreviewed: 205 → 276 件（mature 待ち）
  - OpenRouter 使用: +118 リクエスト（136/950、14%）
  - arXiv は BAN 中のため除外、SO / GitHub / OpenReview の 3 ソースで収集

#### arXiv 429 対策 TODO（2026-05-18 調査結果）
- ✅ `arxiv.py`: `Retry-After` ヘッダー読み取り対応済み（arXiv は現状ヘッダーを返さない）
- ✅ `seed_only.py`: `memory_manager.add()` 呼び出し誤り修正済み（以前から 0 件追加が続いていた）
- ✅ `seed_only.py`: `openreview` を `RATE_LIMITS` に追加済み
- ✅ `schema.py`: `SourceType.OPENREVIEW` 追加済み
- ✅ **iptestserver スタブテスト実施（2026-05-19）**: arXiv / OpenReview / SO / GitHub 全 4 ソース正常動作確認
  - OpenReview 429 バックオフ: minutes_level 0→4 昇格・RESET 後正常復帰確認済み
  - テストコマンド: `STUB=http://localhost:8002 PYTHONPATH=/mnt/d/Projects/claude_work/MED poetry run python scripts/test_retriever_stub.py`
- ✅ **seed_only.py クエリキャッシュ実装（2026-05-20）**: 同一クエリの重複送信を防止し BAN リスクを低減
  - `metadata.db` に `seed_query_log` テーブル追加（PRIMARY KEY: query_hash × source）
  - `MetadataStore.is_query_cached(query, source, ttl_days=7)` / `record_query()` 追加
  - `seed_only.py --cache-ttl-days N`（default=7, 0=無効）。0件結果でもキャッシュ記録
  - iptestserver 3シナリオ全 PASS: 初回送信2件 / 2回目0件 / ttl=0再送信2件
  - テストコマンド: `STUB=http://localhost:8002 poetry run python scripts/test_retriever_stub.py --test-cache`
- ✅ **バックオフインターバル統一（2026-05-21）**: `wait_secs()` を全 level で `multiplier * 2^level` 式に統一
  - arXiv level 0: 3s → **10s**（level1=20s, level2=40s, level3=80s→ban は変更なし）
  - OpenReview level 0: 1s → 5s（level1以上は変更なし）
  - `BACKOFF_BASE_SECS` を `arxiv.py` から削除（`persistent_backoff.wait_secs` の `base` 引数を廃止）
  - 背景: VPN 経由（NordVPN 共有 IP）での arXiv アクセスで即 429 発生。インターバル延長で再発防止。
- 🟡 **arXiv 解除後の動作確認手順**:
  1. `curl -o /dev/null -w "%{http_code}" "https://export.arxiv.org/api/query?search_query=all:FAISS&max_results=1"`
  2. 200 確認後に GUI Seeder タブの arXiv チェックを ON にしてサーバーポーリング再開
- ⏸️ **mature 後回し**: 未処理件数が少ないため優先度低。LM Studio / FastFlowLM はローカル待機中。
- OpenRouter 再活用候補: `openai/gpt-4o-mini` 系 — 厳格さが若干不足するため「切り口の異なるドメイン」のmatureに限定使用を検討
- 🟡 **OpenRouter 再活用**: GPT-OSS-120b を異なる観点でのmaturation（F-2と連携）

### F-2. seed_from_docs.py 本番実行 ⏸️ **arXiv BAN 解除待ち**
> 📄 `plan_programming_seed.md` カテゴリ I〜L（見込み 2,150〜4,200件）

**2026-04-28 現状**:
- URLリスト（Arch Wiki / Python docs / Linux Command Line）: ✅ 概ね seeded 済み
- github_docs: seeding 実施中（--mature なし、ローカルモデルで消化）
  - 追加余地大: Django(0件), TypeScript(2件), MDN JS/CSS(64件)
  - 2026-04-28 seed_from_docs.py --source github_docs --max-files 100 実行中

**man-pages 制約**:
- `mkerrisk/man-pages` は troff 形式（.1/.7）のため github_docs_fetcher 非対応
- 代替案: `man7.org` の HTML を url_list で取得（`data/doc_urls/man7org.txt` 作成が必要）
- 🟡 `data/doc_urls/man7org.txt` 作成（man1コマンド / man7概念 優先URL列挙）

- 🟡 GitHub ドキュメントリポジトリ（成熟 repos はローカル mature 待ち、arXiv BAN 解除後に再開）
  ```bash
  poetry run python scripts/seed_from_docs.py --source github_docs --max-files 100
  ```
- 🟡 questions_bridge.txt での seed 継続
  ```bash
  poetry run python scripts/seed_only.py --questions-file scripts/questions_bridge.txt
  ```

### F-3. needs_update 再mature ⏸️ **後回し（未処理少）**
- 現状: needs_update **326件**（arXiv中心）。件数少ないため次フェーズまで保留。
  ```bash
  poetry run python scripts/remature_needs_update.py --provider openrouter --model nvidia/nemotron-3-nano-30b-a3b:free --limit 200
  ```

### F-4. seed_blacklist ✅ **完了（2026-04-09）**
- ✅ `src/memory/metadata_store.py` — `seed_blacklist` テーブル追加
- ✅ `reviewer.py` — rejected 判定時に自動登録
- ✅ `seed_and_mature.py` — fetch後・dedup前にblacklistチェック
- ✅ `seed_from_docs.py` — Phase 1.5 にblacklistフィルタ挿入
- 現状: 172件登録済み（既存rejected文書から自動投入）

### F-5. Chunker 改善 — API リファレンス形式への対応 🟡（クリーナー修正済み）

**背景:** github_docs の needs_update 4,705件（Node.js 1,658件 + cpython 1,563件が主因）

**原因と修正状況:**

| 問題 | ファイル | 状態 |
|------|---------|------|
| `[関数名][]` Markdown 内部リンク | `_clean_markdown()` | ✅ 修正済み |
| `> Stability: N - ...` Node.js 固有行 | `_clean_nodejs_markdown()` | ✅ 修正済み |
| `{TypeName}` Node.js 型参照 | `_clean_nodejs_markdown()` | ✅ 修正済み（2026-04-29） |
| `~module.attr` RST チルダ参照 | `_clean_rst()` | ✅ 修正済み（2026-04-29） |
| `text <anchor>` RST インラインターゲット残留 | `_clean_rst()` | ✅ 修正済み（2026-04-29） |
| セクション境界考慮チャンク分割 | `chunk_markdown()` `min_body_lines=3` | ✅ 実装済み |

**残課題（運用）:**

- 🟡 既存 needs_update docs を新クリーナーで再投入（`rebuild_github_docs.py`）
  ```bash
  # 動作確認（dry-run）
  poetry run python scripts/rebuild_github_docs.py --dry-run --limit 5
  # 本番（全 github_docs の needs_update を再チャンク化）
  poetry run python scripts/rebuild_github_docs.py --limit 100
  ```
- 🟡 cpython/fastapi/rust の needs_update も同コマンドで再投入可能
- 🟢 `**See also:**` / `**History:**` セクションのみのチャンク除外（現状 min_body_lines でほぼカバー）

### F-7. 永続バックオフ汎用化 ✅ **完了（2026-05-09）**
- ✅ `src/rag/retrievers/persistent_backoff.py` — `arxiv_backoff.py` を汎用クラスに昇格
  - `BackoffState` dataclass + `PersistentBackoffStore(source_name, db_path)`
  - テーブル名 = `{source_name}_backoff`（arXiv は既存 `arxiv_backoff` テーブル継続使用）
  - `minutes_level` 1日1段階緩和 / `days_level` 1週1段階緩和 / ban 期間 = 2^(N-1) 日
  - 純粋関数: `wait_secs()` / `ban_days()` / `apply_relaxation()` / `is_banned()`
- ✅ `arxiv_backoff.py` 削除 → `arxiv.py` のインポートを `persistent_backoff` に更新
- ✅ `ArXivRetriever.current_backoff_state()` — クラスメソッドで状態参照可
  - BAN 確認コマンド: `poetry run python -c "import asyncio; from src.rag.retrievers.arxiv import ArXivRetriever; print(asyncio.run(ArXivRetriever.current_backoff_state()))"`

### F-8. OpenReview retriever ✅ **完了（2026-05-09）**
> semantic_map_plan_en.md Layer A（高信頼性学術ソース）実装

- ✅ `src/rag/retrievers/openreview.py` — OpenReview API v2 対応 retriever
  - 対象: ICLR / NeurIPS の accepted 論文（poster / oral / spotlight）
  - クエリ照合: title + abstract へのクライアント側スコアリング（語出現率）
  - `PersistentBackoffStore("openreview", "data/openreview_backoff.db")` で 429 永続管理
  - base=1s / multiplier=5 / ban_threshold=120s / max_level=5（API: 60req/min）
  - デフォルト venue: ICLR 2025/2024・NeurIPS 2024/2023（`retrievers.yaml` で変更可）
- ✅ `RetrieverRouter` に登録（`_SOURCE_CONCURRENCY["openreview"] = 1`）
- ✅ `configs/retrievers.yaml` に `openreview:` セクション追加
- ✅ **venue ループ rate-limit 修正（2026-05-19）**: `_do_search()` の 4 venue イテレーション間に `BACKOFF_BASE_SECS`（1s）待機を追加
  - 修正前: 4 req / 556ms ≈ 7 req/s（60 req/min 制限超過 → 429 の原因）
  - 修正後: 4 req / 3.6s ≈ 1.1 req/s → 制限内。iptestserver スタブで動作確認済み
- 🟢 ACL / EMNLP 等の venue 追加（`retrievers.yaml` の `venues:` に追記するだけ）

### F-6. レトリーバー品質改善 ✅ **完了（2026-04-28）**

UMAP分析でソース別クラスター分布を確認後、retriever層の3問題を修正。

- ✅ **P1** `src/rag/retrievers/tavily.py` — `include_raw_content` デフォルト `True` → `False`
  - 全文取得（平均7k〜最大120k文字）→ スニペット（300〜500文字）に切り替え
  - chunker への過剰負荷・関係ないページ全文混入を防止
- ✅ **P2** `src/rag/retrievers/github.py` — Contents API でファイル本文取得に全面改修
  - Code Search API でファイル特定 → Contents API（base64デコード）で実コンテンツ取得
  - `GitHubDocsFetcher._clean_content()` で拡張子別クリーニング（md/rst/text）
  - `source="github_docs"` を設定 → `chunker.py` が markdown chunker を使用
  - 旧実装の `content=item.get("path","")` （ファイルパス文字列のみ・平均78文字）を廃止
- ✅ **P3** `src/rag/retrievers/stackoverflow.py` — HTMLエンティティデコード＋ハードカット廃止
  - `import html as html_lib` 追加
  - `html_lib.unescape()` を `_strip_html()` 後に適用（`&quot;` `&amp;` 等を正しくデコード）
  - `content=body[:2000]` / `content=q_body[:2000]` のハードカットを削除 → chunker に委譲

---

## G. ローカル Teacher 設定 ✅ **完了（2026-04-09）** / OpenRouterモデル調査 ✅ **完了（2026-04-11）**

### FastFlowLM (NPU)
- ✅ `configs/llm_config.local.yaml` — `fastflowlm` プロバイダー追加
  - `qwen3.5:9b`（Q4_1・NPU）採用
  - IFBench: 9b Q4_1≈57% / 4b Q4_1≈50% / 2b Q4_1≈35%
  - reviewer用途は9b推奨（4bはJSON失敗率増のリスク）
- ✅ 全32モデルベンチマーク完了（decode速度 / JSON出力品質）

### LM Studio
- ✅ `configs/llm_config.local.yaml` — `lmstudio` プロバイダー設定済み
  - `qwen3.5-9b`（BF16 IFBench 64.5%）推奨
- ✅ **埋め込みサーバー対応（2026-05-21）**: `text-embedding-all-minilm-l6-v2` をロードして `EMBEDDING_PROVIDER_URL` で利用可能
  - `Embedder`: プローブ成功時に LMStudio 経由、10秒タイムアウト時にローカルモデルへ自動切替
  - `QueryRewriter`: Qwen2.5-0.5B-Instruct (`QWEN_PROVIDER_URL`) で正常動作確認
  - flan-t5-small GGUF は LMStudio で空レスポンスのためローカルモデル使用
- ✅ **`.env` プロバイダー URL 読み込み修正（2026-05-21）**: pydantic-settings のネスト子モデル制約により `.env` の値が子モデルにマップされない問題を修正
  - 原因: `EmbeddingConfig` / `QueryRewriterConfig` は `BaseModel`（非 `BaseSettings`）のため、フラット env var がネストフィールドに自動マップされない
  - 修正: `Settings` にトップレベルフィールドを 6 個追加（`embedding_provider_url/model`, `flan_t5_provider_url/model`, `qwen_provider_url/model`）
  - `Embedder._load_model()` / `_probe_provider()` / `QueryRewriter.initialize()` で `get_settings()` 経由の参照に統一
  - `.env` 設定: `EMBEDDING_PROVIDER_URL`, `EMBEDDING_PROVIDER_MODEL`, `FLAN_T5_PROVIDER_URL`, `FLAN_T5_PROVIDER_MODEL`, `QWEN_PROVIDER_URL`, `QWEN_PROVIDER_MODEL` がすべて正常読み込み確認済み
- ✅ **FAISS 再インデックス（2026-05-21）**: `scripts/reindex_faiss.py` 新規作成・実行完了
  - 背景: `data/faiss_indices/code/index.faiss` が破損（39MB→10.8MB 途中書き込み）、iptestserver スタブテストの `close()` で空インデックスが上書きされた
  - `reindex_faiss.py`: approved 文書を SQLite から全件取得 → LMStudio Embedder で再 embed → FAISS に直接追加（metadata.db は変更なし）
  - 再構築結果: code 11,486 / academic 7 / general 17 ベクトル復元完了（約 11 分、64件/バッチ）
  - `--dry-run` / `--limit N` / `--domain` オプション対応

### OpenRouter モデル調査
- ✅ `docs/openrouter_models.md` — 無料モデルベンチマーク・429問題・FastFlowLM評価を記録
- ✅ デフォルト変更: `nemotron-nano-12b` → `nemotron-3-nano-30b-a3b:free`（Apr10実績: 承認率65%）
- ✅ `model_rate_limits` 実装（全モデル 1 RPM・`openai_compatible.py` / `gateway.py`）

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
- ✅ **LMStudio 外部プロバイダー対応（2026-05-21）**
  - `FLAN_T5_PROVIDER_URL` / `QWEN_PROVIDER_URL` 環境変数で外部推論サーバーを指定可能
  - 起動時に 10 秒タイムアウト + 空レスポンス検出プローブ → 失敗時はローカルモデルへ自動フォールバック
  - flan-t5 GGUF は LMStudio で空レスポンス（seq2seq→causal LM 構造ミスマッチ）→ 自動ローカルフォールバック済み
  - Qwen2.5-0.5B は LMStudio `/v1/chat/completions` で正常動作確認済み
  - `configs/default.yaml` の `query_rewriter.qwen_provider_url` でも設定可能
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

---

## N. med_enhancement_seed.md 起源タスク
> 📄 `med_enhancement_seed.md` / `med_seed_papers.md`
> 論文出所: S1〜S4（note.com レビュー）+ A1〜HE3（セッション調査）
> 実装ロードマップ: Phase 1 → Phase 2 → Phase 3 の順に着手

---

### N-Phase1: 思考ログ + k値 外出し（IDEA-001, 002）

#### N-1. Structured Thought Log（IDEA-001）✅
> 根拠: S1（Context Engineering 2.0, 2510.26493）

- ✅ `thought_logs` テーブル作成（`src/memory/metadata_store.py`）
  ```sql
  CREATE TABLE thought_logs (
      id TEXT PRIMARY KEY, timestamp TEXT,
      input TEXT, reasoning JSON, output TEXT,
      reward REAL, self_eval JSON, pattern_id TEXT
  );
  ```
  - `reasoning`: `[{step, thought, confidence}]` 形式
  - `self_eval`: `{accuracy, relevance, completeness, improvement_notes}`
- ✅ `ThoughtLog` Pydantic モデル追加（`src/memory/schema.py`）
- ✅ MetadataStore CRUD: `save_thought_log()` / `get_thought_log()` / `list_thought_logs()`
  - `get_pattern_success_rate()` — パターン別成功率集計
  - `list_patterns_above_threshold()` — KG 登録候補抽出 SQL
- ✅ `SelfEval` + `SelfEvaluator` — 報酬変換パイプライン（`src/training/rewards/self_evaluator.py`）
  - 重み: accuracy×0.40 + relevance×0.35 + completeness×0.25
  - `to_reward(SelfEval) → float` / `build_log(...) → ThoughtLog`
- ✅ `PatternExtractor` — KG 自動登録（`src/training/rewards/pattern_extractor.py`）
  - `success_rate > 0.9` かつ `n >= 5` → `pattern:<id>` Entity として KG に登録
  - 既存ノードは properties 更新のみ（重複なし）
- **接続先**: A-2（ReasoningTrace ✅）/ D（KG ✅）/ E（GRPO）
- **残課題**: N-4（LLM 呼び出しによる `self_evaluate()` 実装）で Teacher との接続を完成させる

#### N-2. FAISS k-value Calibration（IDEA-002）🟡
> 根拠: S2（ICL is Provably Bayesian, 2510.10981）— k=3〜5 で指数収束 O(e^{-ck})
> **RLVR知見（S6）により優先度昇格**: k値とコンテキスト品質がStudentの「見かけの賢さ」を決定

- ✅ FAISS 取得数 `k` を `configs/default.yaml` に外出し（`rag.faiss_k: 5`）
  - `memory_manager.py` の `search()` / `search_hybrid()` が `get_settings().rag.faiss_k` を参照
- ✅ k=3/5/7/10 での検索精度比較実験スクリプト作成（`scripts/eval_faiss_k.py`）
  - 実験結果 (n=30): k=3 MRR=0.139 / k=5 MRR=0.239 ← 最高 / k=10 MRR=0.209
  - **結論: k=5 が最適（ICL Bayesian 収束理論 k=3〜5 と一致）。現設定を維持**
- 🟡 Observer（FAISS検索精度）と Solver（Student推論精度）を**独立評価**する実験設計
  - `observation_accuracy`（正しい根拠を取得できたか）と `solver_accuracy`（根拠から正しく推論できたか）を分離
  - どちらが弱いかを診断 → k値調整 vs RLVR訓練強化の意思決定に使用（N-OQ-6参照）
- **接続先**: F（シード継続）/ K（CRAG QueryRewriter）/ N-4拡張案A

---

### N-Phase2: カリキュラム + 報酬 + KG 自動更新（IDEA-003〜005）

#### N-3. Teacher Curriculum Generator（IDEA-003）🟡
> 根拠: S3（PSV self-play, 2512.18160）+ S4（Agent0, 2511.16043）+ S6（COvolve: 環境自動生成+難化）

- 🟡 `TeacherCurriculumGenerator` クラス実装（`src/memory/maturation/` 配下）
  - `generate_problem(difficulty: "easier"|"frontier"|"harder") → Problem`
  - 難易度判定は `student_success_rate` と Chance-Level Threshold（IDEA-009）を参照
  - COvolve の「Teacher=環境生成役 / Student=ポリシー役」分業と同構造
- 🟡 Verifier 実装（まずルールベース: 形式チェック + 正解照合）
  - 将来: LLM-as-judge へ移行（タイミングは Open Question N-OQ-1）
  - SC1対応: Teacher入力のサニタイズ層を Verifier 前段に挿入（教育的・仮説的表現でラッピングされた汚染入力への警戒）
- 🟡 Student 成功率 EMA トラッカー実装
  - `ema = α * current_success + (1-α) * ema_prev`（α=0.1 推奨）
  - 設計参照: hantani記事「1問題ごとに即Verify → 不明点ログ → 修正してから次へ」
- **接続先**: E（TrainingDataGate ✅）/ Phase B（CurriculumController ✅）/ N-4拡張案A

#### N-4. GRPO Reward from Self-Evaluation（IDEA-004）🟡
> 根拠: S1 + S3 + S6（Observer/Solver分業）+ S7（RLTF: テキストフィードバックRL, 2602.02482）

- 🟡 **基本実装**: `compute_reward()` （`src/training/rewards/` 配下）
  ```python
  # 仮重みは暫定値; N-OQ-2 で Ablation して最適化
  base = 0.5 * accuracy + 0.3 * relevance + 0.2 * completeness
  ```
  - Verifier 不合格時は `-1.0` を返す（早期ペナルティ）
  - `style_target` が渡された場合: `0.7 * base + 0.3 * style_score`（med_hyp_style_g.md 連携）
- 🟡 `AccuracyEvaluator` / `RelevanceEvaluator` / `CompletenessEvaluator` のスタブ実装

- 🟡 **拡張案A（Phase2優先）: Observer/Solver分離報酬** （S6 COvolve/Observer-Solver知見）
  ```python
  # observation_accuracy低 → FAISSのk値・Hyperbolic距離を改善（IDEA-002へフィードバック）
  # solver_accuracy低      → StudentのRLVR訓練を強化（IDEA-003へフィードバック）
  def compute_reward_obs_solver(observation_accuracy, solver_accuracy, verifier_result,
                                 w_obs=0.4, w_sol=0.6) -> float
  ```
  - Phase2 で先行実装。N-2の独立評価実験結果を受けて重みを調整

- 🟢 **拡張案B（Phase3以降）: テキスト批評報酬** （S7 RLTF）
  ```python
  # Teacher批評 → 改善率をスコア化 → 改善能力を1回目に転写
  def compute_reward_rltf(output, teacher_critique, second_attempt, verifier_result) -> float
  ```
  - Phase1: Teacher批評をthought_logsに記録（N-1の拡張）
  - Phase2: 批評 → 改善率をスコア化
  - Phase3: 改善能力を1回目に転写（RLTF本来）
  - MED_INTEGRATION.md Phase5の「Teacher批評フィードバックAPI」と同一

- **接続先**: E（GRPO ✅骨格）/ N-3（Verifier）/ N-2（Observer診断）/ Phase 5（StyleExtractor）

#### N-5. Knowledge Graph Auto-Update（IDEA-005）🟡
> 根拠: S1（Context Engineering 2.0）+ **SC1（AI Agent Traps, SSRN 6372438）— RAG汚染対策**

- 🟡 KG 自動更新トリガー実装（`src/knowledge_graph/` 配下）
  - `thought_logs.reward > 0.9` かつ 類似パターン未登録 → 新ノード追加
  - 重複検出: FAISS 近傍検索で cos_sim > 0.95 なら既存ノードへのエッジのみ追加
- 🟡 Hyperbolic エッジ重み実装（KGエッジ生成時のみ）
  ```python
  import geoopt
  manifold = geoopt.PoincareBall(c=1.0)
  edge_weight = 1.0 / (1.0 + manifold.dist(h_a, h_b))
  ```
  - float64 使用（数値安定性確保）。推論速度影響は N-OQ-3 で測定
- 🟡 `pyproject.toml` に `geoopt` 追加（`poetry add geoopt`）
- 🟡 **SC1対応: SourceTrustScore + Provenance記録**（自動登録前の検証ゲート）
  ```python
  @dataclass
  class SourceTrustScore:
      source_url: str
      domain_type: str   # "arxiv" | "github" | "web" | "user_input"
      provenance: str    # 出所の追跡チェーン
      sanitized: bool
      trust_score: float # 0.0〜1.0
  ```
  - `trust_score < threshold` → 自動KG登録をブロックして手動確認キューに追加
  - KGエッジ生成時に `provenance` を記録（Latent Memory Poisoning対策）
  - `thought_logs` と共に記録（N-1と連携）
- **接続先**: D（KG ✅）/ HE1（2005.02819 seed済み）/ HE2/HE3（seed済み）/ N-9（セキュリティ）

---

### N-Phase3: 曖昧さ認識 + IN-DEDUCTIVE（IDEA-008〜010）

#### N-6. Ambiguity-Aware RAG（IDEA-008）🟢
> 根拠: A1（2304.14399 seed済み）/ A2（2505.11679 seed済み）/ A3, A4（未seed）
> **S5（Transformer=BP, 2603.17063）**: grounded/ungrounded分離の理論的根拠確立

- 🟢 **grounded/ungrounded分離設計**（S5知見）
  - FAISSで根拠が得られる場合（grounded）とそうでない場合（ungrounded）を明示的に区別
  - ungroundedクエリに対して「根拠未定義」を明示する応答パスを設計
  - BP理論: 「LLMは間違っているのではなく、正誤が存在しない空間で動いている」
- 🟢 `compute_semantic_entropy(query: str) → float` 実装
  - Kuhn et al. 2023 "Semantic Uncertainty" を参照（arXiv ID 要確認: N-S-1）
- 🟢 `generate_interpretations(query: str) → list[str]` — 複数解釈生成
- 🟢 `merge_and_rerank(results_list, k) → list[SearchResult]` — RRF で統合
- 🟢 `ambiguity_aware_search()` を `RetrieverRouter` に組み込み
  - SC1対応: Webコンテンツ取込時のHTMLソース vs レンダリング差分検出（Content Injection対策）
- **接続先**: K（CRAG QueryRewriter）/ N-7/8/ S5（2603.17063 seed予定）

#### N-7. Chance-Level Threshold 再設計（IDEA-009）🟢
> 根拠: 個人実験 + L1（Gemma2）/ L2（2410.16682 seed済み）/ L3（Focal Loss）
> Status: needs-redesign（Hard版 → Soft版への移行が必要）

- 🟢 Soft版（Chance-Focal）実装:
  ```python
  weight = max(0, 1 - p * n_classes) ** gamma  # γ=2 から Ablation
  ```
- 🟢 `scale` 最適値の Ablation 実験設計（N-OQ-4）
- 🟢 `n_classes` の動的決定ロジック（タスク種別ごとに変動）
- **接続先**: N-6（閾値設計）/ N-8（ルーティング）/ E（GRPO報酬）

#### N-8. IN-DEDUCTIVE Hybrid 推論（IDEA-010）🟢
> 根拠: H1（IN-DEDUCTIVE LSHTC3）/ H2（MoE Shazeer 2017）/ H3（DID ACL 2025）
> **S5（Transformer=BP）**: 演繹パス=Attention（メッセージ伝播）、帰納パス=FFN（ベイズ更新）と構造同型
> N-6/7 実装後に着手

- 🟢 `teacher_classifier` — グループ確率を出力する分類器
  - SC1対応: Teacher演繹パスの入力段階でフィルタリング（汚染されたTeacher判定が全下流を誤誘導するリスク）
- 🟢 `inductive_deductive_search()` — 確信度 ≥ Chance-Level で演繹パス、そうでなければ帰納パス
  - 設計参照: hantani記事「レビュー依頼→Codex CLI直接実行（演繹）/ SPEC.md不足→段階的設計（帰納）」
- 🟢 IDEA-008/009/010 の統合テスト設計
- **接続先**: K（CRAG）/ L（NEAT）/ TRIDENT（ルーティング）/ S5（BP理論的根拠）

---

---

### N-9. RAG/KG セキュリティ強化（SC1 AI Agent Traps）🟡
> 根拠: SC1（Franklin et al. 2026, Google DeepMind, SSRN 6372438）
> 「自律AIエージェントがウェブを行動するとき、情報環境そのものが脆弱性になる」

SC1が特定した6種類のトラップのうちMEDに直接影響するもの:

| 攻撃種別 | 影響するコンポーネント | 対策タスク |
|---------|------------------|-----------|
| RAG Knowledge Poisoning | IDEA-005（KG自動更新） | SourceTrustScore実装（N-5に統合） |
| Latent Memory Poisoning | IDEA-005, KG全般 | KGエッジ生成時のProvenance記録（N-5に統合） |
| Contextual Learning Traps | IDEA-003（Verifier） | Teacher入力のサニタイズ層（N-3に統合） |
| Content Injection | IDEA-008（曖昧さ認識RAG） | HTMLソース vs レンダリング差分検出（N-6に統合） |
| Semantic Manipulation | IDEA-010（演繹パス） | Teacher入力フィルタリング（N-8に統合） |
| Oversight & Critic Evasion | IDEA-003（Verifier全般） | 教育的表現でラッピングされた汚染入力への警戒 |

- 🟡 **SourceTrustScore** データクラス実装（`src/memory/schema.py` に追加）— N-5と同時実施
- 🟡 **seed_from_docs.py** に trust_score チェックゲートを追加
  - `domain_type="web"` のコンテンツは trust_score 評価必須
  - `domain_type="arxiv" | "github"` は高信頼ソースとして default trust_score=0.9
- 🟡 **メモ**: SC1はSSRN 6372438のみ（arXiv未登録）→ seed不可

- **接続先**: N-5（SourceTrustScore）/ N-3（サニタイズ）/ N-6（Content Injection）/ N-8（演繹パスフィルタ）

---

### N-Seed: 未取得論文の seed 追加

#### N-S-0. 新規追加済み論文（4/27 seed可能）🟡

`data/doc_urls/med_papers.txt` 追記済み。`seed_arxiv_ids.py` で投入予定:

| 論文 | arXiv ID | セクション | 状態 |
|------|---------|-----------|------|
| Transformer = Belief Propagation | 2603.17063 | S5 ★★★★★ | ✅ txt追加済み |
| RLTF (Textual Feedback RL) | 2602.02482 | S7 ★★★★ | ✅ txt追加済み |
| AI Agent Traps (SC1) | SSRN 6372438 | SC1 ★★★★★ | ❌ arXiv未登録・seed不可 |

```bash
poetry run python scripts/seed_arxiv_ids.py
```

#### N-S-1. 未 arXiv ID 論文の調査・追加 🟡

以下は `med_seed_papers.md` に記載されているが arXiv ID が不明。調査後 `data/doc_urls/med_papers.txt` に追記して `seed_arxiv_ids.py` で投入：

| 論文 | セクション | 調査状況 |
|------|-----------|---------|
| Kuhn et al. 2023 "Semantic Uncertainty" ICLR | IDEA-008 根拠 | arXiv ID 未確認 |
| "Can LLMs Faithfully Express Their Uncertainty?" EMNLP 2024 | A3 | arXiv ID 未確認 |
| "Do LLMs Estimate Uncertainty Well?" ICLR 2025 | A4 | arXiv ID 未確認 |
| Gemma 2 Technical Report (Google DeepMind 2024) | L1 | 2408.00118 候補（要確認） |
| Focal Loss (Lin et al., ICCV 2017) | L3 | 1708.02002 候補（要確認） |
| MoE (Shazeer et al., 2017) | H2 | 1701.06538 候補（要確認） |
| DID Framework (ACL 2025) | H3 | arXiv ID 未確認 |
| ST2: "Can LLMs Identify Authorship?" (Huang 2024) | ST2 | arXiv ID 未確認 |
| S6: COvolve / Observer-Solver / Medical AI Scientist | S6 | 記事まとめのみ・個別ID要調査 |

**追加コマンド** （ID 確認後）:
```bash
# data/doc_urls/med_papers.txt に ID を追記してから
poetry run python scripts/seed_arxiv_ids.py
```

---

### N-OQ: Open Questions（調査・実験が必要）

| ID | 問い | 関連 IDEA |
|----|------|----------|
| N-OQ-1 | Verifier を ルールベース→LLM-as-judge に移行するタイミング | IDEA-003 |
| N-OQ-2 | GRPO 報酬重み最適値（0.5/0.3/0.2 は仮設定）の Ablation | IDEA-004 |
| N-OQ-3 | Hyperbolic float64 計算が推論速度に与える影響（実測） | IDEA-005 |
| N-OQ-4 | Chance-Level Threshold の `scale` 最適値 Ablation 設計 | IDEA-009 |
| N-OQ-5 | StyloMetrix の日本語対応状況確認（pip install + 動作テスト） | med_hyp_style_g.md |
| N-OQ-6 | Observer/Solver分離評価で「どちらが弱いか」が判明した場合の優先改善順序 | IDEA-002/004 |
| N-OQ-7 | k値拡張で浅いStudentモデルを補完できる上限はどこか（RLVR環境での実測） | IDEA-002 |
| N-OQ-8 | NEAT開始タイミング: Phase2のRLVRフィットネス関数が安定したら即開始でよいか | TRIDENT Phase4 |

---

## O. 埋め込み空間診断・arXiv↔実装ブリッジング

> **現状の問題**: academic=11ベクトル / code=25,389ベクトル（2026-04-27時点）
> FAISSのacademic空間とcode空間が断絶している。
> UMAPで構造を可視化してから、2つのドメインを接続するSEEDと設問を設計する。

---

### O-1. FAISS 埋め込み分布 UMAP 可視化 ✅ **完了（2026-04-27）**
> TRIDENT「埋め込み空間最適化 Task 2」の前提条件

- ✅ `scripts/visualize_faiss.py` 実装・実行完了
  - `data/analysis/faiss_umap_20260427_2154.png` / `.csv` 出力済み
  - 実行: `poetry run python scripts/visualize_faiss.py --sample 2000`

**分析結果（2026-04-27、sample n=2,035）**:

```
断絶度（academic ↔ code 重心間距離）: 5.227  ← 大きな断絶あり

code クラスター内部の分離:
  docs cluster    (右上): github_docs (+9.53, +6.27) + web_docs (+9.37, +6.06)
  research cluster(左上): arxiv (+1.01, +10.98) + tavily (+2.09) + github (+2.35)
  Q&A（中間）          : stackoverflow (+5.14, +9.69) ← 自然なブリッジ位置

academic domain (11点): 両クラスターから断絶（孤立島状態）

孤立点Top: rust-lang/book SUMMARY, tldr-pages abbr, logging.handlers
ブリッジ候補: code/arxiv の KV-Cache / 並列探索論文群
```

**主な含意**:
- arxiv論文はcodeドメインに「research cluster」として吸収されており、MEDに特化した academic domain (11点) はその外側に孤立
- stackoverflow が自然なブリッジになっている → 理論と実装の橋渡しには Q&A 形式のSEEDが有効
- ブリッジSEEDの最優先候補: ML実装チュートリアル (refactoring.guru, pytorch.org) + SO高評価回答

- ✅ `poetry add umap-learn` 追加済み
- **接続先**: TRIDENT Task 2（埋め込み空間最適化）/ O-2（SEEDに反映済み）

---

### O-2. arXiv↔実装ブリッジング SEED 設計 🟡
> O-1の可視化結果を受けて実施

#### ブリッジSEED 3種別

**種別A: 実装ドキュメント（理論の実装側）**

| 対象 | URL / arXiv | 橋渡しする概念 | 優先度 | 状態 |
|------|------------|--------------|--------|------|
| sentence-transformers公式ドキュメント | `data/doc_urls/bridge_sbert.txt` (9 URL) | 384次元埋め込み ↔ all-MiniLM-L6-v2 | 🔴 | ✅ URLリスト作成済み・seed待ち |
| FAISS公式ドキュメント | `facebookresearch/faiss` (README等 md) | ベクトル検索理論 ↔ faiss.IndexFlatIP | 🔴 | ✅ github_doc_repos.yaml 追加済み・seed待ち |
| GRPO実装解説 | `huggingface/trl/docs/source` (62 md) | GRPO報酬理論 ↔ TRL実装 | 🟡 | ✅ github_doc_repos.yaml 追加済み・seed待ち |
| geoopt使い方ガイド | GitHub/README | Poincaré Ball理論 ↔ geoopt.PoincareBall | 🟡 | 🟡 GitHubに公式docs無し・方針検討 |
| NetworkX チュートリアル | docs.networkx | KGグラフ理論 ↔ nx.DiGraph操作 | 🟡 | 🟡 rst形式・URLリスト方式が適切 |

**種別B: ペーパー+実装ペア（seed済みarXivの公式コード）**

| 論文 | arXiv | 公式実装 | 状態 |
|------|-------|---------|------|
| HypStructure | 2412.01023 ✅seed済 | github.com/... | 🟡 実装リポを追加seed |
| geoopt | 2005.02819 ✅seed済 | github.com/geoopt/geoopt | 🟡 実装リポを追加seed |
| sentence-BERT | — | sbert.net | 🟡 調査・seed |

**種別C: 変換型ドキュメント（数式↔コードのマッピング）**

以下のようなドキュメントが空間上で最も効果的なブリッジになる見込み:
- 「ICLのPosterior Variance をFAISS k値で実装するには」（S2→code実装説明）
- 「Belief Propagation のAttentionとしての実装」（S5→Transformer実装説明）
- 「Hyperbolic距離のgeooptでの計算手順」（HE1→コード）

これらは既存ドキュメントにない場合、**Teacherによる合成ドキュメント生成**で補完できる。

#### ブリッジSEED 追加コマンド（O-1後に実施）
```bash
# TRL (GRPO) + FAISS github_docs を seed（seed のみ、mature はローカルモデルで）
poetry run python scripts/seed_from_docs.py --source github_docs --max-files 70

# sentence-transformers URLリストを seed
poetry run python scripts/seed_from_docs.py --source url_list --url-file data/doc_urls/bridge_sbert.txt
```

---

### O-3. arXiv↔実装横断設問 + 設計思想設問の設計 🟡
> O-1可視化結果を受けて更新（2026-04-27）
> **plan_programming_seed.md** カテゴリ D（GoF）/ E（UML）/ F（アーキテクチャ）/ H（自然言語橋渡し）も対象

**UMAP分析からの知見**:
- stackoverflow（+5.14）が自然なブリッジ → Q&A形式設問が最も効果的
- docs cluster と research cluster を跨ぐ設問を設計する
- 「理論用語 + 実装要求」型だけでなく「設計判断 + 理由説明」型も有効（UML空間活用）

横断設問は「理論用語 + 実装要求」または「設計判断 + 根拠」を同一質問に含めることで、
FAISSがacademic・code・design知識の複数ドメインを検索する必要が生じる問いを作る。

#### 設問テンプレート（8パターン）

**パターン1: 理論→実装変換型**
```
「[論文名/理論概念] の [特定の数式/アルゴリズム] を
 Python で実装するには？コード例を含めて説明してください」

例: 「ICL の Posterior Variance O(e^{-ck}) を FAISS k=5 の検索結果で
     実際に下げられるか、サンプルコードで確認する方法は？」
```

**パターン2: 実装→理論説明型**
```
「[ライブラリ/コード] の [特定の機能] の背後にある
 理論的な根拠を説明してください」

例: 「faiss.IndexFlatIP が内積検索で cosine similarity に相当するのはなぜか、
     線形代数的に説明してください」
```

**パターン3: 比較型（論文Aと実装B）**
```
「[論文の提案手法] と [既存実装] の違いを実装レベルで説明してください」

例: 「HypStructure の双曲空間正則化と通常のL2正則化を
     geoopt と torch でそれぞれ実装した場合の違いは？」
```

**パターン4: デバッグ・最適化型**
```
「[理論的に期待される挙動] に対して [実装で観察される問題] が
 起きている。原因と対処法を教えてください」

例: 「Poincaré Ball の expmap0 で float32 を使うと NaN が出る。
     理論的な数値安定性の観点から原因と float64 への切り替え方法を説明してください」
```

**パターン5: 設計判断型（理論的根拠を要求）**
```
「[実装上の設計選択] について、[理論]の観点からどの選択が正しいか説明してください」

例: 「FAISS で k=3 と k=7 どちらを選ぶべきか。
     ICL の Bayesian 収束理論と検索コストのトレードオフを踏まえて判断してください」
```

**パターン6: UMLモデリング型**（plan_programming_seed.md カテゴリE）
```
「[システムコンポーネント群] の関係を UML [図種別] で表してください。
 [設計上の制約/原則] の観点でなぜこの構造になるか説明してください」

例: 「MEDのTeacher / Student / Verifier / KGStore の関係を
     UMLクラス図で表し、依存の向きがSOLID DIの観点からなぜこうなるか説明してください」
例: 「FAISS検索→maturation→KG登録 の流れを UMLシーケンス図で表してください」
```

**パターン7: GoFデザインパターン選択型**（plan_programming_seed.md カテゴリD）
```
「[設計上の問題] に対して GoF パターンの [X] か [Y] どちらが適切か、
 理由とコード例を示してください」

例: 「RAGの検索戦略（FAISS/Tavily/SO）を切り替えるには Strategy と Command
     どちらが適切か？provider切り替えのコード例を含めて説明してください」
例: 「LLMプロバイダー切り替えに Factory vs Abstract Factory どちらを使うべきか、
     OpenRouter/FastFlowLM/LMStudio の統合を例に説明してください」
```

**パターン8: アーキテクチャ改善型**（plan_programming_seed.md カテゴリF）
```
「[現在の実装の設計問題] を [アーキテクチャパターン] の観点から改善するには？
 変更前後のコンポーネント構造を示してください」

例: 「seed_and_mature.py がFAISS・SQLite・外部APIを直接呼ぶモノリシック構造を
     Hexagonal Architecture で改善するとどうなるか？依存の向きを示してください」
例: 「MEDのTeacher-Student設計にCQRSを適用した場合、読み書きモデルはどう分離されるか」
```

#### 設問リスト（初期40問案）

MED の現在のSEED内容（code 25k + academic 11）を踏まえた横断設問（コード実装 + 設計思想）:

| # | 設問 | 要求ドメイン |
|---|------|------------|
| Q1 | ICL Posterior Variance を下げるために k=5 で十分か検証するコードは？ | academic+code |
| Q2 | sentence-transformers の all-MiniLM-L6-v2 が 384次元を選んだ理由は？ | academic+code |
| Q3 | FAISS IndexFlatIP と IndexFlatL2 の精度差を余弦類似度の理論から説明 | academic+code |
| Q4 | geoopt PoincaréBall で float32/float64 切り替えが必要な数値的理由 | academic+code |
| Q5 | GRPO の報酬関数を TRL で実装する際の accuracy/relevance 分離方法 | academic+code |
| Q6 | Belief Propagation と Transformer Attention の対応をコードで確認するには？ | academic+code |
| Q7 | HypStructure の正則化損失を PyTorch で実装する最小コード例 | academic+code |
| Q8 | IN-DEDUCTIVE パス切り替えを FAISS の group_probs で実装する方法 | academic+code |
| Q9 | NetworkX に Hyperbolic エッジ重みを追加するとグラフ探索がどう変わるか | academic+code |
| Q10 | Focal Loss の γ=2 を Chance-Level Threshold に変換するコードは？ | academic+code |
| Q11 | CRAG Query Rewriter の flan-t5 モデルを locally ロードする最短コード | code+general |
| Q12 | FAISS に add する前の L2 正規化が inner_product = cosine になる数学的証明 | academic+code |
| Q13 | Semantic Entropy を LLM の複数出力から計算する Python 実装 | academic+code |
| Q14 | aiosqlite で thought_logs を非同期挿入するベストプラクティス | code |
| Q15 | EMA トラッカー（α=0.1）を numpy で実装してStudent成功率を追跡するコード | code+academic |
| Q16 | TensorNEAT の NeatIndexer を FAISS の DomainIndex に接続するアダプタ実装 | code |
| Q17 | geoopt manifold.dist の計算コストをプロファイルして float64 影響を測定するスクリプト | code+academic |
| Q18 | Transformer の層数と推論深さの対応（BP理論）を実験的に確認するには？ | academic+code |
| Q19 | RAG で grounded/ungrounded を自動分類する FAISS 類似度閾値の設定方法 | academic+code |
| Q20 | KG の trust_score < 0.5 の文書を SQLite でフィルタリングするクエリ | code |

**パターン6〜8（設計思想）: UML・GoF・アーキテクチャ設問**

| # | 設問 | パターン | 要求ドメイン |
|---|------|---------|------------|
| Q21 | MEDのTeacher/Student/Verifier/KGStoreの関係をUMLクラス図で表し依存の向きを説明 | P6: UML | design+code |
| Q22 | FAISS検索→maturation→KG登録 の流れをUMLシーケンス図で表してください | P6: UML | design+code |
| Q23 | RAGの検索戦略切り替えにStrategy vs Commandどちらが適切か、FAISS/Tavilyの例で | P7: GoF | design+code |
| Q24 | LLMプロバイダー切り替えにFactory vs Abstract Factoryどちらを使うべきか | P7: GoF | design+code |
| Q25 | Decorator パターンを使って FAISS 検索にログ記録とキャッシュを透過的に追加するには | P7: GoF | design+code |
| Q26 | ObserverパターンでKG自動更新トリガー（thought_logs.reward > 0.9）を実装するには | P7: GoF | design+code |
| Q27 | seed_and_mature.py のモノリシック構造を Hexagonal Architecture で改善するとどうなるか | P8: Arch | design+code |
| Q28 | MEDのTeacher-Student設計にCQRSを適用すると読み書きモデルはどう分離されるか | P8: Arch | design+code |
| Q29 | SOLID原則のうち依存性逆転(DIP)をFAISSIndexとDomainIndexの設計で説明してください | P8: Arch | design+code |
| Q30 | Clean Architectureの「依存の向き」をMEDの src/memory / src/rag / src/llm で示してください | P8: Arch | design+code |
| Q31 | GoFのProxyパターンを使ってFAISSインデックスにアクセス制御を追加するコード例 | P7: GoF | design+code |
| Q32 | Template Methodパターンでseed_and_mature / seed_from_docs の共通骨格を抽出するには | P7: GoF | design+code |
| Q33 |状態機械図（State Machine）でmaturationのreview_statusの遷移を表してください | P6: UML | design+code |
| Q34 | CompositeパターンでKGの階層ノード（親→子概念）を表現するPython実装例 | P7: GoF | design+code |
| Q35 | StrategyパターンでChance-Level ThresholdのHard版とSoft版を交換可能にする設計 | P7: GoF | design+code |
| Q36 | UMLコンポーネント図でMED / TRIDENT / 外部API間のインターフェースを表してください | P6: UML | design+code |
| Q37 | Event Sourcing でthought_logsの変更履歴を追跡するとどのような設計になるか | P8: Arch | design+code |
| Q38 | PythonのProtocol vs ABCでMEDSkillStoreProtocolを実装する設計上の違い | P8: Arch | design+code |
| Q39 | Facadeパターンで複雑なFAISSとKGの検索を統一APIに包むコード設計 | P7: GoF | design+code |
| Q40 | Microservices vs Monolithのトレードオフをmature jobとseed jobの分離判断で説明 | P8: Arch | design+code |

- ✅ Q1〜Q40 を `scripts/questions_bridge.txt` として保存（2026-04-27）
- 🟡 plan_programming_seed.md カテゴリD/E/FのSEED投入後に設問品質を再評価
- 🟡 stackoverflow型（Q&A形式）のSEEDを優先追加（UMAP分析でブリッジ位置と確認済み）

---

### O-4. ユーザー主観収集 SEED 設計 🟡
> 📄 `plan_programming_seed.md` カテゴリ M

**背景**: 既存 SEED（A〜L）は「事実・仕様空間」を中心に構成されており、学習者の「経験・判断空間」が欠けている。
UMAP で docs クラスタと academic クラスタの2島が分離している状況で、
「どこで詰まるか / なぜそちらを選んだか」という人間の経験軸が認知空間に不在。

#### ソース別の役割整理

| ソース | 役割 | 認知空間上の位置づけ |
|--------|------|-------------------|
| 技術ブログ「1から勉強してみた」 | 経験空間の軸形成 | docs〜academic の**中間**に学習者視点の島を作る |
| Stack Overflow | エラートレンド・トピッククラスタ（信号） | 軸にはならない。抜け漏れが多く均一性に欠ける |
| GitHub Issues (closed bugs) | エラーメッセージ→原因→解決策のトリプレット | docs クラスタ内で「失敗パターン」の密度を高める |

#### 技術ブログ収集方針

優先ソース（英語）:
- **dev.to**: `tag:python` / `tag:machinelearning` × `reactions≥50` × 「beginners」タグ付き
- **Medium / Towards Data Science**: `claps≥200` の実務移行記・比較記事
- **Hashnode**: 長期シリーズ記事（同著者が入門→応用と続けているもの）

収集形式:
- 「1から学んだ」「入門」「ハマった」「移行した」系の記事を優先
- 単発記事より**継続シリーズ**（入門→中級→応用の時系列記録）を最優先

#### SO の再定義（エラートレンド専用）

```bash
# SO エラートレンド収集クエリ案
# 「よく起きるエラー × 解決済み」 をスコープとする
# 例: FAISS エラーパターン
"FAISS IndexFlatIP returns wrong results"
"sentence-transformers encoding batch size error"
"torch.cuda.OutOfMemoryError when training"
"aiosqlite OperationalError database is locked"
"fastapi 422 Unprocessable Entity pydantic validation"
```

#### GitHub Issues 収集方針

```
対象リポジトリ:
  pytorch/pytorch / huggingface/transformers / facebookresearch/faiss / tiangolo/fastapi

フィルタ:
  - is:issue is:closed label:bug comments:>10
  - 再現コード付き優先（エラーメッセージ→原因→解決策のトリプレット構造）
```

#### maturation 切り口（`user_perspective` variant）

ユーザー主観コンテンツは通常の「正確性優先」評価とは異なる軸でレビューする:

| 通常コンテンツ | `user_perspective` コンテンツ |
|-------------|---------------------------|
| 事実として正しいか | 経験として真正か |
| 説明が完全か | 詰まりポイントが具体的か |
| 現行仕様と一致するか | 主観的判断の理由が語られているか |
| 技術的複雑さ = 難易度 | 学習者の心理的障壁 = 難易度 |

実装: `source_extra` に `"content_type": "user_perspective"` を付与し、Verifier が評価軸を切り替える

#### 実装 TODO

- 🟡 `data/doc_urls/user_blogs.txt` 作成（dev.to / Medium の「1から学んだ」系URLリスト）
- 🟡 `data/doc_urls/github_issues.txt` または `scripts/seed_github_issues.py` 設計
- 🟡 SO エラートレンド収集クエリリスト作成（`scripts/questions_so_errors.txt`）
- 🟡 Verifier `user_perspective` variant プロンプト設計

---

### O-5. 教師-生徒設計哲学 — 認知類型モデルと MED への含意 🔵（UMAP島充実後）

**前提条件**: O-1〜O-4 の SEED 投入により UMAP の各島が充実した後に実施する

#### 理解の4類型

| 類型 | 特徴 | LLM対応 | 教師としての特性 |
|-----|------|---------|--------------|
| **1. 天才型**（感性優位） | 最初の直感が答えに直結、過程をほぼ省略。過程を説明できないことが多い | 良いパラメーターを偶然引いた状態（再現不可） | 教えられない。見せて真似させるのみ |
| **2. 高能力型**（大コンテキスト） | 必要な全情報を最後まで保持しながら積み上げ、または積み上げを効率化 | 超大コンテキスト・全情報を適切に要約して最高効率で出力 | 同スペック同士には教えられる。生徒がどこで躓いたか分からないため普遍的な教師にはなれない |
| **3. WM制約型**（ワーキングメモリ不足） | 必要な情報がこぼれやすく積み上げが非効率になる | 小コンテキスト・要約が粗くなる | どんな工夫が必要かを知っているため優秀な教師になれる可能性がある。**ただしこのタイプで教師モデルとしての品質を達成できた場合に限る** |
| **4. 前提修正型**（認識の歪み経験） | 常識を疑うことを知っている。認識の違いを認識できる | 事前知識の誤りを訂正した経験がある | 相手に合わせる最適化が得意。例: 日本語話者は省略するスタイル、フランス語話者は全部言うスタイル — この認識差を知ることで相互理解が深まる |

#### 教師の構造と出力の分岐

```
教師の基盤: 2（大コンテキスト知識）
  × 4（認識差を認識する能力）
  × 3（制約条件下の工夫を知っている）
```

| 目標 | アプローチ | 生徒の到達点 | MED上の対応 |
|-----|----------|-----------|-----------|
| **ワーカー育成**（1型出力） | 2+4 が 3 に教え、とりあえず結果を出させる | 最低条件を満たす作業者。教師にはなれない | 組み込み・エッジモデル向け（外部メモリ制約が厳しい場合） |
| **自律型育成**（2型の振舞い） | 2+4 が 3 に教え、外部メモリで 2 のように振舞わせる | 外部メモリ込みで高能力型と同等に機能できる | **MEDが目指す状態** |

エッジモデルの場合は外部メモリ側にも制約が生じるため、どこまで内部知識化できるかの考察が別途必要。

#### MED への含意

**MEDが目指すもの**: 小さいモデル（3型 = WM制約）が外部メモリ（FAISS）を使って大きいモデル（2型）のように振舞う

**⚠️ 現在の設計上の注意点**:
> 「構築している論理で内部知識と外部コンテキストを混同している点がある」

| | 人間 | MED |
|--|------|-----|
| 内部知識 | 長期記憶・スキーマ（ワーキングメモリ消費を削減する） | モデルパラメーター（LoRA含む） |
| 外部コンテキスト | ワーキングメモリで処理する入力情報 | FAISS 検索結果 + RAG コンテキスト |
| 前提認識の歪み（4型） | 内部知識の誤りが補正を失敗させる | FAISS に誤情報・冗長情報 → 検索結果が前提を歪める |
| 対策 | 4型経験（常識を疑う） | seed_blacklist + maturation 品質管理 |

人間では「内部知識で代替」することでワーキングメモリの消費が大幅に減る。この節約が可能なのは内部知識が正確なときだけで、歪んでいると4型の問題として現れる。

MEDでは外部知識を扱うために認知空間（FAISS空間）が必要。前提認識を歪まないよう、**内部知識**（モデルパラメーター）はなるべく最小・精選する方針を取る — 誤った前提をパラメーターに焼き込まないことが重要。FAISS は「半外部知識」として位置づけ、更新によって対応する。

#### FAISS = 半外部知識としての知識更新サイクル

世の中では常識が覆る再発見がよく起こる。外部知識は更新で対応できる。この観点で FAISS と NEAT の更新サイクルは一般的なモデル再訓練と同等の位置づけになりうる:

| フェーズ | 内容 | 一般的なML対応 |
|---------|------|-------------|
| **FAISS 更新** | RAG で使いながら知識を集積 → maturation で精査 → Trusted として組み込み | 学習データの追加・更新 |
| **NEAT 更新** | 新しい連想トポロジを構築して知識を検索しやすくする | モデルアーキテクチャの再設計 |

**⚠️ NEAT 更新の懸念**:
NEAT の連想トポロジに知識を紐づけると、トポロジが進化するたびに検索の振舞いが変わる。
「Windows の OS 更新のたびに UI が激変するストレス」に近い問題が生じる可能性がある。
→ NEAT のトポロジ変化の影響範囲を局所化する設計が将来的に必要。

#### 教師 SEED の実装方針（UMAP島充実後）

UMAP 上で各島が充実したら、「2型が4を踏まえて3に教える」観点のコンテンツを追加 SEED する:

| SEED 種別 | 内容 | UMAP上の期待位置 |
|---------|------|--------------|
| 4型視点の解説記事 | 「この前提が間違っていた → ここが変わった」型 | academic 〜 docs の間 |
| 3型向け工夫記事 | 「ワーキングメモリを節約する書き方」「段階的理解のロードマップ」 | docs 〜 user_perspective の間 |
| 教師の言語化 | 「なぜこう教えるか」を明示した解説（チュートリアル批評・カリキュラム設計） | design 〜 academic の間 |

#### 実装 TODO（UMAP島充実後）

- 🔵 FAISS 内で「内部知識で代替可能な冗長な外部知識」を識別するUMAP分析
- 🔵 maturation reviewer に「内部知識との重複度」評価軸を追加する variant 設計
- 🔵 Teacher プロンプトに「4型視点（認識差の認識）」と「3型制約（段階的説明）」を組み込む
- 🔵 Student の「外部コンテキスト依存度 vs 内部知識活用度」を測定する評価指標設計
- 🔵 エッジモデル向け: 外部メモリなしで最低限機能する「内部知識の最小セット」の特定

---

### O-OQ: Open Questions

| ID | 問い |
|----|------|
| O-OQ-1 | UMAP後の空白地帯に対して種別Cの合成ドキュメントを生成する場合、Teacher APIのコストとSEEDの多様性のバランスをどうとるか |
| O-OQ-2 | academicとcodeの中間に意図的に置く「ブリッジ文書」の最適な粒度（1概念1文書 vs 複数概念まとめ）|
| O-OQ-3 | TRIDENTのAssociationFn重み進化において、academic↔codeブリッジ文書がcontext_embとして機能するか |
| O-OQ-4 | MEDのFAISS外部知識のうち「モデル内部知識で代替できる部分」の割合はどう計測するか（内部 vs 外部の境界定量化） |
| O-OQ-5 | エッジモデル向けに「外部メモリなしで最低限機能する内部知識セット」を特定するための蒸留実験設計 |
| O-OQ-6 | NEATトポロジ更新が検索振舞いに与える影響範囲をどう局所化するか（トポロジ変化の互換性設計） |

---

---

## P. サイクル管理システム（Knowledge Collection Cycle）✅ **完了（2026-05-01）**

Gap Detection → Enrich → Dispatch の自律サイクルパイプライン。

### P1d. Orchestrator ✅
- ✅ `src/cycle/schema.py` — `CollectionTask` / `GapType` (`small_cluster`, `unreviewed_backlog`, `source_imbalance`, `low_quality`)
- ✅ `src/cycle/gap_detector.py` — UMAP 島分析から CollectionTask リスト生成
- ✅ `src/cycle/query_generator.py` — LLM 支援検索クエリ生成
- ✅ `src/cycle/cycle_store.py` — `cycle_runs` / `cycle_tasks` SQLite 永続化
- ✅ `src/cycle/orchestrator.py` — `OrchestratorConfig` + `Orchestrator.run_cycle()`
  - `_phase_detect()` → `_phase_enrich()` → `_phase_dispatch()` の 3 フェーズステートマシン
  - `_MATURE_GAP_TYPES` (`unreviewed_backlog`, `low_quality`) → `mature_only()` 呼び出し（上限 200 件）
  - `_COLLECTOR_GAP_TYPES` (`small_cluster`, `source_imbalance`) → 収集ログのみ（QueryRunner 未実装）
- ✅ `src/cycle/__init__.py` — `CycleStore`, `Orchestrator`, `OrchestratorConfig` エクスポート追加
- ✅ `scripts/run_cycle.py` — CLI エントリーポイント（`--detect-only`, `--enrich-only`, デフォルト full run）

### P3. サイクルモニタリング GUI タブ ✅
- ✅ `src/gui/tabs/cycle.py` — 読み取り専用モニタリングタブ（259 行）
  - UMAP Islands 散布図（Plotly Express、最大 8,000 点）
  - サイクル実行履歴テーブル（直近 20 件）
  - 最新サイクル タスク一覧テーブル（reason[:80] / keywords[:4]）
  - `⟳ 更新` ボタン → 全コンポーネント一括更新
- ✅ `src/gui/app.py` — `🔄 サイクル` タブとして追加（`🎓 学習` の前）

### P4a-b. プランビューア & 実行コントロール ✅
- ✅ `src/gui/tabs/plan.py` — 245 行
  - **P4a**: run-id ドロップダウン（直近 50 件）+ サマリー Markdown + タスク詳細 DataFrame
    - keywords・queries・signals の全文表示（P3 は切り詰め、P4 は完全表示）
    - `⟳ リスト更新` でドロップダウンを最新化
  - **P4b**: Provider ドロップダウン + Model テキスト + `▶ Run Cycle` ボタン
    - DB バックのロック: 直近 30 分以内に `status='running'` な run がある場合はブロック
    - `threading.Thread(daemon=True)` + `asyncio.new_event_loop()` でバックグラウンド実行
- ✅ `src/gui/app.py` — `📋 プラン` タブとして追加（`🔄 サイクル` の直後）

### 残課題

#### P-R1. Seeder / Reviewer 分離 ✅ **完了（2026-05-02）**
- ✅ `src/cycle/orchestrator.py`: `_dispatch_mature()` から `mature_only()` 呼び出しを除去
  - UNREVIEWED_BACKLOG / LOW_QUALITY → ギャップを記録して `done` にマークするのみ
  - 実際のレビューは Reviewer タブ（ReviewerSession）がマルチスレッドで担当
  - `OrchestratorConfig.persona` / `mature_interval` は後方互換のため残存（未使用）
  - `_MATURE_LIMIT_CAP` 定数を削除
- ✅ `src/cycle/__init__.py` / ドキュメント更新

#### P-R2. Reviewer タブ実装 ✅ **完了（2026-05-01）**
- ✅ `src/cycle/reviewer_worker.py` (279行) — ReviewTask / SlotConfig / ReviewerConfig / ReviewerSession
  - `build_task_list()`: unreviewed + needs_update を DB から取得してメモリ上タスクリスト構築
  - `_get_next_task()`: ロック + ランダムスリープ (100-1000ms) でタスクを排他取得
  - `_worker_thread()` + `_worker_async()`: スロット毎のデーモンスレッド（asyncio.new_event_loop）
  - `ReviewerSession.stop()`: 停止フラグ → join(timeout) → 生存スレッドは放棄
- ✅ `src/gui/tabs/reviewer.py` (185行) — 4スロット UI + 進捗モニター
  - スロット毎: Provider ドロップダウン / Model テキスト / ペルソナ CheckboxGroup
  - `▶ レビュー開始` / `■ 停止` ボタン
  - `gr.Timer(10秒)` による自動ポーリング（Gradio 5+ のみ、旧版は手動 `⟳`）
  - タスク一覧 DataFrame（最大 200 件）+ ETA 表示

#### P-R3. ペルソナ対 source_type マッピング改善 ✅ **完了（2026-05-02）**
`_DOMAIN_FLAG_MAP`（seed_and_mature.py:109）を更新:
- `arxiv` → `strict`（学術論文）
- `stackoverflow` → `practical_reference`（Q&A実践内容）
- `web_docs` → `practical_reference`（manページ・wiki系）
- `tavily` → `practical_reference`（一般Webスニペット）
- `github` → `on_domain`（コードファイル・現状維持）
- `github_docs` → `on_domain`（APIリファレンス・seed_from_docs.py 専任）

#### P-GUI-1. GUI 入力パラメーター localStorage 永続化 ✅ **完了（2026-05-03）**
ページリロード後も全入力値を復元する。`elem_id="med-{tab}-{component}"` をキーに使用。

- ✅ `src/gui/tabs/plan.py` — provider / model（2コンポーネント）
- ✅ `src/gui/tabs/chat.py` — provider / model / mode / memory,RAG チェック / timeout h,m,s / CRAG 設定 4 個（13コンポーネント）
- ✅ `src/gui/tabs/reviewer.py` — limit / timeout / low_q / スロット 1-4 × (provider, model, personas)（15コンポーネント）
- ✅ `src/gui/tabs/training.py` — algorithm / adapter / reward / sliders 3 / TinyLoRA 3 / reward 重み 5（14コンポーネント）
- ✅ `src/gui/app.py` — `app.load(fn=None, js=..., outputs=[...])` でページロード時に一括復元
  - Dropdown / Radio / Checkbox → `.change()` で即保存、Textbox → `.blur()` で保存
  - optional Dropdown（null 許容）は未保存時に `localStorage.removeItem` → デフォルト `null` で復元

#### P-R4. 文書側ペルソナ指定フィールド追加 🟡（P-R3 後）
現状: DB の documents テーブルにペルソナ指定カラムなし。domain_flag で代替。
TODO: `required_persona TEXT DEFAULT NULL` を追加 → Reviewer ワーカーがペルソナ対応文書のみ処理できるようにする。
DB マイグレーション: `ALTER TABLE documents ADD COLUMN required_persona TEXT DEFAULT NULL;`

#### P-R5. QueryRunner（Seeder）実装 ✅ **完了（2026-05-02）**
- ✅ `src/cycle/query_runner.py` (232行) — `QueryRunner` / `QueryRunnerConfig`
  - `initialize()`: RetrieverRouter / Embedder / MemoryManager / Deduplicator を共有初期化
  - `run_task(task)`: queries × 外部検索 → 関連性フィルタ(cosine) → ブラックリスト → dedup → `mm.add()`
  - `SOURCE_IMBALANCE`: `dominant_source` を除外したソースリストで検索（多様化）
  - `SMALL_CLUSTER`: 全利用可能ソースで検索（拡充）
  - mature なし（Reviewer タブが担当）
- ✅ `src/cycle/orchestrator.py`: `_dispatch_needs_collector()` → `_dispatch_collector()` に置換
  - `_phase_dispatch()` で QueryRunner を1インスタンス生成・再利用、finally で close
- ✅ `src/cycle/__init__.py`: `QueryRunner`, `QueryRunnerConfig` エクスポート追加

#### P-BUG-1. バグ修正（2026-05-03）✅
- ✅ **reviewer Slot 2-4 provider エラー**: 未保存時に `''` を復元 → Gradio が choices バリデーションエラー
  - 修正: 保存側を `removeItem`（null クリア）に変更、復元側デフォルトを `null` に変更
- ✅ **QueryGenerator モデル未伝播**: プランタブで指定したモデルが LLM 呼び出しに渡らずプロバイダーデフォルトモデルが使われていた
  - 修正: `QueryGenerator.__init__` に `model: Optional[str]` 追加、`_call_llm` に `model=self._model` 追加
  - 修正: `Orchestrator._phase_enrich` で `model=self._cfg.model or None` を渡すよう修正

#### P-GUI-2. シーダータブ ポーリング強化 ✅ **完了（2026-05-04）**
- ✅ **⏸ 停止 / ▶ 再開ボタン**: `_polling_paused` フラグで Timer コールバックを一時停止
  - サーバー再起動後の誤検知（古い `running` run を遷移と誤認）を手動回避できる
- ✅ **ETA 表示**: `running` 中のみ `ETA: 残り Xm Ys (done/total 完了)` を status_md に追記
  - 完了済みタスクの平均処理時間から残り時間を算出
- ✅ **タブ名変更**: 🔄 サイクル → 📊 アナリティクス / 📋 プラン → 🌱 シーダー（`src/gui/app.py`）

#### P-BUG-2. レトリーバー レート制限・並列制御強化 ✅ **完了（2026-05-05）**
- ✅ **arXiv 429 対策**:
  - レート制限 5s → **10s** に引き上げ（`src/rag/retriever.py`）
  - 429 受信時に 15s/45s 指数バックオフリトライ（最大2回）を追加（`src/rag/retrievers/arxiv.py`）
- ✅ **SO 400/429 対策**: レート制限を **12s** に設定（`src/rag/retriever.py`）
- ✅ **SO 日次上限 300件**: `DailyUsageTracker` を流用し `data/openrouter_usage.db` で使用回数を管理
  - `_RATE_LIMIT_COUNTS = {"stackoverflow": 300}` — ソース別日次上限を定義
  - `BaseRetriever.search()` でセマフォ取得前に日次チェック → 超過時は `[]` を返してスキップ
  - `_get_daily_tracker()` — モジュールレベル遅延初期化（`asyncio.Lock` 不使用、SO concurrency=1 で二重初期化防止）
- ✅ **ソース別同時リクエスト上限**: `BaseRetriever` にインスタンス単位のセマフォを追加
  - arXiv: 1 / SO: 1 / GitHub: 2 / Tavily: 2（異なるソース間の並列実行は維持）
  - `asyncio.Semaphore` をインスタンスに保持（`_get_sem()` 遅延生成）— イベントループ跨ぎ安全
- ✅ **enrich_concurrency デフォルト**: 3 → **1**（LMStudio等ローカルプロバイダーの同時リクエスト積み上がりを防止）

#### P-GUI-3. Reviewer タブ キュー件数表示 ✅ **完了（2026-05-05）**
- ✅ `_get_queue_counts()` / `_format_queue_md()` 追加（`src/gui/tabs/reviewer.py`）
  - SQLite から `review_status` 別件数を集計（unreviewed / needs_update / 合計）
- ✅ 実行設定アコーディオン内に件数 Markdown + `⟳ 件数更新` ボタンを配置
- ✅ **⟳ ボタンクリック** → DB 再クエリして最新件数を反映

#### P-BUG-3. DB ロック解除手順の確立 ✅ **完了（2026-05-05）**
サーバー中断でシーダーが `running` のまま止まる問題（hot-reload による `.claude/` 読み取り権限エラーが原因）
- ✅ `fuser data/metadata.db` でロック保持プロセス確認 → サーバー終了後は自動解除
- ✅ WAL ファイル（`-wal`/`-shm`）はサーバー終了後に残存しないことを確認
- ✅ `running` 状態の `cycle_runs` レコードを `error` に書き換えるワンライナー手順を確立
  ```python
  conn.execute("UPDATE cycle_runs SET status='error', finished_at=datetime('now'), "
               "summary='Interrupted: server shutdown / DB lock' WHERE status='running'")
  ```
- ⚠️ **根本対策**: `--reload-exclude .claude` を uvicorn 起動時に指定（`.claude/` 読み取りエラーを防止）

#### P-R6. 実験的: レビュー結果によるブリッジ文書生成 🟢（審査プロセス設計後）
Reviewer の分析結果を使って UMAP 上の島間ブリッジを伸ばす合成文書生成。
⚠️ ハルシネーション連鎖リスクが高い。実施前に以下が必要:
- 合成文書の審査プロセス設計（Verifier + trust_score チェックゲート）
- 生成→審査→FAISS 追加 の各ステップの品質テスト
- 生成元が「レビュー結果」であることの provenance 記録（N-5 SourceTrustScore と連携）

#### P-R7. Reviewer タブ ポーリング機能 ✅ **完了（2026-05-22）**

- ✅ `ReviewerConfig` に `lock_sleep_min_ms` / `lock_sleep_max_ms` / `ui_poll_interval_sec` フィールドを追加
- ✅ `ReviewerConfig.TEST_PRESET()` — limit=5, timeout=5s, sleep=0〜1ms, ui_poll=1s
- ✅ `ReviewerConfig.PROD_PRESET()` — デフォルト値（200件、60s、100〜1000ms、10s）
- ✅ `_worker_thread/_worker_async` を `cfg: ReviewerConfig` 受け取りに変更（db_path を cfg から取得）
- ✅ `_get_next_task` / `_finish_task` が `lock_sleep_min_ms/max_ms` を参照
- ✅ `reviewer.py` Timer interval を `ReviewerConfig.PROD_PRESET().ui_poll_interval_sec` から取得
- ✅ `tests/unit/test_reviewer_worker.py` — TEST_PRESET fixture 統合テスト 9件追加（全 PASS）

**備考**: `reviewer_worker.py` が 313 行に増加（300行上限超え）。→ **P-R10** で分離予定。

#### P-R10. reviewer_worker.py モジュール分離 ✅ **完了（2026-05-23）**

`python-strict.md` の 300 行制限超過（旧 313 行）に対応。

- ✅ `src/cycle/reviewer_config.py` 新規作成 — `ReviewTask` / `SlotConfig` / `ReviewerConfig` を移動（58行）
- ✅ `src/cycle/reviewer_worker.py` — worker 関数 + `ReviewerSession` + `get_persona_choices` のみに縮小（268行）
- ✅ `src/gui/tabs/reviewer.py` のインポートパスを更新（`reviewer_config` から `ReviewerConfig` / `SlotConfig`）
- ✅ `tests/unit/test_reviewer_worker.py` のインポートを更新 — 全 9 件 PASS

#### P-R8. シーダータブへの自動ポーリング ✅ **完了（2026-05-04）**
`gr.Timer(5秒)` で status_md を更新、`running→done/error` 遷移時のみ run_dd を再ロード。
停止ボタン・ETA 表示は P-GUI-2 で実装済み。

#### P-R9. シーダータブの「タスクスキップ」オーバーライド機能 🟢

#### P-SYS-1. ジャーナルシステム構築 ✅ **完了（2026-05-08）** → **Stop Hook 削除済み**
Stop Hook → LMStudio Gemma 4 31B → SQLite FTS5 によるセッション自動記録。その後 Stop Hook から削除（AWEP ClaudeJournal に移行）。

- ✅ `~/.claude/journal/scripts/journal_hook.sh` — スクリプト残存（フック未登録）
- ✅ `~/.claude/journal/scripts/conv_to_text.py` — JSONL → プレーンテキスト変換
- ✅ `~/.claude/journal/scripts/summarize.py` — LMStudio API 呼び出し（Gemma 4 31B）
- ✅ `~/.claude/journal/scripts/register_topics.py` — `journal.db` (SQLite FTS5) へ登録
- ✅ `~/.claude/journal/scripts/search_journal.py` — トピック検索 / 最近セッション取得 CLI
- ~~`.claude/settings.json` Stop Hook に `journal_hook.sh` を追加~~ → **削除済み**（P-SYS-2 に移行）

#### P-SYS-2. AWEP（ai_workspace_event_platform）ClaudeJournal プラグイン統合（2026-05-17 更新）
`~/.claude/settings.json` に3つのフックを追加。全イベントを AWEP サーバー（localhost:8001）へ転送。
APIの使い方・統合方針は `.claude/rules/awep-journal.md` / `forUser/rules/awep-journal.md` を参照。

- ✅ **フック追加**（`.claude/settings.json`）:
  - `awep-pre`：PreToolUse → `POST http://localhost:8001/ingest`（ツール呼び出し前）
  - `awep-post`：PostToolUse → `POST http://localhost:8001/ingest`（ツール呼び出し後）
  - `awep-stop`：Stop → `POST http://localhost:8001/ingest`（セッション終了時）
  - いずれも `|| true` で AWEP サーバー未起動時のエラーを無視（サイレントフォールバック）
- ✅ **AWEP 検索 API 実装（AWEP STEP 5、2026-05-16 完了）**:
  - `GET /search/conversations` — FTS5 全文検索（trigram、最低3文字）
  - `GET /search/topics` — トピックキーワード部分一致
  - `GET /context/recent` — 最近N会話サマリー注入
  - `scripts/context_hook.py` / `scripts/topic_hook.py` — UserPromptSubmit オプショナルフック
- ✅ **API ルール文書化（2026-05-17）**: `.claude/rules/awep-journal.md` に API 仕様・クライアントパターン・統合ロードマップを記録
- 🟡 **UserPromptSubmit フック有効化（オプション）**:
  ```bash
  cd /mnt/d/Projects/claude_work/ai_workspace_evnet_platform
  ./scripts/install-hooks.sh --with-context   # 最近の会話サマリー注入
  # または
  ./scripts/install-hooks.sh --with-topics    # FTS5マッチした関連会話注入
  ```
- 🟡 **AWEP サーバー起動手順の文書化**: `docs/awep_setup.md` にまとめる（未作成）
- 🟢 **5-3: セマンティック検索** `GET /search/semantic`（AWEP 側未実装）→ 完了後に `topic_hook.py` を FAISS 版に切り替え（awep-journal.md §4-1）
- 🟢 **5-5: 双方向連携パイプライン設計**: AWEP サマリー → MED エピソード FAISS 投入 / MED 検索結果 → AWEP KG（awep-journal.md §4-2/4-3 / **Q-3a** 参照）
- ✅ **hookフォーマット修正（2026-05-21）**: curl が raw Claude CLI ペイロードをそのまま送信していたため 422 エラーが発生していた問題を修正
  - 修正前: stdin を直接 `/ingest` に転送（`source` / `payload` フィールドなし → Pydantic 422）
  - 修正後: `python3 -c "import json,sys; ..."` でラップし `{"source":"claude_cli","payload":{...}}` 形式に変換してから送信
  - 対象フック: `awep-pre`（PreToolUse）/ `awep-post`（PostToolUse）/ `awep-stop`（Stop）の3件
- **注意**: P-SYS-1（journal_hook.sh）は Stop Hook から削除済み。スクリプトファイル（`~/.claude/journal/scripts/`）は残存しているが、現在は何も呼び出していない。Stop Hook は pytest runner と awep-stop のみ。

#### P-BUG-4. QueryRunner バグ修正 ✅ **完了（2026-05-20）**

- ✅ **0件取得時のレートリミット消滅修正**: `_run_query()` でキャッシュ済みソースを除外した後、
  全ソースがキャッシュ済みでも `record_query()` が呼ばれず TTL が更新されなかった問題を修正
- ✅ **QueryRunner クエリキャッシュ追加**: `QueryRunnerConfig.cache_ttl_days`（デフォルト 7日）
  - `_run_query()` の検索前に `is_query_cached(query, source)` でソース別にスキップ判定
  - 検索後に `record_query(query, source, result_count)` でキャッシュ記録
  - `seed_only.py` 側とテーブル共有（`seed_query_log`）

---

#### P-QE. クエリ生成環境拡張 ✅ **完了（2026-05-21）**

> 参照: `plan_programming_seed.md` / 会話（2026-05-20〜21）
>
> `GapDetector → QueryGenerator` パイプラインに UMAP 空間的構造（島間距離・理論↔実装の偏り）の
> 活用と 0 件クエリのフィードバックループを追加。5 タスクすべて実装・テスト完了。
>
> **iptestserver WebGUI 経路（QueryRunner）スタブテスト結果（2026-05-21）**:
> - Test 1 (SMALL_CLUSTER 全ソース)              : PASS
> - Test 2 (SOURCE_IMBALANCE dominant 除外)       : PASS
> - Test 3 (QueryRunner キャッシュ TTL=7)         : PASS
> - Test 4 (INTER_ISLAND_BRIDGE 全ソース) [P-QE-1]: PASS
> - Test 5 (0件ピボット発火)              [P-QE-4/5]: PASS（empty モード使用）
> - `scripts/test_queryrunner_stub.py` で実施（Orchestrator → QueryRunner → RetrieverRouter 経路）
> - 前提修正: `.env` の `EMBEDDING_PROVIDER_URL` が未読み込みだったため LMStudio 使用に修正してから実施

##### P-QE-1. 島間距離分析 → 橋渡しクエリ生成 ✅ **完了（2026-05-21）**

**背景**: 離れた島ペアを検出し、中間領域を埋める論文・実装を探すクエリを生成したい

- ✅ `src/cycle/umap_islands.py` に `detect_isolated_pairs(iset, min_dist_percentile=75)` 追加
  - 全島ペアの重心間ユークリッド距離を計算し、距離上位ペアを `(island_a, island_b, dist)` で返す
- ✅ `src/cycle/schema.py` に `GapType.INTER_ISLAND_BRIDGE = "inter_island_bridge"` 追加
- ✅ `src/cycle/gap_detector.py` に `_detect_inter_island_bridges()` 追加
  - `signals` に `island_a/b`（ネスト）, `bridge_dist`, `source_dist`, `theory_pct`, `impl_pct` を含める
- ✅ `src/cycle/query_prompts.py`（新規）の `_build_prompt()` に `INTER_ISLAND_BRIDGE` プロンプトを追加
  - 「2トピックを橋渡しする論文・実装を探す」という明示指示
- ✅ `src/cycle/orchestrator.py` の `_COLLECTOR_GAP_TYPES` に `INTER_ISLAND_BRIDGE` 追加

##### P-QE-2. 理論↔実装橋渡しクエリ強化 ✅ **完了（2026-05-21）**

**背景**: arxiv 偏りの島に対し GitHub・SO の実装・運用コンテンツを積極収集したい

- ✅ `src/cycle/gap_detector.py` の `_island_signals()` に以下を追加:
  - `theory_pct`: arxiv 占有率
  - `impl_pct`: github + stackoverflow 占有率
- ✅ `src/cycle/query_prompts.py` の `_build_prompt()` で `signals` の `theory_pct` を参照:
  - `theory_pct > 0.70` かつ `impl_pct < 0.10` → `"Find GitHub repositories / implementation guides / operational notes"` ヒント追加

##### P-QE-3. SMALL_CLUSTER プロンプト強化（関連研究・派生研究） ✅ **完了（2026-05-21）**

**背景**: 小さい島のクエリを「多様なソース」から「後続・派生・関連研究」方向にシフトしたい

- ✅ `src/cycle/query_prompts.py` の `SMALL_CLUSTER` ヒント文を更新:
  - 変更後: `"This topic is under-represented. Find follow-up work, derivative research, or implementations related to: [sample titles]"`

##### P-QE-4. QueryGenerator ピボット機能（0件フィードバック） ✅ **完了（2026-05-21）**

**背景**: 0件だったクエリを検知し、視点・角度を変えた代替クエリを自動生成したい

- ✅ `src/cycle/query_generator.py` に `enrich_pivot(task, zero_result_queries)` 追加
  - `zero_result_queries`: 0件だったクエリ文字列のリスト
  - `src/cycle/query_prompts.py` の `_build_pivot_prompt()` を使用（モジュール分割）
  - 返り値は通常の `enrich` と同じ `CollectionTask`（`queries` フィールドを上書き）

##### P-QE-5. QueryRunner への 0件リトライ制御 ✅ **完了（2026-05-21）**

**背景**: P-QE-4 で生成したピボットクエリを実際に追加実行したい

- ✅ `src/cycle/query_runner.py` の `run_task()` に 0件検出ループを追加:
  - `_run_query()` の返り値を `None`（全キャッシュ済み）/ `0`（0件）/ `N`（N件）に変更
  - `QueryRunnerConfig` に `pivot_threshold: float = 0.5`, `pivot_enabled: bool = True` 追加
  - 0件クエリ数 / 全クエリ数 ≥ `pivot_threshold` で `_run_pivot()` を呼び出し
  - `_run_pivot()` 内で `QueryGenerator.enrich_pivot()` を呼び、ピボットクエリを追加実行

---

#### P-SYS-3. 型不一致チェッカー実装 ✅ **完了（2026-05-18）**
AWEP `src/analysis/type_check/` を MED に移植。静的（AST+CGA）+ 動的（typeguard）の2層型検出。

- ✅ `src/analysis/type_check/` サブパッケージ作成（6モジュール）
  - `ast_extractor.py` — FunctionInfo / CallSiteInfo 抽出
  - `call_graph.py` — CallEdge / build_call_graph / build_reverse_call_graph
  - `type_mismatch.py` — TypeMismatch / detect_mismatches（プリミティブのみ・false positive 防止）
  - `checker.py` — check_directory / CheckResult（CLI: `python -m src.analysis.type_check.checker src/`）
  - `dynamic_capture.py` — pytest プラグイン（TypeCheckError を JSONL にキャプチャ）
  - `dynamic_checker.py` — オフライン解析器（JSONL → JSON+Markdown レポート）
- ✅ `tests/unit/test_type_checker.py` — 35テスト（AST抽出/呼び出しグラフ/不一致検出/統合）
- ✅ `tests/conftest.py` — `pytest_plugins = ["src.analysis.type_check.dynamic_capture"]` 追加
- ✅ `pyproject.toml` — `typeguard>=4.0.0` dev extras に追加
- ✅ `.gitignore` — `runtime/` 追加（動的チェックレポート出力先）
- ✅ `.claude/rules/type-mismatch.md` / `forUser/rules/type-mismatch.md` 作成
- **ガードテスト**: `test_no_type_mismatches_in_src_analysis`（`src/analysis/` スコープ限定）
  - `src/` 全体は false positive 153件（`execute`/`max`/`min` の名前衝突）→ 将来 CGA 改善後に拡張予定
- **動的チェッカー実行**: `poetry run pytest tests/unit/ --typeguard-packages=src -q`（オンデマンド）

---

## Q. エピソード記憶ゾーニング（Episodic Memory Zoning）🟡

> **設計方針**: 人間の記憶モデルを参考に、知識記憶（semantic memory）とエピソード記憶（episodic memory）を
> 別 FAISS インデックスで管理する。エピソードは時系列減衰スコアでランキングし、知識検索と混合しない。

### 現行ゾーン構成（知識記憶のみ）

```
data/faiss_indices/
  code/       ← コード・技術文書          （知識記憶ゾーン）
  academic/   ← arXiv 論文               （知識記憶ゾーン）
  general/    ← 汎用                     （知識記憶ゾーン）
```

### 追加するゾーン

```
data/faiss_indices/
  episodic/   ← 会話・作業・思考ログ       （エピソード記憶ゾーン）← 新設
```

**エピソードゾーンに入るデータ源（3種）:**

| 源 | 接続タスク | SourceType |
|----|-----------|-----------|
| AWEP 会話サマリー | Q-3a | `AWEP`（新設） |
| thought_logs（N-1）| Q-3b | `TEACHER`（既存）+ zone フラグ |
| 会話ターン（A-1）| Q-3c | `MANUAL`（既存）+ zone フラグ |

---

### Q-1. Schema 拡張 ✅ **完了（2026-05-23）**

- ✅ `src/memory/schema.py` — `Domain.EPISODIC = "episodic"` 追加
- ✅ `src/memory/schema.py` — `SourceType.AWEP = "awep"` 追加
- ✅ `src/memory/schema.py` — `Document` に `memory_zone: Literal["knowledge", "episodic"] = "knowledge"` フィールド追加
  - `SourceType.AWEP` → `model_validator(mode="after")` で `"episodic"` に自動設定
  - `src/memory/metadata_store.py` — `memory_zone` 列マイグレーション追加（`_MIGRATION_ADD_MEMORY_ZONE`）

---

### Q-2. FAISS エピソードインデックス追加 ✅ **完了（2026-05-23）**

- ✅ `configs/faiss_config.yaml` に `episodic:` セクション追加:
  ```yaml
  episodic:
    dim: 384
    initial_type: "Flat"
    metric: "inner_product"
    nprobe: 32
    scale_rules:
      - threshold: 100000
        migrate_to: "HNSW32"
  ```
- ✅ `configs/default.yaml` に `rag.episodic_enabled/k/decay_halflife_days/min_score` 4パラメーター追加
- ✅ `src/common/config.py` の `RAGConfig` に同4フィールド追加
- ✅ `FAISSIndexManager` の `get_domain()` がフォールバック生成するため自動対応を確認

---

### Q-3. エピソードデータ投入パイプライン

#### Q-3a. AWEP 会話サマリー → episodic FAISS ✅ **完了（2026-05-23）**
- ✅ `scripts/seed_from_awep.py` 新規作成
  - `/context/recent?n=20` + `--search-query` で会話サマリー収集（`/sessions` はタイムアウト回避）
  - `Document(memory_zone="episodic", source_type=SourceType.AWEP, created_at=...)` として MED FAISS + metadata DB へ投入
  - カーソル管理: `data/awep_cursor.db` に取込済み conversation_id を保存（差分取込）
  - `--dry-run` / `--reset-cursor` / `--limit N` / `--search-query QUERY` オプション対応
  - 初回実行: 20 件 → episodic FAISS に 20 vectors 投入済み

#### Q-3b. thought_logs → episodic FAISS（N-1 連携）✅ **完了（2026-05-23）**
- ✅ `MemoryManager.save_thought_log(log: ThoughtLog) -> str` 追加
  - `store.save_thought_log()` (DB) + `Document(memory_zone="episodic", SourceType.THOUGHT_LOG)` で FAISS 投入
  - content = `[Input]\n{input}\n\n[Output]\n{output}`（reasoning は省略）
  - `SourceType.THOUGHT_LOG` を `schema.py` に追加、`_auto_set_episodic_zone` に組み込み
- ✅ `tests/unit/test_episodic_hooks.py` — TestSaveThoughtLog 2件 PASS

#### Q-3c. 会話ターン → episodic FAISS（A-1 連携）✅ **完了（2026-05-23）**
- ✅ `MemoryManager.save_turn_to_episodic(turn: Turn) -> str` 追加
  - 20文字未満はスキップ（空文字返却）
  - `source.extra` に `{session_id, role}` 保存、`SourceType.CONVERSATION` を追加
  - `_auto_set_episodic_zone` に `THOUGHT_LOG` / `CONVERSATION` 追加
- ✅ `MEDPipeline.query()` Step N に `episodic_enabled` 時の非同期保存フック追加
  - `asyncio.ensure_future(mm.save_turn_to_episodic(user_turn / assistant_turn))`（両ロール保存）
- ✅ `tests/unit/test_episodic_hooks.py` — TestSaveTurnToEpisodic 4件 PASS

---

### Q-4. Recency-Weighted Episodic Retrieval ✅ **完了（2026-05-23）**

**設計:**
- `FAISSIndexManager.search_episodic(query, k, decay_halflife_days)` を新設
- スコアリング式（指数減衰）:
  ```python
  # decay_halflife_days は configs/default.yaml で調整
  age_days = (now - doc.created_at).days
  recency_weight = 2 ** (-age_days / decay_halflife_days)
  final_score = cosine_sim * recency_weight
  ```
- `k` は知識ゾーン検索と独立して設定可能（`rag.episodic_k` を外出し）

**チューニング対象パラメーター（configs/default.yaml）:**

| パラメーター | 初期値 | 意味 |
|-------------|--------|------|
| `rag.episodic_k` | 3 | エピソード取得件数 |
| `rag.episodic_decay_halflife_days` | 30 | 半減期（30日で重みが0.5倍） |
| `rag.episodic_min_score` | 0.0 | 足切りスコア（0.0=無効） |
| `rag.episodic_enabled` | false | エピソードゾーン検索の有効フラグ |

**検索フロー（知識とエピソードの分離）:**
```
クエリ
  ├─ search_knowledge(domains=[code, academic, general], k=5)  ← 常時
  └─ search_episodic(domain=episodic, k=3, decay=30d)          ← episodic_enabled=true のみ
         ↓ RRF または concat でマージ
     最終ランキング（知識 + エピソード混在なし → 別セクションで提示）
```

- ✅ `src/memory/memory_manager.py` に `search_episodic()` 追加（指数減衰スコアリング実装済み）
- ✅ `MEDPipeline.query()` Step 1b に episodic 検索を統合（`RetrieverRouter` ではなく Pipeline 層で実装）
  - `get_settings().rag.episodic_enabled` が true のみ動作（デフォルト false）
  - `episodic_results` を `faiss_results` 末尾に結合して LLM コンテキストに渡す
- 🟢 GUI の Chat タブに「エピソード参照」トグル追加（`episodic_enabled` を動的切替）

---

### Q-5. 将来: エピソード→知識への固定化（Consolidation）🟢

人間の記憶固定化（hippocampus → neocortex）に相当するプロセス。

- 🟢 条件: エピソードが閾値以上の頻度で参照された場合 → knowledge ゾーンに昇格
  - `episodic_access_count >= consolidation_threshold`（例: 5回参照）かつ `reward_avg > 0.8`
- 🟢 `scripts/consolidate_episodic.py` — 対象エピソードを knowledge ゾーン（domain=general）に再投入、episodic から削除

---

## R. MED モジュール分離・責務分離 🟡

> **背景**: MED は現在、Memory / RAG / LLM / Cycle / GUI / KG / Training など多数の責務を単一リポジトリで担っている。
> TVKB 構想への発展と AWEP との疎結合連携を見据え、各責務の境界を明確にして段階的に分離可能な構造へ移行する。
> **進め方**: まず R-0 で既存の結合状態を調査・記録し、その結果を踏まえて R-1 で分離方針を決定する。

---

### R-0. 既存実装の独立性確認（依存関係調査）✅ **完了（2026-05-22）**

**成果物**: `docs/module_dependency_report.md` / `runtime/dep_graph.json` / `scripts/dep_graph.py`

**主要な調査結果**:
- 真の構造的循環依存: **0件**（遅延 import による擬似循環 13件はランタイム問題なし）
- 層跨ぎ依存: **1件**（`src.orchestrator → src.sandbox`、sandbox をL3に再分類で解消可）
- `src.llm.gateway` fan-in=33 / `src.memory.schema` fan-in=22 が集中ポイント
- `src.memory ↔ src.llm` パッケージ循環: `llm.response_generator` が `memory.schema` 型を使用。`src.common.models` 移動で解消可能
- **密結合**: `src.orchestrator.pipeline`（fan-out=14）、`src.memory.maturation`（llm + memory 両依存）
- **要整理**: `src.memory.schema`（600行超）、`src.cycle.reviewer_worker`（P-R10で対応予定）

**推奨アクション（R-1 以降）**: P-R10 → schema分割 → orchestrator3層化 → maturation Port化の順

**目的**: どのモジュールがどのモジュールに依存しているかを可視化し、密結合・循環依存・SRP 違反箇所を特定する。

#### R-0-1. import グラフ可視化

```bash
# pydeps でグラフ出力（未導入の場合は poetry add --group dev pydeps）
poetry run pydeps src --max-bacon=3 --cluster --noshow -o runtime/dep_graph.svg

# 循環依存チェック（stdlib のみ使用）
poetry run python -c "
import ast, pathlib, sys
src = pathlib.Path('src')
imports = {}
for f in src.rglob('*.py'):
    mod = str(f.with_suffix('')).replace('/', '.')
    tree = ast.parse(f.read_text())
    imports[mod] = [
        n.names[0].name if isinstance(n, ast.Import) else n.module
        for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
        if getattr(n, 'module', None) and 'src.' in (getattr(n, 'module', '') or '')
    ]
for mod, deps in imports.items():
    for dep in deps:
        if mod in imports.get(dep, []):
            print(f'循環依存: {mod} ↔ {dep}')
"
```

#### R-0-2. 責務別モジュール評価表

調査対象と評価観点：

| モジュール | 推定責務 | 主な調査観点 |
|-----------|---------|------------|
| `src/memory/` | FAISS + SQLite 永続化 | RAG / maturation に直接依存していないか |
| `src/rag/` | 外部検索（arXiv / SO / GitHub / Tavily） | memory / llm を import していないか |
| `src/llm/` | LLM ゲートウェイ | 上位レイヤーから独立しているか |
| `src/memory/maturation/` | Teacher レビューパイプライン | llm + memory 両方に依存（密結合候補） |
| `src/cycle/` | 自律サイクル制御 | 依存先の広さ（Orchestrator が何を import するか） |
| `src/knowledge_graph/` | KG（NetworkX + Neo4j） | memory との結合度 |
| `src/training/` | GRPO + TinyLoRA 骨格 | 他から独立しているか（現状未接続想定） |
| `src/auth/` | JWT 認証 | 他モジュールへの依存なしか |
| `src/conversation/` | セッション・ターン管理 | memory / auth との結合度 |
| `src/gui/` | Gradio 9 タブ | 全モジュールを直接 import していないか（Facade 不在の確認） |
| `src/analysis/` | 型不一致チェッカー | 独立（外部 src 依存なし・確認済み想定） |

#### R-0-3. 評価基準

| 評価項目 | 良い状態 | 問題のある状態 |
|---------|---------|-------------|
| 循環依存 | なし | A→B→A のサイクル |
| 層跨ぎ依存 | 上位→下位のみ | 下位モジュールが上位を import |
| SRP 遵守 | 1モジュール1責務 | 複数の「主語」が存在する |
| インターフェース境界 | ABC / Protocol で定義 | 具象クラスを直接 import |
| テスト独立性 | モックなしで単体テスト可 | 複数モジュールのモックが必要 |

#### 成果物

- 🟡 調査実施 → 結果を `docs/module_dependency_report.md` に出力
  - 循環依存リスト
  - 密結合・SRP 違反箇所（優先度付き）
  - 「切り離しやすい順」ランキング
- 🟡 R-1（分離方針決定）への入力として整理

---

### R-1. 責務マップと分離方針決定 🟢（R-0 後）

R-0 の調査結果を踏まえ、分離の優先順位と手法を決定する。

- 🟢 「切り離しやすいモジュール」から順に独立化計画を作成
- 🟢 インターフェース（Protocol / ABC）導入が必要な境界を特定
- 🟢 TVKB の Journal / Knowledge Base / Retriever 層への対応付け確認
- 🟢 分離後の `src/` ディレクトリ構造案を `docs/module_separation_plan.md` に記述

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
| GUI: Gradio 9タブ（chat / memory / sandbox / cycle / plan / reviewer / training / guide / settings） | ✅ |
| GUI localStorage 永続化: 全入力タブ 44コンポーネント（plan/chat/reviewer/training）| ✅ |
| CI: GitHub Actions + ruff + pytest 1096テスト | ✅ |
| A-1: src/auth/ + src/conversation/ + JWT + セッション管理 | ✅ |
| A-2: ReasoningTrace / ThinkingExtractor / Extended Thinking | ✅ |
| J: restic + NAS バックアップ基盤 | ✅ |
| K: CRAG Query Rewriter 4戦略 + タイムアウト伝播 | ✅ |
| P: サイクル管理（Orchestrator + cycle/plan GUI タブ） | ✅ |
| P-SYS-1: ジャーナルシステム（Stop Hook → Gemma 4 31B → SQLite FTS5） | ✅ |
| P-SYS-3: 型不一致チェッカー（src/analysis/type_check/ + 35テスト + type-mismatch.md ルール） | ✅ |
