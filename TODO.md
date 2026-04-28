# TODO.md — MED フレームワーク 残作業一覧

> 最終更新: 2026-04-27
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

**現状: approved 10,111件 / FAISS code 25,389 vectors（2026-04-28）**
- ✅ approved 10,000件目標 達成済み（Apr 28確認）
- needs_update: 6,304件（大幅増。主因: github_docs NodeJS内部リンク問題 → F-5で対処中）
- mature はローカルモデル（LM Studio / FastFlowLM）が継続稼働中

### F-1. 日次 seed_and_mature ジョブ 🟡（mature はローカル継続中）
- ✅ Apr 9: approved +302件
- ✅ Apr 10: approved +175件
- ✅ **approved 10,000件 目標達成**（2026-04-28確認）
- mature 継続: LM Studio / FastFlowLM がローカルで稼働。OpenRouter は停止中（厳格さ不足 + 429問題）
- OpenRouter 再活用候補: `openai/gpt-4o-mini` 系 — 厳格さが若干不足するため「切り口の異なるドメイン」のmatureに限定使用を検討
- 🟡 **OpenRouter 再活用**: GPT-OSS-120b を異なる観点でのmaturation（F-2と連携）

### F-2. seed_from_docs.py 本番実行 🟡（mature なし seed 継続中）
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

- 🟡 GitHub ドキュメントリポジトリ（成熟 repos はローカル mature 待ち）
  ```bash
  poetry run python scripts/seed_from_docs.py --source github_docs --max-files 100
  ```
- 🟡 questions_bridge.txt での seed 継続
  ```bash
  poetry run python scripts/seed_only.py --questions-file scripts/questions_bridge.txt
  ```

### F-3. needs_update 再mature 🟡
- 現状: needs_update **326件**（arXiv中心）
  ```bash
  poetry run python scripts/remature_needs_update.py --provider openrouter --model nvidia/nemotron-3-nano-30b-a3b:free --limit 200
  ```

### F-4. seed_blacklist ✅ **完了（2026-04-09）**
- ✅ `src/memory/metadata_store.py` — `seed_blacklist` テーブル追加
- ✅ `reviewer.py` — rejected 判定時に自動登録
- ✅ `seed_and_mature.py` — fetch後・dedup前にblacklistチェック
- ✅ `seed_from_docs.py` — Phase 1.5 にblacklistフィルタ挿入
- 現状: 172件登録済み（既存rejected文書から自動投入）

### F-5. Chunker 改善 — API リファレンス形式への対応 🟡

**背景:** Node.js API ドキュメント（`nodejs/node doc/api/`）を seed したところ
大量の HOLD が発生（承認率 31%）。原因は2つ：

1. **内部リンク記法の残存** — Markdown の `[関数名][]` 形式リンクが未解決のまま残り、
   LLM が「他セクションへの参照を含む断片」と判定して `needs_supplement=true` にする
2. **チャンク単独での文脈不足** — API リファレンスは前後のセクションを前提に書かれており、
   分割後のチャンクが単独では意味が完結しない

**対応方針:**

- 🟡 **前処理クリーナーの強化** (`src/rag/chunker.py` または `github_docs_fetcher.py`)
  - `[xxx][]` / `[xxx][yyy]` 形式の未解決内部リンクを除去または展開
  - `> Stability: N - ...` のような Node.js 固有メタ行を除去
  - `**See also:**` / `**History:**` セクションのみのチャンクを除外

- 🟡 **セクション境界を考慮したチャンク分割**
  - API リファレンス形式（`### func(args)` の見出し単位）を1チャンクとして扱う
  - 見出し＋直下の説明文が最小単位。3文未満になる見出しは次のセクションと結合する

- 🟡 **min_meaningful_sentences フィルタ**
  - 実質的な文が3文未満のチャンクを fetch 段階で除外
  - 現行の `min_chunk_len=100` は文字数のみでコンテンツ密度を見ていない

**再有効化手順:**
```bash
# F-5 実装後に Node.js を再有効化
# data/doc_urls/github_doc_repos.yaml の nodejs/node を enabled: true に変更してから実行
poetry run python scripts/seed_from_docs.py --source github_docs --max-files 80 --dry-run
```

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

#### N-2. FAISS k-value Calibration（IDEA-002）🔴
> 根拠: S2（ICL is Provably Bayesian, 2510.10981）— k=3〜5 で指数収束 O(e^{-ck})
> **RLVR知見（S6）により優先度昇格**: k値とコンテキスト品質がStudentの「見かけの賢さ」を決定

- 🔴 FAISS 取得数 `k` を `configs/default.yaml` に外出し（現状ハードコード）
  ```yaml
  retrieval:
    k: 5  # 推奨範囲: 3〜5（理論値）
  ```
- 🟡 k=3/5/7/10 での検索精度比較実験スクリプト作成（MRR / Recall@k 計測）
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

| 対象 | URL / arXiv | 橋渡しする概念 | 優先度 |
|------|------------|--------------|--------|
| sentence-transformers公式ドキュメント | GitHub/README | 384次元埋め込み ↔ all-MiniLM-L6-v2 | 🔴 |
| FAISS公式ドキュメント | GitHub/README | ベクトル検索理論 ↔ faiss.IndexFlatIP | 🔴 |
| geoopt使い方ガイド | GitHub/README | Poincaré Ball理論 ↔ geoopt.PoincareBall | 🟡 |
| GRPO実装解説 | GitHub/README | GRPO報酬理論 ↔ TRL/VERLコード | 🟡 |
| NetworkX チュートリアル | docs.networkx | KGグラフ理論 ↔ nx.DiGraph操作 | 🟡 |

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
# 種別A/B: github_docs_fetcher でREADME等を取得
poetry run python scripts/seed_from_docs.py --source github_docs --mature --provider openrouter

# 種別A: URLリストに追加してから実行
# data/doc_urls/url_list.txt に sentence-transformers / FAISS docs を追記
poetry run python scripts/seed_from_docs.py --source url_list --mature --provider openrouter
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
