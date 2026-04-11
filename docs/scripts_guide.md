# scripts/ ガイド

最終更新: 2026-04-11

`scripts/` 以下のスクリプト一覧、用途、現状（現役 / 将来 / 非推奨）をまとめる。

---

## 凡例

| 記号 | 意味 |
|------|------|
| ✅ | 現役・推奨 |
| 🔵 | 将来フェーズ用（骨格のみ・現時点では未使用） |
| 🟡 | 補助用途（必要に応じて使用） |
| ⚠️ | 非推奨（上位スクリプトに統合済み） |
| 🗑️ | 一時スクリプト（実行済み・削除候補） |

---

## メインワークフロー（日次運用）

### `scripts/sh/run_job.sh` ✅ **推奨エントリーポイント**

```bash
# seed+mature（新規取得→審査）
bash scripts/sh/run_job.sh openrouter nvidia/nemotron-3-nano-30b-a3b:free 20 --seed-mature --limit 150

# mature のみ（既存 unreviewed を処理）
bash scripts/sh/run_job.sh openrouter nvidia/nemotron-3-nano-30b-a3b:free 20 --mature-only --limit 400

# FastFlowLM で mature のみ
bash scripts/sh/run_job.sh fastflowlm qwen3.5:9b 15 --mature-only --limit 200
```

**機能:**
- バックグラウンドで `seed_and_mature.py` を起動
- 指定間隔（分）ごとに進捗を自動表示
  - `[処理件数/全体]` / PASS・HOLD・FAIL カウント・承認率
  - OpenRouter 当日使用量
  - 最新ログ末尾
- ログは `/tmp/med_job_<provider>_<model>_<timestamp>.log` に出力

---

### `scripts/seed_and_mature.py` ✅ **メイン Python スクリプト**

`run_job.sh` から呼ばれる実体。直接実行も可。

```bash
# seed + mature（フル実行）
poetry run python scripts/seed_and_mature.py \
    --questions scripts/questions.txt \
    --exclude-sources tavily \
    --top-k 5 --limit 150 \
    --provider openrouter \
    --model nvidia/nemotron-3-nano-30b-a3b:free

# mature のみ
poetry run python scripts/seed_and_mature.py \
    --mature-only --provider openrouter --limit 400

# seed のみ（審査スキップ）
poetry run python scripts/seed_and_mature.py \
    --questions scripts/questions.txt --no-mature
```

**UTC リセット自動停止（OpenRouter のみ）:**
- UTC 23:50〜23:55: 待機ループに入り 00:00 を待つ
- UTC 23:55 超: mature ループを break して安全停止

**主な引数:**

| 引数 | 説明 |
|------|------|
| `--questions` | 質問ファイル（省略時は内蔵 30 問） |
| `--mature-only` | 既存 unreviewed のみ審査（seed スキップ） |
| `--no-mature` | seed のみ（審査スキップ） |
| `--provider` | Teacher プロバイダー（openrouter / fastflowlm / lmstudio / anthropic） |
| `--model` | モデル override |
| `--limit` | mature-only 時の最大処理件数 |
| `--top-k` | クエリごとの RAG 取得件数 |
| `--exclude-sources` | 除外ソース（例: `tavily`） |
| `--dry-run` | 質問一覧を確認するだけ（実行なし） |

---

### `scripts/seed_from_docs.py` ✅

公式ドキュメント（GitHub リポジトリ / URL リスト）を FAISS に投入する。
arXiv・SO ではカバーできない大量の実用ドキュメント用。

```bash
# GitHub ドキュメントリポジトリ（github_doc_repos.yaml）
poetry run python scripts/seed_from_docs.py --source github_docs --max-files 100 --mature --provider openrouter

# URL リスト（data/doc_urls/*.txt）
poetry run python scripts/seed_from_docs.py --source url_list --mature --provider openrouter
```

**参照設定ファイル:** `data/doc_urls/github_doc_repos.yaml` / `archwiki.txt` / `python_docs.txt` / `linux_command_line.txt`

---

### `scripts/remature_needs_update.py` ✅

`review_status = needs_update` のドキュメントを再審査する。

```bash
poetry run python scripts/remature_needs_update.py \
    --provider openrouter \
    --model nvidia/nemotron-3-nano-30b-a3b:free \
    --limit 200
```

**現状:** needs_update 498 件（arXiv 中心）

---

## 状態確認

### `scripts/sh/check_progress.sh` ✅ **推奨**

```bash
bash scripts/sh/check_progress.sh
```

表示内容: DB 件数（ステータス別）/ FAISS vectors / OpenRouter 当日使用量

---

### `scripts/check_usage.py` 🟡

OpenRouter 使用量の詳細（ジョブ別内訳あり）。`check_progress.sh` より詳細が必要なとき。

```bash
poetry run python scripts/check_usage.py          # 今日のサマリー
poetry run python scripts/check_usage.py --jobs 20 # 最近20ジョブ
poetry run python scripts/check_usage.py --all     # 全日履歴
```

---

## モデルテスト

### `scripts/sh/test_models.sh` ✅

複数モデルを少数件数（デフォルト 3 件）で比較テスト。

```bash
bash scripts/sh/test_models.sh \
    nvidia/nemotron-3-nano-30b-a3b:free \
    nvidia/nemotron-3-super-120b-a12b:free \
    google/gemma-4-31b-it:free
```

表示内容: 応答時間・質量スコア・承認率・レビュー内容。

---

### `scripts/test_teacher.py` 🟡

Teacher モデルの接続確認・JSON 出力品質テスト。初回設定時や新プロバイダー追加時に使用。

```bash
poetry run python scripts/test_teacher.py --ping --provider openrouter
poetry run python scripts/test_teacher.py --provider openrouter --test json
```

---

### `scripts/openrouter_model_test.py` 🟡

OpenRouter の全無料モデルを一括ベンチマークし結果を `data/openrouter_test.db` に保存する。
新モデルが追加されたときや定期調査時に使用。結果は `docs/openrouter_models.md` に反映する。

```bash
poetry run python scripts/openrouter_model_test.py --init   # DB初期化 & モデル一覧取得
poetry run python scripts/openrouter_model_test.py --test   # 未テストモデルをテスト
poetry run python scripts/openrouter_model_test.py --show   # 結果一覧表示
```

---

## システム管理

### `scripts/launch_gui.py` ✅

Gradio Web GUI を起動する。

```bash
poetry run python scripts/launch_gui.py
poetry run python scripts/launch_gui.py --port 7861 --share
```

---

### `scripts/backup_data.sh` ✅

`data/` を restic でバックアップし NAS・B2 クラウドに同期する。

```bash
bash scripts/backup_data.sh              # フルバックアップ
bash scripts/backup_data.sh --local-only # NAS/B2 なし
bash scripts/backup_data.sh --list       # スナップショット一覧
```

---

### `scripts/claude_commit.sh` 🟡

Claude 名義で `git commit & push` するラッパー。Claude Code から直接コミットする場合は不要。

---

## 将来フェーズ用（現時点では骨格のみ）

| スクリプト | フェーズ | 説明 |
|-----------|---------|------|
| `train_student.py` | Phase 3 | GRPO+TinyLoRA で Student モデルを学習。VERL/trl 統合が必要 |
| `evaluate_student.py` | Phase 3 | 学習済み Student をベンチマーク評価 |
| `generate_query_rewrite_data.py` | Phase K | CRAG 用 Teacher データ生成（クエリ書き換えペア） |
| `train_query_rewriter.py` | Phase K | FLAN-T5 / Qwen を CRAG タスクで SFT |

---

## 補助・特殊用途

### `scripts/seed_conversation.py` 🟡

会話 JSON ファイルから知識を FAISS に投入する。手動で作成したナレッジを追加する際に使用。

### `scripts/seed_test_users.py` 🟡 開発環境専用

テストユーザーを `data/users.db` に登録する。**本番環境では実行しないこと。**

---

## 非推奨（上位スクリプトに統合済み）

| スクリプト | 非推奨理由 | 代替 |
|-----------|-----------|------|
| `seed_memory.py` ⚠️ | seed のみ。審査なし | `seed_and_mature.py --no-mature` |
| `mature_memory.py` ⚠️ | 機能が分散。インターフェースが古い | `seed_and_mature.py --mature-only` / `remature_needs_update.py` |

---

## 削除候補（一時スクリプト・実行済み）

以下は特定の一時的なデータ修正のために作成したスクリプトで、既に実行済み。

| スクリプト | 実行目的 | 実行日 |
|-----------|---------|--------|
| `_cleanup_so_needs_update.py` 🗑️ | SO の needs_update 18 件を削除 | 2026-03-31 |
| `_cleanup_tavily.py` 🗑️ | Tavily の needs_update+rejected を大量削除（628 件） | 2026-04-01 |
| `_cleanup_tavily_needs_update.py` 🗑️ | Tavily の needs_update 11 件を削除 | 2026-04-03 |

これらは `_` プレフィックスで非実行対象を明示している。削除しても問題ない。

---

## データファイル

| ファイル | 説明 |
|---------|------|
| `scripts/questions.txt` | seed 用質問集（262 問 / 25 カテゴリ）。`seed_and_mature.py` の `--questions` に渡す |

---

## 推奨ワークフロー（日次運用）

```
1. 状態確認
   bash scripts/sh/check_progress.sh

2. ジョブ起動（OpenRouter mature-only）
   bash scripts/sh/run_job.sh openrouter nvidia/nemotron-3-nano-30b-a3b:free 20 --mature-only --limit 400

3. ジョブ起動（FastFlowLM seed+mature、並行）
   bash scripts/sh/run_job.sh fastflowlm qwen3.5:9b 15 --mature-only --limit 200

4. 新モデルを試す場合
   bash scripts/sh/test_models.sh model1 model2

5. needs_update を再審査
   poetry run python scripts/remature_needs_update.py --provider openrouter --limit 200

6. OpenRouter 使用量の詳細確認（必要時）
   poetry run python scripts/check_usage.py --jobs 10
```

**UTC 自動停止（OpenRouter）:**
`seed_and_mature.py` は UTC 23:50（JST 08:50）に待機を開始し、23:55（JST 08:55）を超えると自動停止する。
JST 09:00（UTC 00:00）以降に新しいジョブを起動すること。
