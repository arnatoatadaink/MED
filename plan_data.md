# plan_data.md — データ世代管理計画（restic + NAS）

## 目的

MED の `data/` ディレクトリ（FAISS インデックス・SQLite DB・学習済みアダプタ等）を
Git 外でバージョン管理し、NAS へ安全にアーカイブする。

### 背景・動機

- `data/` は数百MB〜数GBに成長する見込み（FAISS 10,000docs 目標）
- Git / Git LFS には不向き（差分が効かないバイナリ、ローカルで完結したい）
- NAS（SMB）が利用可能 → ローカル差分管理 + NAS同期のハイブリッド構成を採用
- Git コミットとデータスナップショットの紐付けでトレーサビリティを確保

---

## アーキテクチャ

```
data/                       ← バックアップ対象
  ├── faiss_indices/        ← FAISS ベクトルインデックス
  ├── metadata.db           ← SQLite メタデータ
  ├── users.db              ← 認証DB
  ├── conversations.db      ← 会話履歴DB
  ├── adapters/             ← TinyLoRA 学習済みアダプタ (~1KB each)
  ├── training_logs/        ← 学習ログ
  └── models/               ← ローカル埋め込みモデル

        │
        ▼  restic backup (暗号化・重複排除・差分)

.restic/repo/               ← ローカル restic リポジトリ (D: ドライブ)
  ├── config
  ├── data/                 ← 暗号化チャンク
  ├── index/
  ├── keys/
  └── snapshots/

        │
        ▼  rsync --no-perms --no-owner --no-group --no-times --delete

/mnt/z/med-backup/          ← NAS (\\192.168.2.105\Public\datarepository)
```

### 設計判断

| 項目 | 選択 | 理由 |
|------|------|------|
| 差分管理ツール | restic | 暗号化・重複排除・スナップショットが標準。CLI が簡潔 |
| ローカルリポジトリ | `.restic/repo/` (プロジェクト内) | Git の `.gitignore` で除外。プロジェクトと同居で管理が容易 |
| NAS 同期方式 | rsync (SMB互換フラグ) | SMB は Unix パーミッション非対応。`--no-perms --no-owner --no-group --no-times` で回避 |
| Git 連携 | スナップショットタグに `git:<hash>` を自動付与 | どのコミット時点のデータか追跡可能 |
| パスワード管理 | `.restic/password.txt` (chmod 600) | `.gitignore` で除外。NAS側にはリポジトリごと同期されるため復元可能 |
| DVC / Git LFS | 不採用 | Git 依存が強すぎる。NAS直接アーカイブの要件に合わない |

---

## 実装ステップ

### Step 1: restic ローカルリポジトリ初期化 — ✅ 完了

```bash
mkdir -p .restic
echo "<password>" > .restic/password.txt
chmod 600 .restic/password.txt
restic init --repo .restic/repo --password-file .restic/password.txt
```

- `.gitignore` に `.restic/` 追加済み

### Step 2: バックアップ自動化スクリプト — ✅ 完了

`scripts/backup_data.sh` を作成:

```bash
bash scripts/backup_data.sh                    # バックアップ + NAS同期
bash scripts/backup_data.sh --local-only       # NAS同期なし
bash scripts/backup_data.sh --tag "milestone"  # カスタムタグ追加
bash scripts/backup_data.sh --list             # スナップショット一覧
bash scripts/backup_data.sh --restore latest   # 最新を data/ に復元
```

自動タグ:
- `git:<短縮ハッシュ>` — 現在の Git HEAD
- `docs:<件数>` — metadata.db のドキュメント数（sqlite3 必要）
- カスタムタグ（`--tag` 指定時）

### Step 3: NAS マウント・同期 — ✅ 完了

```bash
# NAS マウント（WSL2）
sudo mount -t drvfs Z: /mnt/z

# SMB互換 rsync（スクリプト内で自動実行）
rsync -av --no-perms --no-owner --no-group --no-times --delete \
    .restic/repo/ /mnt/z/med-backup/
```

**注意**: WSL2 の drvfs/SMB マウントは `fsync` 未対応。restic を NAS 上で直接実行すると
`sync: input/output error` が発生するため、ローカル→rsync の2段構成を採用。

### Step 4: Windows バッチファイル — 🟡 未実施

Windows 側から直接実行できるよう `.bat` ラッパーを用意する（任意）:

```bat
@echo off
wsl bash -c "cd /mnt/d/Projects/claude_work/MED && bash scripts/backup_data.sh %*"
```

### Step 5: 復元手順の検証 — 🟡 未実施

```bash
# NAS からローカルに restic リポジトリを復元
rsync -av --no-perms --no-owner --no-group --no-times \
    /mnt/z/med-backup/ .restic/repo/

# スナップショット確認
bash scripts/backup_data.sh --list

# 特定スナップショットを復元
bash scripts/backup_data.sh --restore <snapshot-id>
```

### Step 6: 定期バックアップ・保持ポリシー — 🟢 将来

- cron / タスクスケジューラによる定期実行
- `restic forget --keep-last 10 --keep-daily 7 --keep-weekly 4 --prune` で古いスナップショット整理
- バックアップ前の整合性チェック: `restic check`

---

## 運用ガイドライン

### いつバックアップを取るか

| タイミング | タグ例 |
|-----------|--------|
| seed_memory.py 実行後 | `seed-python`, `seed-faiss` |
| mature_memory.py 実行後 | `mature-review` |
| train_student.py 実行後 | `train-grpo-v1` |
| マイルストーン達成時 | `milestone-1000docs` |
| 破壊的変更の前 | `before-schema-migration` |

### Git コミットとの対応

スナップショットタグに `git:<hash>` が自動付与されるため、
任意のスナップショットがどのコード状態に対応するか追跡可能:

```bash
# スナップショット一覧でタグ確認
bash scripts/backup_data.sh --list

# 対応する Git コミットの内容確認
git log --oneline git:<hash>
```

### NAS が利用不可の場合

`--local-only` でローカルのみバックアップ。NAS 復旧後に手動同期:

```bash
bash scripts/backup_data.sh --local-only
# ... NAS復旧後 ...
rsync -av --no-perms --no-owner --no-group --no-times --delete \
    .restic/repo/ /mnt/z/med-backup/
```

---

## 現在のスナップショット

| ID | 日時 | タグ | サイズ |
|----|------|------|--------|
| `aab4c96b` | 2026-03-23 13:51 | `git:e08a035`, `docs:419`, `initial` | ベースライン |
| `12927731` | 2026-03-23 13:58 | `git:e08a035`, `docs:unknown`, `test-sync` | +121MB (会話Seed含む) |
