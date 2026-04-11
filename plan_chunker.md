# plan_chunker.md — Chunker 改善計画

> 作成: 2026-04-12
> 関連 TODO: F-5
> 背景: Node.js API ドキュメントを seed したところ承認率 31% → needs_update 459件が発生

---

## 問題分析

### 問題1: reflink 記法の残存（最大の原因）

`_clean_markdown()` は `![alt](url)` 形式の画像バッジは除去しているが、
**Markdown 参照スタイルリンク（reflink）**が未対応:

```markdown
The [net.Server.listen()][] event is emitted...   ← [xxx][] 形式（空ラベル）
See [child_process.spawn][spawn] for details...   ← [xxx][yyy] 形式（named ref）
```

リンク定義（`[spawn]: #child_processspawn`）は文書末尾に置かれるため、
チャンク化後のテキストには **未解決のリンク記法が残る**。
LLM が「他セクションへの参照を含む断片」と判定 → `needs_supplement=true` → HOLD

---

### 問題2: 見出し単位の意味的まとまりを無視した分割

現在の `Chunker.chunk_text()` は段落（`\n\n`）を基本単位として文字数で結合する。
Node.js API ドキュメントは以下の構造を持つ:

```markdown
### child_process.spawn(command[, args][, options])
<!-- Added in: v0.1.90 -->
> Stability: 2 - Stable

* `command` {string} ...
* `args` {string[]} ...

Spawns a new process using the given `command`...
```

この **「見出し＋本文」が1つのAPI定義**だが、文字数ベースでは:
- 短い定義が複数混合されて1チャンクになる（前後の文脈が混在）
- 長い定義の見出しと本文が別チャンクに分断される

---

### 問題3: Node.js 固有メタ行

以下のパターンはコンテンツ密度がなく HOLD になる原因:

```markdown
> Stability: 2 - Stable           ← 安定性評価（1行）
> Stability: 0 - Deprecated
```

```markdown
**See also:**
* [`child_process.exec()`][]      ← reflink 除去後は空リスト行になる
* [`child_process.fork()`][]
```

> ✅ **History について**: Markdown ソースでは `<!-- YAML\nadded: vX.X.X\nchanges: ... -->` 形式の
> HTML コメントとして記述されている。現行の `_clean_markdown()` の HTML コメント除去で**既に対応済み**。
> レンダリング後の「History テーブル」はソースには存在しない。

---

### 問題4: `min_chunk_len=100` は文字数のみ

上記メタ行が3行あれば100文字を超えてフィルタをすり抜ける。
実質的な説明文（コードブロック・見出し・メタ行を除いた文）が少ないチャンクを除外できない。

---

## 改善アプローチ

### 案A: クリーナー強化のみ（最小変更）

`github_docs_fetcher.py` の `_clean_markdown()` に汎用処理を追加:

1. **reflink 除去**（汎用）
   ```python
   # [xxx][] → xxx
   text = re.sub(r'\[([^\]]+)\]\[\]', r'\1', text)
   # [xxx][yyy] → xxx
   text = re.sub(r'\[([^\]]+)\]\[[^\]]+\]', r'\1', text)
   # リンク定義行を除去: [label]: url "title"
   text = re.sub(r'^\[[^\]]+\]:\s+\S+.*$', '', text, flags=re.MULTILINE)
   ```

2. **Node.js 固有メタ行除去**（要 label 判定 → 論点2で確定後に実装）
   ```python
   # Stability 注記（> Stability: 2 - Stable 形式）
   text = re.sub(r'^>\s*Stability:\s*\d+\s*-\s*.+$', '', text, flags=re.MULTILINE)
   # See also セクション（reflink 除去後に残る空リスト）
   text = re.sub(r'^\*\*See also:\*\*\n(\*\s+.+\n?)+', '', text, flags=re.MULTILINE)
   # ※ History は <!-- YAML ... --> HTML コメントとして記述されており、
   #   _clean_markdown() の HTML コメント除去で既に対応済み
   ```

**メリット**: 変更範囲が小さい
**デメリット**: 見出し単位の分断（問題2）は解決しない

---

### 案B: Heading-aware Chunker（構造改善）

Markdown の見出し（`##`, `###`）を**チャンク強制境界**として扱う新ロジック:

```
入力:
  ### spawn(command)       ← 見出し = 新チャンク開始
  Added in: v0.1.90
  Spawns a new process...  ← 本文

  ### spawnSync(command)   ← 見出し = 次のチャンク開始
  ...

出力:
  チャンク1: "spawn(command)\nAdded in: v0.1.90\nSpawns a new process..."
  チャンク2: "spawnSync(command)\n..."
```

**隣接する短いセクションの結合**: 本文が3文未満の場合は次のセクションと結合する。

実装方針:
- `chunk_text()` を修正するか、`chunk_markdown()` を新メソッドとして追加するか → **論点3**
- `chunk_result()` で `source_type == GITHUB_DOCS` の場合に `chunk_markdown()` を呼び出す

**メリット**: API リファレンス全般（Node.js / cpython / MDN）に有効
**デメリット**: `chunk_text()` のロジックが増える

---

### 案C: A + B の組み合わせ（推奨）

| ステップ | 内容 | 適用範囲 | 状態 |
|---------|------|---------|------|
| C-1 | reflink 除去 (`_clean_markdown()` 修正) | 全 Markdown ソース | ⬜ 検討中（論点1参照） |
| C-2a | Stability 除去 + `source_extra` 保存 | `cleaner_profile: "nodejs_api"` | ✅ 実装済み |
| C-2b | Added in 抽出 → `source_extra.added_in` 保存 | `cleaner_profile: "nodejs_api"` | ✅ 実装済み |
| C-3 | Heading-aware Chunker (`chunk_markdown()` 追加) | `source == "github_docs"` | ✅ 実装済み |
| C-4 | min_meaningful_sentences フィルタ | 全ソース（閾値: 3文） | ✅ `chunk_markdown()` に組み込み済み |

---

## 未解決の論点

### 論点1: Heading-aware Chunker を汎用化するか `GITHUB_DOCS` 専用にするか
- arXiv の論文や SO の回答にも `###` 見出しは出現する
- 汎用化すると既存の Tavily / arXiv / SO チャンク品質への影響を確認が必要
- **暫定方針**: `source_type in {GITHUB_DOCS, WEB_DOCS}` の場合のみ適用し、効果確認後に汎用化を検討

### 論点2: Node.js 固有メタ行の書式確認 ← **要確認**
以下の URL で実際の Markdown 書式を確認し、正規表現パターンを確定する:
- `https://github.com/nodejs/node/blob/main/doc/api/child_process.md`
- `https://github.com/nodejs/node/blob/main/doc/api/crypto.md`

確認項目:
- `Stability:` 行の正確な書式（`> Stability:` vs `<!-- Stability: -->`）
- `History:` テーブルの前後の構造
- `See also:` セクションの終端パターン
- Added in / Changed in の記述方式

### 論点3: `chunk_text()` を修正するか `chunk_markdown()` を新メソッドとして追加するか
- 既存の Tavily / arXiv / SO / GitHub Code への影響を避けるなら**新メソッドが安全**
- 汎用化を見据えるなら `chunk_text(mode="markdown")` のような引数追加も検討可能
- **暫定方針**: `chunk_markdown()` を新メソッドとして追加し、`chunk_result()` 内で source_type に応じて切り替える

---

## 実装状況

### ✅ 実装済み（2026-04-12）

**`src/rag/github_docs_fetcher.py`**
- `_extract_nodejs_meta(text)` — HTML コメント内 YAML と Stability blockquote からメタを抽出
  - `added_in: "v0.1.90"` / `stability_level: 2` / `stability_label: "Stable"`
  - `source_extra` に自動保存（`RawResult.metadata` 経由）
- `_clean_nodejs_markdown(text)` — 標準クリーナー + Stability 行除去
- `fetch_repo()` — `cleaner_profile: "nodejs_api"` 設定時にプロファイル別処理

**`src/rag/chunker.py`**
- `chunk_markdown(text, min_body_lines=3)` — GITHUB_DOCS 向け見出し単位チャンカー
  - `##` / `###` を強制チャンク境界として使用
  - 本文が `min_body_lines` 未満のセクションは次と結合
  - `chunk_size` 超えのセクションは `chunk_text()` にフォールバック
- `chunk_result()` — `source == "github_docs"` の場合 `chunk_markdown()` を自動選択

**`data/doc_urls/github_doc_repos.yaml`**
- `nodejs/node` に `cleaner_profile: "nodejs_api"` を追加

### ⬜ 未実装（検討中）

**C-1: reflink 除去**（論点1 — 下記参照）
- `[関数名][]` / `[xxx][yyy]` → テキスト保持 or 関連情報として DB 属性化
- See also セクションのリンクを `source_extra.related_refs` として保存する案も検討中
  - DB / BI ツールでの活用 or FAISS エントリへの再マッピングが有望

---

## 再有効化手順（F-5 実装後）

```bash
# 1. Node.js を再有効化
# data/doc_urls/github_doc_repos.yaml の nodejs/node を enabled: true に変更

# 2. dry-run で確認
poetry run python scripts/seed_from_docs.py --source github_docs --max-files 20 --dry-run

# 3. 少量で承認率を確認（目標 70% 以上）
poetry run python scripts/seed_from_docs.py --source github_docs --max-files 20 --mature --provider openrouter

# 4. 問題なければ本番実行
poetry run python scripts/seed_from_docs.py --source github_docs --max-files 80 --mature --provider openrouter
```
