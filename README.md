# MED — Memory Environment Distillation

> RAG × FAISS × LLM × TinyLoRA — Teacher モデルが FAISS メモリを育て、Student モデルに「記憶の使い方」を蒸留するシステム。

---

## アーキテクチャ

```
ブラウザ
  └── Gradio WebGUI (port 7860)
        │
        ▼
  FastAPI Orchestrator (port 8000)
        │
        ├── QueryParser  ── クエリ意図の分類 (LLM)
        │
        └── ModelRouter  ── 複雑度に応じたモデル振り分け
              │
      ┌───────┼───────┐
      ▼       ▼       ▼
   simple  moderate  complex
      │       │       │
  Student  Student  Teacher
  (Qwen-7B  + RAG   (Claude / GPT)
  TinyLoRA)  + KG
      │       │       │
      └───────┴───────┘
              │
      ┌───────▼────────────────────┐
      │       MED RAG Layer         │
      │                             │
      │  ┌─────────┐               │
      │  │  FAISS  │ ← 意味検索     │
      │  │ (海馬)   │               │
      │  └────┬────┘               │
      │       │   ┌──────────────┐ │
      │       ├──▶│ Knowledge    │ │
      │       │   │ Graph (概念) │ │
      │       │   └──────┬───────┘ │
      │       │          │         │
      │       │   ┌──────▼───────┐ │
      │       └──▶│  SQL / BI    │ │
      │           │ (宣言的記憶)  │ │
      │           └──────────────┘ │
      │                             │
      │  外部RAG: GitHub / SO /     │
      │          Tavily / arXiv     │
      └──────────────┬──────────────┘
                     │
             Docker Sandbox
             (コード実行・検証)
```

### 記憶の三層構造

| 層 | 実装 | 役割 |
|---|---|---|
| 海馬（連想記憶） | FAISS + sentence-transformers | 意味検索・エピソード記憶 |
| 概念地図（意味記憶） | Knowledge Graph (NetworkX) | Entity 間の関係・構造 |
| ノート（宣言的記憶） | SQLite / BI | 正確な値・集計クエリ |

### 外部 RAG の動作

外部 RAG は LLM に「ツールを呼ばせる」のではなく、**LLM に渡す前にパイプライン側が自動的に検索→蓄積**する。

```
クエリ受信
  → RetrieverRouter（並列検索）
      → GitHub / StackOverflow / Tavily / arXiv
  → ResultVerifier（関連性フィルタ）
  → Chunker（チャンク分割）
  → FAISS へ蓄積（次回以降はローカル検索で高速応答）
  → 検索結果をコンテキストとして LLM へ注入
  → LLM が回答生成（コンテキストを読むだけ）
```

---

## セットアップ

### 必要環境

- Python 3.11+
- Docker（Sandbox 機能を使う場合）
- 8 GB RAM 以上推奨（Student モデルを動かす場合は GPU 推奨）

### インストール

```bash
git clone <repo-url>
cd MED
pip install -e ".[dev]"
```

### API キー設定

`.env` ファイルをプロジェクトルートに作成:

```env
# AI プロバイダー（使うものだけ設定）
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
TOGETHER_API_KEY=...

# 外部検索 API（使うものだけ設定）
TAVILY_API_KEY=tvly-...
GITHUB_TOKEN=ghp_...       # 任意：レート制限緩和
```

### 使用できる AI プロバイダー

| プロバイダー | 種別 | 設定方法 |
|---|---|---|
| **Anthropic** (Claude Sonnet / Haiku) | クラウド API | `ANTHROPIC_API_KEY` |
| **OpenAI** (GPT-4o / GPT-4o-mini) | クラウド API | `OPENAI_API_KEY` |
| **Together.ai** (Llama / Qwen / Mixtral 等) | クラウド API | `TOGETHER_API_KEY` |
| **Ollama** | ローカル | `http://localhost:11434`（設定不要） |
| **LM Studio** | ローカル | GUI 設定タブ → カスタムプロバイダー |
| **vLLM** | ローカル / サーバー | GUI 設定タブ → カスタムプロバイダー |
| **任意の OpenAI 互換サーバー** | ローカル / クラウド | GUI 設定タブ → カスタムプロバイダー |

**カスタムプロバイダー（LM Studio / vLLM 等）の追加手順:**
1. GUI → 設定タブ → カスタムプロバイダー
2. 名前・エンドポイント URL（例: `http://localhost:1234/v1`）・モデル名を入力
3. 「追加」→「🔌 接続テスト」で疎通確認
4. チャットタブのプロバイダードロップダウンに即反映

### 使用できる検索 API（外部 RAG）

| ソース | 取得できる情報 | API キー |
|---|---|---|
| **GitHub** | コード・Issue・PR | 任意（なくても動作、レート制限あり） |
| **StackOverflow** | Q&A・解決策 | 不要 |
| **Tavily** | Web 全般（最新情報） | 必要（`TAVILY_API_KEY`） |
| **arXiv** | 論文・研究 | 不要 |

API キー未設定のソースは自動的にスキップされます。FAISS メモリにすでに関連ドキュメントが蓄積されている場合は外部検索なしで応答します。

### Docker（Sandbox）

```bash
docker compose up -d
```

---

## 使い方

### GUI 起動（推奨）

```bash
python scripts/launch_gui.py
# → http://localhost:7860 をブラウザで開く
```

### タブ構成

| タブ | 機能 |
|---|---|
| 💬 チャット | RAG + LLM クエリ。プロバイダー・モード・タイムアウトを設定できる |
| 🧠 FAISSメモリ | 蓄積ドキュメントの確認・管理・統計 |
| ⚙️ サンドボックス | コードエディタ + Docker 実行環境 |
| 🎓 学習 | Student モデルの学習制御・進捗可視化 |
| ⚙️ 設定 | API キー管理・カスタムプロバイダー登録・接続テスト |
| 📚 ガイド | ドキュメント Q&A チャットBot |

### チャットモード

| モード | 動作 |
|---|---|
| `auto` | クエリ複雑度に応じて Student / Teacher を自動選択 |
| `simple` | Student モデル（軽量・高速） |
| `teacher` | Teacher モデル（高精度） |

**メモリ使用 ON** → FAISS から関連ドキュメントをコンテキストとして注入
**外部 RAG ON** → GitHub / SO / Tavily / arXiv から検索して FAISS に蓄積

### バックエンド API のみ起動

```bash
python -m uvicorn src.orchestrator.server:app --port 8000 --reload
```

```bash
# クエリ例
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Python でリストをソートするには？", "use_rag": true}'
```

### メモリ初期化（シードデータ投入）

```bash
python scripts/seed_memory.py
```

### Student モデル学習

```bash
python scripts/train_student.py
```

---

## 設定ファイル

| ファイル | 内容 |
|---|---|
| `configs/default.yaml` | 全体設定（埋め込み次元・サンドボックス等） |
| `configs/llm_config.yaml` | プロバイダー設定・タスク別ルーティング |
| `configs/faiss_config.yaml` | FAISS インデックス設定（ドメイン別） |
| `configs/retrievers.yaml` | 外部検索ソース設定 |
| `configs/training.yaml` | 学習アルゴリズム・アダプタ設定 |

---

## 技術スタック

| レイヤー | 技術 |
|---|---|
| Web GUI | Gradio |
| API サーバー | FastAPI |
| 埋め込み | sentence-transformers (all-MiniLM-L6-v2, 384 次元) |
| ベクトル検索 | FAISS |
| Knowledge Graph | NetworkX |
| メタデータ | SQLite (aiosqlite) |
| コンテナ | Docker |
| Student 学習 | VERL (GRPO) / trl / TinyLoRA |
