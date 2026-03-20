# MED — Memory Environment Distillation

> RAG × FAISS × Knowledge Graph × LLM × TinyLoRA
> Teacher モデルが FAISS メモリを育て、Student モデルに「記憶の使い方」を蒸留するシステム。

---

## 実装状況

| フェーズ | 内容 | 状態 |
|---------|------|------|
| Phase 1 | コア基盤（FAISS / RAG / LLM / Sandbox / Orchestrator） | ✅ 完了 |
| Phase 1.5 | Knowledge Graph（NetworkX）+ Fusion/Rerank | ✅ 完了 |
| Phase 2 | メモリ成熟 + SQL/BI MCP + Cross-Encoder | ✅ 完了 |
| Phase 3 | 学習フレームワーク骨格（GRPO / PPO / DPO / TinyLoRA） | ✅ 骨格完了 |
| Phase 4 | Graph-aware ルーティング + 運用最適化 | ✅ 完了 |
| GUI | Gradio 6タブ構成 | ✅ 完了 |
| テスト | unit tests 36ファイル / 1079件 pass | ✅ 完了 |
| モデル | all-MiniLM-L6-v2 ローカル配置済み（HF Hub 不要） | ✅ 完了 |

**将来対応:** GRPO/TinyLoRA 本番学習（VERL/trl 統合） / Neo4j 移行 / Docker E2E テスト

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
      ┌───────▼─────────────────────┐
      │        MED RAG Layer         │
      │                              │
      │  FAISS (海馬・連想記憶)       │
      │    ↕ KG Bridge               │
      │  Knowledge Graph (概念地図)   │
      │    ↕ Structured Filter       │
      │  SQL / BI (宣言的記憶)        │
      │                              │
      │  外部RAG: GitHub / SO /      │
      │          Tavily / arXiv      │
      └──────────────┬───────────────┘
                     │
             Docker Sandbox
             (コード実行・検証)
```

### 記憶の三層構造

| 層 | 実装 | 役割 |
|---|---|---|
| 海馬（連想記憶） | FAISS + all-MiniLM-L6-v2 (384次元) | 意味検索・エピソード記憶 |
| 概念地図（意味記憶） | Knowledge Graph (NetworkX) | Entity 間の関係・構造・ルーティング |
| ノート（宣言的記憶） | SQLite / BI MCP | 正確な値・集計クエリ |

---

## セットアップ

### 必要環境

- Python 3.11+
- Docker（Sandbox 機能を使う場合）
- 8 GB RAM 以上推奨

### インストール

```bash
git clone <repo-url>
cd MED
pip install -e ".[dev]"
```

### 埋め込みモデル

`all-MiniLM-L6-v2` はリポジトリに同梱済み（`data/models/all-MiniLM-L6-v2/`）。
HuggingFace Hub への接続は不要です。`configs/default.yaml` に設定済み。

```yaml
embedding:
  model: "all-MiniLM-L6-v2"
  cache_dir: "data/models"   # ローカルモデルを参照
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

| プロバイダー | 種別 | 設定 |
|---|---|---|
| Anthropic (Claude) | クラウド API | `ANTHROPIC_API_KEY` |
| OpenAI (GPT-4o) | クラウド API | `OPENAI_API_KEY` |
| Together.ai | クラウド API | `TOGETHER_API_KEY` |
| Ollama | ローカル | 設定不要 (`localhost:11434`) |
| LM Studio / vLLM / OpenAI互換 | ローカル | GUI 設定タブで登録 |

### 使用できる外部 RAG ソース

| ソース | 取得できる情報 | API キー |
|---|---|---|
| GitHub | コード・Issue・PR | 任意（なくても動作） |
| StackOverflow | Q&A・解決策 | 不要 |
| Tavily | Web 全般（最新情報） | 必要 |
| arXiv | 論文・研究 | 不要 |

未設定のソースは自動スキップ。FAISS に関連ドキュメントが蓄積済みであれば外部検索なしで応答します。

### Docker（Sandbox）

```bash
docker compose up -d
```

---

## 使い方

### GUI 起動（推奨）

```bash
python scripts/launch_gui.py
# → http://localhost:7860
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

### バックエンド API のみ起動

```bash
python -m uvicorn src.orchestrator.server:app --port 8000 --reload
```

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Python でリストをソートするには？", "use_rag": true}'
```

### スクリプト

```bash
# FAISS メモリにシードデータを投入（外部 RAG 経由）
python scripts/seed_memory.py --query "FAISS vector search" --domain code

# JSON ファイルから直接投入
python scripts/seed_memory.py --input-file docs.json

# メモリ成熟（Teacher による品質審査・難易度付与）
python scripts/mature_memory.py

# Student モデル学習
python scripts/train_student.py
```

---

## 設定ファイル

| ファイル | 内容 |
|---|---|
| `configs/default.yaml` | 全体設定（埋め込み・サンドボックス等） |
| `configs/llm_config.yaml` | プロバイダー設定・タスク別ルーティング |
| `configs/faiss_config.yaml` | FAISS インデックス設定（ドメイン別） |
| `configs/retrievers.yaml` | 外部検索ソース設定 |
| `configs/training.yaml` | 学習アルゴリズム・アダプタ設定 |
| `configs/model_router.yaml` | Model Router・KG 参照設定 |

---

## ディレクトリ構成（主要部）

```
MED/
├── src/
│   ├── orchestrator/       # FastAPI + QueryParser + ModelRouter
│   ├── llm/                # LLMGateway + 4プロバイダー
│   ├── rag/                # RetrieverRouter + GitHub/SO/Tavily/arXiv
│   ├── memory/             # FAISS + SQLite + スコアリング + 成熟管理
│   ├── knowledge_graph/    # NetworkX KG + EntityExtractor + RouterBridge
│   ├── retrieval/          # QueryClassifier + RRF Fusion/Rerank
│   ├── mcp_tools/          # SQL/BI MCP ツール
│   ├── training/           # GRPO/PPO/DPO/SFT + TinyLoRA/LoRA（骨格）
│   ├── sandbox/            # Docker コード実行
│   ├── gui/                # Gradio 6タブ
│   └── common/             # Config + Logger
├── data/
│   ├── models/
│   │   └── all-MiniLM-L6-v2/   # 埋め込みモデル（同梱）
│   └── faiss_indices/           # FAISS インデックス（ドメイン別）
├── configs/                # YAML 設定ファイル群
├── scripts/                # seed / mature / train / launch
└── tests/                  # unit (36ファイル / 1079件) + integration
```

---

## 技術スタック

| レイヤー | 技術 |
|---|---|
| Web GUI | Gradio |
| API サーバー | FastAPI (非同期) |
| 埋め込み | sentence-transformers / all-MiniLM-L6-v2 (384次元) |
| ベクトル検索 | FAISS (IndexFlatIP → IVF 自動移行) |
| Knowledge Graph | NetworkX (→ Neo4j: 将来) |
| メタデータ | SQLite (aiosqlite) |
| Fusion / Rerank | RRF (Reciprocal Rank Fusion) |
| コンテナ | Docker |
| Student 学習 | GRPO + TinyLoRA 骨格実装（VERL/trl 統合: 将来） |
| 設定 | pydantic-settings + YAML |
