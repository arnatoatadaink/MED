# アーキテクチャ概要

## システム全体図

```
ブラウザ / CLI
    │
    ▼
Gradio WebGUI (5タブ)
    │
    ▼
Orchestrator (FastAPI :8000)
    │
    ├── Query Parser (LLM)
    │       クエリの複雑さを判定 → simple / moderate / complex
    │
    ├── Graph-aware Model Router
    │       │
    │       ├── simple   → Student (Qwen-7B + TinyLoRA)
    │       ├── moderate → Student + FAISS + 外部RAG
    │       └── complex  → Teacher (Claude / GPT)
    │
    ├── MED RAG Layer
    │       │
    │       ├── FAISS (意味検索)
    │       │       ↕ KG Bridge (Fusion / Rerank)
    │       ├── Knowledge Graph (関係・構造)
    │       │       ↕ Structured Filter
    │       └── SQL / BI (正確検索)
    │
    └── Docker Sandbox
            コード実行・セキュリティ・自動リトライ
```

## コンポーネント一覧

| コンポーネント | ファイル | 役割 |
|-------------|---------|------|
| **Orchestrator** | `src/orchestrator/server.py` | FastAPI エントリポイント |
| **Query Parser** | `src/orchestrator/query_parser.py` | LLM ベース意図分類 |
| **Model Router** | `src/orchestrator/model_router.py` | Teacher/Student 振り分け |
| **LLM Gateway** | `src/llm/gateway.py` | プロバイダー抽象化 |
| **FAISS Memory** | `src/memory/faiss_index.py` | ドメイン別ベクトルインデックス |
| **Memory Manager** | `src/memory/memory_manager.py` | FAISS + SQLite 原子的操作 |
| **Knowledge Graph** | `src/knowledge_graph/store.py` | NetworkX → Neo4j |
| **RAG Retriever** | `src/rag/retriever.py` | GitHub / SO / Tavily / arXiv |
| **Fusion Reranker** | `src/retrieval/fusion_reranker.py` | RRF ベース結果融合 |
| **Docker Sandbox** | `src/sandbox/manager.py` | コード実行・セキュリティ |
| **Training Pipeline** | `src/training/pipeline.py` | GRPO + TinyLoRA 学習制御 |

## データフロー（チャットクエリ）

```
1. ユーザー入力
      ↓
2. Query Parser: 複雑さ判定 (simple / moderate / complex)
      ↓
3. Model Router: Teacher or Student を選択
      ↓
4. FAISS Memory: 類似ドキュメントを K 件取得
      ↓
5. 外部RAG (必要なら): GitHub / SO / Tavily を検索
      ↓
6. Knowledge Graph: Entity 関係性でコンテキストを補強
      ↓
7. LLM: コンテキスト付きで回答生成
      ↓
8. コード含む場合: Docker Sandbox で実行・検証
      ↓
9. 回答を FAISS + KG に保存（有用性スコア付き）
      ↓
10. ユーザーに返答
```

## 記憶システムの設計思想

MEDの記憶設計は人の認知モデルに対応しています:

```
海馬（連想記憶）    ← FAISS         高速・近似・エピソード
概念地図（意味記憶） ← Knowledge Graph 関係性・構造・ルーティング補助
ノート（宣言的記憶） ← SQL / BI      正確・構造化・集計
```

KG は FAISS と SQL/BI の**橋渡し層**として機能し、単独で答えを出さずルーティングと融合を担います。

## ディレクトリ構造

```
src/
├── orchestrator/       FastAPI + Query Parser + Model Router
├── llm/               LLM プロバイダー抽象化
│   └── providers/     anthropic / openai / ollama / vllm
├── rag/               外部検索パイプライン
│   └── retrievers/    github / stackoverflow / tavily / arxiv
├── memory/            FAISS + SQLite メモリ管理
│   ├── scoring/       freshness / usefulness / composite
│   ├── learning/      LTR / Cross-Encoder / feedback
│   └── maturation/    reviewer / difficulty_tagger / quality_metrics
├── knowledge_graph/   KG ストア + Entity 抽出
├── retrieval/         query_classifier + fusion_reranker
├── mcp_tools/         SQL / BI クエリ
├── training/          GRPO + TinyLoRA 学習フレームワーク
│   ├── algorithms/    grpo / ppo / dpo / sft
│   ├── adapters/      tinylora / lora / lora_xs
│   └── rewards/       composite / code_exec / teacher_eval
├── sandbox/           Docker 管理 + セキュリティ
└── gui/               Gradio Web UI (5タブ)
    └── tabs/          chat / memory / sandbox / training / settings / guide
```
