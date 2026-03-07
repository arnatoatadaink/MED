MED Framework — Knowledge Graph 統合計画

Claude Code 引き継ぎ用 plan.md 作成日: 2026-03-07 ステータス: 設計フェーズ

1. 背景・目的

現状のMEDアーキテクチャ

Query → ModelRouter → FAISS (非構造化・意味検索)                     → SQL/BI (構造化・正確検索)                     → Teacher/Student LLM 

課題

問題 詳細 FAISSの近似誤り データ増加で連想精度が低下（人の「誤想起」に相当） SQL/FAISSの断絶 意味検索と構造検索が独立しており、融合した推論が不可 関係性の欠如 Entity間の関係（因果・階層・時系列）がどちらにも存在しない

目的

FAISS（連想記憶）とSQL/BI（宣言的記憶）の橋渡し層として Knowledge Graph（KG）を導入し、MEDの推論品質を向上させる。

2. 設計思想

人の思考との対応関係

FAISS           → 海馬（エピソード記憶・連想）    速いが近似 Knowledge Graph → 概念地図（意味記憶・関係性）    中間層 SQL/BI          → 書き起こしたノート（宣言的記憶） 遅いが正確 

設計原則





既存コンポーネントを壊さない — FAISS・SQL/BIはそのまま保持



KGは橋渡しに徹する — KG単独で答えを出すのではなくルーティングと融合を担う



段階的導入 — Phase 1〜3に分けて動作確認しながら拡張



コスト意識 — GraphDB新規導入は最小構成から。既存インフラ優先

3. ターゲットアーキテクチャ

                         ┌─────────────────────────────────┐                          │         MED RAG Layer            │                          │                                  │ Query ──→ Graph-aware ───┼──→ FAISS        ←──┐            │            Router        │    (意味検索)        │  Fusion /  │                          │         │        KG Bridge  Rerank│                          │         ↓           │            │                          │  Knowledge Graph ───┘            │                          │    (関係・構造)   ←──┐            │                          │         │            │            │                          │         ↓            │            │                          │    SQL / BI      Structured       │                          │    (正確検索)    Filter           │                          └──────────┬──────────────────────-┘                                     ↓                              Teacher / Student LLM 

各コンポーネントの責務

コンポーネント 役割 変更 FAISS 非構造化テキストの意味検索 変更なし Knowledge Graph Entity・関係性の管理、検索ルーティング補助 新規追加 SQL/BI MCP 構造化データの正確なクエリ 変更なし（MCP拡張） ModelRouter クエリ分類 → KG参照後にルーティング判断 拡張 Fusion/Rerank FAISS + KG + SQL 結果の統合 新規追加

4. Knowledge Graph 技術選定

候補比較

技術 特徴 MEDとの相性 Neo4j 最も成熟、Cypher言語、豊富なPythonサポート ◎ 本格運用向き NetworkX Pure Python、軽量、インメモリ ◎ Phase1プロトタイプ向き Cognee Vector+Graph統合ライブラリ、自動グラフ構築 ○ 将来検討 Weaviate Vector+Structured統合DB ○ FAISS置き換え候補

推奨方針

Phase 1: NetworkX（インメモリ、依存ゼロ、プロトタイプ） Phase 2: Neo4j（永続化、本格クエリ） Phase 3: Cognee / Weaviate 評価（FAISS統合検討） 

5. 実装計画

Phase 1 — KGプロトタイプ（推定工数: 3〜5日）

目標: NetworkXでKGを構築し、ModelRouterがKGを参照できる最小実装

タスク:





[ ] med/knowledge_graph/ ディレクトリ作成



[ ] KnowledgeGraphStore クラス実装（NetworkX backend）





add_entity(entity_id, label, properties)



add_relation(src, dst, relation_type, properties)



query_neighbors(entity_id, depth=2)



find_path(src_entity, dst_entity)



[ ] EntityExtractor 実装（Teacher API呼び出しでEntityを自動抽出）



[ ] FAISS格納時にKGへも自動登録するパイプライン



[ ] ModelRouter にKG参照ロジック追加





クエリからEntity候補を抽出



KGでEntity存在確認 → ルーティング判断に利用



[ ] 単体テスト追加

成果物:

med/   knowledge_graph/     __init__.py     store.py          # KnowledgeGraphStore     extractor.py      # EntityExtractor     router_bridge.py  # RouterとKGの接続   tests/     test_knowledge_graph.py 

Phase 2 — SQL/BI MCP統合 + Fusion層（推定工数: 5〜7日）

目標: KG・FAISS・SQLの3検索結果をFusion/Rerankで統合

タスク:





[ ] SQL/BI MCPツール実装





SQLQueryTool: テキスト → SQL変換 → 実行



BIAggregationTool: 集計クエリ（COUNT/SUM/AVG）



[ ] クエリ分類ロジック強化

class QueryClassifier:    SEMANTIC   = "faiss"      # 「〜に似たものは？」    FACTUAL    = "sql"        # 「〜の件数は？」「〜の値は？」    RELATIONAL = "kg"         # 「〜と〜の関係は？」    HYBRID     = "all"        # 複合クエリ 



[ ] FusionReranker 実装





FAISS スコア + KG パス長 + SQL 確実性スコアで重み付け



Reciprocal Rank Fusion (RRF) ベース



[ ] Neo4j 移行スクリプト（NetworkX → Neo4j）



[ ] 統合テスト追加

成果物:

med/   mcp_tools/     sql_query_tool.py     bi_aggregation_tool.py   retrieval/     fusion_reranker.py     query_classifier.py 

Phase 3 — Student訓練品質向上（推定工数: 5日〜）

目標: KG由来の構造的根拠をStudent訓練データに反映

タスク:





[ ] KGパスをTeacherへのプロンプトに含める（CoT強化）



[ ] 訓練データ生成時にKG根拠をアノテーション



[ ] GRPO報酬関数にKG整合性スコアを追加検討



[ ] 評価指標にEntity精度・関係再現率を追加

6. ディレクトリ構成（変更後）

med/ ├── knowledge_graph/          # ★ 新規 │   ├── __init__.py │   ├── store.py              # KnowledgeGraphStore (NetworkX/Neo4j) │   ├── extractor.py          # EntityExtractor │   └── router_bridge.py      # Router連携 ├── retrieval/                # ★ 新規 │   ├── query_classifier.py   # クエリ分類 │   └── fusion_reranker.py    # Fusion/Rerank ├── mcp_tools/                # ★ 拡張 │   ├── faiss_tool.py         # 既存 │   ├── sql_query_tool.py     # 新規 │   └── bi_aggregation_tool.py# 新規 ├── router/ │   └── model_router.py       # KG参照ロジック追加 ├── memory/ │   └── faiss_store.py        # 既存（KG登録フック追加） └── training/     └── grpo_trainer.py       # 既存（Phase3で拡張） 

7. 依存ライブラリ

# Phase 1 networkx = ">=3.0" spacy = ">=3.7"           # Entity抽出（軽量モデル使用）  # Phase 2 neo4j = ">=5.0"           # Neo4j Python driver sqlalchemy = ">=2.0"      # SQL抽象化  # Phase 3（評価） cognee = ">=0.1"          # 将来検討 

8. 参照アーキテクチャ・論文

名称 参照ポイント GraphRAG (Microsoft, 2024) Vector + KG統合の基本設計 HippoRAG (2024) 海馬モデルのRAG実装、誤想起軽減 Self-RAG (2023) 検索ルーティングの判断ロジック REALM (2020) 外部知識統合の基礎研究 RRF (Reciprocal Rank Fusion) Fusionアルゴリズム

9. 作業開始チェックリスト（Claude Code用）

# 1. 現状確認 cat CLAUDE.md ls med/  # 2. Phase 1 開始 mkdir -p med/knowledge_graph mkdir -p med/retrieval  # 3. 既存テスト確認 pytest tests/ -v  # 4. KnowledgeGraphStore から実装開始 # → med/knowledge_graph/store.py 

10. 判断が必要な未決事項

項目 選択肢 優先度 KG永続化タイミング Phase1でNetworkX+pickle / 即Neo4j 中 Entity抽出モデル spaCy小モデル / Teacher API呼び出し 高 SQL対象DB SQLite（ローカル） / PostgreSQL 中 KGスキーマ設計 汎用 / MED特化 高

推奨: Entity抽出はTeacher API（Gemini）を使い精度優先。SQLはSQLiteから開始。

このplanはMED Framework Knowledge Graph統合の基本設計です。 実装中に発見した課題はCLAUDE.mdに追記してください。