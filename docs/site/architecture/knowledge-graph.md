# Knowledge Graph

FAISS（連想記憶）と SQL/BI（宣言的記憶）の橋渡し層として機能する知識グラフです。

## 設計原則

- **橋渡しに徹する**: KG 単独で答えを出さず、ルーティングと融合を担う
- **既存非破壊**: FAISS・SQL/BI を壊さず橋渡し層として追加
- **段階的導入**: NetworkX（Phase 1.5）→ Neo4j（Phase 2+）

## 記憶モデルとの対応

```
海馬（連想記憶）    ← FAISS         高速・近似・エピソード
概念地図（意味記憶） ← Knowledge Graph 関係性・構造・ルーティング補助
ノート（宣言的記憶） ← SQL / BI      正確・構造化・集計
```

## 主要クラス

### KnowledgeGraphStore

```python
# src/knowledge_graph/store.py
class KnowledgeGraphStore:
    def add_entity(self, entity_id: str, label: str, props: dict) -> None
    def add_relation(self, src: str, rel_type: str, dst: str, props: dict) -> None
    def query_neighbors(self, entity_id: str, depth: int = 1) -> list[dict]
    def find_path(self, src: str, dst: str, max_hops: int = 3) -> list[str]
    def search_entities(self, query: str, top_k: int = 5) -> list[dict]
```

### EntityExtractor

```python
# src/knowledge_graph/extractor.py
class EntityExtractor:
    # Teacher API を使用して Entity・関係性を抽出
    async def extract(self, text: str) -> dict:
        # returns: {"entities": [...], "relations": [...]}
```

## フェーズ別技術選定

| フェーズ | バックエンド | 理由 |
|---------|-----------|------|
| Phase 1.5 | **NetworkX** | インメモリ・依存ゼロ・プロトタイプ向き |
| Phase 2+ | **Neo4j** | 永続化・Cypher クエリ・本格運用向き |

## FAISS 格納時の自動登録

ドキュメントを FAISS に追加すると、自動的に KG にも登録されます:

```python
# memory_manager.py (概念)
async def add_document(doc: Document):
    # 1. FAISS に埋め込みを追加
    await faiss_index.add(doc)

    # 2. SQLite メタデータを保存
    await metadata_store.insert(doc)

    # 3. KG に Entity・関係性を登録 (フック)
    entities = await entity_extractor.extract(doc.content)
    for entity in entities:
        kg_store.add_entity(entity)
```

## クエリ分類との連携

```python
# query_classifier.py
class QueryType(Enum):
    SEMANTIC   = "semantic"    # → FAISS で処理
    FACTUAL    = "factual"     # → SQL/BI で処理
    RELATIONAL = "relational"  # → KG で処理
    HYBRID     = "hybrid"      # → Fusion/Rerank で処理
```

## 永続化

Phase 1.5 では NetworkX グラフを pickle で保存します:

```bash
# 保存先
data/knowledge_graph/graph.pkl
```

Phase 2 で Neo4j に移行する際は移行スクリプトを使います（実装予定）。
