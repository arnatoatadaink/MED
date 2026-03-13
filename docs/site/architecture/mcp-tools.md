# MCP Tools — SQL / BI

構造化データへの正確なクエリを提供する MCP (Model Context Protocol) ツールです。

## 概要

| ツール | ファイル | 役割 |
|-------|---------|------|
| **SQL Query Tool** | `src/mcp_tools/sql_query_tool.py` | テキスト → SQL 変換 → 実行 |
| **BI Aggregation Tool** | `src/mcp_tools/bi_aggregation_tool.py` | COUNT / SUM / AVG 集計クエリ |

## SQL Query Tool

自然言語のクエリを SQL に変換して SQLite で実行します。

```python
# 使用例
result = await sql_query_tool.query(
    "Python ドキュメントで承認済みのものを信頼度順に 10 件取得"
)
# → SELECT * FROM documents
#   WHERE domain='python' AND review_status='approved'
#   ORDER BY confidence DESC LIMIT 10
```

### 対象テーブル

Phase 2 は SQLite の `documents` テーブルを対象とします:

| カラム | 型 | 説明 |
|-------|-----|------|
| `doc_id` | TEXT | ドキュメント ID |
| `title` | TEXT | タイトル |
| `domain` | TEXT | ドメイン |
| `quality_score` | REAL | Teacher 審査スコア |
| `confidence` | REAL | 信頼度 |
| `review_status` | TEXT | approved / rejected / unreviewed |
| `difficulty` | TEXT | beginner / intermediate / advanced / expert |
| `teacher_id` | TEXT | 生成した Teacher ID |
| `created_at` | TEXT | 登録日時 |

## BI Aggregation Tool

集計クエリを自然言語で実行します:

```python
# 使用例
result = await bi_tool.aggregate(
    "ドメイン別の承認済みドキュメント数を集計"
)
# → {
#     "python": 3200,
#     "math": 1500,
#     "general": 800,
#     "system": 500
#   }
```

対応集計関数: `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`, `GROUP BY`

## QueryClassifier との連携

```python
query_type = classifier.classify(user_query)

if query_type == QueryType.FACTUAL:
    # "Python ドキュメントは何件ありますか？" → SQL/BI
    result = await sql_query_tool.query(user_query)
elif query_type == QueryType.SEMANTIC:
    # "二分探索の実装方法" → FAISS
    result = await faiss_index.search(user_query)
```

## 将来の拡張

Phase 3 以降では SQLite から PostgreSQL への移行を予定:

```yaml
# configs/default.yaml (将来)
database:
  backend: postgresql
  host: localhost
  port: 5432
  name: med_db
```
