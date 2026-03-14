# Fusion / Rerank

FAISS・Knowledge Graph・SQL/BI の検索結果を統合してランキングする層です。

## Reciprocal Rank Fusion (RRF)

複数の検索結果リストを統合する際に RRF を使用します:

```
RRF スコア = Σ 1 / (k + rank_i)
k = 60 (定数)
```

RRF は各システムの絶対スコアではなく**順位**を使うため、
スコールスケールの違いに影響されません。

## 検索フロー

```
クエリ
    ↓
QueryClassifier → SEMANTIC / FACTUAL / RELATIONAL / HYBRID
    ↓
┌──────────┬──────────┬──────────┐
│  FAISS   │   KG     │  SQL/BI  │
│ 意味検索  │ 関係検索  │ 構造検索  │
└──────────┴──────────┴──────────┘
    ↓              ↓         ↓
         FusionReranker (RRF)
                 ↓
         統合ランキング結果
```

## クエリタイプ別の処理

| クエリタイプ | FAISS | KG | SQL/BI | 説明 |
|------------|-------|-----|--------|------|
| `SEMANTIC` | ✅ | ─ | ─ | 意味的類似検索のみ |
| `FACTUAL` | ─ | ─ | ✅ | 正確な事実検索のみ |
| `RELATIONAL` | ─ | ✅ | ─ | Entity 間関係の検索のみ |
| `HYBRID` | ✅ | ✅ | ✅ | 全システムの RRF 統合 |

## FusionReranker API

```python
# src/retrieval/fusion_reranker.py
class FusionReranker:
    async def fuse(
        self,
        faiss_results: list[ScoredDoc],
        kg_results: list[ScoredDoc],
        sql_results: list[ScoredDoc],
        k: int = 60,
    ) -> list[ScoredDoc]:
        ...
```

## Phase 2 での CrossEncoder 再ランキング

RRF 統合後、上位 K 件を CrossEncoder でさらに再ランキングします:

```
RRF 統合結果 (上位 20 件)
    ↓
CrossEncoder (Bi-encoder より高精度)
    ↓
最終ランキング (上位 5 件)
```

| 方式 | 速度 | 精度 | フェーズ |
|------|------|------|---------|
| FAISS (Bi-encoder) | O(1) | 近似 | Phase 1〜 |
| RRF Fusion | O(N) | 中 | Phase 1.5〜 |
| CrossEncoder | O(N) | 高精度 | Phase 2〜 |
