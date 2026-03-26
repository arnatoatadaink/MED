# plan_neat_hyp_e.md
> Status: plan | Created: 2026-03-26
> 元仮説: hyp_h_neat.md > 仮説E
> 実装はClaude Code CLI（WSL2）で行う

---

## 目的

FAISSの検索結果を `association_fn(query, candidate, context)` で
再スコアリングし、文脈依存・非対称な連想検索を実現する。

アーキテクチャは将来NEATで進化させることを想定した
プレースホルダー設計とする。

---

## 設計

### インターフェース

```python
results = searcher.search(
    query_emb=q,       # 検索クエリのembedding
    context_emb=ctx,   # 文脈（直前の会話 / タスクタイプなど）
    k=5
)
```

### パイプライン

```
query_emb
    │
    ▼
FAISS.search(k * 3件) ─── 候補を広く取る
    │
    ▼
association_fn(query, candidate, context)  ← ここがCPPN的な3項関数
    │                                         現状はMLPで実装
    ▼
rerank(top-k)
    │
    ▼
List[SearchResult]
```

### association_fn の設計（現状 → 将来）

```
現状 (MLP固定アーキテクチャ):
    score = w0 * cosine(q, c)
          + w1 * cosine(q, ctx)
          + w2 * cosine(c, ctx)
          + w3 * cosine(q-ctx, c)   ← 文脈差分ベクトルとの類似度
    w_i は学習可能パラメータ（初期値は等重み）

将来 (NEAT進化):
    score = CPPN(q, c, ctx)
    ↑ このアーキテクチャ自体をNEATで進化させる
```

### SearchResult

```python
@dataclass
class SearchResult:
    index:       int
    text:        str
    base_score:  float    # FAISSのコサイン類似度
    assoc_score: float    # association_fn の出力
    final_score: float    # alpha * base + (1-alpha) * assoc
    embedding:   np.ndarray
```

---

## ファイル構成

```
context_search.py       # メイン実装
  └─ AssociationFn      # 3項スコア関数（NEAT進化のプレースホルダー）
  └─ ContextSensitiveSearch  # FAISSラッパー
  └─ SearchResult       # 結果データクラス

test_context_search.py  # 動作確認スクリプト
  └─ numpy fallback テスト（faiss/torch なしで動作確認）
  └─ 文脈ありなしの比較テスト
```

---

## 実装タスク（Claude CLI向け）

### Phase 1: コア実装

- [ ] `AssociationFn` — numpy版（torch不要）を先に実装
  - `fit(feedback_pairs)` で重みを更新できる設計
  - 重みをJSONで保存・ロードできる
- [ ] `ContextSensitiveSearch`
  - `build_index(embeddings, texts)`
  - `search(query_emb, context_emb, k, alpha=0.5)`
  - faiss未インストール時はnumpyブルートフォースにフォールバック
- [ ] `SearchResult` dataclass

### Phase 2: テスト

- [ ] `test_context_search.py`
  - ダミーembedding（np.random）で動作確認
  - `context_emb=None`（文脈なし）と `context_emb=ctx`（文脈あり）で
    結果の差を比較出力
  - スコアの内訳（base / assoc / final）をログ出力

### Phase 3: MEDへの統合確認

- [ ] MEDの既存FAISSモジュールへの差し込み方を確認
- [ ] `context_emb` の生成元を決定
  - 候補A: 直前のTeacher応答のembedding
  - 候補B: タスクタイプの固定embedding（"creative" / "factual" など）
  - 候補C: 直近N件のやり取りのmean pooling

---

## 評価指標

```
定量:
  文脈ありなしでの検索結果の差（順位変動率）
  同一クエリ・異なるcontextでの結果多様性

定性:
  「犬」+ context="科学"  → 哺乳類・条件反射 が上位に来るか
  「犬」+ context="創作"  → 忠誠・孤独・友情 が上位に来るか
```

---

## 昇格履歴

| From | To | Date | 理由 |
|------|----|------|------|
| hyp_h_neat.md 仮説E | plan_neat_hyp_e.md | 2026-03-26 | MLP版が数十行で実装可能と判断 |

---

## Update Log

| Date | Note |
|------|------|
| 2026-03-26 | Initial plan from hyp_h_neat.md 仮説E昇格 |
