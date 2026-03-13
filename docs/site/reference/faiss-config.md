# faiss_config.yaml リファレンス

`configs/faiss_config.yaml` の全フィールドの説明です。

## 設定例

```yaml
# 埋め込みモデル
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 768
  device: "cpu"           # cpu / cuda / mps

# インデックス設定
index:
  type: "auto"            # auto / flat / ivf / ivfpq
  metric: "ip"            # ip (内積) / l2 (L2距離)
  ivf_nlist: 100          # IVF のクラスタ数
  pq_m: 8                 # PQ の部分空間数
  auto_threshold: 100000  # flat → IVF の自動切り替え件数

# ドメイン別インデックス
domains:
  - python
  - math
  - general
  - system

# 検索パラメータ
search:
  default_top_k: 5
  max_top_k: 50
  nprobe: 10              # IVF 検索時の探索クラスタ数

# 保存先
storage:
  index_dir: "data/faiss_indices"
  metadata_db: "data/metadata.db"
```

## インデックス種別の選択基準

| ドキュメント数 | 推奨インデックス | 特徴 |
|-------------|---------------|------|
| ≤ 100,000 | `IndexFlatIP` | 完全探索・最高精度 |
| 100,000〜1M | `IndexIVFFlat` | 近似・高速 |
| > 1M | `IndexIVFPQ` | 圧縮・最小メモリ |

`type: "auto"` を指定すると `auto_threshold` に基づいて自動切り替えします。

## 埋め込みモデル

| モデル | 次元 | 用途 |
|-------|------|------|
| `all-MiniLM-L6-v2` | 384 | 汎用・高速（デフォルト） |
| `all-mpnet-base-v2` | 768 | 高精度（推奨） |
| `microsoft/unixcoder-base` | 768 | コード特化 |

!!! note
    次元数を変更した場合は既存インデックスを再構築する必要があります。
