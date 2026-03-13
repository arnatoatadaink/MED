# FAISSメモリ管理

ベクトルインデックス（FAISS）に知識を蓄積・検索するメモリ管理機能です。

## 概念図

```
テキスト → [Embedder: all-MiniLM-L6-v2] → ベクトル (768次元)
                                               ↓
                                     [FAISS IndexFlatIP]
                                               ↓
クエリ → ベクトル化 → 上位 K 件を取得 → LLM に渡して回答生成
```

## インデックス構成

| フェーズ | インデックス種別 | 条件 |
|---------|---------------|------|
| Phase 1 | `IndexFlatIP` (完全探索) | ドキュメント数 ≤ 100,000 |
| Phase 2+ | `IndexIVFFlat` (近似) | ドキュメント数 > 100,000 |
| Phase 3+ | `IndexIVFPQ` (圧縮) | メモリ制約がある場合 |

自動切り替えは `src/memory/faiss_index.py` が担当します。

## ドメイン

| ドメイン | 用途 |
|---------|------|
| `python` | Python コード・ライブラリ・バグ修正 |
| `math` | 数式・アルゴリズム・最適化 |
| `general` | 汎用知識・概念説明 |
| `system` | システム設計・アーキテクチャ |

ドメインごとに独立した FAISS インデックスが作られます。
クエリ時は全ドメインを横断検索するか、特定ドメインに絞ることができます。

## GUI でのドキュメント追加

1. **🧠 FAISSメモリ** タブを開く
2. 「ドキュメント追加」タブを選択
3. タイトル・本文・ドメインを入力
4. 「**追加**」ボタンを押す

即座に埋め込みが計算され FAISS に登録されます。

## スクリプトでのバルク投入

```bash
# シードデータを一括投入
python scripts/seed_memory.py --domain python --file data/python_docs.jsonl

# メモリ品質審査（Phase 2）
python scripts/mature_memory.py --limit 100
```

## セマンティック検索

```bash
# API で直接検索
curl http://localhost:8000/memory/search \
  -H "Content-Type: application/json" \
  -d '{"query": "binary search python", "top_k": 5}'
```

レスポンス例:

```json
{
  "results": [
    {
      "doc_id": "abc123",
      "title": "Binary Search Implementation",
      "score": 0.94,
      "domain": "python",
      "preview": "def binary_search(arr, target): ..."
    }
  ]
}
```

## メモリを育てるには

1. **チャットで質問する** → Teacher の回答が自動的にメモリに保存
2. **スクリプトでシードデータを投入** → `scripts/seed_memory.py`
3. **Phase 2 成熟プロセスを実行** → `scripts/mature_memory.py`

詳細は [Phase 2 メモリ成熟](../phase2/overview.md) を参照してください。

## スコアリング

各ドキュメントは 3 種類のスコアの合成値でランキングされます:

| スコア | 説明 | 重み |
|-------|------|------|
| **Usefulness** | 有用性（正確性・完全性・明瞭性） | 0.5 |
| **Freshness** | 鮮度（ドメイン別指数減衰） | 0.3 |
| **Teacher Trust** | Teacher モデルの信頼度 | 0.2 |

```
composite_score = usefulness × freshness × teacher_trust
```
