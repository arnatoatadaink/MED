# チャット機能

RAG と FAISS メモリを活用して Teacher / Student モデルへクエリを送ります。

## モデルモード

| モード | 動作 | 推奨用途 |
|--------|------|---------|
| **auto** *(デフォルト)* | クエリ複雑さを自動判定してモデルを選択 | 通常使用 |
| **student** | Qwen-7B + TinyLoRA を強制使用 | 高速・低コストが優先のとき |
| **teacher** | Claude / GPT API を強制使用 | 高精度が必要なとき |

## 検索オプション

| オプション | ON の効果 |
|------------|---------|
| **FAISSメモリ使用** | 過去の会話・登録済みドキュメントをベクトル検索して参照 |
| **外部RAG使用** | GitHub / Stack Overflow / Tavily をリアルタイム検索して回答に組み込む |

どちらも OFF にするとモデルのパラメトリック知識のみで回答します。

## クエリルーティングの仕組み

```python
# src/orchestrator/query_parser.py より

# 1. Query Parser が複雑さを判定
complexity = query_parser.classify(user_query)
# → "simple" | "moderate" | "complex"

# 2. Model Router が担当モデルを決定
if complexity == "simple":
    model = student_model           # Qwen-7B + TinyLoRA
elif complexity == "moderate":
    model = student_model + faiss   # + FAISS 検索
else:
    model = teacher_model           # Claude / GPT
```

## サンプルプロンプト

GUI のチャットタブ内「💡 サンプルプロンプト」アコーディオンから
ワンクリックで以下のサンプルを試せます:

- Python 二分探索実装
- FAISS とは何か
- コードのデバッグ支援
- ソート比較
- 機械学習の評価指標
- pandas DataFrame 処理

## Tips

- `Shift+Enter` で改行、`Enter` または「送信」で送信
- 「**履歴クリア**」ボタンで会話をリセット（FAISS メモリはクリアされません）
- コード質問は **auto / teacher** モードが適切（Sandbox 自動実行が走ります）
- 長いコードブロックは ` ``` ` で囲むと構文ハイライトされます

## レスポンス情報パネル

送信後、右カラムに表示:

```
モデル: claude-sonnet-4-20250514
レイテンシ: 1,234 ms
ソース:
  [1] github.com/... score=0.92
  [2] stackoverflow.com/... score=0.87
```
