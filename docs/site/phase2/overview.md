# Phase 2 — メモリ成熟 概要

Teacher モデルが FAISS メモリを「成熟」させるフェーズです。
品質の低いドキュメントを審査・タグ付けし、Student の学習品質を高めます。

## Phase 2 品質目標

| 目標 | 値 | 確認場所 |
|------|-----|---------|
| ドキュメント数 | ≥ 10,000 docs | 成熟管理 → 品質レポート |
| 平均信頼度 | ≥ 0.7 | 同上 |
| コード実行成功率 | ≥ 80% | 同上 |

## Phase 2 の構成要素

```
Phase 2 メモリ成熟
    │
    ├── Teacher 信頼度評価 (TeacherRegistry)
    │       Teacher ごとの信頼度を EWMA で管理
    │       CompositeScore に自動反映
    │
    ├── MemoryReviewer
    │       Teacher LLM が未審査ドキュメントを品質審査
    │       approved / rejected のラベル付け
    │
    ├── DifficultyTagger
    │       beginner / intermediate / advanced / expert の難易度付与
    │       Student 学習のカリキュラム順序を決定
    │
    └── CrossEncoder (再ランキング)
            FAISS 上位 K 件を意味的に再ランキング
            Bi-encoder より高精度
```

## GUI からのアクセス

1. **🧠 FAISSメモリ** タブを開く
2. **🔬 成熟管理 (Phase 2)** サブタブを選択

3 つのサブタブが利用可能:

| サブタブ | 機能 |
|---------|------|
| 品質レポート | Phase 2 進捗を ASCII プログレスバーで可視化 |
| Teacher 信頼度 | 各 Teacher の trust_score / フィードバック数を一覧表示 |
| 一括審査 | limit / concurrency を指定して未審査ドキュメントを一括処理 |

## API エンドポイント

| エンドポイント | メソッド | 説明 |
|--------------|---------|------|
| `/maturation/quality` | GET | 品質レポート取得 |
| `/maturation/teachers` | GET | Teacher プロファイル一覧 |
| `/maturation/review` | POST | 未審査ドキュメントの一括審査 |

## スクリプトから実行

```bash
# メモリ成熟を実行（一括審査 + 難易度タグ付け）
python scripts/mature_memory.py --limit 100

# Teacher 信頼度を確認
python scripts/mature_memory.py --show-teachers

# 難易度タグ付けのみ
python scripts/mature_memory.py --tag-difficulty --limit 200
```

## 次のステップ

- [Teacher 信頼度評価の仕組み](teacher-trust.md)
- [MemoryReviewer のフロー](reviewer.md)
- [品質メトリクスの読み方](quality-metrics.md)
