# MemoryReviewer

Teacher LLM が未審査ドキュメントを品質審査し、`approved` / `rejected` ラベルを付与します。

## 審査フロー

```
MetadataStore.get_unreviewed(limit) で未審査ドキュメントを取得
    ↓
Teacher LLM に品質評価を依頼（JSON 形式）
    ↓
{quality_score, confidence, approved, reason} を抽出
    ↓
MetadataStore.update_quality() で DB を更新
    ↓
review_status を "approved" / "rejected" に変更
```

## 審査プロンプト

Teacher LLM には以下の形式で評価を依頼します:

```
以下のドキュメントの品質を 0.0〜1.0 で評価してください。

タイトル: {title}
本文: {content[:500]}

JSON で回答してください:
{
  "quality_score": 0.0-1.0,
  "confidence": 0.0-1.0,
  "approved": true/false,
  "reason": "評価理由"
}
```

## 承認閾値

```python
# reviewer.py
APPROVAL_THRESHOLD = 0.6

approved = quality_score >= APPROVAL_THRESHOLD
```

## GUI からの一括審査

**🧠 FAISSメモリ → 🔬 成熟管理 (Phase 2) → 一括審査** タブ:

| パラメータ | 範囲 | 説明 |
|-----------|------|------|
| limit | 1〜200 | 審査するドキュメントの上限数 |
| concurrency | 1〜10 | 同時並列リクエスト数 |

「**一括審査を実行**」ボタンを押すと結果テーブルが表示されます:

```
┌─────────────┬───────┬───────────────────────┐
│ 項目         │ 値    │                       │
├─────────────┼───────┼───────────────────────┤
│ 審査済み     │ 50    │                       │
│ 承認         │ 38    │ ████████░░ 76%        │
│ 却下         │ 12    │ ██░░░░░░░░ 24%        │
│ 平均品質     │ 0.74  │                       │
│ 平均信頼度   │ 0.81  │                       │
└─────────────┴───────┴───────────────────────┘
```

## API から実行

```bash
curl -X POST http://localhost:8000/maturation/review \
  -H "Content-Type: application/json" \
  -d '{"limit": 50, "concurrency": 5}'
```

レスポンス:

```json
{
  "reviewed": 50,
  "approved": 38,
  "rejected": 12,
  "avg_quality": 0.74,
  "avg_confidence": 0.81
}
```

## DifficultyTagger との連携

審査完了後、承認されたドキュメントに難易度タグを付与します:

| タグ | 説明 | Student 学習順序 |
|------|------|----------------|
| `beginner` | 基礎概念・簡単な実装 | 1番目 |
| `intermediate` | 標準的な実装・応用 | 2番目 |
| `advanced` | 複雑なアルゴリズム・最適化 | 3番目 |
| `expert` | 研究レベル・システム設計 | 4番目 |

Student はこの順序でカリキュラムを提示されます。
