# 品質メトリクス

Phase 2 の進捗状況を数値で追跡します。

## Phase 2 目標値

| メトリクス | 目標 | 意味 |
|-----------|------|------|
| `total_docs` | ≥ 10,000 | FAISS に登録されたドキュメント総数 |
| `avg_confidence` | ≥ 0.7 | Teacher 審査時の平均信頼度スコア |
| `exec_success_rate` | ≥ 0.8 | コードドキュメントの実行成功率 |

## GUI での確認

**🧠 FAISSメモリ → 🔬 成熟管理 (Phase 2) → 品質レポート** タブ:

```
## Phase 2 進捗

ドキュメント数   ██████░░░░  6,000 / 10,000
平均信頼度      ████████░░  0.74 / 0.70  ✅
実行成功率      ███████░░░  70% / 80%

## Phase 2 目標達成状況
目標達成: ❌ (未達成: total_docs, exec_success_rate)

## 審査状況
未審査:  1,200 docs
承認:    4,500 docs  (75%)
却下:    300 docs    (5%)

## 難易度分布
beginner:     ████░░░░░░  40%
intermediate: ████░░░░░░  35%
advanced:     ██░░░░░░░░  20%
expert:       █░░░░░░░░░  5%
```

## API レスポンス形式

`GET /maturation/quality`:

```json
{
  "total_docs": 6000,
  "approved_docs": 4500,
  "rejected_docs": 300,
  "unreviewed_docs": 1200,
  "avg_confidence": 0.74,
  "avg_quality_score": 0.71,
  "exec_success_rate": 0.70,
  "phase2_progress": {
    "total_docs": { "current": 6000, "target": 10000, "pct": 60.0 },
    "avg_confidence": { "current": 0.74, "target": 0.7, "met": true },
    "exec_success_rate": { "current": 0.70, "target": 0.80, "met": false }
  },
  "meets_phase2_goal": false,
  "difficulty_distribution": {
    "beginner": 1800,
    "intermediate": 1575,
    "advanced": 900,
    "expert": 225
  }
}
```

## ドメイン別フィルタ

ドロップダウンから特定ドメインを選択すると、そのドメインのみの品質レポートを表示します:

```bash
# API でドメイン指定
curl "http://localhost:8000/maturation/quality?domain=python"
```

## Phase 2 達成後

`meets_phase2_goal: true` になると、Phase 3 の学習フレームワークを本格稼働できます:

```
Phase 2 完了 (10,000 docs / confidence≥0.7 / exec_success≥80%)
    ↓
Phase 3: GRPO + TinyLoRA 本番学習
    - 成熟したメモリで Student を訓練
    - KG パスを Teacher プロンプトに含めて CoT 強化
    - 評価指標に Entity 精度・関係再現率を追加
```
