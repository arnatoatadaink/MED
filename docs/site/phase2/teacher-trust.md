# Teacher 信頼度評価

Teacher モデルごとの信頼度を SQLite で管理し、検索スコアに自動反映させるシステムです。

## 仕組み

```
Teacher が回答生成
    ↓
FeedbackCollector がフィードバックイベントを蓄積
    ↓
TeacherFeedbackPipeline.flush() で TeacherRegistry を更新
    ↓
EWMA (指数加重移動平均) で trust_score を更新
    ↓
CompositeScorer が trust_score を composite_score に乗算
    ↓
信頼度の低い Teacher のドキュメントは検索結果で後退
```

## Trust Score の更新アルゴリズム

| フィードバック数 | 更新方式 | 理由 |
|--------------|---------|------|
| n ≤ 10 件 | **Welford 法**（オンライン真の平均） | 初期は偏りを抑える |
| n > 10 件 | **EWMA**（α = 0.05） | 最近のフィードバックを重視 |

```python
# EWMA 更新式
trust_score = (1 - α) * trust_score + α * reward
# α = 0.05: 直近 20 件のフィードバックが約 63% の重みを占める
```

**最小値**: 0.05（信頼度が低くても完全排除はしない）

## teacher_id の形式

```
{provider}:{model}
例: anthropic:claude-sonnet-4-20250514
    openai:gpt-4o
    ollama:llama3.1:8b
```

## データモデル

```python
# TeacherProfile (teacher_registry.py)
@dataclass
class TeacherProfile:
    teacher_id: str           # "anthropic:claude-sonnet-4-20250514"
    provider: str             # "anthropic"
    trust_score: float        # 0.05 ~ 1.0 (EWMA)
    total_docs: int           # 生成したドキュメント総数
    avg_reward: float         # 平均報酬
    n_feedback: int           # フィードバック受信回数
    updated_at: datetime
```

## CompositeScore への反映

```python
# composite_scorer.py より
composite_score = (
    usefulness_score * 0.5
    + freshness_score * 0.3
) * teacher_trust  # ← teacher_trust を最後に乗算
```

`teacher_trust` が 0.5 の Teacher のドキュメントは、
同等の usefulness / freshness を持つ信頼度 1.0 の Teacher のドキュメントと比較して
検索スコアが半分になります。

## FeedbackCollector

フィードバックは `FeedbackCollector` に非同期で蓄積され、
`TeacherFeedbackPipeline.flush()` を呼ぶまで DB には書き込まれません。

```python
# 使用例
collector = FeedbackCollector()
collector.record(
    teacher_id="anthropic:claude-sonnet-4-20250514",
    reward=0.85,
    doc_id="abc123",
    query="binary search implementation",
)

# バッチで DB に反映
pipeline = TeacherFeedbackPipeline(registry, collector)
await pipeline.flush()
```

## GUI での確認

**🧠 FAISSメモリ → 🔬 成熟管理 (Phase 2) → Teacher 信頼度** タブで
全 Teacher のプロファイルを Dataframe 形式で確認できます。

| 列 | 説明 |
|----|------|
| teacher_id | モデル識別子 |
| provider | プロバイダー名 |
| trust_score | 現在の信頼度 (0.05〜1.0) |
| total_docs | 生成ドキュメント数 |
| avg_reward | 平均報酬スコア |
| n_feedback | フィードバック受信回数 |
| updated_at | 最終更新日時 |
