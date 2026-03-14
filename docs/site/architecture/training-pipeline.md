# 学習パイプライン

Teacher-Student 二層構造による強化学習パイプラインの詳細です。

## 全体フロー

```
① データ準備
   Teacher 応答 + FAISS 成熟ドキュメントで訓練セットを構築

② SFT ウォームアップ (src/training/algorithms/sft.py)
   Student が基本的な応答形式を学ぶ

③ GRPO + TinyLoRA (src/training/algorithms/grpo.py)
   Student が FAISS 検索・活用スキルを RL で習得

④ 評価 (src/training/evaluation/)
   ベンチマークで Teacher との差を測定
```

## GRPO の動作

```python
# grpo.py より概念
for batch in training_data:
    # G 個の応答を生成
    responses = [student.generate(query) for _ in range(G)]

    # 各応答の報酬を計算
    rewards = [composite_reward(r, query) for r in responses]

    # グループ相対報酬 (GRPO の核心)
    baseline = mean(rewards)
    advantages = [r - baseline for r in rewards]

    # TinyLoRA アダプタを更新
    loss = -mean(log_prob * advantage for each response)
    optimizer.step(loss)
```

## TinyLoRA の構造

```
Frozen Pretrained Weights (Qwen-7B)
    ↓ (凍結)
TinyLoRA Adapter (~13 パラメータ)
    - frozen_rank: 2
    - projection_dim: 4
    - tie_factor: 7
    ↓ (学習対象)
Output
```

全パラメータ数の 0.0001% 以下を更新するだけで、
検索・活用スキルを習得できます（TinyLoRA 論文より）。

## Reward 関数の詳細

```python
# composite.py より
class CompositeReward:
    weights = {
        "correctness":        0.35,
        "retrieval_quality":  0.20,
        "exec_success":       0.20,
        "memory_utilization": 0.15,
        "efficiency":         0.10,
    }

    async def compute(self, response, query, context) -> float:
        scores = {
            "correctness":        await self._correctness(response, query),
            "retrieval_quality":  await self._retrieval_quality(context),
            "exec_success":       await self._exec_success(response),
            "memory_utilization": self._memory_utilization(context),
            "efficiency":         self._efficiency(response),
        }
        return sum(self.weights[k] * v for k, v in scores.items())
```

## 評価指標

| 指標 | 説明 |
|------|------|
| `student_vs_teacher` | Student と Teacher の回答品質比 |
| `retrieval_precision` | FAISS 検索の適合率 |
| `gsm8k_accuracy` | GSM8K 数学ベンチマーク |
| `code_pass_rate` | コード実行成功率 |
| `latency_p50/p99` | 応答レイテンシ |

```bash
# ベンチマーク実行
python scripts/evaluate_student.py \
    --adapter data/adapters/tinylora_latest.pt \
    --benchmark gsm8k,code_eval,med_internal
```

## Phase 3 KG 統合予定

Phase 3 では学習パイプラインに KG を統合します:

- KG パスを Teacher プロンプトの CoT に含める
- 訓練データ生成時に KG 根拠をアノテーション
- GRPO 報酬関数に KG 整合性スコアを追加
- 評価指標に Entity 精度・関係再現率を追加
