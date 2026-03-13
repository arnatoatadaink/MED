# training.yaml リファレンス

`configs/training.yaml` の全フィールドの説明です。

## 設定例

```yaml
# デフォルトアルゴリズム
algorithm: "grpo"         # grpo / ppo / dpo / sft / reinforce

# デフォルトアダプタ
adapter: "tinylora"       # tinylora / lora / lora_xs / full_ft

# デフォルトReward関数
reward: "composite"       # composite / code_exec / teacher_eval / hybrid

# GRPO 設定
grpo:
  group_size: 8           # 1クエリに対して生成する応答数
  clip_ratio: 0.2         # PPO クリップ比率
  kl_coeff: 0.01          # KL 正則化係数
  learning_rate: 1e-5
  batch_size: 16
  max_epochs: 10

# TinyLoRA 設定
tinylora:
  frozen_rank: 2
  projection_dim: 4
  tie_factor: 7

# LoRA 設定
lora:
  rank: 16
  alpha: 32
  dropout: 0.1
  target_modules: ["q_proj", "v_proj"]

# Composite Reward 重み
composite_reward:
  correctness: 0.35
  retrieval_quality: 0.20
  exec_success: 0.20
  memory_utilization: 0.15
  efficiency: 0.10

# 学習データ
data:
  train_domain: "all"     # all / python / math / general
  min_difficulty: "beginner"
  max_difficulty: "expert"
  split_ratio: 0.9        # train/val 分割比

# 評価設定
evaluation:
  interval_steps: 100     # 評価実行間隔（ステップ数）
  benchmarks:
    - gsm8k
    - code_eval
    - med_internal
```

## アルゴリズム別パラメータ

=== "GRPO"
    ```yaml
    grpo:
      group_size: 8         # 推奨: 4〜16
      clip_ratio: 0.2       # PPO と同じ意味
      kl_coeff: 0.01        # 大きいと保守的
    ```

=== "PPO"
    ```yaml
    ppo:
      clip_ratio: 0.2
      value_coeff: 0.5
      entropy_coeff: 0.01
      gae_lambda: 0.95
    ```

=== "DPO"
    ```yaml
    dpo:
      beta: 0.1             # KL 正則化強度
      reference_free: false
    ```

=== "SFT"
    ```yaml
    sft:
      max_seq_length: 2048
      packing: true
    ```
