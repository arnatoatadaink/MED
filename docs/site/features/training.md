# 学習フレームワーク

Teacher が成熟させた FAISS メモリを使って、Student モデルに「メモリの使い方」を RL で学習させます。

## 設計思想

!!! quote "TinyLoRA 論文より"
    「知識は既にある。使い方だけ RL で教える」

MED では**知識を FAISS 外部メモリに蓄積**し、Student には「検索・活用スキル」だけを TinyLoRA アダプタで学習させます。

## 学習パイプライン（3段階）

```
① SFT ウォームアップ
   Teacher の応答を教師データとして Student に教師あり学習
        ↓
② GRPO + TinyLoRA
   Student が FAISS を検索 → 回答生成 → Reward 計算 → アダプタ更新
        ↓
③ 評価
   ベンチマークで Teacher との差を測定
```

## TinyLoRA 設定

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `frozen_rank` | 2 | 凍結するランク数 |
| `projection_dim` | 4 | 射影次元 |
| `tie_factor` | 7 | 重み共有因子 |
| アダプタサイズ | ~1 KB | ファイルサイズ |

`data/adapters/` に保存され、チャットタブの Student モードで自動的に使用されます。

## Reward 関数

| 信号 | 重み | 説明 |
|------|------|------|
| `correctness` | 0.35 | 回答の正確性 |
| `retrieval_quality` | 0.20 | FAISS 検索の適切さ |
| `exec_success` | 0.20 | コード実行成功率 |
| `memory_utilization` | 0.15 | メモリの活用度 |
| `efficiency` | 0.10 | レイテンシ・トークン効率 |

## 対応アルゴリズム / アダプタ

=== "アルゴリズム"

    | アルゴリズム | 用途 |
    |------------|------|
    | **GRPO** *(デフォルト)* | Group Relative Policy Optimization |
    | **PPO** | Proximal Policy Optimization |
    | **DPO** | Direct Preference Optimization |
    | **SFT** | Supervised Fine-Tuning（ウォームアップ用）|
    | **REINFORCE** | 基本的な Policy Gradient |

=== "アダプタ"

    | アダプタ | パラメータ数 | 用途 |
    |---------|------------|------|
    | **TinyLoRA** *(デフォルト)* | ~13 | 極少パラメータ RL |
    | **LoRA** | ~数千 | 標準的なファインチューニング |
    | **LoRA-XS** | ~数百 | TinyLoRA と LoRA の中間 |
    | **Full FT** | 全パラメータ | フルファインチューニング |

## GUI での操作

1. **🎓 学習** タブを開く
2. アルゴリズムとアダプタを選択
3. 「**開始**」ボタンをクリック
4. Loss / Reward のリアルタイムグラフを確認
5. 「**停止**」ボタンでアダプタを保存して終了

## コマンドラインから実行

```bash
# 学習を開始
python scripts/train_student.py \
    --algorithm grpo \
    --adapter tinylora \
    --epochs 10 \
    --domain python

# 評価
python scripts/evaluate_student.py \
    --adapter data/adapters/tinylora_latest.pt \
    --benchmark gsm8k
```

## ログ

Weights & Biases (W&B) に学習ログを送ります:

```bash
export WANDB_API_KEY=your-key
python scripts/train_student.py --wandb-project MED-training
```
