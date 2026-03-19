# plan_training_a.md — MED 訓練システム詳細設計

作成日: 2026-03-19
参照元: `src/training/`, `src/memory/maturation/`, `CLAUDE.md`

---

## 1. 全体像：何を誰に教えるか

```
「知識はFAISSメモリに蓄積する。
 小モデルには RLで検索・利用方法だけ教える」
                         — TinyLoRA論文応用
```

| 役割 | モデル | やること |
|------|--------|---------|
| **Teacher** | Claude / GPT-4o | 訓練データを生成・採点する |
| **Student** | Qwen2.5-7B | FAISSメモリを使いながら回答することを学習する |
| **FAISS** | 外部メモリ | 知識を格納。Studentが検索して使う |

Studentが学ぶのは「新しい知識」ではなく「FAISSをうまく引いて回答を組み立てるスキル」。

---

## 2. 訓練データの作り方（2経路）

### 経路A — Teacher が自動生成（`SeedBuilder`）

```
SeedBuilder.build(topic, domain, n_samples)
    │
    ├─ Teacher LLM に「技術文書を書いて」と依頼
    │   ・難易度: beginner / intermediate / advanced / expert
    │   ・形式:  document（200〜400語）or qa（Q&A ペア）
    │
    └─ 生成物を Document として FAISS + SQLite に保存
       (review_status = APPROVED: Teacher生成なので即採用)
```

**生成プロンプト例（document）:**
```
Topic: Python FAISS vector search
Difficulty: intermediate
Domain: code

Generate a technical document:
```

**難易度分布（デフォルト、n=20の場合）:**
```
beginner:     8件   ←少し多め
intermediate: 4件
advanced:     4件
expert:       4件
```

---

### 経路B — 実運用ログから蓄積（自動）

Orchestratorが処理したリクエストが自動的に `TrainingBatch` の素材になる:

```
ユーザークエリ
    │
    ▼
Student が FAISS を引いて回答
    │
    ├─ metadata に記録: {
    │    doc_ids: ["..."],          # 使ったFAISSドキュメント
    │    context_scores: [0.87],    # FAISSのスコア
    │    exec_success: True,        # コード実行が成功したか
    │    used_context: True,
    │  }
    │
    └─ (prompt, response, metadata) → TrainingBatch に変換
```

---

## 3. 訓練データの単位：`TrainingBatch`

```python
@dataclass
class TrainingBatch:
    prompts:   list[str]    # 入力クエリ
    responses: list[str]    # Studentが生成した回答
    rewards:   list[float]  # ← GRPO実行時に後から設定される（事前固定ではない）
    metadata:  list[dict]   # FAISSスコア・実行結果など
```

`rewards` は**学習ループの中で動的に計算**される点が重要。
事前にラベル付けした「正解データ」は不要。

---

## 4. 報酬関数（`CompositeReward`）

Studentの出力品質を5つの観点で採点し、加重合計する:

```
composite = 0.35 × correctness        ← Teacher LLM が 0〜1 で採点
          + 0.20 × retrieval_quality   ← FAISSスコアの平均
          + 0.20 × exec_success        ← Dockerでコードが通ったか（0 or 1）
          + 0.10 × efficiency          ← 応答の長さが適切か（20〜2000字が理想）
          + 0.15 × memory_utilization  ← FAISSを1〜3件使えたか
```

**correctness の採点プロンプト（Teacher に渡す）:**
```
Question: {prompt[:300]}

Answer: {response[:500]}

Rate answer correctness 0.0-1.0. Reply with only a number.
```

---

## 5. 訓練アルゴリズム：GRPO

**ロス計算式:**
```
L_GRPO = -mean( (r_i - r_group_mean) / (r_group_std + ε) × log π(a_i|s_i) )
```

**グループ内相対化の意味:**
- バッチ内の平均報酬より良い回答 → 確率を上げる
- バッチ内の平均報酬より悪い回答 → 確率を下げる
- KLペナルティ不要 → 小モデル（7B）でも安定

**設定値（デフォルト）:**
```python
epsilon = 1e-8       # 正規化安定項
clip_ratio = None    # PPOスタイルclippingは無効（オプション）
entropy_coef = 0.01  # 探索促進ボーナス
```

---

## 6. アダプタ：TinyLoRA

学習可能なのは **B行列のみ**（frozen_rank × projection_dim = 2×4 = **8パラメータ**）。

```
入力 x → [Aで圧縮（凍結）] → [Bで射影（学習）] × scale → 元の出力に加算
       (hidden_dim × frozen_rank)  (frozen_rank × projection_dim)
```

```python
TinyLoRAAdapter(
    hidden_dim=4096,     # Qwen2.5-7B の隠れ次元
    frozen_rank=2,       # A の列数（凍結）
    projection_dim=4,    # B の列数（学習可能）
    tie_factor=7,        # 同じ (A,B) を最大7層で共有
    alpha=1.0,
)
# → trainable_params ≈ 2×4 = 8
```

保存サイズ: `frozen_rank × projection_dim × 4 bytes` ≈ **32バイト** (〜1KB with metadata)

---

## 7. 3段階パイプライン（`TrainingPipeline`）

```
data_source（AsyncIterator[TrainingBatch] or list）
    │
    ▼
Stage 1: SFT ウォームアップ（100 steps, lr=5e-5）
    │  SFTアルゴリズムで基礎的な回答能力を付ける
    │  → 損失: 教師あり cross-entropy
    │
    ▼
Stage 2: GRPO + TinyLoRA 本学習（900 steps, lr=1e-4）
    │  各ステップで:
    │  1. CompositeReward.compute_batch(batch) → rewards を設定
    │  2. GRPOAlgorithm.compute_loss(batch, model, adapter)
    │  3. AdamW.step()
    │  4. 200 steps ごとにチェックポイント保存
    │
    ▼
Stage 3: 評価（最終バッチで eval_reward を計算）
    │
    ▼
TrainingResult（total_steps, final_loss, best_reward, checkpoint_path）
```

---

## 8. 評価フレームワーク（既存実装）

### 8-1. `BenchmarkSuite` — Q&Aペアでのバッチ評価

```python
# 組み込みベンチマークデータセット
"qa_retrieval"   : [("What is FAISS?", "..."), ("Explain TinyLoRA", "..."), ...]
"code_generation": [("Write FAISS index code", "..."), ...]
"math_reasoning" : [("Solve: 2x+3=7", "4"), ...]

# 実行
suite = BenchmarkSuite(student_evaluator)
report = await suite.run(student_model, benchmark_names=["qa_retrieval"])
# → BenchmarkReport(overall_score, per_benchmark_scores)
```

### 8-2. `StudentEvaluator` — 1サンプルごとの多次元評価

```python
# EvalSample ごとに以下を計算
{
    "retrieval_accuracy": ...,  # FAISSが正しい文書を引けたか
    "answer_quality":     ...,  # Teacher LLM が採点
    "exec_success_rate":  ...,  # コード実行成功率
    "avg_reward":         ...,  # CompositeRewardの平均
}
```

### 8-3. `TeacherComparison` — Teacher vs Student の比較

```python
# 同じクエリに対して Teacher と Student 両方を走らせて比較
result = await comparison.compare(query, student_answer, teacher_answer)
# ComparisonResult(
#     student_score=0.71,
#     teacher_score=0.93,
#     winner="teacher",
#     quality_gap=0.22,
#     feedback="Student lacks detail on...",
# )
```

---

## 9. 「面接形式」評価について（現状と不足点）

### 現状できること

| 機能 | 実装状況 | ファイル |
|------|---------|---------|
| Q&Aペアでバッチ評価 | ✅ 実装済み | `benchmark_suite.py` |
| Teacher が採点 | ✅ 実装済み | `teacher_eval.py` |
| Student vs Teacher 比較 | ✅ 実装済み | `teacher_comparison.py` |
| セッションID管理 | ✅ スキーマあり | `models.py` |

### 面接形式（多ターン・文脈継続）に不足しているもの

```
現状: Q1 → A1 → [採点] → Q2 → A2 → [採点]  ← 独立した1問1答
目標: Q1 → A1 → Q2（A1を踏まえた追加質問）→ A2 → ...  ← 面接
```

不足コンポーネント:
1. **会話履歴ストア** — セッションIDに紐づく (Q, A) 履歴の保持
2. **フォローアップ質問生成器** — 前の回答に基づいて次の質問をTeacherが作る
3. **文脈継続評価** — 前のターンと矛盾しないか・深化しているかの指標
4. **適応的難易度調整** — 正解率に応じて次の質問を難しくする

→ 設計詳細は `TODO.md` Section A-1 / 将来の `plan_training_b.md` で扱う

---

## 10. 実装状況サマリ

| レイヤー | 実装 | 状態 |
|---------|------|------|
| データ生成 | `SeedBuilder` (Teacher自動生成) | ✅ 動作可能（API key要） |
| データ構造 | `TrainingBatch` / `Document` | ✅ 完成 |
| 報酬計算 | `CompositeReward` (5指標) | ✅ 完成 |
| GRPOアルゴリズム | `GRPOAlgorithm` | ✅ 骨格完成、実モデル接続待ち |
| アダプタ | `TinyLoRAAdapter` (8パラメータ) | ✅ 完成 |
| 3段階パイプライン | `TrainingPipeline` | ✅ 骨格完成、実モデル接続待ち |
| バッチローダー | データソース → TrainingBatch変換 | ⬜ 未実装（最重要ボトルネック） |
| 実Studentモデル接続 | Qwen2.5-7B + vLLM | ⬜ 未実装 |
| 評価 | `BenchmarkSuite` / `StudentEvaluator` / `TeacherComparison` | ✅ 完成 |
| 面接形式評価 | 多ターン・文脈継続 | ⬜ 未設計 |

---

## 11. 今すぐ動かせる最小手順

```bash
# Step 1: FAISSにシードを投入（SeedBuilder、API key があれば即実行可能）
python scripts/seed_memory.py --topic "Python async" --domain code --n 50

# Step 2: 成熟済みドキュメントをバッチ化（要実装: バッチローダー）
# → memory_manager.search() の結果を TrainingBatch に変換するスクリプトが必要

# Step 3: パイプライン実行（Studentモデルが必要）
python scripts/train_student.py --config configs/training.yaml

# Step 4: 評価
python scripts/evaluate_student.py --benchmarks qa_retrieval code_generation
```

**現在のボトルネック2点:**
1. `Step 2` のバッチローダー未実装
2. `Step 3` での実Studentモデル（Qwen2.5-7B）ロード未実装
