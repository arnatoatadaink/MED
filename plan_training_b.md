# plan_training_b.md — 面接形式テスト・訓練 全体方針

作成日: 2026-03-19
参照元: セッション研究調査（LLM-as-an-Interviewer, IQA-EVAL, MultiChallenge, mtRAG, RAGEN, REFUEL, CURIO）

---

## 全体構造：3層アーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│  評価層（テスト）                                          │
│  LLM-as-an-Interviewer / IQA-EVAL / MultiChallenge /    │
│  AssumptionCorrection / mtRAG                           │
├─────────────────────────────────────────────────────────┤
│  訓練層（アルゴリズム選択）                                │
│  GRPO/StarPO → StarPO-S → REFUEL → CURIO               │
├─────────────────────────────────────────────────────────┤
│  データ品質層（MED固有）                                   │
│  Teacher品質ゲーティング / 分散フィルタ /                  │
│  KGカバレッジ監視 / Cross-Encoder疑似報酬                 │
└─────────────────────────────────────────────────────────┘
```

---

## 1. 評価層（テスト）

### 1-1. LLM-as-an-Interviewer（圧迫深掘りテスト）
- **出典**: arXiv:2412.10424（2024年12月）
- **方針**: テストとして全面取り込み
- **MEDでの使い方**:
  - Teacherがインタビュアーとして「なぜそう思ったの？」を繰り返す
  - 回答の一貫性・深化能力・矛盾への対応を評価
  - 最終的に「インタビューレポート」を自動生成
- **測定する軸**:
  - 最初の回答品質
  - フィードバックへの適応力（答えを深化できるか）
  - フォローアップへの対応力

```
Q1 → A1 → [なぜ？] → A2 → [具体的には？] → A3 → [矛盾してない？] → A4
                     └─ 前の回答を踏まえた追加質問（コンテキスト継続）
```

---

### 1-2. IQA-EVAL（ペルソナ別評価自動化）
- **出典**: NeurIPS 2024 (arXiv:2408.13545)
- **方針**: テストとして取り込み + テストデータ生成用途にも活用
- **位置づけの整理**:

| 用途 | 内容 | MEDでの対応 |
|------|------|------------|
| 評価自動化 | 人間評価者の代替 | Teacherがペルソナ付きLEAとして評価 |
| テストデータ生成 | 複雑・曖昧な質問を自動生成 | FAISSシードデータ生成に活用 |
| QA品質保持 | リリース前の回帰テスト | CI/CDの一部として組み込み |

- **ペルソナ例**: 初心者ユーザー / 専門家ユーザー / 懐疑的なユーザー
- **LLM-as-an-Interviewerとの違い**:
  - IQA-EVAL: ユーザー側をシミュレート → モデルの回答を評価（適応力を測る）
  - LLM-as-an-Interviewer: インタビュアー側として深掘り（深化能力を測る）

---

### 1-3. MultiChallenge（長期指示維持テスト）
- **出典**: arXiv:2501.17399（2025年）
- **方針**: テストとして取り込み
- **LLM-as-an-Interviewerとの違い**:

```
LLM-as-an-Interviewer  ← 縦方向
  「前の回答への一貫性」を深掘りで試す

MultiChallenge          ← 横方向
  「前のターンの指示・文脈を何ターン後まで守れるか」を試す
```

- **4カテゴリ**:

| カテゴリ | 内容 |
|----------|------|
| instruction retention | 複数ターン前の指示を守り続けられるか |
| inference memory | ユーザー情報を正しく推論・記憶できるか |
| reliable versioned editing | 文書を複数回修正しても整合性を保てるか |
| self-coherence | 自分の前の発言と矛盾しないか |

- **参考値**: Claude 3.5 Sonnet でさえ正解率 41.4%（静的ベンチでは高得点でも多ターンで大幅低下）

---

### 1-4. AssumptionCorrection（MEDオリジナル・前提訂正テスト）
- **方針**: 新規追加推奨（既存研究のどれとも被らない）
- **背景**: 既存の3研究は「前提の変質を能動的に訂正するか」を測っていない
- **測定内容**:
  1. ユーザーが間違った前提で質問する
  2. モデルが前提の誤りに気づいて指摘するか
  3. 指摘後に正しい前提で回答できるか
- **関連**: メタ認知検証（自分が何を知っていて何を知らないか）を挟むと効果的
  - 応答中に「この前提で合っていますか？」確認ステップを訓練データに意図的に含める

```python
# テストパターン例
AssumptionCorrectionTest:
  入力: 「PythonのGILはマルチコアを活用できますよね？」（誤前提）
  期待: 誤りを指摘 → 正しい前提で説明
  不合格: 誤前提をそのまま受け入れて回答
```

---

### 1-5. mtRAG（RAG×多ターン精度テスト）
- **出典**: MIT Press TACL（2025年）
- **方針**: テストとして取り込み
- **データ**: 110会話 × 平均7.7ターン = 842タスク（4ドメイン）
- **評価観点**:
  - 後半ターンで検索精度が落ちるか（MEDのFAISS繰り返し検索でも同様の劣化リスク）
  - 回答不可能な質問を正しく断れるか
  - 前のターンに依存する質問（非独立質問）を処理できるか

---

## 2. 訓練層（アルゴリズム選択）

### 2-1. GRPO / StarPO（デフォルト）
- **位置づけ**: 既存実装（`src/training/algorithms/grpo.py`）を多ターン化したもの
- **GRPO ≒ 並列検証プロンプト**（認識の整理）:
  ```
  推論時: 複数回答を並列生成 → 最良を選ぶ（並列検証プロンプト）
  訓練時: 複数回答の相対評価でモデルの重みを更新（GRPO）
  ← 同じ発想を推論と訓練で使い分けている
  ```
- **StarPO の追加要素**:
  - ステップ単位でなく軌跡全体（複数ターン分）をまとめてGRPO最適化
  - `<think>推論</think><ans>行動</ans>` の思考過程トークンを含む

---

### 2-2. StarPO-S（GRPO崩壊対策・強化版）
- **出典**: RAGEN arXiv:2504.20073（2025年）
- **方針**: 選択可能な訓練方法として追加。GRPO崩壊（Echo Trap）検出時のフォールバック
- **Echo Trap ≈ エコーチャンバー現象**:

| | SNSのエコーチャンバー | GRPOのEcho Trap |
|--|--|--|
| フィードバック源 | 「いいね」・フォロワー | 報酬関数 |
| 強化されるもの | 共感を得やすい意見パターン | 報酬を得やすい文体パターン |
| 失われるもの | 異なる視点 | 多様な推論戦略 |
| 検出 | 徐々に起きる | reward variance の崖で急に現れる |

- **StarPO-S の3つの介入**:

```
① 分散ベース軌跡フィルタリング
   高分散バッチ（良い回答も悪い回答もある）→ 学習効率が高い → 採用
   低分散バッチ（全部良い or 全部悪い）    → 学習効率が低い → 除外
   実績: 上位25〜50%の高分散ロールアウトのみで安定性・性能が大幅改善

② Criticの組み込み（PPOスタイル）
   純粋なGRPO（Critic不要）より安定。崩壊リスクが高い場面で有効

③ 非対称クリッピング + KL制約緩和
   良い方向への更新を強め / 悪化方向を抑制
   KL制約を外して探索を促進（DAPO手法の転用）
```

---

### 2-3. REFUEL（長い多ターン会話特化）
- **出典**: arXiv:2410.04612（2024年）
- **方針**: 選択可能な訓練方法として追加。形式が独自で他手法と被らない
- **核心的な仕組み**:
  ```
  通常のActor-Critic: Actor + Critic = 2モデル必要

  REFUEL: Q値の差分 Q(s,a₁) - Q(s,a₂) を1モデルで回帰
          → 同一プレフィックスから2つの完了系列を生成して差分を計算
          → モデル1つで済む・多ターン特化・逐次更新対応
  ```
- **手法比較**:

| 手法 | 多ターン | Critic不要 | 特徴 |
|------|---------|-----------|------|
| DPO | ✗ | ✓ | シンプルだが多ターン不向き |
| PPO | ✓ | ✗ | 安定だがコスト高 |
| GRPO | ✓ | ✓ | 並列サンプリング依存 |
| **REFUEL** | **✓** | **✓** | **多ターン特化・軽量・逐次更新** |
| StarPO | ✓ | 選択可 | 軌跡全体最適化 |

- **特に強い場面**: 会話が長くなるほどDPO/REBELに対して優位性が増す
- **実績**: Llama-3-8B を REFUEL で訓練 → Llama-3.1-70B（9倍大）を長い多ターン会話で上回った

---

### 2-4. CURIO（情報利得報酬・好奇心駆動）
- **出典**: arXiv:2504.03206（2025年）
- **方針**: 選択可能な訓練方法として追加。`memory_utilization` 報酬の代替/補完として有力
- **仕組み**:
  ```
  報酬 = 外的報酬（正解） + 内発的報酬（ユーザーのことをより理解できたか）

  「次の質問でユーザーについて何か新しいことが分かるような質問をしろ」
  というインセンティブ → 自然な深掘り質問が生まれる
  ```
- **MEDでの対応**:
  - CompositeRewardの `memory_utilization(0.15)` を部分的にCURIOの情報利得報酬に置き換える
  - FAISSの検索結果から「新たに得た情報量」を内発的報酬として計算する設計が可能

---

## 3. データ品質層（MED固有・差別化ポイント）

既存研究にはなく、Teacher-Student構造を持つMEDだからこそ実装できる層。

### 3-1. Teacher品質ゲーティング（訓練前フィルタ）

```python
class TrainingDataGate:
    """
    StarPO-S の分散フィルタ（訓練中・動的）と組み合わせる。

    Teacher品質ゲート（MED固有）: 訓練前にTeacherが精査（静的）
    分散フィルタ（StarPO-S）:     訓練中にリアルタイム判断（動的）
    → 両者を組み合わせると「良質かつ学習効率の高い」バッチを確保できる
    """
    def gate(self, batch) -> bool:
        teacher_quality = self.teacher.review(batch)   # reviewer.py 活用（既実装）
        reward_variance = self.calc_variance(batch)
        # Teacherが品質を保証した上で、高分散なものだけを訓練に通す
        return teacher_quality > θ_q and reward_variance > θ_v
```

- **既存実装との対応**: `src/memory/maturation/reviewer.py` を流用

---

### 3-2. 難易度カーリキュラム動的調整（訓練中フィルタ）

```
現在のMED: difficulty_tagger.py で事前タグ付け（静的カーリキュラム）

拡張案:
  Teacherが訓練中の損失推移を監視
  損失が低すぎる（簡単すぎる） → 難しいデータを追加
  損失が高すぎる（難しすぎる） → 中間難易度を増やす
  → リアルタイムのCurriculum Learning
```

- **既存実装との対応**: `src/memory/maturation/difficulty_tagger.py` の動的版

---

### 3-3. KGカバレッジ監視（Echo Trap早期検出）

```
Echo Trapの通常の検出: reward variance で代用（StarPO-S）

MEDでの改善案:
  KGのエンティティカバレッジを多様性指標として使う
  「同じエンティティ経路ばかり使っている」= Echo Trap の早期シグナル
  → router_bridge.py が既にエンティティ追跡しているため実装コストが低い

多様性スコア = ユニークエンティティ数 / 総エンティティ参照数
この値が閾値以下になったら警告 → StarPO-SまたはREFUELに切り替えトリガー
```

- **既存実装との対応**: `src/knowledge_graph/router_bridge.py` を流用

---

### 3-4. Cross-Encoder疑似報酬（Teacher APIコスト削減）

```
現状の課題: CompositeRewardの correctness(0.35) は毎ステップTeacher API呼び出しが必要
            → コスト高・レイテンシ高

改善案:
  cross_encoder.py（既実装）を訓練ループの疑似報酬として使う

  Student回答 → Cross-Encoder（Teacher品質の代理）→ 疑似報酬
                                                   ↓
                              Teacher APIを全ステップで呼ばずに済む
                              定期的なTeacher確認（N stepに1回）で品質を補正

REFUEL組み合わせ案:
  Cross-Encoder(回答A, 回答B) → 優劣スコア → REFUEL的な差分更新
  （完全にTeacher不要の多ターン訓練ループが実現）
```

- **既存実装との対応**: `src/memory/learning/cross_encoder.py` を流用

---

## 4. 実装優先順位

### フェーズ区切りの方針

コンテキスト規模・デバッグ規模を抑えるため、各フェーズは独立してテスト可能な単位で区切る。

```
Phase B-1: データ品質層の基盤（最優先・他フェーズの前提）
  → TrainingDataGate + 分散フィルタ計算
  → 既存 reviewer.py + difficulty_tagger.py の流用のため実装コスト小

Phase B-2: 評価フレームワーク拡張（テスト追加）
  → LLM-as-an-Interviewer テストループ
  → MultiChallenge テストスイート
  → AssumptionCorrection テスト（MED固有）

Phase B-3: 訓練アルゴリズム拡張
  → StarPO-S の分散フィルタをGRPOに追加
  → REFUEL の実装（多ターン特化）
  → CURIO の報酬関数追加

Phase B-4: 統合・モニタリング
  → KGカバレッジ監視によるEcho Trap検出
  → Cross-Encoder疑似報酬によるTeacher APIコスト削減
  → IQA-EVAL によるペルソナ別自動評価
  → mtRAG ベンチマーク統合
```

---

## 5. 既存実装との対応表

| plan_training_b の新要素 | 活用する既存実装 | 新規実装の規模 |
|--------------------------|----------------|--------------|
| Teacher品質ゲーティング | `reviewer.py` | 小（ラッパー） |
| 難易度カーリキュラム動的調整 | `difficulty_tagger.py` | 中 |
| KGカバレッジ監視 | `router_bridge.py` | 小（監視フック追加） |
| Cross-Encoder疑似報酬 | `cross_encoder.py` | 中 |
| StarPO-S 分散フィルタ | `grpo.py`（骨格） | 中 |
| REFUEL | なし | 大（新規） |
| CURIO 情報利得報酬 | `composite.py` | 中 |
| LLM-as-an-Interviewer テスト | `teacher_comparison.py` | 中 |
| MultiChallenge テスト | `benchmark_suite.py` | 中 |
| AssumptionCorrection テスト | なし | 中（新規） |
| mtRAG テスト | `benchmark_suite.py` | 中 |
| IQA-EVAL テストデータ生成 | `seed_builder.py` | 中 |

---

## 6. 参照論文

| 研究 | 論文 | 用途 |
|------|------|------|
| LLM-as-an-Interviewer | arXiv:2412.10424 | 評価 |
| IQA-EVAL | arXiv:2408.13545 (NeurIPS 2024) | 評価・データ生成 |
| MultiChallenge | arXiv:2501.17399 | 評価 |
| mtRAG | MIT Press TACL 2025 | 評価 |
| RAGEN / StarPO-S | arXiv:2504.20073 | 訓練 |
| REFUEL | arXiv:2410.04612 | 訓練 |
| CURIO | arXiv:2504.03206 | 訓練（報酬） |

---

## 7. plan_training_a.md / plan_think.md との関係

```
plan_training_a.md  → 訓練システムの基盤設計
                       （GRPO・TinyLoRA・3段階パイプライン・CompositeReward）
                       本ファイルの訓練層はこれを拡張する

plan_think.md       → TeacherのReasoningTrace抽出・保存スキーム
                       本ファイルの Teacher品質ゲーティング・Cross-Encoder疑似報酬
                       と組み合わせると、思考過程を訓練データとしても活用できる

plan_training_b.md  → 面接形式テスト・多ターン訓練・データ品質層の追加方針
（本ファイル）         上記2ファイルを統合した plan_training.md の中間設計書
```
