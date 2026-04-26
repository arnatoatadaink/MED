# med_enhancement_seed.md
> Status: active | Created: 2026-03-26 | Last Updated: 2026-03-26
> 対象システム: MED（Memory Environment Distillation）
> 論文出所: med_seed_papers.md 参照
> TRIDENT連携: trident_hyp_neat.md / trident_plan_hyp_e.md 参照

---

## Overview

MEDへの応用アイデアのシード集。
各IDEAは「理論的根拠 → MEDへの対応 → 実装ギャップ → 具体的タスク」の形式で記述。

---

## Source Articles（note.com）

| ID | 論文 | arXiv | MED関連度 |
|----|------|-------|----------|
| S1 | Context Engineering 2.0 | 2510.26493 | ★★★★★ |
| S2 | ICL is Provably Bayesian Inference | 2510.10981 | ★★★★★ |
| S3 | Propose, Solve, Verify (self-play) | 2512.18160 | ★★★★ |
| S4 | Agent0: Self-Evolving from Zero Data | 2511.16043 | ★★★★ |

詳細は `med_seed_papers.md` 参照。

---

## MED / TRIDENT 統合マップ

```
MED（記憶・学習・知識管理）担当:
  FAISS インデクサ（all-MiniLM-L6-v2 384次元）
  Hyperbolic投影層（KGエッジ生成時のみ、geoopt）
  IDEA-001〜010（記憶・検索・報酬系）
  med_hyp_style_g.md（スタイル抽出）

TRIDENT（トポロジー・ルーティング）担当:
  trident_hyp_neat.md（仮説A/B/C/D/F）
  trident_plan_hyp_e.md（CPPN連想関数の進化）
  TensorNEAT（GPU進化エンジン）
  Type A/B/C インターフェースコントラクト

境界が曖昧な部分:
  仮説B（KGエッジ）: MED→MLP版、TRIDENT→長期CPPN進化
  IN-DEDUCTIVE Gate（IDEA-010）: MED→Teacherパス、TRIDENT→ルーティング
  連想関数（IDEA-E）: MED→MLP実装、TRIDENT→将来NEAT進化
```

---

## Hyperbolic Embedding 利用方針

```python
# 現状維持: FAISSはEuclidean空間のまま
embeddings = sentence_model.encode(texts)  # (N, 384)
faiss_index.add(embeddings)

# Hyperbolic投影（KGエッジ生成時のみ追加）
import geoopt, torch
manifold = geoopt.PoincareBall(c=1.0)

def to_hyperbolic(emb: np.ndarray) -> torch.Tensor:
    t = torch.tensor(emb, dtype=torch.float64)  # float64推奨
    t = t / (t.norm() + 1e-8) * 0.9  # 境界内に収める
    return manifold.expmap0(t)

# KGエッジ重み（階層関係を保持）
def hyperbolic_edge_weight(emb_a, emb_b, context_emb):
    ha, hb, hc = to_hyperbolic(emb_a), to_hyperbolic(emb_b), to_hyperbolic(context_emb)
    base = 1.0 / (1.0 + manifold.dist(ha, hb))
    ctx  = 1.0 / (1.0 + manifold.dist(ha, hc)) + 1.0 / (1.0 + manifold.dist(hb, hc))
    return base + 0.3 * ctx
```

---

## Ideas

### IDEA-001: Structured Thought Log（思考ログ構造化）
- **Source**: S1 | **Priority**: A | **Status**: `draft` | **担当**: MED

```sql
CREATE TABLE thought_logs (
    id          TEXT PRIMARY KEY,
    timestamp   TEXT,
    input       TEXT,
    reasoning   JSON,   -- [{step, thought, confidence}]
    output      TEXT,
    reward      REAL,
    self_eval   JSON,   -- {accuracy, relevance, completeness, improvement_notes}
    pattern_id  TEXT    -- KGへの参照
);
```

**実装ギャップ**:
- [ ] self_evaluate() の出力をGRPO報酬に変換するパイプライン
- [ ] パターン抽出（success_rate > 0.9 → KG登録）ロジック

---

### IDEA-002: FAISS k-value Calibration（k値チューニング）
- **Source**: S2 + **S6（RLVR知見により重要度昇格）** | **Priority**: **A（昇格）** | **Status**: `draft` | **担当**: MED

Posterior Variance が O(e^{-ck}) で指数的減少。k=3〜5で実用的収束。

```
k < 3: Posterior Variance高 → タスク推定不安定
k = 3〜5: 実用的収束ゾーン（理論値）← 推奨
k > 7: コンテキスト消費増 / 収束余剰
```

**RLVR知見による重要度昇格（S6）**:
```
RLVRモデル（知識外付け・推論特化）を採用する場合、
FAISSのk値とコンテキスト品質がStudentの「見かけの賢さ」を決定する。
浅いStudentでも深いTransformer相当の出力が得られるかどうかは
k値と検索精度の実験で検証可能。

検証すべき仮説:
  「k=5 + Hyperbolic距離 + context_emb」の組み合わせで
  ベースモデルの深さを補完できるか
  → IDEA-002はMEDの中核仮説の検証実験になる
```

**実装ギャップ**:
- [ ] k値を config.yaml に外出し
- [ ] k=3/5/7/10での精度比較実験
- [ ] Observer（FAISS検索精度）とSolver（Student推論精度）を**独立評価**する実験設計

---

### IDEA-003: Teacher Curriculum Generator（カリキュラム自動生成）
- **Source**: S3 + S4 | **Priority**: B | **Status**: `draft`
- **担当**: MED（Teacherパス）/ TRIDENT（難易度判定）

```python
def generate_curriculum(student_success_rate: float) -> Problem:
    threshold = 1.0 / n_classes  # Chance-Level（IDEA-009と統合）
    if student_success_rate > 0.9:
        difficulty = "harder"
    elif student_success_rate < threshold:
        difficulty = "easier"
    else:
        difficulty = "frontier"
    return teacher.propose(difficulty=difficulty)
```

**実装ギャップ**:
- [ ] TeacherにCurriculumGeneratorモードを追加
- [ ] Verifier（ルールベースから開始）
- [ ] Student成功率のEMAトラッカー

#### 実装パターン参照: hantani記事（2026-04-01）`sketch`
> note.com/hantani/n/n9e1b0c170514
> 「Claude Codeで作ってCodex CLIでレビューする、AI駆動開発の全自動化をSkillで試した話」

記事のskill構造がIDEA-003の「カリキュラム生成→Verify→修正ループ」の実装例として機能している。

```
記事の構造 → IDEA-003への対応:

start-skill（親skill・状態を見て判断）
  ↕ 対応
Teacher（状態を見てカリキュラムを判断）

spec-to-design の1ドキュメントごとのループ:
  1. ドキュメント作成（Claude Code）
  2. Codex CLIで個別レビュー
  3. QandA.mdに不明点追記
  4. 修正してから次へ
  ↕ 対応
IDEA-003のVerifyループ:
  1. Studentが問題を解く
  2. Verifierで検証
  3. QandA.md的な「不明点ログ」に記録（IDEA-001の思考ログ）
  4. 修正・再学習してから次の問題へ

記事の知見:
  「全部まとめて作ってから最後にレビュー」を避ける
  → 1ドキュメント（1問題）ごとにVerifyを通す設計が重要
  → MEDでもStudent学習を「一括→評価」より「逐次→即Verify」で回す
```

---

### IDEA-004: GRPO Reward from Self-Evaluation（自己評価→報酬変換）
- **Source**: S1 + S3 + **S6（Observer/Solver）** + **S7（RLTF）** | **Priority**: B | **Status**: `draft` | **担当**: MED

```python
def compute_reward(output, reference, verifier_result, style_target=None):
    if not verifier_result.passed:
        return -1.0
    accuracy     = eval_accuracy(output, reference)     # 0〜1
    relevance    = eval_relevance(output, context)      # 0〜1
    completeness = eval_completeness(output, task)      # 0〜1
    base = 0.5 * accuracy + 0.3 * relevance + 0.2 * completeness
    if style_target is not None:
        style_score = cosine(style_extractor.extract(output).personal,
                             style_target.personal)
        return 0.7 * base + 0.3 * style_score
    return base
```

#### 拡張案A: Observer/Solver分離報酬（S6）

```python
def compute_reward_obs_solver(
    observation_accuracy: float,   # FAISSが正しい根拠を取得できたか
    solver_accuracy: float,        # Studentが根拠から正しく推論できたか
    verifier_result,
    w_obs=0.4, w_sol=0.6           # 重みは調整可能
) -> float:
    """
    Observer（FAISS検索精度）とSolver（Student推論精度）を独立評価。
    どちらの能力が不足しているかを分離して診断できる。
    
    observation_accuracy低 → FAISSのk値・Hyperbolic距離を改善（IDEA-002）
    solver_accuracy低      → StudentのRLVR訓練を強化（IDEA-003）
    """
    if not verifier_result.passed:
        return -1.0
    return w_obs * observation_accuracy + w_sol * solver_accuracy
```

#### 拡張案B: テキスト批評報酬（S7 RLTF）

```python
def compute_reward_rltf(
    output: str,
    teacher_critique: str,         # Teacherのテキスト批評
    second_attempt: str,           # 批評を受けた後の2回目の回答
    verifier_result
) -> float:
    """
    1ビット（良/悪）の代わりに、テキスト批評による高次元報酬。
    「どこが・なぜ違うか」をRLTFで転写する。
    
    実装順序:
      Phase 1: Teacher批評をログに記録（IDEA-001の拡張）
      Phase 2: 批評 → 改善率をスコア化
      Phase 3: 改善能力を1回目の回答に転写（本来のRLTF）
    """
    if not verifier_result.passed:
        return -1.0
    improvement = similarity(second_attempt, reference) \
                - similarity(output, reference)
    return max(0.0, improvement)   # 改善した分だけ報酬
```

**統合方針**:
```
Phase 2まで: 拡張案Aを先行実装（Observer/Solver分離）
Phase 3以降: 拡張案Bを追加（RLTF・より実装コストが高い）
現状の基本報酬関数は維持・拡張案はオプション
```

---

### IDEA-005: Knowledge Graph Auto-Update（KG自動更新）
- **Source**: S1 | **Priority**: B | **Status**: `draft` | **担当**: MED

```
自動更新トリガー:
  thought_logs.reward > 0.9 かつ 類似パターン未登録 → KGに新ノード追加
  
KGエッジ重み（Hyperbolic版）:
  edge_weight = 1.0 / (1.0 + manifold.dist(h_a, h_b))
```

---

### IDEA-006: Dynamic LoRA Rank Expansion（NEAT着想）
- **Source**: trident_hyp_neat.md 仮説A
- **Priority**: C | **Status**: `hypothesis` | **担当**: TRIDENT主 / MED従

詳細は `trident_hyp_neat.md 仮説A` 参照。

---

### IDEA-007: CPPN-inspired KG Edge Weighting（HyperNEAT着想）
- **Source**: trident_hyp_neat.md 仮説B
- **Priority**: C | **Status**: `hypothesis` | **担当**: TRIDENT（長期）/ MED（短期MLP）

短期MLP版: `trident_plan_hyp_e.md` 参照。

---

### IDEA-008: Ambiguity-Aware RAG（曖昧さ認識検索）
- **Source**: Liu 2023 + Hu 2025 + EMNLP 2024
- **Priority**: B | **Status**: `draft` | **担当**: MED

**失敗モードの3層構造**:
```
層1: 入力の曖昧さを「検出」できない
層2: 内部の不確信を「表現」できない
層3: 不確実性を「定量化」できない
```

```python
def ambiguity_aware_search(query_emb, context_emb, k=5):
    ambiguity = compute_semantic_entropy(query)
    threshold = 1.0 / n_groups  # Chance-Level（IDEA-009）
    
    if ambiguity > threshold:
        # 帰納パス: 複数解釈で並列検索
        interpretations = generate_interpretations(query)
        return merge_and_rerank(
            [faiss_search(i) for i in interpretations], k=k)
    else:
        # 演繹パス（IDEA-010と統合）
        return context_sensitive_search(query_emb, context_emb, k=k)
```

**確認すべき論文**:
- [ ] Kuhn et al. (2023) "Semantic Uncertainty" ICLR

---

### IDEA-009: Chance-Level Threshold（確率閾値サンプル選択）
- **Source**: 個人実験（視覚モデル）
- **Priority**: B | **Status**: `needs-redesign` | **担当**: MED

**実装の正確な記述（コードレビュー済み）**:
```python
def make_chance_clip_crossentropy(n_classes, scale=1.0):
    """
    threshold = scale / n_classes
    p >= threshold → loss = 0（高確信サンプルを除外）
    p <  threshold → 損失増幅（低確信サンプルに集中）
    
    元実験: 10クラス × scale=1.0 → threshold=0.1 = ランダム確率（偶然の一致）
    """
    threshold = scale / n_classes
    mul = tf.constant(1.0 / threshold, dtype=tf.float32)
    @tf.function
    def crossentropy(yTrue, yPred):
        return CategoricalCrossentropy()(
            yTrue, tf.clip_by_value(yPred * mul, 1e-10, 1.0))
    return crossentropy
```

**Soft版（Chance-Focal）**:
```python
weight = max(0, 1 - p * n_classes) ** gamma
# p = 1/n_classes でweight=0になる自然な境界
```

**一般化テーブル**:
| n_classes | scale=0.5 | scale=1.0 | scale=2.0 |
|-----------|-----------|-----------|-----------|
| 10 | 0.050 | **0.100** | 0.200 |
| 100 | 0.005 | 0.010 | 0.020 |

---

### IDEA-010: IN-DEDUCTIVE ハイブリッド推論
- **Source**: IN-DEDUCTIVE（LSHTC3）+ MoE + DID（ACL 2025）
- **Priority**: B | **Status**: `draft`
- **担当**: MED（Teacherパス）/ TRIDENT（ルーティング）

```python
def inductive_deductive_search(query_emb, context_emb, k=5):
    group_probs = teacher_classifier(query_emb)
    threshold = 1.0 / n_groups  # IDEA-009と統合
    
    if group_probs.max() >= threshold:
        # 演繹パス（MEDが担当）
        return faiss_search_in_group(query_emb, group_probs.argmax(), k)
    else:
        # 帰納パス（IDEA-008が担当）
        return ambiguity_aware_search(query_emb, context_emb, k*3)
```

**IDEA-008/009/010の統合**:
```
IDEA-009 → 切り替え閾値を設計
IDEA-010 → 確信度でパスを動的選択
IDEA-008 → 帰納パスでの具体的な検索戦略
```

#### 実装パターン参照: hantani記事（2026-04-01）`sketch`
> note.com/hantani/n/n9e1b0c170514

記事のstart-skill構造がIN-DEDUCTIVEの「演繹/帰納パス切り替え」の実装例として機能している。

```
記事の構造 → IDEA-010への対応:

start-skill の判定ルール:
  if レビュー依頼 or 大きなコード差分:
      → Codex CLI を直接実行（演繹パス: 明確なタスク）
  elif SPEC.mdあり、設計書不足:
      → spec-to-design を実行（帰納パス: 段階的な証拠収集）
  ↕ 対応
IDEA-010の切り替えロジック:
  if 確信度 >= threshold:
      → 演繹パス（Teacher絞り込み: 目的が明確な場合）
  else:
      → 帰納パス（IDEA-008曖昧さ認識RAG: 複数解釈が必要な場合）

記事の知見:
  「今は何をすべきか」をworkspace状態から判断する親役が必要
  → MEDでもTeacherが現在のStudent状態を見て
    「演繹（絞り込み）か帰納（並列探索）か」を判断する役割を持つ

  Codex CLI = 外部Verifier（作る役と見る役を分ける）
  → MEDの「Teacher（出題）× Student（解答）× Verifier（検証）」
    の3役分担と同じ構造
```

---

## Implementation Roadmap

```
Phase 0（現在）: FAISS + SQL + KG の基本構造

Phase 1: IDEA-001, IDEA-002 → 思考ログ確定、k値外出し
  + IDEA-002拡張: Observer/Solver独立評価の実験設計

Phase 2: IDEA-003, IDEA-004, IDEA-005 → カリキュラム・報酬・KG+Hyperbolic
  + IDEA-004拡張A: Observer/Solver分離報酬（S6）
  + IDEA-005拡張: SC1対応 trust_score + Provenance記録

Phase 3: IDEA-008, IDEA-009, IDEA-010 → 曖昧さ対応・IN-DEDUCTIVE
  + grounded/ungrounded分離（S5 BP理論）

Phase 4: TRIDENT連携 → trident_plan_hyp_e.md（CPPN連想関数）
  ← RLVR知見によりNEAT開始タイミングをPhase 2並行に前倒し検討
     「RLVRのフィットネス関数が確定したらNEATをすぐ始めてよい」

Phase 5: スタイル統合 → med_hyp_style_g.md（StyleExtractor）

Phase 6（追加）: RLTF統合 → IDEA-004拡張B（テキスト批評報酬）
```

---

## Open Questions

- Verifierを「ルールベース」→「LLM-as-judge」に移行するタイミングは？
- GRPO報酬重みの最適値（accuracy:0.5 / relevance:0.3 / completeness:0.2 は仮設定）
- Hyperbolicの float64計算が推論速度に与える影響
- scale（Chance-Level Threshold）の最適値をAblationでどう探索するか
- StyloMetrixの日本語対応状況（要確認）
- **Observer/Solver分離評価で「どちらが弱いか」が判明した場合の優先改善順序**
- **IDEA-002（k値）とStudent能力の相互作用: k値拡張で浅いモデルを補完できる上限は?**
- **NEAT開始のタイミング: Phase 2のRLVRフィットネス関数が安定したら即開始でよいか**

---

## Update Log

| Date | Note |
|------|------|
| 2026-03-26 | Initial draft from 4 note.com articles |
| 2026-03-26 | Added IDEA-008, IDEA-009（コードレビュー済み）, IDEA-010 |
| 2026-03-26 | 全体最新化: MED/TRIDENT分離マップ、Hyperbolic実装方針、Roadmap更新 |
| 2026-04-04 | IDEA-003/010にhantani記事の実装パターンをsketchとして追記 |
| 2026-04-26 | IDEA-002優先度昇格（RLVR知見）、IDEA-004にObserver/Solver分離報酬とRLTFを追加、RoadmapにNEAT前倒し検討を追記 |
