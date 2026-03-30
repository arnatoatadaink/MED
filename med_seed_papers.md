# med_seed_papers.md
> Created: 2026-03-26
> 対象システム: MED（Memory Environment Distillation）
> 出所: note.com/tender_peony902 の記事レビュー + 本セッションで調査した論文群

---

## セクション1: note.com論文レビュー（MED関連上位）

### S1: Context Engineering 2.0
- **arXiv**: 2510.26493
- **記事**: https://note.com/tender_peony902/n/n76c24bac6b98（2026-01-26）
- **MED関連度**: ★★★★★

**核心**:
- メタ認知AIシステムとして、思考ログ（reasoning_chain + confidence + self_evaluation）を構造化
- 4フェーズ学習サイクル: 実行 → 自己評価 → パターン抽出 → 適応
- 知識グラフを継続更新し、success_rate > 0.9 のパターンを自動汎化

**MEDへの接続（IDEA-001）**:
- reasoning_chain → TeacherのFAISS検索文脈に対応
- self_evaluation（accuracy/relevance/completeness）→ GRPOの報酬設計に直結
- パターン抽出 → KG（NetworkX）への自動登録
- 思考ログの永続化 → SQL構造化ノート

---

### S2: ICL is Provably Bayesian Inference
- **arXiv**: 2510.10981
- **記事**: https://note.com/tender_peony902/n/nd4d15933733b（2025-10-27）
- **MED関連度**: ★★★★★

**核心**:
- ICLをメタ学習として定式化。リスクを2成分に分解:
  - Bayes Gap: pretrainingスケール（Teacher依存）で改善
  - Posterior Variance: コンテキスト例数kで指数的減少 O(e^{-ck})

**MEDへの接続（IDEA-002）**:
- FAISS取得数 k の設計根拠: k=3〜5で実用的収束（理論値）
- Bayes GapはTeacher=Claude APIで既に小さい
- OOD安定性: Bayes Gapのみ影響を受けるため、未知ドメインでの耐性設計の指針に

---

### S3: Propose, Solve, Verify（self-play）
- **arXiv**: 2512.18160
- **記事**: https://note.com/tender_peony902/n/n8eeb8ac86c5b（2026-01-14）
- **MED関連度**: ★★★★

**核心**:
- PSVループ: Propose（問題生成）→ Solve（解答）→ Verify（形式的検証）
- 形式的検証器が合否判定し、合格データのみ学習に採用
- pass@1で最大9.6倍の性能改善

**MEDへの接続（IDEA-003）**:
- Propose → TeacherによるStudentカリキュラム自動生成
- Solve → StudentのGRPO推論
- Verify → GRPOの報酬関数（形式的検証器として）
- 難易度適応: Student成功率に応じて問題を動的生成

---

### S4: Agent0（Self-Evolving from Zero Data）
- **arXiv**: 2511.16043
- **記事**: https://note.com/tender_peony902/n/n33ce5d16d9f4（2025-11-25）
- **MED関連度**: ★★★★

**核心**:
- 外部データなしで2エージェント（カリキュラム・エグゼキューター）を競わせる
- ツール統合型推論（Python等）で難問を解く
- 数学的推論18%・一般推論24%向上

**MEDへの接続（IDEA-003と統合）**:
- カリキュラム・エージェント → Teacher（問題生成役）
- エグゼキューター・エージェント → Student（推論実行役）
- ゼロデータ自己進化 → naiveなStudentからの立ち上げと一致

---

## セクション2: LLM不確実性・曖昧さ研究

### A1: "We're afraid language models aren't modeling ambiguity"
- **arXiv**: 2304.14399 (Liu et al., EMNLP 2023)
- **MED関連度**: IDEA-008（曖昧さ認識RAG）の根拠

**要点**:
- GPT-4でも曖昧な入力に対して単一解釈に強制収束
- 曖昧さの解消成功率はプロンプトの言い回しで大きく変動

---

### A2: "Ambiguity in LLMs is a concept missing problem"
- **arXiv**: 2505.11679 (Hu et al., 2025)
- **MED関連度**: IDEA-008の根拠

**要点**:
- 曖昧さはLLMの潜在空間における「概念の欠損」から生じる
- Sparse Autoencoderで欠損概念を補うと曖昧さが検出可能になる

---

### A3: "Can Large Language Models Faithfully Express Their Uncertainty?"
- **会議**: EMNLP 2024
- **MED関連度**: IDEA-008の根拠

**要点**:
- 標準デコードでは断定的な回答が生成される
- 不確実性プロンプトを与えても確信度と言語表現が乖離

---

### A4: "Do LLMs Estimate Uncertainty Well?" (ICLR 2025)
- **MED関連度**: IDEA-008の根拠

**要点**:
- 事実UQ手法は指示追従タスクの不確実性に不適合
- 指示追従では事実の正確さより「応答が指示に従っているか」が問題

---

## セクション3: 損失関数・学習安定化研究

### L1: Gemma 2 Technical Report（Google DeepMind, 2024）
- **MED関連度**: IDEA-009（Chance-Level Threshold）の接続先

**要点**:
- Attention logitsを50.0でcap、final logitsを30.0でcap
- Soft cap: logit / cap → tanh → × cap（なめらかな制限）
- Gemma 3ではsoft-capping → QK-normに置き換え

---

### L2: "Methods of Improving LLM Training Stability" (Rybakov et al., 2024)
- **arXiv**: 2410.16682
- **MED関連度**: IDEA-009の接続先

**要点**:
- QK LayerNorm + Softmax Cappingの組み合わせで学習率1.5倍向上
- 発散時はQKV・Proj・FC2層のL2ノルムが2倍以上成長

---

### L3: Focal Loss（Lin et al., ICCV 2017）
- **MED関連度**: IDEA-009のSoft版への移行根拠

**要点**:
- 高確信サンプルの損失を縮小（完全ゼロにはしない）
- Chance-Focal Soft版の設計基盤: weight = max(0, 1-p*n_classes)^γ

---

## セクション4: 階層分類・ハイブリッド推論

### H1: IN-DEDUCTIVE（LSHTC3優勝システム, Rubin & Watanabe 2016）
- **MED関連度**: IDEA-010（IN-DEDUCTIVEハイブリッド）の根拠

**要点**:
- 帰納的学習（ボトムアップ: 葉→根）× 演繹的分類（トップダウン: 根→葉）
- LSHTC3チャレンジで他システムを上回った

---

### H2: MoE（Mixture of Experts: Shazeer et al., 2017）
- **MED関連度**: IDEA-010の参照アーキテクチャ

**要点**:
- Gate（ルーター）= 演繹的（どのエキスパートを使うか）
- Experts（並列）= 帰納的（各エキスパートが独立判断）

---

### H3: DID Framework（De-In-Ductive, ACL 2025）
- **MED関連度**: IDEA-010の理論的背景

**要点**:
- 問題構造に応じて演繹・帰納を動的に切り替え
- IO→CoT→ToT→DIDの進化

---

## セクション5: 著者帰属・スタイル抽出研究

### ST1: Nini (2023) "A Theory of Linguistic Individuality for Authorship Analysis"
- **出版**: Cambridge University Press
- **med_hyp_style_g.md の根拠**

**要点**:
- 人の書き方スタイルは一貫したスタイルモデルを形成する
- LLMはこのモデルに合致するテキストを生成することで真のパーソナライゼーションに近づける

---

### ST2: Huang et al. (2024) "Can Large Language Models Identify Authorship?"
- **med_hyp_style_g.md の根拠**

**要点**:
- T5 EncoderでAuthorship Signatureを学習
- LLMが少量サンプルからスタイル識別できることを実証

---

### ST3: EMNLP 2025 "LLMs Still Struggle to Imitate Implicit Writing Styles"
- **arXiv**: 2509.14543
- **med_hyp_style_g.md の根拠**

**要点**:
- LLMは明示的な指示なしには個人の暗黙的なスタイルを模倣困難
- 逆に言えば個人語彙は抽出可能なシグナルが存在する証明

---

## セクション6: Hyperbolic Embedding（embedding研究）

### HE1: geoopt（Kochurov et al., 2020）
- **arXiv**: 2005.02819
- **GitHub**: https://github.com/geoopt/geoopt
- **MEDへの接続**: 仮説B（KGエッジCPPN）のHyperbolic実装基盤

**要点**:
- PyTorchネイティブのリーマン幾何最適化ライブラリ
- Poincaréボール・Lorentz（双曲面）・積多様体をサポート
- `pip install geoopt` で即利用可能
- float64推奨（数値安定性）

---

### HE2: Multi-Relational Hyperbolic Word Embeddings（EACL 2024）
- **arXiv**: 2305.07303
- **GitHub**: https://github.com/neuro-symbolic-ai/multi_relational_hyperbolic_word_embeddings
- **MEDへの接続**: KGエッジ生成の実装参考

**要点**:
- 自然言語定義から双曲空間のword embeddingを学習
- 概念の包含関係・差異化関係を幾何的制約で保持

---

### HE3: HypStructure（NeurIPS 2024）
- **arXiv**: 2412.01023
- **MEDへの接続**: 階層ラベル構造をembeddingに組み込む正則化

**要点**:
- 双曲空間を使った構造的正則化（HypStructure）
- 既存のtask lossと組み合わせ可能なシンプルな実装

---

## IDEAとの対応表

| IDEA | 根拠論文 |
|------|---------|
| IDEA-001（思考ログ） | S1（Context Engineering 2.0） |
| IDEA-002（FAISS k値） | S2（ICL is Bayesian Inference） |
| IDEA-003（カリキュラム生成） | S3（PSV self-play）+ S4（Agent0） |
| IDEA-004（GRPO報酬） | S1 + S3 |
| IDEA-005（KG自動更新） | S1 |
| IDEA-008（曖昧さ認識RAG） | A1, A2, A3, A4 |
| IDEA-009（Chance-Level Threshold） | L1, L2, L3 + 個人実験 |
| IDEA-010（IN-DEDUCTIVE） | H1, H2, H3 |
| med_hyp_style_g.md（仮説G） | ST1, ST2, ST3 |
| 仮説B（KGエッジCPPN） | HE1, HE2 |

---

## Update Log

| Date | Note |
|------|------|
| 2026-03-26 | Initial: note.com記事レビューからの論文群を整理 |
| 2026-03-26 | セクション3-6: 本セッション中に調査した論文群を追加 |
