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

---

## セクション7: 3/26〜4/26 新規論文（note.com更新分）

### S5: Transformer = Belief Propagation（構造同型）
- **arXiv**: 2603.17063
- **記事**: https://note.com/tender_peony902/n/n444d1f563d35（2026-04-04）
- **MED関連度**: ★★★★★

**核心**:
Transformerは確率的推論（Belief Propagation）と構造的に同一であることを証明。近似でもアナロジーでもなく数学的同型。

```
対応関係:
  Transformerのトークン変数 = ベイズネットのノード
  Attention          = メッセージ伝播（gather）
  FFN                = ベイズ更新（update）
  重み W             = 因子ポテンシャル
  層数 L             = 推論ステップ数

一意性定理:
  正確なベイズ推論をするTransformerは必ずBP構造になる
  → 学習 = BPへの収束過程

ハルシネーションの再定義:
  「LLMは間違っているのではなく、正誤が存在しない空間で動いている」
  grounded（検索で根拠あり）vs ungrounded（根拠未定義）の構造的分離
```

**MEDへの接続**:

| 接続先 | 内容 |
|--------|------|
| IDEA-008（曖昧さ認識RAG） | grounded/ungrounded分離の理論的根拠。FAISSで根拠が得られる場合（grounded）とそうでない場合（ungrounded）を明示的に区別する設計が正当化される |
| IDEA-010（IN-DEDUCTIVE） | 演繹パス = Attention（メッセージ伝播）、帰納パス = FFN（ベイズ更新）に対応。IN-DEDUCTIVEの2パス設計がTransformerの内部構造と同型 |
| 仮説D（推論トポロジー） | 「層数 = 推論深さ」の理論的根拠。TRIDENTのルーティングが推論深さを制御するという設計の基盤 |

**プロンプト設計への含意**:
```
「以下の情報のみを根拠として...」という grounding の明示が
BP理論的に正当化される → MEDのFAISS検索コンテキスト付与の設計原則
```

---

### S6: AGIロードマップ論文群（4/1まとめ）
- **記事**: https://note.com/tender_peony902/n/nef0b383e6a9d（2026-04-01）
- **収録論文**: Deep Research of Deep Research / COvolve / Observer-Solver / Medical AI Scientist
- **MED関連度**: ★★★★

**核心（COvolve）**:
LLMが「環境」と「ポリシー」の両方をPythonコードで自動生成し、2人零和ゲームで対立的に進化させるオープンエンド学習。ナッシュ均衡を使って過去の学習を忘れない「メタポリシー」を構築。

**核心（Observer/Solver分業）**:
マルチモーダルLLMの推論でObserver（観察者）が視覚証拠をキャプション化し、Solver（解決者）が回答する2役体制。報酬を観察精度と解答精度に分離することで学習が安定。

**MEDへの接続**:

| 接続先 | 内容 |
|--------|------|
| IDEA-003（カリキュラム生成） | COvolveの「環境自動生成＋難化」= Teacherが問題難易度を自動調整する設計と対応。Teacherが環境生成役、Studentがポリシー役 |
| IDEA-004（GRPO報酬） | Observer/Solver分業の報酬分離設計。「観察の正確さ」と「解答の正確さ」を別スコアにすることで報酬関数を細粒度化できる |
| TRIDENT仮説A | NEATの「最小構造から必要に応じて複雑化」がCOvolveの共進化ループと構造的に一致 |

---

### S7: RLTF（テキストフィードバックRL）
- **arXiv**: 2602.02482
- **記事**: https://note.com/tender_peony902/n/n4bb0dabd9535（2026-02-16）
- **MED関連度**: ★★★★

**核心**:
従来のRLHFは報酬が1ビット（良い/悪い）のみ。RLTFは「1回目の回答→テキストでダメ出し→2回目が改善」という改善能力を1回目に転写する。模範解答（高コスト）ではなくテキスト批評（低コスト）でスケールする学習。

```
従来RLHF:
  報酬 = 1ビット（良/悪）
  「なぜ間違えたか」が分からない → 学習効率が悪い

RLTF:
  報酬 = テキスト批評（「ここが・なぜ違うか」）
  改善能力を1回目に転写 → 学習効率が大幅向上
```

**MEDへの接続（IDEA-004）**:
```
現状のIDEA-004:
  reward = 0.5*accuracy + 0.3*relevance + 0.2*completeness

RLTF拡張案:
  Teacher が生成するテキスト批評自体を報酬信号として使う
  reward_text = teacher.critique(student_output)
  → GRPOの報酬関数にテキスト批評を組み込む

メリット:
  「正解/不正解」より高次元な学習信号
  Studentが「何がどう違うか」を学べる
  模範解答なしにスケール可能
```

---

---

## セクション9: セキュリティ・ロバスト性（新規カテゴリ）

### SC1: AI Agent Traps（Google DeepMind, 2026）
- **SSRN**: 6372438
- **PDF**: https://www.rivista.ai/wp-content/uploads/2026/04/ssrn-6372438.pdf
- **記事**: https://note.com/tender_peony902/n/n88158a8bf42c（2026-04-25）
- **MED関連度**: ★★★★★（RAG/KGへの直接的脅威）

**核心**:
自律AIエージェントがウェブを行動するとき、情報環境そのものが脆弱性になる。6種類のトラップを分類した最初の体系的フレームワーク。

```
6種類のトラップ分類:
  Content Injection   → 知覚層への攻撃（HTML/CSS隠蔽命令）
  Semantic Manipulation → 推論層への攻撃（バイアス・フレーミング）
  Cognitive State     → 記憶・学習層への攻撃（RAG汚染・潜在記憶汚染）
  Behavioural Control → 行動層への攻撃（隠れジェイルブレイク・データ窃取）
  Systemic            → マルチエージェント動態への攻撃（Sybil攻撃・連鎖崩壊）
  Human-in-the-Loop   → 人間監督者への攻撃（認知バイアス悪用）
```

**実験的エビデンス**:
- HTML aria-label等への敵対的命令注入で15〜29%のケースで生成要約が改変
- 人間作成のプロンプトインジェクションで最大86%のシナリオでエージェントを部分的にハイジャック
- 動的クローキング（エージェント検出→別コンテンツ配信）が実証されている

**MEDへの接続**:

| 攻撃種別 | 影響するIDEA/仮説 | 対策方針 |
|---------|----------------|---------|
| RAG Knowledge Poisoning | IDEA-005（KG自動更新） | 自動登録前に検証ゲートを追加。success_rate閾値だけでなくソース信頼度スコアを導入 |
| Latent Memory Poisoning | IDEA-005, 仮説B | KGエッジ生成時の出所追跡（Provenance）を記録 |
| Contextual Learning Traps | IDEA-003, IDEA-004 | few-shot demonstrationsの汚染検出。Teacher入力のサニタイズ層 |
| Content Injection | IDEA-008（曖昧さ認識RAG） | Webコンテンツ取込時のHTMLソース vs レンダリング差分検出 |
| Semantic Manipulation | IDEA-010（IN-DEDUCTIVE） | Teacher演繹パスの入力段階でフィルタリング。汚染されたTeacher判定が全下流を誤誘導するリスク |
| Oversight & Critic Evasion | IDEA-003（Verifier） | 教育的・仮説的・red-teaming的表現でラッピングされた入力への警戒 |

**ソース信頼度スコアの設計方針**:
```python
@dataclass
class SourceTrustScore:
    source_url: str
    domain_type: str        # "arxiv" | "github" | "web" | "user_input"
    provenance: str         # 出所の追跡チェーン
    sanitized: bool         # サニタイズ済みフラグ
    trust_score: float      # 0.0〜1.0
    
    # MEDのKG登録時にthought_logsと共に記録
    # trust_score < threshold → 自動登録をブロックしてQandA.mdに切り出す
```

**引用**:
- Franklin et al. (2026) "AI Agent Traps", Google DeepMind, SSRN 6372438
- Greshake et al. (2023) 間接プロンプトインジェクションの先駆け研究

---

## セクション8: 技術トレンドサマリー（3/26〜4/26）

| トレンド | 内容 | MEDとの関係 |
|---------|------|------------|
| Transformer理論の完成 | TransformerがBP（Belief Propagation）と構造同一であると証明（S5） | IDEA-008/010・仮説Dの理論的根拠が確立 |
| エージェントの自己進化 | COvolveに代表される「環境もポリシーも自動生成して共進化」が主流化 | TRIDENTのNEAT進化ループと同じ発想が学術的に追認 |
| 報酬信号の高次元化 | 1ビット（良/悪）→ テキスト批評（RLTF）→ 多次元スコアへ移行 | IDEA-004の報酬設計方向と一致。拡張設計の論拠 |
| 役割分担の優位性 | Observer/Solver、Teacher/Studentの分業が単一モデル全処理より優れると実証継続 | MEDのTeacher-Student設計の理論的追い風 |
| AI科学者エージェント | 「仮説→実験→論文」を自律的に回すエージェントが実用化フェーズへ | MEDのStudent自律学習の将来形として参照可能 |

---

## IDEAとの対応表（更新）

| IDEA | 根拠論文 |
|------|---------|
| IDEA-001（思考ログ） | S1（Context Engineering 2.0） |
| IDEA-002（FAISS k値） | S2（ICL is Bayesian Inference） |
| IDEA-003（カリキュラム生成） | S3（PSV self-play）+ S4（Agent0）+ S6（COvolve）|
| IDEA-004（GRPO報酬） | S1 + S3 + S7（RLTF） |
| IDEA-005（KG自動更新） | S1 + **SC1（RAG汚染・Provenance設計）** |
| IDEA-008（曖昧さ認識RAG） | A1, A2, A3, A4 + S5 + **SC1（Content Injection対策）** |
| IDEA-009（Chance-Level Threshold） | L1, L2, L3 + 個人実験 |
| IDEA-010（IN-DEDUCTIVE） | H1, H2, H3 + S5 + **SC1（Teacher入力サニタイズ）** |
| med_hyp_style_g.md（仮説G） | ST1, ST2, ST3 |
| 仮説B（KGエッジCPPN） | HE1, HE2 |
| 仮説D（推論トポロジー） | S5（層数=推論深さの理論的根拠） |

---

## Update Log

| Date | Note |
|------|------|
| 2026-03-26 | Initial: note.com記事レビューからの論文群を整理 |
| 2026-03-26 | セクション3-6: 本セッション中に調査した論文群を追加 |
| 2026-04-25 | セクション7-8: 3/26〜4/26の新規論文（S5/S6/S7）と技術トレンドサマリーを追加 |
| 2026-04-25 | セクション9: SC1（AI Agent Traps, Google DeepMind）をセキュリティカテゴリとして追加 |
