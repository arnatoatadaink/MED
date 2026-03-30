# med_hyp_style_g.md
> Status: sketch | Created: 2026-03-26
> 対象システム: MED（Memory Environment Distillation）
> 関連: trident_hyp_neat.md 仮説F（言語×NEAT）の MED側担当部分

---

## 概要

言語の「表層的スタイル」を3層に分解し、それぞれに特化したモデルを接続するシステム。
個人レベルの特徴（口調・語彙）と言語レベルの特徴を分離して解析・利用する。

---

## 設計の概要

```
入力テキスト（日本語）
    │
    ├─→ 形態素解析（SudachiPy等）
    │        ↓
    │   [層3モデル] 言語構造ベクトル ← 言語レベルの特徴
    │
    ├─→ 語彙特徴量抽出（StyloMetrix / LIWC-J）
    │        ↓
    │   [層2モデル] 個人語彙ベクトル ← 個人の特徴（語彙）
    │
    └─→ トーン特徴量抽出（文末・感情表現）
             ↓
        [層1モデル] 個人口調ベクトル ← 個人の特徴（口調）

             ↓ 層1+2を統合
        [個人スタイルベクトル（StyleVector.personal）]
             ↓
        ← FAISSのcontext_embとして活用（IDEA-E）
        ← KGのノード属性として登録（IDEA-005）
        ← GRPOのスタイル一貫性報酬に使用（IDEA-004）
```

---

## 層1: 個人レベル・口調（Tone）
> **Status: sketch**

### 抽出対象

```
文末表現パターン: 〜ですね / 〜だよ / 〜じゃん / 〜っす
感情表現の頻度・種類
文の長さ分布
質問・命令・感嘆の比率
句読点・絵文字の使用傾向
```

### 接続する研究

```
Stylometry (StyloMetrix, 2025):
  語彙・文法・句読点パターンを人間設計の特徴量として抽出
  LightGBMで分類 → 解釈可能かつ高速
  
「Better Call Claude」(ACL 2025):
  最新LLM（GPT-4o / Claude）は文センベルでのスタイル変化を検出できる
  → LLM-as-judgeでのトーン評価が実用的
```

### 実装候補

```python
# StyloMetrix（Python）
from stylo_metrix import StyloMetrix
sm = StyloMetrix("ja")  # 日本語対応確認が必要
metrics = sm.transform([text])
tone_features = metrics[["punct_ratio", "sentence_length_mean", ...]]
```

---

## 層2: 個人レベル・語彙（Vocabulary）
> **Status: sketch**

### 抽出対象

```
語彙の豊富さ（TTR: Type-Token Ratio）
専門語彙・固有名詞の分布
慣用句・口語表現の使用頻度
同義語選択パターン（「使う」vs「利用する」vs「活用する」）
LIWC的な心理カテゴリ分類
```

### 接続する研究

```
Nini (2023) "A Theory of Linguistic Individuality for Authorship Analysis":
  人の書き方スタイルは一貫したスタイルモデルを形成する
  → 個人語彙は抽出可能なシグナルとして存在することの理論的根拠

Huang et al. (2024) "Can Large Language Models Identify Authorship?":
  T5 EncoderでAuthorship Signatureを学習
  LLMが少量サンプルからスタイル識別できることを実証

「LLMs Still Struggle to Imitate Implicit Writing Styles」(EMNLP 2025):
  LLMは明示的な指示なしには個人語彙スタイルを模倣困難
  → 逆に個人語彙は抽出可能なシグナルが存在する
```

### 実装候補

```python
# LIWC-J（日本語版LIWCの有無要確認）
# 代替: 辞書ベースで心理カテゴリ分類

def extract_vocab_features(text, morphemes):
    tokens = [m.surface() for m in morphemes]
    types = set(tokens)
    ttr = len(types) / len(tokens)  # Type-Token Ratio
    
    # 専門語彙比率（技術辞書との照合）
    # 慣用句検出（慣用句辞書との照合）
    return {"ttr": ttr, ...}
```

---

## 層3: 言語レベル・形態素構造
> **Status: sketch（ツールが揃っている）**

### 抽出対象

```
形態素の品詞分布（名詞率・動詞率・形容詞率）
敬語レベル（素体 / 丁寧体 / 尊敬語 / 謙譲語）
接続詞パターン
助詞の使用傾向
述語のアスペクト（完了・継続・習慣）
```

### 日本語形態素解析ツール（現在利用可能）

```
SudachiPy（推奨・最も柔軟）:
  複数粒度の分割に対応（短単位・中単位・長単位）
  pip install sudachipy sudachidict-full
  
GiNZA（依存関係解析が必要な場合）:
  spaCyベースで依存構造まで取得可能
  pip install ginza ja_ginza
  
MeCab + NEologd（高速処理が最優先なら）:
  定番だが更新が遅い、新語は NEologd で補完

spaCy ja_core_news（Transformerベース）:
  精度重視の場合
```

```python
# SudachiPyの例
import sudachipy
import sudachidict_full

tokenizer = sudachipy.Dictionary().create()
morphemes = tokenizer.tokenize(text)

pos_dist = {}
for m in morphemes:
    pos = m.part_of_speech()[0]  # 品詞
    pos_dist[pos] = pos_dist.get(pos, 0) + 1

# 敬語検出
keigo_level = detect_keigo_level(morphemes)
```

---

## StyleVector の設計

```python
@dataclass
class StyleVector:
    personal: np.ndarray   # 層1+2統合（個人の「声」）次元: 64-128次元想定
    language: np.ndarray   # 層3（言語構造）次元: 32次元想定
    metadata: dict         # {"language": "ja", "register": "casual", ...}
    
    def to_context_emb(self) -> np.ndarray:
        """FAISSのcontext_embとして使用できる形式に変換"""
        return self.personal  # 層3は別途KGで管理
```

---

## MEDへの統合

### FAISS（仮説E / IDEA-010と統合）

```python
# context_emb として StyleVector.personal を活用
results = searcher.search(
    query_emb=query,
    context_emb=style_vector.personal,  # ← ここに個人スタイルを注入
    k=5
)
# → 「この人らしい検索」が可能になる
```

### KG（仮説B / IDEA-005と統合）

```python
# StyleVectorをKGノード属性として登録
kg.add_node(
    node_id=doc_id,
    embedding=doc_emb,
    style=style_vector,  # ← ノード属性として追加
)
# → 同じスタイルの記憶同士をエッジで接続
```

### GRPO報酬（IDEA-004と統合）

```python
def compute_reward_with_style(output, reference, style_target):
    # 既存の精度・関連性スコア
    base_reward = compute_base_reward(output, reference)
    
    # スタイル一貫性スコア（追加）
    output_style = style_extractor.extract(output)
    style_consistency = cosine(
        output_style.personal,
        style_target.personal
    )
    
    # 合算
    return 0.7 * base_reward + 0.3 * style_consistency
```

---

## 評価指標

```
層1（口調）: 同一著者の別テキスト間での口調ベクトル類似度
層2（語彙）: Authorship Attributionタスクでの精度
層3（言語）: 敬語レベル分類精度

統合評価: スタイル逸脱検出（PostSaver的なユースケース）
  → 「自分らしくない投稿」を検出できるか
  → StyleVector.personal の距離でアラート
```

---

## 実装ギャップ

- [ ] SudachiPy動作確認: `pip install sudachipy sudachidict-full`
- [ ] StyloMetrix日本語対応確認: https://github.com/ZILiAT-NASK/StyloMetrix
- [ ] LIWC-J（日本語版）の有無を確認
- [ ] 層1+2の次元設計（64次元 or 128次元）
- [ ] StyleVectorのFAISS互換形式への変換

---

## 参照すべきツール・研究

```
ツール:
  SudachiPy: https://github.com/WorksApplications/SudachiPy
  GiNZA: https://github.com/megagonlabs/ginza
  StyloMetrix: https://github.com/ZILiAT-NASK/StyloMetrix

研究:
  Nini (2023) "A Theory of Linguistic Individuality for Authorship Analysis"
    Cambridge University Press
  Huang et al. (2024) "Can Large Language Models Identify Authorship?"
  EMNLP 2025 "LLMs Still Struggle to Imitate Implicit Writing Styles"
  ACL 2025 "Better Call Claude: Can LLMs Detect Changes of Writing Style?"
```

---

## Update Log

| Date | Note |
|------|------|
| 2026-03-26 | hyp_h_neat.md 仮説Gから独立、med_hyp_style_g.md として作成 |
| 2026-03-26 | SudachiPy・GiNZA・StyloMetrixの実装情報を追加 |
| 2026-03-26 | GRPO報酬統合案（スタイル一貫性スコア）を追加 |
