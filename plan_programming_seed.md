# plan_programming_seed.md — プログラミング・論理空間拡張 Seed 計画

## 目的

現在の FAISS メモリ（approx 4,000 docs）は arXiv 論文主体で「研究知識」に偏っている。
本計画では以下 2 軸を拡充し、Student モデルの連想検索品質を高める。

| 軸 | 目標 | 主な効果 |
|----|------|---------|
| **言語空間拡張** | ライブラリ・標準 API の使用パターンを蓄積 | 「何が使えるか」の引き出しを増やす |
| **論理空間拡張** | 設計・構造・手法の「なぜ/どうする」を蓄積 | 問題→解法のマッピングを強化する |

---

## カテゴリ別計画

### A. Python 標準ライブラリ（言語空間）

**目的**: Python の built-in と stdlib の使用パターン・適切な選択基準を記憶させる

| 優先度 | モジュール群 | 代表トピック |
|-------|------------|------------|
| 高 | `itertools`, `functools`, `collections` | 高階関数・データ構造イディオム |
| 高 | `asyncio`, `concurrent.futures` | 非同期・並列処理パターン |
| 高 | `dataclasses`, `typing`, `abc` | 型システム・抽象化パターン |
| 中 | `pathlib`, `os`, `sys`, `subprocess` | ファイル・プロセス操作 |
| 中 | `json`, `csv`, `sqlite3`, `pickle` | データ永続化 |
| 中 | `logging`, `unittest`, `pytest` | 品質管理パターン |
| 低 | `re`, `textwrap`, `string` | テキスト処理 |

**収集形式**: チュートリアル説明＋コード例＋「なぜこのモジュールを使うか」の説明

---

### B. ML/AI ライブラリ（言語空間）

**目的**: PyTorch・numpy 等のコア API パターンを記憶させる

| 優先度 | ライブラリ | 対象トピック |
|-------|----------|------------|
| 高 | `torch` | Tensor 操作・autograd・nn.Module・DataLoader |
| 高 | `numpy` | ブロードキャスト・einsum・ベクトル化操作 |
| 高 | `scikit-learn` | Pipeline・Estimator API・CV パターン |
| 高 | `pydantic` | バリデーション・型強制・モデル設計 |
| 中 | `transformers` | Pipeline API・tokenizer・model.generate() |
| 中 | `faiss` | Index 操作・検索・シリアライズパターン |
| 中 | `aiosqlite`, `httpx`, `fastapi` | 非同期 I/O パターン |
| 低 | `pandas` | DataFrame 操作・グループ集計 |

---

### C. アルゴリズム設計パターン（論理空間の核）

**目的**: 問題クラス → 解法クラスのマッピングを記憶させる。UML と相補的に機能する。

| カテゴリ | 具体的な内容 | 論理空間への寄与 |
|---------|------------|---------------|
| **探索・グラフ** | BFS/DFS/Dijkstra/A*、ユースケース比較 | 「状態空間の探索戦略」の位置情報 |
| **動的計画法** | メモ化・ボトムアップ・状態定義の方法 | 「最適部分構造の見つけ方」 |
| **分割統治** | 再帰分解・マスター定理の直感 | 「問題を小さくする方法」 |
| **ヒューリスティック** | 貪欲法が有効な条件・反例パターン | 「近似解の選択基準」 |
| **計算量分析** | Big-O の空間的直感・実測との乖離 | 「コストの位置づけ」 |
| **データ構造選択** | ハッシュ vs ツリー vs リスト の選択基準 | 「操作コストの空間的配置」 |

---

### D. GoF デザインパターン（言語×論理の橋渡し）

**目的**: 「このパターンを使う理由（Why）→ 実装の形（How）→ Python での慣習」の三層構造を蓄積する

UML のクラス図と直結するため、本カテゴリが言語空間と論理空間の橋渡し役として最も重要。

| 分類 | パターン | 特に重要な「Why」 |
|-----|---------|----------------|
| **生成** | Factory, Abstract Factory, Builder, Singleton | オブジェクト生成の責務分離 |
| **構造** | Adapter, Decorator, Proxy, Composite, Facade | インターフェース変換・透過的拡張 |
| **振舞** | Strategy, Observer, Command, Template Method, Iterator | アルゴリズムの交換可能性・疎結合 |
| **Python 慣習** | Protocol vs ABC、duck typing vs explicit interface | 静的型との比較 |
| **アンチパターン** | God Object、Spaghetti、Magic Number | 「何をしてはいけないか」の逆マッピング |

---

### E. UML・ソフトウェア設計（論理空間のマクロ構造）

**目的**: 設計の「なぜこの構造か」の位置情報をマッピングする

| 図の種類 | 論理空間での役割 |
|---------|--------------|
| **クラス図** | 静的構造・関係性（継承・集約・依存）の空間配置 |
| **シーケンス図** | 時系列の相互作用・責務の流れ |
| **状態機械図** | 状態遷移・不変条件の記述 |
| **アクティビティ図** | 制御フロー・並列処理の可視化 |
| **コンポーネント図** | モジュール境界・インターフェース定義 |

収集内容:
- 図の「読み方・書き方」の説明テキスト
- 具体的なシステム設計への適用例（Python/ML システム）
- 「この図をいつ使うか」の判断基準

---

### F. ソフトウェアアーキテクチャパターン（マクロ論理空間）

**目的**: システム全体の構造選択の「Why」を蓄積する

| パターン | Why を蓄積する観点 |
|---------|----------------|
| **SOLID 原則** | 変更容易性・依存関係の方向性 |
| **Hexagonal Architecture** | 外部依存の分離・テスト容易性 |
| **CQRS / Event Sourcing** | 読み書き分離・状態の追跡性 |
| **Clean Architecture** | 関心の分離・依存の向き |
| **Microservices vs Monolith** | スケーリング vs 複雑性のトレードオフ |
| **Martin Fowler Patterns** | Enterprise パターンの適用判断 |

---

### G. 型システム・形式的推論（論理×言語の橋渡し）

**目的**: Python の型ヒントの「なぜ」を支える背景知識を蓄積する

| トピック | 論理空間での役割 |
|---------|--------------|
| 型安全性の直感的説明 | 「コンパイル時 vs 実行時のエラー」の位置情報 |
| 代数的データ型（Optional, Union） | 「値の存在/非存在の表現」 |
| 型推論の仕組み | 型の「伝播」パターン |
| Protocol vs ABC | 構造的部分型 vs 名義的部分型 |
| Design by Contract | 前提条件・事後条件・不変条件 |

---

### H. 自然言語でのアルゴリズム説明（橋渡し特化）

**目的**: 「コード ↔ 論理 ↔ 自然言語」の三角形を閉じる

これが連想記憶の品質に最も直接的に寄与する。
コードを見て「なぜこう書くか」を自然言語で説明するドキュメント群。

| 形式 | 例 |
|-----|---|
| 疑似コード＋意図説明 | "This loop maintains an invariant that..." |
| Why コメント付きコード | "Using a deque here because..." |
| Before/After リファクタリング説明 | "Changed X to Y because..." |
| コードレビューコメント集 | 実際の設計判断の言語化 |

---

## 優先順位サマリー

```
Phase A（言語空間 優先）:
  1. GoFデザインパターン     ← 言語×論理の両方に効く
  2. アルゴリズム設計パターン ← 論理空間の核
  3. Python stdlib / ML libs ← 言語空間の充実

Phase B（論理空間 強化）:
  4. UML + アーキテクチャ    ← 論理空間のマクロ構造
  5. 型システム              ← 言語×論理の橋渡し

Phase C（深化）:
  6. 自然言語アルゴリズム説明 ← 橋渡し品質の精度向上
  7. アンチパターン集         ← 逆マッピング強化
```

---

## 収集量の目安

| カテゴリ | 見込みドキュメント数 | 優先度 |
|---------|------------------|-------|
| A. Python stdlib | 200〜400 | 高 |
| B. ML/AI ライブラリ | 300〜500 | 高 |
| C. アルゴリズム設計 | 150〜300 | 高 |
| D. GoFパターン | 100〜200 | 最高 |
| E. UML | 100〜200 | 中 |
| F. アーキテクチャ | 100〜200 | 中 |
| G. 型システム | 50〜100 | 中 |
| H. 自然言語説明 | 200〜400 | 中 |
| **合計** | **1,200〜2,300** | |

---

## 言語方針

- **全コンテンツ英語で統一**
- 収集クエリ・チャンク・レビュープロンプトすべて英語
- 日本語混入リスク: 低（arXiv/SO/ドキュメントサイトは英語主体）
- 現行 DB の日本語 6 件（manual 5 件 + tavily 1 件）は必要に応じて削除可能

---

## 取得先候補（次フェーズで決定）

| カテゴリ | 候補ソース |
|---------|----------|
| Python docs | docs.python.org, realpython.com, python-patterns.guide |
| ML libs | pytorch.org/docs, scikit-learn.org, numpy.org |
| GoF パターン | refactoring.guru, sourcemaking.com |
| アルゴリズム | cp-algorithms.com, algorithmist.com |
| UML | uml-diagrams.org, plantuml.com/guide |
| アーキテクチャ | martinfowler.com, patterns.dev |
| 型システム | mypy.readthedocs.io, peps.python.org |
| 自然言語説明 | Stack Overflow (高評価 answer), GitHub code review |

取得方法:
- **Tavily**: Web 検索（credits に注意）
- **arXiv**: 実装論文・パターン論文
- **SO**: 高評価 Q&A
- **直接 URL フェッチ**: docs サイトは `url_fetcher.py` で直接取得
- **手動 seed**: 厳選ドキュメントを manual ソースとして投入

---

## questions.txt への追加案

```
# === Python Standard Library Patterns ===
Python itertools and functools patterns for functional programming
Python asyncio event loop and task management patterns
Python dataclasses vs NamedTuple vs TypedDict: when to use which
Python collections module: Counter, deque, defaultdict use cases
Python typing module: Protocol, TypeVar, Generic patterns

# === Software Design Patterns ===
GoF Strategy pattern implementation in Python
Decorator pattern vs Python function decorators: design differences
Observer pattern and event-driven architecture in Python
Factory method vs Abstract Factory: when to use which
Python ABC vs Protocol: structural vs nominal subtyping

# === Algorithm Design ===
Dynamic programming: identifying optimal substructure in problems
Graph traversal BFS vs DFS: use case decision criteria
Greedy algorithm validity conditions and counterexamples
Divide and conquer algorithm design with Master theorem intuition
Hash map vs BST vs sorted list: time-space tradeoff analysis

# === Software Architecture ===
SOLID principles applied to Python class design
Hexagonal architecture: separating domain from infrastructure
Clean architecture dependency rule in Python projects
CQRS pattern: separating read and write models
Composition over inheritance: practical Python examples

# === UML and System Design ===
UML class diagram notation: association aggregation composition
Sequence diagram for async Python service interactions
State machine design for document processing pipelines
Component diagram for microservice dependency visualization
Activity diagram for parallel workflow representation
```
