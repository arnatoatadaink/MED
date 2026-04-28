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

### I. JavaScript / TypeScript（言語空間拡張）

**目的**: ブラウザ・Node.js 環境の実装パターンと落とし穴を蓄積する

| 優先度 | トピック | 代表内容 |
|-------|---------|---------|
| 高 | イベントループ・非同期 | microtask/macrotask、Promise、async/await |
| 高 | モジュールシステム | ESM vs CJS、dynamic import、tree-shaking |
| 高 | TypeScript 型システム | generics、conditional types、satisfies |
| 中 | React/Vue パターン | hooks 最適化、状態管理、コンポーネント設計 |
| 中 | Node.js ランタイム | streams、worker_threads、child_process |
| 低 | ビルドツール | Webpack/Vite 設定、bundle 最適化 |

---

### J. HTML/CSS & Web 標準（言語空間拡張）

**目的**: Web プラットフォームの現代的な仕様と設計パターンを蓄積する

| 優先度 | トピック | 代表内容 |
|-------|---------|---------|
| 高 | CSS レイアウト | Grid vs Flexbox 選択基準 |
| 高 | Web パフォーマンス | Core Web Vitals、LCP/FID/CLS 最適化 |
| 高 | セキュリティヘッダー | CSP、CORS、XSS 防止パターン |
| 中 | Web Components | Shadow DOM、カスタム要素、slot |
| 中 | アクセシビリティ | ARIA ロール、キーボードナビゲーション |
| 低 | HTTP キャッシュ | Cache-Control、ETag 戦略 |

---

### K. サーバー・インフラ環境（運用知識）

**目的**: 実働環境のセットアップパターンと運用判断基準を蓄積する

| 優先度 | トピック | 代表内容 | 取得先 |
|-------|---------|---------|-------|
| 最高 | Linux コマンド・概念 | man-pages（man1/man2/man7） | GitHub API: mkerrisk/man-pages |
| 最高 | Linux 実践運用 | systemd、ネットワーク、権限、プロセス | Arch Wiki (URLリスト) |
| 高 | Docker | multi-stage build、Compose ネットワーキング | SO / GitHub |
| 高 | Python web サーバー | WSGI vs ASGI、gunicorn/uvicorn 設定 | SO / docs |
| 高 | nginx | reverse proxy、upstream、TLS 設定 | Arch Wiki / docs |
| 中 | データベース | PostgreSQL インデックス戦略、Redis パターン | SO / docs |
| 中 | CI/CD | GitHub Actions matrix、caching、artifact | GitHub docs |
| 低 | Linux From Scratch | OS構築から学ぶカーネル・ブート概念 | URLリスト |

**Linux 取得方針:**
- `mkerrisk/man-pages`: GitHub API で man1（コマンド）・man7（概念）を優先取得
- Arch Wiki: 運用頻度の高いページを `data/doc_urls/archwiki.txt` にキュレーション
- The Linux Command Line (linuxcommand.org): 教科書としてチャプター単位でURL列挙

---

### L. リリースノート・バージョン変化（保守知識）★

**目的**: バージョンアップによる破壊的変更・非推奨化・API 移行を蓄積する
→ 詳細設計は `plan_version_aware.md` を参照

| 優先度 | 対象 | 主な変更点 |
|-------|------|---------|
| 最高 | Python 3.10〜3.13 | match文、例外グループ、distutils削除、型改善 |
| 最高 | Pydantic v1→v2 | validator→field_validator、パフォーマンス刷新 |
| 高 | FastAPI 0.9x→0.10x | Pydantic v2対応、breaking changes |
| 高 | PyTorch 1.x→2.x | compile()、TorchDynamo、legacy API廃止 |
| 高 | React 16→18 | Concurrent Mode、自動バッチング、hooks |
| 中 | numpy 1.x→2.x | dtype変更、削除API |
| 中 | transformers | AutoModel APIの変遷、tokenizer変更 |
| 低 | SQLAlchemy 1.4→2.0 | query API廃止、2.0スタイル |

**収集形式**: 各バージョンの「何が壊れるか」「どう移行するか」を中心に収集

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

### M. ユーザー主観・技術ブログ（経験空間）

**目的**: 実際の学習者・実務者の「主観的体験」を蓄積し、認知空間に「人間の経験軸」を形成する

既存カテゴリとの違い:

| カテゴリ | 何を蓄積するか | 認知空間での役割 |
|---------|-------------|--------------|
| A〜L | 「何が正しいか / どう動くか」（事実・仕様） | 事実空間の軸 |
| **M** | 「どこで詰まったか / なぜこちらを選んだか」（経験・判断） | **経験空間の軸** |

Student モデルが「ユーザーがどこで詰まるか」「どの説明が刺さるか」を学習するためには、
仕様書には書かれない「主観的体験の記録」が必要。

| 優先度 | ソース形式 | 代表内容 | 取得先 |
|-------|---------|---------|-------|
| 最高 | 「1から勉強してみた」ブログ | 入門〜運用まで時系列で記録した学習日記 | dev.to, Zenn (英語記事), Medium |
| 最高 | チュートリアル完走記 | 公式チュートリアルを試した感想・詰まった箇所 | Hashnode, 長期運営の個人ブログ |
| 高 | 実務移行記 | 「本番で初めてXを使ったときのハマりどころ」 | dev.to, Medium/Towards Data Science |
| 高 | 比較・選定記 | 「AとBを比べてBを選んだ理由」（主観的判断の言語化） | dev.to, Medium |
| 中 | エラー解決記 | 「このエラーに3時間ハマった → 原因はXだった」 | 個人ブログ, dev.to |
| 中 | コードレビュー記録 | 「このPRでこう指摘された → なぜそれが正しいか」 | GitHub discussions |

**収集の質基準**:
- 単発記事より「シリーズ継続記事」（入門→中級→応用の同一著者記録）を優先
- タイトルに「[初心者向け]」「入門」「から学ぶ」が含まれるものを優先
- dev.to の「reactions 50以上」または Medium の「claps 200以上」の記事

---

#### SO の位置づけ（認知空間軸として再定義）

SO は「認知空間の均一な軸」にはならない（質問・回答の抜け漏れが多く、空間の島にならない）。
代わりに以下の用途に特化させる:

| 用途 | 使う | 使わない |
|-----|-----|--------|
| 認知空間の軸形成 | | ✗（抜け漏れが多く均一性に欠ける） |
| エラートレンド収集 | ✓ | |
| トピッククラスタ形成 | ✓（高頻度タグ → 関心の密度マップ） | |
| 特定技術の「詰まりやすい箇所」特定 | ✓（質問タイトルがそのまま詰まりポイント） | |

→ SO は「空間の軸」ではなく「信号」として使用する。
高頻度エラーパターンを seed することで、エラーメッセージ→原因→解決策のマッピング密度を高める。

---

#### GitHub Issues の位置づけ（エラートレンド軸）

```
対象リポジトリ（実際のユーザーエラー報告が集まるもの）:
  - pytorch/pytorch         — RuntimeError / CUDA OOM など
  - huggingface/transformers — モデルロード / tokenizer 挙動
  - facebookresearch/faiss  — IndexFlatIP / GPU インデックス
  - tiangolo/fastapi        — Pydantic バリデーション / CORS 問題

収集方針:
  - "is:issue is:closed label:bug" ＋ コメント数10以上（多くの人が遭遇）
  - 再現コード付きのissue優先（エラーメッセージ→原因→解決策のトリプレット）
  - "is:issue" タイトルに "How to" / "Error" / "failed" を含むもの

用途: エラーメッセージ → 原因 → 解決策のトリプレットとして seed
```

---

#### maturation 切り口（ユーザー主観コンテンツ専用）

通常コンテンツ（事実確認・正確性優先）とは異なるレビュー軸が必要:

| 評価軸 | 通常コンテンツ | ユーザー主観コンテンツ |
|-------|-------------|------------------|
| 正確性 | 事実として正しいか | 経験として真正か（嘘の体験談でないか） |
| 完全性 | 説明が網羅されているか | 詰まりポイントが具体的か |
| 最新性 | 現行仕様と一致するか | 判断の理由が語られているか |
| 難易度 | 技術的複雑さで評価 | 学習者から見た「心理的障壁」で評価 |

**実装案**: `source_extra` に `"content_type": "user_perspective"` を付与
→ Verifier は `needs_supplement` より `is_authentic_experience` を重視する variant で評価

---

## 優先順位サマリー

```
Phase A（言語空間 優先）:
  1. GoFデザインパターン        ← 言語×論理の両方に効く
  2. アルゴリズム設計パターン    ← 論理空間の核
  3. Python stdlib / ML libs   ← 言語空間の充実
  4. リリースノート / バージョン変化 ← 保守知識（plan_version_aware.md参照）

Phase B（論理空間 強化）:
  5. UML + アーキテクチャ       ← 論理空間のマクロ構造
  6. 型システム                 ← 言語×論理の橋渡し
  7. JavaScript / TypeScript   ← Web フロントエンド言語空間
  8. HTML/CSS & Web 標準        ← Web プラットフォーム知識
  9. サーバー・インフラ          ← 運用環境知識

Phase C（深化）:
  10. 自然言語アルゴリズム説明   ← 橋渡し品質の精度向上
  11. アンチパターン集           ← 逆マッピング強化
  12. ユーザー主観・技術ブログ   ← 経験空間の軸形成（SOはエラートレンド限定）
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
| I. JavaScript/TS | 200〜400 | 高 |
| J. HTML/CSS/Web | 100〜200 | 中 |
| K. サーバー/インフラ | 150〜300 | 高 |
| L. リリースノート | 200〜400 | 最高 |
| M. ユーザー主観・技術ブログ | 300〜600 | 高 |
| **合計** | **2,450〜4,800** | |

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
| JavaScript/TS | MDN Web Docs, javascript.info, TypeScript Handbook |
| HTML/CSS | MDN, web.dev (Google), CSS-Tricks |
| サーバー/インフラ | Docker docs, nginx.org, DigitalOcean tutorials |
| リリースノート | docs.python.org/whatsnew, github.com/*/CHANGELOG.md, PyPI release history |
| ユーザー主観 | dev.to (reactions≥50), Medium/TDS (claps≥200), Hashnode, 長期運営個人ブログ |
| エラートレンド（SO） | SO 高頻度タグ × エラーパターン（認知軸ではなくシグナル収集） |
| エラートレンド（Issues） | pytorch/pytorch, huggingface/transformers, faiss, fastapi の closed bugs |

取得方法:
- **Tavily**: Web 検索（credits に注意、基本除外）
- **arXiv**: 実装論文・パターン論文
- **SO**: 高評価 Q&A
- **GitHub API 拡張（案A）**: ドキュメントリポジトリからファイル内容を取得
  - `github.py` にファイル内容取得モードを追加
  - レート制限: 1 req/sec（Contents API 5,000/hr の範囲内）
  - 対象: mkerrisk/man-pages, mdn/content, nodejs/node/doc, python/cpython/Doc
- **URLリストフェッチャー（案C）**: キュレーテッドURLをバッチ取得
  - `url_fetcher.py` にバッチモードを追加（`data/doc_urls/*.txt` を入力）
  - レート制限: サイト別（docs.python.org: 1.5s, Arch Wiki: 5s, デフォルト: 2s）
  - 対象: Arch Wiki、The Linux Command Line、docs.python.org whatsnew
- **手動 seed**: 厳選ドキュメントを manual ソースとして投入

### URL リスト管理ファイル（data/doc_urls/）
```
data/doc_urls/
├── archwiki.txt          # Arch Wiki 運用ページ一覧
├── python_docs.txt       # docs.python.org/whatsnew + 重要ページ
├── linux_command_line.txt # linuxcommand.org チャプター
├── mdn_web_api.txt       # MDN Web API リファレンス（補完用）
└── release_notes.txt     # 各ライブラリのリリースノートURL
```

### GitHub ドキュメントリポジトリ一覧（案A）
```
data/doc_urls/github_doc_repos.yaml
  - repo: mkerrisk/man-pages
    path_prefix: man1/        # コマンドリファレンス優先
    path_prefix: man7/        # 概念・プロトコル
    rate_sec: 1.0
  - repo: mdn/content
    path_prefix: files/en-us/web/javascript/reference/
    path_prefix: files/en-us/web/css/
    rate_sec: 1.0
  - repo: nodejs/node
    path_prefix: doc/api/
    rate_sec: 1.0
  - repo: python/cpython
    path_prefix: Doc/library/
    version_tag: v3.12.0      # バージョン固定取得
    rate_sec: 1.0
```

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
