# MED — Memory Environment Distillation

**RAG × FAISS × LLM × TinyLoRA** による Teacher-Student メモリ蒸留システム。

---

## 何ができるのか

```
あなたの質問
    ↓
複雑さを自動判定
    ├── 簡単 → Student モデル (Qwen-7B + TinyLoRA)  高速・無料
    ├── 中程度 → Student + FAISS記憶 + 外部RAG検索
    └── 複雑 → Teacher モデル (Claude / GPT)         高精度
    ↓
コードを含む場合 → Docker Sandbox で実行・検証
    ↓
回答をメモリに蓄積（次回の検索に活用）
```

## コアコンセプト

!!! quote "TinyLoRA 論文 (Morris et al., 2026) より"
    「知識は大モデルに既にある。RL は使い方だけを教える」

MEDはこれを発展させ、**知識を FAISS 外部メモリに蓄積**し、小モデルには「メモリの検索・活用スキル」だけを RL で教えます。

## 記憶のモデル

| コンポーネント | 人の記憶への対応 | 役割 |
|--------------|----------------|------|
| **FAISS** | 海馬（エピソード記憶・連想） | 意味的類似検索 |
| **Knowledge Graph** | 概念地図（意味記憶・関係性） | Entity 間の構造把握 |
| **SQL / BI** | ノートに書き起こした知識（宣言的記憶） | 正確な構造化クエリ |

## クイックリンク

<div class="grid cards" markdown>

-   :rocket: **[セットアップ](getting-started/setup.md)**

    5 分で MED を起動する手順

-   :zap: **[クイックスタート](getting-started/quickstart.md)**

    はじめてのクエリを送るまで

-   :brain: **[FAISSメモリ](features/memory.md)**

    ベクトルメモリの使い方と育て方

-   :microscope: **[Phase 2 メモリ成熟](phase2/overview.md)**

    Teacher 信頼度評価と品質審査

</div>

## バージョン履歴

| 版 | 主要変更 |
|----|---------|
| v1 | MemN2N ベースの初期設計 |
| v2 | MemN2N → FAISS 置換、Iterative Retrieval 導入 |
| v3 | LLM 統合、LTR / Cross-Encoder / Adapter、有用性管理 |
| **v4（現行）** | Teacher-Student 二層構造、GRPO + TinyLoRA、Knowledge Graph、Phase 2 成熟管理 |
