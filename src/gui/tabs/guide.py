"""src/gui/tabs/guide.py — セットアップウィザード & ガイドタブ。

セットアップ状態をリアルタイム検出し、手順と機能説明を提供する。
各セクションはラジオボタンで切り替え可能。
"""

from __future__ import annotations

import os

import gradio as gr

from src.gui.utils import ORCHESTRATOR_URL, is_api_alive

# ────────────────────────────────────────────────────────────────
# セットアップ状態チェック
# ────────────────────────────────────────────────────────────────

_KNOWN_PROVIDER_KEYS: dict[str, str] = {
    "anthropic":    "ANTHROPIC_API_KEY",
    "openai":       "OPENAI_API_KEY",
    "azure_openai": "AZURE_OPENAI_API_KEY",
    "together":     "TOGETHER_API_KEY",
}
_LOCAL_PROVIDERS = {"ollama", "vllm"}


def _get_primary_provider() -> str:
    from pathlib import Path
    import yaml
    cfg_path = Path(__file__).parent.parent.parent.parent / "configs" / "llm_config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("primary_provider", "anthropic")
    return "anthropic"


def _check_setup() -> tuple[list[dict], bool]:
    """
    セットアップ状態を確認する。
    Returns (steps, all_ok)
    各 step = {"label": str, "status": "ok"|"warn"|"err", "hint": str}
    """
    steps: list[dict] = []
    primary = _get_primary_provider()

    # ① プロバイダー設定
    steps.append({
        "label": f"プロバイダー設定",
        "status": "ok",
        "detail": f"主要プロバイダー: `{primary}`",
        "hint": "🔧 設定 → プロバイダー設定 → プリセット から変更できます",
    })

    # ② APIキー
    if primary in _LOCAL_PROVIDERS:
        steps.append({
            "label": "APIキー",
            "status": "ok",
            "detail": f"`{primary}` はローカル実行のためAPIキー不要",
            "hint": "",
        })
    elif primary in _KNOWN_PROVIDER_KEYS:
        env_name = _KNOWN_PROVIDER_KEYS[primary]
        val = os.environ.get(env_name, "")
        if val:
            masked = val[:4] + "****" + val[-4:] if len(val) > 8 else "****"
            steps.append({
                "label": f"APIキー ({env_name})",
                "status": "ok",
                "detail": f"設定済み: `{masked}`",
                "hint": "",
            })
        else:
            steps.append({
                "label": f"APIキー ({env_name})",
                "status": "err",
                "detail": "未設定",
                "hint": f"🔧 設定 → APIキー タブで `{env_name}` を入力してください",
            })
    else:
        # カスタムプロバイダー — api_key_env を読む
        from pathlib import Path
        import yaml
        cfg_path = Path(__file__).parent.parent.parent.parent / "configs" / "llm_config.yaml"
        cfg = {}
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
        p_cfg = cfg.get("providers", {}).get(primary, {})
        env_name = p_cfg.get("api_key_env", "")
        val = os.environ.get(env_name, "") if env_name else ""
        if not env_name:
            steps.append({
                "label": "APIキー",
                "status": "warn",
                "detail": "カスタムプロバイダー (api_key_env 未設定)",
                "hint": "🔧 設定 → カスタムプロバイダー で api_key_env を設定してください",
            })
        elif val:
            steps.append({
                "label": f"APIキー ({env_name})",
                "status": "ok",
                "detail": "設定済み",
                "hint": "",
            })
        else:
            steps.append({
                "label": f"APIキー ({env_name})",
                "status": "err",
                "detail": "未設定",
                "hint": f"🔧 設定 → APIキー タブで `{env_name}` を入力してください",
            })

    # ③ オーケストレーター接続
    alive = is_api_alive()
    steps.append({
        "label": "オーケストレーター (FastAPI)",
        "status": "ok" if alive else "warn",
        "detail": f"接続中: `{ORCHESTRATOR_URL}`" if alive else f"未接続 ({ORCHESTRATOR_URL})",
        "hint": (
            ""
            if alive
            else "以下のコマンドで起動してください:\n```bash\nuvicorn src.orchestrator.server:app --reload --port 8000\n```"
        ),
    })

    all_ok = all(s["status"] == "ok" for s in steps)
    return steps, all_ok


def _render_setup_status() -> str:
    steps, all_ok = _check_setup()
    icons = {"ok": "✅", "warn": "⚠️", "err": "❌"}
    lines: list[str] = []

    if all_ok:
        lines.append("### ✅ セットアップ完了 — すぐに使えます\n")
    else:
        lines.append("### セットアップ状況\n")

    for s in steps:
        icon = icons[s["status"]]
        lines.append(f"**{icon} {s['label']}**: {s['detail']}")
        if s["hint"]:
            lines.append(f"> {s['hint']}\n")

    if not all_ok:
        lines.append("\n---\n_問題を解決してから「再チェック」ボタンを押してください。_")

    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────
# ガイドコンテンツ定義
# ────────────────────────────────────────────────────────────────

_GUIDE_SECTIONS: dict[str, str] = {

    "🚀 セットアップ手順": """\
## セットアップ手順

MED を使い始めるための 4 ステップです。

---

### Step 1 — プロバイダープリセットを選ぶ

**🔧 設定 → プロバイダー設定 → プリセット** タブへ移動します。

| プリセット | 必要なもの | 月コスト目安 |
|-----------|-----------|------------|
| **Anthropic Claude** *(推奨)* | ANTHROPIC_API_KEY | ~$1–10 |
| **OpenAI GPT-4o** | OPENAI_API_KEY | ~$1–10 |
| **Ollama (ローカル)** | Ollama サーバー | 無料 |
| **vLLM (ローカル)** | vLLM サーバー | 無料 |
| **Together.ai** | TOGETHER_API_KEY | ~$1–5 |
| **Azure OpenAI** | AZURE_OPENAI_API_KEY | 従量制 |

プリセットを選択して「**適用**」ボタンを押すと `llm_config.yaml` が書き換わります。

---

### Step 2 — APIキーを設定する

**🔧 設定 → APIキー** タブで対応するキーを入力し「適用」を押します。

> ⚠️ セッション中のみ有効です。永続化するには `.env` ファイルに記載してください。
>
> ```
> ANTHROPIC_API_KEY=sk-ant-...
> OPENAI_API_KEY=sk-...
> ```

**ローカルプロバイダー（Ollama / vLLM）** の場合はキー不要です。
Ollama は別ターミナルで起動しておいてください:
```bash
ollama serve
ollama pull llama3.1:8b
```

---

### Step 3 — オーケストレーターを起動する

プロジェクトルートで以下を実行します:
```bash
uvicorn src.orchestrator.server:app --reload --port 8000
```

起動すると画面上部のバナーが **接続中** に変わります。

> 🔸 オーケストレーター未起動でも GUI は動作しますが、
> チャットタブには **[MOCK]** 応答が返ります。

---

### Step 4 — はじめてのクエリを送る

**💬 チャット** タブを開きます。
サンプルプロンプト（クイックスタートボタン）をクリックするか、自由に入力してください。

```
Python で二分探索を実装してください
FAISSとは何ですか？
このコードのバグを直してください: [コードを貼り付け]
```

「**送信**」を押すと Teacher / Student モデルが選択され、
FAISSメモリと外部RAGを組み合わせた回答が返ります。
""",

    "💬 チャット機能": """\
## チャット機能

Teacher / Student モデルへのクエリを送り、RAG と FAISS メモリを活用した回答を受け取ります。

---

### モデルモード

| モード | 動作 |
|--------|------|
| **auto** *(デフォルト)* | クエリの複雑さを自動判定してモデルを選択 |
| **student** | Student モデル（Qwen-7B+TinyLoRA）を強制使用。高速・低コスト |
| **teacher** | Teacher モデル（Claude/GPT API）を強制使用。高精度 |

---

### FAISSメモリ / 外部RAG

| オプション | 効果 |
|------------|------|
| **FAISSメモリ使用** ON | 過去の会話・蓄積した知識をベクトル検索して参照 |
| **外部RAG使用** ON | GitHub / Stack Overflow / Tavily をリアルタイム検索して回答に組み込む |

どちらもOFFにすると、モデルの知識のみで回答します。

---

### レスポンス情報

送信後に右カラムに表示されます:

- **モデル**: 実際に使われたモデル名
- **レイテンシ**: 応答時間 (ms)
- **参照ソース**: 外部RAGが使用したURL・スコア

---

### Tips

- `Shift+Enter` で改行、`Enter` または「送信」で送信
- 「**履歴クリア**」ボタンで会話をリセット（FAISSメモリはクリアされません）
- **コード質問**は auto / teacher モードが向いています（サンドボックス実行を内部で行います）
""",

    "🧠 FAISSメモリ機能": """\
## FAISSメモリ機能

ベクトルインデックス（FAISS）に知識を蓄積・検索するメモリ管理ツールです。

---

### 概念図

```
テキスト → [Embedder: all-MiniLM-L6-v2] → ベクトル(768次元)
                                              ↓
                                    [FAISS IndexFlatIP]
                                              ↓
クエリ → ベクトル化 → 上位K件を取得 → LLMに渡して回答生成
```

---

### 各セクションの説明

**インデックス統計**
- 登録ドキュメント数・ドメイン分布・最終更新時刻を表示
- 「更新」ボタンでリアルタイム反映

**セマンティック検索**
- クエリを入力 → FAISS がコサイン類似度で上位 K 件を返す
- スコアと本文プレビューを一覧表示

**ドキュメント追加**
- タイトル・本文・ドメイン（python / general / math など）を入力
- 「追加」で即座に埋め込み → FAISS に登録

---

### ドメイン

| ドメイン | 用途 |
|---------|------|
| `python` | Pythonコード・ライブラリ |
| `math` | 数式・アルゴリズム |
| `general` | 汎用知識 |
| `system` | システム設計・アーキテクチャ |

---

### メモリを育てるには

1. チャットで Teacher モデルに質問する → 応答が自動的にメモリに保存される
2. `scripts/seed_memory.py` でバルクデータを投入する
3. `scripts/mature_memory.py` で Teacher が品質審査・難易度タグ付けを行う
""",

    "⚙️ サンドボックス機能": """\
## サンドボックス機能

Docker コンテナ内でコードを安全に実行します。
ネットワーク無効・読み取り専用FS・メモリ制限付きの隔離環境です。

---

### セキュリティポリシー

| 制限 | 設定値 |
|------|--------|
| ネットワーク | 無効（`--network none`）|
| ファイルシステム | 読み取り専用（書き込みは `/tmp` のみ）|
| CPU | 1コア |
| メモリ | 256MB |
| タイムアウト | 30秒 |
| 禁止コマンド | `rm -rf /`, `fork bomb`, `os.system` 等 |

---

### 使い方

1. 「コード入力」エリアに実行したいコードを貼り付ける
2. 言語を選択（Python / JavaScript / Bash）
3. 「実行」ボタンを押す
4. 標準出力・エラー出力・実行時間が表示される

---

### 自動リトライ

実行が失敗した場合:
1. エラーを LLM が解析してコードを修正提案
2. ユーザーの確認後、修正版を再実行
3. 最大 3 回までリトライ

---

### チャットとの連携

チャットタブでコード生成を依頼すると、内部でサンドボックスが自動実行されます。
実行成功した場合のみ回答として返されるため、**動作保証済みのコード**を受け取れます。
""",

    "🎓 学習フレームワーク": """\
## 学習フレームワーク

Teacher モデルが成熟させた FAISS メモリを使って、
Student モデル（Qwen-7B）に「**メモリの使い方**」を GRPO + TinyLoRA で学習させます。

---

### 概念（TinyLoRA 論文より）

> 「知識は既にある。使い方だけ RL で教える」

- **Teacher**: Claude / GPT API — 高精度・コスト高
- **Student**: Qwen2.5-7B-Instruct — 高速・低コスト
- **TinyLoRA**: 13 パラメータのアダプタ（~1KB）で Student に FAISSの使い方を教える

---

### 学習パイプライン（3段階）

```
① SFT ウォームアップ
  Teacher の応答を教師データとして Student に SFT（教師あり学習）

② GRPO + TinyLoRA
  Student が FAISS を検索 → 回答生成 → Reward 計算 → アダプタ更新

③ 評価
  ベンチマークで Teacher との差を測定
```

---

### Reward 関数の内訳

| 信号 | 重み | 説明 |
|------|------|------|
| correctness | 0.35 | 回答の正確性 |
| retrieval_quality | 0.20 | FAISS 検索の適切さ |
| exec_success | 0.20 | コード実行成功率 |
| memory_utilization | 0.15 | メモリの活用度 |
| efficiency | 0.10 | レイテンシ・トークン効率 |

---

### 学習タブの操作

1. **アルゴリズム選択**: GRPO（デフォルト）/ PPO / DPO / SFT
2. **アダプタ選択**: TinyLoRA（デフォルト）/ LoRA / LoRA-XS / Full FT
3. **「開始」ボタン**: 学習ジョブを起動（バックグラウンド実行）
4. **進捗モニター**: Loss / Reward のリアルタイムグラフ
5. **「停止」ボタン**: 学習を中断してアダプタを保存

---

### 学習済みアダプタ

`data/adapters/` に保存されます（~1KB/アダプタ）。
チャットタブで Student モードを選択すると自動的に最新アダプタが使用されます。
""",

    "🏗️ アーキテクチャ概要": """\
## アーキテクチャ概要

---

### システム全体図

```
ブラウザ (Gradio GUI)
    │
    ▼
Orchestrator (FastAPI :8000)
    │
    ├── Query Parser (LLM)
    │       クエリの複雑さを判定
    │
    ├── Model Router
    │       simple   → Student (Qwen-7B + TinyLoRA)
    │       moderate → Student + FAISSメモリ + 外部RAG
    │       complex  → Teacher (Claude / GPT)
    │
    ├── RAG Pipeline
    │       GitHub / Stack Overflow / Tavily / arXiv
    │       → LLM 裏どり → チャンク化
    │
    ├── FAISS Memory
    │       ドメイン別ベクトルインデックス
    │       SQLite メタデータ（有用性スコア）
    │       Iterative Retrieval（マルチホップ）
    │
    └── Docker Sandbox
            コード実行・セキュリティ制限・自動リトライ
```

---

### データフロー（チャットクエリの場合）

```
1. ユーザー入力
      ↓
2. Query Parser が複雑さを判定 (simple / moderate / complex)
      ↓
3. simple/moderate → Student が FAISS を検索
   complex         → Teacher が回答生成
      ↓
4. 外部RAG（必要なら）GitHub/SO/Tavily を検索
      ↓
5. コード含む場合 → Docker Sandbox で実行・検証
      ↓
6. 回答を FAISS に保存（有用性スコア付き）
      ↓
7. ユーザーに返答
```

---

### ファイル構成（主要部分）

```
src/
├── orchestrator/    FastAPI + Query Parser + Model Router
├── llm/            LLM プロバイダー抽象化 (Claude / GPT / Ollama / vLLM)
├── rag/            外部検索パイプライン
├── memory/         FAISS + SQLite メモリ管理
├── training/       GRPO + TinyLoRA 学習フレームワーク
├── sandbox/        Docker Sandbox 管理
└── gui/            Gradio Web UI (このアプリ)
```

---

### 設定ファイル

| ファイル | 内容 |
|---------|------|
| `configs/default.yaml` | ホスト・ポート・埋め込みモデル |
| `configs/llm_config.yaml` | プロバイダー設定・タスクルーティング |
| `configs/faiss_config.yaml` | インデックス種別・次元数 |
| `configs/training.yaml` | 学習アルゴリズム・ハイパーパラメータ |
| `configs/sandbox_policy.yaml` | セキュリティ制限 |
| `configs/retrievers.yaml` | 外部検索 API 設定 |
""",
}

# ────────────────────────────────────────────────────────────────
# タブ UI 構築
# ────────────────────────────────────────────────────────────────

def build_tab() -> None:
    """Gradio Blocks コンテキスト内でガイドタブを描画する。"""

    # ── セットアップウィザード ────────────────────────────────────
    with gr.Accordion("🧭 セットアップウィザード", open=True):
        status_md = gr.Markdown(_render_setup_status())
        recheck_btn = gr.Button("再チェック", variant="secondary", size="sm")
        recheck_btn.click(fn=_render_setup_status, outputs=[status_md])

    gr.Markdown("---")

    # ── ガイドセクション ─────────────────────────────────────────
    section_names = list(_GUIDE_SECTIONS.keys())

    with gr.Row():
        with gr.Column(scale=1, min_width=180):
            gr.Markdown("### セクション")
            section_radio = gr.Radio(
                choices=section_names,
                value=section_names[0],
                show_label=False,
            )

        with gr.Column(scale=4):
            content_md = gr.Markdown(
                _GUIDE_SECTIONS[section_names[0]],
                height=540,
            )

    section_radio.change(
        fn=lambda s: _GUIDE_SECTIONS.get(s, ""),
        inputs=[section_radio],
        outputs=[content_md],
    )
