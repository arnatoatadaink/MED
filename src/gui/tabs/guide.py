"""src/gui/tabs/guide.py — セットアップウィザード & ガイドタブ。

セットアップ状態をリアルタイム検出し、手順と機能説明を提供する。
各セクションはラジオボタンで切り替え可能。

案C: MEDアシスタント — MkDocs ドキュメントを FAISS 検索して回答する Q&A チャットBot。
"""

from __future__ import annotations

import os

import gradio as gr

from src.gui.utils import GRADIO_MAJOR, ORCHESTRATOR_URL, is_api_alive

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
        "label": "プロバイダー設定",
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

オーケストレーター（FastAPI バックエンド）と GUI は **別プロセス** です。
**先にオーケストレーターを起動してから** GUI を開いてください。

**ターミナル 1（オーケストレーター）:**
```bash
cd /path/to/MED
uvicorn src.orchestrator.server:app --reload --port 8000
```

**ターミナル 2（Gradio GUI）:**
```bash
cd /path/to/MED
python -m src.gui.app
# または: med-gui --port 7860
```

起動すると画面上部のステータスバーが **🟢 接続中** に変わります。

> 🔸 **オーケストレーター未起動でも GUI は動作します（モックモード）。**
> チャットタブには **[MOCK]** 応答が返ります。
> 実際の LLM 応答には Step 3 の起動が必要です。

**接続確認:**
```bash
curl http://localhost:8000/health
# → {"status":"ok","pipeline_initialized":true}
```

---

### Step 4 — はじめてのクエリを送る

**💬 チャット** タブを開きます。
サンプルプロンプト（クイックスタートボタン）をクリックするか、自由に入力してください。

```
Python で二分探索を実装してください
FAISSとは何ですか？
このコードのバグを直してください: [コードを貼り付け]
```

右カラムの **モデル設定** でプロバイダーとモデルを選択できます:

| 設定 | 動作 |
|------|------|
| プロバイダー: **auto** | 設定ファイル (`llm_config.yaml`) のデフォルト |
| プロバイダー: **anthropic** + モデル: `claude-opus-4-6` | 最高精度 |
| プロバイダー: **ollama** + モデル: `llama3.1:8b` | ローカル・無料 |

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

### LLMプロバイダー / モデルの選択

右カラムの **「LLM プロバイダー / モデル」** セクションで変更できます:

| プロバイダー | 代表モデル | 特徴 |
|-------------|-----------|------|
| **auto** *(デフォルト)* | 設定ファイル依存 | `llm_config.yaml` の `primary_provider` を使用 |
| **anthropic** | claude-sonnet-4-20250514, claude-opus-4-6 | 高精度・最新 |
| **openai** | gpt-4o, gpt-4o-mini | 汎用・マルチモーダル |
| **ollama** | llama3.1:8b, mistral:7b | ローカル実行・無料 |
| **vllm** | Qwen2.5-7B-Instruct | 高速推論・ローカル |

モデル名フィールドを空白にするとプロバイダーのデフォルトモデルが使われます。
プロバイダーを選択すると代表的なモデル名の例が表示されます。

---

### レスポンス情報

送信後に右カラムに表示されます:

- **プロバイダー**: 実際に使われたプロバイダー
- **モデル**: 実際に使われたモデル名
- **トークン**: 入力 ↑ / 出力 ↓ のトークン数
- **コンテキスト**: 参照した FAISS ドキュメント件数
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

    "🔬 Phase 2: メモリ成熟": """\
## Phase 2: メモリ成熟 (Week 4-5)

Teacher が FAISS メモリを「成熟」させるフェーズです。
品質の低いドキュメントを審査・タグ付け・削除し、Student の学習品質を高めます。

---

### Phase 2 品質目標

| 目標 | 値 | 確認場所 |
|------|-----|---------|
| ドキュメント数 | ≥ 10,000 docs | FAISSメモリ → 成熟管理 → 品質レポート |
| 平均信頼度 | ≥ 0.7 | 同上 |
| 実行成功率 | ≥ 80% | 同上 |

---

### Teacher 信頼度評価（追加実装済み）

Teacher モデルごとの信頼度を SQLite で管理し、
検索スコアに自動反映させるシステムです。

```
Teacher が回答生成
    ↓
FeedbackCollector がフィードバックを蓄積
    ↓
TeacherFeedbackPipeline.flush() で TeacherRegistry を更新
    ↓
EWMA（指数加重移動平均）で trust_score を更新
    ↓
CompositeScorer が trust_score を composite_score に乗算
    ↓
信頼度の低い Teacher のドキュメントは検索で後退
```

**Trust Score の更新アルゴリズム:**
- フィードバック ≤ 10 件: Welford 法（真の平均）
- フィードバック > 10 件: EWMA（α = 0.05 の固定学習率）
- trust_score 最小値: 0.05（完全に排除はしない）

---

### MemoryReviewer — Teacher 品質審査

`🧠 FAISSメモリ → 🔬 成熟管理 → 一括審査` から実行できます。

**審査フロー:**
1. `MetadataStore.get_unreviewed()` で未審査ドキュメントを取得
2. Teacher LLM に品質評価を依頼（JSON形式で返答）
3. `quality_score` / `confidence` / `approved` / `reason` を抽出
4. `MetadataStore.update_quality()` で DB を更新
5. `review_status` を `approved` / `rejected` に変更

**承認閾値:** `quality_score ≥ 0.6` → approved

---

### DifficultyTagger — 難易度タグ付け

`scripts/mature_memory.py` から実行:
```bash
python scripts/mature_memory.py --limit 100 --tag-difficulty
```

難易度 (beginner / intermediate / advanced / expert) は
Student の学習カリキュラム順序決定に使われます:
- `beginner` → `intermediate` → `advanced` → `expert` の順に提示

---

### CrossEncoder — Phase 2 再ランキング

基本の FAISS 検索（IndexFlatIP / コサイン類似度）に加えて、
Cross-Encoder による意味的再ランキングを提供します。

| 方式 | 速度 | 精度 |
|------|------|------|
| FAISS (Bi-encoder) | 高速 (O(1)) | 近似 |
| Cross-Encoder (Phase 2) | 低速 (O(N)) | 高精度 |

Phase 2 以降は上位 K 件を FAISS で取得した後、Cross-Encoder で再ランキングします。

---

### Teacher 信頼度を確認するには

**GUI から:**
1. `🧠 FAISSメモリ` タブを開く
2. `🔬 成熟管理 (Phase 2)` サブタブを選択
3. `Teacher 信頼度` タブで各モデルの trust_score を確認

**スクリプトから:**
```bash
python scripts/mature_memory.py --show-teachers
```
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

    gr.Markdown("---")

    # ── MEDアシスタント（案C）──────────────────────────────────────
    _build_assistant_section()


# ────────────────────────────────────────────────────────────────
# MEDアシスタント（案C） — ドキュメント Q&A チャットBot
# ────────────────────────────────────────────────────────────────

# サンプル質問リスト
_SAMPLE_QUESTIONS = [
    "FAISSメモリの使い方を教えてください",
    "プロバイダーの設定方法は？",
    "Phase 2 メモリ成熟とは何ですか？",
    "TinyLoRA と GRPO の仕組みを説明してください",
    "サンドボックスのセキュリティポリシーは？",
    "Knowledge Graph はどう機能しますか？",
]


def _init_engine_status() -> str:
    """エンジン初期化状態のメッセージを返す。"""
    try:
        from src.gui.docs_chat import get_engine
        engine = get_engine()
        engine.initialize()
        n = len(engine._chunks)
        backend = "FAISS ベクトル検索" if isinstance(
            engine._searcher,
            type(engine._searcher),  # duck typing
        ) else "キーワード検索"
        # より具体的な判定
        try:
            import faiss  # noqa: F401
            from sentence_transformers import SentenceTransformer  # noqa: F401
            backend = "FAISS ベクトル検索"
        except ImportError:
            backend = "キーワード検索"
        return (
            f"✅ ドキュメント **{n} チャンク** 読み込み済み — "
            f"検索方式: **{backend}**"
        )
    except Exception as e:
        return f"⚠️ エンジン初期化エラー: {e}"


def _respond(message: str, history: list) -> tuple[list, str]:
    """ユーザーのメッセージを受け取り、更新された履歴と空の入力を返す。"""
    if not message.strip():
        return history, ""
    try:
        from src.gui.docs_chat import get_engine
        engine = get_engine()
        engine.initialize()
        answer = engine.answer(message)
    except Exception as e:
        answer = f"❌ エラーが発生しました: {e}"

    if GRADIO_MAJOR >= 5:
        new_history = list(history) + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
    else:
        new_history = list(history) + [(message, answer)]

    return new_history, ""


def _set_sample(sample: str) -> str:
    """サンプル質問ボタンが押されたときにテキストボックスに反映する。"""
    return sample


def _build_assistant_section() -> None:
    """MEDアシスタント UI を構築する。"""

    with gr.Accordion("🤖 MEDアシスタント — ドキュメント Q&A", open=False):

        gr.Markdown(
            "**MED のドキュメント**（セットアップ・機能・アーキテクチャ等）に関する"
            "質問に回答します。\n\n"
            "オーケストレーター起動中は LLM による回答、"
            "未起動時は関連ドキュメントのチャンクをそのまま表示します。"
        )

        # エンジン状態表示
        engine_status = gr.Markdown("⏳ エンジン初期化中...")

        # チャット履歴
        if GRADIO_MAJOR >= 5:
            chatbot = gr.Chatbot(
                label="MEDアシスタント",
                type="messages",
                height=420,
                show_copy_button=True,
                bubble_full_width=False,
                avatar_images=(None, "🤖"),
            )
        else:
            chatbot = gr.Chatbot(
                label="MEDアシスタント",
                type="messages",
                height=420,
                show_copy_button=True,
                avatar_images=(None, "🤖"),
            )

        # 入力行
        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="MEDについて質問してください... (例: FAISSの使い方は？)",
                show_label=False,
                scale=5,
                lines=1,
            )
            send_btn = gr.Button("送信", variant="primary", scale=1, min_width=80)
            clear_btn = gr.Button("クリア", variant="secondary", scale=1, min_width=70)

        # サンプル質問ボタン
        with gr.Accordion("💡 サンプル質問", open=False):
            with gr.Row(wrap=True):
                sample_btns = [
                    gr.Button(q, size="sm", variant="secondary")
                    for q in _SAMPLE_QUESTIONS
                ]

        # イベント接続
        send_btn.click(
            fn=_respond,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input],
        )
        msg_input.submit(
            fn=_respond,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input],
        )
        clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, msg_input])

        for btn in sample_btns:
            btn.click(fn=_set_sample, inputs=[btn], outputs=[msg_input])

        # Accordion が開かれたときにエンジンを初期化（load イベントで代用）
        # Gradio では Accordion の open イベントが無いため、
        # ページロード時に非同期で初期化する
        chatbot.change(fn=None, outputs=[])  # dummy to ensure component is registered

    # ページロード時にエンジン初期化ステータスを更新
    # （build_tab が Blocks.load より後に呼ばれるので直接ロードできないため、
    #   Accordion open ではなくここで初期化コールを仕込む）
    engine_status.value = _init_engine_status()
