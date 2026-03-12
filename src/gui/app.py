"""src/gui/app.py — Gradio Web GUI メインアプリケーション。

タブ構成:
  1. チャット      — RAG + LLM クエリインターフェース
  2. FAISSメモリ   — インデックス統計・検索・追加
  3. コードサンドボックス — Docker 安全実行
  4. 設定         — APIキー・YAML設定編集
  5. 学習         — GRPO + TinyLoRA ダッシュボード

起動方法:
    python scripts/launch_gui.py [--host HOST] [--port PORT] [--share]
    # または直接
    python -m src.gui.app
"""

from __future__ import annotations

import argparse

import gradio as gr

# タブモジュール
from src.gui.tabs import chat, memory, sandbox, settings, training
from src.gui.components.status_bar import get_status_markdown

# ────────────────────────────────────────────────────────────────
# テーマ & CSS
# ────────────────────────────────────────────────────────────────

_CUSTOM_CSS = """
/* ヘッダーバー — 色はすべて CSS 変数経由でテーマ切り替えに追従 */
.med-header {
    background: var(--med-header-bg,
        linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%));
    padding: 16px 24px;
    border-radius: 8px;
    margin-bottom: 8px;
}
.med-header h1 {
    color: var(--button-primary-background-fill, #e94560);
    margin: 0;
    font-size: 1.6rem;
}
.med-header p {
    color: var(--block-label-text-color, #a0a8b8);
    margin: 4px 0 0 0;
    font-size: 0.9rem;
}
/* ステータスバー */
.status-bar {
    background: var(--block-background-fill, #1e1e2e);
    border: 1px solid var(--block-border-color, #2d2d4e);
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 0.85rem;
}
/* タブ強調 */
.tab-nav button {
    font-weight: 600;
}
"""

_THEME = gr.themes.Soft(
    primary_hue="rose",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
).set(
    body_background_fill="#0d0d1a",
    body_background_fill_dark="#0d0d1a",
    block_background_fill="#1a1a2e",
    block_background_fill_dark="#1a1a2e",
    block_border_color="#2d2d4e",
    block_border_color_dark="#2d2d4e",
    block_label_text_color="#a0a8b8",
    input_background_fill="#12122a",
    input_background_fill_dark="#12122a",
    button_primary_background_fill="#e94560",
    button_primary_background_fill_hover="#c73652",
    button_primary_text_color="#ffffff",
)

# ────────────────────────────────────────────────────────────────
# クライアントサイド テーマ切り替え JavaScript
# ────────────────────────────────────────────────────────────────
# CSS カスタムプロパティを documentElement にインラインで上書きすることで
# Gradio のテーマ変数を JS 実行時に置き換える。
# localStorage にテーマ名 / システム追従フラグを保存して再読み込み後も維持。

_THEME_INIT_JS = """() => {
    const THEMES = {
        "MED Dark": {
            "--body-background-fill":                    "#0d0d1a",
            "--block-background-fill":                   "#1a1a2e",
            "--block-border-color":                      "#2d2d4e",
            "--block-label-text-color":                  "#a0a8b8",
            "--input-background-fill":                   "#12122a",
            "--button-primary-background-fill":          "#e94560",
            "--button-primary-background-fill-hover":    "#c73652",
            "--button-primary-text-color":               "#ffffff",
            "--body-text-color":                         "#e0e0f0",
            "--block-title-text-color":                  "#c0c8d8",
            "--med-header-bg":                           "linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)"
        },
        "Soft Light": {
            "--body-background-fill":                    "#f9fafb",
            "--block-background-fill":                   "#ffffff",
            "--block-border-color":                      "#e5e7eb",
            "--block-label-text-color":                  "#6b7280",
            "--input-background-fill":                   "#f3f4f6",
            "--button-primary-background-fill":          "#f43f5e",
            "--button-primary-background-fill-hover":    "#e11d48",
            "--button-primary-text-color":               "#ffffff",
            "--body-text-color":                         "#374151",
            "--block-title-text-color":                  "#1f2937",
            "--med-header-bg":                           "linear-gradient(135deg, #dbeafe 0%, #e0e7ff 50%, #ede9fe 100%)"
        },
        "Ocean Dark": {
            "--body-background-fill":                    "#0c1929",
            "--block-background-fill":                   "#132235",
            "--block-border-color":                      "#1e3a5f",
            "--block-label-text-color":                  "#7ec8e3",
            "--input-background-fill":                   "#0a1520",
            "--button-primary-background-fill":          "#0ea5e9",
            "--button-primary-background-fill-hover":    "#0284c7",
            "--button-primary-text-color":               "#ffffff",
            "--body-text-color":                         "#e0f2fe",
            "--block-title-text-color":                  "#bae6fd",
            "--med-header-bg":                           "linear-gradient(135deg, #132235 0%, #0c2040 50%, #091525 100%)"
        },
        "Monochrome": {
            "--body-background-fill":                    "#111111",
            "--block-background-fill":                   "#1e1e1e",
            "--block-border-color":                      "#333333",
            "--block-label-text-color":                  "#888888",
            "--input-background-fill":                   "#161616",
            "--button-primary-background-fill":          "#e0e0e0",
            "--button-primary-background-fill-hover":    "#cccccc",
            "--button-primary-text-color":               "#111111",
            "--body-text-color":                         "#e0e0e0",
            "--block-title-text-color":                  "#cccccc",
            "--med-header-bg":                           "linear-gradient(135deg, #1e1e1e 0%, #161616 50%, #0a0a0a 100%)"
        },
        "Forest": {
            "--body-background-fill":                    "#0d1a0d",
            "--block-background-fill":                   "#162616",
            "--block-border-color":                      "#254525",
            "--block-label-text-color":                  "#7db87d",
            "--input-background-fill":                   "#0f1f0f",
            "--button-primary-background-fill":          "#4caf50",
            "--button-primary-background-fill-hover":    "#388e3c",
            "--button-primary-text-color":               "#ffffff",
            "--body-text-color":                         "#c8e6c9",
            "--block-title-text-color":                  "#a5d6a7",
            "--med-header-bg":                           "linear-gradient(135deg, #162616 0%, #0f2020 50%, #0a1a10 100%)"
        }
    };

    /* CSS カスタムプロパティを documentElement のインラインスタイルに
       直接上書きする（シートより優先度が高い）。 */
    function applyTheme(name) {
        const vars = THEMES[name];
        if (!vars) return;
        const root = document.documentElement;
        /* 旧テーマ変数を一旦すべて除去してから新テーマを適用 */
        Object.keys(THEMES["MED Dark"]).forEach(k => root.style.removeProperty(k));
        Object.entries(vars).forEach(([k, v]) => root.style.setProperty(k, v));
        try {
            localStorage.setItem("med-theme", name);
            localStorage.setItem("med-use-system", "false");
        } catch(e) {}
        window.__MED_CURRENT_THEME__ = name;
    }

    /* OS の prefers-color-scheme を読み取り対応テーマを適用 */
    function applySystemTheme() {
        const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
        applyTheme(dark ? "MED Dark" : "Soft Light");
        try { localStorage.setItem("med-use-system", "true"); } catch(e) {}
    }

    /* グローバルに公開 — settings.py の JS イベントから呼び出す */
    window.MED_THEMES        = THEMES;
    window.MED_applyTheme    = applyTheme;
    window.MED_applySystemTheme = applySystemTheme;

    /* OS テーマ変更をリアルタイム追従 */
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
        try {
            if (localStorage.getItem("med-use-system") === "true") applySystemTheme();
        } catch(e) {}
    });

    /* ページ読み込み時に保存済みテーマを復元 */
    try {
        if (localStorage.getItem("med-use-system") === "true") {
            applySystemTheme();
        } else {
            applyTheme(localStorage.getItem("med-theme") || "MED Dark");
        }
    } catch(e) {
        applyTheme("MED Dark");
    }
}"""


# ────────────────────────────────────────────────────────────────
# アプリ構築
# ────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    """Gradio Blocks アプリケーションを構築して返す。"""

    with gr.Blocks(
        title="MED — RAG × FAISS × LLM",
        theme=_THEME,
        css=_CUSTOM_CSS,
        analytics_enabled=False,
    ) as app:

        # ── ヘッダー ────────────────────────────────────────────
        gr.HTML("""
        <div class="med-header">
          <h1>MED</h1>
          <p>Memory Environment Distillation — RAG × FAISS × LLM × TinyLoRA</p>
        </div>
        """)

        # ── ステータスバー ───────────────────────────────────────
        with gr.Row():
            status_bar = gr.Markdown(
                get_status_markdown(),
                elem_classes=["status-bar"],
            )
            refresh_status_btn = gr.Button(
                "⟳", size="sm", scale=0, min_width=40, variant="secondary"
            )
        refresh_status_btn.click(fn=get_status_markdown, outputs=[status_bar])

        # ── メインタブ ──────────────────────────────────────────
        with gr.Tabs(elem_classes=["tab-nav"]):

            with gr.TabItem("💬 チャット"):
                gr.Markdown(
                    "_Teacher / Student モデルへのクエリ。RAGとFAISSメモリを活用した応答を返します。_"
                )
                chat.build_tab()

            with gr.TabItem("🧠 FAISSメモリ"):
                gr.Markdown(
                    "_ドメイン別ベクトルインデックスの統計表示・検索・ドキュメント追加。_"
                )
                memory.build_tab()

            with gr.TabItem("⚙️ サンドボックス"):
                gr.Markdown(
                    "_Docker Sandbox でコードを安全に実行します。ネットワーク無効・読み取り専用FS。_"
                )
                sandbox.build_tab()

            with gr.TabItem("🎓 学習"):
                gr.Markdown(
                    "_GRPO + TinyLoRA 学習の設定・制御・進捗モニタリング。_"
                )
                training.build_tab()

            with gr.TabItem("🔧 設定"):
                gr.Markdown(
                    "_APIキーの設定と YAML 設定ファイルの編集。_"
                )
                settings.build_tab()

        # ── フッター ────────────────────────────────────────────
        gr.Markdown(
            "<div style='text-align:center; color:var(--block-label-text-color, #555);"
            " margin-top:16px; font-size:0.8rem;'>"
            "MED v0.4.0 — RAG × FAISS × LLM × Memory Environment Distillation"
            "</div>"
        )

        # ── テーマ初期化 JS (ページ読み込み時に localStorage から復元) ──
        app.load(fn=None, js=_THEME_INIT_JS)

    return app


# ────────────────────────────────────────────────────────────────
# 起動ヘルパー
# ────────────────────────────────────────────────────────────────

def launch(
    host: str = "0.0.0.0",
    port: int = 7860,
    share: bool = False,
    debug: bool = False,
) -> None:
    """Gradio アプリを起動する。"""
    app = build_app()
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        debug=debug,
        show_error=True,
        favicon_path=None,
    )


# ────────────────────────────────────────────────────────────────
# エントリーポイント
# ────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MED Gradio Web GUI")
    parser.add_argument("--host", default="0.0.0.0", help="バインドホスト (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="ポート番号 (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Gradio share URLを生成")
    parser.add_argument("--debug", action="store_true", help="デバッグモード")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    launch(host=args.host, port=args.port, share=args.share, debug=args.debug)
