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

from src.gui.components.status_bar import get_status_markdown

# タブモジュール
from src.gui.tabs import chat, memory, sandbox, settings, training

# ────────────────────────────────────────────────────────────────
# テーマ & CSS
# ────────────────────────────────────────────────────────────────

_CUSTOM_CSS = """
/* ヘッダーバー */
.med-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 16px 24px;
    border-radius: 8px;
    margin-bottom: 8px;
}
.med-header h1 {
    color: #e94560;
    margin: 0;
    font-size: 1.6rem;
}
.med-header p {
    color: #a0a8b8;
    margin: 4px 0 0 0;
    font-size: 0.9rem;
}
/* ステータスバー */
.status-bar {
    background: #1e1e2e;
    border: 1px solid #2d2d4e;
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
            "<div style='text-align:center; color:#555; margin-top:16px; font-size:0.8rem;'>"
            "MED v0.4.0 — RAG × FAISS × LLM × Memory Environment Distillation"
            "</div>"
        )

    return app


# ────────────────────────────────────────────────────────────────
# 起動ヘルパー
# ────────────────────────────────────────────────────────────────

def launch(
    host: str = "0.0.0.0",
    port: int = 7860,
    share: bool = False,
    debug: bool = False,
    reload: bool = False,
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
