"""src/gui/tabs/sandbox.py — コードサンドボックス実行タブ。

Docker Sandbox (src/sandbox/) でコードを安全に実行する。
sandbox.manager が未実装またはAPIオフライン時はモックを返す。
"""

from __future__ import annotations

import time

import gradio as gr
import httpx

from src.gui.utils import GRADIO_MAJOR, ORCHESTRATOR_URL, is_api_alive

_EXAMPLE_PYTHON = """\
# FAISSインデックス動作確認の例
import numpy as np

dim = 4
vectors = np.random.rand(10, dim).astype("float32")
query = np.random.rand(1, dim).astype("float32")

# コサイン類似度 (正規化してIP)
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
vectors_norm = vectors / norms
query_norm = query / np.linalg.norm(query)

scores = (vectors_norm @ query_norm.T).flatten()
top3 = np.argsort(scores)[::-1][:3]

print("Top-3 results:")
for rank, idx in enumerate(top3, 1):
    print(f"  {rank}. index={idx}, score={scores[idx]:.4f}")
"""

_EXAMPLE_BASH = """\
#!/bin/bash
echo "Python version:"
python3 --version

echo "\\nInstalled packages (top 10):"
pip list 2>/dev/null | head -10

echo "\\nMemory usage:"
free -h 2>/dev/null || vm_stat
"""

_EXAMPLES = {
    "Python — FAISS類似度計算": ("python", _EXAMPLE_PYTHON),
    "Bash — 環境確認": ("bash", _EXAMPLE_BASH),
    "空 (Python)": ("python", ""),
    "空 (Bash)": ("bash", ""),
}

_SUPPORTED_LANGS = ["python", "bash", "javascript", "ruby"]


# ────────────────────────────────────────────────────────────────
# 実行ヘルパー
# ────────────────────────────────────────────────────────────────

def _execute_via_api(code: str, language: str, timeout: int) -> dict:
    r = httpx.post(
        f"{ORCHESTRATOR_URL}/sandbox/execute",
        json={"code": code, "language": language, "timeout": timeout},
        timeout=float(timeout + 10),
    )
    r.raise_for_status()
    return r.json()


def _mock_execute(code: str, language: str) -> dict:
    time.sleep(0.5)
    line_count = len(code.strip().splitlines())
    return {
        "stdout": (
            f"[MOCK実行 — Docker Sandbox未接続]\n"
            f"言語: {language} | コード行数: {line_count}\n\n"
            "実際の実行には以下が必要です:\n"
            "  1. Docker デーモンが起動中\n"
            "  2. FastAPI オーケストレーター起動:\n"
            "     uvicorn src.orchestrator.server:app --reload\n"
            "  3. sandbox イメージのビルド:\n"
            "     docker build -f Dockerfile.sandbox -t med-sandbox ."
        ),
        "stderr": "",
        "exit_code": 0,
        "execution_time_ms": 500,
        "memory_mb": 0,
    }


def _run_code(code: str, language: str, timeout: int) -> tuple[str, str, str]:
    """コードを実行し (stdout, stderr, メタ情報) を返す。"""
    if not code.strip():
        return "", "", "❌ コードが空です"

    if is_api_alive():
        try:
            result = _execute_via_api(code, language, timeout)
        except Exception as e:
            result = {"stdout": "", "stderr": str(e), "exit_code": -1, "execution_time_ms": 0}
    else:
        result = _mock_execute(code, language)

    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    exit_code = result.get("exit_code", 0)
    exec_ms = result.get("execution_time_ms", 0)
    mem_mb = result.get("memory_mb", 0)

    status_icon = "✅" if exit_code == 0 else "❌"
    meta = (
        f"{status_icon} exit={exit_code} | "
        f"時間: {exec_ms}ms | "
        f"メモリ: {mem_mb}MB"
    )
    return stdout, stderr, meta


# ────────────────────────────────────────────────────────────────
# タブ UI 構築
# ────────────────────────────────────────────────────────────────

def build_tab() -> None:
    """Gradio Blocks コンテキスト内でサンドボックスタブを描画する。"""

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### コードエディター")
            with gr.Row():
                lang_selector = gr.Dropdown(
                    choices=_SUPPORTED_LANGS,
                    value="python",
                    label="言語",
                    scale=1,
                )
                timeout_slider = gr.Slider(
                    5, 120, value=30, step=5, label="タイムアウト (秒)", scale=2
                )
                example_selector = gr.Dropdown(
                    choices=list(_EXAMPLES.keys()),
                    value=None,
                    label="サンプルコード",
                    scale=2,
                )

            code_editor = gr.Code(
                value=_EXAMPLE_PYTHON,
                language="python",
                label="コード",
                lines=20,
            )

            with gr.Row():
                run_btn = gr.Button("▶  実行", variant="primary", scale=3)
                clear_btn = gr.Button("クリア", variant="secondary", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("### 実行結果")
            exec_meta = gr.Markdown("_実行後に表示_")
            # Gradio 6.x: show_copy_button 削除 → buttons=["copy"]
            _copy_kwargs: dict = (
                {"buttons": ["copy"]} if GRADIO_MAJOR >= 6 else {"show_copy_button": True}
            )
            stdout_box = gr.Textbox(
                label="標準出力 (stdout)",
                lines=10,
                interactive=False,
                **_copy_kwargs,
            )
            stderr_box = gr.Textbox(
                label="標準エラー (stderr)",
                lines=5,
                interactive=False,
                **_copy_kwargs,
            )

            gr.Markdown("### セキュリティポリシー")
            gr.Markdown(
                "- ネットワーク: **無効**\n"
                "- ファイルシステム: **読み取り専用**\n"
                "- メモリ上限: **256 MB**\n"
                "- CPU: **0.5 コア**\n"
                "- 禁止syscall: mount, ptrace, reboot"
            )

    # イベント接続
    def _load_example(example_name):
        if example_name and example_name in _EXAMPLES:
            lang, code = _EXAMPLES[example_name]
            return code, lang
        # gr.update() は Gradio 6.x で deprecated → gr.skip() を使用
        return gr.skip(), gr.skip()

    example_selector.change(
        fn=_load_example,
        inputs=[example_selector],
        outputs=[code_editor, lang_selector],
    )

    def _on_lang_change(lang):
        return gr.update(language=lang)

    lang_selector.change(fn=_on_lang_change, inputs=[lang_selector], outputs=[code_editor])

    run_btn.click(
        fn=_run_code,
        inputs=[code_editor, lang_selector, timeout_slider],
        outputs=[stdout_box, stderr_box, exec_meta],
    )

    clear_btn.click(
        fn=lambda: ("", "", "_実行後に表示_"),
        outputs=[stdout_box, stderr_box, exec_meta],
    )
