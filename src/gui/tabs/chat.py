"""src/gui/tabs/chat.py — チャット/クエリタブ。

Teacher/Student モデルへのクエリと RAG 結果を表示する。
FastAPI オーケストレーターが起動中の場合はHTTP経由で通信し、
未起動時はモックレスポンスを返す。

Gradio バージョン差異:
  5.x: タプル形式履歴 [(user, bot), ...]、show_copy_button、bubble_full_width
  6.x: 辞書形式履歴 [{"role":..., "content":...}, ...]、buttons=["copy"]
       bubble_full_width 削除(代替なし)
"""

from __future__ import annotations

import time
from typing import Generator

import gradio as gr
import httpx

from src.gui.utils import GRADIO_MAJOR, ORCHESTRATOR_URL, is_api_alive

# ────────────────────────────────────────────────────────────────
# API クライアントヘルパー
# ────────────────────────────────────────────────────────────────


def _query_api(
    prompt: str,
    mode: str,
    use_memory: bool,
    use_rag: bool,
) -> dict:
    payload = {
        "prompt": prompt,
        "mode": mode,
        "use_memory": use_memory,
        "use_rag": use_rag,
    }
    r = httpx.post(f"{ORCHESTRATOR_URL}/query", json=payload, timeout=60.0)
    r.raise_for_status()
    return r.json()


def _mock_response(prompt: str, mode: str) -> dict:
    """APIが未起動の場合のモックレスポンス。"""
    time.sleep(0.3)
    return {
        "answer": (
            f"[MOCK — オーケストレーター未接続]\n\n"
            f"モード: **{mode}**\n\n"
            f"クエリ受信: `{prompt[:80]}{'...' if len(prompt) > 80 else ''}`\n\n"
            "FastAPI サーバーを起動すると実際の応答が返ります:\n"
            "```bash\nuvicorn src.orchestrator.server:app --reload\n```"
        ),
        "sources": [],
        "model_used": f"mock-{mode}",
        "retrieval_count": 0,
        "latency_ms": 300,
    }


# ────────────────────────────────────────────────────────────────
# チャット履歴処理
# ────────────────────────────────────────────────────────────────

def _format_sources(sources: list) -> str:
    if not sources:
        return "_ソースなし_"
    lines = []
    for i, s in enumerate(sources, 1):
        title = s.get("title", "untitled")
        url = s.get("url", "")
        score = s.get("score", 0.0)
        lines.append(f"{i}. **{title}**  score={score:.3f}  {url}")
    return "\n".join(lines)


def respond(
    message: str,
    history: list,
    mode: str,
    use_memory: bool,
    use_rag: bool,
) -> Generator[tuple[list, str, str], None, None]:
    """Gradio チャット用ストリーミング風ジェネレータ。"""
    if not message.strip():
        yield history, "", "_入力が空です_"
        return

    # 即座に「思考中…」を表示
    # Gradio 6.x: 辞書形式  /  5.x: タプル形式
    if GRADIO_MAJOR >= 6:
        thinking_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⏳ 処理中…"},
        ]
    else:
        thinking_history = history + [(message, "⏳ 処理中…")]
    yield thinking_history, "", ""

    try:
        if is_api_alive():
            result = _query_api(message, mode, use_memory, use_rag)
        else:
            result = _mock_response(message, mode)
    except Exception as e:
        result = {
            "answer": f"❌ エラー: {e}",
            "sources": [],
            "model_used": "error",
            "latency_ms": 0,
        }

    answer = result.get("answer", "")
    sources_md = _format_sources(result.get("sources", []))
    model_used = result.get("model_used", "unknown")
    latency = result.get("latency_ms", 0)
    meta = f"モデル: `{model_used}` | レイテンシ: {latency}ms"

    if GRADIO_MAJOR >= 6:
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
    else:
        new_history = history + [(message, answer)]
    yield new_history, meta, sources_md


# ────────────────────────────────────────────────────────────────
# タブ UI 構築
# ────────────────────────────────────────────────────────────────

def build_tab() -> None:
    """Gradio Blocks コンテキスト内でチャットタブを描画する。"""
    with gr.Row():
        with gr.Column(scale=3):
            # Gradio 5.x: show_copy_button + bubble_full_width (deprecated but works)
            # Gradio 6.x: buttons=["copy"]、bubble_full_width は削除(代替なし)
            #             type="messages" で辞書形式履歴を使用
            if GRADIO_MAJOR >= 6:
                # Gradio 6.x: messages形式(辞書)のみサポート、type引数は廃止
                chatbot = gr.Chatbot(
                    label="チャット",
                    height=480,
                    buttons=["copy"],
                )
            else:
                chatbot = gr.Chatbot(
                    label="チャット",
                    height=480,
                    show_copy_button=True,
                    bubble_full_width=False,
                )
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="クエリを入力してください… (Shift+Enter で改行)",
                    show_label=False,
                    lines=2,
                    scale=9,
                )
                send_btn = gr.Button("送信", variant="primary", scale=1)
            clear_btn = gr.Button("履歴クリア", variant="secondary", size="sm")

        with gr.Column(scale=1):
            gr.Markdown("### 設定")
            mode_radio = gr.Radio(
                choices=["auto", "student", "teacher"],
                value="auto",
                label="モデルモード",
                info="auto: Router が自動選択",
            )
            use_memory_chk = gr.Checkbox(value=True, label="FAISSメモリ使用")
            use_rag_chk = gr.Checkbox(value=True, label="外部RAG使用")

            gr.Markdown("### レスポンス情報")
            meta_box = gr.Markdown("_送信後に表示_")

            gr.Markdown("### 参照ソース")
            sources_box = gr.Markdown("_ソースなし_")

    # イベント接続
    send_inputs = [msg_box, chatbot, mode_radio, use_memory_chk, use_rag_chk]
    send_outputs = [chatbot, meta_box, sources_box]

    def _on_send(message, history, mode, use_memory, use_rag):
        for update in respond(message, history, mode, use_memory, use_rag):
            yield update[0], update[1], update[2]

    send_btn.click(
        fn=_on_send,
        inputs=send_inputs,
        outputs=send_outputs,
    ).then(fn=lambda: "", outputs=[msg_box])

    msg_box.submit(
        fn=_on_send,
        inputs=send_inputs,
        outputs=send_outputs,
    ).then(fn=lambda: "", outputs=[msg_box])

    clear_btn.click(fn=lambda: ([], "", "_ソースなし_"), outputs=[chatbot, meta_box, sources_box])
