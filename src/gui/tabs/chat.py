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
from collections.abc import Generator

import gradio as gr
import httpx

from src.gui.utils import GRADIO_MAJOR, ORCHESTRATOR_URL, get_all_provider_choices, is_api_alive

# プロバイダーごとのよく使うモデル例
_MODEL_EXAMPLES: dict[str, list[str]] = {
    "anthropic": [
        "claude-sonnet-4-20250514",
        "claude-opus-4-6",
        "claude-haiku-4-5-20251001",
    ],
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    "ollama": ["llama3.1:8b", "llama3.1:70b", "mistral:7b", "codellama:13b"],
    "vllm": ["Qwen/Qwen2.5-7B-Instruct"],
}


# ────────────────────────────────────────────────────────────────
# API クライアントヘルパー
# ────────────────────────────────────────────────────────────────


_TIMEOUT_MIN = 5          # 最小タイムアウト: 5秒
_TIMEOUT_MAX = 86400      # 最大タイムアウト: 24時間
_TIMEOUT_DEFAULT = 300    # デフォルト: 5分


def _calc_timeout(hours: int, minutes: int, seconds: int) -> int:
    """時/分/秒 → 秒数を返す。範囲は [5, 86400] にクランプ。"""
    total = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    return max(_TIMEOUT_MIN, min(_TIMEOUT_MAX, total))


def _query_api(
    query: str,
    mode: str,
    use_memory: bool,
    use_rag: bool,
    provider: str | None,
    model: str | None,
    timeout_seconds: int = _TIMEOUT_DEFAULT,
) -> dict:
    payload: dict = {
        "query": query,
        "use_memory": use_memory,
        "use_rag": use_rag,
    }
    if mode and mode != "auto":
        payload["mode"] = mode
    if provider:
        payload["provider"] = provider
    if model:
        payload["model"] = model
    payload["timeout_seconds"] = timeout_seconds
    # クライアント側タイムアウトはサーバー側より少し長く設定（接続/応答バッファ）
    http_timeout = timeout_seconds + 10
    r = httpx.post(f"{ORCHESTRATOR_URL}/query", json=payload, timeout=http_timeout)
    r.raise_for_status()
    return r.json()


def _mock_response(query: str, mode: str, provider: str | None, model: str | None) -> dict:
    """APIが未起動の場合のモックレスポンス。"""
    time.sleep(0.3)
    provider_label = provider or "config依存"
    model_label = model or "config依存"
    return {
        "answer": (
            f"[MOCK — オーケストレーター未接続]\n\n"
            f"モード: **{mode}** | プロバイダー: **{provider_label}** | モデル: **{model_label}**\n\n"
            f"クエリ受信: `{query[:80]}{'...' if len(query) > 80 else ''}`\n\n"
            "FastAPI サーバーを起動すると実際の応答が返ります:\n"
            "```bash\nuvicorn src.orchestrator.server:app --reload --port 8000\n```"
        ),
        "provider": provider_label,
        "model": model_label,
        "input_tokens": 0,
        "output_tokens": 0,
        "context_doc_count": 0,
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
    provider_choice: str,
    model_name: str,
    timeout_h: int,
    timeout_m: int,
    timeout_s: int,
) -> Generator[tuple[list, str, str], None, None]:
    """Gradio チャット用ストリーミング風ジェネレータ。"""
    if not message.strip():
        yield history, "", "_入力が空です_"
        return

    timeout_seconds = _calc_timeout(timeout_h, timeout_m, timeout_s)

    # プロバイダー / モデルを解決
    actual_provider = None if provider_choice.startswith("auto") else provider_choice
    actual_model = model_name.strip() or None

    # 即座に「思考中…」を表示
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
            result = _query_api(
                message, mode, use_memory, use_rag,
                actual_provider, actual_model, timeout_seconds,
            )
        else:
            result = _mock_response(message, mode, actual_provider, actual_model)
    except Exception as e:
        result = {
            "answer": f"❌ エラー: {e}",
            "provider": "error",
            "model": "error",
            "input_tokens": 0,
            "output_tokens": 0,
            "context_doc_count": 0,
        }

    answer = result.get("answer", "")
    sources_md = _format_sources(result.get("sources", []))
    provider_used = result.get("provider", "unknown")
    model_used = result.get("model", "unknown")
    in_tok = result.get("input_tokens", 0)
    out_tok = result.get("output_tokens", 0)
    ctx_count = result.get("context_doc_count", 0)
    meta = (
        f"プロバイダー: `{provider_used}` | モデル: `{model_used}`  \n"
        f"トークン: ↑{in_tok} ↓{out_tok} | コンテキスト: {ctx_count}件"
    )

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

def build_tab() -> gr.Dropdown:
    """Gradio Blocks コンテキスト内でチャットタブを描画する。

    Returns:
        provider_dd: 設定タブ・app.load から選択肢を更新するために返す。
    """
    with gr.Row():
        with gr.Column(scale=3):
            if GRADIO_MAJOR >= 6:
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

            # ── サンプルプロンプト ───────────────────────────────
            with gr.Accordion("💡 サンプルプロンプト", open=False):
                gr.Markdown("_クリックで入力欄にセットされます_")
                _SAMPLES = [
                    ("Python 二分探索",       "Python で二分探索（バイナリサーチ）を実装してください"),
                    ("FAISS とは",            "FAISSとは何ですか？どのような場面で使われますか？"),
                    ("コードレビュー依頼",     "以下のコードのバグを見つけて修正してください:\n```python\ndef add(a, b):\n    return a - b\n```"),
                    ("RAG の仕組み",          "RAG（Retrieval-Augmented Generation）の仕組みをわかりやすく説明してください"),
                    ("TinyLoRA とは",         "TinyLoRA とはどのような技術ですか？通常の LoRA との違いを教えてください"),
                    ("GPT と Claude の違い",  "GPT-4o と Claude Sonnet の特徴と使い分けを比較してください"),
                ]
                for label, prompt in _SAMPLES:
                    btn = gr.Button(label, size="sm", variant="secondary")
                    btn.click(fn=lambda p=prompt: p, outputs=[msg_box])

        with gr.Column(scale=1):
            gr.Markdown("### モデル設定")
            mode_radio = gr.Radio(
                choices=["auto", "student", "teacher"],
                value="auto",
                label="モデルモード",
                info="auto: Router が自動選択",
            )
            use_memory_chk = gr.Checkbox(value=True, label="FAISSメモリ使用")
            use_rag_chk = gr.Checkbox(value=True, label="外部RAG使用")

            gr.Markdown("#### LLM プロバイダー / モデル")
            _initial_choices = get_all_provider_choices()
            provider_dd = gr.Dropdown(
                choices=_initial_choices,
                value=_initial_choices[0],
                label="プロバイダー",
                info="設定ファイルの primary_provider を使う場合は auto のまま",
            )
            model_box = gr.Textbox(
                placeholder="空白=設定ファイルのデフォルト",
                label="モデル名 (任意)",
                lines=1,
            )

            # プロバイダー選択時にモデル例を表示
            model_hint = gr.Markdown("_モデル名例が選択後に表示されます_")

            def _update_model_hint(prov: str) -> str:
                if prov.startswith("auto"):
                    return "_モデル名例: プロバイダーを選択してください_"
                examples = _MODEL_EXAMPLES.get(prov, [])
                if not examples:
                    return f"_{prov} のモデル名を直接入力してください_"
                joined = " / ".join(f"`{m}`" for m in examples)
                return f"_例: {joined}_"

            provider_dd.change(fn=_update_model_hint, inputs=[provider_dd], outputs=[model_hint])

            gr.Markdown("#### タイムアウト設定")
            gr.Markdown(
                "_範囲: 5秒 〜 24時間。デフォルトは 5分 (0h 5m 0s)。_",
            )
            with gr.Row():
                timeout_h = gr.Number(
                    value=0, minimum=0, maximum=24, step=1,
                    label="時 (h)", precision=0, scale=1,
                )
                timeout_m = gr.Number(
                    value=5, minimum=0, maximum=59, step=1,
                    label="分 (m)", precision=0, scale=1,
                )
                timeout_s = gr.Number(
                    value=0, minimum=0, maximum=59, step=1,
                    label="秒 (s)", precision=0, scale=1,
                )
            timeout_display = gr.Markdown("_設定値: 300秒 (5分)_")

            def _update_timeout_display(h, m, s):
                t = _calc_timeout(h, m, s)
                if t >= 3600:
                    label = f"{t // 3600}時間 {(t % 3600) // 60}分 {t % 60}秒"
                elif t >= 60:
                    label = f"{t // 60}分 {t % 60}秒"
                else:
                    label = f"{t}秒"
                return f"_設定値: **{t}秒** ({label})_"

            for w in (timeout_h, timeout_m, timeout_s):
                w.change(
                    fn=_update_timeout_display,
                    inputs=[timeout_h, timeout_m, timeout_s],
                    outputs=[timeout_display],
                )

            gr.Markdown("### レスポンス情報")
            meta_box = gr.Markdown("_送信後に表示_")

            gr.Markdown("### 参照ソース")
            sources_box = gr.Markdown("_ソースなし_")

    # イベント接続
    send_inputs = [
        msg_box, chatbot, mode_radio, use_memory_chk, use_rag_chk,
        provider_dd, model_box,
        timeout_h, timeout_m, timeout_s,
    ]
    send_outputs = [chatbot, meta_box, sources_box]

    def _on_send(message, history, mode, use_memory, use_rag, provider_choice, model_name):
        for update in respond(message, history, mode, use_memory, use_rag, provider_choice, model_name):
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

    return provider_dd
