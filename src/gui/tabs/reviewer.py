"""src/gui/tabs/reviewer.py — マルチスレッド Reviewer 制御タブ

モデルスロット設定・ペルソナ選択・開始/停止・進捗モニタリング。
"""

from __future__ import annotations

import time
from typing import Optional

import gradio as gr
import pandas as pd

from src.cycle.reviewer_worker import (
    ReviewerConfig,
    ReviewerSession,
    SlotConfig,
    get_persona_choices,
)
from src.gui.utils import get_all_provider_choices

_MAX_SLOTS = 4
_session: Optional[ReviewerSession] = None


# ── セッション管理 ────────────────────────────────────────────

def _start_review(
    limit: int,
    timeout_sec: int,
    include_low_quality: bool,
    # slot 1
    p1: str, m1: str, ps1: list[str],
    # slot 2
    p2: str, m2: str, ps2: list[str],
    # slot 3
    p3: str, m3: str, ps3: list[str],
    # slot 4
    p4: str, m4: str, ps4: list[str],
) -> str:
    global _session
    if _session and _session.get_stats()["is_running"]:
        return "⚠️ 既にセッションが実行中です。停止してから再実行してください。"

    slots: list[SlotConfig] = []
    for provider, model, personas in [(p1, m1, ps1), (p2, m2, ps2), (p3, m3, ps3), (p4, m4, ps4)]:
        provider = (provider or "").strip()
        if not provider or provider.startswith("auto"):
            continue
        if not personas:
            continue
        slots.append(SlotConfig(provider=provider, model=(model or "").strip(), personas=personas))

    if not slots:
        return "⚠️ 有効なスロットがありません。Provider とペルソナを指定してください。"

    cfg = ReviewerConfig(
        slots=slots,
        limit=int(limit),
        timeout_sec=int(timeout_sec),
        include_low_quality=include_low_quality,
    )
    _session = ReviewerSession(cfg)
    n = _session.build()
    if n == 0:
        return "ℹ️ レビュー対象文書が見つかりません（unreviewed / needs_update がゼロ）。"
    _session.start()
    return f"🚀 セッション開始 — {n} 件 / {len(slots)} スロット"


def _stop_review() -> str:
    global _session
    if _session is None:
        return "ℹ️ 実行中のセッションはありません。"
    _session.stop()
    return "🛑 停止要求を送信しました。各スレッドの終了を待っています。"


def _get_status_md() -> str:
    if _session is None:
        return "_セッション未起動_"
    s = _session.get_stats()
    c = s["counts"]
    running = "実行中 ▶" if s["is_running"] else "停止済 ■"
    eta = f"残り ~{s['eta_sec']}秒" if s["eta_sec"] is not None else "ETA 計算中"
    elapsed = s["elapsed_sec"]
    return (
        f"**{running}** | "
        f"合計 {s['total']} 件 | "
        f"完了 {c['done']} | "
        f"処理中 {c['in_progress']} | "
        f"エラー {c['error']} | "
        f"待機 {c['pending']} | "
        f"経過 {elapsed}秒 | {eta}"
    )


def _get_task_df() -> pd.DataFrame:
    if _session is None:
        return pd.DataFrame(columns=["doc_id", "source", "persona", "status", "assigned"])
    rows = _session.get_task_rows()
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["doc_id", "source", "persona", "status", "assigned"]
    )


def _refresh_all():
    return _get_status_md(), _get_task_df()


# ── タブ構築 ───────────────────────────────────────────────────

def build_tab() -> None:
    """Reviewer 制御タブを構築する。"""
    provider_choices = get_all_provider_choices()
    persona_choices = get_persona_choices()

    # ── 実行設定 ───────────────────────────────────────────────
    with gr.Accordion("実行設定", open=True):
        with gr.Row():
            limit_nb = gr.Number(label="最大件数", value=200, minimum=1, maximum=2000, scale=1)
            timeout_nb = gr.Number(label="タイムアウト(秒)", value=60, minimum=10, maximum=600, scale=1)
            low_q_cb = gr.Checkbox(label="needs_update も含む", value=True, scale=1)

    # ── モデルスロット ─────────────────────────────────────────
    gr.Markdown("#### モデルスロット（最大 4）")
    slot_inputs: list[tuple[gr.Dropdown, gr.Textbox, gr.CheckboxGroup]] = []
    for i in range(1, _MAX_SLOTS + 1):
        with gr.Row():
            p = gr.Dropdown(
                choices=provider_choices,
                label=f"Slot {i} Provider",
                value="fastflowlm" if i == 1 else None,
                scale=2,
            )
            m = gr.Textbox(
                label="Model (空欄=デフォルト)",
                placeholder="例: gemini-2.0-flash",
                scale=3,
            )
            ps = gr.CheckboxGroup(
                choices=persona_choices,
                label="対応ペルソナ",
                value=["auto"] if i == 1 else [],
                scale=4,
            )
        slot_inputs.append((p, m, ps))

    # ── 開始/停止 ──────────────────────────────────────────────
    gr.Markdown("---")
    with gr.Row():
        start_btn = gr.Button("▶ レビュー開始", variant="primary", scale=2)
        stop_btn = gr.Button("■ 停止", variant="stop", scale=1)
    action_status = gr.Markdown("_ここにアクション結果が表示されます。_")

    # ── 進捗モニター ───────────────────────────────────────────
    gr.Markdown("#### 進捗モニター")
    with gr.Row():
        refresh_btn = gr.Button("⟳ 更新", size="sm", scale=0, min_width=80, variant="secondary")
        status_md = gr.Markdown(_get_status_md())

    task_df = gr.Dataframe(
        value=_get_task_df(),
        interactive=False,
        wrap=False,
        label="タスク一覧（最大 200 件）",
    )

    # Gradio 5+ の Timer で 10 秒ごと自動更新
    try:
        timer = gr.Timer(value=10)
        timer.tick(fn=_refresh_all, outputs=[status_md, task_df])
    except AttributeError:
        pass  # 旧バージョンは手動更新のみ

    # ── イベント接続 ───────────────────────────────────────────
    all_slot_inputs = [c for trio in slot_inputs for c in trio]

    start_btn.click(
        fn=_start_review,
        inputs=[limit_nb, timeout_nb, low_q_cb, *all_slot_inputs],
        outputs=[action_status],
    )
    stop_btn.click(fn=_stop_review, outputs=[action_status])
    refresh_btn.click(fn=_refresh_all, outputs=[status_md, task_df])
