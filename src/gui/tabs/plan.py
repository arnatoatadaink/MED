"""src/gui/tabs/plan.py — P4a プランビューア & P4b 実行コントロール。

P4a: 過去サイクルの詳細プランを run-id ドロップダウンで閲覧。
P4b: プロバイダー選択 + "Run Cycle" ボタン（DB バック ロック付き）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from src.gui.utils import get_all_provider_choices

_DB_PATH = Path("data/metadata.db")
_LOCK_WINDOW_MIN = 30  # これ以内に running なら二重起動ブロック

log = logging.getLogger(__name__)


# ── SQLite ヘルパー ─────────────────────────────────────────────

def _query(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    """metadata.db を読み取り専用でクエリする。"""
    if not _DB_PATH.exists():
        return []
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


# ── P4a: プランデータ取得 ───────────────────────────────────────

def _get_run_choices() -> list[str]:
    """直近 50 run の run_id リストを返す。最新が先頭。"""
    rows = _query(
        "SELECT run_id, started_at, status FROM cycle_runs "
        "ORDER BY started_at DESC LIMIT 50"
    )
    return [
        f"{r['run_id'][:12]}  [{r['status']}]  {r['started_at'][:16]}"
        for r in rows
    ] or ["(履歴なし)"]


def _run_id_from_choice(choice: str) -> str | None:
    """ドロップダウン表示文字列から run_id を復元する。"""
    if not choice or choice == "(履歴なし)":
        return None
    return choice.split()[0].replace("…", "")


def _get_plan_detail(choice: str) -> tuple[str, pd.DataFrame]:
    """選択 run のサマリー Markdown と詳細タスク DataFrame を返す。"""
    run_id_prefix = _run_id_from_choice(choice)
    if not run_id_prefix:
        return "_run を選択してください。_", pd.DataFrame()

    runs = _query(
        "SELECT * FROM cycle_runs WHERE run_id LIKE ?",
        (run_id_prefix + "%",),
    )
    if not runs:
        return "_データなし。_", pd.DataFrame()

    r = runs[0]
    summary_md = (
        f"**Run ID**: `{r['run_id']}`\n\n"
        f"**Status**: {r['status']}  |  "
        f"**Started**: {(r['started_at'] or '')[:19]}  |  "
        f"**Finished**: {(r['finished_at'] or '—')[:19]}\n\n"
        f"**Summary**: {r.get('summary') or '—'}"
    )

    tasks = _query(
        "SELECT gap_type, priority, status, reason, keywords, queries, signals "
        "FROM cycle_tasks WHERE run_id=? ORDER BY priority DESC",
        (r["run_id"],),
    )
    if not tasks:
        return summary_md, pd.DataFrame(
            columns=["gap_type", "priority", "status", "reason", "keywords", "queries", "signals"]
        )

    rows = []
    for t in tasks:
        try:
            kw = ", ".join(json.loads(t["keywords"])) if t["keywords"] else ""
        except Exception:
            kw = t["keywords"] or ""
        try:
            qs = " | ".join(json.loads(t["queries"])) if t["queries"] else ""
        except Exception:
            qs = t["queries"] or ""
        try:
            sig = json.dumps(json.loads(t["signals"]), ensure_ascii=False) if t["signals"] else ""
        except Exception:
            sig = t["signals"] or ""
        rows.append({
            "gap_type": t["gap_type"],
            "priority": round(float(t["priority"] or 0), 3),
            "status": t["status"],
            "reason": t["reason"] or "",
            "keywords": kw,
            "queries": qs,
            "signals": sig,
        })

    return summary_md, pd.DataFrame(rows)


# ── P4b: サイクル起動ロジック ───────────────────────────────────

def _is_cycle_running() -> bool:
    """直近 LOCK_WINDOW_MIN 分以内に status='running' な run が存在するか。"""
    cutoff = datetime.fromtimestamp(
        datetime.now(timezone.utc).timestamp() - _LOCK_WINDOW_MIN * 60,
        tz=timezone.utc,
    ).strftime("%Y-%m-%d %H:%M:%S")
    rows = _query(
        "SELECT 1 FROM cycle_runs WHERE status='running' AND started_at >= ? LIMIT 1",
        (cutoff,),
    )
    return bool(rows)


def _run_cycle_background(provider: str, model: str) -> None:
    """別スレッドでサイクルを起動する。"""
    from src.cycle.orchestrator import Orchestrator, OrchestratorConfig

    cfg = OrchestratorConfig(
        provider=provider,
        model=model or None,
    )
    orch = Orchestrator(config=cfg)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        run_id = loop.run_until_complete(orch.run_cycle())
        log.info("Cycle complete — run_id=%s", run_id)
    except Exception:
        log.exception("Cycle error")
    finally:
        loop.close()


def _trigger_cycle(provider: str, model: str) -> str:
    """Run Cycle ボタンのハンドラー。ステータス文字列を返す。"""
    p = (provider or "fastflowlm").strip()
    if p.startswith("auto"):
        p = "fastflowlm"

    if _is_cycle_running():
        return f"⚠️ サイクルが既に実行中です（直近 {_LOCK_WINDOW_MIN} 分以内に running あり）。"

    t = threading.Thread(
        target=_run_cycle_background,
        args=(p, model or ""),
        daemon=True,
        name="cycle-runner",
    )
    t.start()
    return f"🚀 サイクル起動しました（provider={p}）。サイクルタブで進捗を確認してください。"


# ── タブ構築 ───────────────────────────────────────────────────

def build_tab() -> tuple[gr.Dropdown, gr.Textbox]:
    """プランビューア & 実行コントロールタブを構築する。

    Returns:
        (provider_dd, model_tb) — localStorage 復元用に app.py へ渡す。
    """

    # ── P4b: 実行コントロール ──────────────────────────────────
    with gr.Accordion("▶ サイクル実行", open=True):
        with gr.Row():
            provider_dd = gr.Dropdown(
                choices=get_all_provider_choices(),
                value="fastflowlm",
                label="Provider",
                scale=2,
                elem_id="med-plan-provider",
            )
            model_tb = gr.Textbox(
                label="Model (空欄=デフォルト)",
                placeholder="例: gemini-2.0-flash",
                scale=3,
                elem_id="med-plan-model",
            )
            run_btn = gr.Button("▶ Run Cycle", variant="primary", scale=1)
        trigger_status = gr.Markdown("_ここにステータスが表示されます。_")

        run_btn.click(
            fn=_trigger_cycle,
            inputs=[provider_dd, model_tb],
            outputs=[trigger_status],
        )

    # localStorage 保存（Dropdown: change、Textbox: blur）
    provider_dd.change(
        fn=None,
        inputs=[provider_dd],
        js="(v) => { localStorage.setItem('med-plan-provider', v ?? ''); }",
    )
    model_tb.blur(
        fn=None,
        inputs=[model_tb],
        js="(v) => { localStorage.setItem('med-plan-model', v ?? ''); }",
    )

    gr.Markdown("---")

    # ── P4a: プランビューア ────────────────────────────────────
    gr.Markdown("#### 📋 サイクル履歴ビューア")
    with gr.Row():
        choices = _get_run_choices()
        run_dd = gr.Dropdown(
            choices=choices,
            value=choices[0] if choices else None,
            label="Run ID を選択",
            scale=4,
        )
        refresh_dd_btn = gr.Button("⟳ リスト更新", size="sm", scale=1, min_width=90, variant="secondary")

    plan_summary = gr.Markdown(_get_plan_detail(choices[0] if choices else "")[0])
    plan_table = gr.Dataframe(
        value=_get_plan_detail(choices[0] if choices else "")[1],
        interactive=False,
        wrap=True,
        label="タスク詳細（keywords / queries / signals 全文）",
    )

    def _on_run_select(choice: str) -> tuple[str, pd.DataFrame]:
        return _get_plan_detail(choice)

    def _refresh_dd() -> "gr.update":
        new_choices = _get_run_choices()
        return gr.update(choices=new_choices, value=new_choices[0] if new_choices else None)

    run_dd.change(
        fn=_on_run_select,
        inputs=[run_dd],
        outputs=[plan_summary, plan_table],
    )
    refresh_dd_btn.click(
        fn=_refresh_dd,
        outputs=[run_dd],
    )

    return provider_dd, model_tb
