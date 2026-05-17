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

_SOURCES = ["github", "stackoverflow", "tavily", "arxiv", "openreview"]
_SOURCE_LABELS = {
    "github": "GitHub",
    "stackoverflow": "StackOverflow",
    "tavily": "Tavily",
    "arxiv": "arXiv",
    "openreview": "OpenReview",
}
_SOURCE_DEFAULTS = {
    "github": True,
    "stackoverflow": True,
    "tavily": True,
    "arxiv": False,   # BAN 中 — 5/17 以降に解除確認してから ON にする
    "openreview": True,
}

log = logging.getLogger(__name__)

# 遷移検知: {run_id: prev_status}
_last_run_state: dict[str, str] = {}
# ポーリング一時停止フラグ
_polling_paused: bool = False


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


# ── P4c: ポーリングヘルパー ────────────────────────────────────

def _get_latest_run() -> dict[str, Any] | None:
    """最新の cycle_runs レコードを返す。"""
    rows = _query(
        "SELECT run_id, status, started_at, finished_at, summary "
        "FROM cycle_runs ORDER BY started_at DESC LIMIT 1"
    )
    return rows[0] if rows else None


def _calc_eta(run: dict[str, Any]) -> str:
    """実行中 run の終了見込み時間を返す。running 以外は空文字。"""
    if run["status"] != "running":
        return ""
    counts = _query(
        "SELECT status, COUNT(*) as n FROM cycle_tasks WHERE run_id=? GROUP BY status",
        (run["run_id"],),
    )
    by_status = {r["status"]: r["n"] for r in counts}
    total = sum(by_status.values())
    done = by_status.get("done", 0)
    if total == 0:
        return ""
    started = run.get("started_at") or ""
    try:
        started_dt = datetime.fromisoformat(started)
        if started_dt.tzinfo is None:
            started_dt = started_dt.replace(tzinfo=timezone.utc)
        elapsed = (datetime.now(timezone.utc) - started_dt).total_seconds()
        if done > 0:
            remaining = total - done
            eta_secs = int(elapsed / done * remaining)
            eta_m, eta_s = divmod(eta_secs, 60)
            return f"  ETA: 残り {eta_m}m {eta_s}s ({done}/{total} 完了)"
        return f"  ETA: 計算中 ({done}/{total} 完了)"
    except Exception:
        return ""


def _format_status_md(run: dict[str, Any], eta: str = "") -> str:
    """run レコードをステータス Markdown に整形する。"""
    status = run["status"]
    icon = {"running": "🔄", "done": "✅", "error": "❌"}.get(status, "❓")
    started = run.get("started_at") or ""
    elapsed_str = ""
    if started:
        try:
            started_dt = datetime.fromisoformat(started)
            if started_dt.tzinfo is None:
                started_dt = started_dt.replace(tzinfo=timezone.utc)
            ref_dt = (
                datetime.fromisoformat(run["finished_at"])
                if run.get("finished_at")
                else datetime.now(timezone.utc)
            )
            if ref_dt.tzinfo is None:
                ref_dt = ref_dt.replace(tzinfo=timezone.utc)
            secs = int((ref_dt - started_dt).total_seconds())
            elapsed_str = f"  ⏱ {secs // 60}m {secs % 60}s"
        except Exception:
            pass
    run_id_short = run["run_id"][:12]
    summary = run.get("summary") or "—"
    return (
        f"{icon} **{status.upper()}** `{run_id_short}`{elapsed_str}{eta}  \n"
        f"Summary: {summary}"
    )


def _toggle_polling() -> str:
    """ポーリング一時停止/再開トグル。ボタン表示ラベルを返す。"""
    global _polling_paused
    _polling_paused = not _polling_paused
    return "▶ 再開" if _polling_paused else "⏸ 停止"


def _poll_cycle_status() -> tuple[str, Any, Any, Any]:
    """Timer コールバック: status_md と（遷移時のみ）ドロップダウンを更新する。

    _polling_paused が True のときは全コンポーネントを no-op で返す。
    running→done/error 遷移を検知したときだけ run_dd の choices を再ロードする。
    value= を指定しないためユーザーの選択は保持される。
    """
    if _polling_paused:
        return gr.update(), gr.update(), gr.update(), gr.update()

    run = _get_latest_run()
    if run is None:
        return "_サイクル未実行。_", gr.update(), gr.update(), gr.update()

    run_id = run["run_id"]
    status = run["status"]
    prev = _last_run_state.get(run_id)
    _last_run_state[run_id] = status

    eta = _calc_eta(run)
    status_md = _format_status_md(run, eta)

    if prev == "running" and status in ("done", "error"):
        new_choices = _get_run_choices()
        note = "\n\n_↑ 完了しました。ドロップダウンから最新 run を選択してください。_"
        return (
            status_md + note,
            gr.update(choices=new_choices),
            gr.update(),
            gr.update(),
        )

    return status_md, gr.update(), gr.update(), gr.update()


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


def _run_cycle_background(
    provider: str, model: str, disabled_sources: frozenset[str]
) -> None:
    """別スレッドでサイクルを起動する。"""
    from src.cycle.orchestrator import Orchestrator, OrchestratorConfig

    cfg = OrchestratorConfig(
        provider=provider,
        model=model or None,
        disabled_sources=set(disabled_sources),
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


def _trigger_cycle(
    provider: str,
    model: str,
    src_github: bool,
    src_stackoverflow: bool,
    src_tavily: bool,
    src_arxiv: bool,
    src_openreview: bool,
) -> str:
    """Run Cycle ボタンのハンドラー。ステータス文字列を返す。"""
    p = (provider or "fastflowlm").strip()
    if p.startswith("auto"):
        p = "fastflowlm"

    if _is_cycle_running():
        return f"⚠️ サイクルが既に実行中です（直近 {_LOCK_WINDOW_MIN} 分以内に running あり）。"

    src_flags = {
        "github": src_github,
        "stackoverflow": src_stackoverflow,
        "tavily": src_tavily,
        "arxiv": src_arxiv,
        "openreview": src_openreview,
    }
    disabled = frozenset(s for s, enabled in src_flags.items() if not enabled)
    disabled_label = ", ".join(sorted(disabled)) if disabled else "なし"

    t = threading.Thread(
        target=_run_cycle_background,
        args=(p, model or "", disabled),
        daemon=True,
        name="cycle-runner",
    )
    t.start()
    return (
        f"🚀 サイクル起動しました（provider={p}）。"
        f"無効ソース: {disabled_label}。"
        "サイクルタブで進捗を確認してください。"
    )


# ── タブ構築 ───────────────────────────────────────────────────

def build_tab() -> tuple[gr.Dropdown, gr.Textbox, dict[str, gr.Checkbox]]:
    """プランビューア & 実行コントロールタブを構築する。

    Returns:
        (provider_dd, model_tb, src_checks) — localStorage 復元用に app.py へ渡す。
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

        # ── ソース設定 ─────────────────────────────────────────
        with gr.Accordion("🗂️ ソース設定", open=True):
            gr.Markdown(
                "取得するソースを選択します。"
                "**arXiv は BAN 中 — 5/17 以降に解除確認してから ON にしてください。**"
            )
            with gr.Row():
                src_checks: dict[str, gr.Checkbox] = {}
                for src in _SOURCES:
                    src_checks[src] = gr.Checkbox(
                        label=_SOURCE_LABELS[src],
                        value=_SOURCE_DEFAULTS[src],
                        scale=1,
                        elem_id=f"med-plan-src-{src}",
                    )

        trigger_status = gr.Markdown("_ここにステータスが表示されます。_")

        run_btn.click(
            fn=_trigger_cycle,
            inputs=[provider_dd, model_tb, *src_checks.values()],
            outputs=[trigger_status],
        )
        with gr.Row():
            poll_toggle_btn = gr.Button(
                "⏸ 停止", size="sm", min_width=80, scale=0, variant="secondary"
            )
            with gr.Column(scale=10):
                cycle_status_md = gr.Markdown("_ポーリング待機中…_")

    # localStorage 保存（Dropdown: change、Textbox: blur、Checkbox: change）
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
    for src, chk in src_checks.items():
        chk.change(
            fn=None,
            inputs=[chk],
            js=f"(v) => {{ localStorage.setItem('med-plan-src-{src}', JSON.stringify(v)); }}",
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

    # ── ポーリング Timer（Gradio 5+ のみ）─────────────────────
    try:
        timer = gr.Timer(value=5)
        timer.tick(
            fn=_poll_cycle_status,
            outputs=[cycle_status_md, run_dd, plan_summary, plan_table],
        )
        poll_toggle_btn.click(fn=_toggle_polling, outputs=[poll_toggle_btn])
    except AttributeError:
        log.warning("gr.Timer not available — plan tab polling disabled")

    return provider_dd, model_tb, src_checks
