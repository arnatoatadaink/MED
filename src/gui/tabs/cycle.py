"""src/gui/tabs/cycle.py — サイクルモニタリングタブ (P3)

Gap Detection → Enrich → Dispatch サイクルの状態を
読み取り専用で可視化する。実行は scripts/run_cycle.py から行う。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

_DB_PATH    = Path("data/metadata.db")
_CACHE_PATH = Path("data/umap_cache.npz")
_MAX_PLOT   = 8_000  # scatter に描画する最大点数

_STATUS_COLORS = {
    "approved":  "#4caf50",
    "rejected":  "#f44336",
    "unreviewed": "#9e9e9e",
    "needs_update": "#ff9800",
    "orphaned":  "#607d8b",
}

_GAP_COLORS = {
    "small_cluster":      "#42a5f5",
    "unreviewed_backlog": "#ff7043",
    "source_imbalance":   "#ab47bc",
    "low_quality":        "#ef5350",
}


# ── SQLite 直読み ──────────────────────────────────────────────

def _query(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    """metadata.db を読み取り専用で直接クエリする。"""
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


# ── サマリー ───────────────────────────────────────────────────

def _get_summary() -> str:
    """最新サイクル実行の概要を Markdown で返す。"""
    runs = _query(
        "SELECT * FROM cycle_runs ORDER BY started_at DESC LIMIT 1"
    )
    if not runs:
        return "_サイクル実行履歴なし。`poetry run python scripts/run_cycle.py` で初回実行してください。_"

    r = runs[0]
    task_rows = _query(
        "SELECT status, COUNT(*) as cnt FROM cycle_tasks WHERE run_id=? GROUP BY status",
        (r["run_id"],),
    )
    task_summary = "  ".join(f"**{t['status']}**: {t['cnt']}" for t in task_rows) or "—"

    cache_info = ""
    if _CACHE_PATH.exists():
        import time
        age_h = (time.time() - _CACHE_PATH.stat().st_mtime) / 3600
        cache_info = f"  |  UMAP キャッシュ: **{age_h:.1f}h** 前"

    finished = r["finished_at"] or "—"
    return (
        f"最終実行: `{r['run_id'][:8]}…`  "
        f"状態: **{r['status']}**  "
        f"開始: {r['started_at'][:19]}  "
        f"終了: {finished[:19] if finished != '—' else '—'}"
        f"{cache_info}\n\n"
        f"タスク: {task_summary}"
    )


# ── UMAP 散布図 ────────────────────────────────────────────────

def _build_plot() -> Any:
    """UMAP キャッシュから island 着色散布図を返す。"""
    try:
        import plotly.express as px
        from src.cycle.umap_islands import compute_islands
    except ImportError:
        return None

    iset = compute_islands(cache_path=_CACHE_PATH)
    if iset.n_docs == 0:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.update_layout(
            title="UMAP データなし — umap_analysis.py --save-cache を実行してください",
            template="plotly_dark",
            height=420,
        )
        return fig

    # island ラベル配列を構築
    n = iset.n_docs
    island_label = ["noise"] * n
    id_map: dict[str, int] = {}
    for island in iset.islands:
        for did in island.doc_ids:
            id_map[did] = island.id
    for i, did in enumerate(iset.doc_ids):
        if did in id_map:
            island_label[i] = f"Island #{id_map[did]}"

    # gap タスクがあるか確認（最新 run）
    latest_runs = _query(
        "SELECT run_id FROM cycle_runs WHERE status='done' ORDER BY finished_at DESC LIMIT 1"
    )
    gap_map: dict[int, str] = {}  # island_id → gap_type
    if latest_runs:
        tasks = _query(
            "SELECT signals, gap_type FROM cycle_tasks WHERE run_id=?",
            (latest_runs[0]["run_id"],),
        )
        import json
        for t in tasks:
            try:
                sig = json.loads(t["signals"])
                gap_map[sig["cluster_id"]] = t["gap_type"]
            except Exception:
                pass

    emb = iset.embedding
    xs, ys = emb[:, 0].tolist(), emb[:, 1].tolist()

    # 点が多すぎる場合はサブサンプル
    idx_list = list(range(n))
    if n > _MAX_PLOT:
        import random
        random.seed(42)
        idx_list = sorted(random.sample(idx_list, _MAX_PLOT))

    rows = []
    for i in idx_list:
        did = iset.doc_ids[i]
        iid = id_map.get(did, -1)
        gap = gap_map.get(iid, "") if iid >= 0 else ""
        rows.append({
            "x":             xs[i],
            "y":             ys[i],
            "island":        island_label[i],
            "review_status": iset.review_status[i],
            "source_type":   iset.source_type[i],
            "quality":       round(iset.quality[i], 2),
            "gap_type":      gap or "—",
            "doc_id":        did[:12],
        })

    df = pd.DataFrame(rows)
    stale_note = " ⚠️ stale" if iset.stale else ""
    fig = px.scatter(
        df, x="x", y="y",
        color="review_status",
        color_discrete_map=_STATUS_COLORS,
        hover_data={
            "x": False, "y": False,
            "doc_id": True, "island": True,
            "source_type": True, "quality": True, "gap_type": True,
        },
        title=(
            f"UMAP Islands — {iset.n_docs:,} docs / "
            f"{iset.n_clusters} islands / "
            f"noise {iset.noise_count}{stale_note}"
        ),
        template="plotly_dark",
        opacity=0.55,
    )
    fig.update_traces(marker_size=3)
    fig.update_layout(height=440, margin=dict(l=20, r=20, t=40, b=20))
    return fig


# ── テーブルデータ ──────────────────────────────────────────────

def _get_runs() -> pd.DataFrame:
    rows = _query(
        "SELECT run_id, started_at, finished_at, status, summary "
        "FROM cycle_runs ORDER BY started_at DESC LIMIT 20"
    )
    if not rows:
        return pd.DataFrame(columns=["run_id", "started_at", "status", "summary"])
    df = pd.DataFrame(rows)
    df["run_id"] = df["run_id"].str[:12]
    df["started_at"] = df["started_at"].str[:19]
    df["finished_at"] = df["finished_at"].fillna("—").str[:19]
    return df[["run_id", "started_at", "finished_at", "status", "summary"]]


def _get_tasks() -> pd.DataFrame:
    """最新サイクルのタスク一覧を返す。"""
    latest = _query(
        "SELECT run_id FROM cycle_runs ORDER BY started_at DESC LIMIT 1"
    )
    if not latest:
        return pd.DataFrame(columns=["gap_type", "priority", "status", "reason"])
    run_id = latest[0]["run_id"]
    rows = _query(
        "SELECT gap_type, priority, status, reason, keywords "
        "FROM cycle_tasks WHERE run_id=? ORDER BY priority DESC",
        (run_id,),
    )
    if not rows:
        return pd.DataFrame(columns=["gap_type", "priority", "status", "reason"])
    df = pd.DataFrame(rows)
    df["priority"] = df["priority"].round(3)
    df["reason"] = df["reason"].str[:80]
    import json
    df["keywords"] = df["keywords"].apply(
        lambda v: ", ".join(json.loads(v)[:4]) if v else ""
    )
    return df[["gap_type", "priority", "status", "reason", "keywords"]]


def _refresh_all():
    return _get_summary(), _build_plot(), _get_runs(), _get_tasks()


# ── タブ構築 ───────────────────────────────────────────────────

def build_tab() -> None:
    """サイクルモニタリングタブを構築する。"""
    with gr.Row():
        summary_md = gr.Markdown(_get_summary())
        refresh_btn = gr.Button("⟳ 更新", size="sm", scale=0, min_width=80, variant="secondary")

    with gr.Row():
        plot_out = gr.Plot(value=_build_plot(), label="UMAP Islands")

    gr.Markdown("#### サイクル実行履歴")
    runs_table = gr.Dataframe(
        value=_get_runs(),
        interactive=False,
        wrap=False,
    )

    gr.Markdown("#### 最新サイクル タスク一覧")
    tasks_table = gr.Dataframe(
        value=_get_tasks(),
        interactive=False,
        wrap=False,
    )

    refresh_btn.click(
        fn=_refresh_all,
        outputs=[summary_md, plot_out, runs_table, tasks_table],
    )
