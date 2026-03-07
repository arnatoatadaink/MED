"""src/gui/tabs/memory.py — FAISSメモリビュータブ。

ドメイン別インデックスの統計表示、ドキュメント検索・追加・削除を提供。
memory_manager が実装済みの場合は直接呼び出し、未実装の場合はAPIまたはモックを使用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import gradio as gr
import httpx

_ORCHESTRATOR_URL = "http://localhost:8000"
_FAISS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "faiss_indices"


def _is_api_alive() -> bool:
    try:
        r = httpx.get(f"{_ORCHESTRATOR_URL}/health", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False


# ────────────────────────────────────────────────────────────────
# データ取得ヘルパー
# ────────────────────────────────────────────────────────────────

def _get_memory_stats() -> dict:
    """ドメイン別メモリ統計を取得。"""
    if _is_api_alive():
        try:
            r = httpx.get(f"{_ORCHESTRATOR_URL}/memory/stats", timeout=5.0)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass

    # ローカルファイルから推定 / モック
    domains = ["code", "academic", "general"]
    stats = {}
    for domain in domains:
        idx_path = _FAISS_DIR / domain
        count = 0
        if idx_path.exists():
            index_file = idx_path / "index.faiss"
            count = index_file.stat().st_size // 768 // 4 if index_file.exists() else 0
        stats[domain] = {
            "doc_count": count,
            "index_type": "IndexFlatIP" if count < 10000 else "IndexIVFFlat",
            "avg_confidence": 0.0,
            "status": "active" if count > 0 else "empty",
        }
    return {"domains": stats, "api_connected": False}


def _search_memory(query: str, domain: str, top_k: int) -> list:
    """メモリ検索。"""
    if _is_api_alive():
        try:
            r = httpx.post(
                f"{_ORCHESTRATOR_URL}/memory/search",
                json={"query": query, "domain": domain, "top_k": top_k},
                timeout=10.0,
            )
            if r.status_code == 200:
                return r.json().get("results", [])
        except Exception:
            pass
    return [
        {
            "id": f"mock-{i}",
            "content": f"[モック] '{query}' に関連するドキュメント {i + 1}",
            "score": round(0.95 - i * 0.05, 3),
            "domain": domain,
            "source": "mock",
        }
        for i in range(min(top_k, 3))
    ]


def _add_document(content: str, domain: str, source: str) -> dict:
    """ドキュメント追加。"""
    if _is_api_alive():
        try:
            r = httpx.post(
                f"{_ORCHESTRATOR_URL}/memory/add",
                json={"content": content, "domain": domain, "source": source},
                timeout=10.0,
            )
            return r.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    return {
        "success": False,
        "error": "APIオフライン — ドキュメントは追加されませんでした（モード）",
        "doc_id": None,
    }


# ────────────────────────────────────────────────────────────────
# UI ヘルパー
# ────────────────────────────────────────────────────────────────

def _stats_to_df(stats: dict) -> list[list]:
    rows = []
    for domain, info in stats.get("domains", {}).items():
        rows.append([
            domain,
            info.get("doc_count", 0),
            info.get("index_type", "—"),
            f"{info.get('avg_confidence', 0.0):.3f}",
            info.get("status", "unknown"),
        ])
    return rows


def _results_to_md(results: list) -> str:
    if not results:
        return "_結果なし_"
    lines = []
    for r in results:
        score = r.get("score", 0.0)
        content = r.get("content", "")[:200]
        domain = r.get("domain", "")
        source = r.get("source", "")
        lines.append(
            f"**スコア {score:.3f}** | ドメイン: `{domain}` | ソース: `{source}`\n\n"
            f"{content}\n\n---"
        )
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────
# タブ UI 構築
# ────────────────────────────────────────────────────────────────

def build_tab() -> None:
    """Gradio Blocks コンテキスト内でメモリタブを描画する。"""

    with gr.Tabs():

        # ── 統計ビュー ──────────────────────────────────────────
        with gr.TabItem("統計"):
            gr.Markdown("### ドメイン別インデックス統計")
            stats_table = gr.Dataframe(
                headers=["ドメイン", "ドキュメント数", "インデックス型", "平均信頼度", "状態"],
                datatype=["str", "number", "str", "str", "str"],
                interactive=False,
                label="FAISSインデックス状態",
            )
            api_status = gr.Markdown()
            refresh_btn = gr.Button("更新", variant="secondary")

            def _refresh():
                stats = _get_memory_stats()
                df = _stats_to_df(stats)
                connected = stats.get("api_connected", False)
                status_md = (
                    "🟢 **APIオンライン** — リアルデータ"
                    if connected
                    else "🔴 **APIオフライン** — ローカル推定値"
                )
                return df, status_md

            refresh_btn.click(fn=_refresh, outputs=[stats_table, api_status])
            # 初期ロード
            stats_table.value = _stats_to_df(_get_memory_stats())

        # ── 検索 ────────────────────────────────────────────────
        with gr.TabItem("検索"):
            gr.Markdown("### FAISSメモリ検索")
            with gr.Row():
                search_query = gr.Textbox(label="検索クエリ", placeholder="検索したい内容を入力…")
                search_domain = gr.Dropdown(
                    choices=["all", "code", "academic", "general"],
                    value="all",
                    label="ドメイン",
                )
                top_k_slider = gr.Slider(1, 20, value=5, step=1, label="Top-K")
            search_btn = gr.Button("検索", variant="primary")
            search_results = gr.Markdown("_検索結果がここに表示されます_")

            def _do_search(query, domain, top_k):
                if not query.strip():
                    return "_クエリを入力してください_"
                results = _search_memory(query, domain, int(top_k))
                return _results_to_md(results)

            search_btn.click(
                fn=_do_search,
                inputs=[search_query, search_domain, top_k_slider],
                outputs=[search_results],
            )
            search_query.submit(
                fn=_do_search,
                inputs=[search_query, search_domain, top_k_slider],
                outputs=[search_results],
            )

        # ── ドキュメント追加 ────────────────────────────────────
        with gr.TabItem("追加"):
            gr.Markdown("### ドキュメントをメモリに追加")
            add_content = gr.Textbox(
                label="コンテンツ",
                placeholder="追加するテキストを貼り付けてください…",
                lines=8,
            )
            with gr.Row():
                add_domain = gr.Dropdown(
                    choices=["code", "academic", "general"],
                    value="code",
                    label="ドメイン",
                )
                add_source = gr.Textbox(label="ソースURL / 識別子", placeholder="https://…")
            add_btn = gr.Button("追加", variant="primary")
            add_result = gr.Markdown()

            def _do_add(content, domain, source):
                if not content.strip():
                    return "❌ コンテンツが空です"
                result = _add_document(content, domain, source or "manual")
                if result.get("success"):
                    doc_id = result.get("doc_id", "—")
                    return f"✅ 追加成功 — doc_id: `{doc_id}`"
                else:
                    return f"❌ {result.get('error', '不明なエラー')}"

            add_btn.click(
                fn=_do_add,
                inputs=[add_content, add_domain, add_source],
                outputs=[add_result],
            )
