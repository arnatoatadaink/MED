"""src/gui/tabs/memory.py — FAISSメモリビュータブ。

ドメイン別インデックスの統計表示、ドキュメント検索・追加・削除を提供。
Phase 2 成熟管理（品質レポート・Teacher信頼度・一括審査）を含む。
memory_manager が実装済みの場合は直接呼び出し、未実装の場合はAPIまたはモックを使用。
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import httpx

from src.gui.utils import ORCHESTRATOR_URL, is_api_alive

_FAISS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "faiss_indices"


# ────────────────────────────────────────────────────────────────
# データ取得ヘルパー
# ────────────────────────────────────────────────────────────────

def _get_memory_stats() -> dict:
    """ドメイン別メモリ統計を取得。"""
    if is_api_alive():
        try:
            r = httpx.get(f"{ORCHESTRATOR_URL}/memory/stats", timeout=5.0)
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
    if is_api_alive():
        try:
            r = httpx.post(
                f"{ORCHESTRATOR_URL}/memory/search",
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


# ────────────────────────────────────────────────────────────────
# Phase 2 成熟管理ヘルパー
# ────────────────────────────────────────────────────────────────

def _get_quality_report(domain: str | None = None) -> dict:
    """Phase 2 品質レポートを取得。"""
    if is_api_alive():
        try:
            params = {"domain": domain} if domain and domain != "all" else {}
            r = httpx.get(f"{ORCHESTRATOR_URL}/maturation/quality", params=params, timeout=10.0)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass

    # モックデータ
    return {
        "total_docs": 0, "approved_docs": 0, "rejected_docs": 0, "pending_docs": 0,
        "avg_confidence": 0.0, "avg_teacher_quality": 0.0, "avg_composite_score": 0.0,
        "exec_success_rate": 0.0, "avg_retrieval_count": 0.0,
        "difficulty_distribution": {},
        "approval_rate": 0.0, "meets_phase2_goal": False,
        "phase2_progress": {"docs": 0.0, "confidence": 0.0, "exec_success": 0.0},
        "doc_target": 10000, "confidence_target": 0.7, "exec_success_target": 0.8,
        "_mock": True,
    }


def _get_teachers() -> dict:
    """Teacher 信頼度プロファイルを取得。"""
    if is_api_alive():
        try:
            r = httpx.get(f"{ORCHESTRATOR_URL}/maturation/teachers", timeout=5.0)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
    return {"teachers": [], "_mock": True}


def _run_review(limit: int, concurrency: int) -> dict:
    """未審査ドキュメントを一括審査する。"""
    if is_api_alive():
        try:
            r = httpx.post(
                f"{ORCHESTRATOR_URL}/maturation/review",
                json={"limit": limit, "concurrency": concurrency},
                timeout=120.0,
            )
            if r.status_code == 200:
                return r.json()
            return {"error": r.text}
        except Exception as e:
            return {"error": str(e)}
    return {"error": "APIオフライン — オーケストレーターを起動してください"}


def _quality_report_to_md(r: dict) -> str:
    """品質レポートを Markdown テーブルに変換。"""
    if r.get("_mock"):
        prefix = "> 🔴 **APIオフライン** — モックデータを表示\n\n"
    else:
        prefix = "> 🟢 **APIオンライン** — リアルデータ\n\n"

    prog = r.get("phase2_progress", {})

    def _bar(ratio: float, width: int = 20) -> str:
        filled = int(ratio * width)
        return "█" * filled + "░" * (width - filled) + f" {ratio*100:.0f}%"

    goal_icon = "✅" if r.get("meets_phase2_goal") else "🔄"

    lines = [
        prefix,
        f"### {goal_icon} Phase 2 目標進捗\n",
        f"| 目標 | 現在値 | 目標値 | 進捗 |",
        f"|------|--------|--------|------|",
        f"| ドキュメント数 | `{r['total_docs']:,}` | `{r['doc_target']:,}` "
        f"| `{_bar(prog.get('docs', 0))}` |",
        f"| 平均信頼度 | `{r['avg_confidence']:.3f}` | `{r['confidence_target']}` "
        f"| `{_bar(prog.get('confidence', 0))}` |",
        f"| 実行成功率 | `{r['exec_success_rate']:.1%}` | `{r['exec_success_target']:.0%}` "
        f"| `{_bar(prog.get('exec_success', 0))}` |",
        "",
        f"### レビュー状況\n",
        f"| 承認 | 却下 | 未審査 | 承認率 |",
        f"|------|------|--------|--------|",
        f"| `{r['approved_docs']:,}` | `{r['rejected_docs']:,}` "
        f"| `{r['pending_docs']:,}` | `{r.get('approval_rate', 0):.1%}` |",
        "",
        f"| 平均品質スコア | `{r['avg_teacher_quality']:.3f}` |",
        f"|------|------|",
        f"| 平均複合スコア | `{r['avg_composite_score']:.3f}` |",
        f"| 平均検索回数 | `{r['avg_retrieval_count']:.1f}` |",
    ]

    diff = r.get("difficulty_distribution", {})
    if diff:
        lines += ["", "### 難易度分布\n",
                  "| 難易度 | 件数 |", "|--------|------|"]
        for k, v in sorted(diff.items()):
            lines.append(f"| `{k}` | `{v:,}` |")

    return "\n".join(lines)


def _teachers_to_rows(data: dict) -> list[list]:
    rows = []
    for t in data.get("teachers", []):
        rows.append([
            t.get("teacher_id", "—"),
            t.get("provider", "—"),
            f"{t.get('trust_score', 0):.4f}",
            t.get("total_docs", 0),
            f"{t.get('avg_reward', 0):.4f}",
            t.get("n_feedback", 0),
            (t.get("updated_at") or "—")[:19],
        ])
    return rows


def _add_document(content: str, domain: str, source: str) -> dict:
    """ドキュメント追加。"""
    if is_api_alive():
        try:
            r = httpx.post(
                f"{ORCHESTRATOR_URL}/memory/add",
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
            _initial_stats = _get_memory_stats()
            stats_table = gr.Dataframe(
                value=_stats_to_df(_initial_stats),
                headers=["ドメイン", "ドキュメント数", "インデックス型", "平均信頼度", "状態"],
                datatype=["str", "number", "str", "str", "str"],
                interactive=False,
                label="FAISSインデックス状態",
            )
            _init_connected = _initial_stats.get("api_connected", False)
            api_status = gr.Markdown(
                "🟢 **APIオンライン** — リアルデータ"
                if _init_connected
                else "🔴 **APIオフライン** — ローカル推定値"
            )
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

        # ── Phase 2 成熟管理 ────────────────────────────────────
        with gr.TabItem("🔬 成熟管理 (Phase 2)"):

            with gr.Tabs():

                # ── 品質レポート ─────────────────────────────────
                with gr.TabItem("品質レポート"):
                    gr.Markdown(
                        "### Phase 2 メモリ品質レポート\n"
                        "目標: **10,000 docs / 信頼度 ≥ 0.7 / 実行成功率 ≥ 80%**"
                    )
                    with gr.Row():
                        qr_domain = gr.Dropdown(
                            choices=["all", "code", "academic", "general"],
                            value="all",
                            label="ドメインフィルタ",
                            scale=2,
                        )
                        qr_refresh_btn = gr.Button("更新", variant="secondary", scale=1)
                    qr_md = gr.Markdown(_quality_report_to_md(_get_quality_report()))

                    def _refresh_quality(domain):
                        d = domain if domain != "all" else None
                        return _quality_report_to_md(_get_quality_report(d))

                    qr_refresh_btn.click(fn=_refresh_quality, inputs=[qr_domain], outputs=[qr_md])
                    qr_domain.change(fn=_refresh_quality, inputs=[qr_domain], outputs=[qr_md])

                # ── Teacher 信頼度 ───────────────────────────────
                with gr.TabItem("Teacher 信頼度"):
                    gr.Markdown(
                        "### Teacher 信頼度プロファイル\n"
                        "各 Teacher モデルの trust_score（EWMA で自動更新）。\n"
                        "スコアが高いほど検索結果で優遇されます。"
                    )
                    _init_teachers = _get_teachers()
                    teachers_table = gr.Dataframe(
                        value=_teachers_to_rows(_init_teachers),
                        headers=["teacher_id", "provider", "trust_score",
                                 "total_docs", "avg_reward", "n_feedback", "updated_at"],
                        datatype=["str", "str", "str", "number", "str", "number", "str"],
                        interactive=False,
                        label="Teacher プロファイル",
                    )
                    teacher_status = gr.Markdown(
                        "_モックデータ_" if _init_teachers.get("_mock") else
                        f"登録済み Teacher: **{len(_init_teachers.get('teachers', []))}** 件"
                    )
                    t_refresh_btn = gr.Button("更新", variant="secondary", size="sm")

                    def _refresh_teachers():
                        data = _get_teachers()
                        rows = _teachers_to_rows(data)
                        if data.get("_mock"):
                            status = "🔴 APIオフライン — モックデータ"
                        else:
                            n = len(data.get("teachers", []))
                            status = f"🟢 登録済み Teacher: **{n}** 件"
                        return rows, status

                    t_refresh_btn.click(fn=_refresh_teachers, outputs=[teachers_table, teacher_status])

                # ── 一括審査 ─────────────────────────────────────
                with gr.TabItem("一括審査"):
                    gr.Markdown(
                        "### 未審査ドキュメント一括審査\n"
                        "Teacher LLM が各ドキュメントを評価し、"
                        "`quality_score` / `confidence` / `review_status` を更新します。\n\n"
                        "> ⚠️ オーケストレーターが起動中かつ Teacher API キーが設定済みの場合のみ実行できます。"
                    )
                    with gr.Row():
                        review_limit = gr.Slider(
                            1, 200, value=50, step=1,
                            label="最大審査件数",
                            scale=3,
                        )
                        review_concurrency = gr.Slider(
                            1, 10, value=5, step=1,
                            label="並列数",
                            scale=1,
                        )
                    review_btn = gr.Button("審査実行", variant="primary")
                    review_result = gr.Markdown()

                    def _do_review(limit, concurrency):
                        result = _run_review(int(limit), int(concurrency))
                        if "error" in result:
                            return f"❌ {result['error']}"
                        return (
                            f"✅ **審査完了**\n\n"
                            f"| 項目 | 件数 |\n|------|------|\n"
                            f"| 審査数 | `{result.get('reviewed', 0)}` |\n"
                            f"| 承認 | `{result.get('approved', 0)}` |\n"
                            f"| 却下 | `{result.get('rejected', 0)}` |\n"
                            f"| 平均品質スコア | `{result.get('avg_quality', 0):.3f}` |\n"
                            f"| 平均信頼度 | `{result.get('avg_confidence', 0):.3f}` |"
                        )

                    review_btn.click(
                        fn=_do_review,
                        inputs=[review_limit, review_concurrency],
                        outputs=[review_result],
                    )
