"""src/cycle/query_prompts.py — LLM プロンプトビルダー (query_generator 用)

query_generator.py が 300 行を超えるため、プロンプト生成関数を分離。
"""

from __future__ import annotations

from src.cycle.schema import CollectionTask, GapType


def _build_prompt(task: CollectionTask, titles: list[str]) -> str:
    """CollectionTask から LLM 用プロンプトを組み立てる。"""
    sig = task.signals
    src_lines = "\n".join(
        f"  {src}: {cnt} docs"
        for src, cnt in sorted(
            sig.get("source_dist", {}).items(),
            key=lambda x: -x[1],
        )
    )
    title_lines = "\n".join(f"  - {t}" for t in titles) if titles else "  (none available)"

    # gap_type 別ヒント
    if task.gap_type == GapType.INTER_ISLAND_BRIDGE:
        id_a = sig.get("island_a", {}).get("cluster_id", "A")
        id_b = sig.get("island_b", {}).get("cluster_id", "B")
        gap_hint = (
            f"Find papers or implementations that bridge cluster #{id_a} and cluster #{id_b}. "
            "Look for cross-domain surveys, comparative studies, or tools that combine both areas."
        )
    else:
        gap_hint = {
            GapType.SMALL_CLUSTER: (
                "This topic is under-represented. Find follow-up work, derivative research, "
                "or implementations related to the sample titles above. "
                "Prefer queries like 'follow-up on X', 'X applied to Y', 'implementation of X'."
            ),
            GapType.UNREVIEWED_BACKLOG: (
                "This cluster has many unreviewed docs. Find high-quality sources."
            ),
            GapType.SOURCE_IMBALANCE: (
                f"The cluster relies too heavily on '{sig.get('dominant_source', 'unknown')}'. "
                "Suggest queries for other sources."
            ),
            GapType.LOW_QUALITY: (
                "Quality is low here. Prioritize authoritative, detailed sources."
            ),
        }.get(task.gap_type, "")

    # theory/impl バイアスの追加ヒント
    theory_pct = sig.get("theory_pct", 0.0)
    impl_pct = sig.get("impl_pct", 0.0)
    if isinstance(theory_pct, (int, float)) and theory_pct > 0.70 and impl_pct < 0.10:
        gap_hint += (
            " The cluster is heavily theory-focused (mostly arXiv). "
            "Also suggest GitHub repos, code implementations, operational notes, or tutorials."
        )

    return f"""You are a knowledge collection assistant for a machine learning memory system.
A gap was detected in the knowledge base. Your task: generate search keywords and queries
to find documents that would fill this gap.

Gap type: {task.gap_type.value}
Reason: {task.reason}
Cluster size: {sig.get('size', '?')} docs  quality_avg: {sig.get('q_avg', 0):.2f}
Approved: {sig.get('approved_pct', 0) * 100:.0f}%

Source distribution:
{src_lines}

Sample document titles (what already exists in this cluster):
{title_lines}

Hint: {gap_hint}

Generate 4-6 search keywords and 4-6 search queries in English.
Keywords should be concise terms (1-4 words each).
Queries should be natural language questions or search strings suitable for arXiv / GitHub / web search.

Respond ONLY with valid JSON, no explanation:
{{"keywords": ["...", "..."], "queries": ["...", "..."]}}"""


def _build_pivot_prompt(
    task: CollectionTask,
    titles: list[str],
    zero_result_queries: list[str],
) -> str:
    """0件クエリに基づくピボット用プロンプト。"""
    sig = task.signals
    zero_lines = "\n".join(f"  - {q}" for q in zero_result_queries[:6])
    title_lines = "\n".join(f"  - {t}" for t in titles) if titles else "  (none available)"

    return f"""You are a knowledge collection assistant for a machine learning memory system.
The following search queries returned zero results. Please generate alternative queries
with different angles, perspectives, or terminology.

Gap type: {task.gap_type.value}
Cluster size: {sig.get('size', '?')} docs  quality_avg: {sig.get('q_avg', 0):.2f}

Sample document titles (what exists in this cluster):
{title_lines}

Queries that returned zero results (do NOT repeat these):
{zero_lines}

Generate 4-6 alternative search keywords and 4-6 alternative search queries in English.
Try different angles: synonyms, related fields, specific sub-topics, or implementation perspectives.

Respond ONLY with valid JSON, no explanation:
{{"keywords": ["...", "..."], "queries": ["...", "..."]}}"""
