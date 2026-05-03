"""src/cycle/query_generator.py — LLM 支援による検索クエリ生成

GapDetector が生成した CollectionTask の signals を読み、
LLM (fastflowlm / lmstudio) を使って keywords と queries を補完する。

LLM 不使用の fallback も備える（プロバイダ未接続時に自動適用）。
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Optional

from src.cycle.schema import CollectionTask, GapType

logger = logging.getLogger(__name__)

# ---- 設定 ----------------------------------------------------------

_DEFAULT_PROVIDER = "fastflowlm"
_DEFAULT_DB_PATH  = "data/metadata.db"
_SAMPLE_TITLE_LIMIT = 8   # プロンプトに含めるサンプルタイトル数
_MAX_QUERY_RETRIES  = 2   # JSON parse 失敗時のリトライ数


# ---- QueryGenerator ------------------------------------------------

class QueryGenerator:
    """CollectionTask に keywords / queries を LLM で補完するクラス。

    Args:
        provider:   使用する LLM プロバイダ名 (llm_config.local.yaml のキー)。
        db_path:    metadata.db のパス（サンプルタイトル取得用）。
        temperature: LLM 生成温度。創造的クエリには 0.7 程度を推奨。
        max_tokens: LLM 最大出力トークン数。
    """

    def __init__(
        self,
        provider: str = _DEFAULT_PROVIDER,
        model: Optional[str] = None,
        db_path: str = _DEFAULT_DB_PATH,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> None:
        self._provider    = provider
        self._model       = model
        self._db_path     = db_path
        self._temperature = temperature
        self._max_tokens  = max_tokens
        self._gateway: Optional[object] = None

    # ---- public ----------------------------------------------------

    async def enrich(self, task: CollectionTask) -> CollectionTask:
        """task.keywords / task.queries を LLM で補完して返す。

        LLM が利用できない場合は fallback を適用する。
        """
        titles = await self._fetch_titles(task.signals.get("sample_doc_ids", []))
        try:
            gateway = await self._get_gateway()
            keywords, queries = await self._call_llm(task, titles, gateway)
        except Exception as exc:
            logger.warning("LLM call failed (%s); using fallback heuristic", exc)
            keywords, queries = _fallback(task)

        task.keywords = keywords
        task.queries  = queries
        return task

    async def enrich_batch(
        self,
        tasks: list[CollectionTask],
        concurrency: int = 3,
    ) -> list[CollectionTask]:
        """複数タスクを並列で enrich する。"""
        import asyncio
        sem = asyncio.Semaphore(concurrency)

        async def _bounded(t: CollectionTask) -> CollectionTask:
            async with sem:
                return await self.enrich(t)

        return list(await asyncio.gather(*[_bounded(t) for t in tasks]))

    # ---- private ---------------------------------------------------

    async def _get_gateway(self) -> object:
        """LLMGateway のシングルトンを返す（遅延初期化）。"""
        if self._gateway is None:
            from src.llm.gateway import LLMGateway
            self._gateway = LLMGateway()
        return self._gateway

    async def _fetch_titles(self, doc_ids: list[str]) -> list[str]:
        """DB からサンプル doc_id に対応するタイトルを取得する（同期を thread で）。"""
        if not doc_ids:
            return []

        import asyncio

        def _query() -> list[str]:
            try:
                conn = sqlite3.connect(self._db_path)
                placeholders = ",".join("?" * len(doc_ids))
                rows = conn.execute(
                    f"SELECT source_title FROM documents WHERE id IN ({placeholders})",
                    doc_ids[:_SAMPLE_TITLE_LIMIT],
                ).fetchall()
                conn.close()
                return [r[0] for r in rows if r[0]]
            except Exception:
                return []

        return await asyncio.to_thread(_query)

    async def _call_llm(
        self,
        task: CollectionTask,
        titles: list[str],
        gateway: object,
    ) -> tuple[list[str], list[str]]:
        """LLM にプロンプトを送り (keywords, queries) を返す。"""
        prompt = _build_prompt(task, titles)

        for attempt in range(_MAX_QUERY_RETRIES + 1):
            resp = await gateway.complete(
                prompt,
                provider=self._provider,
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            text = resp.content.strip()
            result = _parse_json(text)
            if result is not None:
                keywords = [str(k) for k in result.get("keywords", [])][:8]
                queries  = [str(q) for q in result.get("queries", [])][:8]
                if keywords or queries:
                    return keywords, queries
            logger.debug("JSON parse failed on attempt %d; retrying", attempt + 1)

        raise ValueError("LLM returned no parseable keywords/queries")


# ---- プロンプト ビルダー ------------------------------------------

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

    gap_hint = {
        GapType.SMALL_CLUSTER:      "This topic area is under-represented. Find diverse sources.",
        GapType.UNREVIEWED_BACKLOG: "This cluster has many unreviewed docs. Find high-quality sources.",
        GapType.SOURCE_IMBALANCE:   f"The cluster relies too heavily on '{sig.get('dominant_source', 'unknown')}'. Suggest queries for other sources.",
        GapType.LOW_QUALITY:        "Quality is low here. Prioritize authoritative, detailed sources.",
    }.get(task.gap_type, "")

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


# ---- JSON パーサー ------------------------------------------------

_JSON_RE = re.compile(r'\{.*?\}', re.DOTALL)


def _parse_json(text: str) -> Optional[dict]:
    """LLM レスポンスから JSON オブジェクトを抽出する。"""
    # コードブロックを除去
    text = re.sub(r'```(?:json)?\s*', '', text).strip()
    # 最初の {} を探す
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


# ---- フォールバック -----------------------------------------------

def _fallback(task: CollectionTask) -> tuple[list[str], list[str]]:
    """LLM 不使用の簡易フォールバック。

    signals の dominant_source・gap_type からキーワードをヒューリスティックに生成する。
    """
    sig = task.signals
    dominant = sig.get("dominant_source", "")
    size     = sig.get("size", 0)

    # gap_type ベースのキーワード接頭語
    prefix_map = {
        GapType.SMALL_CLUSTER:      ["tutorial", "guide", "introduction"],
        GapType.UNREVIEWED_BACKLOG: ["review", "survey", "overview"],
        GapType.SOURCE_IMBALANCE:   ["alternative", "comparison", "benchmark"],
        GapType.LOW_QUALITY:        ["best practices", "authoritative", "documentation"],
    }
    prefixes = prefix_map.get(task.gap_type, ["guide"])

    # reason から名詞フレーズを簡易抽出
    words = re.findall(r'\b[A-Za-z][a-z]{3,}\b', task.reason)
    content_words = [w.lower() for w in words if w.lower() not in
                     {"island", "docs", "cluster", "from", "only", "needs", "more",
                      "documents", "prioritize", "maturation", "consider"}]
    content_words = list(dict.fromkeys(content_words))[:3]

    keywords = (prefixes[:2] + content_words)[:6]
    queries  = [
        f"{kw} {dominant} documentation" if dominant else f"{kw} documentation"
        for kw in keywords[:3]
    ]

    logger.info(
        "Fallback keywords for task %s: %s", task.task_id[:8], keywords
    )
    return keywords, queries
