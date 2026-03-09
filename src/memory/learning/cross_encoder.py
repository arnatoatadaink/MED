"""src/memory/learning/cross_encoder.py — Cross-Encoder リランカー

LLM を Cross-Encoder として使用し、クエリとドキュメントの関連性を直接スコアリングする。
Phase 1 の線形 LTR (ltr_ranker.py) の後継として導入。

設計方針:
- 入力: (query, document_content) のペア
- 出力: 0.0〜1.0 の関連性スコア
- LLM に YES/NO + スコアを問い合わせる「LLM-as-Reranker」パターン
- バッチ処理と並列化 (asyncio.Semaphore) をサポート

使い方:
    from src.memory.learning.cross_encoder import CrossEncoder

    encoder = CrossEncoder(gateway)
    score = await encoder.score(query, document)
    ranked = await encoder.rerank(query, documents, top_k=5)
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from src.llm.gateway import LLMGateway
from src.memory.schema import Document, SearchResult

logger = logging.getLogger(__name__)

_SCORE_SYSTEM = """\
You are a relevance judge. Given a query and a document, rate how relevant the document is.
Respond with ONLY a number between 0.0 and 1.0 (e.g. 0.8).
1.0 = perfectly relevant, 0.0 = completely irrelevant."""

_SCORE_PROMPT = """\
Query: {query}

Document:
{text}

Relevance score (0.0-1.0):"""


class CrossEncoder:
    """LLM ベースの Cross-Encoder リランカー。

    Args:
        gateway: LLMGateway インスタンス。
        provider: 優先プロバイダ（省略時はデフォルト）。
        max_doc_chars: ドキュメントの最大文字数。
        max_concurrency: 同時 LLM 呼び出し数の上限。
        fallback_score: LLM 失敗時のデフォルトスコア。
    """

    def __init__(
        self,
        gateway: LLMGateway,
        provider: Optional[str] = None,
        max_doc_chars: int = 1000,
        max_concurrency: int = 5,
        fallback_score: float = 0.5,
    ) -> None:
        self._gateway = gateway
        self._provider = provider
        self._max_doc_chars = max_doc_chars
        self._max_concurrency = max_concurrency
        self._fallback_score = fallback_score

    async def score(self, query: str, doc: Document) -> float:
        """クエリとドキュメントの関連性スコアを計算する。

        Returns:
            関連性スコア (0.0〜1.0)。LLM 失敗時は fallback_score。
        """
        text = doc.content[:self._max_doc_chars]
        prompt = _SCORE_PROMPT.format(query=query, text=text)

        try:
            response = await self._gateway.complete(
                prompt,
                system=_SCORE_SYSTEM,
                provider=self._provider,
                max_tokens=10,
                temperature=0.0,
            )
            return self._parse_score(response.content)
        except Exception:
            logger.exception(
                "CrossEncoder.score failed for query=%r doc=%s; using fallback",
                query[:50], doc.id,
            )
            return self._fallback_score

    async def score_batch(
        self,
        query: str,
        docs: list[Document],
    ) -> list[float]:
        """複数ドキュメントのスコアを並列計算する。

        Returns:
            docs と同じ順序のスコアリスト。
        """
        import asyncio
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def _score_with_sem(doc: Document) -> float:
            async with semaphore:
                return await self.score(query, doc)

        return list(await asyncio.gather(*[_score_with_sem(d) for d in docs]))

    async def rerank(
        self,
        query: str,
        search_results: list[SearchResult],
        top_k: Optional[int] = None,
    ) -> list[SearchResult]:
        """SearchResult リストを Cross-Encoder スコアで再ランク付けする。

        Args:
            query: 検索クエリ。
            search_results: FAISS 検索結果リスト。
            top_k: 返す上位件数（省略時は全件）。

        Returns:
            Cross-Encoder スコア降順に並び替えた SearchResult リスト。
        """
        if not search_results:
            return []

        docs = [sr.document for sr in search_results]
        scores = await self.score_batch(query, docs)

        reranked = sorted(
            zip(search_results, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = [sr for sr, _score in reranked]
        if top_k is not None:
            results = results[:top_k]

        logger.debug(
            "CrossEncoder.rerank: %d → %d results for query=%r",
            len(search_results), len(results), query[:50],
        )
        return results

    def _parse_score(self, content: str) -> float:
        """LLM レスポンスから 0.0〜1.0 のスコアを抽出する。"""
        content = content.strip()
        # 数値部分を抽出
        match = re.search(r"([0-9]*\.?[0-9]+)", content)
        if match:
            try:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))  # クランプ
            except ValueError:
                pass
        logger.warning("CrossEncoder: could not parse score from %r", content[:50])
        return self._fallback_score
