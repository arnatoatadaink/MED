"""src/rag/verifier.py — LLM ベース検索結果裏どり (Verifier)

検索結果がクエリに関連しているかを LLM で検証し、
無関係なノイズを除去する。

使い方:
    from src.rag.verifier import ResultVerifier

    verifier = ResultVerifier(gateway)
    verified = await verifier.verify(results, query="Python FAISS usage")
"""

from __future__ import annotations

import logging
from typing import Optional

from src.llm.gateway import LLMGateway
from src.rag.retriever import RawResult

logger = logging.getLogger(__name__)

_VERIFY_PROMPT = """\
Given the following search result and user query, determine if the result is relevant.
Respond with only "YES" or "NO".

Query: {query}

Search Result:
Title: {title}
Content: {content}

Is this result relevant to the query? (YES/NO):"""


class ResultVerifier:
    """LLM を使って検索結果の関連性を検証する。

    Args:
        gateway: LLMGateway インスタンス。None の場合は検証をスキップ（全結果通過）。
        max_content_length: LLM に渡す最大コンテンツ文字数。
        provider: 使用する LLM プロバイダ（省略時はデフォルト）。
    """

    def __init__(
        self,
        gateway: Optional[LLMGateway] = None,
        max_content_length: int = 500,
        provider: Optional[str] = None,
    ) -> None:
        self._gateway = gateway
        self._max_len = max_content_length
        self._provider = provider

    async def verify(
        self,
        results: list[RawResult],
        query: str,
        max_concurrency: int = 5,
    ) -> list[RawResult]:
        """検索結果を LLM で検証し、関連するものだけ返す。

        Args:
            results: 検証対象の検索結果リスト。
            query: 元のクエリ。
            max_concurrency: 同時に LLM を呼ぶ最大件数。

        Returns:
            LLM が "YES" と判断した結果のリスト。
        """
        if not results:
            return []

        if self._gateway is None:
            logger.debug("Verifier: no gateway, passing all %d results", len(results))
            return results

        import asyncio

        semaphore = asyncio.Semaphore(max_concurrency)

        async def _check(result: RawResult) -> Optional[RawResult]:
            async with semaphore:
                try:
                    relevant = await self._is_relevant(result, query)
                    return result if relevant else None
                except Exception:
                    logger.exception(
                        "Verifier failed for url=%s; passing through", result.url
                    )
                    return result  # 検証失敗時は通過させる

        tasks = [_check(r) for r in results]
        checked = await asyncio.gather(*tasks)
        verified = [r for r in checked if r is not None]

        logger.info(
            "Verifier: %d/%d results passed for query=%r",
            len(verified), len(results), query[:50],
        )
        return verified

    async def _is_relevant(self, result: RawResult, query: str) -> bool:
        """単一の検索結果が関連するかを LLM で判定する。"""
        assert self._gateway is not None

        content_snippet = result.content[:self._max_len]
        prompt = _VERIFY_PROMPT.format(
            query=query,
            title=result.title[:200],
            content=content_snippet,
        )

        response = await self._gateway.complete(
            prompt,
            provider=self._provider,
            max_tokens=10,
            temperature=0.0,
        )

        answer = response.content.strip().upper()
        return answer.startswith("YES")
