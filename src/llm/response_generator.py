"""src/llm/response_generator.py — RAG 結果を使ったレスポンス生成

FAISS 検索結果（SearchResult）と外部 RAG 結果（RawResult）を文脈として
LLM にプロンプトを送り、回答を生成する。

使い方:
    from src.llm.response_generator import ResponseGenerator

    generator = ResponseGenerator(gateway)
    response = await generator.generate(
        query="Python でリストをソートするには？",
        context_docs=faiss_results,
    )
    print(response.answer)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.llm.gateway import LLMGateway, LLMMessage, LLMResponse
from src.memory.schema import SearchResult

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM = """\
You are a knowledgeable technical assistant. Use the provided context documents \
to answer the user's question accurately. If the context doesn't contain enough \
information, say so clearly. Always cite which context document you used."""

_RESPONSE_TEMPLATE = """\
Context documents:
{context}

User question: {query}

Please provide a comprehensive answer based on the context above."""


@dataclass
class GeneratedResponse:
    """LLM が生成したレスポンス。"""

    answer: str
    query: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    context_doc_ids: list[str] = field(default_factory=list)


class ResponseGenerator:
    """FAISS 検索結果を文脈として LLM レスポンスを生成する。

    Args:
        gateway: LLMGateway インスタンス。
        system_prompt: システムプロンプト（省略時はデフォルト）。
        max_context_docs: 文脈に含める最大ドキュメント数。
        max_context_chars: 文脈全体の最大文字数。
        provider: 使用プロバイダ（省略時はデフォルト）。
    """

    def __init__(
        self,
        gateway: LLMGateway,
        system_prompt: Optional[str] = None,
        max_context_docs: int = 5,
        max_context_chars: int = 4000,
        provider: Optional[str] = None,
    ) -> None:
        self._gateway = gateway
        self._system = system_prompt or _DEFAULT_SYSTEM
        self._max_docs = max_context_docs
        self._max_chars = max_context_chars
        self._provider = provider

    async def generate(
        self,
        query: str,
        context_docs: list[SearchResult],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> GeneratedResponse:
        """クエリと検索結果から回答を生成する。

        Args:
            query: ユーザーのクエリ。
            context_docs: FAISS 検索の SearchResult リスト。
            max_tokens: 最大出力トークン数。
            temperature: 温度パラメータ。

        Returns:
            GeneratedResponse オブジェクト。
        """
        # 文脈を構築
        context_text, used_ids = self._build_context(context_docs)

        prompt = _RESPONSE_TEMPLATE.format(
            context=context_text,
            query=query,
        )

        messages = [
            LLMMessage(role="system", content=self._system),
            LLMMessage(role="user", content=prompt),
        ]

        raw: LLMResponse = await self._gateway.complete_messages(
            messages,
            provider=self._provider,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        logger.debug(
            "ResponseGenerator: query=%r in=%d out=%d provider=%s",
            query[:50], raw.input_tokens, raw.output_tokens, raw.provider,
        )

        return GeneratedResponse(
            answer=raw.content,
            query=query,
            provider=raw.provider,
            model=raw.model,
            input_tokens=raw.input_tokens,
            output_tokens=raw.output_tokens,
            context_doc_ids=used_ids,
        )

    def _build_context(
        self,
        docs: list[SearchResult],
    ) -> tuple[str, list[str]]:
        """検索結果から文脈文字列を構築する。

        Returns:
            (context_text, doc_id_list)
        """
        selected = docs[: self._max_docs]
        parts: list[str] = []
        used_ids: list[str] = []
        total_chars = 0

        for idx, sr in enumerate(selected, start=1):
            doc = sr.document
            snippet = doc.content[: self._max_chars // max(1, len(selected))]
            part = f"[{idx}] {snippet}"
            if total_chars + len(part) > self._max_chars:
                break
            parts.append(part)
            used_ids.append(doc.id)
            total_chars += len(part)

        return "\n\n".join(parts), used_ids
