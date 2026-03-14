"""src/retrieval/query_classifier.py — クエリ分類器

クエリを SEMANTIC / FACTUAL / RELATIONAL / HYBRID の4種に分類する。
Fusion/Reranker がどの検索バックエンドを使うかを決定するルーティング判断に使用する。

分類基準:
- SEMANTIC: 概念的・意味論的な質問（「○○とは何か」「○○の特徴は」）
- FACTUAL: 具体的な事実・数値・日付（「○○はいつ」「○○の値は」）
- RELATIONAL: エンティティ間の関係・パス探索（「○○と○○の関係」「○○が○○に影響する理由」）
- HYBRID: 複数タイプが混在するクエリ

使い方:
    from src.retrieval.query_classifier import QueryClassifier, QueryType

    classifier = QueryClassifier(gateway)
    qtype = await classifier.classify("What is the relationship between FAISS and ANN?")
    # → QueryType.RELATIONAL
"""

from __future__ import annotations

import logging
import re
from enum import Enum

from src.llm.gateway import LLMGateway

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """クエリタイプ。"""

    SEMANTIC = "semantic"      # 概念・意味論的な質問
    FACTUAL = "factual"        # 具体的な事実・数値
    RELATIONAL = "relational"  # エンティティ間の関係性
    HYBRID = "hybrid"          # 複合タイプ


_CLASSIFY_SYSTEM = """\
Classify the query into ONE of: semantic, factual, relational, hybrid.

- semantic: conceptual questions about meaning, definition, explanation (What is X? How does X work?)
- factual: specific facts, numbers, dates, names (When was X? What value does X have?)
- relational: relationships, connections, paths between entities (How does X relate to Y? Why does X affect Y?)
- hybrid: multiple types combined

Respond with ONLY one word: semantic, factual, relational, or hybrid."""

_CLASSIFY_PROMPT = "Classify: {query}"

_TYPE_MAP: dict[str, QueryType] = {
    "semantic": QueryType.SEMANTIC,
    "factual": QueryType.FACTUAL,
    "relational": QueryType.RELATIONAL,
    "hybrid": QueryType.HYBRID,
}

# キーワードベースのフォールバック分類器
_RELATIONAL_PATTERNS = re.compile(
    r"\b(relat\w*|connect\w*|link\w*|between|affect\w*|depend\w*|interact\w*|associ\w*|path\b|how\s+\w+\s+work\s+together)",
    re.IGNORECASE,
)
_FACTUAL_PATTERNS = re.compile(
    r"\b(when|where|who|which|how many|how much|what year|version|number|count|date|value)\b",
    re.IGNORECASE,
)


class QueryClassifier:
    """LLM ベースのクエリ分類器。

    Args:
        gateway: LLMGateway インスタンス。
        provider: 優先プロバイダ。
        use_fallback: LLM 失敗時にキーワードベースの分類にフォールバックするか。
        default_type: フォールバック分類も失敗した場合のデフォルトタイプ。
    """

    def __init__(
        self,
        gateway: LLMGateway,
        provider: str | None = None,
        use_fallback: bool = True,
        default_type: QueryType = QueryType.SEMANTIC,
    ) -> None:
        self._gateway = gateway
        self._provider = provider
        self._use_fallback = use_fallback
        self._default_type = default_type

    async def classify(self, query: str) -> QueryType:
        """クエリタイプを分類する。

        Returns:
            QueryType。LLM 失敗時はキーワードベース or デフォルトにフォールバック。
        """
        try:
            response = await self._gateway.complete(
                _CLASSIFY_PROMPT.format(query=query),
                system=_CLASSIFY_SYSTEM,
                provider=self._provider,
                max_tokens=10,
                temperature=0.0,
            )
            type_str = response.content.strip().lower().split()[0]
            qtype = _TYPE_MAP.get(type_str)
            if qtype is not None:
                logger.debug("QueryClassifier: %r → %s", query[:50], qtype)
                return qtype
            logger.warning("Unknown query type: %r; falling back", type_str)
        except Exception:
            logger.exception("QueryClassifier LLM call failed for: %r", query[:50])

        if self._use_fallback:
            return self._keyword_classify(query)
        return self._default_type

    async def classify_batch(self, queries: list[str]) -> list[QueryType]:
        """複数クエリを並列分類する。"""
        import asyncio
        return list(await asyncio.gather(*[self.classify(q) for q in queries]))

    def _keyword_classify(self, query: str) -> QueryType:
        """キーワードベースのフォールバック分類。"""
        is_relational = bool(_RELATIONAL_PATTERNS.search(query))
        is_factual = bool(_FACTUAL_PATTERNS.search(query))

        if is_relational and is_factual:
            return QueryType.HYBRID
        if is_relational:
            return QueryType.RELATIONAL
        if is_factual:
            return QueryType.FACTUAL
        return self._default_type
