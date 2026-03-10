"""src/orchestrator/query_parser.py — LLMベースクエリ解析器

受信クエリを LLM で解析し、意図・ドメイン・複雑度・エンティティを抽出する。
ModelRouter がルーティング判断に使用する。

解析結果:
- intent: "qa" | "code" | "explain" | "search" | "compute"
- domain: "code" | "academic" | "general"
- complexity: "simple" | "moderate" | "complex"
- entities: 主要エンティティ名のリスト
- requires_execution: コード実行が必要か
- requires_retrieval: 外部検索が必要か

使い方:
    from src.orchestrator.query_parser import QueryParser, ParsedQuery

    parser = QueryParser(gateway)
    parsed = await parser.parse("How do I implement FAISS index?")
    # → ParsedQuery(intent="code", domain="code", complexity="moderate", ...)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from src.llm.gateway import LLMGateway

logger = logging.getLogger(__name__)

_PARSE_SYSTEM = """\
You are a query analyzer. Analyze the user query and respond with ONLY valid JSON:
{
  "intent": "qa|code|explain|search|compute",
  "domain": "code|academic|general",
  "complexity": "simple|moderate|complex",
  "entities": ["entity1", "entity2"],
  "requires_execution": true/false,
  "requires_retrieval": true/false
}

Intent definitions:
- qa: factual question-answering
- code: code generation/debugging
- explain: explanation of concept
- search: information retrieval
- compute: mathematical/logical computation

Complexity:
- simple: single step, direct answer
- moderate: multi-step, some context needed
- complex: multi-hop reasoning, research-level"""

_PARSE_PROMPT = "Analyze this query: {query}"


@dataclass
class ParsedQuery:
    """解析済みクエリ。"""

    raw: str
    intent: str = "qa"
    domain: str = "general"
    complexity: str = "moderate"
    entities: list[str] = field(default_factory=list)
    requires_execution: bool = False
    requires_retrieval: bool = True

    @property
    def is_simple(self) -> bool:
        return self.complexity == "simple"

    @property
    def is_complex(self) -> bool:
        return self.complexity == "complex"

    @property
    def is_code(self) -> bool:
        return self.intent == "code" or self.domain == "code"


# キーワードベースのフォールバック解析
_CODE_KEYWORDS = re.compile(
    r"\b(code|implement|function|class|algorithm|bug|debug|error|syntax|import|pip|"
    r"python|javascript|java|c\+\+|rust|go|sql|bash|script|API|library)\b",
    re.IGNORECASE,
)
_COMPUTE_KEYWORDS = re.compile(
    r"\b(calculate|compute|solve|equation|integral|derivative|matrix|probability|"
    r"statistic|math|formula)\b",
    re.IGNORECASE,
)
_EXPLAIN_KEYWORDS = re.compile(
    r"\b(explain|describe|what is|what are|how does|why|define|definition|concept)\b",
    re.IGNORECASE,
)
_COMPLEX_KEYWORDS = re.compile(
    r"\b(compare|analyze|design|architect|trade-off|optimize|research|survey|"
    r"implement.*system|end-to-end)\b",
    re.IGNORECASE,
)


class QueryParser:
    """LLM ベースのクエリ解析器。

    Args:
        gateway: LLMGateway インスタンス。
        provider: 優先プロバイダ。
        use_fallback: LLM 失敗時にキーワードベース解析にフォールバックするか。
    """

    def __init__(
        self,
        gateway: LLMGateway,
        provider: Optional[str] = None,
        use_fallback: bool = True,
    ) -> None:
        self._gateway = gateway
        self._provider = provider
        self._use_fallback = use_fallback

    async def parse(self, query: str) -> ParsedQuery:
        """クエリを解析する。

        Returns:
            ParsedQuery オブジェクト。LLM 失敗時はキーワードベース解析結果。
        """
        try:
            response = await self._gateway.complete(
                _PARSE_PROMPT.format(query=query),
                system=_PARSE_SYSTEM,
                provider=self._provider,
                max_tokens=150,
                temperature=0.0,
            )
            parsed = self._parse_json(response.content)
            result = ParsedQuery(
                raw=query,
                intent=parsed.get("intent", "qa"),
                domain=parsed.get("domain", "general"),
                complexity=parsed.get("complexity", "moderate"),
                entities=parsed.get("entities", []),
                requires_execution=bool(parsed.get("requires_execution", False)),
                requires_retrieval=bool(parsed.get("requires_retrieval", True)),
            )
            logger.debug(
                "QueryParser: intent=%s domain=%s complexity=%s entities=%s",
                result.intent, result.domain, result.complexity, result.entities,
            )
            return result
        except Exception:
            logger.exception("QueryParser LLM failed for: %r", query[:50])
            if self._use_fallback:
                return self._keyword_parse(query)
            return ParsedQuery(raw=query)

    def _parse_json(self, content: str) -> dict:
        content = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("`").strip()
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            content = match.group(0)
        return json.loads(content)

    def _keyword_parse(self, query: str) -> ParsedQuery:
        """キーワードベースのフォールバック解析。"""
        intent = "qa"
        domain = "general"
        complexity = "moderate"
        requires_execution = False

        if _CODE_KEYWORDS.search(query):
            intent = "code"
            domain = "code"
            requires_execution = True
        elif _COMPUTE_KEYWORDS.search(query):
            intent = "compute"
        elif _EXPLAIN_KEYWORDS.search(query):
            intent = "explain"

        if _COMPLEX_KEYWORDS.search(query):
            complexity = "complex"
        elif len(query.split()) <= 8:
            complexity = "simple"

        return ParsedQuery(
            raw=query,
            intent=intent,
            domain=domain,
            complexity=complexity,
            entities=[],
            requires_execution=requires_execution,
            requires_retrieval=True,
        )
