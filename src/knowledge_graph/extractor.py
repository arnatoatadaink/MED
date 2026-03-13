"""src/knowledge_graph/extractor.py — EntityExtractor (Teacher API 使用)

ドキュメントのテキストから Entity と Relation を抽出し、KG に登録する。
Teacher API（Claude/GPT）を使って高精度な抽出を行う。

設計方針:
- Teacher API（精度優先）を推奨
- JSON 形式で Entity/Relation を抽出
- 抽出失敗時は空リストを返す（KG 登録をスキップ）

使い方:
    from src.knowledge_graph.extractor import EntityExtractor
    from src.knowledge_graph.store import KnowledgeGraphStore

    kg = KnowledgeGraphStore()
    extractor = EntityExtractor(gateway)
    await extractor.extract_and_register(doc, kg)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.llm.gateway import LLMGateway
from src.memory.schema import Document

if TYPE_CHECKING:
    from src.knowledge_graph.store import KnowledgeGraphStore

logger = logging.getLogger(__name__)

_EXTRACT_SYSTEM = """\
You are a knowledge graph expert. Extract entities and relations from the given text.
Output ONLY valid JSON in the following format, nothing else:
{
  "entities": [
    {"name": "EntityName", "type": "concept|language|library|algorithm|framework|tool|other"}
  ],
  "relations": [
    {"source": "Entity1", "target": "Entity2", "type": "uses|extends|causes|part_of|implements|related_to", "weight": 0.8}
  ]
}
Rules:
- Entity names should be concise (1-4 words)
- Extract 3-10 entities and 2-8 relations maximum
- Weight is 0.0-1.0 indicating confidence"""

_EXTRACT_PROMPT = """\
Extract entities and relations from the following text:

{text}"""


@dataclass
class ExtractedKG:
    """抽出された KG 情報。"""

    entities: list[dict]   # [{"name": str, "type": str}]
    relations: list[dict]  # [{"source": str, "target": str, "type": str, "weight": float}]
    doc_id: str
    success: bool


class EntityExtractor:
    """LLM を使って Entity と Relation を抽出する。

    Args:
        gateway: LLMGateway インスタンス。
        provider: 優先プロバイダ（省略時はデフォルト）。
        max_text_length: LLM に渡す最大文字数。
    """

    def __init__(
        self,
        gateway: LLMGateway,
        provider: str | None = None,
        max_text_length: int = 1000,
    ) -> None:
        self._gateway = gateway
        self._provider = provider
        self._max_text = max_text_length

    async def extract(self, doc: Document) -> ExtractedKG:
        """ドキュメントから Entity と Relation を抽出する。

        Returns:
            ExtractedKG オブジェクト。LLM 失敗時は success=False の空データ。
        """
        text = doc.content[:self._max_text]
        prompt = _EXTRACT_PROMPT.format(text=text)

        try:
            response = await self._gateway.complete(
                prompt,
                system=_EXTRACT_SYSTEM,
                provider=self._provider,
                max_tokens=512,
                temperature=0.0,
            )
            parsed = self._parse_response(response.content)
            return ExtractedKG(
                entities=parsed.get("entities", []),
                relations=parsed.get("relations", []),
                doc_id=doc.id,
                success=True,
            )
        except Exception:
            logger.exception("Entity extraction failed for doc=%s", doc.id)
            return ExtractedKG(entities=[], relations=[], doc_id=doc.id, success=False)

    def _parse_response(self, content: str) -> dict:
        """LLM レスポンスから JSON を抽出する。"""
        # コードブロック除去
        content = re.sub(r"```(?:json)?\s*", "", content).strip()
        content = content.rstrip("`").strip()

        # JSON 部分を抽出
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            content = match.group(0)

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response: %r", content[:200])
            return {"entities": [], "relations": []}

    async def extract_and_register(
        self,
        doc: Document,
        kg: KnowledgeGraphStore,
    ) -> ExtractedKG:
        """抽出結果を KG に登録する。

        Args:
            doc: 対象ドキュメント。
            kg: 登録先 KnowledgeGraphStore。

        Returns:
            ExtractedKG（登録済み）。
        """

        extracted = await self.extract(doc)
        if not extracted.success:
            return extracted

        # Entity 登録
        for ent in extracted.entities:
            name = ent.get("name", "").strip()
            etype = ent.get("type", "concept")
            if name:
                kg.add_entity(name, entity_type=etype, doc_id=doc.id)

        # Relation 登録
        for rel in extracted.relations:
            src = rel.get("source", "").strip()
            tgt = rel.get("target", "").strip()
            rtype = rel.get("type", "related_to")
            weight = float(rel.get("weight", 1.0))
            if src and tgt:
                kg.add_relation(src, tgt, relation_type=rtype, weight=weight)

        logger.debug(
            "Registered %d entities, %d relations for doc=%s",
            len(extracted.entities), len(extracted.relations), doc.id,
        )
        return extracted
