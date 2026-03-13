"""src/memory/maturation/difficulty_tagger.py — 難易度タガー

ドキュメントの内容を LLM で分析し、DifficultyLevel を付与する。
Student モデルのカリキュラム学習（beginner → expert 順提示）に使用する。

使い方:
    from src.memory.maturation.difficulty_tagger import DifficultyTagger

    tagger = DifficultyTagger(gateway)
    level = await tagger.tag(doc)
    # → DifficultyLevel.intermediate
"""

from __future__ import annotations

import logging

from src.llm.gateway import LLMGateway
from src.memory.schema import DifficultyLevel, Document

logger = logging.getLogger(__name__)

_TAG_SYSTEM = """\
You are an expert at assessing technical difficulty. Classify the difficulty of the given text.
Respond with ONLY one of: beginner, intermediate, advanced, expert

Criteria:
- beginner: Basic concepts, no prior knowledge needed, simple examples
- intermediate: Requires some background, moderate complexity
- advanced: Deep technical knowledge required, complex algorithms/patterns
- expert: Cutting-edge research, specialized expertise needed"""

_TAG_PROMPT = "Classify the difficulty of this text:\n\n{text}"

_LEVEL_MAP: dict[str, DifficultyLevel] = {
    "beginner": DifficultyLevel.BEGINNER,
    "intermediate": DifficultyLevel.INTERMEDIATE,
    "advanced": DifficultyLevel.ADVANCED,
    "expert": DifficultyLevel.EXPERT,
}


class DifficultyTagger:
    """LLM でドキュメントの難易度を分類する。

    Args:
        gateway: LLMGateway インスタンス。
        provider: 優先プロバイダ（省略時はデフォルト）。
        max_text_length: LLM に渡す最大文字数。
        default_level: LLM 失敗時のデフォルト難易度。
    """

    def __init__(
        self,
        gateway: LLMGateway,
        provider: str | None = None,
        max_text_length: int = 800,
        default_level: DifficultyLevel = DifficultyLevel.INTERMEDIATE,
    ) -> None:
        self._gateway = gateway
        self._provider = provider
        self._max_text = max_text_length
        self._default_level = default_level

    async def tag(self, doc: Document) -> DifficultyLevel:
        """ドキュメントの難易度を分類する。

        Returns:
            DifficultyLevel（LLM 失敗時はデフォルト値）。
        """
        text = doc.content[:self._max_text]
        prompt = _TAG_PROMPT.format(text=text)

        try:
            response = await self._gateway.complete(
                prompt,
                system=_TAG_SYSTEM,
                provider=self._provider,
                max_tokens=20,
                temperature=0.0,
            )
            level_str = response.content.strip().lower().split()[0]
            level = _LEVEL_MAP.get(level_str, self._default_level)
            logger.debug("Difficulty for doc=%s: %s → %s", doc.id, level_str, level)
            return level
        except Exception:
            logger.exception("Difficulty tagging failed for doc=%s; using default", doc.id)
            return self._default_level

    async def tag_batch(self, docs: list[Document]) -> dict[str, DifficultyLevel]:
        """複数ドキュメントの難易度を一括分類する。

        Returns:
            {doc_id: DifficultyLevel} の辞書。
        """
        import asyncio
        results = await asyncio.gather(*[self.tag(doc) for doc in docs])
        return {doc.id: level for doc, level in zip(docs, results)}
