"""src/memory/maturation/seed_builder.py — シードデータ構築

FAISSメモリの初期シードドキュメントを構築する。
Teacher Model に「良質なサンプル」を生成させ、メモリの品質基盤を確立する。

使い方:
    from src.memory.maturation.seed_builder import SeedBuilder

    builder = SeedBuilder(gateway, memory_manager)
    await builder.build(
        topic="Python FAISS vector search",
        domain="code",
        n_samples=20,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.llm.gateway import LLMGateway
from src.memory.memory_manager import MemoryManager
from src.memory.schema import DifficultyLevel, Document, ReviewStatus, SourceMeta, SourceType

logger = logging.getLogger(__name__)

_SEED_SYSTEM = """\
You are a technical content creator building a knowledge base.
Generate a high-quality, self-contained technical document about the given topic.
The document should be educational, accurate, and practical.
Write approximately 200-400 words."""

_SEED_PROMPT = """\
Topic: {topic}
Difficulty: {difficulty}
Domain: {domain}

Generate a technical document:"""

_SEED_QA_SYSTEM = """\
Generate a high-quality Q&A pair for a technical knowledge base.
Output in format:
Q: [question]
A: [detailed answer]"""

_SEED_QA_PROMPT = "Generate a Q&A about: {topic} (difficulty: {difficulty})"


@dataclass
class SeedResult:
    """シードビルド結果。"""

    docs_created: int
    docs_failed: int
    doc_ids: list[str]


class SeedBuilder:
    """Teacher Model でシードドキュメントを生成する。

    Args:
        gateway: LLMGateway インスタンス。
        memory_manager: MemoryManager インスタンス（生成物を追加）。
        provider: 優先プロバイダ（省略時はデフォルト）。
        teacher_id: Teacher モデル識別子（素性追跡用）。省略時は gateway のデフォルトモデル。
    """

    def __init__(
        self,
        gateway: LLMGateway,
        memory_manager: MemoryManager,
        provider: str | None = None,
        teacher_id: str | None = None,
    ) -> None:
        self._gateway = gateway
        self._mm = memory_manager
        self._provider = provider
        self._teacher_id = teacher_id or "unknown"

    async def build(
        self,
        topic: str,
        domain: str = "general",
        n_samples: int = 10,
        difficulty_distribution: dict[str, int] | None = None,
        seed_type: str = "document",
    ) -> SeedResult:
        """指定トピックのシードドキュメントを生成・保存する。

        Args:
            topic: 生成するコンテンツのトピック。
            domain: 対象ドメイン。
            n_samples: 生成するドキュメント数。
            difficulty_distribution: 難易度別の件数。
                例: {"beginner": 3, "intermediate": 4, "advanced": 2, "expert": 1}
            seed_type: "document" または "qa"（Q&A ペア）。

        Returns:
            SeedResult オブジェクト。
        """
        # 難易度分布の構築
        if difficulty_distribution is None:
            difficulty_distribution = self._default_distribution(n_samples)

        tasks = []
        for diff_str, count in difficulty_distribution.items():
            diff = DifficultyLevel(diff_str)
            for _ in range(count):
                tasks.append((topic, domain, diff, seed_type))

        created_ids: list[str] = []
        failed = 0

        import asyncio
        semaphore = asyncio.Semaphore(3)  # 同時 3 件まで

        async def _generate_one(task_args) -> str | None:
            topic_, domain_, diff_, stype = task_args
            async with semaphore:
                return await self._generate_doc(topic_, domain_, diff_, stype)

        results = await asyncio.gather(*[_generate_one(t) for t in tasks])

        for doc_id in results:
            if doc_id is not None:
                created_ids.append(doc_id)
            else:
                failed += 1

        logger.info(
            "SeedBuilder: topic=%r domain=%s created=%d failed=%d",
            topic, domain, len(created_ids), failed,
        )
        return SeedResult(
            docs_created=len(created_ids),
            docs_failed=failed,
            doc_ids=created_ids,
        )

    async def _generate_doc(
        self,
        topic: str,
        domain: str,
        difficulty: DifficultyLevel,
        seed_type: str,
    ) -> str | None:
        """1 件のシードドキュメントを生成して保存する。"""
        try:
            if seed_type == "qa":
                content = await self._generate_qa(topic, difficulty)
            else:
                content = await self._generate_document(topic, domain, difficulty)

            if not content.strip():
                return None

            doc = Document(
                content=content,
                domain=domain,
                difficulty=difficulty,
                review_status=ReviewStatus.APPROVED,  # Teacher 生成なのでそのまま承認
                source=SourceMeta(
                    source_type=SourceType.TEACHER,
                    title=f"Seed: {topic} ({difficulty.value})",
                    tags=["seed", domain, difficulty.value],
                ).set_teacher(self._teacher_id, provider=self._provider),
            )
            return await self._mm.add(doc)

        except Exception:
            logger.exception("Failed to generate seed doc for topic=%r difficulty=%s", topic, difficulty)
            return None

    async def _generate_document(
        self, topic: str, domain: str, difficulty: DifficultyLevel
    ) -> str:
        response = await self._gateway.complete(
            _SEED_PROMPT.format(topic=topic, difficulty=difficulty.value, domain=domain),
            system=_SEED_SYSTEM,
            provider=self._provider,
            max_tokens=600,
            temperature=0.7,
        )
        return response.content

    async def _generate_qa(self, topic: str, difficulty: DifficultyLevel) -> str:
        response = await self._gateway.complete(
            _SEED_QA_PROMPT.format(topic=topic, difficulty=difficulty.value),
            system=_SEED_QA_SYSTEM,
            provider=self._provider,
            max_tokens=400,
            temperature=0.5,
        )
        return response.content

    def _default_distribution(self, n_samples: int) -> dict[str, int]:
        """デフォルト難易度分布（beginner 多め）。"""
        base = max(1, n_samples // 4)
        remainder = n_samples - base * 4
        return {
            "beginner": base + remainder,
            "intermediate": base,
            "advanced": base,
            "expert": base,
        }
