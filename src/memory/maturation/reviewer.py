"""src/memory/maturation/reviewer.py — Teacher 品質審査

Teacher Model が FAISSメモリ内のドキュメントを審査し、
品質スコア（teacher_quality）・難易度・ReviewStatus を更新する。

メモリ品質目標（Phase 2）:
- 10,000 docs
- confidence > 0.7
- 実行成功率 > 80%

使い方:
    from src.memory.maturation.reviewer import MemoryReviewer

    reviewer = MemoryReviewer(gateway, store)
    result = await reviewer.review(doc)
    print(result.quality_score, result.approved)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from src.llm.gateway import LLMGateway
from src.memory.metadata_store import MetadataStore
from src.memory.schema import Document, ReviewStatus

logger = logging.getLogger(__name__)

_REVIEW_SYSTEM = """\
You are a quality reviewer for a technical knowledge base.
Evaluate the given document and respond with ONLY valid JSON:
{
  "quality_score": 0.0-1.0,
  "confidence": 0.0-1.0,
  "approved": true/false,
  "reason": "brief explanation"
}

Quality criteria:
- Accuracy: Is the information correct?
- Completeness: Is it self-contained and useful?
- Clarity: Is it clear and well-written?
- Relevance: Is it relevant for technical learning?

Approve if quality_score >= 0.6"""

_REVIEW_PROMPT = "Review this document:\n\n{text}"


@dataclass
class ReviewResult:
    """審査結果。"""

    doc_id: str
    quality_score: float
    confidence: float
    approved: bool
    reason: str
    review_status: ReviewStatus


class MemoryReviewer:
    """Teacher Model でドキュメントを審査・品質スコア更新する。

    Args:
        gateway: LLMGateway インスタンス。
        store: MetadataStore インスタンス（品質スコア更新に使用）。
        provider: 優先プロバイダ（省略時はデフォルト）。
        max_text_length: LLM に渡す最大文字数。
        approval_threshold: この値以上で approved=True。
    """

    def __init__(
        self,
        gateway: LLMGateway,
        store: MetadataStore,
        provider: str | None = None,
        max_text_length: int = 1200,
        approval_threshold: float = 0.6,
    ) -> None:
        self._gateway = gateway
        self._store = store
        self._provider = provider
        self._max_text = max_text_length
        self._threshold = approval_threshold

    async def review(self, doc: Document) -> ReviewResult:
        """ドキュメントを審査し、MetadataStore を更新する。

        Returns:
            ReviewResult オブジェクト。
        """
        text = doc.content[:self._max_text]
        prompt = _REVIEW_PROMPT.format(text=text)

        try:
            response = await self._gateway.complete(
                prompt,
                system=_REVIEW_SYSTEM,
                provider=self._provider,
                max_tokens=200,
                temperature=0.0,
            )
            parsed = self._parse_response(response.content)
        except Exception:
            logger.exception("Review failed for doc=%s; marking as failed", doc.id)
            parsed = {
                "quality_score": 0.0,
                "confidence": 0.3,
                "approved": False,
                "reason": "Review failed due to LLM error",
            }

        quality_score = float(parsed.get("quality_score", 0.0))
        confidence = float(parsed.get("confidence", 0.5))
        approved = bool(parsed.get("approved", False))
        reason = str(parsed.get("reason", ""))

        review_status = ReviewStatus.APPROVED if approved else ReviewStatus.REJECTED

        # MetadataStore を更新
        try:
            await self._store.update_quality(
                doc.id,
                teacher_quality=quality_score,
                review_status=review_status.value,
                confidence=confidence,
            )
        except Exception:
            logger.exception("Failed to update quality for doc=%s", doc.id)

        result = ReviewResult(
            doc_id=doc.id,
            quality_score=quality_score,
            confidence=confidence,
            approved=approved,
            reason=reason,
            review_status=review_status,
        )
        logger.debug(
            "Reviewed doc=%s: quality=%.2f approved=%s",
            doc.id, quality_score, approved,
        )
        return result

    async def review_batch(
        self,
        docs: list[Document],
        concurrency: int = 5,
    ) -> list[ReviewResult]:
        """複数ドキュメントを並列審査する。"""
        import asyncio
        semaphore = asyncio.Semaphore(concurrency)

        async def _review_with_sem(doc: Document) -> ReviewResult:
            async with semaphore:
                return await self.review(doc)

        return list(await asyncio.gather(*[_review_with_sem(d) for d in docs]))

    async def review_unreviewed(
        self,
        limit: int = 100,
    ) -> list[ReviewResult]:
        """未審査ドキュメントを一括審査する。"""
        docs = await self._store.get_unreviewed(limit=limit)
        if not docs:
            logger.info("No unreviewed documents found")
            return []
        logger.info("Reviewing %d unreviewed documents", len(docs))
        return await self.review_batch(docs)

    def _parse_response(self, content: str) -> dict:
        content = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("`").strip()
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            content = match.group(0)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse review JSON: %r", content[:200])
            return {"quality_score": 0.5, "confidence": 0.5, "approved": False, "reason": "parse error"}
