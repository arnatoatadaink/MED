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
  "needs_supplement": true/false,
  "reason": "brief explanation"
}

Quality criteria:
- Accuracy: Is the information correct?
- Completeness: Is it self-contained and useful?
- Clarity: Is it clear and well-written?
- Relevance: Is it relevant for technical learning?

Note on domain_flag:
- on_domain: CS/ML content. Apply standard quality criteria.
- off_domain: Non-CS/ML field (physics, math, etc.). This content is intentionally
  retained for associative memory diversity. Approve if the document is high-quality
  within its own field, even if not directly CS/ML relevant. Lower the relevance
  weight and focus on accuracy and clarity instead.

Set needs_supplement=true if the document meets ANY of these conditions:
1. Fragment / incomplete: truncated mid-sentence, missing context to be understood
   standalone, or is clearly a partial excerpt needing surrounding content.
2. Thin / shallow: fewer than ~3 meaningful sentences of substance, only contains
   a title/header/install command with no explanation, or is a navigation/UI
   description with no actual knowledge content.

When needs_supplement=true, set approved=false regardless of quality_score.
Approve if quality_score >= 0.6 AND needs_supplement=false."""

_REVIEW_PROMPT = """\
Document metadata:
- content_type: {content_type}
- categories: {categories}
- domain_flag: {domain_flag}

Document text:
{text}"""


@dataclass
class ReviewResult:
    """審査結果。"""

    doc_id: str
    quality_score: float
    confidence: float
    approved: bool
    needs_supplement: bool
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
        model: str | None = None,
        max_text_length: int = 1200,
        approval_threshold: float = 0.6,
    ) -> None:
        self._gateway = gateway
        self._store = store
        self._provider = provider
        self._model = model
        self._max_text = max_text_length
        self._threshold = approval_threshold

    async def review(self, doc: Document) -> ReviewResult:
        """ドキュメントを審査し、MetadataStore を更新する。

        Returns:
            ReviewResult オブジェクト。
        """
        import json as _json
        text = doc.content[:self._max_text]
        extra = doc.source.extra or {}
        content_type = extra.get("content_type", "unknown")
        categories = ", ".join(extra.get("categories", [])) or "unknown"
        domain_flag = extra.get("domain_flag", "unknown")
        prompt = _REVIEW_PROMPT.format(
            content_type=content_type,
            categories=categories,
            domain_flag=domain_flag,
            text=text,
        )

        try:
            response = await self._gateway.complete(
                prompt,
                system=_REVIEW_SYSTEM,
                provider=self._provider,
                model=self._model,
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
        needs_supplement = bool(parsed.get("needs_supplement", False))
        reason = str(parsed.get("reason", ""))

        # quality >= 0.7 なら needs_supplement でも APPROVED として扱う（内容は十分）
        if needs_supplement and quality_score >= 0.7:
            needs_supplement = False  # 品質十分なので保留扱いしない
            approved = True
        if needs_supplement:
            review_status = ReviewStatus.NEEDS_UPDATE
        elif approved:
            review_status = ReviewStatus.APPROVED
        else:
            review_status = ReviewStatus.REJECTED

        # MetadataStore を更新
        try:
            await self._store.update_quality(
                doc.id,
                teacher_quality=quality_score,
                review_status=review_status.value,
                confidence=confidence,
            )
            # rejected 時はブラックリストに登録（同一 URL/タイトルの再取得を防ぐ）
            if review_status == ReviewStatus.REJECTED:
                await self._store.add_to_blacklist(
                    source_type=doc.source.source_type.value if doc.source else "",
                    source_url=doc.source.url or "" if doc.source else "",
                    source_title=doc.source.title or "" if doc.source else "",
                    reason="rejected",
                )
        except Exception:
            logger.exception("Failed to update quality for doc=%s", doc.id)

        result = ReviewResult(
            doc_id=doc.id,
            quality_score=quality_score,
            confidence=confidence,
            approved=approved,
            needs_supplement=needs_supplement,
            reason=reason,
            review_status=review_status,
        )
        logger.debug(
            "Reviewed doc=%s: quality=%.2f approved=%s needs_supplement=%s",
            doc.id, quality_score, approved, needs_supplement,
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
