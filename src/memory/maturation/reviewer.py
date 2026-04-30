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
from src.memory.maturation.personas import AUTO_PERSONA, ReviewerPersona, get_persona_or_raise
from src.memory.metadata_store import MetadataStore
from src.memory.schema import Document, ReviewStatus

logger = logging.getLogger(__name__)

# thinking モデルの無効化は llm_config.local.yaml の extra_params で制御
# (fastflowlm: chat_template_kwargs.enable_thinking=false)
# thinking ON のまま低 temperature を指定すると合理化推論が発生し過承認の原因になる

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
- practical_reference: Man pages, wikis, command references, system documentation
  (e.g. Arch Wiki, Linux command references, Python stdlib docs). Approve if the
  content is accurate, actionable, and useful as a quick reference for system
  operation or programming — even if it is brief, tabular, or lacks narrative
  explanation. Do NOT penalise list/table format or terse style. Relevance criterion:
  is this useful to a developer or system administrator?

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


_REVIEW_PROMPT_PERSONA = """\
Document metadata:
- content_type: {content_type}
- categories: {categories}

Document text:
{text}"""


class MemoryReviewer:
    """Teacher Model でドキュメントを審査・品質スコア更新する。

    Args:
        gateway: LLMGateway インスタンス。
        store: MetadataStore インスタンス（品質スコア更新に使用）。
        provider: 優先プロバイダ（省略時はデフォルト）。
        persona: ペルソナ名 ("auto" / "on_domain" / "off_domain" /
                 "practical_reference" / "strict")。
                 "auto" (デフォルト) は doc の domain_flag で動的選択。
        max_text_length: LLM に渡す最大文字数。
    """

    def __init__(
        self,
        gateway: LLMGateway,
        store: MetadataStore,
        provider: str | None = None,
        model: str | None = None,
        persona: str = AUTO_PERSONA,
        max_text_length: int = 1200,
    ) -> None:
        self._gateway = gateway
        self._store = store
        self._provider = provider
        self._model = model
        self._max_text = max_text_length
        # auto 以外はペルソナを起動時に検証してキャッシュ
        if persona != AUTO_PERSONA:
            self._fixed_persona: ReviewerPersona | None = get_persona_or_raise(persona)
        else:
            self._fixed_persona = None
        self._persona_name = persona

    async def review(self, doc: Document) -> ReviewResult:
        """ドキュメントを審査し、MetadataStore を更新する。

        Returns:
            ReviewResult オブジェクト。
        """
        text = doc.content[:self._max_text]
        extra = doc.source.extra or {}
        content_type = extra.get("content_type", "unknown")
        categories = ", ".join(extra.get("categories", [])) or "unknown"
        domain_flag = extra.get("domain_flag", "unknown")

        if self._fixed_persona is not None:
            # 明示的ペルソナ: ペルソナ固有のシステムプロンプト + domain_flag なしの短いプロンプト
            system = self._fixed_persona.system_prompt
            persona_label = self._fixed_persona.name
            threshold = self._fixed_persona.approval_threshold
            prompt = _REVIEW_PROMPT_PERSONA.format(
                content_type=content_type,
                categories=categories,
                text=text,
            )
        else:
            # auto: domain_flag をプロンプトに渡して LLM が動的選択
            system = _REVIEW_SYSTEM
            persona_label = domain_flag
            threshold = 0.6
            prompt = _REVIEW_PROMPT.format(
                content_type=content_type,
                categories=categories,
                domain_flag=domain_flag,
                text=text,
            )

        try:
            response = await self._gateway.complete(
                prompt,
                system=system,
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

        # quality >= threshold なら LLM の approved 判定を優先（ペルソナ固有閾値を使用）
        if not approved and not needs_supplement and quality_score >= threshold:
            approved = True
        # quality >= 0.7 なら needs_supplement でも APPROVED として扱う（内容は十分）
        if needs_supplement and quality_score >= 0.7:
            needs_supplement = False
            approved = True
        if needs_supplement:
            review_status = ReviewStatus.NEEDS_UPDATE
        elif approved:
            review_status = ReviewStatus.APPROVED
        else:
            review_status = ReviewStatus.HOLD  # REJECTED → HOLD（再審査可能）

        # composite_score: teacher_quality * 0.6 + confidence * 0.4
        composite_score = round(quality_score * 0.6 + confidence * 0.4, 4)

        # teacher_id: "provider/model" 形式で審査モデルを記録
        teacher_id: str | None = None
        if self._provider or self._model:
            parts = [p for p in (self._provider, self._model) if p]
            teacher_id = "/".join(parts)

        # documents テーブル更新（後方互換 — 最終審査結果を保持）
        try:
            await self._store.update_quality(
                doc.id,
                teacher_quality=quality_score,
                review_status=review_status.value,
                confidence=confidence,
                composite_score=composite_score,
                teacher_id=teacher_id,
            )
        except Exception:
            logger.exception("Failed to update quality for doc=%s", doc.id)

        # doc_reviews テーブル更新（モデル×ペルソナ単位で競合なく保存）
        try:
            await self._store.save_review(
                doc_id=doc.id,
                teacher_id=teacher_id or "unknown",
                persona=persona_label,
                quality_score=quality_score,
                confidence=confidence,
                approved=approved,
                needs_supplement=needs_supplement,
                reason=reason,
                composite_score=composite_score,
            )
        except Exception:
            logger.exception("Failed to save doc_review for doc=%s", doc.id)

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
