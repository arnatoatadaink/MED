"""src/llm/thinking_extractor.py — Teacher 思考過程の抽出

Anthropic: Extended Thinking API を呼び出し ThinkingBlock を取得
他プロバイダ: CoTプロンプト（reasoning_extraction.yaml）で構造化思考を引き出す

技術的注意点:
- Extended Thinking: Anthropicが公式提供するAPI機能（ToS準拠）
- CoTプロンプト: Wei et al. (2022) で確立した公開手法
- CoTで引き出した思考は post-hoc rationalization の可能性がある
  （モデルの実際の計算過程ではなく「もっともらしい説明の後付け生成」）
- Extended Thinkingのほうが実際の推論トークン列に近く信頼性が高い
- 保存データには trace_method で区別する

使い方:
    extractor = ThinkingExtractor(gateway)
    response, trace = await extractor.extract(query, context_docs)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from src.llm.gateway import LLMGateway, LLMMessage, LLMResponse
from src.memory.schema import KnowledgeType, ReasoningTrace, TraceMethod

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompt_templates" / "reasoning_extraction.yaml"


def _load_cot_prompt() -> dict[str, str]:
    """CoT 抽出プロンプトテンプレートを読み込む。"""
    if _PROMPT_PATH.exists():
        with open(_PROMPT_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {"system": "", "user_template": "{query}"}


def _parse_xml_section(text: str, tag: str) -> str:
    """<tag>...</tag> の中身を抽出する。"""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _parse_json_or_lines(text: str) -> list[Any]:
    """JSON配列 or 行リストとしてパースする。"""
    text = text.strip()
    if not text:
        return []
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    # 行ごとに分割（- 付きリスト or 改行区切り）
    lines = [line.strip().lstrip("- ").strip() for line in text.split("\n") if line.strip()]
    return lines


class ThinkingExtractor:
    """思考過程の抽出を担当するファサード。

    Anthropic: Extended Thinking API を呼び出し ThinkingBlock を取得
    他プロバイダ: CoTプロンプト（reasoning_extraction.yaml）で構造化思考を引き出す

    Args:
        gateway: LLMGateway。
        default_provider: デフォルトプロバイダ（None で gateway デフォルト）。
    """

    def __init__(
        self,
        gateway: LLMGateway,
        default_provider: str | None = None,
    ) -> None:
        self._gateway = gateway
        self._default_provider = default_provider
        self._cot_prompt = _load_cot_prompt()

    async def extract(
        self,
        query: str,
        context_docs: list[Any] | None = None,
        provider: str | None = None,
        *,
        enable_extended_thinking: bool = True,
        thinking_budget_tokens: int = 8000,
    ) -> tuple[LLMResponse, ReasoningTrace]:
        """クエリに対してTeacher呼び出しを行い、回答と思考過程を返す。

        Args:
            query: ユーザークエリ。
            context_docs: 参照ドキュメント（SearchResult or Document リスト）。
            provider: 使用するプロバイダ（None でデフォルト）。
            enable_extended_thinking: Anthropic Extended Thinking を使うか。
            thinking_budget_tokens: thinking ブロックの最大トークン数。

        Returns:
            (LLMResponse, ReasoningTrace) のタプル。
        """
        target_provider = provider or self._default_provider

        # Anthropic プロバイダで Extended Thinking を使う場合
        if enable_extended_thinking and self._is_anthropic(target_provider):
            return await self._extract_extended_thinking(
                query, context_docs, target_provider, thinking_budget_tokens,
            )
        else:
            return await self._extract_cot_prompt(
                query, context_docs, target_provider,
            )

    # ── Extended Thinking（Anthropic）────────────────────────────

    async def _extract_extended_thinking(
        self,
        query: str,
        context_docs: list[Any] | None,
        provider: str | None,
        thinking_budget_tokens: int,
    ) -> tuple[LLMResponse, ReasoningTrace]:
        """Extended Thinking API で思考過程を取得する。"""
        context_text = self._format_context(context_docs)
        messages = [LLMMessage(role="user", content=f"{context_text}\n\n{query}")]

        response = await self._gateway.complete(
            messages,
            provider=provider,
            enable_thinking=True,
            thinking_budget_tokens=thinking_budget_tokens,
        )

        trace = ReasoningTrace(
            query=query,
            answer=response.content,
            raw_thinking=response.thinking_text,
            trace_method=TraceMethod.EXTENDED_THINKING,
            teacher_model=response.model,
            teacher_provider=response.provider,
            thinking_tokens=response.thinking_tokens,
            confidence=0.8,  # Extended Thinking は CoT より信頼度高
        )

        # raw_thinking から構造化情報を抽出する試み
        if response.thinking_text:
            self._parse_raw_thinking(trace, response.thinking_text)

        return response, trace

    # ── CoT プロンプト（他プロバイダ）────────────────────────────

    async def _extract_cot_prompt(
        self,
        query: str,
        context_docs: list[Any] | None,
        provider: str | None,
    ) -> tuple[LLMResponse, ReasoningTrace]:
        """CoTプロンプトで構造化思考を引き出す。"""
        context_text = self._format_context(context_docs)

        system_prompt = self._cot_prompt.get("system", "")
        user_template = self._cot_prompt.get("user_template", "{query}")
        user_content = user_template.format(query=query, context=context_text)

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_content),
        ]
        response = await self._gateway.complete(messages, provider=provider)

        trace = ReasoningTrace(
            query=query,
            answer=response.content,
            trace_method=TraceMethod.COT_PROMPT,
            teacher_model=response.model,
            teacher_provider=response.provider,
            confidence=0.5,  # CoT は post-hoc rationalization の可能性あり
        )

        # レスポンスから XML セクションをパース
        self._parse_cot_response(trace, response.content)

        return response, trace

    # ── パーサー ────────────────────────────────────────────────

    def _parse_raw_thinking(self, trace: ReasoningTrace, thinking: str) -> None:
        """Extended Thinking の生テキストから構造化情報を抽出する（ベストエフォート）。"""
        # XML タグがある場合はパース
        ka = _parse_xml_section(thinking, "knowledge_audit")
        if ka:
            trace.knowledge_audit = _parse_json_or_lines(ka)

        rc = _parse_xml_section(thinking, "reasoning_chain")
        if rc:
            trace.reasoning_chain = _parse_json_or_lines(rc)

        jc = _parse_xml_section(thinking, "judgment_criteria")
        if jc:
            trace.judgment_criteria = _parse_json_or_lines(jc)

        rr = _parse_xml_section(thinking, "retrieval_rationale")
        if rr:
            trace.retrieval_rationale = _parse_json_or_lines(rr)

    def _parse_cot_response(self, trace: ReasoningTrace, content: str) -> None:
        """CoT レスポンスの XML セクションをパースする。"""
        ka = _parse_xml_section(content, "knowledge_audit")
        if ka:
            trace.knowledge_audit = _parse_json_or_lines(ka)
            # 思考部分を raw_thinking に保存
            trace.raw_thinking = content

        rc = _parse_xml_section(content, "reasoning_chain")
        if rc:
            trace.reasoning_chain = _parse_json_or_lines(rc)

        jc = _parse_xml_section(content, "judgment_criteria")
        if jc:
            trace.judgment_criteria = _parse_json_or_lines(jc)

        rr = _parse_xml_section(content, "retrieval_rationale")
        if rr:
            trace.retrieval_rationale = _parse_json_or_lines(rr)

        # primary_knowledge_type を推定
        if trace.knowledge_audit:
            kinds = [
                item.get("kind", "factual") if isinstance(item, dict) else "factual"
                for item in trace.knowledge_audit
            ]
            # 最頻出の kind を primary にする
            if kinds:
                from collections import Counter
                most_common = Counter(kinds).most_common(1)[0][0]
                try:
                    trace.primary_knowledge_type = KnowledgeType(most_common)
                except ValueError:
                    pass

    # ── ユーティリティ ──────────────────────────────────────────

    def _is_anthropic(self, provider: str | None) -> bool:
        """指定プロバイダが Anthropic かどうかを判定する。"""
        if provider == "anthropic":
            return True
        if provider is None:
            # デフォルトプロバイダが anthropic かチェック
            return "anthropic" in self._gateway._providers and \
                self._gateway._providers["anthropic"].is_available()
        return False

    @staticmethod
    def _format_context(docs: list[Any] | None) -> str:
        """コンテキストドキュメントをテキストにフォーマットする。"""
        if not docs:
            return "（参照ドキュメントなし）"
        parts = []
        for i, doc in enumerate(docs, 1):
            if hasattr(doc, "document"):
                # SearchResult
                content = doc.document.content
                doc_id = doc.document.id
            elif hasattr(doc, "content"):
                # Document
                content = doc.content
                doc_id = getattr(doc, "id", f"doc_{i}")
            else:
                content = str(doc)
                doc_id = f"doc_{i}"
            parts.append(f"[{doc_id}] {content[:500]}")
        return "\n\n".join(parts)
