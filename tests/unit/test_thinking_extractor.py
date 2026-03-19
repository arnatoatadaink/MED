"""tests/unit/test_thinking_extractor.py — ThinkingExtractor 単体テスト"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm.gateway import LLMResponse
from src.llm.thinking_extractor import (
    ThinkingExtractor,
    _parse_json_or_lines,
    _parse_xml_section,
)
from src.memory.schema import KnowledgeType, ReasoningTrace, TraceMethod

# ── ヘルパー関数テスト ────────────────────────────────────────────────────

class TestParseXmlSection:
    def test_extracts_content(self):
        text = "<foo>hello world</foo>"
        assert _parse_xml_section(text, "foo") == "hello world"

    def test_multiline(self):
        text = "<foo>\nline1\nline2\n</foo>"
        result = _parse_xml_section(text, "foo")
        assert "line1" in result
        assert "line2" in result

    def test_missing_tag_returns_empty(self):
        assert _parse_xml_section("no tags here", "foo") == ""

    def test_strips_whitespace(self):
        assert _parse_xml_section("<foo>  content  </foo>", "foo") == "content"


class TestParseJsonOrLines:
    def test_json_array(self):
        result = _parse_json_or_lines('["a", "b", "c"]')
        assert result == ["a", "b", "c"]

    def test_json_array_of_dicts(self):
        data = [{"item": "x", "kind": "factual"}]
        result = _parse_json_or_lines(json.dumps(data))
        assert result == data

    def test_line_list_with_dashes(self):
        text = "- item1\n- item2\n- item3"
        result = _parse_json_or_lines(text)
        assert result == ["item1", "item2", "item3"]

    def test_plain_lines(self):
        text = "step1\nstep2\nstep3"
        result = _parse_json_or_lines(text)
        assert result == ["step1", "step2", "step3"]

    def test_empty_string(self):
        assert _parse_json_or_lines("") == []

    def test_whitespace_only(self):
        assert _parse_json_or_lines("   \n   ") == []


# ── モックファクトリー ───────────────────────────────────────────────────

def _make_gateway(
    *,
    provider_name: str = "openai",
    has_anthropic: bool = False,
    content: str = "answer text",
    thinking_text: str | None = None,
    thinking_tokens: int = 0,
) -> MagicMock:
    gateway = MagicMock()
    gateway._providers = {}

    if has_anthropic:
        anthropic_prov = MagicMock()
        anthropic_prov.is_available.return_value = True
        gateway._providers["anthropic"] = anthropic_prov

    response = LLMResponse(
        content=content,
        provider=provider_name,
        model="test-model",
        input_tokens=10,
        output_tokens=20,
        thinking_text=thinking_text,
        thinking_tokens=thinking_tokens,
    )
    gateway.complete = AsyncMock(return_value=response)
    return gateway


# ── ThinkingExtractor テスト ─────────────────────────────────────────────

class TestThinkingExtractorIsAnthropic:
    def test_explicit_anthropic_provider(self):
        gw = _make_gateway()
        extractor = ThinkingExtractor(gw)
        assert extractor._is_anthropic("anthropic") is True

    def test_explicit_openai_provider(self):
        gw = _make_gateway()
        extractor = ThinkingExtractor(gw)
        assert extractor._is_anthropic("openai") is False

    def test_none_with_anthropic_available(self):
        gw = _make_gateway(has_anthropic=True)
        extractor = ThinkingExtractor(gw)
        assert extractor._is_anthropic(None) is True

    def test_none_without_anthropic(self):
        gw = _make_gateway(has_anthropic=False)
        extractor = ThinkingExtractor(gw)
        assert extractor._is_anthropic(None) is False


class TestThinkingExtractorFormatContext:
    def test_none_docs(self):
        result = ThinkingExtractor._format_context(None)
        assert "なし" in result or "ドキュメント" in result

    def test_empty_list(self):
        result = ThinkingExtractor._format_context([])
        assert "なし" in result or "ドキュメント" in result

    def test_document_with_content_attr(self):
        doc = MagicMock()
        doc.content = "test content"
        doc.id = "d1"
        del doc.document  # document 属性なし
        result = ThinkingExtractor._format_context([doc])
        assert "test content" in result
        assert "d1" in result

    def test_search_result_with_document(self):
        inner = MagicMock()
        inner.content = "inner content"
        inner.id = "inner_id"
        sr = MagicMock()
        sr.document = inner
        result = ThinkingExtractor._format_context([sr])
        assert "inner content" in result


class TestExtractCoTPrompt:
    """非Anthropicプロバイダ向けCoT抽出テスト"""

    @pytest.mark.asyncio
    async def test_basic_cot_response(self):
        gw = _make_gateway(
            provider_name="openai",
            has_anthropic=False,
            content="Some answer",
        )
        extractor = ThinkingExtractor(gw, default_provider="openai")
        response, trace = await extractor.extract(
            "Test query", enable_extended_thinking=False
        )
        assert response.content == "Some answer"
        assert trace.trace_method == TraceMethod.COT_PROMPT
        assert trace.query == "Test query"
        assert trace.confidence == 0.5

    @pytest.mark.asyncio
    async def test_cot_parses_xml_sections(self):
        content = """
<knowledge_audit>["step a", "step b"]</knowledge_audit>
<reasoning_chain>["reason1", "reason2"]</reasoning_chain>
<judgment_criteria>["criteria1"]</judgment_criteria>
<retrieval_rationale>[{"doc_id": "d1", "why": "relevant", "trust": "高"}]</retrieval_rationale>
Final answer here.
"""
        gw = _make_gateway(
            provider_name="openai",
            has_anthropic=False,
            content=content,
        )
        extractor = ThinkingExtractor(gw, default_provider="openai")
        _, trace = await extractor.extract(
            "Test query", enable_extended_thinking=False
        )
        assert trace.knowledge_audit == ["step a", "step b"]
        assert trace.reasoning_chain == ["reason1", "reason2"]
        assert trace.judgment_criteria == ["criteria1"]
        assert trace.retrieval_rationale is not None
        assert trace.raw_thinking == content  # CoT では raw_thinking に全体保存

    @pytest.mark.asyncio
    async def test_cot_knowledge_type_inference(self):
        content = """
<knowledge_audit>[{"item":"x","kind":"factual","needs_retrieval":false,"confidence":"高"}]</knowledge_audit>
Answer.
"""
        gw = _make_gateway(
            provider_name="openai",
            has_anthropic=False,
            content=content,
        )
        extractor = ThinkingExtractor(gw, default_provider="openai")
        _, trace = await extractor.extract(
            "Query", enable_extended_thinking=False
        )
        assert trace.primary_knowledge_type == KnowledgeType.FACTUAL


class TestExtractExtendedThinking:
    """Anthropic Extended Thinking 抽出テスト"""

    @pytest.mark.asyncio
    async def test_extended_thinking_uses_anthropic(self):
        gw = _make_gateway(
            provider_name="anthropic",
            has_anthropic=True,
            content="ET answer",
            thinking_text="<reasoning_chain>[\"think1\"]</reasoning_chain>",
            thinking_tokens=200,
        )
        extractor = ThinkingExtractor(gw, default_provider="anthropic")
        response, trace = await extractor.extract(
            "ET query",
            enable_extended_thinking=True,
            thinking_budget_tokens=5000,
        )

        # gateway.complete は enable_thinking=True で呼ばれる
        call_kwargs = gw.complete.call_args[1]
        assert call_kwargs.get("enable_thinking") is True
        assert call_kwargs.get("thinking_budget_tokens") == 5000

        assert trace.trace_method == TraceMethod.EXTENDED_THINKING
        assert trace.thinking_tokens == 200
        assert trace.confidence == 0.8

    @pytest.mark.asyncio
    async def test_extended_thinking_parses_thinking_text(self):
        thinking = """
<knowledge_audit>[{"item":"FAISS","kind":"factual","needs_retrieval":false,"confidence":"高"}]</knowledge_audit>
<reasoning_chain>["step1","step2"]</reasoning_chain>
"""
        gw = _make_gateway(
            provider_name="anthropic",
            has_anthropic=True,
            content="answer",
            thinking_text=thinking,
            thinking_tokens=100,
        )
        extractor = ThinkingExtractor(gw, default_provider="anthropic")
        _, trace = await extractor.extract("Q", enable_extended_thinking=True)

        assert trace.reasoning_chain == ["step1", "step2"]
        assert trace.knowledge_audit is not None

    @pytest.mark.asyncio
    async def test_no_thinking_text_leaves_fields_none(self):
        gw = _make_gateway(
            provider_name="anthropic",
            has_anthropic=True,
            content="plain answer",
            thinking_text=None,
        )
        extractor = ThinkingExtractor(gw, default_provider="anthropic")
        _, trace = await extractor.extract("Q", enable_extended_thinking=True)

        assert trace.raw_thinking is None
        assert trace.reasoning_chain == []

    @pytest.mark.asyncio
    async def test_falls_back_to_cot_when_extended_disabled(self):
        gw = _make_gateway(has_anthropic=True, content="cot answer")
        extractor = ThinkingExtractor(gw, default_provider="anthropic")
        _, trace = await extractor.extract(
            "Q", enable_extended_thinking=False
        )
        assert trace.trace_method == TraceMethod.COT_PROMPT


# ── reasoning_trace ストレージ統合テスト ─────────────────────────────────

class TestReasoningTraceStorage:
    """metadata_store の reasoning_traces CRUD テスト"""

    @pytest.mark.asyncio
    async def test_save_and_get(self):
        from src.memory.metadata_store import MetadataStore

        store = MetadataStore(db_path=":memory:")
        await store.initialize()

        trace = ReasoningTrace(
            query="What is FAISS?",
            answer="FAISS is a vector search library.",
            raw_thinking="thinking content",
            trace_method=TraceMethod.EXTENDED_THINKING,
            teacher_model="claude-opus-4-6",
            teacher_provider="anthropic",
            thinking_tokens=150,
            confidence=0.8,
            reasoning_chain=["step1", "step2"],
            primary_knowledge_type=KnowledgeType.FACTUAL,
            doc_ids=["doc_1", "doc_2"],
        )

        trace_id = await store.save_reasoning_trace(trace)
        assert trace_id == trace.id

        fetched = await store.get_reasoning_trace(trace_id)
        assert fetched is not None
        assert fetched.query == "What is FAISS?"
        assert fetched.trace_method == TraceMethod.EXTENDED_THINKING
        assert set(fetched.doc_ids) == {"doc_1", "doc_2"}
        assert fetched.reasoning_chain == ["step1", "step2"]
        assert fetched.primary_knowledge_type == KnowledgeType.FACTUAL

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self):
        from src.memory.metadata_store import MetadataStore

        store = MetadataStore(db_path=":memory:")
        await store.initialize()
        result = await store.get_reasoning_trace("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_traces(self):
        from src.memory.metadata_store import MetadataStore

        store = MetadataStore(db_path=":memory:")
        await store.initialize()

        for i in range(3):
            t = ReasoningTrace(
                query=f"Query {i}",
                answer=f"Answer {i}",
                trace_method=TraceMethod.COT_PROMPT,
                teacher_model="gpt-4o",
                teacher_provider="openai",
            )
            await store.save_reasoning_trace(t)

        listed = await store.list_reasoning_traces(limit=10)
        assert len(listed) == 3

    @pytest.mark.asyncio
    async def test_list_filter_by_method(self):
        from src.memory.metadata_store import MetadataStore

        store = MetadataStore(db_path=":memory:")
        await store.initialize()

        for method in [TraceMethod.EXTENDED_THINKING, TraceMethod.COT_PROMPT, TraceMethod.EXTENDED_THINKING]:
            t = ReasoningTrace(
                query="Q",
                answer="A",
                trace_method=method,
                teacher_model="m",
                teacher_provider="p",
            )
            await store.save_reasoning_trace(t)

        et_list = await store.list_reasoning_traces(trace_method=TraceMethod.EXTENDED_THINKING)
        assert len(et_list) == 2

        cot_list = await store.list_reasoning_traces(trace_method=TraceMethod.COT_PROMPT)
        assert len(cot_list) == 1

    @pytest.mark.asyncio
    async def test_upsert_overwrites(self):
        from src.memory.metadata_store import MetadataStore

        store = MetadataStore(db_path=":memory:")
        await store.initialize()

        trace = ReasoningTrace(
            query="Q",
            answer="original",
            trace_method=TraceMethod.COT_PROMPT,
            teacher_model="m",
            teacher_provider="p",
        )
        await store.save_reasoning_trace(trace)

        # 同じ ID で上書き
        trace.answer = "updated"
        await store.save_reasoning_trace(trace)

        fetched = await store.get_reasoning_trace(trace.id)
        assert fetched.answer == "updated"

        listed = await store.list_reasoning_traces()
        assert len(listed) == 1  # 重複なし
