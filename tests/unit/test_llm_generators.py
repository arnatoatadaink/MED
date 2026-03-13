"""tests/unit/test_llm_generators.py — ResponseGenerator / CodeGenerator の単体テスト"""

from __future__ import annotations

import pytest

from src.llm.code_generator import CodeGenerator
from src.llm.gateway import LLMGateway, LLMMessage, LLMResponse
from src.llm.response_generator import ResponseGenerator
from src.memory.schema import Document, SearchResult, SourceMeta, SourceType

# ──────────────────────────────────────────────
# モック Gateway
# ──────────────────────────────────────────────


class MockGateway(LLMGateway):
    def __init__(self, response_text: str = "mock response") -> None:
        self._response_text = response_text
        self._providers = {}
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0
        self.last_messages: list[LLMMessage] = []

    async def complete_messages(self, messages, **kwargs) -> LLMResponse:
        self.last_messages = list(messages)
        self._call_count += 1
        return LLMResponse(
            content=self._response_text,
            provider="mock",
            model="mock-model",
            input_tokens=50,
            output_tokens=100,
        )


def _make_search_result(content: str = "test content") -> SearchResult:
    doc = Document(
        content=content,
        domain="code",
        source=SourceMeta(source_type=SourceType.MANUAL),
    )
    return SearchResult(document=doc, score=0.9, query="test")


# ──────────────────────────────────────────────
# ResponseGenerator
# ──────────────────────────────────────────────


class TestResponseGenerator:
    @pytest.mark.asyncio
    async def test_generate_returns_response(self) -> None:
        gw = MockGateway("This is the answer.")
        gen = ResponseGenerator(gw)
        result = await gen.generate("test query", context_docs=[_make_search_result()])
        assert result.answer == "This is the answer."
        assert result.query == "test query"

    @pytest.mark.asyncio
    async def test_generate_empty_context(self) -> None:
        gw = MockGateway("Answer without context.")
        gen = ResponseGenerator(gw)
        result = await gen.generate("test query", context_docs=[])
        assert result.answer == "Answer without context."
        assert result.context_doc_ids == []

    @pytest.mark.asyncio
    async def test_generate_includes_system_prompt(self) -> None:
        gw = MockGateway()
        gen = ResponseGenerator(gw, system_prompt="custom system")
        await gen.generate("query", context_docs=[])
        roles = [m.role for m in gw.last_messages]
        assert "system" in roles
        system_msg = next(m for m in gw.last_messages if m.role == "system")
        assert system_msg.content == "custom system"

    @pytest.mark.asyncio
    async def test_generate_tracks_doc_ids(self) -> None:
        gw = MockGateway()
        gen = ResponseGenerator(gw, max_context_docs=3)
        docs = [_make_search_result(f"content {i}") for i in range(5)]
        result = await gen.generate("query", context_docs=docs)
        assert len(result.context_doc_ids) <= 3

    @pytest.mark.asyncio
    async def test_generate_token_counts(self) -> None:
        gw = MockGateway()
        gen = ResponseGenerator(gw)
        result = await gen.generate("query", context_docs=[])
        assert result.input_tokens == 50
        assert result.output_tokens == 100

    @pytest.mark.asyncio
    async def test_context_char_limit(self) -> None:
        gw = MockGateway()
        gen = ResponseGenerator(gw, max_context_chars=100)
        # 長いドキュメントが切り詰められることを確認
        long_doc = _make_search_result("A" * 1000)
        result = await gen.generate("query", context_docs=[long_doc])
        # プロンプトが構築できることを確認（エラーなし）
        assert result.answer is not None


# ──────────────────────────────────────────────
# CodeGenerator
# ──────────────────────────────────────────────


class TestCodeGenerator:
    @pytest.mark.asyncio
    async def test_generate_extracts_code_block(self) -> None:
        code_response = "```python\ndef hello():\n    return 'world'\n```\nThis function returns 'world'."
        gw = MockGateway(code_response)
        gen = CodeGenerator(gw)
        result = await gen.generate("write hello function")
        assert result.code == "def hello():\n    return 'world'"
        assert "world" in result.explanation
        assert result.is_complete is True

    @pytest.mark.asyncio
    async def test_generate_no_code_block_fallback(self) -> None:
        gw = MockGateway("just some text without code block")
        gen = CodeGenerator(gw)
        result = await gen.generate("task")
        assert result.code == "just some text without code block"
        assert result.is_complete is False

    @pytest.mark.asyncio
    async def test_default_language_python(self) -> None:
        gw = MockGateway("```python\nx = 1\n```")
        gen = CodeGenerator(gw)
        result = await gen.generate("task")
        assert result.language == "python"

    @pytest.mark.asyncio
    async def test_custom_language(self) -> None:
        gw = MockGateway("```javascript\nconsole.log('hi')\n```")
        gen = CodeGenerator(gw, default_language="javascript")
        result = await gen.generate("task", language="javascript")
        assert result.language == "javascript"
        assert "console.log" in result.code

    @pytest.mark.asyncio
    async def test_context_code_included_in_prompt(self) -> None:
        gw = MockGateway("```python\npass\n```")
        gen = CodeGenerator(gw)
        await gen.generate("extend this", context_code="def existing(): pass")
        prompt = next(m.content for m in gw.last_messages if m.role == "user")
        assert "existing" in prompt

    @pytest.mark.asyncio
    async def test_empty_context_code_excluded(self) -> None:
        gw = MockGateway("```python\npass\n```")
        gen = CodeGenerator(gw)
        await gen.generate("task", context_code="")
        prompt = next(m.content for m in gw.last_messages if m.role == "user")
        # コンテキストセクションがなければ "context_code" テキストも含まれない
        assert "Existing code context:" not in prompt

    @pytest.mark.asyncio
    async def test_result_has_metadata(self) -> None:
        gw = MockGateway("```python\npass\n```")
        gen = CodeGenerator(gw)
        result = await gen.generate("task")
        assert result.provider == "mock"
        assert result.model == "mock-model"
        assert result.task == "task"

    def test_extract_code_with_backtick_code_block(self) -> None:
        gen = CodeGenerator.__new__(CodeGenerator)
        content = "```\ndef foo():\n    pass\n```"
        code, explanation, complete = gen._extract_code(content, "python")
        assert "def foo" in code
        assert complete is True
