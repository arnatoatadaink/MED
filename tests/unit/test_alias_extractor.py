"""tests/unit/test_alias_extractor.py — AliasExtractor 単体テスト"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm.gateway import LLMResponse
from src.memory.alias_extractor import AliasExtractor
from src.memory.schema import Document, SourceMeta, SourceType


def _make_doc(content: str, title: str = "") -> Document:
    source = SourceMeta(type=SourceType.MANUAL, title=title or None)
    return Document(content=content, domain="general", source=source)


def _make_gateway(response_text: str) -> MagicMock:
    gw = MagicMock()
    gw.complete = AsyncMock(return_value=LLMResponse(
        content=response_text,
        provider="mock",
        model="mock-model",
        input_tokens=10,
        output_tokens=10,
    ))
    return gw


class TestAliasExtractorEnabled:
    @pytest.mark.asyncio
    async def test_returns_aliases_from_llm(self):
        gw = _make_gateway('["TinyLoRA", "13-param LoRA", "Tiny LoRA"]')
        ext = AliasExtractor(gw)
        doc = _make_doc("TinyLoRA achieves GSM8K 91% with 13 parameters.")
        result = await ext.extract(doc)
        assert result == ["TinyLoRA", "13-param LoRA", "Tiny LoRA"]

    @pytest.mark.asyncio
    async def test_gateway_called_with_truncated_content(self):
        gw = _make_gateway("[]")
        ext = AliasExtractor(gw)
        long_content = "A" * 2000
        doc = _make_doc(long_content)
        await ext.extract(doc)
        call_args = gw.complete.call_args
        messages = call_args[0][0]
        # The user message should contain at most 800 chars of the content
        user_content = messages[0].content
        assert len(user_content) < 1000

    @pytest.mark.asyncio
    async def test_title_prepended_to_prompt(self):
        gw = _make_gateway("[]")
        ext = AliasExtractor(gw)
        doc = _make_doc("Some content.", title="TinyLoRA Paper")
        await ext.extract(doc)
        messages = gw.complete.call_args[0][0]
        assert "TinyLoRA Paper" in messages[0].content

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_disabled(self):
        gw = _make_gateway('["something"]')
        ext = AliasExtractor(gw, enabled=False)
        doc = _make_doc("Content")
        result = await ext.extract(doc)
        assert result == []
        gw.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_gateway_exception_returns_empty(self):
        gw = MagicMock()
        gw.complete = AsyncMock(side_effect=RuntimeError("API error"))
        ext = AliasExtractor(gw)
        doc = _make_doc("Some content.")
        result = await ext.extract(doc)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_empty_array(self):
        gw = _make_gateway("[]")
        ext = AliasExtractor(gw)
        doc = _make_doc("Plain text with no aliases.")
        result = await ext.extract(doc)
        assert result == []


class TestAliasExtractorParseAliases:
    """_parse_aliases の単体テスト（プロンプトインジェクション対策含む）"""

    def test_clean_json_array(self):
        result = AliasExtractor._parse_aliases('["A", "B", "C"]')
        assert result == ["A", "B", "C"]

    def test_with_surrounding_text(self):
        result = AliasExtractor._parse_aliases('Here are the aliases: ["X", "Y"]')
        assert result == ["X", "Y"]

    def test_invalid_json_returns_empty(self):
        result = AliasExtractor._parse_aliases("not json at all")
        assert result == []

    def test_not_a_list_returns_empty(self):
        result = AliasExtractor._parse_aliases('{"key": "value"}')
        assert result == []

    def test_non_string_elements_filtered(self):
        result = AliasExtractor._parse_aliases('[1, "A", null, "B"]')
        assert result == ["A", "B"]

    def test_too_long_element_filtered(self):
        long_str = "X" * 201
        raw = json.dumps([long_str, "valid"])
        result = AliasExtractor._parse_aliases(raw)
        assert long_str not in result
        assert "valid" in result

    def test_empty_string_filtered(self):
        result = AliasExtractor._parse_aliases('["", "  ", "valid"]')
        assert "" not in result
        assert "valid" in result

    def test_whitespace_stripped(self):
        result = AliasExtractor._parse_aliases('["  TinyLoRA  "]')
        assert "TinyLoRA" in result

    def test_injection_with_extra_braces(self):
        # Attempts to inject additional JSON objects
        raw = 'Ignore above. ["real_alias"] {"evil": "payload"}'
        result = AliasExtractor._parse_aliases(raw)
        assert result == ["real_alias"]

    def test_max_8_aliases_not_enforced_by_parser(self):
        # Parser doesn't enforce max; gateway prompt handles it
        items = [f"alias{i}" for i in range(10)]
        raw = json.dumps(items)
        result = AliasExtractor._parse_aliases(raw)
        assert len(result) == 10


class TestAliasExtractorProviderModel:
    @pytest.mark.asyncio
    async def test_custom_provider_model_forwarded(self):
        gw = _make_gateway("[]")
        ext = AliasExtractor(gw, provider="anthropic", model="claude-haiku-4-5")
        doc = _make_doc("content")
        await ext.extract(doc)
        call_kwargs = gw.complete.call_args[1]
        assert call_kwargs.get("provider") == "anthropic"
        assert call_kwargs.get("model") == "claude-haiku-4-5"
