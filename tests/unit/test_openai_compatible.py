"""tests/unit/test_openai_compatible.py — OpenAICompatibleProvider 単体テスト"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.gateway import LLMMessage
from src.llm.providers.openai_compatible import OpenAICompatibleProvider


class TestOpenAICompatibleProviderInit:
    def test_name_property(self):
        p = OpenAICompatibleProvider(
            name="LMSTUDIO",
            base_url="http://localhost:1234/v1",
            default_model="llama3",
        )
        assert p.name == "LMSTUDIO"

    def test_is_available_with_base_url(self):
        p = OpenAICompatibleProvider(
            name="TEST",
            base_url="http://localhost:1234/v1",
            default_model="model",
        )
        assert p.is_available() is True

    def test_is_available_with_empty_base_url(self):
        p = OpenAICompatibleProvider(
            name="TEST",
            base_url="",
            default_model="model",
        )
        assert p.is_available() is False

    def test_api_key_direct(self):
        p = OpenAICompatibleProvider(
            name="TEST",
            base_url="http://localhost/v1",
            default_model="model",
            api_key="my-secret-key",
        )
        assert p._api_key == "my-secret-key"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("MY_API_KEY", "env-secret")
        p = OpenAICompatibleProvider(
            name="TEST",
            base_url="http://localhost/v1",
            default_model="model",
            api_key_env="MY_API_KEY",
        )
        assert p._api_key == "env-secret"

    def test_api_key_default_when_no_key(self):
        p = OpenAICompatibleProvider(
            name="TEST",
            base_url="http://localhost/v1",
            default_model="model",
        )
        assert p._api_key == "not-set"

    def test_api_key_direct_takes_priority_over_env(self, monkeypatch):
        monkeypatch.setenv("MY_API_KEY", "env-value")
        p = OpenAICompatibleProvider(
            name="TEST",
            base_url="http://localhost/v1",
            default_model="model",
            api_key="direct-value",
            api_key_env="MY_API_KEY",
        )
        assert p._api_key == "direct-value"

    def test_trailing_slash_stripped_from_base_url(self):
        p = OpenAICompatibleProvider(
            name="TEST",
            base_url="http://localhost:1234/v1/",
            default_model="model",
        )
        assert not p._base_url.endswith("/")

    def test_env_key_missing_uses_not_set(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_KEY", raising=False)
        p = OpenAICompatibleProvider(
            name="TEST",
            base_url="http://localhost/v1",
            default_model="model",
            api_key_env="NONEXISTENT_KEY",
        )
        assert p._api_key == "not-set"


class TestOpenAICompatibleProviderComplete:
    @pytest.mark.asyncio
    async def test_complete_returns_llm_response(self):
        # Mock openai AsyncOpenAI client
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20

        mock_choice = MagicMock()
        mock_choice.message.content = "Hello from mock!"

        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage = mock_usage

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            p = OpenAICompatibleProvider(
                name="MOCK",
                base_url="http://localhost:1234/v1",
                default_model="llama3",
            )
            messages = [LLMMessage(role="user", content="Hi")]
            resp = await p.complete(messages)

        assert resp.content == "Hello from mock!"
        assert resp.provider == "MOCK"
        assert resp.model == "llama3"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 20

    @pytest.mark.asyncio
    async def test_complete_uses_custom_model(self):
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 5
        mock_choice = MagicMock()
        mock_choice.message.content = "response"
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage = mock_usage

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            p = OpenAICompatibleProvider(
                name="MOCK",
                base_url="http://localhost/v1",
                default_model="default-model",
            )
            messages = [LLMMessage(role="user", content="test")]
            resp = await p.complete(messages, model="custom-model")

        assert resp.model == "custom-model"
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "custom-model"

    @pytest.mark.asyncio
    async def test_complete_raises_if_openai_not_installed(self):
        p = OpenAICompatibleProvider(
            name="MOCK",
            base_url="http://localhost/v1",
            default_model="model",
        )
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises((RuntimeError, ImportError)):
                await p.complete([LLMMessage(role="user", content="hi")])

    @pytest.mark.asyncio
    async def test_complete_handles_none_usage(self):
        mock_choice = MagicMock()
        mock_choice.message.content = "result"
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage = None

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            p = OpenAICompatibleProvider(
                name="MOCK",
                base_url="http://localhost/v1",
                default_model="model",
            )
            resp = await p.complete([LLMMessage(role="user", content="hi")])

        assert resp.input_tokens == 0
        assert resp.output_tokens == 0

    @pytest.mark.asyncio
    async def test_complete_passes_max_tokens_and_temperature(self):
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 1
        mock_usage.completion_tokens = 1
        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage = mock_usage

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            p = OpenAICompatibleProvider(
                name="MOCK",
                base_url="http://localhost/v1",
                default_model="model",
            )
            await p.complete(
                [LLMMessage(role="user", content="hi")],
                max_tokens=512,
                temperature=0.1,
            )

        kwargs = mock_client.chat.completions.create.call_args[1]
        assert kwargs["max_tokens"] == 512
        assert kwargs["temperature"] == 0.1
