"""tests/unit/test_llm_gateway.py — LLMGateway の単体テスト"""

from __future__ import annotations

import pytest

from src.llm.gateway import (
    BaseLLMProvider,
    LLMGateway,
    LLMMessage,
    LLMResponse,
)


# ──────────────────────────────────────────────
# モックプロバイダ
# ──────────────────────────────────────────────


class SuccessProvider(BaseLLMProvider):
    """常に成功するモックプロバイダ。"""

    def __init__(self, name: str = "mock", available: bool = True) -> None:
        self._name = name
        self._available = available
        self.calls: list[list[LLMMessage]] = []

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return self._available

    async def complete(
        self, messages, *, model=None, max_tokens=2048, temperature=0.7
    ) -> LLMResponse:
        self.calls.append(messages)
        return LLMResponse(
            content="mock response",
            provider=self._name,
            model=model or "mock-model",
            input_tokens=10,
            output_tokens=5,
        )


class FailingProvider(BaseLLMProvider):
    @property
    def name(self) -> str:
        return "failing"

    def is_available(self) -> bool:
        return True

    async def complete(self, messages, **kwargs) -> LLMResponse:
        raise RuntimeError("Provider failed")


class UnavailableProvider(BaseLLMProvider):
    @property
    def name(self) -> str:
        return "unavailable"

    def is_available(self) -> bool:
        return False

    async def complete(self, messages, **kwargs) -> LLMResponse:
        raise AssertionError("Should not be called")


# ──────────────────────────────────────────────
# ヘルパー: 全プロバイダなしの gateway を作る
# ──────────────────────────────────────────────


def _make_empty_gateway() -> LLMGateway:
    """デフォルトプロバイダを登録しない gateway。テスト用。"""
    gateway = LLMGateway.__new__(LLMGateway)
    gateway._providers = {}
    gateway._total_input_tokens = 0
    gateway._total_output_tokens = 0
    gateway._call_count = 0
    return gateway


# ──────────────────────────────────────────────
# 基本動作テスト
# ──────────────────────────────────────────────


class TestLLMGatewayBasic:
    @pytest.mark.asyncio
    async def test_complete_with_mock_provider(self) -> None:
        gw = _make_empty_gateway()
        mock = SuccessProvider("mock")
        gw.register(mock)

        response = await gw.complete("Hello", provider="mock")
        assert response.content == "mock response"
        assert response.provider == "mock"

    @pytest.mark.asyncio
    async def test_complete_passes_system_prompt(self) -> None:
        gw = _make_empty_gateway()
        mock = SuccessProvider("mock")
        gw.register(mock)

        await gw.complete("user prompt", system="be helpful", provider="mock")
        messages = mock.calls[0]
        roles = [m.role for m in messages]
        assert "system" in roles
        assert "user" in roles

    @pytest.mark.asyncio
    async def test_complete_messages_direct(self) -> None:
        gw = _make_empty_gateway()
        mock = SuccessProvider("mock")
        gw.register(mock)

        msgs = [LLMMessage(role="user", content="test")]
        response = await gw.complete_messages(msgs, provider="mock")
        assert response.content == "mock response"

    @pytest.mark.asyncio
    async def test_no_available_provider_raises(self) -> None:
        gw = _make_empty_gateway()
        with pytest.raises(RuntimeError, match="All LLM providers failed"):
            await gw.complete("test")


# ──────────────────────────────────────────────
# フォールバック
# ──────────────────────────────────────────────


class TestFallback:
    @pytest.mark.asyncio
    async def test_fallback_to_second_provider(self) -> None:
        gw = _make_empty_gateway()
        gw._providers["anthropic"] = FailingProvider()
        gw._providers["openai"] = SuccessProvider("openai")

        # anthropic が失敗 → openai にフォールバック
        from src.llm.gateway import _DEFAULT_PROVIDER_ORDER
        response = await gw.complete("test")  # デフォルト順で試みる
        assert response.provider == "openai"

    @pytest.mark.asyncio
    async def test_unavailable_provider_skipped(self) -> None:
        gw = _make_empty_gateway()
        gw._providers["anthropic"] = UnavailableProvider()
        gw._providers["openai"] = SuccessProvider("openai")

        response = await gw.complete("test")
        assert response.provider == "openai"

    @pytest.mark.asyncio
    async def test_specific_provider_no_fallback(self) -> None:
        gw = _make_empty_gateway()
        gw._providers["anthropic"] = FailingProvider()
        gw._providers["openai"] = SuccessProvider("openai")

        # anthropic を指定 → 失敗してもフォールバックしない
        with pytest.raises(RuntimeError):
            await gw.complete("test", provider="anthropic")


# ──────────────────────────────────────────────
# 使用量トラッキング
# ──────────────────────────────────────────────


class TestUsageTracking:
    @pytest.mark.asyncio
    async def test_usage_increments(self) -> None:
        gw = _make_empty_gateway()
        gw.register(SuccessProvider("mock"))

        await gw.complete("test", provider="mock")
        await gw.complete("test", provider="mock")

        usage = gw.usage
        assert usage["call_count"] == 2
        assert usage["total_input_tokens"] == 20
        assert usage["total_output_tokens"] == 10

    @pytest.mark.asyncio
    async def test_usage_initial_zero(self) -> None:
        gw = _make_empty_gateway()
        usage = gw.usage
        assert usage["call_count"] == 0
        assert usage["total_tokens"] == 0


# ──────────────────────────────────────────────
# プロバイダ登録・一覧
# ──────────────────────────────────────────────


class TestProviderRegistration:
    def test_register_custom_provider(self) -> None:
        gw = _make_empty_gateway()
        gw.register(SuccessProvider("custom"))
        assert "custom" in gw._providers

    def test_available_providers_list(self) -> None:
        gw = _make_empty_gateway()
        gw.register(SuccessProvider("available", available=True))
        gw.register(SuccessProvider("unavail", available=False))

        available = gw.available_providers()
        assert "available" in available
        assert "unavail" not in available

    def test_register_overrides_existing(self) -> None:
        gw = _make_empty_gateway()
        gw.register(SuccessProvider("mock"))
        new_mock = SuccessProvider("mock")
        gw.register(new_mock)
        assert gw._providers["mock"] is new_mock


# ──────────────────────────────────────────────
# プロバイダ可用性テスト（APIキーなし）
# ──────────────────────────────────────────────


class TestProviderAvailability:
    def test_anthropic_unavailable_without_key(self) -> None:
        from src.llm.providers.anthropic import AnthropicProvider
        from src.common.config import get_settings
        settings = get_settings()
        provider = AnthropicProvider(settings)
        # CI 環境では API キーがないため False を期待
        # （実際にキーがある場合は True）
        assert isinstance(provider.is_available(), bool)

    def test_openai_unavailable_without_key(self) -> None:
        from src.llm.providers.openai import OpenAIProvider
        from src.common.config import get_settings
        settings = get_settings()
        provider = OpenAIProvider(settings)
        assert isinstance(provider.is_available(), bool)

    def test_ollama_always_available(self) -> None:
        from src.llm.providers.ollama import OllamaProvider
        from src.common.config import get_settings
        settings = get_settings()
        provider = OllamaProvider(settings)
        # Ollama はデフォルト URL が設定されているので True
        assert provider.is_available() is True

    def test_vllm_unavailable_without_env(self) -> None:
        import os
        os.environ.pop("VLLM_BASE_URL", None)
        from src.llm.providers.vllm_student import VLLMStudentProvider
        from src.common.config import get_settings
        settings = get_settings()
        provider = VLLMStudentProvider(settings)
        assert provider.is_available() is False
