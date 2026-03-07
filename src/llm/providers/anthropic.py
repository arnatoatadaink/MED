"""src/llm/providers/anthropic.py — Anthropic (Claude) プロバイダ"""

from __future__ import annotations

import logging
from typing import Optional

from src.common.config import Settings
from src.llm.gateway import BaseLLMProvider, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API プロバイダ。"""

    @property
    def name(self) -> str:
        return "anthropic"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def is_available(self) -> bool:
        return bool(self._settings.anthropic_api_key)

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> LLMResponse:
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

        client = anthropic.AsyncAnthropic(api_key=self._settings.anthropic_api_key)
        model_name = model or self._settings.llm.anthropic.default_model

        # system プロンプトを分離
        system_content: Optional[str] = None
        user_messages = []
        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            else:
                user_messages.append({"role": msg.role, "content": msg.content})

        kwargs: dict = {
            "model": model_name,
            "max_tokens": max_tokens,
            "messages": user_messages,
        }
        if temperature != 1.0:
            kwargs["temperature"] = temperature
        if system_content:
            kwargs["system"] = system_content

        raw = await client.messages.create(**kwargs)

        content = raw.content[0].text if raw.content else ""
        return LLMResponse(
            content=content,
            provider=self.name,
            model=model_name,
            input_tokens=raw.usage.input_tokens,
            output_tokens=raw.usage.output_tokens,
            raw=raw,
        )
