"""src/llm/providers/openai.py — OpenAI (GPT) プロバイダ"""

from __future__ import annotations

import logging

from src.common.config import Settings
from src.llm.gateway import BaseLLMProvider, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT API プロバイダ。"""

    @property
    def name(self) -> str:
        return "openai"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def is_available(self) -> bool:
        return bool(self._settings.openai_api_key)

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: float | None = None,
    ) -> LLMResponse:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        client = AsyncOpenAI(api_key=self._settings.openai_api_key)
        model_name = model or self._settings.llm.openai.default_model

        oai_messages = [{"role": m.role, "content": m.content} for m in messages]

        raw = await client.chat.completions.create(
            model=model_name,
            messages=oai_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        content = raw.choices[0].message.content or ""
        usage = raw.usage
        return LLMResponse(
            content=content,
            provider=self.name,
            model=model_name,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            raw=raw,
        )
