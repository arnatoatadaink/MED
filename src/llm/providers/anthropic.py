"""src/llm/providers/anthropic.py — Anthropic (Claude) プロバイダ"""

from __future__ import annotations

import logging

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
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        enable_thinking: bool = False,
        thinking_budget_tokens: int = 8000,
    ) -> LLMResponse:
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

        client = anthropic.AsyncAnthropic(api_key=self._settings.anthropic_api_key)
        model_name = model or self._settings.llm.anthropic.default_model

        # system プロンプトを分離
        system_content: str | None = None
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

        if enable_thinking:
            # Extended Thinking: temperature は 1.0 固定（API要件）
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget_tokens,
            }
            # Extended Thinking 有効時は temperature を設定しない（API が 1.0 を強制）
            kwargs.pop("temperature", None)
        else:
            if temperature != 1.0:
                kwargs["temperature"] = temperature

        if system_content:
            kwargs["system"] = system_content

        raw = await client.messages.create(**kwargs)

        # レスポンスパース: thinking ブロックと text ブロックを分離
        thinking_text = None
        thinking_tokens = 0
        content_text = ""
        for block in raw.content:
            if block.type == "thinking":
                thinking_text = block.thinking
                # SDK バージョンによって thinking_tokens の取得方法が異なる
                thinking_tokens = getattr(block, "thinking_tokens", 0)
            elif block.type == "text":
                content_text = block.text

        # thinking ブロックがなかった場合（通常応答）
        if not content_text and raw.content:
            content_text = raw.content[0].text if hasattr(raw.content[0], "text") else ""

        return LLMResponse(
            content=content_text,
            provider=self.name,
            model=model_name,
            input_tokens=raw.usage.input_tokens,
            output_tokens=raw.usage.output_tokens,
            raw=raw,
            thinking_text=thinking_text,
            thinking_tokens=thinking_tokens,
        )
