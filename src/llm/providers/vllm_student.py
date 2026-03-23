"""src/llm/providers/vllm_student.py — vLLM Student Model プロバイダ

Student Model (Qwen2.5-7B + TinyLoRA) を vLLM 経由で呼び出す。
OpenAI 互換 API を使用するため openai ライブラリを流用する。

環境変数 VLLM_BASE_URL が設定されている場合に有効化される。
"""

from __future__ import annotations

import logging
import os

from src.common.config import Settings
from src.llm.gateway import BaseLLMProvider, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)


class VLLMStudentProvider(BaseLLMProvider):
    """vLLM で動く Student Model プロバイダ（OpenAI 互換 API）。"""

    @property
    def name(self) -> str:
        return "vllm_student"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._vllm_base_url = os.environ.get("VLLM_BASE_URL", "")

    def is_available(self) -> bool:
        """VLLM_BASE_URL 環境変数が設定されている場合のみ有効。"""
        return bool(self._vllm_base_url)

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        timeout: float | None = None,
    ) -> LLMResponse:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        effective_timeout = timeout if timeout is not None else 120.0
        client = AsyncOpenAI(
            api_key="not-needed",  # vLLM は API キー不要
            base_url=self._vllm_base_url,
            timeout=effective_timeout,
        )
        model_name = model or self._settings.training.student_model.name

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
