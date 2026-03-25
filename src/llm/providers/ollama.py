"""src/llm/providers/ollama.py — Ollama (ローカル LLM) プロバイダ"""

from __future__ import annotations

import logging

from src.common.config import Settings
from src.llm.gateway import BaseLLMProvider, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Ollama ローカル LLM プロバイダ（HTTP REST API 経由）。"""

    @property
    def name(self) -> str:
        return "ollama"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def is_available(self) -> bool:
        """Ollama は API キー不要。常に利用可能（接続エラーは complete 時に発生）。"""
        return bool(self._settings.llm.ollama.base_url)

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
            import httpx
        except ImportError:
            raise RuntimeError("httpx package not installed. Run: pip install httpx")

        model_name = model or self._settings.llm.ollama.default_model
        base_url = self._settings.llm.ollama.base_url.rstrip("/")

        # Ollama の /api/chat エンドポイントを使用
        payload = {
            "model": model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        content = data.get("message", {}).get("content", "")
        prompt_eval_count = data.get("prompt_eval_count", 0)
        eval_count = data.get("eval_count", 0)

        return LLMResponse(
            content=content,
            provider=self.name,
            model=model_name,
            input_tokens=prompt_eval_count,
            output_tokens=eval_count,
            raw=data,
        )
