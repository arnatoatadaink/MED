"""src/llm/providers/openai_compatible.py — OpenAI互換エンドポイント汎用プロバイダー

LM Studio / vLLM / Together.ai / Azure OpenAI 等、
`/v1/chat/completions` を提供する任意のサーバーに接続する。

ベースURLの形式:
    http://localhost:1234/v1   ← LM Studio デフォルト
    http://localhost:8001/v1   ← vLLM
    https://api.together.xyz/v1
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from src.llm.gateway import BaseLLMProvider, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(BaseLLMProvider):
    """OpenAI互換 `/v1/chat/completions` エンドポイント汎用プロバイダー。

    APIキーが不要なローカルサーバー（LM Studio 等）は api_key_env を省略でき、
    その場合 `api_key="not-set"` を使用する（openai SDK のローカル接続に必要なダミー値）。

    Args:
        name: プロバイダー識別名 (例: "LMSTUDIO")
        base_url: エンドポイントの /v1 まで (例: "http://localhost:1234/v1")
        default_model: モデル名省略時のデフォルト
        api_key: 直接渡すAPIキー (api_key_env より優先)
        api_key_env: APIキーを取得する環境変数名
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        default_model: str,
        api_key: str = "",
        api_key_env: str = "",
        timeout: float = 600.0,
        default_max_tokens: int | None = None,
        default_temperature: float | None = None,
        extra_params: dict | None = None,
        requests_per_minute: int | None = None,
    ) -> None:
        self._name = name
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._timeout = timeout
        self._default_max_tokens = default_max_tokens
        self._default_temperature = default_temperature
        self._extra_params = extra_params or {}
        # APIキー解決: 直接値 → 環境変数 → ダミー値 (ローカル用)
        if api_key:
            self._api_key = api_key
        elif api_key_env:
            self._api_key = os.environ.get(api_key_env, "not-set")
        else:
            self._api_key = "not-set"
        # レート制限 (requests_per_minute が設定されている場合)
        if requests_per_minute and requests_per_minute > 0:
            self._rate_interval = 60.0 / requests_per_minute  # 秒/リクエスト
        else:
            self._rate_interval = 0.0
        self._rate_lock = asyncio.Lock()
        self._last_request_time: float = 0.0

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        """ベースURLが設定されていれば利用可能とみなす。

        ローカルサーバーはAPIキー不要なため、URLの有無のみでチェックする。
        """
        return bool(self._base_url)

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        enable_thinking: bool = False,
        thinking_budget_tokens: int = 8000,
        timeout: float | None = None,
    ) -> LLMResponse:
        # レート制限: requests_per_minute が設定されている場合、間隔を守る
        if self._rate_interval > 0:
            async with self._rate_lock:
                now = time.monotonic()
                wait = self._rate_interval - (now - self._last_request_time)
                if wait > 0:
                    logger.debug("Rate limit wait: %.2fs for provider %s", wait, self._name)
                    await asyncio.sleep(wait)
                self._last_request_time = time.monotonic()

        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        # プロバイダーデフォルト値を適用（yaml 設定 > 呼び出し側デフォルト）
        effective_timeout = timeout if timeout is not None else self._timeout
        effective_max_tokens = self._default_max_tokens if self._default_max_tokens is not None else max_tokens
        effective_temperature = self._default_temperature if self._default_temperature is not None else temperature

        client = AsyncOpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=effective_timeout,
        )
        model_name = model or self._default_model

        oai_messages = [{"role": m.role, "content": m.content} for m in messages]

        # extra_params から追加パラメータを適用（extra_body 等）
        create_kwargs: dict = {
            "model": model_name,
            "messages": oai_messages,
            "max_tokens": effective_max_tokens,
            "temperature": effective_temperature,
        }
        if self._extra_params:
            create_kwargs["extra_body"] = self._extra_params

        raw = await client.chat.completions.create(**create_kwargs)

        choice = raw.choices[0]
        content = choice.message.content or ""
        usage = raw.usage

        # Qwen3.5 等の思考モデル対応: reasoning_content を thinking_text に保存
        # content が空で reasoning_content がある場合は思考途中で max_tokens に達した可能性
        thinking_text: str | None = None
        thinking_tokens = 0
        reasoning = getattr(choice.message, "reasoning_content", None)
        if reasoning:
            thinking_text = reasoning
            # 思考トークン数は概算（usage に内訳がない場合）
            if usage and not content:
                thinking_tokens = usage.completion_tokens or 0
                logger.warning(
                    "Provider %s: content is empty but reasoning_content has %d chars. "
                    "The model may need more max_tokens (current=%d) to produce output "
                    "after thinking.",
                    self._name, len(reasoning), effective_max_tokens,
                )

        return LLMResponse(
            content=content,
            provider=self._name,
            model=model_name,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            thinking_text=thinking_text,
            thinking_tokens=thinking_tokens,
            raw=raw,
        )
