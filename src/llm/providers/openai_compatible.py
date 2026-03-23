"""src/llm/providers/openai_compatible.py — OpenAI互換エンドポイント汎用プロバイダー

LM Studio / vLLM / Together.ai / Azure OpenAI 等、
`/v1/chat/completions` を提供する任意のサーバーに接続する。

ベースURLの形式:
    http://localhost:1234/v1   ← LM Studio デフォルト
    http://localhost:8001/v1   ← vLLM
    https://api.together.xyz/v1
"""

from __future__ import annotations

import logging
import os

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
    ) -> None:
        self._name = name
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._timeout = timeout
        # APIキー解決: 直接値 → 環境変数 → ダミー値 (ローカル用)
        if api_key:
            self._api_key = api_key
        elif api_key_env:
            self._api_key = os.environ.get(api_key_env, "not-set")
        else:
            self._api_key = "not-set"

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
        max_tokens: int = 2048,
        temperature: float = 0.7,
        enable_thinking: bool = False,
        thinking_budget_tokens: int = 8000,
        timeout: float | None = None,
    ) -> LLMResponse:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        # リクエスト単位の timeout が指定されていればそちらを優先、なければコンストラクタ値
        effective_timeout = timeout if timeout is not None else self._timeout
        client = AsyncOpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=effective_timeout,
        )
        model_name = model or self._default_model

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
            provider=self._name,
            model=model_name,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            raw=raw,
        )
