"""src/llm/gateway.py — LLM プロバイダ抽象化ゲートウェイ

複数の LLM プロバイダ（Claude/GPT/Ollama/vLLM）を統一インターフェースで利用する。
プロバイダの選択・フォールバック・使用量トラッキングを担当する。

使い方:
    from src.llm.gateway import LLMGateway

    gateway = LLMGateway()
    response = await gateway.complete("Python のリストをソートするには？")
    print(response.content)

    # 特定プロバイダを指定
    response = await gateway.complete("...", provider="anthropic")
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.common.config import Settings, get_settings

logger = logging.getLogger(__name__)


# ============================================================================
# データクラス
# ============================================================================


@dataclass
class LLMMessage:
    """LLM への入力メッセージ。"""

    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class LLMResponse:
    """LLM からのレスポンス。"""

    content: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    raw: Any | None = None  # プロバイダ固有の生レスポンス


# ============================================================================
# 抽象基底クラス
# ============================================================================


class BaseLLMProvider(ABC):
    """LLM プロバイダの抽象基底クラス。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """プロバイダ名。"""
        ...

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """メッセージリストを受け取り、補完レスポンスを返す。"""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """プロバイダが利用可能か確認する（APIキー存在チェック等）。"""
        ...


# ============================================================================
# ゲートウェイ
# ============================================================================

# デフォルトのフォールバック順序
_DEFAULT_PROVIDER_ORDER = ["anthropic", "openai", "ollama"]


class LLMGateway:
    """複数 LLM プロバイダを管理するゲートウェイ。

    優先順位: 指定プロバイダ → デフォルト順（anthropic → openai → ollama）

    Args:
        settings: Settings オブジェクト。省略時は get_settings() を使用。
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._providers: dict[str, BaseLLMProvider] = {}
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0
        self._register_providers()

    def _register_providers(self) -> None:
        """利用可能なプロバイダを登録する。"""
        from src.llm.providers.anthropic import AnthropicProvider
        from src.llm.providers.ollama import OllamaProvider
        from src.llm.providers.openai import OpenAIProvider
        from src.llm.providers.vllm_student import VLLMStudentProvider

        candidates = [
            AnthropicProvider(self._settings),
            OpenAIProvider(self._settings),
            OllamaProvider(self._settings),
            VLLMStudentProvider(self._settings),
        ]
        for provider in candidates:
            self._providers[provider.name] = provider
            logger.debug(
                "Registered provider: %s (available=%s)", provider.name, provider.is_available()
            )

    def register(self, provider: BaseLLMProvider) -> None:
        """カスタムプロバイダを登録する。"""
        self._providers[provider.name] = provider
        logger.info("Registered custom provider: %s", provider.name)

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """テキストプロンプトを送り、補完を返す。

        Args:
            prompt: ユーザーのプロンプト文字列。
            system: システムプロンプト（省略可）。
            provider: 使用プロバイダ名。省略時はデフォルト / フォールバック順。
            model: モデル名。省略時はプロバイダのデフォルト。
            max_tokens: 最大出力トークン数。
            temperature: 温度パラメータ。

        Returns:
            LLMResponse オブジェクト。
        """
        messages: list[LLMMessage] = []
        if system:
            messages.append(LLMMessage(role="system", content=system))
        messages.append(LLMMessage(role="user", content=prompt))
        return await self.complete_messages(
            messages,
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def complete_messages(
        self,
        messages: list[LLMMessage],
        *,
        provider: str | None = None,
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """メッセージリストを送り、補完を返す。

        指定プロバイダが使用できない場合はフォールバック順を試みる。
        """
        providers_to_try = self._build_provider_order(provider)

        last_exc: Exception | None = None
        for prov_name in providers_to_try:
            prov = self._providers.get(prov_name)
            if prov is None:
                logger.warning("Provider not registered: %s", prov_name)
                continue
            if not prov.is_available():
                logger.debug("Provider %s not available; skipping", prov_name)
                continue

            try:
                start = time.monotonic()
                response = await prov.complete(
                    messages,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                response.latency_ms = (time.monotonic() - start) * 1000

                self._total_input_tokens += response.input_tokens
                self._total_output_tokens += response.output_tokens
                self._call_count += 1

                logger.debug(
                    "LLM call via %s: in=%d out=%d latency=%.0fms",
                    prov_name, response.input_tokens, response.output_tokens, response.latency_ms,
                )
                return response

            except Exception as exc:
                logger.warning("Provider %s failed: %s", prov_name, exc)
                last_exc = exc
                continue

        raise RuntimeError(
            f"All LLM providers failed. Tried: {providers_to_try}. "
            f"Last error: {last_exc}"
        )

    def _build_provider_order(self, requested: str | None) -> list[str]:
        """試みるプロバイダ名のリストを返す。"""
        if requested is not None:
            return [requested]  # 指定プロバイダのみ（フォールバックなし）
        return list(_DEFAULT_PROVIDER_ORDER)

    @property
    def usage(self) -> dict:
        """累積使用量統計を返す。"""
        return {
            "call_count": self._call_count,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
        }

    def available_providers(self) -> list[str]:
        """利用可能なプロバイダ名のリストを返す。"""
        return [name for name, prov in self._providers.items() if prov.is_available()]
