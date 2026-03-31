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
from pathlib import Path
from typing import Any

import yaml

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
    # Extended Thinking / CoT で得た思考テキスト
    thinking_text: str | None = None
    thinking_tokens: int = 0


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
        max_tokens: int = 4096,
        temperature: float = 0.7,
        enable_thinking: bool = False,
        thinking_budget_tokens: int = 8000,
        timeout: float | None = None,
    ) -> LLMResponse:
        """メッセージリストを受け取り、補完レスポンスを返す。

        Args:
            enable_thinking: Extended Thinking を有効化する（Anthropic のみ）。
            thinking_budget_tokens: thinking ブロックの最大トークン数。
            timeout: リクエストタイムアウト秒数（None = プロバイダデフォルト）。
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """プロバイダが利用可能か確認する（APIキー存在チェック等）。"""
        ...


# ============================================================================
# ゲートウェイ
# ============================================================================

# 組み込みプロバイダーのデフォルトフォールバック順序
_BUILTIN_PROVIDER_ORDER = ["anthropic", "openai", "ollama"]
# llm_config.yaml から primary_provider / カスタムプロバイダーを読むパス
_CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"
_LLM_CONFIG_PATH = _CONFIGS_DIR / "llm_config.yaml"
# ローカル専用設定（git 管理外）— ローカルバックエンドのプロバイダーを格納する
_LLM_CONFIG_LOCAL_PATH = _CONFIGS_DIR / "llm_config.local.yaml"
_KNOWN_PROVIDERS = {"anthropic", "openai", "ollama", "vllm", "azure_openai", "together"}


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
        # llm_config.yaml に保存されたカスタムプロバイダーを追加ロード
        self._load_custom_providers()

    def _load_custom_providers(self) -> None:
        """llm_config.yaml および llm_config.local.yaml からカスタムプロバイダーを登録する。

        組み込み名 (anthropic, openai, ollama, vllm, azure_openai, together) 以外の
        エントリーを OpenAICompatibleProvider として登録する。
        type が "openai_compatible" または未指定のものが対象。

        llm_config.local.yaml は git 管理外のローカル専用設定ファイル。
        ローカルバックエンド（LM Studio / vLLM 等）のエンドポイントはこちらに保存される。
        """
        config_files = [_LLM_CONFIG_PATH, _LLM_CONFIG_LOCAL_PATH]
        for config_path in config_files:
            if not config_path.exists():
                continue
            try:
                with open(config_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                for name, conf in cfg.get("providers", {}).items():
                    if name in _KNOWN_PROVIDERS:
                        continue
                    if not isinstance(conf, dict):
                        continue
                    ptype = conf.get("type", "openai_compatible")
                    if ptype not in ("openai_compatible", "other"):
                        logger.debug("Skipping custom provider %s with unsupported type %s", name, ptype)
                        continue
                    base_url = conf.get("base_url", "")
                    if not base_url:
                        logger.warning("Custom provider %s has no base_url; skipping", name)
                        continue
                    from src.llm.providers.openai_compatible import OpenAICompatibleProvider
                    api_key_env = conf.get("api_key_env", "")
                    # プロバイダー固有パラメータ（yaml で任意指定）
                    default_max_tokens = conf.get("max_tokens")
                    if default_max_tokens is not None:
                        default_max_tokens = int(default_max_tokens)
                    default_temperature = conf.get("temperature")
                    if default_temperature is not None:
                        default_temperature = float(default_temperature)
                    extra_params = conf.get("extra_params") or {}
                    requests_per_minute = conf.get("requests_per_minute")
                    if requests_per_minute is not None:
                        requests_per_minute = int(requests_per_minute)
                    provider = OpenAICompatibleProvider(
                        name=name,
                        base_url=base_url,
                        default_model=conf.get("default_model", ""),
                        api_key_env=api_key_env,
                        timeout=float(conf.get("timeout", 600)),
                        default_max_tokens=default_max_tokens,
                        default_temperature=default_temperature,
                        extra_params=extra_params,
                        requests_per_minute=requests_per_minute,
                    )
                    self._providers[name] = provider
                    logger.info(
                        "Loaded custom provider: %s (base_url=%s model=%s local=%s)",
                        name, base_url, conf.get("default_model", ""),
                        config_path == _LLM_CONFIG_LOCAL_PATH,
                    )
            except Exception:
                logger.exception("Failed to load custom providers from %s", config_path)

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
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: float | None = None,
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
            timeout=timeout,
        )

    async def complete_messages(
        self,
        messages: list[LLMMessage],
        *,
        provider: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: float | None = None,
    ) -> LLMResponse:
        """メッセージリストを送り、補完を返す。

        指定プロバイダが使用できない場合はフォールバック順を試みる。
        """
        providers_to_try = self._build_provider_order(provider)

        last_exc: Exception | None = None
        skip_reasons: list[str] = []
        for prov_name in providers_to_try:
            prov = self._providers.get(prov_name)
            if prov is None:
                reason = f"{prov_name}: not registered"
                logger.warning("Provider not registered: %s", prov_name)
                skip_reasons.append(reason)
                continue
            if not prov.is_available():
                reason = f"{prov_name}: not available"
                logger.debug("Provider %s not available; skipping", prov_name)
                skip_reasons.append(reason)
                continue

            try:
                start = time.monotonic()
                response = await prov.complete(
                    messages,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
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
                logger.warning("Provider %s failed: %s", prov_name, exc, exc_info=True)
                last_exc = exc
                skip_reasons.append(f"{prov_name}: {type(exc).__name__}: {exc}")
                continue

        raise RuntimeError(
            f"All LLM providers failed. Tried: {providers_to_try}. "
            f"Details: {skip_reasons}. Last error: {last_exc}"
        )

    def _build_provider_order(self, requested: str | None) -> list[str]:
        """試みるプロバイダ名のリストを返す。

        - 明示指定あり: そのプロバイダのみ（フォールバックなし）
        - auto (None): llm_config.yaml の primary_provider を先頭に、
          その後 _BUILTIN_PROVIDER_ORDER でフォールバック
        """
        if requested is not None:
            return [requested]

        # llm_config.yaml から primary_provider を読み込む
        primary: str | None = None
        try:
            if _LLM_CONFIG_PATH.exists():
                with open(_LLM_CONFIG_PATH, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                primary = cfg.get("primary_provider")
        except Exception:
            pass

        if primary and primary not in _BUILTIN_PROVIDER_ORDER:
            # カスタムプロバイダーが primary の場合も組み込みへフォールバック
            return [primary] + list(_BUILTIN_PROVIDER_ORDER)
        if primary:
            # 組み込みプロバイダーが primary の場合は先頭に置いてフォールバック
            order = [primary] + [p for p in _BUILTIN_PROVIDER_ORDER if p != primary]
            return order
        return list(_BUILTIN_PROVIDER_ORDER)

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
