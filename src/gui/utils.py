"""src/gui/utils.py — GUI共通ユーティリティ。

複数タブで共有するヘルパー関数と定数を提供する。
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import httpx
import yaml

# オーケストレーター接続先 (configs/default.yaml の gui.orchestrator_url と合わせる)
ORCHESTRATOR_URL = "http://localhost:8000"

# Gradio メジャーバージョン (バージョン別 API 分岐に使用)
GRADIO_MAJOR = int(gr.__version__.split(".")[0])

_CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"
_BASE_PROVIDERS = ["auto (設定ファイル依存)", "anthropic", "openai", "ollama", "vllm"]
_KNOWN_PROVIDERS = {"anthropic", "openai", "ollama", "vllm", "azure_openai", "together"}


def is_api_alive() -> bool:
    """FastAPI オーケストレーターが起動中かどうかを返す。"""
    try:
        r = httpx.get(f"{ORCHESTRATOR_URL}/health", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False


def get_all_provider_choices() -> list[str]:
    """llm_config.yaml および llm_config.local.yaml からカスタムプロバイダーを含む全プロバイダーリストを返す。

    ページ初期ロード時およびカスタムプロバイダーの追加・削除後に呼び出すことで、
    チャットタブのドロップダウンを常に最新状態に保つ。
    llm_config.local.yaml は git 管理外のローカル専用設定ファイル。
    """
    custom: list[str] = []
    for config_file in [
        _CONFIGS_DIR / "llm_config.yaml",
        _CONFIGS_DIR / "llm_config.local.yaml",
    ]:
        try:
            if config_file.exists():
                with open(config_file, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                for k in cfg.get("providers", {}):
                    if k not in _KNOWN_PROVIDERS and k not in custom:
                        custom.append(k)
        except Exception:
            pass
    return _BASE_PROVIDERS + custom
