"""src/gui/utils.py — GUI共通ユーティリティ。

複数タブで共有するヘルパー関数と定数を提供する。
"""

from __future__ import annotations

import httpx

# オーケストレーター接続先 (configs/default.yaml の gui.orchestrator_url と合わせる)
ORCHESTRATOR_URL = "http://localhost:8000"


def is_api_alive() -> bool:
    """FastAPI オーケストレーターが起動中かどうかを返す。"""
    try:
        r = httpx.get(f"{ORCHESTRATOR_URL}/health", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False
