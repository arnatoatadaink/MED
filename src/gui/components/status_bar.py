"""src/gui/components/status_bar.py — システム状態表示コンポーネント。

APIオーケストレーターおよびDockerの接続状態をリアルタイムで表示する。
"""

from __future__ import annotations

import httpx

_ORCHESTRATOR_URL = "http://localhost:8000"


def get_status_markdown() -> str:
    """API・Dockerの接続状態をMarkdown文字列で返す。"""
    # FastAPI
    try:
        r = httpx.get(f"{_ORCHESTRATOR_URL}/health", timeout=1.0)
        api_ok = r.status_code == 200
        api_info = r.json() if api_ok else {}
    except Exception:
        api_ok = False
        api_info = {}

    api_icon = "🟢" if api_ok else "🔴"
    api_label = "オーケストレーター: **オンライン**" if api_ok else "オーケストレーター: **オフライン**"

    # Docker (簡易チェック — dockerデーモンのsocket存在確認)
    import os
    docker_ok = os.path.exists("/var/run/docker.sock")
    docker_icon = "🟢" if docker_ok else "🟡"
    docker_label = "Docker: **利用可能**" if docker_ok else "Docker: **未確認**"

    version = api_info.get("version", "—")
    extra = f" v{version}" if version != "—" else ""

    return (
        f"{api_icon} {api_label}{extra} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"{docker_icon} {docker_label}"
    )
