"""tests/conftest.py — 共有フィクスチャ & カスタムマーカー定義

pytest マーカー:
  - pytest.mark.docker   : Docker デーモン必須テスト（CI の docker-tests ジョブで実行）
  - pytest.mark.slow     : 実行に数秒以上かかるテスト

Docker が利用できない環境では `@pytest.mark.docker` テストは自動スキップされる。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# カスタムマーカー登録
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "docker: mark test as requiring a running Docker daemon",
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (may take several seconds)",
    )


# ---------------------------------------------------------------------------
# Docker 可用性チェック（セッション共有）
# ---------------------------------------------------------------------------

def _is_docker_available() -> bool:
    """Docker デーモンに接続できるか確認する。"""
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def docker_available() -> bool:
    """Docker デーモンが利用可能かどうかを返すフィクスチャ。"""
    return _is_docker_available()


def pytest_runtest_setup(item: pytest.Item) -> None:
    """docker マーカーがついたテストを Docker なしの環境でスキップする。"""
    if item.get_closest_marker("docker"):
        if not _is_docker_available():
            pytest.skip("Docker daemon not available")


# ---------------------------------------------------------------------------
# 共通 tmp_path ラッパー（非同期テスト向け）
# ---------------------------------------------------------------------------

@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------------------
# 共通 MetadataStore フィクスチャ（integration テスト向け）
# ---------------------------------------------------------------------------

@pytest.fixture
async def fresh_store(tmp_path: Path):
    """初期化済みの MetadataStore を提供し、テスト後にクローズする。"""
    from src.memory.metadata_store import MetadataStore

    store = MetadataStore(db_path=str(tmp_path / "test.db"))
    await store.initialize()
    yield store
    await store.close()


# ---------------------------------------------------------------------------
# 共通 TeacherRegistry フィクスチャ
# ---------------------------------------------------------------------------

@pytest.fixture
async def fresh_registry(tmp_path: Path):
    """初期化済みの TeacherRegistry を提供し、テスト後にクローズする。"""
    from src.memory.teacher_registry import TeacherRegistry

    registry = TeacherRegistry(tmp_path / "registry.db")
    await registry.initialize()
    yield registry
    await registry.close()
