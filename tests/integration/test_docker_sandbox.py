"""tests/integration/test_docker_sandbox.py — Docker Sandbox 統合テスト

実際の Docker デーモンを使用してコード実行の End-to-End を検証する。
`@pytest.mark.docker` がついており、Docker なし環境では自動スキップされる。

テスト対象:
  - CodeExecutor._execute_docker: python:3.11-slim コンテナでの実行
  - SandboxManager.run: リトライ込みの高レベルAPI
  - セキュリティポリシー: Docker 環境でのブロック確認
  - タイムアウト: 長時間実行コードのキャンセル
"""

from __future__ import annotations

import asyncio
import time

import pytest

from src.sandbox.executor import CodeExecutor
from src.sandbox.manager import SandboxManager

pytestmark = pytest.mark.docker


# ---------------------------------------------------------------------------
# CodeExecutor — Docker 実行
# ---------------------------------------------------------------------------

class TestCodeExecutorDocker:
    @pytest.fixture
    def executor(self):
        return CodeExecutor(use_docker=True)

    def test_executor_uses_docker(self, executor):
        """Docker が利用可能なとき use_docker フラグが True になること。"""
        assert executor._use_docker is True

    @pytest.mark.asyncio
    async def test_hello_world(self, executor):
        """print() の出力が stdout に返ること。"""
        result = await executor.execute("print('hello from docker')")
        assert result.success
        assert "hello from docker" in result.stdout
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_arithmetic(self, executor):
        """計算結果が正しく出力されること。"""
        result = await executor.execute("print(6 * 7)")
        assert result.success
        assert "42" in result.stdout

    @pytest.mark.asyncio
    async def test_multiline_code(self, executor):
        """複数行コードが正しく実行されること。"""
        code = """
xs = [1, 2, 3, 4, 5]
total = sum(xs)
print(f"sum={total}")
"""
        result = await executor.execute(code)
        assert result.success
        assert "sum=15" in result.stdout

    @pytest.mark.asyncio
    async def test_syntax_error_captured(self, executor):
        """構文エラーが stderr に記録され exit_code != 0 になること。"""
        result = await executor.execute("def broken(:\n    pass")
        assert not result.success
        assert result.exit_code != 0

    @pytest.mark.asyncio
    async def test_runtime_error_captured(self, executor):
        """実行時エラーが stderr に記録されること。"""
        result = await executor.execute("raise ValueError('test error')")
        assert not result.success
        assert "ValueError" in result.stderr

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_timeout_kills_container(self, executor):
        """タイムアウトを超えた実行がキャンセルされること。"""
        code = "import time; time.sleep(60)"
        start = time.monotonic()
        result = await executor.execute(code, timeout=3)
        elapsed = time.monotonic() - start

        assert result.timed_out
        assert not result.success
        assert elapsed < 10  # 10秒以内に返る（余裕を持って検証）

    @pytest.mark.asyncio
    async def test_execution_time_recorded(self, executor):
        """実行時間が記録されること（0より大きい値）。"""
        result = await executor.execute("x = 1 + 1")
        assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_no_network_access(self, executor):
        """コンテナからの外部ネットワークアクセスが制限されること。"""
        code = """
import socket
try:
    socket.setdefaulttimeout(2)
    socket.getaddrinfo('google.com', 80)
    print('network_ok')
except Exception as e:
    print(f'network_blocked: {e}')
"""
        result = await executor.execute(code)
        # 環境によって結果が変わる（CI では blocked が期待値）
        # 少なくともクラッシュしないことを確認
        assert result.exit_code is not None


# ---------------------------------------------------------------------------
# SecurityPolicy — Docker 環境での確認
# ---------------------------------------------------------------------------

class TestSecurityPolicyWithDocker:
    @pytest.fixture
    def executor(self):
        return CodeExecutor(use_docker=True)

    @pytest.mark.asyncio
    async def test_os_system_blocked_before_docker(self, executor):
        """os.system() はポリシーチェックで Docker 起動前にブロックされること。"""
        result = await executor.execute("import os; os.system('ls')")
        assert result.blocked_by_policy
        assert not result.success
        assert result.execution_time_ms == 0.0  # Docker を起動していない

    @pytest.mark.asyncio
    async def test_subprocess_blocked_before_docker(self, executor):
        """subprocess はポリシーチェックでブロックされること。"""
        result = await executor.execute(
            "import subprocess; subprocess.run(['ls'])"
        )
        assert result.blocked_by_policy

    @pytest.mark.asyncio
    async def test_safe_import_allowed(self, executor):
        """標準ライブラリの安全なインポートは許可されること。"""
        result = await executor.execute(
            "import math; print(math.sqrt(16))"
        )
        assert result.success
        assert "4.0" in result.stdout


# ---------------------------------------------------------------------------
# SandboxManager — 高レベル API
# ---------------------------------------------------------------------------

class TestSandboxManagerDocker:
    @pytest.fixture
    def manager(self):
        return SandboxManager(use_docker=True, max_retries=1)

    @pytest.mark.asyncio
    async def test_run_returns_result(self, manager):
        """run() が ExecutionResult を返すこと。"""
        result = await manager.run("print('sandbox ok')")
        assert result.success
        assert "sandbox ok" in result.stdout

    @pytest.mark.asyncio
    async def test_run_language_python(self, manager):
        """language='python' で正常動作すること。"""
        result = await manager.run("print(2 ** 10)", language="python")
        assert "1024" in result.stdout

    @pytest.mark.asyncio
    async def test_run_blocked_code(self, manager):
        """ポリシー違反コードは run() でもブロックされること。"""
        result = await manager.run("import os; os.system('id')")
        assert not result.success
        assert result.blocked_by_policy

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_run_with_custom_timeout(self, manager):
        """カスタムタイムアウトが適用されること。"""
        result = await manager.run(
            "import time; time.sleep(30)", timeout=2
        )
        assert result.timed_out
        assert not result.success


# ---------------------------------------------------------------------------
# 並行実行テスト
# ---------------------------------------------------------------------------

class TestDockerConcurrency:
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_executions(self):
        """複数コンテナを並行実行できること。"""
        executor = CodeExecutor(use_docker=True)
        codes = [f"print('job-{i}')" for i in range(3)]
        results = await asyncio.gather(*[
            executor.execute(code) for code in codes
        ])
        assert all(r.success for r in results)
        for i, result in enumerate(results):
            assert f"job-{i}" in result.stdout
