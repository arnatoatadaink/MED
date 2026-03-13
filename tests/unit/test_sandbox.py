"""tests/unit/test_sandbox.py — Sandbox モジュールの単体テスト"""

from __future__ import annotations

import pytest

from src.sandbox.executor import CodeExecutor
from src.sandbox.manager import SandboxManager
from src.sandbox.security import SecurityPolicy

# ──────────────────────────────────────────────
# SecurityPolicy
# ──────────────────────────────────────────────


class TestSecurityPolicy:
    def test_safe_code_passes(self) -> None:
        policy = SecurityPolicy()
        code = "x = 1 + 2\nprint(x)"
        is_safe, violations = policy.check_code(code)
        assert is_safe
        assert violations == []

    def test_os_system_blocked(self) -> None:
        policy = SecurityPolicy()
        code = "import os\nos.system('ls')"
        is_safe, violations = policy.check_code(code)
        assert not is_safe
        assert len(violations) > 0

    def test_subprocess_blocked(self) -> None:
        policy = SecurityPolicy()
        code = "import subprocess\nsubprocess.run(['ls'])"
        is_safe, violations = policy.check_code(code)
        assert not is_safe

    def test_eval_blocked(self) -> None:
        policy = SecurityPolicy()
        code = "eval('1 + 2')"
        is_safe, violations = policy.check_code(code)
        assert not is_safe

    def test_exec_blocked(self) -> None:
        policy = SecurityPolicy()
        code = "exec('x = 1')"
        is_safe, violations = policy.check_code(code)
        assert not is_safe

    def test_custom_policy_no_blocked_patterns(self) -> None:
        policy = SecurityPolicy(blocked_patterns=[])
        code = "os.system('rm -rf /')"
        is_safe, violations = policy.check_code(code)
        assert is_safe  # パターンなし → 通過

    def test_violations_list_non_empty(self) -> None:
        policy = SecurityPolicy()
        _, violations = policy.check_code("eval('x')")
        assert len(violations) > 0
        assert all(isinstance(v, str) for v in violations)


# ──────────────────────────────────────────────
# CodeExecutor (subprocess モード)
# ──────────────────────────────────────────────


class TestCodeExecutor:
    def _make_executor(self) -> CodeExecutor:
        return CodeExecutor(use_docker=False)  # テストは subprocess モード

    @pytest.mark.asyncio
    async def test_execute_hello_world(self) -> None:
        executor = self._make_executor()
        result = await executor.execute("print('hello world')")
        assert result.success
        assert "hello world" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_math(self) -> None:
        executor = self._make_executor()
        result = await executor.execute("print(2 + 2)")
        assert result.success
        assert "4" in result.stdout

    @pytest.mark.asyncio
    async def test_syntax_error_captured(self) -> None:
        executor = self._make_executor()
        result = await executor.execute("def broken(:")
        assert result.exit_code != 0
        assert not result.success

    @pytest.mark.asyncio
    async def test_runtime_error_captured(self) -> None:
        executor = self._make_executor()
        result = await executor.execute("raise ValueError('test error')")
        assert result.exit_code != 0
        assert "ValueError" in result.stderr

    @pytest.mark.asyncio
    async def test_timeout(self) -> None:
        executor = self._make_executor()
        result = await executor.execute("import time; time.sleep(100)", timeout=1)
        assert result.timed_out
        assert not result.success

    @pytest.mark.asyncio
    async def test_blocked_by_policy(self) -> None:
        policy = SecurityPolicy()
        executor = CodeExecutor(policy=policy, use_docker=False)
        result = await executor.execute("import os; os.system('ls')")
        assert result.blocked_by_policy
        assert not result.success

    @pytest.mark.asyncio
    async def test_execution_time_recorded(self) -> None:
        executor = self._make_executor()
        result = await executor.execute("x = 1")
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_multiline_code(self) -> None:
        executor = self._make_executor()
        code = "def add(a, b):\n    return a + b\nprint(add(3, 4))"
        result = await executor.execute(code)
        assert result.success
        assert "7" in result.stdout


# ──────────────────────────────────────────────
# SandboxManager
# ──────────────────────────────────────────────


class TestSandboxManager:
    def _make_manager(self) -> SandboxManager:
        return SandboxManager(use_docker=False, max_retries=1)

    @pytest.mark.asyncio
    async def test_run_success(self) -> None:
        manager = self._make_manager()
        result = await manager.run("print('ok')")
        assert result.success
        assert "ok" in result.stdout

    @pytest.mark.asyncio
    async def test_run_failure(self) -> None:
        manager = self._make_manager()
        result = await manager.run("raise RuntimeError('oops')")
        assert not result.success

    @pytest.mark.asyncio
    async def test_to_memory_result(self) -> None:
        from src.sandbox.executor import ExecutionResult as SandboxResult
        manager = self._make_manager()

        sandbox_result = SandboxResult(
            stdout="hello", stderr="", exit_code=0,
            execution_time_ms=100.0, language="python"
        )
        mem_result = manager.to_memory_result(sandbox_result)
        assert mem_result.success is True
        assert mem_result.stdout == "hello"

    @pytest.mark.asyncio
    async def test_blocked_policy_no_retry(self) -> None:
        """ポリシーブロックはリトライしない。"""
        manager = self._make_manager()
        result = await manager.run("import os; os.system('ls')")
        assert result.blocked_by_policy
