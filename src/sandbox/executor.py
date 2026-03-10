"""src/sandbox/executor.py — コード実行エンジン

Docker コンテナ内でコードを安全に実行する。
docker-py が利用可能な場合は Docker コンテナを使用し、
利用できない場合は subprocess を使ったフォールバック実行を行う（開発時のみ）。

使い方:
    from src.sandbox.executor import CodeExecutor

    executor = CodeExecutor()
    result = await executor.execute("print('hello')", language="python")
    print(result.stdout)
    print(result.stderr)
    print(result.exit_code)
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.sandbox.security import SecurityPolicy, _DEFAULT_POLICY

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """コード実行結果。"""

    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: float
    language: str
    timed_out: bool = False
    blocked_by_policy: bool = False
    policy_violations: list[str] = None

    def __post_init__(self) -> None:
        if self.policy_violations is None:
            self.policy_violations = []

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.blocked_by_policy


class CodeExecutor:
    """Docker または subprocess でコードを実行する。

    Args:
        policy: セキュリティポリシー。
        use_docker: Docker を使用するか（False = subprocess フォールバック）。
        docker_image: 使用する Docker イメージ。
    """

    def __init__(
        self,
        policy: Optional[SecurityPolicy] = None,
        use_docker: bool = True,
        docker_image: str = "python:3.11-slim",
    ) -> None:
        self._policy = policy or _DEFAULT_POLICY
        self._use_docker = use_docker and self._docker_available()
        self._docker_image = docker_image

        if not self._use_docker:
            logger.warning(
                "Docker not available or disabled; using subprocess fallback (dev only)"
            )

    def _docker_available(self) -> bool:
        """Docker が利用可能か確認する。"""
        try:
            import docker
            client = docker.from_env()
            client.ping()
            return True
        except Exception:
            return False

    async def execute(
        self,
        code: str,
        language: str = "python",
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """コードを実行する。

        Args:
            code: 実行するコード。
            language: プログラミング言語（現在は "python" のみサポート）。
            timeout: タイムアウト秒数（省略時はポリシーのデフォルト）。

        Returns:
            ExecutionResult オブジェクト。
        """
        effective_timeout = timeout or self._policy.timeout_seconds

        # セキュリティチェック
        is_safe, violations = self._policy.check_code(code)
        if not is_safe:
            logger.warning("Code blocked by policy: %s", violations)
            return ExecutionResult(
                stdout="",
                stderr=f"Code blocked by security policy: {'; '.join(violations)}",
                exit_code=1,
                execution_time_ms=0.0,
                language=language,
                blocked_by_policy=True,
                policy_violations=violations,
            )

        if self._use_docker:
            return await self._execute_docker(code, language, effective_timeout)
        else:
            return await self._execute_subprocess(code, language, effective_timeout)

    async def _execute_docker(
        self,
        code: str,
        language: str,
        timeout: int,
    ) -> ExecutionResult:
        """Docker コンテナでコードを実行する。"""
        try:
            import docker
            from docker.errors import ContainerError, ImageNotFound
        except ImportError:
            logger.warning("docker package not available; falling back to subprocess")
            return await self._execute_subprocess(code, language, timeout)

        start = time.monotonic()
        try:
            client = docker.from_env()
            run_kwargs = {
                "image": self._docker_image,
                "command": ["python", "-c", code] if language == "python" else ["sh", "-c", code],
                "remove": True,
                "mem_limit": f"{self._policy.memory_limit_mb}m",
                "network_disabled": not self._policy.allow_network,
                "stdout": True,
                "stderr": True,
            }

            # 非同期でブロッキング Docker 呼び出しを実行
            loop = asyncio.get_event_loop()
            container_output = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: client.containers.run(**run_kwargs)),
                timeout=float(timeout),
            )

            elapsed = (time.monotonic() - start) * 1000
            stdout = container_output.decode("utf-8", errors="replace")
            stdout = stdout[: self._policy.max_output_length]

            return ExecutionResult(
                stdout=stdout,
                stderr="",
                exit_code=0,
                execution_time_ms=elapsed,
                language=language,
            )

        except asyncio.TimeoutError:
            elapsed = (time.monotonic() - start) * 1000
            return ExecutionResult(
                stdout="",
                stderr=f"Execution timed out after {timeout}s",
                exit_code=124,
                execution_time_ms=elapsed,
                language=language,
                timed_out=True,
            )
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            return ExecutionResult(
                stdout="",
                stderr=str(exc),
                exit_code=1,
                execution_time_ms=elapsed,
                language=language,
            )

    async def _execute_subprocess(
        self,
        code: str,
        language: str,
        timeout: int,
    ) -> ExecutionResult:
        """subprocess でコードを実行する（開発/テスト用フォールバック）。"""
        start = time.monotonic()

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py" if language == "python" else ".sh",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            cmd = ["python", tmp_path] if language == "python" else ["sh", tmp_path]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=float(timeout)
                )
                elapsed = (time.monotonic() - start) * 1000
                stdout = stdout_b.decode("utf-8", errors="replace")[: self._policy.max_output_length]
                stderr = stderr_b.decode("utf-8", errors="replace")[: self._policy.max_output_length]
                return ExecutionResult(
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=proc.returncode or 0,
                    execution_time_ms=elapsed,
                    language=language,
                )
            except asyncio.TimeoutError:
                proc.kill()
                elapsed = (time.monotonic() - start) * 1000
                return ExecutionResult(
                    stdout="",
                    stderr=f"Execution timed out after {timeout}s",
                    exit_code=124,
                    execution_time_ms=elapsed,
                    language=language,
                    timed_out=True,
                )
        finally:
            Path(tmp_path).unlink(missing_ok=True)
