"""src/sandbox/manager.py — Docker Sandbox 管理

CodeExecutor をラップして自動リトライ・結果変換を提供する。

使い方:
    from src.sandbox.manager import SandboxManager

    manager = SandboxManager()
    result = await manager.run("print('hello')", language="python")
"""

from __future__ import annotations

import logging
from typing import Optional

from src.memory.schema import ExecutionResult as MemoryExecutionResult
from src.sandbox.executor import CodeExecutor, ExecutionResult
from src.sandbox.security import SecurityPolicy, _DEFAULT_POLICY

logger = logging.getLogger(__name__)


class SandboxManager:
    """Docker Sandbox 管理クラス。

    CodeExecutor をラップし、自動リトライや結果のスキーマ変換を行う。

    Args:
        policy: セキュリティポリシー。
        max_retries: 実行失敗時の最大リトライ回数。
        use_docker: Docker を使用するか。
    """

    def __init__(
        self,
        policy: Optional[SecurityPolicy] = None,
        max_retries: int = 2,
        use_docker: bool = True,
    ) -> None:
        self._policy = policy or _DEFAULT_POLICY
        self._max_retries = max_retries
        self._executor = CodeExecutor(policy=self._policy, use_docker=use_docker)

    async def run(
        self,
        code: str,
        language: str = "python",
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """コードを実行する（リトライ付き）。

        タイムアウトやポリシーブロックはリトライしない。
        システムエラー（Docker クラッシュ等）のみリトライする。
        """
        last_result: Optional[ExecutionResult] = None

        for attempt in range(self._max_retries + 1):
            result = await self._executor.execute(code, language=language, timeout=timeout)

            # 再試行不要なケース
            if result.success or result.timed_out or result.blocked_by_policy:
                return result

            # 実行自体は動いたが exit_code != 0 (コードエラー) → リトライ不要
            if result.exit_code != 0 and result.stderr:
                return result

            last_result = result
            if attempt < self._max_retries:
                logger.warning(
                    "Sandbox attempt %d/%d failed; retrying", attempt + 1, self._max_retries
                )

        return last_result or ExecutionResult(
            stdout="", stderr="All attempts failed", exit_code=1,
            execution_time_ms=0.0, language=language,
        )

    def to_memory_result(self, result: ExecutionResult) -> MemoryExecutionResult:
        """ExecutionResult を MemoryExecutionResult に変換する。

        memory.schema.ExecutionResult として FAISSメモリへのフィードバックに使用。
        """
        return MemoryExecutionResult(
            success=result.success,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            execution_time_ms=result.execution_time_ms,
            timed_out=result.timed_out,
        )
