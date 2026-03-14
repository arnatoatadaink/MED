"""src/sandbox/retry_handler.py — サンドボックス実行リトライ制御

コード実行失敗時の自動リトライ + エラー解析 + 自動修正ループを担当する。

フロー:
    1. executor.run(code) → ExecutionResult
    2. 失敗なら ErrorAnalyzer でエラー分析
    3. fixed_code があれば再試行（最大 max_retries 回）
    4. 成功 or 上限到達で RetryResult を返す

使い方:
    from src.sandbox.retry_handler import RetryHandler

    handler = RetryHandler(executor, gateway)
    result = await handler.run_with_retry(code, language="python")
    print(result.success, result.final_output, result.attempts)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RetryResult:
    """リトライ制御の最終結果。"""

    success: bool
    final_output: str = ""
    final_error: str = ""
    attempts: int = 0
    final_code: str = ""
    error_history: list[dict] = field(default_factory=list)

    @property
    def was_fixed(self) -> bool:
        """途中でコードが修正されたか。"""
        return self.attempts > 1 and self.success


class RetryHandler:
    """サンドボックス実行のリトライ制御クラス。

    Args:
        executor: SandboxExecutor インスタンス。
        gateway: LLMGateway インスタンス（エラー分析に使用、省略可）。
        max_retries: 最大リトライ回数（初回実行を含まない）。
        retry_delay: リトライ間の待機秒数。
        use_error_analyzer: True のとき LLM でエラー解析・自動修正を試みる。
    """

    def __init__(
        self,
        executor: object,
        gateway: object | None = None,
        max_retries: int = 2,
        retry_delay: float = 0.5,
        use_error_analyzer: bool = True,
    ) -> None:
        self._executor = executor
        self._gateway = gateway
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._use_error_analyzer = use_error_analyzer

        self._analyzer: object | None = None
        if gateway is not None and use_error_analyzer:
            try:
                from src.llm.error_analyzer import ErrorAnalyzer
                self._analyzer = ErrorAnalyzer(gateway)
            except Exception:
                logger.warning("Failed to initialize ErrorAnalyzer; proceeding without fix")

    async def run_with_retry(
        self,
        code: str,
        language: str = "python",
        timeout_seconds: int = 10,
    ) -> RetryResult:
        """コードを実行し、失敗時にリトライする。

        Args:
            code: 実行コード。
            language: プログラミング言語。
            timeout_seconds: 実行タイムアウト秒数。

        Returns:
            RetryResult。
        """
        current_code = code
        error_history: list[dict] = []

        for attempt in range(1, self._max_retries + 2):  # 初回 + max_retries 回
            logger.debug("RetryHandler: attempt=%d/%d", attempt, self._max_retries + 1)

            try:
                result = await self._executor.run(
                    current_code,
                    language=language,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                logger.exception("Executor raised exception on attempt %d", attempt)
                error_history.append({
                    "attempt": attempt,
                    "error": str(exc),
                    "code": current_code,
                })
                if attempt > self._max_retries:
                    return RetryResult(
                        success=False,
                        final_error=str(exc),
                        attempts=attempt,
                        final_code=current_code,
                        error_history=error_history,
                    )
                await asyncio.sleep(self._retry_delay)
                continue

            if result.success:
                return RetryResult(
                    success=True,
                    final_output=result.stdout,
                    attempts=attempt,
                    final_code=current_code,
                    error_history=error_history,
                )

            # 失敗: エラー分析
            error_text = result.stderr or result.error_message or ""
            error_history.append({
                "attempt": attempt,
                "error": error_text[:500],
                "code": current_code,
            })
            logger.warning(
                "Execution failed (attempt %d): %s",
                attempt, error_text[:100],
            )

            if attempt > self._max_retries:
                break

            # 自動修正を試みる
            fixed_code = await self._try_fix(current_code, error_text)
            if fixed_code and fixed_code.strip() != current_code.strip():
                logger.info("RetryHandler: applying auto-fix on attempt %d", attempt)
                current_code = fixed_code
            else:
                logger.info("RetryHandler: no fix available; retrying as-is")

            if self._retry_delay > 0:
                await asyncio.sleep(self._retry_delay)

        return RetryResult(
            success=False,
            final_error=error_history[-1]["error"] if error_history else "",
            attempts=self._max_retries + 1,
            final_code=current_code,
            error_history=error_history,
        )

    async def _try_fix(self, code: str, error_output: str) -> str | None:
        """エラー解析で修正コードを取得する。"""
        if self._analyzer is None:
            return None
        try:
            analysis = await self._analyzer.analyze(code, error_output)
            if analysis.has_fix:
                return analysis.fixed_code
        except Exception:
            logger.exception("ErrorAnalyzer failed in RetryHandler")
        return None
