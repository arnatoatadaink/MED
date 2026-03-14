"""src/training/rewards/code_exec.py — コード実行報酬

レスポンス中のコードを実行して成否を報酬とする。
SandboxManager を使用してセキュアに実行する。

使い方:
    from src.training.rewards.code_exec import CodeExecReward
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.training.base import RewardFunction
from src.training.registry import TrainingRegistry

logger = logging.getLogger(__name__)

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


@TrainingRegistry.reward("code_exec")
class CodeExecReward(RewardFunction):
    """コード実行成功報酬。

    Args:
        sandbox: SandboxManager インスタンス (省略時はスタブ)。
        success_score: 実行成功時のスコア。
        failure_score: 実行失敗時のスコア。
        no_code_score: コードなし時のスコア。
    """

    def __init__(
        self,
        sandbox: Any = None,
        success_score: float = 1.0,
        failure_score: float = 0.0,
        no_code_score: float = 0.5,
    ) -> None:
        self._sandbox = sandbox
        self._success_score = success_score
        self._failure_score = failure_score
        self._no_code_score = no_code_score

    @property
    def name(self) -> str:
        return "code_exec"

    async def compute(
        self,
        prompt: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """レスポンス中のコードを実行して報酬を返す。"""
        meta = metadata or {}

        # メタデータに既存結果があれば使う
        if "exec_success" in meta:
            return self._success_score if meta["exec_success"] else self._failure_score

        # コードブロックを抽出
        code = self._extract_code(response)
        if not code:
            return self._no_code_score

        # サンドボックス実行
        if self._sandbox is not None:
            try:
                result = await self._sandbox.run(code)
                return self._success_score if result.success else self._failure_score
            except Exception:
                logger.exception("CodeExecReward: sandbox execution failed")
                return self._failure_score

        # サンドボックスなし: コードが存在すれば中間スコア
        return (self._success_score + self._failure_score) / 2

    def _extract_code(self, response: str) -> str:
        """レスポンスからコードブロックを抽出する。"""
        match = _CODE_BLOCK_RE.search(response)
        if match:
            return match.group(1).strip()
        return ""
