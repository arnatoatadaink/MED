"""src/sandbox — Docker サンドボックス実行モジュール

CodeExecutor  : ephemeral コンテナ方式（セキュリティポリシー付き、単発実行向け）
DockerRuntime : persistent コンテナ方式（バルク実行向け、MEDの知識収集に使用）
SandboxManager: CodeExecutor のリトライラッパー
"""

from src.sandbox.docker_runtime import DockerRuntime
from src.sandbox.executor import CodeExecutor, ExecutionResult
from src.sandbox.manager import SandboxManager
from src.sandbox.security import SecurityPolicy, _DEFAULT_POLICY

__all__ = [
    "DockerRuntime",
    "CodeExecutor",
    "ExecutionResult",
    "SandboxManager",
    "SecurityPolicy",
    "_DEFAULT_POLICY",
]
