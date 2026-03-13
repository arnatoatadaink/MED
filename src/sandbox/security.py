"""src/sandbox/security.py — コード実行のセキュリティポリシー

Docker コンテナでのコード実行に適用するセキュリティ制限を管理する。
禁止パターン・リソース制限・ネットワーク制御などを提供する。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 危険なパターン（Python）
_DANGEROUS_PATTERNS = [
    r"\bos\.system\b",
    r"\bsubprocess\b",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\b__import__\s*\(",
    r"\bopen\s*\(.+['\"]w['\"]",      # ファイル書き込み
    r"\bsocket\b",
    r"\bimport\s+socket\b",
    r"\bimport\s+subprocess\b",
    r"\bimport\s+os\b",               # os モジュール全体
    r"\bimport\s+sys\b",
    r"rm\s+-rf",
    r"\bshutil\b",
]

# 許可するインポートの許可リスト（明示的に許可しないものは拒否）
_ALLOWED_IMPORTS = frozenset([
    "math", "random", "datetime", "json", "re", "collections",
    "itertools", "functools", "typing", "dataclasses",
    "numpy", "pandas", "scipy", "sklearn",
    "pathlib",  # 読み取り専用で許可
])


@dataclass
class SecurityPolicy:
    """コード実行セキュリティポリシー。

    Attributes:
        allow_network: ネットワークアクセスを許可するか。
        allow_filesystem_write: ファイルシステムへの書き込みを許可するか。
        memory_limit_mb: メモリ上限 (MB)。
        cpu_limit: CPU 使用率上限 (0.0〜1.0)。
        timeout_seconds: 実行タイムアウト秒数。
        max_output_length: 標準出力の最大文字数。
    """

    allow_network: bool = False
    allow_filesystem_write: bool = False
    memory_limit_mb: int = 256
    cpu_limit: float = 0.5
    timeout_seconds: int = 30
    max_output_length: int = 10000
    blocked_patterns: list[str] = field(default_factory=lambda: list(_DANGEROUS_PATTERNS))

    def check_code(self, code: str) -> tuple[bool, list[str]]:
        """コードを安全性チェックする。

        Returns:
            (is_safe, list_of_violations)
        """
        violations: list[str] = []

        for pattern in self.blocked_patterns:
            if re.search(pattern, code):
                violations.append(f"Blocked pattern: {pattern}")

        return len(violations) == 0, violations


_DEFAULT_POLICY = SecurityPolicy()
