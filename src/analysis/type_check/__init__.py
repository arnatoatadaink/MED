"""Type mismatch detection: AST + Call Graph Analysis (CGA) sub-package."""

from .checker import CheckResult, check_directory

__all__ = ["check_directory", "CheckResult"]
