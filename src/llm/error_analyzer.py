"""src/llm/error_analyzer.py — エラー分析器

コード実行エラーや LLM エラーを分析し、修正提案を生成する。
SandboxManager → ErrorAnalyzer → CodeGenerator の連携で自動修正ループを実現する。

使い方:
    from src.llm.error_analyzer import ErrorAnalyzer

    analyzer = ErrorAnalyzer(gateway)
    analysis = await analyzer.analyze(code, error_output)
    # → ErrorAnalysis(error_type="NameError", suggestion="変数名を確認してください", ...)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from src.llm.gateway import LLMGateway

logger = logging.getLogger(__name__)

_ANALYZE_SYSTEM = """\
You are a Python debugging expert. Analyze the error and provide a fix.
Respond with ONLY valid JSON:
{
  "error_type": "error class name",
  "error_line": line number or null,
  "root_cause": "brief explanation",
  "suggestion": "how to fix it",
  "fixed_code": "corrected code or empty string if unsure"
}"""

_ANALYZE_PROMPT = """\
Code:
```python
{code}
```

Error output:
{error}

Analyze and provide fix."""

# 正規表現ベースのエラータイプ検出
_ERROR_PATTERNS = [
    (re.compile(r"NameError"), "NameError"),
    (re.compile(r"TypeError"), "TypeError"),
    (re.compile(r"ValueError"), "ValueError"),
    (re.compile(r"AttributeError"), "AttributeError"),
    (re.compile(r"ImportError|ModuleNotFoundError"), "ImportError"),
    (re.compile(r"IndexError"), "IndexError"),
    (re.compile(r"KeyError"), "KeyError"),
    (re.compile(r"ZeroDivisionError"), "ZeroDivisionError"),
    (re.compile(r"SyntaxError"), "SyntaxError"),
    (re.compile(r"IndentationError"), "IndentationError"),
    (re.compile(r"RuntimeError"), "RuntimeError"),
    (re.compile(r"TimeoutError|Timeout"), "TimeoutError"),
]
_LINE_PATTERN = re.compile(r"line (\d+)", re.IGNORECASE)


@dataclass
class ErrorAnalysis:
    """エラー分析結果。"""

    error_type: str
    error_line: Optional[int]
    root_cause: str
    suggestion: str
    fixed_code: str = ""
    raw_error: str = ""

    @property
    def has_fix(self) -> bool:
        return bool(self.fixed_code.strip())


class ErrorAnalyzer:
    """コード実行エラー分析器。

    Args:
        gateway: LLMGateway インスタンス。
        provider: 優先プロバイダ。
        max_code_chars: LLM に渡すコードの最大文字数。
        max_error_chars: LLM に渡すエラーの最大文字数。
    """

    def __init__(
        self,
        gateway: LLMGateway,
        provider: Optional[str] = None,
        max_code_chars: int = 1500,
        max_error_chars: int = 500,
    ) -> None:
        self._gateway = gateway
        self._provider = provider
        self._max_code = max_code_chars
        self._max_error = max_error_chars

    async def analyze(
        self,
        code: str,
        error_output: str,
    ) -> ErrorAnalysis:
        """エラーを分析して修正提案を生成する。

        Args:
            code: エラーが発生したコード。
            error_output: stderr / 例外トレースバック。

        Returns:
            ErrorAnalysis オブジェクト。
        """
        # ローカルパターンマッチ（高速フォールバック）
        error_type = self._detect_error_type(error_output)
        error_line = self._detect_line(error_output)

        try:
            import json
            response = await self._gateway.complete(
                _ANALYZE_PROMPT.format(
                    code=code[:self._max_code],
                    error=error_output[:self._max_error],
                ),
                system=_ANALYZE_SYSTEM,
                provider=self._provider,
                max_tokens=400,
                temperature=0.0,
            )
            content = re.sub(r"```(?:json)?\s*", "", response.content).strip().rstrip("`")
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                return ErrorAnalysis(
                    error_type=data.get("error_type", error_type),
                    error_line=data.get("error_line") or error_line,
                    root_cause=str(data.get("root_cause", "")),
                    suggestion=str(data.get("suggestion", "")),
                    fixed_code=str(data.get("fixed_code", "")),
                    raw_error=error_output,
                )
        except Exception:
            logger.exception("ErrorAnalyzer LLM call failed")

        # フォールバック: パターンマッチのみ
        return ErrorAnalysis(
            error_type=error_type,
            error_line=error_line,
            root_cause=f"{error_type} detected in output",
            suggestion=self._default_suggestion(error_type),
            fixed_code="",
            raw_error=error_output,
        )

    def _detect_error_type(self, error_output: str) -> str:
        for pattern, name in _ERROR_PATTERNS:
            if pattern.search(error_output):
                return name
        return "UnknownError"

    def _detect_line(self, error_output: str) -> Optional[int]:
        m = _LINE_PATTERN.search(error_output)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
        return None

    def _default_suggestion(self, error_type: str) -> str:
        _SUGGESTIONS = {
            "NameError": "Check variable/function names for typos.",
            "TypeError": "Check argument types and count.",
            "ValueError": "Check input value constraints.",
            "AttributeError": "Verify the object has the expected attribute.",
            "ImportError": "Install the missing package with pip.",
            "IndexError": "Check list/array bounds.",
            "KeyError": "Verify dictionary key exists before access.",
            "ZeroDivisionError": "Add a zero-check before division.",
            "SyntaxError": "Check indentation and syntax.",
            "IndentationError": "Fix indentation consistency.",
            "TimeoutError": "Optimize code or increase timeout.",
        }
        return _SUGGESTIONS.get(error_type, "Review the error message and traceback.")
