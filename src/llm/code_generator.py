"""src/llm/code_generator.py — コード生成特化 LLM インターフェース

コード生成に特化したプロンプト設計で、実行可能なコードスニペットを生成する。
生成コードをコードブロックから抽出し、Sandbox で実行できる形式で返す。

使い方:
    from src.llm.code_generator import CodeGenerator

    generator = CodeGenerator(gateway)
    result = await generator.generate(
        task="Python でリストを逆順にする関数を書いて",
        language="python",
        context_code="# 既存コードがあれば",
    )
    print(result.code)
    print(result.explanation)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from src.llm.gateway import LLMGateway, LLMMessage

logger = logging.getLogger(__name__)

_CODE_SYSTEM = """\
You are an expert software engineer. Generate clean, working, well-commented code. \
Always wrap code in a single markdown code block with the language specified. \
Include a brief explanation after the code block."""

_CODE_TEMPLATE = """\
Task: {task}

Language: {language}
{context_section}
Generate the code:"""

_CONTEXT_SECTION = """
Existing code context:
```{language}
{context_code}
```
"""


@dataclass
class CodeResult:
    """コード生成結果。"""

    code: str
    explanation: str
    language: str
    task: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    is_complete: bool = True  # コードブロックが正常に抽出できた場合 True


class CodeGenerator:
    """コード生成特化 LLM インターフェース。

    Args:
        gateway: LLMGateway インスタンス。
        provider: 優先プロバイダ（省略時はデフォルト）。
        default_language: デフォルト言語（"python"）。
    """

    def __init__(
        self,
        gateway: LLMGateway,
        provider: Optional[str] = None,
        default_language: str = "python",
    ) -> None:
        self._gateway = gateway
        self._provider = provider
        self._default_language = default_language

    async def generate(
        self,
        task: str,
        language: Optional[str] = None,
        context_code: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,  # コードは低温で安定性重視
    ) -> CodeResult:
        """コード生成タスクを LLM に送り、結果を返す。

        Args:
            task: コード生成タスクの説明。
            language: プログラミング言語（省略時はデフォルト）。
            context_code: 既存コードのコンテキスト（省略可）。
            max_tokens: 最大出力トークン数。
            temperature: 温度パラメータ（低いほど安定）。

        Returns:
            CodeResult オブジェクト。
        """
        lang = language or self._default_language

        context_section = ""
        if context_code and context_code.strip():
            context_section = _CONTEXT_SECTION.format(
                language=lang,
                context_code=context_code.strip(),
            )

        prompt = _CODE_TEMPLATE.format(
            task=task,
            language=lang,
            context_section=context_section,
        )

        messages = [
            LLMMessage(role="system", content=_CODE_SYSTEM),
            LLMMessage(role="user", content=prompt),
        ]

        raw = await self._gateway.complete_messages(
            messages,
            provider=self._provider,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        code, explanation, is_complete = self._extract_code(raw.content, lang)

        logger.debug(
            "CodeGenerator: task=%r lang=%s code_len=%d provider=%s",
            task[:50], lang, len(code), raw.provider,
        )

        return CodeResult(
            code=code,
            explanation=explanation,
            language=lang,
            task=task,
            provider=raw.provider,
            model=raw.model,
            input_tokens=raw.input_tokens,
            output_tokens=raw.output_tokens,
            is_complete=is_complete,
        )

    def _extract_code(
        self, content: str, language: str
    ) -> tuple[str, str, bool]:
        """LLM の出力からコードブロックと説明を抽出する。

        Returns:
            (code, explanation, is_complete)
        """
        # ```python ... ``` または ``` ... ``` にマッチ
        pattern = rf"```(?:{re.escape(language)}|)\s*\n(.*?)\n```"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)

        if match:
            code = match.group(1).strip()
            # コードブロック以降をexplanationとして取得
            after_block = content[match.end():].strip()
            explanation = after_block[:1000] if after_block else ""
            return code, explanation, True

        # コードブロックが見つからない場合はコンテンツ全体をコードとして扱う
        logger.warning("No code block found in LLM output; using raw content")
        return content.strip(), "", False
