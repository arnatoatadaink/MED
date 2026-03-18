"""src/memory/alias_extractor.py — ドキュメントエイリアス抽出

取り込んだドキュメントの主概念に対する別名・略称・通称を Teacher API で抽出する。
SQLite の aliases 列に保存し、用語不一致（Vocabulary Mismatch）を緩和する。

セキュリティ設計:
  - 外部ドキュメント内容は先頭 800 文字にトランケートして LLM に渡す
  - 出力フォーマットを JSON 配列に厳格に制限（プロンプトインジェクション緩和）
  - JSON パース失敗時は空リストを返す（安全なデフォルト）
  - 抽出結果は文字列データとして SQLite に保存されるだけ（コード実行されない）

コスト設計:
  - ドキュメント取り込み時に一回だけ実行（クエリ時には実行しない）
  - Haiku / gpt-4o-mini 使用で ~$0.0001/件、10,000 件 ≈ $1

使い方:
    extractor = AliasExtractor(gateway)
    aliases = await extractor.extract(doc)
    # → ["TinyLoRA", "13-param LoRA", "Learning to Reason in 13 Parameters"]
"""

from __future__ import annotations

import json
import logging
import re

from src.llm.gateway import LLMGateway, LLMMessage
from src.memory.schema import Document

logger = logging.getLogger(__name__)

# ── プロンプト ──────────────────────────────────────────────────────
_ALIAS_SYSTEM = """\
You are a metadata extraction assistant.
Your ONLY task: extract alternative names, nicknames, abbreviations, and aliases \
for the MAIN CONCEPT discussed in the given text.

Output ONLY a valid JSON array of strings, nothing else.
Example output: ["TinyLoRA", "13-param LoRA", "Learning to Reason in 13 Parameters"]

Rules:
- Maximum 8 aliases.
- Include the most common variants (full title, short name, abbreviation, nickname).
- If no clear aliases exist, output [].
- Do NOT output explanations, markdown, or any text outside the JSON array."""

_ALIAS_PROMPT = """\
Extract aliases for the main concept in this text (truncated to 800 chars):

{text}"""

# ── 入力の最大長（セキュリティ: 攻撃面の縮小） ────────────────────
_MAX_INPUT_CHARS = 800


class AliasExtractor:
    """Teacher API を用いてドキュメントの主概念エイリアスを抽出する。

    Args:
        gateway:      LLMGateway インスタンス。
        provider:     使用するプロバイダー（省略時は gateway デフォルト）。
        model:        使用するモデル（省略時は gateway デフォルト、軽量モデル推奨）。
        enabled:      False にすると extract() が常に空リストを返す（無効化）。
    """

    def __init__(
        self,
        gateway: LLMGateway,
        provider: str | None = None,
        model: str | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        self._gateway = gateway
        self._provider = provider
        self._model = model
        self._enabled = enabled

    async def extract(self, doc: Document) -> list[str]:
        """ドキュメントからエイリアスリストを抽出する。

        Args:
            doc: 対象ドキュメント。

        Returns:
            エイリアス文字列のリスト。失敗時は空リスト。
        """
        if not self._enabled:
            return []

        # セキュリティ: 先頭 800 文字に制限（プロンプトインジェクション攻撃面を縮小）
        truncated = doc.content[:_MAX_INPUT_CHARS]

        # ドキュメントタイトルがある場合は先頭に追加（精度向上）
        title = doc.source.title or ""
        if title:
            truncated = f"Title: {title}\n\n{truncated}"

        messages = [
            LLMMessage(role="user", content=_ALIAS_PROMPT.format(text=truncated)),
        ]

        try:
            response = await self._gateway.complete(
                messages,
                system=_ALIAS_SYSTEM,
                provider=self._provider,
                model=self._model,
                max_tokens=256,
            )
            raw = response.content.strip()

            # JSON 配列を抽出（余分なテキストが混入しても安全にパース）
            aliases = self._parse_aliases(raw)
            logger.debug(
                "AliasExtractor: doc=%s → %d aliases: %s",
                doc.id[:8], len(aliases), aliases,
            )
            return aliases

        except Exception:
            logger.debug("AliasExtractor: extraction failed for doc=%s (non-fatal)", doc.id[:8])
            return []

    @staticmethod
    def _parse_aliases(raw: str) -> list[str]:
        """LLM 出力から JSON 配列を安全にパースする。

        プロンプトインジェクションで余分なテキストが混入しても、
        JSON 配列部分だけを抽出してパースする。失敗時は空リストを返す。
        """
        # JSON 配列部分のみを抽出
        m = re.search(r'\[.*?\]', raw, re.DOTALL)
        if not m:
            return []
        try:
            result = json.loads(m.group())
            if not isinstance(result, list):
                return []
            # 文字列のみ抽出、長すぎる値は除去（注入対策）
            return [
                str(item).strip()
                for item in result
                if isinstance(item, str) and 0 < len(str(item).strip()) <= 200
            ]
        except (json.JSONDecodeError, ValueError):
            return []
