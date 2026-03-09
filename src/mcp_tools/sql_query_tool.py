"""src/mcp_tools/sql_query_tool.py — テキスト → SQL 変換・実行ツール

自然言語クエリを LLM で SQL に変換し、SQLite DB に対して実行する。
MED 記憶モデルの「ノート（宣言的記憶）」層を担う。

設計:
- テキスト入力 → LLM で SQL 生成 → aiosqlite で実行
- SELECT のみ許可（DDL/DML をブロック）
- 結果を列名付き辞書リストで返す

使い方:
    from src.mcp_tools.sql_query_tool import SQLQueryTool

    tool = SQLQueryTool(gateway, db_path="data/metadata.db")
    results = await tool.query("Show me the top 5 documents by quality score")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import aiosqlite

from src.llm.gateway import LLMGateway

logger = logging.getLogger(__name__)

_SQL_SYSTEM = """\
You are a SQL expert. Convert the natural language question to a SQLite SELECT query.
Output ONLY the SQL query, no explanation, no markdown.

Available tables:
- documents(id TEXT, content TEXT, domain TEXT, difficulty TEXT, review_status TEXT,
            retrieval_count INTEGER, selection_count INTEGER,
            positive_feedback INTEGER, negative_feedback INTEGER,
            teacher_quality REAL, execution_success_rate REAL,
            freshness REAL, composite_score REAL, confidence REAL,
            created_at TEXT, updated_at TEXT)

Rules:
- Use SELECT only. No INSERT, UPDATE, DELETE, DROP, CREATE, ALTER.
- Limit results to 100 rows unless specified otherwise."""

_SQL_PROMPT = "Convert to SQL: {question}"

_BLOCKED_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|PRAGMA|ATTACH)\b",
    re.IGNORECASE,
)


@dataclass
class SQLResult:
    """SQL 実行結果。"""

    question: str
    sql: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def row_count(self) -> int:
        return len(self.rows)


class SQLQueryTool:
    """テキスト → SQL 変換・実行ツール。

    Args:
        gateway: LLMGateway インスタンス（SQL 生成に使用）。
        db_path: SQLite DB ファイルパス。
        provider: 優先 LLM プロバイダ。
        max_rows: 返す最大行数。
    """

    def __init__(
        self,
        gateway: LLMGateway,
        db_path: str = ":memory:",
        provider: Optional[str] = None,
        max_rows: int = 100,
    ) -> None:
        self._gateway = gateway
        self._db_path = db_path
        self._provider = provider
        self._max_rows = max_rows

    async def query(self, question: str) -> SQLResult:
        """自然言語クエリを SQL に変換して実行する。

        Args:
            question: 自然言語の質問文。

        Returns:
            SQLResult オブジェクト。
        """
        # SQL 生成
        try:
            sql = await self._generate_sql(question)
        except Exception:
            logger.exception("SQL generation failed for: %r", question)
            return SQLResult(question=question, sql="", error="SQL generation failed")

        # セキュリティチェック
        if _BLOCKED_KEYWORDS.search(sql):
            logger.warning("Blocked SQL: %r", sql[:100])
            return SQLResult(question=question, sql=sql, error="Non-SELECT statement blocked")

        # 実行
        try:
            rows = await self._execute(sql)
            logger.debug("SQLQueryTool: %d rows for question=%r", len(rows), question[:50])
            return SQLResult(question=question, sql=sql, rows=rows[:self._max_rows])
        except Exception as e:
            logger.exception("SQL execution failed: %r", sql[:100])
            return SQLResult(question=question, sql=sql, error=str(e))

    async def execute_raw(self, sql: str) -> SQLResult:
        """SQL を直接実行する（テスト・デバッグ用）。

        SELECT のみ許可。
        """
        question = f"[raw] {sql[:80]}"
        if _BLOCKED_KEYWORDS.search(sql):
            return SQLResult(question=question, sql=sql, error="Non-SELECT statement blocked")
        try:
            rows = await self._execute(sql)
            return SQLResult(question=question, sql=sql, rows=rows[:self._max_rows])
        except Exception as e:
            return SQLResult(question=question, sql=sql, error=str(e))

    async def _generate_sql(self, question: str) -> str:
        response = await self._gateway.complete(
            _SQL_PROMPT.format(question=question),
            system=_SQL_SYSTEM,
            provider=self._provider,
            max_tokens=200,
            temperature=0.0,
        )
        sql = response.content.strip()
        # コードブロックの除去
        sql = re.sub(r"```(?:sql)?\s*", "", sql).strip().rstrip("`").strip()
        return sql

    async def _execute(self, sql: str) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(sql) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
