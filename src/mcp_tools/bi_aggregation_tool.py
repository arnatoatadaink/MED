"""src/mcp_tools/bi_aggregation_tool.py — BI 集計クエリツール

COUNT / SUM / AVG などの集計クエリを安全に実行する。
SQLQueryTool の上位レイヤとして、集計専用の高レベル API を提供する。

設計:
- 集計関数: COUNT, SUM, AVG, MIN, MAX
- グループ化: GROUP BY domain, difficulty, review_status など
- テーブルは metadata DB の documents / usefulness_scores に限定
- LLM 不要（SQL テンプレートを直接組み立てる）

使い方:
    from src.mcp_tools.bi_aggregation_tool import BIAggregationTool

    tool = BIAggregationTool(db_path="data/metadata.db")
    result = await tool.count("documents", group_by="domain")
    result = await tool.average("usefulness_scores", column="composite", group_by="domain")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import aiosqlite

logger = logging.getLogger(__name__)

AggFunc = Literal["COUNT", "SUM", "AVG", "MIN", "MAX"]

_ALLOWED_TABLES = {"documents"}
_ALLOWED_COLUMNS = {
    "documents": {
        "id", "content", "domain", "difficulty", "review_status",
        "retrieval_count", "selection_count",
        "positive_feedback", "negative_feedback",
        "teacher_quality", "execution_success_rate",
        "freshness", "composite_score", "confidence",
        "created_at", "updated_at",
    },
}


@dataclass
class AggResult:
    """集計クエリ結果。"""

    sql: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def scalar(self) -> Optional[Any]:
        """単一行・単一列の結果を返す（COUNT(*) など）。"""
        if self.rows and len(self.rows) == 1:
            values = list(self.rows[0].values())
            if values:
                return values[0]
        return None


class BIAggregationTool:
    """BI 集計クエリツール。

    Args:
        db_path: SQLite DB ファイルパス。
        max_rows: 返す最大行数。
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        max_rows: int = 1000,
    ) -> None:
        self._db_path = db_path
        self._max_rows = max_rows

    async def count(
        self,
        table: str,
        column: str = "*",
        where: Optional[str] = None,
        group_by: Optional[str] = None,
    ) -> AggResult:
        """COUNT 集計を実行する。

        Args:
            table: 対象テーブル名。
            column: カウント対象カラム（デフォルト: *）。
            where: WHERE 句（SQL インジェクション注意 — 信頼できる値のみ渡すこと）。
            group_by: GROUP BY カラム名。

        Returns:
            AggResult。
        """
        return await self._aggregate("COUNT", table, column, where, group_by)

    async def sum(
        self,
        table: str,
        column: str,
        where: Optional[str] = None,
        group_by: Optional[str] = None,
    ) -> AggResult:
        """SUM 集計を実行する。"""
        return await self._aggregate("SUM", table, column, where, group_by)

    async def average(
        self,
        table: str,
        column: str,
        where: Optional[str] = None,
        group_by: Optional[str] = None,
    ) -> AggResult:
        """AVG 集計を実行する。"""
        return await self._aggregate("AVG", table, column, where, group_by)

    async def min(
        self,
        table: str,
        column: str,
        where: Optional[str] = None,
        group_by: Optional[str] = None,
    ) -> AggResult:
        """MIN 集計を実行する。"""
        return await self._aggregate("MIN", table, column, where, group_by)

    async def max(
        self,
        table: str,
        column: str,
        where: Optional[str] = None,
        group_by: Optional[str] = None,
    ) -> AggResult:
        """MAX 集計を実行する。"""
        return await self._aggregate("MAX", table, column, where, group_by)

    async def domain_stats(self) -> AggResult:
        """ドメイン別ドキュメント数を返す。"""
        sql = (
            "SELECT domain, COUNT(*) as doc_count "
            "FROM documents GROUP BY domain ORDER BY doc_count DESC"
        )
        return await self._execute(sql)

    async def difficulty_distribution(self) -> AggResult:
        """難易度別ドキュメント数を返す。"""
        sql = (
            "SELECT difficulty, COUNT(*) as doc_count "
            "FROM documents GROUP BY difficulty ORDER BY doc_count DESC"
        )
        return await self._execute(sql)

    async def quality_summary(self) -> AggResult:
        """品質スコアの統計サマリを返す。"""
        sql = (
            "SELECT "
            "  COUNT(*) as total_docs, "
            "  ROUND(AVG(teacher_quality), 3) as avg_quality, "
            "  ROUND(AVG(composite_score), 3) as avg_composite, "
            "  ROUND(AVG(confidence), 3) as avg_confidence "
            "FROM documents"
        )
        return await self._execute(sql)

    async def review_status_counts(self) -> AggResult:
        """レビューステータス別ドキュメント数を返す。"""
        sql = (
            "SELECT review_status, COUNT(*) as doc_count "
            "FROM documents GROUP BY review_status"
        )
        return await self._execute(sql)

    # ── 内部メソッド ──────────────────────────────

    def _validate_table(self, table: str) -> Optional[str]:
        if table not in _ALLOWED_TABLES:
            return f"Table '{table}' is not allowed. Use: {sorted(_ALLOWED_TABLES)}"
        return None

    def _validate_column(self, table: str, column: str) -> Optional[str]:
        if column == "*":
            return None
        allowed = _ALLOWED_COLUMNS.get(table, set())
        if column not in allowed:
            return f"Column '{column}' is not allowed for table '{table}'"
        return None

    async def _aggregate(
        self,
        func: str,
        table: str,
        column: str,
        where: Optional[str],
        group_by: Optional[str],
    ) -> AggResult:
        err = self._validate_table(table)
        if err:
            return AggResult(sql="", error=err)

        err = self._validate_column(table, column)
        if err:
            return AggResult(sql="", error=err)

        col_expr = column if column == "*" else column
        select_expr = f"{func}({col_expr}) as result"

        if group_by:
            err = self._validate_column(table, group_by)
            if err:
                return AggResult(sql="", error=err)
            select_expr = f"{group_by}, {func}({col_expr}) as result"

        parts = [f"SELECT {select_expr} FROM {table}"]
        if where:
            parts.append(f"WHERE {where}")
        if group_by:
            parts.append(f"GROUP BY {group_by}")
        parts.append(f"LIMIT {self._max_rows}")

        sql = " ".join(parts)
        return await self._execute(sql)

    async def _execute(self, sql: str) -> AggResult:
        try:
            async with aiosqlite.connect(self._db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(sql) as cursor:
                    rows = await cursor.fetchall()
                    return AggResult(sql=sql, rows=[dict(row) for row in rows])
        except Exception as e:
            logger.exception("BIAggregationTool execution failed: %r", sql[:100])
            return AggResult(sql=sql, error=str(e))
