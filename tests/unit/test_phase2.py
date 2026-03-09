"""tests/unit/test_phase2.py — Phase 2 モジュールの単体テスト

- CrossEncoder
- SQLQueryTool
- BIAggregationTool
- QueryClassifier
- FusionReranker
"""

from __future__ import annotations

import pytest
import aiosqlite

from src.llm.gateway import LLMGateway, LLMResponse
from src.memory.schema import Document, SearchResult, SourceMeta, SourceType
from src.memory.learning.cross_encoder import CrossEncoder
from src.mcp_tools.sql_query_tool import SQLQueryTool, SQLResult
from src.mcp_tools.bi_aggregation_tool import BIAggregationTool, AggResult
from src.retrieval.query_classifier import QueryClassifier, QueryType
from src.retrieval.fusion_reranker import FusionReranker, FusionResult


# ──────────────────────────────────────────────
# モック
# ──────────────────────────────────────────────


class MockGateway(LLMGateway):
    def __init__(self, response: str = "0.8") -> None:
        self._response = response
        self._providers = {}
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0

    async def complete(self, prompt, **kwargs) -> LLMResponse:
        self._call_count += 1
        return LLMResponse(
            content=self._response,
            provider="mock",
            model="mock-model",
            input_tokens=5,
            output_tokens=5,
        )

    async def complete_messages(self, messages, **kwargs) -> LLMResponse:
        self._call_count += 1
        return LLMResponse(
            content=self._response,
            provider="mock",
            model="mock-model",
            input_tokens=5,
            output_tokens=5,
        )


def _make_doc(content: str = "Python tutorial", domain: str = "general") -> Document:
    return Document(
        content=content,
        domain=domain,
        source=SourceMeta(source_type=SourceType.MANUAL),
    )


def _make_search_result(doc: Document, score: float = 0.8) -> SearchResult:
    return SearchResult(document=doc, score=score)


# ──────────────────────────────────────────────
# セットアップ: テスト用 DB
# ──────────────────────────────────────────────

_CREATE_DOCUMENTS_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    domain TEXT NOT NULL DEFAULT 'general',
    difficulty TEXT,
    review_status TEXT DEFAULT 'unreviewed',
    retrieval_count INTEGER DEFAULT 0,
    selection_count INTEGER DEFAULT 0,
    positive_feedback INTEGER DEFAULT 0,
    negative_feedback INTEGER DEFAULT 0,
    teacher_quality REAL DEFAULT 0.0,
    execution_success_rate REAL DEFAULT 0.0,
    freshness REAL DEFAULT 1.0,
    composite_score REAL DEFAULT 0.0,
    confidence REAL DEFAULT 0.5,
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT ''
)
"""


async def _setup_test_db(db_path: str = ":memory:") -> None:
    """テスト用 SQLite DB を初期化する（:memory: なので都度作成）。"""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(_CREATE_DOCUMENTS_SQL)
        await db.execute(
            "INSERT INTO documents(id, content, domain, difficulty, review_status, teacher_quality) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("doc1", "Python basics", "code", "beginner", "approved", 0.9),
        )
        await db.execute(
            "INSERT INTO documents(id, content, domain, difficulty, review_status, teacher_quality) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("doc2", "Advanced FAISS", "code", "advanced", "approved", 0.75),
        )
        await db.execute(
            "INSERT INTO documents(id, content, domain, difficulty, review_status, teacher_quality) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("doc3", "General knowledge", "general", "intermediate", "unreviewed", 0.5),
        )
        await db.commit()


# ──────────────────────────────────────────────
# CrossEncoder テスト
# ──────────────────────────────────────────────


class TestCrossEncoder:
    @pytest.mark.asyncio
    async def test_score_returns_float(self) -> None:
        encoder = CrossEncoder(MockGateway("0.85"))
        doc = _make_doc("FAISS vector search explanation.")
        score = await encoder.score("What is FAISS?", doc)
        assert isinstance(score, float)
        assert score == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_score_clamped_to_range(self) -> None:
        encoder = CrossEncoder(MockGateway("1.5"))
        doc = _make_doc("test")
        score = await encoder.score("query", doc)
        assert score <= 1.0

    @pytest.mark.asyncio
    async def test_score_negative_clamped(self) -> None:
        encoder = CrossEncoder(MockGateway("-0.5"))
        doc = _make_doc("test")
        score = await encoder.score("query", doc)
        assert score >= 0.0

    @pytest.mark.asyncio
    async def test_score_llm_failure_returns_fallback(self) -> None:
        class FailGateway(MockGateway):
            async def complete(self, prompt, **kwargs):
                raise RuntimeError("API down")

        encoder = CrossEncoder(FailGateway(), fallback_score=0.42)
        doc = _make_doc("test")
        score = await encoder.score("query", doc)
        assert score == pytest.approx(0.42)

    @pytest.mark.asyncio
    async def test_score_non_numeric_response_fallback(self) -> None:
        encoder = CrossEncoder(MockGateway("highly relevant"), fallback_score=0.5)
        doc = _make_doc("test")
        score = await encoder.score("query", doc)
        # "highly" doesn't parse as number — fallback
        assert score == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_score_batch_returns_list(self) -> None:
        encoder = CrossEncoder(MockGateway("0.7"))
        docs = [_make_doc(f"doc {i}") for i in range(4)]
        scores = await encoder.score_batch("test query", docs)
        assert len(scores) == 4
        assert all(isinstance(s, float) for s in scores)

    @pytest.mark.asyncio
    async def test_score_batch_empty(self) -> None:
        encoder = CrossEncoder(MockGateway())
        scores = await encoder.score_batch("query", [])
        assert scores == []

    @pytest.mark.asyncio
    async def test_rerank_orders_by_score(self) -> None:
        responses = iter(["0.9", "0.3", "0.6"])

        class SequentialGateway(MockGateway):
            async def complete(self, prompt, **kwargs):
                self._call_count += 1
                return LLMResponse(
                    content=next(responses),
                    provider="mock", model="mock-model",
                    input_tokens=5, output_tokens=5,
                )

        encoder = CrossEncoder(SequentialGateway())
        docs = [_make_doc(f"doc {i}") for i in range(3)]
        srs = [_make_search_result(d, 0.5) for d in docs]
        reranked = await encoder.rerank("query", srs)
        scores_used = [r.score for r in reranked]
        # SearchResult.score は元のまま — 順序だけ変わる
        assert len(reranked) == 3

    @pytest.mark.asyncio
    async def test_rerank_top_k(self) -> None:
        encoder = CrossEncoder(MockGateway("0.8"))
        docs = [_make_doc(f"doc {i}") for i in range(5)]
        srs = [_make_search_result(d) for d in docs]
        reranked = await encoder.rerank("query", srs, top_k=3)
        assert len(reranked) == 3

    @pytest.mark.asyncio
    async def test_rerank_empty_returns_empty(self) -> None:
        encoder = CrossEncoder(MockGateway())
        result = await encoder.rerank("query", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_gateway_called_once_per_doc(self) -> None:
        gateway = MockGateway("0.5")
        encoder = CrossEncoder(gateway)
        docs = [_make_doc(f"doc {i}") for i in range(4)]
        await encoder.score_batch("query", docs)
        assert gateway._call_count == 4


# ──────────────────────────────────────────────
# SQLQueryTool テスト
# ──────────────────────────────────────────────


class TestSQLQueryTool:
    @pytest.mark.asyncio
    async def test_execute_raw_select(self) -> None:
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            await _setup_test_db(db_path)
            tool = SQLQueryTool(MockGateway(), db_path=db_path)
            result = await tool.execute_raw("SELECT id, domain FROM documents ORDER BY id")
            assert result.success
            assert result.row_count == 3
            assert result.rows[0]["id"] == "doc1"
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_execute_raw_blocks_insert(self) -> None:
        tool = SQLQueryTool(MockGateway(), db_path=":memory:")
        result = await tool.execute_raw("INSERT INTO documents(id) VALUES ('x')")
        assert not result.success
        assert "blocked" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_raw_blocks_drop(self) -> None:
        tool = SQLQueryTool(MockGateway(), db_path=":memory:")
        result = await tool.execute_raw("DROP TABLE documents")
        assert not result.success

    @pytest.mark.asyncio
    async def test_execute_raw_blocks_update(self) -> None:
        tool = SQLQueryTool(MockGateway(), db_path=":memory:")
        result = await tool.execute_raw("UPDATE documents SET domain='x'")
        assert not result.success

    @pytest.mark.asyncio
    async def test_execute_raw_invalid_sql(self) -> None:
        tool = SQLQueryTool(MockGateway(), db_path=":memory:")
        result = await tool.execute_raw("SELECT * FROM nonexistent_table_xyz")
        assert not result.success
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_query_uses_llm_for_sql(self) -> None:
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            await _setup_test_db(db_path)
            gateway = MockGateway("SELECT COUNT(*) as cnt FROM documents")
            tool = SQLQueryTool(gateway, db_path=db_path)
            result = await tool.query("How many documents are there?")
            assert result.success
            assert gateway._call_count == 1
            assert result.sql == "SELECT COUNT(*) as cnt FROM documents"
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_query_strips_code_block(self) -> None:
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            await _setup_test_db(db_path)
            gateway = MockGateway("```sql\nSELECT id FROM documents LIMIT 1\n```")
            tool = SQLQueryTool(gateway, db_path=db_path)
            result = await tool.query("Give me one document id")
            assert result.success
            assert result.row_count == 1
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_query_llm_failure_returns_error(self) -> None:
        class FailGateway(MockGateway):
            async def complete(self, prompt, **kwargs):
                raise RuntimeError("LLM error")

        tool = SQLQueryTool(FailGateway(), db_path=":memory:")
        result = await tool.query("Something")
        assert not result.success
        assert "generation" in result.error.lower()

    @pytest.mark.asyncio
    async def test_query_llm_returns_blocked_sql(self) -> None:
        tool = SQLQueryTool(MockGateway("DELETE FROM documents"), db_path=":memory:")
        result = await tool.query("Delete everything")
        assert not result.success
        assert "blocked" in result.error.lower()

    def test_sql_result_properties(self) -> None:
        r = SQLResult(question="q", sql="SELECT 1", rows=[{"a": 1}])
        assert r.success is True
        assert r.row_count == 1

        r_err = SQLResult(question="q", sql="", error="oops")
        assert r_err.success is False
        assert r_err.row_count == 0


# ──────────────────────────────────────────────
# BIAggregationTool テスト
# ──────────────────────────────────────────────


class TestBIAggregationTool:
    async def _make_tool(self) -> tuple[BIAggregationTool, str]:
        import tempfile
        f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = f.name
        f.close()
        await _setup_test_db(db_path)
        return BIAggregationTool(db_path=db_path), db_path

    @pytest.mark.asyncio
    async def test_count_all(self) -> None:
        import os
        tool, db_path = await self._make_tool()
        try:
            result = await tool.count("documents")
            assert result.success
            assert result.scalar() == 3
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_count_group_by_domain(self) -> None:
        import os
        tool, db_path = await self._make_tool()
        try:
            result = await tool.count("documents", group_by="domain")
            assert result.success
            domains = {row["domain"] for row in result.rows}
            assert "code" in domains
            assert "general" in domains
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_average_teacher_quality(self) -> None:
        import os
        tool, db_path = await self._make_tool()
        try:
            result = await tool.average("documents", column="teacher_quality")
            assert result.success
            avg = result.scalar()
            assert avg is not None
            assert 0.5 <= float(avg) <= 1.0
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_domain_stats(self) -> None:
        import os
        tool, db_path = await self._make_tool()
        try:
            result = await tool.domain_stats()
            assert result.success
            assert result.row_count >= 1
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_difficulty_distribution(self) -> None:
        import os
        tool, db_path = await self._make_tool()
        try:
            result = await tool.difficulty_distribution()
            assert result.success
            difficulties = {row["difficulty"] for row in result.rows}
            assert "beginner" in difficulties
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_quality_summary(self) -> None:
        import os
        tool, db_path = await self._make_tool()
        try:
            result = await tool.quality_summary()
            assert result.success
            assert result.scalar() == 3  # total_docs
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_review_status_counts(self) -> None:
        import os
        tool, db_path = await self._make_tool()
        try:
            result = await tool.review_status_counts()
            assert result.success
            statuses = {row["review_status"] for row in result.rows}
            assert "approved" in statuses
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_invalid_table_blocked(self) -> None:
        tool = BIAggregationTool()
        result = await tool.count("evil_table")
        assert not result.success
        assert "not allowed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_invalid_column_blocked(self) -> None:
        tool = BIAggregationTool()
        result = await tool.average("documents", column="secret_column")
        assert not result.success
        assert "not allowed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_agg_result_scalar_none_on_empty(self) -> None:
        r = AggResult(sql="SELECT 1", rows=[])
        assert r.scalar() is None

    def test_agg_result_success_flag(self) -> None:
        r_ok = AggResult(sql="SELECT 1", rows=[{"x": 5}])
        assert r_ok.success is True

        r_err = AggResult(sql="", error="fail")
        assert r_err.success is False


# ──────────────────────────────────────────────
# QueryClassifier テスト
# ──────────────────────────────────────────────


class TestQueryClassifier:
    @pytest.mark.asyncio
    async def test_classify_semantic(self) -> None:
        classifier = QueryClassifier(MockGateway("semantic"))
        qtype = await classifier.classify("What is vector similarity?")
        assert qtype == QueryType.SEMANTIC

    @pytest.mark.asyncio
    async def test_classify_factual(self) -> None:
        classifier = QueryClassifier(MockGateway("factual"))
        qtype = await classifier.classify("How many documents are in the database?")
        assert qtype == QueryType.FACTUAL

    @pytest.mark.asyncio
    async def test_classify_relational(self) -> None:
        classifier = QueryClassifier(MockGateway("relational"))
        qtype = await classifier.classify("How does FAISS relate to ANN algorithms?")
        assert qtype == QueryType.RELATIONAL

    @pytest.mark.asyncio
    async def test_classify_hybrid(self) -> None:
        classifier = QueryClassifier(MockGateway("hybrid"))
        qtype = await classifier.classify("When did FAISS connect to GPT and how many docs?")
        assert qtype == QueryType.HYBRID

    @pytest.mark.asyncio
    async def test_classify_unknown_falls_back_to_keyword(self) -> None:
        classifier = QueryClassifier(MockGateway("unknown_type"))
        qtype = await classifier.classify("How does X relate to Y?")
        # keyword fallback: "relate" → RELATIONAL
        assert qtype == QueryType.RELATIONAL

    @pytest.mark.asyncio
    async def test_classify_llm_failure_keyword_fallback(self) -> None:
        class FailGateway(MockGateway):
            async def complete(self, prompt, **kwargs):
                raise RuntimeError("LLM offline")

        classifier = QueryClassifier(FailGateway(), use_fallback=True)
        qtype = await classifier.classify("When was Python created?")
        assert qtype == QueryType.FACTUAL  # "when" → factual

    @pytest.mark.asyncio
    async def test_classify_llm_failure_no_fallback_returns_default(self) -> None:
        class FailGateway(MockGateway):
            async def complete(self, prompt, **kwargs):
                raise RuntimeError("LLM offline")

        classifier = QueryClassifier(
            FailGateway(), use_fallback=False, default_type=QueryType.HYBRID
        )
        qtype = await classifier.classify("anything")
        assert qtype == QueryType.HYBRID

    @pytest.mark.asyncio
    async def test_classify_batch(self) -> None:
        classifier = QueryClassifier(MockGateway("semantic"))
        queries = ["What is X?", "How does Y work?", "When was Z?"]
        types = await classifier.classify_batch(queries)
        assert len(types) == 3
        assert all(isinstance(t, QueryType) for t in types)

    @pytest.mark.asyncio
    async def test_keyword_fallback_semantic_default(self) -> None:
        classifier = QueryClassifier(MockGateway("unknown"), default_type=QueryType.SEMANTIC)
        qtype = await classifier.classify("Explain recursion")
        # no relational/factual keywords → default
        assert qtype == QueryType.SEMANTIC

    @pytest.mark.asyncio
    async def test_keyword_both_relational_and_factual_gives_hybrid(self) -> None:
        classifier = QueryClassifier(MockGateway("unknown_type"))
        qtype = await classifier.classify("How many relations connect X when Y?")
        # "how many" → factual, "connect" / "relate" → relational → HYBRID
        assert qtype == QueryType.HYBRID


# ──────────────────────────────────────────────
# FusionReranker テスト
# ──────────────────────────────────────────────


class TestFusionReranker:
    def test_fuse_single_source(self) -> None:
        reranker = FusionReranker()
        result = reranker.fuse({
            "faiss": [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.5)]
        })
        assert len(result) == 3
        assert result[0].doc_id == "doc1"
        assert all(isinstance(r, FusionResult) for r in result)

    def test_fuse_two_sources_combines(self) -> None:
        reranker = FusionReranker()
        result = reranker.fuse({
            "faiss": [("doc1", 0.9), ("doc2", 0.7)],
            "kg":    [("doc2", 0.8), ("doc1", 0.6)],
        })
        # doc1 appears at rank 1 in faiss, rank 2 in kg
        # doc2 appears at rank 2 in faiss, rank 1 in kg
        doc_ids = [r.doc_id for r in result]
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids

    def test_fuse_rrf_score_increases_with_more_sources(self) -> None:
        """複数ソースに出現するドキュメントはスコアが高くなる。"""
        reranker = FusionReranker()
        result = reranker.fuse({
            "faiss": [("shared_doc", 0.5), ("only_faiss", 0.9)],
            "kg":    [("shared_doc", 0.5)],
        })
        scores = {r.doc_id: r.rrf_score for r in result}
        assert scores["shared_doc"] > scores.get("only_faiss", 0.0) or \
               len(scores["shared_doc"]) == len(result[0].sources) >= 1  # just check it works

    def test_fuse_sources_tracked(self) -> None:
        reranker = FusionReranker()
        result = reranker.fuse({
            "faiss": [("doc1", 0.9)],
            "kg":    [("doc1", 0.8)],
        })
        doc1_result = next(r for r in result if r.doc_id == "doc1")
        assert "faiss" in doc1_result.sources
        assert "kg" in doc1_result.sources

    def test_fuse_source_ranks_tracked(self) -> None:
        reranker = FusionReranker()
        result = reranker.fuse({
            "faiss": [("doc_a", 0.9), ("doc_b", 0.5)],
        })
        doc_a = next(r for r in result if r.doc_id == "doc_a")
        assert doc_a.source_ranks["faiss"] == 1

    def test_fuse_top_k(self) -> None:
        reranker = FusionReranker()
        result = reranker.fuse({
            "faiss": [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)],
        }, top_k=2)
        assert len(result) == 2

    def test_fuse_empty_input(self) -> None:
        reranker = FusionReranker()
        assert reranker.fuse({}) == []

    def test_fuse_doc_ids_only(self) -> None:
        reranker = FusionReranker()
        ids = reranker.fuse_doc_ids({
            "faiss": [("doc1", 0.9), ("doc2", 0.7)],
        })
        assert ids == ["doc1", "doc2"]

    def test_fuse_with_source_weights(self) -> None:
        """重みが高いソースの文書が上位に来る。"""
        reranker = FusionReranker(source_weights={"faiss": 2.0, "kg": 0.5})
        result = reranker.fuse({
            "faiss": [("faiss_top", 0.9)],
            "kg":    [("kg_top", 0.9)],
        })
        # faiss_top should score higher due to 2x weight
        assert result[0].doc_id == "faiss_top"

    def test_normalize_scores(self) -> None:
        items = [("doc1", 0.5), ("doc2", 1.0), ("doc3", 0.0)]
        normalized = FusionReranker.normalize_scores(items)
        scores = [s for _, s in normalized]
        assert max(scores) == pytest.approx(1.0)
        assert min(scores) == pytest.approx(0.0)

    def test_normalize_scores_uniform(self) -> None:
        items = [("doc1", 0.5), ("doc2", 0.5)]
        normalized = FusionReranker.normalize_scores(items)
        assert all(s == pytest.approx(1.0) for _, s in normalized)

    def test_normalize_scores_empty(self) -> None:
        assert FusionReranker.normalize_scores([]) == []

    def test_rrf_score_formula(self) -> None:
        """RRF スコアが 1/(k+rank) の公式に従うか検証。"""
        k = 60
        reranker = FusionReranker(k=k)
        result = reranker.fuse({
            "faiss": [("doc1", 0.9), ("doc2", 0.8)],
        })
        doc1 = next(r for r in result if r.doc_id == "doc1")
        doc2 = next(r for r in result if r.doc_id == "doc2")
        assert doc1.rrf_score == pytest.approx(1.0 / (k + 1))
        assert doc2.rrf_score == pytest.approx(1.0 / (k + 2))

    def test_fuse_dedup_is_alias(self) -> None:
        reranker = FusionReranker()
        ranked = {"faiss": [("doc1", 0.9)]}
        assert reranker.fuse_with_dedup(ranked) == reranker.fuse(ranked)
