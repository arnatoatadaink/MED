"""tests/unit/test_phase4.py — Phase 4 モジュールの単体テスト

テスト対象:
- src/orchestrator/query_parser.py
- src/orchestrator/model_router.py
- src/training/evaluation/student_evaluator.py
- src/training/evaluation/teacher_comparison.py
- src/training/evaluation/benchmark_suite.py
- src/llm/usage_tracker.py
- src/llm/prompt_cache.py
- src/llm/error_analyzer.py
- src/llm/feedback_analyzer.py
- src/memory/deduplicator.py
- src/memory/maturation/quality_metrics.py
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

@dataclass
class _MockResponse:
    content: str


class _MockGateway:
    """LLMGateway スタブ。"""

    def __init__(self, response_content: str = '{"result": "ok"}'):
        self._content = response_content
        self.calls: list[dict] = []

    async def complete(self, prompt: str, **kwargs) -> _MockResponse:
        self.calls.append({"prompt": prompt, **kwargs})
        return _MockResponse(content=self._content)


# ===========================================================================
# QueryParser
# ===========================================================================

class TestParsedQuery:
    def _make(self, **kwargs):
        from src.orchestrator.query_parser import ParsedQuery
        defaults = dict(raw="test", intent="qa", domain="general",
                        complexity="moderate", entities=[], requires_execution=False,
                        requires_retrieval=True)
        defaults.update(kwargs)
        return ParsedQuery(**defaults)

    def test_is_simple(self):
        pq = self._make(complexity="simple")
        assert pq.is_simple
        assert not pq.is_complex

    def test_is_complex(self):
        pq = self._make(complexity="complex")
        assert pq.is_complex
        assert not pq.is_simple

    def test_is_code(self):
        pq = self._make(intent="code")
        assert pq.is_code

    def test_entities_default_empty(self):
        from src.orchestrator.query_parser import ParsedQuery
        pq = ParsedQuery(raw="hello")
        assert pq.entities == []


class TestQueryParser:
    def _parser(self, content: str = ""):
        from src.orchestrator.query_parser import QueryParser
        gw = _MockGateway(content)
        return QueryParser(gw)

    def test_keyword_parse_code(self):
        from src.orchestrator.query_parser import QueryParser
        parser = QueryParser.__new__(QueryParser)
        parser._use_fallback = True
        result = parser._keyword_parse("implement a Python function to sort a list")
        assert result.intent == "code"
        assert result.requires_execution

    def test_keyword_parse_compute(self):
        from src.orchestrator.query_parser import QueryParser
        parser = QueryParser.__new__(QueryParser)
        parser._use_fallback = True
        result = parser._keyword_parse("calculate the sum of 1 to 100")
        assert result.intent == "compute"

    def test_keyword_parse_explain(self):
        from src.orchestrator.query_parser import QueryParser
        parser = QueryParser.__new__(QueryParser)
        parser._use_fallback = True
        result = parser._keyword_parse("explain how FAISS works")
        assert result.intent == "explain"

    def test_keyword_parse_simple(self):
        from src.orchestrator.query_parser import QueryParser
        parser = QueryParser.__new__(QueryParser)
        parser._use_fallback = True
        result = parser._keyword_parse("hello")  # ≤ 8 words
        assert result.complexity == "simple"

    def test_keyword_parse_complex(self):
        from src.orchestrator.query_parser import QueryParser
        parser = QueryParser.__new__(QueryParser)
        parser._use_fallback = True
        result = parser._keyword_parse("research and analyze multiple approaches")
        assert result.complexity == "complex"

    def test_parse_json_success(self):
        content = '{"intent":"code","domain":"code","complexity":"simple","entities":[],"requires_execution":true,"requires_retrieval":false}'
        parser = self._parser(content)
        result = asyncio.get_event_loop().run_until_complete(parser.parse("write code"))
        assert result.intent == "code"
        assert result.requires_execution is True

    def test_parse_llm_failure_uses_fallback(self):
        from src.orchestrator.query_parser import QueryParser
        gw = _MockGateway("INVALID JSON {{")
        parser = QueryParser(gw, use_fallback=True)
        result = asyncio.get_event_loop().run_until_complete(
            parser.parse("implement binary search")
        )
        assert result.raw == "implement binary search"
        assert result.intent in {"qa", "code", "explain", "compute", "search"}


# ===========================================================================
# ModelRouter
# ===========================================================================

class TestRoutingDecision:
    def _make(self, target: str = "student", **kwargs):
        from src.orchestrator.model_router import RoutingDecision
        return RoutingDecision(target=target, **kwargs)

    def test_use_student(self):
        d = self._make("student")
        assert d.use_student
        assert not d.use_teacher

    def test_use_teacher(self):
        d = self._make("teacher")
        assert d.use_teacher
        assert not d.use_student


class TestModelRouter:
    def _router(self, threshold="moderate", always_code=False):
        from src.orchestrator.model_router import ModelRouter
        gw = _MockGateway()
        return ModelRouter(gw, student_complexity_threshold=threshold,
                           always_use_teacher_for_code=always_code)

    def _parsed(self, complexity="moderate", intent="qa", entities=None,
                requires_execution=False, requires_retrieval=True):
        from src.orchestrator.query_parser import ParsedQuery
        return ParsedQuery(
            raw="test query",
            intent=intent,
            domain="general",
            complexity=complexity,
            entities=entities or [],
            requires_execution=requires_execution,
            requires_retrieval=requires_retrieval,
        )

    def test_invalid_threshold(self):
        from src.orchestrator.model_router import ModelRouter
        with pytest.raises(ValueError):
            ModelRouter(_MockGateway(), student_complexity_threshold="invalid")

    def test_simple_goes_to_student(self):
        router = self._router()
        parsed = self._parsed(complexity="simple")
        target, reason = router._select_target(parsed)
        assert target == "student"

    def test_complex_goes_to_teacher(self):
        router = self._router()
        parsed = self._parsed(complexity="complex")
        target, reason = router._select_target(parsed)
        assert target == "teacher"

    def test_moderate_with_moderate_threshold_goes_student(self):
        router = self._router(threshold="moderate")
        parsed = self._parsed(complexity="moderate")
        target, _ = router._select_target(parsed)
        assert target == "student"

    def test_moderate_with_simple_threshold_goes_teacher(self):
        router = self._router(threshold="simple")
        parsed = self._parsed(complexity="moderate")
        target, _ = router._select_target(parsed)
        assert target == "teacher"

    def test_code_forced_teacher(self):
        router = self._router(always_code=True)
        parsed = self._parsed(intent="code")
        target, _ = router._select_target(parsed)
        assert target == "teacher"

    def test_route_no_kg(self):
        router = self._router()
        parsed = self._parsed(complexity="simple")
        decision = asyncio.get_event_loop().run_until_complete(router.route(parsed))
        assert decision.target == "student"
        assert decision.use_faiss is True
        assert decision.use_kg is False

    def test_route_external_rag_for_moderate(self):
        router = self._router()
        parsed = self._parsed(complexity="moderate")
        decision = asyncio.get_event_loop().run_until_complete(router.route(parsed))
        assert decision.use_external_rag is True

    def test_route_sandbox_for_execution(self):
        router = self._router()
        parsed = self._parsed(requires_execution=True)
        decision = asyncio.get_event_loop().run_until_complete(router.route(parsed))
        assert decision.use_sandbox is True


# ===========================================================================
# UsageTracker
# ===========================================================================

class TestUsageTracker:
    def _tracker(self):
        from src.llm.usage_tracker import UsageTracker
        return UsageTracker()

    def test_record_returns_usage_record(self):
        tracker = self._tracker()
        rec = tracker.record("anthropic", "claude-sonnet-4-6",
                             input_tokens=100, output_tokens=50, latency_ms=500)
        assert rec.provider == "anthropic"
        assert rec.model == "claude-sonnet-4-6"
        assert rec.input_tokens == 100
        assert rec.output_tokens == 50

    def test_cost_calculation_known_model(self):
        tracker = self._tracker()
        # claude-sonnet-4-6: input=$3/1M, output=$15/1M
        rec = tracker.record("anthropic", "claude-sonnet-4-6",
                             input_tokens=1_000_000, output_tokens=1_000_000)
        assert abs(rec.cost_usd - 18.0) < 0.01  # 3 + 15

    def test_cost_calculation_unknown_model(self):
        tracker = self._tracker()
        rec = tracker.record("custom", "unknown-model",
                             input_tokens=1_000_000, output_tokens=1_000_000)
        assert rec.cost_usd > 0  # uses default cost

    def test_report_empty(self):
        from src.llm.usage_tracker import UsageSummary
        tracker = self._tracker()
        summary = tracker.report()
        assert isinstance(summary, UsageSummary)
        assert summary.total_input_tokens == 0
        assert summary.total_cost_usd == 0.0

    def test_report_aggregates(self):
        tracker = self._tracker()
        tracker.record("anthropic", "claude-haiku-4-5", input_tokens=100, output_tokens=50)
        tracker.record("anthropic", "claude-haiku-4-5", input_tokens=200, output_tokens=100)
        summary = tracker.report()
        assert summary.total_input_tokens == 300
        assert summary.total_output_tokens == 150
        assert "anthropic" in summary.by_provider

    def test_report_by_model(self):
        tracker = self._tracker()
        tracker.record("openai", "gpt-4o-mini", input_tokens=50, output_tokens=25)
        summary = tracker.report()
        assert "gpt-4o-mini" in summary.by_model

    def test_reset(self):
        tracker = self._tracker()
        tracker.record("anthropic", "claude-haiku-4-5", input_tokens=100, output_tokens=50)
        tracker.reset()
        assert tracker.report().total_input_tokens == 0

    def test_total_calls(self):
        tracker = self._tracker()
        tracker.record("a", "m", input_tokens=10, output_tokens=5)
        tracker.record("a", "m", input_tokens=10, output_tokens=5)
        assert tracker.report().total_calls == 2


# ===========================================================================
# PromptCache
# ===========================================================================

class TestPromptCache:
    def _cache(self, max_size=100, ttl=3600):
        from src.llm.prompt_cache import PromptCache
        return PromptCache(max_size=max_size, ttl_seconds=ttl)

    def test_make_key_deterministic(self):
        cache = self._cache()
        k1 = cache.make_key("hello", system="sys", model="gpt")
        k2 = cache.make_key("hello", system="sys", model="gpt")
        assert k1 == k2

    def test_make_key_differs_on_content(self):
        cache = self._cache()
        k1 = cache.make_key("hello", model="gpt")
        k2 = cache.make_key("world", model="gpt")
        assert k1 != k2

    def test_get_miss(self):
        cache = self._cache()
        assert cache.get("nonexistent") is None

    def test_set_and_get(self):
        cache = self._cache()
        key = cache.make_key("prompt")
        cache.set(key, "response")
        assert cache.get(key) == "response"

    def test_hit_rate(self):
        cache = self._cache()
        key = cache.make_key("p")
        cache.get(key)          # miss
        cache.set(key, "v")
        cache.get(key)          # hit
        cache.get(key)          # hit
        assert cache.hit_rate == pytest.approx(2/3)

    def test_lru_eviction(self):
        cache = self._cache(max_size=2)
        k1 = cache.make_key("p1")
        k2 = cache.make_key("p2")
        k3 = cache.make_key("p3")
        cache.set(k1, "v1")
        cache.set(k2, "v2")
        cache.set(k3, "v3")    # evicts k1 (LRU)
        assert cache.get(k1) is None
        assert cache.get(k2) == "v2"
        assert cache.get(k3) == "v3"

    def test_ttl_expiry(self):
        cache = self._cache(ttl=1)
        key = cache.make_key("prompt")
        cache.set(key, "value")
        assert cache.get(key) == "value"

        # Simulate expired by patching entry's created_at
        entry = cache._cache[key]
        entry.created_at = time.time() - 10  # 10 seconds ago
        assert cache.get(key) is None

    def test_evict_expired(self):
        cache = self._cache(ttl=1)
        key = cache.make_key("prompt")
        cache.set(key, "value")
        entry = cache._cache[key]
        entry.created_at = time.time() - 10
        cache.evict_expired()
        assert len(cache._cache) == 0

    def test_size(self):
        cache = self._cache()
        assert cache.size == 0
        cache.set(cache.make_key("p1"), "v")
        assert cache.size == 1

    def test_clear(self):
        cache = self._cache()
        cache.set(cache.make_key("p"), "v")
        cache.clear()
        assert cache.size == 0


# ===========================================================================
# ErrorAnalyzer
# ===========================================================================

class TestErrorAnalyzer:
    def _analyzer(self, content='{"error_type":"NameError","error_line":1,"suggestion":"fix it","fixed_code":"x = 1"}'):
        from src.llm.error_analyzer import ErrorAnalyzer
        gw = _MockGateway(content)
        return ErrorAnalyzer(gw)

    def test_keyword_detect_name_error(self):
        from src.llm.error_analyzer import ErrorAnalyzer
        a = ErrorAnalyzer.__new__(ErrorAnalyzer)
        error_type = a._detect_error_type("NameError: name 'x' is not defined")
        assert error_type == "NameError"

    def test_keyword_detect_type_error(self):
        from src.llm.error_analyzer import ErrorAnalyzer
        a = ErrorAnalyzer.__new__(ErrorAnalyzer)
        error_type = a._detect_error_type("TypeError: unsupported operand type")
        assert error_type == "TypeError"

    def test_keyword_detect_syntax_error(self):
        from src.llm.error_analyzer import ErrorAnalyzer
        a = ErrorAnalyzer.__new__(ErrorAnalyzer)
        error_type = a._detect_error_type("SyntaxError: invalid syntax")
        assert error_type == "SyntaxError"

    def test_keyword_detect_unknown(self):
        from src.llm.error_analyzer import ErrorAnalyzer
        a = ErrorAnalyzer.__new__(ErrorAnalyzer)
        error_type = a._detect_error_type("something weird happened")
        assert error_type == "UnknownError"

    def test_analyze_llm_success(self):
        analyzer = self._analyzer()
        result = asyncio.get_event_loop().run_until_complete(
            analyzer.analyze("x = undefined_var", "NameError: name 'undefined_var' is not defined")
        )
        assert result.error_type == "NameError"
        assert result.suggestion

    def test_analyze_returns_error_analysis(self):
        from src.llm.error_analyzer import ErrorAnalysis
        analyzer = self._analyzer()
        result = asyncio.get_event_loop().run_until_complete(
            analyzer.analyze("code", "error")
        )
        assert isinstance(result, ErrorAnalysis)

    def test_has_fix_property(self):
        from src.llm.error_analyzer import ErrorAnalysis
        ea = ErrorAnalysis(
            error_type="NameError", error_line=1,
            root_cause="undefined var", suggestion="fix", fixed_code="x = 1"
        )
        assert ea.has_fix is True

    def test_has_fix_empty(self):
        from src.llm.error_analyzer import ErrorAnalysis
        ea = ErrorAnalysis(error_type="NameError", error_line=None, root_cause="x", suggestion="fix")
        assert ea.has_fix is False

    def test_analyze_bad_json_uses_fallback(self):
        from src.llm.error_analyzer import ErrorAnalyzer
        gw = _MockGateway("NOT JSON")
        analyzer = ErrorAnalyzer(gw)
        result = asyncio.get_event_loop().run_until_complete(
            analyzer.analyze("code", "TypeError: x")
        )
        assert result.error_type == "TypeError"


# ===========================================================================
# FeedbackAnalyzer
# ===========================================================================

class TestFeedbackSignal:
    def _signal(self, sentiment="neutral", reward=0.5):
        from src.llm.feedback_analyzer import FeedbackSignal
        return FeedbackSignal(raw_feedback="test", sentiment=sentiment, reward=reward)

    def test_is_positive(self):
        s = self._signal(sentiment="positive")
        assert s.is_positive
        assert not s.is_negative

    def test_is_negative(self):
        s = self._signal(sentiment="negative")
        assert s.is_negative
        assert not s.is_positive

    def test_memory_delta_positive(self):
        s = self._signal(sentiment="positive", reward=0.9)
        assert abs(s.memory_delta - 0.4) < 1e-9

    def test_memory_delta_negative(self):
        s = self._signal(sentiment="negative", reward=0.2)
        assert abs(s.memory_delta - (-0.3)) < 1e-9

    def test_memory_delta_neutral(self):
        s = self._signal(sentiment="neutral", reward=0.5)
        assert abs(s.memory_delta) < 1e-9

    def test_aspects_default_empty(self):
        from src.llm.feedback_analyzer import FeedbackSignal
        s = FeedbackSignal(raw_feedback="x")
        assert s.aspects == []


class TestFeedbackAnalyzer:
    def _analyzer(self, content='{"sentiment":"positive","reward":0.85,"aspects":["helpful"],"update_memory":true,"summary":"great"}'):
        from src.llm.feedback_analyzer import FeedbackAnalyzer
        gw = _MockGateway(content)
        return FeedbackAnalyzer(gw)

    def test_keyword_positive(self):
        from src.llm.feedback_analyzer import FeedbackAnalyzer
        a = FeedbackAnalyzer(use_llm=False)
        result = asyncio.get_event_loop().run_until_complete(a.analyze("great answer, very helpful!"))
        assert result.sentiment == "positive"
        assert result.reward >= 0.7

    def test_keyword_negative(self):
        from src.llm.feedback_analyzer import FeedbackAnalyzer
        a = FeedbackAnalyzer(use_llm=False)
        result = asyncio.get_event_loop().run_until_complete(a.analyze("wrong and incorrect"))
        assert result.sentiment == "negative"
        assert result.reward <= 0.3

    def test_keyword_mixed_neutral(self):
        from src.llm.feedback_analyzer import FeedbackAnalyzer
        a = FeedbackAnalyzer(use_llm=False)
        result = asyncio.get_event_loop().run_until_complete(a.analyze("good but wrong"))
        assert result.sentiment == "neutral"

    def test_keyword_no_match_neutral(self):
        from src.llm.feedback_analyzer import FeedbackAnalyzer
        a = FeedbackAnalyzer(use_llm=False)
        result = asyncio.get_event_loop().run_until_complete(a.analyze("ok"))
        assert result.sentiment == "neutral"

    def test_empty_feedback(self):
        from src.llm.feedback_analyzer import FeedbackAnalyzer
        a = FeedbackAnalyzer(use_llm=False)
        result = asyncio.get_event_loop().run_until_complete(a.analyze("   "))
        assert result.sentiment == "neutral"
        assert result.reward == 0.5

    def test_llm_analyze_success(self):
        analyzer = self._analyzer()
        result = asyncio.get_event_loop().run_until_complete(analyzer.analyze("This was very helpful!"))
        assert result.sentiment == "positive"
        assert result.reward == pytest.approx(0.85)

    def test_analyze_batch(self):
        from src.llm.feedback_analyzer import FeedbackAnalyzer
        a = FeedbackAnalyzer(use_llm=False)
        results = asyncio.get_event_loop().run_until_complete(
            a.analyze_batch(["great!", "wrong!", "ok"])
        )
        assert len(results) == 3


# ===========================================================================
# Deduplicator
# ===========================================================================

class TestDeduplicator:
    def _dedup(self, threshold=0.95):
        from src.memory.deduplicator import Deduplicator
        return Deduplicator(near_dup_threshold=threshold)

    def test_content_hash_deterministic(self):
        from src.memory.deduplicator import Deduplicator
        h1 = Deduplicator.content_hash("hello world")
        h2 = Deduplicator.content_hash("hello world")
        assert h1 == h2

    def test_content_hash_differs(self):
        from src.memory.deduplicator import Deduplicator
        h1 = Deduplicator.content_hash("hello")
        h2 = Deduplicator.content_hash("world")
        assert h1 != h2

    def test_content_hash_is_sha256(self):
        from src.memory.deduplicator import Deduplicator
        h = Deduplicator.content_hash("test")
        assert len(h) == 64

    def test_check_exact_dup(self):
        dedup = self._dedup()
        h = "abc123"
        result = dedup.check(h, existing_hashes={"doc1": h})
        assert result.is_duplicate
        assert result.is_exact
        assert result.duplicate_doc_id == "doc1"
        assert result.similarity == 1.0

    def test_check_no_dup(self):
        dedup = self._dedup()
        result = dedup.check("newhash", existing_hashes={"doc1": "oldhash"})
        assert not result.is_duplicate

    def test_check_near_dup(self):
        dedup = self._dedup(threshold=0.9)
        q = np.ones(4, dtype=np.float32)
        v = np.ones(4, dtype=np.float32)  # cosine = 1.0
        result = dedup.check("newhash", query_vector=q,
                              existing_vectors=[("doc1", v)])
        assert result.is_duplicate
        assert result.is_near
        assert result.similarity == pytest.approx(1.0)

    def test_check_near_dup_below_threshold(self):
        dedup = self._dedup(threshold=0.99)
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # cosine = 0.0
        result = dedup.check("newhash", query_vector=q,
                              existing_vectors=[("doc1", v)])
        assert not result.is_duplicate

    def test_check_near_dup_disabled(self):
        from src.memory.deduplicator import Deduplicator
        dedup = Deduplicator(near_dup_threshold=0.0, check_near_dup=False)
        q = np.ones(4, dtype=np.float32)
        v = np.ones(4, dtype=np.float32)
        result = dedup.check("newhash", query_vector=q,
                              existing_vectors=[("doc1", v)])
        assert not result.is_duplicate

    def test_check_zero_vector(self):
        dedup = self._dedup()
        q = np.zeros(4, dtype=np.float32)
        v = np.ones(4, dtype=np.float32)
        result = dedup.check("newhash", query_vector=q,
                              existing_vectors=[("doc1", v)])
        assert not result.is_duplicate

    def test_filter_batch_unique(self):
        dedup = self._dedup()
        uniq, dups = dedup.filter_batch([("h1", "id1"), ("h2", "id2")])
        assert uniq == ["id1", "id2"]
        assert dups == []

    def test_filter_batch_existing_dup(self):
        dedup = self._dedup()
        uniq, dups = dedup.filter_batch(
            [("h1", "new1")],
            existing_hashes={"old": "h1"},
        )
        assert "new1" in dups
        assert "new1" not in uniq

    def test_filter_batch_intra_dup(self):
        dedup = self._dedup()
        uniq, dups = dedup.filter_batch([("same", "id1"), ("same", "id2")])
        assert "id1" in uniq
        assert "id2" in dups

    def test_dedup_result_properties(self):
        from src.memory.deduplicator import DedupResult
        r = DedupResult(is_duplicate=True, duplicate_type="exact")
        assert r.is_exact
        assert not r.is_near

        r2 = DedupResult(is_duplicate=True, duplicate_type="near", similarity=0.97)
        assert r2.is_near
        assert not r2.is_exact


# ===========================================================================
# QualityMetrics
# ===========================================================================

class TestQualityReport:
    def _report(self, **kwargs):
        from src.memory.maturation.quality_metrics import QualityReport
        defaults = dict(
            total_docs=0,
            approved_docs=0,
            rejected_docs=0,
            pending_docs=0,
            avg_confidence=0.0,
            exec_success_rate=0.0,
        )
        defaults.update(kwargs)
        return QualityReport(**defaults)

    def test_approval_rate_empty(self):
        r = self._report()
        assert r.approval_rate == 0.0

    def test_approval_rate(self):
        r = self._report(total_docs=10, approved_docs=8)
        assert r.approval_rate == pytest.approx(0.8)

    def test_meets_doc_target(self):
        from src.memory.maturation.quality_metrics import QualityReport
        r = QualityReport(total_docs=10_000)
        assert r.meets_doc_target

    def test_not_meets_doc_target(self):
        from src.memory.maturation.quality_metrics import QualityReport
        r = QualityReport(total_docs=100)
        assert not r.meets_doc_target

    def test_meets_confidence_target(self):
        from src.memory.maturation.quality_metrics import QualityReport
        r = QualityReport(avg_confidence=0.75)
        assert r.meets_confidence_target

    def test_meets_exec_success_target(self):
        from src.memory.maturation.quality_metrics import QualityReport
        r = QualityReport(exec_success_rate=0.85)
        assert r.meets_exec_success_target

    def test_meets_phase2_goal_all(self):
        from src.memory.maturation.quality_metrics import QualityReport
        r = QualityReport(
            total_docs=10_000,
            avg_confidence=0.75,
            exec_success_rate=0.85,
        )
        assert r.meets_phase2_goal

    def test_not_meets_phase2_goal_missing_docs(self):
        from src.memory.maturation.quality_metrics import QualityReport
        r = QualityReport(total_docs=100, avg_confidence=0.8, exec_success_rate=0.9)
        assert not r.meets_phase2_goal

    def test_phase2_progress(self):
        from src.memory.maturation.quality_metrics import QualityReport
        r = QualityReport(total_docs=5_000, avg_confidence=0.35, exec_success_rate=0.4)
        p = r.phase2_progress
        assert p["docs"] == pytest.approx(0.5)
        assert p["confidence"] == pytest.approx(0.5)
        assert p["exec_success"] == pytest.approx(0.5)

    def test_phase2_progress_capped_at_1(self):
        from src.memory.maturation.quality_metrics import QualityReport
        r = QualityReport(total_docs=100_000, avg_confidence=0.99, exec_success_rate=0.99)
        p = r.phase2_progress
        assert p["docs"] == 1.0
        assert p["confidence"] == 1.0
        assert p["exec_success"] == 1.0

    def test_summary_string(self):
        from src.memory.maturation.quality_metrics import QualityReport
        r = QualityReport(total_docs=100, avg_confidence=0.5, exec_success_rate=0.6)
        s = r.summary()
        assert "Memory Quality Report" in s
        assert "100" in s

    def test_from_doc_list_empty(self):
        from src.memory.maturation.quality_metrics import QualityMetrics
        r = QualityMetrics.from_doc_list([])
        assert r.total_docs == 0

    def test_from_doc_list(self):
        from src.memory.maturation.quality_metrics import QualityMetrics

        class FakeDoc:
            review_status = "approved"
            confidence = 0.8
            teacher_quality = 0.9
            composite_score = 0.7
            execution_success_rate = 1.0
            retrieval_count = 5
            selection_count = 3
            difficulty = "intermediate"

        docs = [FakeDoc(), FakeDoc()]
        r = QualityMetrics.from_doc_list(docs)
        assert r.total_docs == 2
        assert r.approved_docs == 2
        assert r.avg_confidence == pytest.approx(0.8)
        assert r.exec_success_rate == pytest.approx(1.0)
        assert r.difficulty_distribution == {"intermediate": 2}


class TestQualityMetricsCompute:
    def test_compute_calls_store(self):
        from src.memory.maturation.quality_metrics import QualityMetrics

        class MockStore:
            async def get_stats(self, domain=None):
                return {
                    "total_docs": 500,
                    "approved_docs": 400,
                    "rejected_docs": 50,
                    "pending_docs": 50,
                    "avg_confidence": 0.72,
                    "avg_teacher_quality": 0.75,
                    "avg_composite_score": 0.68,
                    "exec_success_rate": 0.82,
                    "avg_retrieval_count": 3.5,
                    "avg_selection_count": 1.2,
                    "difficulty_distribution": {"easy": 100, "medium": 300, "hard": 100},
                }

        metrics = QualityMetrics(MockStore())
        report = asyncio.get_event_loop().run_until_complete(metrics.compute())
        assert report.total_docs == 500
        assert report.avg_confidence == pytest.approx(0.72)
        assert report.meets_confidence_target

    def test_compute_store_failure(self):
        from src.memory.maturation.quality_metrics import QualityMetrics

        class FailStore:
            async def get_stats(self, domain=None):
                raise RuntimeError("DB error")

        metrics = QualityMetrics(FailStore())
        report = asyncio.get_event_loop().run_until_complete(metrics.compute())
        assert report.total_docs == 0  # empty report on failure

    def test_check_phase2_readiness_not_met(self):
        from src.memory.maturation.quality_metrics import QualityMetrics

        class MockStore:
            async def get_stats(self, domain=None):
                return {"total_docs": 100, "avg_confidence": 0.5, "exec_success_rate": 0.6}

        metrics = QualityMetrics(MockStore())
        ready, missing = asyncio.get_event_loop().run_until_complete(
            metrics.check_phase2_readiness()
        )
        assert not ready
        assert len(missing) >= 1


# ===========================================================================
# StudentEvaluator (stub)
# ===========================================================================

class TestEvalMetrics:
    def test_to_dict(self):
        from src.training.evaluation.student_evaluator import EvalMetrics
        m = EvalMetrics(n_samples=5, avg_reward=0.75, answer_quality=0.8)
        d = m.to_dict()
        assert d["n_samples"] == 5
        assert d["avg_reward"] == pytest.approx(0.75)

    def test_per_sample_default_empty(self):
        from src.training.evaluation.student_evaluator import EvalMetrics
        m = EvalMetrics()
        assert m.per_sample == []


class TestStudentEvaluator:
    def _evaluator(self):
        from src.training.evaluation.student_evaluator import StudentEvaluator
        gw = _MockGateway('{"quality": 0.8, "score": 0.8}')
        return StudentEvaluator(gw)

    def test_evaluate_empty(self):
        evaluator = self._evaluator()
        metrics = asyncio.get_event_loop().run_until_complete(evaluator.evaluate([]))
        assert metrics.n_samples == 0

    def test_evaluate_samples(self):
        from src.training.evaluation.student_evaluator import EvalSample
        evaluator = self._evaluator()
        samples = [EvalSample(query="What is FAISS?", expected_answer="A vector library")]
        metrics = asyncio.get_event_loop().run_until_complete(evaluator.evaluate(samples))
        assert metrics.n_samples == 1
        assert 0.0 <= metrics.avg_reward <= 1.0


# ===========================================================================
# TeacherComparison (stub)
# ===========================================================================

class TestTeacherComparison:
    def _comparison(self):
        from src.training.evaluation.teacher_comparison import TeacherComparison
        content = '{"student_score": 0.7, "teacher_score": 0.9, "winner": "teacher", "quality_gap": 0.2, "feedback": "good"}'
        gw = _MockGateway(content)
        return TeacherComparison(gw)

    def test_compare_returns_result(self):
        from src.training.evaluation.teacher_comparison import ComparisonResult
        comp = self._comparison()
        result = asyncio.get_event_loop().run_until_complete(
            comp.compare("What is FAISS?", "A library", "FAISS is a vector search library by Meta")
        )
        assert isinstance(result, ComparisonResult)

    def test_relative_performance(self):
        from src.training.evaluation.teacher_comparison import ComparisonResult
        r = ComparisonResult(
            query="test", student_score=0.7, teacher_score=0.9,
            winner="teacher", quality_gap=0.2, feedback="ok"
        )
        assert r.relative_performance == pytest.approx(0.7 / 0.9)

    def test_relative_performance_zero_teacher(self):
        from src.training.evaluation.teacher_comparison import ComparisonResult
        r = ComparisonResult(
            query="test", student_score=0.5, teacher_score=0.0,
            winner="student", quality_gap=0.0, feedback=""
        )
        # teacher_score=0 → returns 0 (no division; see implementation)
        assert r.relative_performance == 0.0 or r.relative_performance == 1.0


# ===========================================================================
# BenchmarkSuite
# ===========================================================================

class TestBenchmarkSuite:
    def _suite(self):
        from src.training.evaluation.benchmark_suite import BenchmarkSuite
        gw = _MockGateway('{"quality": 0.75, "score": 0.75}')
        return BenchmarkSuite(gw)

    def test_list_benchmarks(self):
        suite = self._suite()
        names = suite.list_benchmarks()
        assert "code_generation" in names
        assert "qa_retrieval" in names
        assert "math_reasoning" in names

    def test_run_single_benchmark(self):
        from src.training.evaluation.benchmark_suite import BenchmarkReport
        suite = self._suite()
        report = asyncio.get_event_loop().run_until_complete(
            suite.run(benchmark_names=["qa_retrieval"])
        )
        assert isinstance(report, BenchmarkReport)
        summary = report.summary()
        assert "qa_retrieval" in summary["benchmarks"]

    def test_run_all_benchmarks(self):
        from src.training.evaluation.benchmark_suite import BenchmarkReport
        suite = self._suite()
        report = asyncio.get_event_loop().run_until_complete(suite.run())
        assert isinstance(report, BenchmarkReport)

    def test_overall_score(self):
        suite = self._suite()
        report = asyncio.get_event_loop().run_until_complete(
            suite.run(benchmark_names=["qa_retrieval"])
        )
        assert 0.0 <= report.overall_score <= 1.0

    def test_run_invalid_benchmark(self):
        # Unknown benchmark names are skipped with a warning (no raise)
        from src.training.evaluation.benchmark_suite import BenchmarkReport
        suite = self._suite()
        report = asyncio.get_event_loop().run_until_complete(
            suite.run(benchmark_names=["nonexistent"])
        )
        assert isinstance(report, BenchmarkReport)
        assert "nonexistent" not in report.summary().get("benchmarks", {})
