"""tests/unit/test_phase5.py — Phase 5 モジュールの単体テスト

テスト対象:
- src/common/logger.py
- src/common/models.py
- src/sandbox/retry_handler.py
- src/memory/learning/embedding_adapter.py
- src/memory/learning/feedback_collector.py
"""

from __future__ import annotations

import asyncio
import io
import logging
from dataclasses import dataclass

import numpy as np
import pytest

# ===========================================================================
# Logger
# ===========================================================================

class TestSetupLogging:
    def test_setup_text(self):
        from src.common.logger import setup_logging
        stream = io.StringIO()
        setup_logging(level="DEBUG", json_format=False, stream=stream)
        log = logging.getLogger("test_setup_text")
        log.info("hello text")
        output = stream.getvalue()
        assert "hello text" in output

    def test_setup_json(self):
        import json as _json

        from src.common.logger import setup_logging
        stream = io.StringIO()
        setup_logging(level="DEBUG", json_format=True, stream=stream)
        log = logging.getLogger("test_setup_json")
        log.info("hello json")
        line = stream.getvalue().strip().splitlines()[-1]
        data = _json.loads(line)
        assert data["message"] == "hello json"
        assert data["level"] == "INFO"

    def test_level_filtering(self):
        from src.common.logger import setup_logging
        stream = io.StringIO()
        setup_logging(level="WARNING", stream=stream)
        log = logging.getLogger("test_level_filter")
        log.debug("should not appear")
        log.warning("should appear")
        out = stream.getvalue()
        assert "should not appear" not in out
        assert "should appear" in out

    def test_get_logger_returns_logger(self):
        from src.common.logger import get_logger
        log = get_logger("test_module")
        assert isinstance(log, logging.Logger)
        assert log.name == "test_module"


# ===========================================================================
# Common models
# ===========================================================================

class TestQueryRequest:
    def test_defaults(self):
        from src.common.models import QueryRequest
        req = QueryRequest(query="hello")
        assert req.query == "hello"
        assert req.use_memory is True
        assert req.use_rag is True
        assert req.use_sandbox is False

    def test_invalid_empty_query(self):
        from pydantic import ValidationError

        from src.common.models import QueryRequest
        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_max_tokens_bounds(self):
        from pydantic import ValidationError

        from src.common.models import QueryRequest
        with pytest.raises(ValidationError):
            QueryRequest(query="hi", max_tokens=0)
        with pytest.raises(ValidationError):
            QueryRequest(query="hi", max_tokens=99999)


class TestQueryResponse:
    def test_defaults(self):
        from src.common.models import QueryResponse
        resp = QueryResponse(query="q", answer="a")
        assert resp.retrieval_sources == []
        assert resp.execution_result is None
        assert resp.tokens_used == 0


class TestHealthResponse:
    def test_healthy(self):
        from src.common.models import HealthResponse
        h = HealthResponse.healthy({"faiss": "ok"})
        assert h.status == "ok"
        assert h.components["faiss"] == "ok"

    def test_degraded(self):
        from src.common.models import HealthResponse
        h = HealthResponse.degraded("DB down")
        assert h.status == "degraded"
        assert "DB down" in h.components["error"]


class TestSandboxRequest:
    def test_defaults(self):
        from src.common.models import SandboxRequest
        req = SandboxRequest(code="print('hi')")
        assert req.language == "python"
        assert req.timeout_seconds == 10
        assert req.allow_network is False

    def test_invalid_empty_code(self):
        from pydantic import ValidationError

        from src.common.models import SandboxRequest
        with pytest.raises(ValidationError):
            SandboxRequest(code="")


class TestErrorResponse:
    def test_defaults(self):
        from src.common.models import ErrorResponse
        e = ErrorResponse(error="Something failed")
        assert e.code == 500
        assert e.detail is None


class TestTrainingStatusResponse:
    def test_fields(self):
        from src.common.models import TrainingStatusResponse
        s = TrainingStatusResponse(status="running", algorithm="grpo", current_step=10, total_steps=100)
        assert s.status == "running"
        assert s.current_step == 10


# ===========================================================================
# RetryHandler
# ===========================================================================

@dataclass
class _FakeExecResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""


class _FakeExecutor:
    def __init__(self, results: list[_FakeExecResult]):
        self._results = list(results)
        self._calls = 0

    async def run(self, code: str, language: str = "python", timeout_seconds: int = 10):
        result = self._results[min(self._calls, len(self._results) - 1)]
        self._calls += 1
        return result


class TestRetryResult:
    def test_was_fixed_single(self):
        from src.sandbox.retry_handler import RetryResult
        r = RetryResult(success=True, attempts=1)
        assert not r.was_fixed

    def test_was_fixed_multiple(self):
        from src.sandbox.retry_handler import RetryResult
        r = RetryResult(success=True, attempts=2)
        assert r.was_fixed

    def test_was_fixed_failed(self):
        from src.sandbox.retry_handler import RetryResult
        r = RetryResult(success=False, attempts=3)
        assert not r.was_fixed


class TestRetryHandlerSuccess:
    def test_success_first_attempt(self):
        from src.sandbox.retry_handler import RetryHandler
        executor = _FakeExecutor([_FakeExecResult(success=True, stdout="ok")])
        handler = RetryHandler(executor, max_retries=2, retry_delay=0.0)
        result = asyncio.get_event_loop().run_until_complete(
            handler.run_with_retry("print('ok')")
        )
        assert result.success
        assert result.attempts == 1
        assert result.final_output == "ok"

    def test_success_after_retry(self):
        from src.sandbox.retry_handler import RetryHandler
        executor = _FakeExecutor([
            _FakeExecResult(success=False, stderr="NameError: x"),
            _FakeExecResult(success=True, stdout="42"),
        ])
        handler = RetryHandler(executor, max_retries=2, retry_delay=0.0, use_error_analyzer=False)
        result = asyncio.get_event_loop().run_until_complete(
            handler.run_with_retry("x = undefined")
        )
        assert result.success
        assert result.attempts == 2

    def test_fail_all_retries(self):
        from src.sandbox.retry_handler import RetryHandler
        executor = _FakeExecutor([_FakeExecResult(success=False, stderr="Error")])
        handler = RetryHandler(executor, max_retries=1, retry_delay=0.0, use_error_analyzer=False)
        result = asyncio.get_event_loop().run_until_complete(
            handler.run_with_retry("bad code")
        )
        assert not result.success
        assert result.attempts == 2  # initial + 1 retry

    def test_executor_exception(self):
        from src.sandbox.retry_handler import RetryHandler

        class ExplodingExecutor:
            async def run(self, code, **kw):
                raise RuntimeError("Docker down")

        handler = RetryHandler(ExplodingExecutor(), max_retries=0, retry_delay=0.0,
                               use_error_analyzer=False)
        result = asyncio.get_event_loop().run_until_complete(handler.run_with_retry("code"))
        assert not result.success
        assert "Docker down" in result.final_error

    def test_error_history_recorded(self):
        from src.sandbox.retry_handler import RetryHandler
        executor = _FakeExecutor([
            _FakeExecResult(success=False, stderr="SyntaxError"),
            _FakeExecResult(success=False, stderr="NameError"),
        ])
        handler = RetryHandler(executor, max_retries=1, retry_delay=0.0, use_error_analyzer=False)
        result = asyncio.get_event_loop().run_until_complete(handler.run_with_retry("bad"))
        assert len(result.error_history) >= 1


# ===========================================================================
# EmbeddingAdapter
# ===========================================================================

class TestEmbeddingAdapter:
    def _adapter(self, in_dim=8, out_dim=4):
        from src.memory.learning.embedding_adapter import EmbeddingAdapter
        return EmbeddingAdapter(input_dim=in_dim, output_dim=out_dim)

    def test_dimensions(self):
        a = self._adapter(8, 4)
        assert a.input_dim == 8
        assert a.output_dim == 4

    def test_transform_shape(self):
        a = self._adapter(8, 4)
        x = np.random.randn(8).astype(np.float32)
        out = a.transform(x)
        assert out.shape == (4,)

    def test_transform_normalized(self):
        a = self._adapter(8, 4)
        x = np.random.randn(8).astype(np.float32)
        out = a.transform(x)
        norm = np.linalg.norm(out)
        assert abs(norm - 1.0) < 1e-5 or norm < 1e-10  # unit norm or zero

    def test_transform_batch(self):
        a = self._adapter(8, 4)
        x = np.random.randn(3, 8).astype(np.float32)
        out = a.transform(x)
        assert out.shape == (3, 4)

    def test_update_changes_weights(self):
        a = self._adapter(8, 4)
        W_before = a._W.copy()
        x = np.ones(8, dtype=np.float32)
        adapted = a.transform(x)
        a.update(x, adapted, reward=1.0)
        assert not np.allclose(a._W, W_before)

    def test_update_count(self):
        a = self._adapter(8, 4)
        assert a.update_count == 0
        x = np.ones(8, dtype=np.float32)
        a.update(x, a.transform(x), reward=0.8)
        a.update(x, a.transform(x), reward=0.5)
        assert a.update_count == 2

    def test_save_load(self, tmp_path):
        from src.memory.learning.embedding_adapter import EmbeddingAdapter
        a = self._adapter(8, 4)
        x = np.ones(8, dtype=np.float32)
        a.update(x, a.transform(x), reward=1.0)

        path = tmp_path / "adapter.pkl"
        a.save(path)

        b = EmbeddingAdapter.load(path)
        assert b.input_dim == 8
        assert b.output_dim == 4
        assert b.update_count == 1
        assert np.allclose(a._W, b._W)

    def test_reset(self):
        a = self._adapter(8, 4)
        W_init = a._W.copy()
        x = np.ones(8, dtype=np.float32)
        a.update(x, a.transform(x), reward=1.0)
        a.reset()
        assert a.update_count == 0
        # Weights reinitialized (not necessarily same as original since random)
        assert a._W.shape == W_init.shape


# ===========================================================================
# FeedbackCollector
# ===========================================================================

class TestFeedbackEvent:
    def _event(self, reward, feedback_type="click"):
        from src.memory.learning.feedback_collector import FeedbackEvent
        return FeedbackEvent(doc_id="d1", query="q", reward=reward,
                             feedback_type=feedback_type)

    def test_is_positive(self):
        e = self._event(0.8)
        assert e.is_positive
        assert not e.is_negative

    def test_is_negative(self):
        e = self._event(0.2)
        assert e.is_negative
        assert not e.is_positive

    def test_neutral(self):
        e = self._event(0.5)
        assert not e.is_positive
        assert not e.is_negative


class TestFeedbackCollector:
    def _collector(self, max_buffer=100):
        from src.memory.learning.feedback_collector import FeedbackCollector
        return FeedbackCollector(max_buffer=max_buffer)

    def test_record_click(self):
        c = self._collector()
        c.record_click("doc1", query="q", rank=0)
        assert len(c) == 1

    def test_click_rank_adjustment(self):
        c = self._collector()
        c.record_click("d1", rank=0)
        c.record_click("d2", rank=4)
        events = c.peek()
        # rank=0 has higher reward than rank=4
        assert events[0].reward > events[1].reward

    def test_record_explicit_thumbs_up(self):
        c = self._collector()
        c.record_explicit("doc1", thumbs_up=True)
        events = c.peek()
        assert len(events) == 1
        assert events[0].reward == 1.0
        assert events[0].feedback_type == "thumbs_up"

    def test_record_explicit_thumbs_down(self):
        c = self._collector()
        c.record_explicit("doc1", thumbs_up=False)
        assert c.peek()[0].reward == 0.0

    def test_record_explicit_rating(self):
        c = self._collector()
        c.record_explicit("doc1", rating=5.0)
        events = c.peek()
        assert events[0].reward == pytest.approx(1.0)

    def test_record_explicit_rating_min(self):
        c = self._collector()
        c.record_explicit("doc1", rating=1.0)
        assert c.peek()[0].reward == pytest.approx(0.0)

    def test_record_explicit_thumbs_and_rating(self):
        c = self._collector()
        c.record_explicit("doc1", thumbs_up=True, rating=3.0)
        assert len(c) == 2  # both recorded

    def test_drain(self):
        c = self._collector()
        c.record_click("d1")
        c.record_click("d2")
        events = c.drain()
        assert len(events) == 2
        assert len(c) == 0

    def test_peek_does_not_clear(self):
        c = self._collector()
        c.record_click("d1")
        c.peek()
        assert len(c) == 1

    def test_clear(self):
        c = self._collector()
        c.record_click("d1")
        c.clear()
        assert len(c) == 0

    def test_max_buffer_eviction(self):
        c = self._collector(max_buffer=3)
        for i in range(5):
            c.record_click(f"d{i}")
        assert len(c) == 3

    def test_aggregate(self):
        c = self._collector()
        c.record_click("d1", rank=0)
        c.record_click("d1", rank=1)
        c.record_explicit("d1", thumbs_up=True)
        agg = c.aggregate()
        assert "d1" in agg
        a = agg["d1"]
        assert a.n_events == 3
        assert a.click_count == 2
        assert a.thumbs_up == 1

    def test_aggregated_feedback_click_rate(self):
        from src.memory.learning.feedback_collector import AggregatedFeedback
        a = AggregatedFeedback(doc_id="d", n_events=4, click_count=2)
        assert a.click_through_rate == pytest.approx(0.5)

    def test_aggregated_feedback_net_positive(self):
        from src.memory.learning.feedback_collector import AggregatedFeedback
        a = AggregatedFeedback(doc_id="d", thumbs_up=3, thumbs_down=1)
        assert a.net_positive == 2

    def test_record_text_no_analyzer(self):
        c = self._collector()
        asyncio.get_event_loop().run_until_complete(
            c.record_text("d1", text="great answer", query="q")
        )
        events = c.peek()
        assert len(events) == 1
        assert events[0].feedback_type == "text"
        assert events[0].reward == pytest.approx(0.5)  # neutral default
