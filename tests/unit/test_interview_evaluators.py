"""tests/unit/test_interview_evaluators.py — B-2 評価クラスの単体テスト

InterviewEvaluator / MultiChallengeEvaluator / AssumptionCorrectionEvaluator の
LLM 呼び出しをスタブ化してロジックを検証する。
"""

from __future__ import annotations

import asyncio
import sys
import types

# pydantic_settings がない環境向けスタブ（CI 環境と同じ処理）
if "pydantic_settings" not in sys.modules:
    _ps = types.ModuleType("pydantic_settings")

    class _BaseSettings:  # noqa: N801
        model_config: dict = {}
        def __init__(self, **kw: object) -> None: ...
        def model_fields_set(self) -> set: return set()

    class _PydanticBaseSettingsSource:  # noqa: N801
        def __init__(self, settings_cls: type) -> None: ...
        def __call__(self) -> dict: return {}

    _ps.BaseSettings = _BaseSettings  # type: ignore[attr-defined]
    _ps.PydanticBaseSettingsSource = _PydanticBaseSettingsSource  # type: ignore[attr-defined]
    _ps.SettingsConfigDict = lambda **kw: {}  # type: ignore[attr-defined]
    sys.modules["pydantic_settings"] = _ps

import pytest
from unittest.mock import AsyncMock, MagicMock


# ── ゲートウェイスタブ ────────────────────────────────────────

def _make_gateway(content: str = "0.8"):
    """LLMGateway の最小スタブ。complete() が固定テキストを返す。"""
    gw = MagicMock()
    resp = MagicMock()
    resp.content = content
    gw.complete = AsyncMock(return_value=resp)
    return gw


# ────────────────────────────────────────────────────────────────
# TrainingDataGate (B-1 再確認)
# ────────────────────────────────────────────────────────────────

class TestTrainingDataGate:
    def test_high_variance_passes(self):
        from src.training.pipeline import TrainingDataGate, GateConfig, TrainingBatch
        gate = TrainingDataGate(GateConfig(variance_threshold=0.05))
        batch = TrainingBatch(prompts=["q"], responses=["a"], rewards=[0.1, 0.9])
        assert gate.filter(batch) is not None

    def test_low_variance_rejected(self):
        from src.training.pipeline import TrainingDataGate, GateConfig, TrainingBatch
        gate = TrainingDataGate(GateConfig(variance_threshold=0.05))
        batch = TrainingBatch(prompts=["q"], responses=["a"], rewards=[0.5, 0.51])
        assert gate.filter(batch) is None

    def test_low_quality_rejected(self):
        from src.training.pipeline import TrainingDataGate, GateConfig, TrainingBatch
        gate = TrainingDataGate(GateConfig(quality_threshold=0.6, variance_threshold=0.0))
        batch = TrainingBatch(
            prompts=["q"], responses=["a"],
            rewards=[0.1, 0.9],
            metadata=[{"quality_score": 0.3}],
        )
        assert gate.filter(batch) is None

    def test_no_rewards_passes_variance_check(self):
        from src.training.pipeline import TrainingDataGate, GateConfig, TrainingBatch
        gate = TrainingDataGate(GateConfig(variance_threshold=0.1))
        batch = TrainingBatch(prompts=["q"], responses=["a"], rewards=[])
        # rewards が空の場合は分散フィルタをスキップ
        assert gate.filter(batch) is not None

    def test_filter_batches_returns_subset(self):
        from src.training.pipeline import TrainingDataGate, GateConfig, TrainingBatch
        gate = TrainingDataGate(GateConfig(variance_threshold=0.05))
        batches = [
            TrainingBatch(prompts=["q"], responses=["a"], rewards=[0.1, 0.9]),  # pass
            TrainingBatch(prompts=["q"], responses=["a"], rewards=[0.5, 0.51]), # reject
            TrainingBatch(prompts=["q"], responses=["a"], rewards=[0.0, 1.0]),  # pass
        ]
        passed = gate.filter_batches(batches)
        assert len(passed) == 2


# ────────────────────────────────────────────────────────────────
# GRPOAlgorithm StarPO-S 拡張
# ────────────────────────────────────────────────────────────────

class TestGRPOStarPOS:
    def test_variance_filter_skips_low_variance(self):
        """低分散バッチはゼロロスを返してスキップされる。"""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

        from src.training.algorithms.grpo import GRPOAlgorithm
        from src.training.base import TrainingBatch

        algo = GRPOAlgorithm(variance_filter=0.1)
        batch = TrainingBatch(
            prompts=["q1", "q2"],
            responses=["a1", "a2"],
            rewards=[0.5, 0.51],
        )
        loss = algo.compute_loss(batch, None, None)
        assert loss.item() == pytest.approx(0.0)

    def test_high_variance_proceeds(self):
        """高分散バッチは通常通り loss を計算する。"""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

        from src.training.algorithms.grpo import GRPOAlgorithm
        from src.training.base import TrainingBatch

        algo = GRPOAlgorithm(variance_filter=0.05)
        batch = TrainingBatch(
            prompts=["q1", "q2"],
            responses=["a1", "a2"],
            rewards=[0.1, 0.9],
        )
        loss = algo.compute_loss(batch, None, None)
        assert loss is not None

    def test_asymmetric_clip_flag(self):
        """asymmetric_clip=True でインスタンス化できる。"""
        from src.training.algorithms.grpo import GRPOAlgorithm
        algo = GRPOAlgorithm(clip_ratio=0.2, asymmetric_clip=True)
        assert algo._asymmetric_clip is True
        assert algo._clip_ratio == 0.2


# ────────────────────────────────────────────────────────────────
# InterviewEvaluator
# ────────────────────────────────────────────────────────────────

class TestInterviewEvaluator:
    def _make_evaluator(self, max_turns: int = 2):
        from src.training.evaluation.interview_evaluator import InterviewEvaluator
        gw = _make_gateway('{"initial_quality": 0.8, "adaptability": 0.7, "consistency": 0.9, "feedback": "good"}')
        return InterviewEvaluator(gw, max_turns=max_turns)

    def test_run_returns_report(self):
        ev = self._make_evaluator(max_turns=1)
        report = asyncio.get_event_loop().run_until_complete(
            ev.run(None, "What is FAISS?")
        )
        assert report.num_turns == 2  # 初回 + 1追加
        assert 0.0 <= report.overall <= 1.0

    def test_report_scores_in_range(self):
        ev = self._make_evaluator(max_turns=2)
        report = asyncio.get_event_loop().run_until_complete(
            ev.run(None, "Explain TinyLoRA.")
        )
        assert 0.0 <= report.initial_quality <= 1.0
        assert 0.0 <= report.adaptability <= 1.0
        assert 0.0 <= report.consistency <= 1.0

    def test_student_model_callable(self):
        """student_model が callable の場合は呼ばれる。"""
        from src.training.evaluation.interview_evaluator import InterviewEvaluator
        gw = _make_gateway('{"initial_quality": 0.5, "adaptability": 0.5, "consistency": 0.5, "feedback": "ok"}')
        ev = InterviewEvaluator(gw, max_turns=1)
        call_count = {"n": 0}
        def student(prompt):
            call_count["n"] += 1
            return "Some answer"
        report = asyncio.get_event_loop().run_until_complete(
            ev.run(student, "What is GRPO?")
        )
        assert call_count["n"] >= 1
        assert report.num_turns == 2

    def test_summary_averages(self):
        from src.training.evaluation.interview_evaluator import InterviewEvaluator, InterviewReport
        ev = self._make_evaluator()
        reports = [
            InterviewReport(question="q1", overall=0.8, initial_quality=0.8, adaptability=0.8, consistency=0.8),
            InterviewReport(question="q2", overall=0.6, initial_quality=0.6, adaptability=0.6, consistency=0.6),
        ]
        summary = ev.summary(reports)
        assert summary["overall"] == pytest.approx(0.7)

    def test_to_dict(self):
        from src.training.evaluation.interview_evaluator import InterviewReport
        r = InterviewReport(question="q", overall=0.75, initial_quality=0.8,
                            adaptability=0.7, consistency=0.75, feedback="ok", num_turns=3)
        d = r.to_dict()
        assert d["num_turns"] == 3
        assert d["overall"] == pytest.approx(0.75, abs=1e-3)


# ────────────────────────────────────────────────────────────────
# MultiChallengeEvaluator
# ────────────────────────────────────────────────────────────────

class TestMultiChallengeEvaluator:
    def _make_evaluator(self):
        from src.training.evaluation.multi_challenge_evaluator import (
            MultiChallengeEvaluator, ChallengeCase,
        )
        gw = _make_gateway("1")  # judge が "1" を返す = passed
        cases = [
            ChallengeCase(
                category="instruction_retention",
                setup_turns=[("以降JSON形式で返して", "JSON形式を守る")],
                test_question="FAISSとは？",
                condition='回答がJSON形式である',
            ),
            ChallengeCase(
                category="self_coherence",
                setup_turns=[],
                test_question="FAISSの欠点は？",
                condition="前の発言と矛盾しない",
            ),
        ]
        return MultiChallengeEvaluator(gw, cases=cases)

    def test_run_returns_report(self):
        ev = self._make_evaluator()
        report = asyncio.get_event_loop().run_until_complete(ev.run(None))
        assert len(report.results) == 2
        assert 0.0 <= report.overall <= 1.0

    def test_category_scores_present(self):
        ev = self._make_evaluator()
        report = asyncio.get_event_loop().run_until_complete(ev.run(None))
        assert "instruction_retention" in report.category_scores
        assert "self_coherence" in report.category_scores

    def test_all_pass_when_judge_returns_1(self):
        ev = self._make_evaluator()
        report = asyncio.get_event_loop().run_until_complete(ev.run(None))
        assert all(r.passed for r in report.results)
        assert report.overall == pytest.approx(1.0)

    def test_all_fail_when_judge_returns_0(self):
        from src.training.evaluation.multi_challenge_evaluator import (
            MultiChallengeEvaluator, ChallengeCase,
        )
        gw = _make_gateway("0")
        cases = [ChallengeCase(
            category="context_continuity", setup_turns=[],
            test_question="q", condition="c",
        )]
        ev = MultiChallengeEvaluator(gw, cases=cases)
        report = asyncio.get_event_loop().run_until_complete(ev.run(None))
        assert not any(r.passed for r in report.results)
        assert report.overall == pytest.approx(0.0)

    def test_to_dict(self):
        ev = self._make_evaluator()
        report = asyncio.get_event_loop().run_until_complete(ev.run(None))
        d = report.to_dict()
        assert "overall" in d
        assert "category_scores" in d
        assert d["num_cases"] == 2

    def test_add_case(self):
        from src.training.evaluation.multi_challenge_evaluator import (
            MultiChallengeEvaluator, ChallengeCase,
        )
        gw = _make_gateway("1")
        ev = MultiChallengeEvaluator(gw, cases=[])
        ev.add_case(ChallengeCase(category="x", setup_turns=[], test_question="q", condition="c"))
        assert len(ev._cases) == 1


# ────────────────────────────────────────────────────────────────
# AssumptionCorrectionEvaluator
# ────────────────────────────────────────────────────────────────

class TestAssumptionCorrectionEvaluator:
    _GOOD_JUDGE = '{"correction_detected": 1, "correct_answer_given": 1, "quality": 0.9}'
    _BAD_JUDGE  = '{"correction_detected": 0, "correct_answer_given": 0, "quality": 0.2}'

    def _make_evaluator(self, judge_response: str):
        from src.training.evaluation.assumption_correction_evaluator import (
            AssumptionCorrectionEvaluator, AssumptionCase,
        )
        gw = _make_gateway(judge_response)
        cases = [
            AssumptionCase(
                question="GILはマルチコアを使えますよね？",
                wrong_assumption="GILがマルチコア対応",
                correct_fact="GILはマルチコアCPUバウンド並列を妨げる",
                category="python",
            ),
        ]
        return AssumptionCorrectionEvaluator(gw, cases=cases)

    def test_full_marks_when_both_correct(self):
        ev = self._make_evaluator(self._GOOD_JUDGE)
        report = asyncio.get_event_loop().run_until_complete(ev.run(None))
        assert report.correction_rate == pytest.approx(1.0)
        assert report.correct_answer_rate == pytest.approx(1.0)
        assert report.full_marks_rate == pytest.approx(1.0)

    def test_zero_score_when_both_wrong(self):
        ev = self._make_evaluator(self._BAD_JUDGE)
        report = asyncio.get_event_loop().run_until_complete(ev.run(None))
        assert report.correction_rate == pytest.approx(0.0)
        assert report.full_marks_rate == pytest.approx(0.0)

    def test_metacognition_prompt_prepended(self):
        """include_metacognition=True のとき学生へのプロンプトに確認文が含まれる。"""
        from src.training.evaluation.assumption_correction_evaluator import (
            AssumptionCorrectionEvaluator, AssumptionCase,
        )
        gw = _make_gateway(self._GOOD_JUDGE)
        called_with = {}
        original_get_answer = AssumptionCorrectionEvaluator._get_answer

        async def capture_get_answer(self_inner, student_model, prompt):
            called_with["prompt"] = prompt
            return "answer"

        AssumptionCorrectionEvaluator._get_answer = capture_get_answer
        cases = [AssumptionCase(
            question="q", wrong_assumption="w", correct_fact="c", category="test"
        )]
        ev = AssumptionCorrectionEvaluator(gw, cases=cases, include_metacognition=True)
        asyncio.get_event_loop().run_until_complete(ev.run(None))
        AssumptionCorrectionEvaluator._get_answer = original_get_answer

        assert "メタ認知" in called_with.get("prompt", "") or "システム" in called_with.get("prompt", "")

    def test_to_dict(self):
        ev = self._make_evaluator(self._GOOD_JUDGE)
        report = asyncio.get_event_loop().run_until_complete(ev.run(None))
        d = report.to_dict()
        assert "correction_rate" in d
        assert "full_marks_rate" in d
        assert d["num_cases"] == 1

    def test_add_case(self):
        from src.training.evaluation.assumption_correction_evaluator import (
            AssumptionCorrectionEvaluator, AssumptionCase,
        )
        gw = _make_gateway(self._GOOD_JUDGE)
        ev = AssumptionCorrectionEvaluator(gw, cases=[])
        ev.add_case(AssumptionCase(question="q", wrong_assumption="w",
                                   correct_fact="c", category="x"))
        assert len(ev._cases) == 1

    def test_default_cases_exist(self):
        """デフォルト5ケースが組み込まれている。"""
        from src.training.evaluation.assumption_correction_evaluator import (
            AssumptionCorrectionEvaluator, _DEFAULT_CASES,
        )
        assert len(_DEFAULT_CASES) == 5
        categories = {c.category for c in _DEFAULT_CASES}
        assert "python" in categories
        assert "faiss" in categories
