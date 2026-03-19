"""src/training/evaluation — 評価フレームワーク。"""

from src.training.evaluation.assumption_correction_evaluator import (
    AssumptionCase,
    AssumptionCorrectionEvaluator,
    AssumptionCorrectionReport,
    AssumptionResult,
)
from src.training.evaluation.benchmark_suite import BenchmarkReport, BenchmarkSuite
from src.training.evaluation.interview_evaluator import (
    InterviewEvaluator,
    InterviewReport,
    InterviewTurn,
)
from src.training.evaluation.multi_challenge_evaluator import (
    ChallengeCase,
    MultiChallengeEvaluator,
    MultiChallengeReport,
    MultiChallengeResult,
)
from src.training.evaluation.student_evaluator import EvalMetrics, EvalSample, StudentEvaluator
from src.training.evaluation.teacher_comparison import TeacherComparison

__all__ = [
    # 既存
    "BenchmarkSuite",
    "BenchmarkReport",
    "StudentEvaluator",
    "EvalSample",
    "EvalMetrics",
    "TeacherComparison",
    # B-2 新規
    "InterviewEvaluator",
    "InterviewReport",
    "InterviewTurn",
    "MultiChallengeEvaluator",
    "MultiChallengeReport",
    "MultiChallengeResult",
    "ChallengeCase",
    "AssumptionCorrectionEvaluator",
    "AssumptionCorrectionReport",
    "AssumptionResult",
    "AssumptionCase",
]
