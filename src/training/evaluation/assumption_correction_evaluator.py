"""src/training/evaluation/assumption_correction_evaluator.py — 前提訂正評価（MEDオリジナル）

ユーザーが誤った前提で質問したとき、モデルが:
1. 誤りを検出して指摘できるか
2. 正しい前提で回答し直せるか
を評価するMED固有のテスト。

既存研究（LLM-as-an-Interviewer / MultiChallenge）が測定していない観点:
- 「前提の変質を能動的に訂正するか」

使い方:
    evaluator = AssumptionCorrectionEvaluator(gateway)
    report = await evaluator.run(student_model)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.llm.gateway import LLMGateway, LLMMessage

logger = logging.getLogger(__name__)

# Judge プロンプト
_JUDGE_SYSTEM = """\
あなたは回答評価者です。
以下の回答が「誤前提の指摘」と「正しい前提での回答」の両方を行っているか評価してください。

評価基準:
- correction_detected: 誤前提を指摘しているか (0 or 1)
- correct_answer_given: 正しい前提に基づいて回答しているか (0 or 1)
- quality: 全体的な回答品質 (0.0〜1.0)

JSON形式のみ出力: {"correction_detected": 0, "correct_answer_given": 0, "quality": 0.0}"""

_JUDGE_USER = """\
【誤った前提を含む質問】
{question}

【実際の正しい事実】
{correct_fact}

【評価対象の回答】
{answer}

評価（JSON）:"""


@dataclass
class AssumptionCase:
    """前提訂正テストの1ケース。"""
    question: str
    """誤った前提を含む質問。"""
    wrong_assumption: str
    """質問に含まれる誤った前提。"""
    correct_fact: str
    """実際の正しい事実。"""
    category: str = "general"


@dataclass
class AssumptionResult:
    """1ケースの評価結果。"""
    case: AssumptionCase
    answer: str = ""
    correction_detected: bool = False
    correct_answer_given: bool = False
    quality: float = 0.0

    @property
    def score(self) -> float:
        """総合スコア: 指摘(0.5) + 正解(0.5)。"""
        return (0.5 * self.correction_detected + 0.5 * self.correct_answer_given) * self.quality

    @property
    def full_marks(self) -> bool:
        return self.correction_detected and self.correct_answer_given


@dataclass
class AssumptionCorrectionReport:
    """前提訂正評価の全体レポート。"""
    results: list[AssumptionResult] = field(default_factory=list)
    correction_rate: float = 0.0
    correct_answer_rate: float = 0.0
    full_marks_rate: float = 0.0
    overall: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": round(self.overall, 4),
            "correction_rate": round(self.correction_rate, 4),
            "correct_answer_rate": round(self.correct_answer_rate, 4),
            "full_marks_rate": round(self.full_marks_rate, 4),
            "num_cases": len(self.results),
        }

    def __str__(self) -> str:
        return (
            f"AssumptionCorrectionReport("
            f"overall={self.overall:.3f}, "
            f"correction={self.correction_rate:.3f}, "
            f"correct_ans={self.correct_answer_rate:.3f}, "
            f"full={self.full_marks_rate:.3f})"
        )


# ── 組み込みテストケース ──────────────────────────────────────

_DEFAULT_CASES: list[AssumptionCase] = [
    AssumptionCase(
        question="PythonのGILはマルチコアCPUを活用して処理を高速化できますよね？",
        wrong_assumption="GILがマルチコアを活用できる",
        correct_fact="PythonのGIL（Global Interpreter Lock）はCPUバウンドな処理でのマルチコア並列実行を妨げる。"
                     "マルチコアを活用したい場合はmultiprocessingモジュールを使う必要がある。",
        category="python",
    ),
    AssumptionCase(
        question="FAISSはデフォルトでネットワーク越しに分散検索ができますよね？",
        wrong_assumption="FAISSがデフォルトで分散検索に対応している",
        correct_fact="FAISSはデフォルトではシングルマシン向けのライブラリ。"
                     "分散検索にはFaiss + 独自シャーディングか、Milvus/Weaviate等の分散対応VectorDBを使う必要がある。",
        category="faiss",
    ),
    AssumptionCase(
        question="LLMのfine-tuningは必ず大量のGPUメモリが必要ですよね？",
        wrong_assumption="fine-tuningに必ず大量のGPUメモリが必要",
        correct_fact="LoRA/QLoRA等のPEFT手法やTinyLoRAを使えば少ないメモリでfine-tuningが可能。"
                     "TinyLoRAは13パラメータのみ学習する極少パラメータ方式。",
        category="training",
    ),
    AssumptionCase(
        question="RAGシステムでは検索精度を上げるほど回答品質は必ず向上しますよね？",
        wrong_assumption="検索精度と回答品質が常に比例する",
        correct_fact="検索精度が高くても、LLMが検索結果を正しく統合できない場合や、"
                     "検索結果が回答に不要な情報を含む場合は回答品質が下がることがある（lost in the middle問題等）。",
        category="rag",
    ),
    AssumptionCase(
        question="Transformerのアテンション機構は計算量がシーケンス長に対して線形ですよね？",
        wrong_assumption="アテンション計算量がシーケンス長に線形",
        correct_fact="標準的なTransformerのSelf-Attentionの計算量はシーケンス長Nに対してO(N²)。"
                     "線形アテンション（Linformer、Performer等）はO(N)に近似する手法。",
        category="ml_theory",
    ),
]


class AssumptionCorrectionEvaluator:
    """前提訂正評価器（MEDオリジナル）。

    誤前提クエリに対してモデルが誤りを指摘し、
    正しい前提で回答し直せるかを評価する。

    Args:
        gateway: Teacher LLM（Judge 役）。
        cases: テストケースリスト（None でデフォルトケースを使用）。
        teacher_provider: Judge に使用するプロバイダ。
        include_metacognition: True のとき「この前提で合っていますか？」という
            メタ認知確認を Student のシステムプロンプトに含める。
    """

    def __init__(
        self,
        gateway: LLMGateway,
        cases: list[AssumptionCase] | None = None,
        teacher_provider: str | None = None,
        include_metacognition: bool = True,
    ) -> None:
        self._gateway = gateway
        self._cases = list(cases) if cases is not None else list(_DEFAULT_CASES)
        self._provider = teacher_provider
        self._include_metacognition = include_metacognition

    def add_case(self, case: AssumptionCase) -> None:
        """テストケースを追加する。"""
        self._cases.append(case)

    async def run(
        self,
        student_model: Any,
        categories: list[str] | None = None,
    ) -> AssumptionCorrectionReport:
        """前提訂正評価を実行する。

        Args:
            student_model: callable(prompt: str) -> str。
            categories: 実行するカテゴリ（None で全て）。

        Returns:
            AssumptionCorrectionReport。
        """
        cases = self._cases
        if categories:
            cases = [c for c in cases if c.category in categories]

        results: list[AssumptionResult] = []
        for case in cases:
            result = await self._run_case(student_model, case)
            results.append(result)
            logger.info(
                "AssumptionCorrection [%s]: correction=%s correct=%s score=%.2f",
                case.category, result.correction_detected,
                result.correct_answer_given, result.score,
            )

        report = self._make_report(results)
        logger.info("AssumptionCorrection complete: %s", report)
        return report

    # ── 内部メソッド ────────────────────────────────

    async def _run_case(
        self,
        student_model: Any,
        case: AssumptionCase,
    ) -> AssumptionResult:
        """1ケースを実行する。"""
        # メタ認知プロンプトを含める場合はシステムプロンプトを付加
        if self._include_metacognition:
            prompt = (
                "【システム】質問に誤った前提が含まれている場合は必ず指摘してから、"
                "正しい情報に基づいて回答してください。\n\n"
                f"【質問】{case.question}"
            )
        else:
            prompt = case.question

        # Student の回答を取得
        answer = await self._get_answer(student_model, prompt)

        # Judge で採点
        correction, correct_ans, quality = await self._judge(case, answer)

        return AssumptionResult(
            case=case,
            answer=answer,
            correction_detected=correction,
            correct_answer_given=correct_ans,
            quality=quality,
        )

    async def _get_answer(self, student_model: Any, prompt: str) -> str:
        """Student から回答を取得する。"""
        if student_model is not None and callable(student_model):
            try:
                return str(student_model(prompt))
            except Exception:
                pass

        messages = [LLMMessage(role="user", content=prompt)]
        resp = await self._gateway.complete(messages, provider=self._provider)
        return resp.content

    async def _judge(
        self,
        case: AssumptionCase,
        answer: str,
    ) -> tuple[bool, bool, float]:
        """Teacher が採点する。(correction_detected, correct_answer_given, quality) を返す。"""
        messages = [
            LLMMessage(role="system", content=_JUDGE_SYSTEM),
            LLMMessage(
                role="user",
                content=_JUDGE_USER.format(
                    question=case.question,
                    correct_fact=case.correct_fact,
                    answer=answer,
                ),
            ),
        ]
        resp = await self._gateway.complete(messages, provider=self._provider)

        import json
        import re
        try:
            m = re.search(r"\{.*\}", resp.content, re.DOTALL)
            if m:
                data = json.loads(m.group())
                correction = bool(int(data.get("correction_detected", 0)))
                correct_ans = bool(int(data.get("correct_answer_given", 0)))
                quality = float(data.get("quality", 0.5))
                return correction, correct_ans, quality
        except Exception:
            logger.debug("AssumptionCorrectionEvaluator: JSON parse failed")

        return False, False, 0.0

    def _make_report(self, results: list[AssumptionResult]) -> AssumptionCorrectionReport:
        """結果を集計してレポートを生成する。"""
        if not results:
            return AssumptionCorrectionReport()

        n = len(results)
        correction_rate = sum(1 for r in results if r.correction_detected) / n
        correct_answer_rate = sum(1 for r in results if r.correct_answer_given) / n
        full_marks_rate = sum(1 for r in results if r.full_marks) / n
        overall = sum(r.score for r in results) / n

        return AssumptionCorrectionReport(
            results=results,
            correction_rate=correction_rate,
            correct_answer_rate=correct_answer_rate,
            full_marks_rate=full_marks_rate,
            overall=overall,
        )
