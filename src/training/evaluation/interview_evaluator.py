"""src/training/evaluation/interview_evaluator.py — LLM-as-an-Interviewer 評価

圧迫深掘りテスト: Teacher が「なぜ？」「具体的には？」「矛盾してない？」を
繰り返し、Student の回答深化能力・一貫性・フォローアップ対応力を評価する。

参照論文: arXiv:2412.10424 (LLM-as-an-Interviewer, 2024)

使い方:
    evaluator = InterviewEvaluator(gateway)
    report = await evaluator.run(student_model, question="What is FAISS?")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.llm.gateway import LLMGateway, LLMMessage

logger = logging.getLogger(__name__)

# Teacher がフォローアップ質問を生成するプロンプト
_FOLLOWUP_SYSTEM = """\
あなたは技術面接官です。応募者の回答を受けて、より深い理解を引き出す追加質問を1つだけ生成してください。

追加質問のパターン（状況に応じて選択）:
- 「なぜそうなるのか、もう少し詳しく説明してください」
- 「具体的な例を挙げてもらえますか」
- 「〇〇と言いましたが、〇〇との違いは何ですか」
- 「その回答は〇〇の場合でも成立しますか」
- 「前の説明と矛盾しているように見えますが、どう整合しますか」

質問文のみ出力してください。前置き不要。"""

_FOLLOWUP_USER = """\
【元の質問】
{question}

【これまでの会話】
{history}

【最新の回答】
{last_answer}

次の追加質問を生成してください。"""

# 最終的なインタビューレポートを生成するプロンプト
_REPORT_SYSTEM = """\
あなたは技術面接の評価者です。以下の面接記録を読んで、応募者の技術力を評価してください。

以下の3軸で 0.0〜1.0 のスコアを付けてください:
- initial_quality: 最初の回答の正確さ・網羅性
- adaptability: フィードバックへの適応力（深化できたか）
- consistency: 回答間の一貫性（矛盾がないか）

JSON形式で出力してください:
{"initial_quality": 0.0, "adaptability": 0.0, "consistency": 0.0, "feedback": "コメント"}"""

_REPORT_USER = """\
【面接記録】
{transcript}"""


@dataclass
class InterviewTurn:
    """面接の1ターン。"""
    question: str
    answer: str
    turn: int = 0


@dataclass
class InterviewReport:
    """面接評価レポート。"""
    question: str
    turns: list[InterviewTurn] = field(default_factory=list)
    initial_quality: float = 0.0
    adaptability: float = 0.0
    consistency: float = 0.0
    overall: float = 0.0
    feedback: str = ""
    num_turns: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "num_turns": self.num_turns,
            "initial_quality": round(self.initial_quality, 4),
            "adaptability": round(self.adaptability, 4),
            "consistency": round(self.consistency, 4),
            "overall": round(self.overall, 4),
            "feedback": self.feedback,
        }

    def __str__(self) -> str:
        return (
            f"InterviewReport(turns={self.num_turns}, "
            f"overall={self.overall:.3f}, "
            f"init={self.initial_quality:.3f}, "
            f"adapt={self.adaptability:.3f}, "
            f"consist={self.consistency:.3f})"
        )


class InterviewEvaluator:
    """LLM-as-an-Interviewer 評価器。

    Teacher が面接官として Student に追加質問を繰り返し、
    回答深化能力・一貫性・フォローアップ対応力を評価する。

    Args:
        gateway: Teacher LLM（面接官役）。
        max_turns: 追加質問の最大回数（初回質問を除く）。デフォルト3。
        teacher_provider: 使用するプロバイダ（None で gateway デフォルト）。
    """

    def __init__(
        self,
        gateway: LLMGateway,
        max_turns: int = 3,
        teacher_provider: str | None = None,
    ) -> None:
        self._gateway = gateway
        self._max_turns = max_turns
        self._provider = teacher_provider

    async def run(
        self,
        student_model: Any,
        question: str,
    ) -> InterviewReport:
        """1つの質問に対して多ターン面接を実施する。

        Args:
            student_model: callable(prompt: str) -> str のインターフェースを持つ Student。
                           None の場合は gateway を Student 代わりに使用する。
            question: 最初の面接質問。

        Returns:
            InterviewReport。
        """
        turns: list[InterviewTurn] = []
        current_question = question

        for turn_idx in range(self._max_turns + 1):
            # Student に回答させる
            answer = await self._get_answer(student_model, current_question, turns)
            turns.append(InterviewTurn(
                question=current_question,
                answer=answer,
                turn=turn_idx,
            ))
            logger.debug("InterviewEvaluator turn=%d Q=%s A=%s", turn_idx, current_question[:50], answer[:50])

            # 最後のターンはフォローアップ不要
            if turn_idx >= self._max_turns:
                break

            # Teacher がフォローアップ質問を生成
            current_question = await self._generate_followup(question, turns)

        # Teacher がレポートを生成
        report = await self._generate_report(question, turns)
        report.turns = turns
        report.num_turns = len(turns)
        return report

    async def run_batch(
        self,
        student_model: Any,
        questions: list[str],
    ) -> list[InterviewReport]:
        """複数の質問に対して面接を実施する。"""
        reports = []
        for q in questions:
            report = await self.run(student_model, q)
            reports.append(report)
            logger.info("InterviewEvaluator: %s", report)
        return reports

    def summary(self, reports: list[InterviewReport]) -> dict[str, float]:
        """複数レポートの平均スコアを返す。"""
        if not reports:
            return {"overall": 0.0, "initial_quality": 0.0, "adaptability": 0.0, "consistency": 0.0}
        return {
            "overall": sum(r.overall for r in reports) / len(reports),
            "initial_quality": sum(r.initial_quality for r in reports) / len(reports),
            "adaptability": sum(r.adaptability for r in reports) / len(reports),
            "consistency": sum(r.consistency for r in reports) / len(reports),
        }

    # ── 内部メソッド ────────────────────────────────

    async def _get_answer(
        self,
        student_model: Any,
        question: str,
        history: list[InterviewTurn],
    ) -> str:
        """Student から回答を取得する。"""
        if student_model is not None and callable(student_model):
            try:
                context = "\n".join(
                    f"Q: {t.question}\nA: {t.answer}" for t in history
                )
                prompt = f"{context}\nQ: {question}\nA:" if context else f"Q: {question}\nA:"
                result = student_model(prompt)
                return str(result) if result else ""
            except Exception:
                logger.debug("InterviewEvaluator: student_model call failed, using gateway")

        # student_model が None または呼び出し失敗時は gateway を代用
        messages = [LLMMessage(role="user", content=question)]
        resp = await self._gateway.complete(messages, provider=self._provider)
        return resp.content

    async def _generate_followup(
        self,
        original_question: str,
        turns: list[InterviewTurn],
    ) -> str:
        """Teacher がフォローアップ質問を生成する。"""
        history_text = "\n".join(
            f"ターン{t.turn+1} Q: {t.question}\n      A: {t.answer}"
            for t in turns[:-1]
        )
        last_answer = turns[-1].answer

        messages = [
            LLMMessage(role="system", content=_FOLLOWUP_SYSTEM),
            LLMMessage(
                role="user",
                content=_FOLLOWUP_USER.format(
                    question=original_question,
                    history=history_text or "（初回）",
                    last_answer=last_answer,
                ),
            ),
        ]
        resp = await self._gateway.complete(messages, provider=self._provider)
        return resp.content.strip()

    async def _generate_report(
        self,
        question: str,
        turns: list[InterviewTurn],
    ) -> InterviewReport:
        """Teacher が面接記録を採点してレポートを生成する。"""
        transcript = "\n\n".join(
            f"【ターン{t.turn+1}】\n質問: {t.question}\n回答: {t.answer}"
            for t in turns
        )
        messages = [
            LLMMessage(role="system", content=_REPORT_SYSTEM),
            LLMMessage(role="user", content=_REPORT_USER.format(transcript=transcript)),
        ]
        resp = await self._gateway.complete(messages, provider=self._provider)

        # JSON パース
        import json
        import re
        report = InterviewReport(question=question)
        try:
            m = re.search(r"\{.*\}", resp.content, re.DOTALL)
            if m:
                data = json.loads(m.group())
                report.initial_quality = float(data.get("initial_quality", 0.0))
                report.adaptability = float(data.get("adaptability", 0.0))
                report.consistency = float(data.get("consistency", 0.0))
                report.feedback = str(data.get("feedback", ""))
        except Exception:
            logger.debug("InterviewEvaluator: JSON parse failed, using defaults")

        report.overall = (
            report.initial_quality * 0.4
            + report.adaptability * 0.35
            + report.consistency * 0.25
        )
        return report
