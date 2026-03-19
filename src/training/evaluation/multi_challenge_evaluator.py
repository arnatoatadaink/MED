"""src/training/evaluation/multi_challenge_evaluator.py — MultiChallenge 評価

長期指示維持テスト: 複数ターンにわたって指示・文脈・一貫性を維持できるか評価する。

4カテゴリを測定:
1. instruction_retention: 複数ターン前の指示を守り続けられるか
2. inference_memory: ユーザー情報を正しく推論・記憶できるか
3. self_coherence: 自分の前の発言と矛盾しないか
4. context_continuity: 会話の文脈を正しく引き継げるか

参照: arXiv:2501.17399 (MultiChallenge, 2025)
参考値: Claude 3.5 Sonnet でも正解率 41.4%

使い方:
    evaluator = MultiChallengeEvaluator(gateway)
    report = await evaluator.run(student_model)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.llm.gateway import LLMGateway, LLMMessage

logger = logging.getLogger(__name__)

# Judge プロンプト: 指示が守られているかを 0/1 で判定
_JUDGE_SYSTEM = """\
あなたは会話の評価者です。
会話の記録を読み、「評価対象の条件」が最後の回答で守られているかを判定してください。
守られている: 1
守られていない: 0
数字のみ出力してください。"""

_JUDGE_USER = """\
【評価対象の条件】
{condition}

【会話記録】
{transcript}

【最後の回答】
{last_answer}

判定（0 or 1）:"""


@dataclass
class ChallengeCase:
    """MultiChallenge の1テストケース。"""
    category: str
    setup_turns: list[tuple[str, str]]
    """事前に与える (user_message, expected_assistant_behavior) のペア。"""
    test_question: str
    """最後に投げる評価用質問。"""
    condition: str
    """守られるべき条件（Judge プロンプトに渡す）。"""


@dataclass
class MultiChallengeResult:
    """1ケースの評価結果。"""
    case: ChallengeCase
    passed: bool = False
    score: float = 0.0
    last_answer: str = ""


@dataclass
class MultiChallengeReport:
    """MultiChallenge 全体レポート。"""
    results: list[MultiChallengeResult] = field(default_factory=list)
    category_scores: dict[str, float] = field(default_factory=dict)
    overall: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": round(self.overall, 4),
            "category_scores": {k: round(v, 4) for k, v in self.category_scores.items()},
            "num_cases": len(self.results),
            "num_passed": sum(1 for r in self.results if r.passed),
        }

    def __str__(self) -> str:
        cats = ", ".join(f"{k}={v:.2f}" for k, v in self.category_scores.items())
        return f"MultiChallengeReport(overall={self.overall:.3f}, {cats})"


# ── 組み込みテストケース ──────────────────────────────────────

_DEFAULT_CASES: list[ChallengeCase] = [
    # instruction_retention: JSON形式を維持できるか
    ChallengeCase(
        category="instruction_retention",
        setup_turns=[
            ("以降の回答はすべてJSON形式で返してください。例: {\"answer\": \"...\"}",
             "必ずJSON形式で返す"),
        ],
        test_question="Pythonのリストとタプルの違いを教えてください。",
        condition='回答が {"answer": "..."} 形式のJSON文字列になっている',
    ),
    # instruction_retention: 文字数制限を維持できるか
    ChallengeCase(
        category="instruction_retention",
        setup_turns=[
            ("以降の回答はすべて50文字以内で簡潔に答えてください。", "50文字以内で答える"),
            ("Pythonとは何ですか？", "50文字以内の回答"),
        ],
        test_question="FAISSとは何ですか？",
        condition="回答が50文字以内である",
    ),
    # inference_memory: ユーザーのスキルレベルを記憶できるか
    ChallengeCase(
        category="inference_memory",
        setup_turns=[
            ("私はPythonを3年使っていますが、機械学習は初めてです。", "ユーザーのスキルを把握"),
            ("ベクトルって何ですか？", "初心者向けに説明"),
        ],
        test_question="FAISSのインデックスの作り方を教えてください。",
        condition="機械学習初心者向けの説明になっている（専門用語を避けているか、説明している）",
    ),
    # self_coherence: 前の発言と矛盾しないか
    ChallengeCase(
        category="self_coherence",
        setup_turns=[
            ("FAISSとAnnoyの比較をしてください。", "FAISSの特徴を述べる"),
        ],
        test_question="FAISSの最大の欠点は何ですか？",
        condition="前のターンでFAISSについて述べた内容と矛盾していない",
    ),
    # context_continuity: 文脈引き継ぎ
    ChallengeCase(
        category="context_continuity",
        setup_turns=[
            ("ベクトルデータベースの選択肢として、FAISS、Pinecone、Weaviateを検討しています。",
             "3つの選択肢を把握"),
            ("予算が限られています。", "予算制約を把握"),
        ],
        test_question="どれがおすすめですか？",
        condition="3つの選択肢と予算制約の両方を考慮した回答になっている",
    ),
]


class MultiChallengeEvaluator:
    """MultiChallenge 評価器。

    複数ターンにわたる指示維持・記憶・一貫性・文脈継続を評価する。

    Args:
        gateway: Teacher LLM（Judge 役）。
        cases: テストケースリスト（None でデフォルトケースを使用）。
        teacher_provider: Judge に使用するプロバイダ。
    """

    def __init__(
        self,
        gateway: LLMGateway,
        cases: list[ChallengeCase] | None = None,
        teacher_provider: str | None = None,
    ) -> None:
        self._gateway = gateway
        self._cases = list(cases) if cases is not None else list(_DEFAULT_CASES)
        self._provider = teacher_provider

    def add_case(self, case: ChallengeCase) -> None:
        """テストケースを追加する。"""
        self._cases.append(case)

    async def run(
        self,
        student_model: Any,
        categories: list[str] | None = None,
    ) -> MultiChallengeReport:
        """MultiChallenge を実行する。

        Args:
            student_model: callable(prompt: str, history: list) -> str。
            categories: 実行するカテゴリ（None で全て）。

        Returns:
            MultiChallengeReport。
        """
        cases = self._cases
        if categories:
            cases = [c for c in cases if c.category in categories]

        results: list[MultiChallengeResult] = []
        for case in cases:
            result = await self._run_case(student_model, case)
            results.append(result)
            logger.info(
                "MultiChallenge [%s]: passed=%s score=%.2f",
                case.category, result.passed, result.score,
            )

        report = self._make_report(results)
        logger.info("MultiChallenge complete: %s", report)
        return report

    # ── 内部メソッド ────────────────────────────────

    async def _run_case(
        self,
        student_model: Any,
        case: ChallengeCase,
    ) -> MultiChallengeResult:
        """1ケースを実行する。"""
        history: list[tuple[str, str]] = []

        # セットアップターンを実行
        for user_msg, _ in case.setup_turns:
            answer = await self._get_answer(student_model, user_msg, history)
            history.append((user_msg, answer))

        # 評価用質問
        last_answer = await self._get_answer(student_model, case.test_question, history)
        history.append((case.test_question, last_answer))

        # Judge で採点
        score = await self._judge(case, history, last_answer)
        return MultiChallengeResult(
            case=case,
            passed=score >= 0.5,
            score=score,
            last_answer=last_answer,
        )

    async def _get_answer(
        self,
        student_model: Any,
        question: str,
        history: list[tuple[str, str]],
    ) -> str:
        """Student から回答を取得する。"""
        if student_model is not None and callable(student_model):
            try:
                return str(student_model(question, history))
            except Exception:
                pass

        # フォールバック: gateway を使用
        messages = [
            LLMMessage(role="user" if i % 2 == 0 else "assistant", content=msg)
            for pair in history for i, msg in enumerate(pair)
        ]
        messages.append(LLMMessage(role="user", content=question))
        resp = await self._gateway.complete(messages, provider=self._provider)
        return resp.content

    async def _judge(
        self,
        case: ChallengeCase,
        history: list[tuple[str, str]],
        last_answer: str,
    ) -> float:
        """Teacher が条件を満たしているか判定する。0.0 or 1.0。"""
        transcript = "\n".join(
            f"User: {u}\nAssistant: {a}" for u, a in history[:-1]
        )
        messages = [
            LLMMessage(role="system", content=_JUDGE_SYSTEM),
            LLMMessage(
                role="user",
                content=_JUDGE_USER.format(
                    condition=case.condition,
                    transcript=transcript or "（初回）",
                    last_answer=last_answer,
                ),
            ),
        ]
        resp = await self._gateway.complete(messages, provider=self._provider)
        try:
            return float(resp.content.strip()[:1])
        except (ValueError, IndexError):
            return 0.0

    def _make_report(self, results: list[MultiChallengeResult]) -> MultiChallengeReport:
        """結果をカテゴリ別に集計してレポートを生成する。"""
        by_category: dict[str, list[float]] = {}
        for r in results:
            by_category.setdefault(r.case.category, []).append(r.score)

        category_scores = {
            cat: sum(scores) / len(scores)
            for cat, scores in by_category.items()
        }
        overall = sum(category_scores.values()) / len(category_scores) if category_scores else 0.0

        return MultiChallengeReport(
            results=results,
            category_scores=category_scores,
            overall=overall,
        )
