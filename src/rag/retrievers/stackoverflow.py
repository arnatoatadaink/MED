"""src/rag/retrievers/stackoverflow.py — Stack Overflow Search レトリーバー

質問ではなく**回答**を優先的に取得する。

戦略:
  1. search/advanced で回答済み質問を検索
  2. 上位質問の accepted_answer / 高スコア回答を /answers エンドポイントで取得
  3. 回答が取得できなかった場合のみ質問本文をフォールバックとして使用
"""

from __future__ import annotations

import html as html_lib
import logging
import re

from src.rag.retriever import BaseRetriever, RawResult

logger = logging.getLogger(__name__)

_API_BASE = "https://api.stackexchange.com/2.3"
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# プロンプトインジェクションによく使われるパターン
_INJECT_RE = re.compile(
    r"(ignore\s+(previous|above|all)\s+instructions?|"
    r"system\s*prompt|<\|im_start\|>|<\|im_end\|>|"
    r"\[INST\]|\[/INST\]|###\s*instruction|"
    r"you\s+are\s+now|forget\s+everything|disregard\s+)",
    re.IGNORECASE,
)


def _strip_html(html: str) -> str:
    """HTML タグを除去してプレーンテキストにする。"""
    text = _HTML_TAG_RE.sub(" ", html)
    return re.sub(r"\s+", " ", text).strip()


def _sanitize(text: str) -> str:
    """プロンプトインジェクションパターンを無害化する。"""
    return _INJECT_RE.sub("[REDACTED]", text)


class StackOverflowRetriever(BaseRetriever):
    """Stack Exchange API を使った Stack Overflow 検索。

    回答済み質問を検索し、その**回答本文**を返す。
    質問本文は回答が取れない場合のフォールバックとしてのみ使用する。

    Args:
        min_answer_score: 回答の最低スコア。これ未満の回答は除外する。
        prefer_accepted: True なら accepted_answer を優先する。
    """

    def __init__(
        self,
        min_answer_score: int = 1,
        prefer_accepted: bool = True,
    ) -> None:
        self._min_answer_score = min_answer_score
        self._prefer_accepted = prefer_accepted

    @property
    def source_name(self) -> str:
        return "stackoverflow"

    def is_available(self) -> bool:
        return True  # 公開 API のため常に利用可能

    async def _do_search(self, query: str, max_results: int = 5) -> list[RawResult]:
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Step 1: 回答済み質問を検索
            params = {
                "q": query,
                "site": "stackoverflow",
                "pagesize": max_results,
                "order": "desc",
                "sort": "relevance",
                "filter": "withbody",
                "answers": 1,  # 回答が1件以上ある質問
            }
            resp = await client.get(f"{_API_BASE}/search/advanced", params=params)
            resp.raise_for_status()
            data = resp.json()

            questions = data.get("items", [])[:max_results]
            if not questions:
                return []

            # Step 2: 質問IDから回答を一括取得
            question_ids = [str(q["question_id"]) for q in questions]
            question_map = {q["question_id"]: q for q in questions}

            answers_params = {
                "site": "stackoverflow",
                "pagesize": max_results * 3,  # 質問あたり複数回答を想定
                "order": "desc",
                "sort": "votes",
                "filter": "withbody",
            }
            ids_str = ";".join(question_ids)
            ans_resp = await client.get(
                f"{_API_BASE}/questions/{ids_str}/answers",
                params=answers_params,
            )
            ans_resp.raise_for_status()
            ans_data = ans_resp.json()

        # Step 3: 質問ごとにベスト回答を選択
        # question_id → list of answers
        answers_by_q: dict[int, list[dict]] = {}
        for ans in ans_data.get("items", []):
            qid = ans.get("question_id")
            if qid:
                answers_by_q.setdefault(qid, []).append(ans)

        results: list[RawResult] = []
        for q in questions:
            qid = q["question_id"]
            q_title = q.get("title", "")
            q_link = q.get("link", "")
            q_tags = q.get("tags", [])
            q_body = _sanitize(_strip_html(q.get("body", "")))[:1000]  # 質問本文（CoTデータ用）

            ans_list = answers_by_q.get(qid, [])

            # ベスト回答を選択: accepted > 高スコア
            best_ans = self._pick_best_answer(ans_list, q.get("accepted_answer_id"))

            if best_ans:
                body = _sanitize(html_lib.unescape(_strip_html(best_ans.get("body", ""))))
                ans_score = best_ans.get("score", 0)
                is_accepted = best_ans.get("is_accepted", False)

                if ans_score < self._min_answer_score and not is_accepted:
                    logger.debug("Skipping low-score answer (score=%d) for q=%d", ans_score, qid)
                    continue

                results.append(RawResult(
                    title=q_title,
                    content=body,
                    url=q_link,
                    source=self.source_name,
                    score=float(ans_score),
                    metadata={
                        "answer_count": q.get("answer_count", 0),
                        "is_answered": True,
                        "is_accepted_answer": is_accepted,
                        "answer_score": ans_score,
                        "question_score": q.get("score", 0),
                        "view_count": q.get("view_count", 0),
                        "tags": q_tags,
                        "content_type": "answer",
                        "question_body": q_body,
                    },
                ))
            else:
                # フォールバック: 回答なしの場合のみ質問本文を使用（低スコア付与）
                logger.debug("No suitable answer for q=%d, using question body as fallback", qid)
                results.append(RawResult(
                    title=q_title,
                    content=q_body,
                    url=q_link,
                    source=self.source_name,
                    score=max(float(q.get("score", 0)) * 0.3, 0.1),  # 質問は低スコア
                    metadata={
                        "answer_count": q.get("answer_count", 0),
                        "is_answered": q.get("is_answered", False),
                        "question_score": q.get("score", 0),
                        "view_count": q.get("view_count", 0),
                        "tags": q_tags,
                        "content_type": "question_fallback",
                        "question_body": q_body,
                    },
                ))

        logger.info(
            "SO search: %d questions → %d results (%d answers, %d fallbacks)",
            len(questions), len(results),
            sum(1 for r in results if r.metadata.get("content_type") == "answer"),
            sum(1 for r in results if r.metadata.get("content_type") == "question_fallback"),
        )
        return results

    def _pick_best_answer(
        self, answers: list[dict], accepted_id: int | None,
    ) -> dict | None:
        """回答リストからベスト回答を選択する。"""
        if not answers:
            return None

        # accepted answer を優先
        if self._prefer_accepted and accepted_id:
            for ans in answers:
                if ans.get("answer_id") == accepted_id:
                    return ans

        # スコア順（既にsort済みだが念のため）
        sorted_ans = sorted(answers, key=lambda a: a.get("score", 0), reverse=True)
        return sorted_ans[0] if sorted_ans else None
