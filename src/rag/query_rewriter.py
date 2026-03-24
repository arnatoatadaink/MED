"""src/rag/query_rewriter.py — モデルベース Query Rewriter

CRAG (Corrective RAG) 用にユーザーの質問文から最適な検索クエリを生成する。
複数の戦略を提供し、GUI のチェックボックスで選択可能:

1. rule_expand: ルールベース展開 (QueryExpander)  ← 既存
2. flan_t5_rewrite: FLAN-T5-small による seq2seq クエリ書き換え
3. qwen_rewrite: Qwen2.5-0.5B-Instruct によるクエリ書き換え
4. llm_rewrite: Teacher LLM によるクエリ書き換え (Agentic)  ← 既存

動作モード:
- cascade: 安い順に試し、クエリが生成できたら打ち切り（デフォルト）
- parallel: 選択された全戦略を並列実行し、結果を統合

使い方:
    rewriter = QueryRewriter()
    await rewriter.initialize()

    # カスケード: 安い順に試して最初に成功した戦略で停止
    queries = await rewriter.rewrite(
        "Python で FAISS を使う方法",
        strategies=["rule_expand", "flan_t5_rewrite"],
        mode="cascade",
    )

    # パラレル: 全戦略を実行して統合
    queries = await rewriter.rewrite(
        "Python で FAISS を使う方法",
        strategies=["flan_t5_rewrite", "rule_expand"],
        mode="parallel",
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm.gateway import LLMGateway

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_MODEL_DIR = _PROJECT_ROOT / "data" / "models"


@dataclass
class RewriteResult:
    """クエリ書き換え結果。"""

    strategy: str
    original_query: str
    rewritten_queries: list[str] = field(default_factory=list)
    error: str | None = None


class QueryRewriter:
    """複数戦略によるクエリ書き換えエンジン。

    各戦略は独立しており、任意の組み合わせで使用可能。
    結果は重複除去・統合して返す。

    Args:
        model_dir: ローカルモデルの格納ディレクトリ。
        gateway: LLM Gateway (llm_rewrite 戦略用)。
    """

    # 利用可能な戦略名の一覧（GUI 表示用）
    STRATEGIES: dict[str, str] = {
        "rule_expand": "ルールベース展開",
        "flan_t5_rewrite": "FLAN-T5 Query Rewrite",
        "qwen_rewrite": "Qwen2.5-0.5B Query Rewrite",
        "llm_rewrite": "Teacher LLM Rewrite",
    }

    # カスケード順序: コストが低い順
    CASCADE_ORDER: list[str] = [
        "rule_expand",
        "flan_t5_rewrite",
        "qwen_rewrite",
        "llm_rewrite",
    ]

    def __init__(
        self,
        model_dir: Path | None = None,
        gateway: "LLMGateway | None" = None,
    ) -> None:
        self._model_dir = model_dir or _DEFAULT_MODEL_DIR
        self._gateway = gateway

        # モデルインスタンス (lazy load)
        self._flan_t5_model = None
        self._flan_t5_tokenizer = None
        self._qwen_model = None
        self._qwen_tokenizer = None

        self._initialized = False

    async def initialize(self) -> None:
        """利用可能なモデルを検出する（ロードは遅延）。

        fine-tune 済みモデル (*-crag/) があればそちらを優先する。
        """
        # FLAN-T5: fine-tune 済み → ベースの順で検出
        self._flan_t5_path = self._model_dir / "flan-t5-small-crag"
        if not self._flan_t5_path.exists():
            self._flan_t5_path = self._model_dir / "flan-t5-small"
        self._flan_t5_available = self._flan_t5_path.exists()

        # Qwen: fine-tune 済み → ベースの順で検出
        self._qwen_path = self._model_dir / "Qwen2.5-0.5B-Instruct-crag"
        if not self._qwen_path.exists():
            self._qwen_path = self._model_dir / "Qwen2.5-0.5B-Instruct"
        self._qwen_available = self._qwen_path.exists()

        self._initialized = True
        logger.info(
            "QueryRewriter initialized: flan_t5=%s (%s), qwen=%s (%s), llm=%s",
            self._flan_t5_available, self._flan_t5_path.name,
            self._qwen_available, self._qwen_path.name,
            self._gateway is not None,
        )

    def available_strategies(self) -> dict[str, bool]:
        """各戦略の利用可否を返す。"""
        return {
            "rule_expand": True,  # 常に利用可能
            "flan_t5_rewrite": getattr(self, "_flan_t5_available", False),
            "qwen_rewrite": getattr(self, "_qwen_available", False),
            "llm_rewrite": self._gateway is not None,
        }

    async def rewrite(
        self,
        query: str,
        strategies: list[str] | None = None,
        max_queries_per_strategy: int = 3,
        mode: str = "parallel",
    ) -> list[RewriteResult]:
        """指定された戦略でクエリを書き換える。

        Args:
            query: 元のクエリ文字列。
            strategies: 使用する戦略名のリスト (None = rule_expand のみ)。
            max_queries_per_strategy: 各戦略が返す最大クエリ数。
            mode: 実行モード。
                - "parallel": 全戦略を実行し結果を統合。
                - "cascade": コストが低い順に試し、クエリが生成できたら停止。

        Returns:
            戦略ごとの RewriteResult リスト。
        """
        if strategies is None:
            strategies = ["rule_expand"]

        if mode == "cascade":
            return await self._rewrite_cascade(query, strategies, max_queries_per_strategy)
        return await self._rewrite_parallel(query, strategies, max_queries_per_strategy)

    async def _rewrite_parallel(
        self,
        query: str,
        strategies: list[str],
        max_queries: int,
    ) -> list[RewriteResult]:
        """全戦略を実行して結果を統合する。"""
        results: list[RewriteResult] = []
        for strat in strategies:
            results.append(await self._run_strategy(strat, query, max_queries))
        return results

    async def _rewrite_cascade(
        self,
        query: str,
        strategies: list[str],
        max_queries: int,
    ) -> list[RewriteResult]:
        """コストが低い順に試し、クエリが生成できた時点で停止する。

        CASCADE_ORDER に基づいて strategies をソートし、
        利用可能 かつ 有効なクエリを返した最初の戦略で停止する。
        """
        available = self.available_strategies()
        # strategies を CASCADE_ORDER の順序でソート
        ordered = sorted(
            strategies,
            key=lambda s: self.CASCADE_ORDER.index(s) if s in self.CASCADE_ORDER else 999,
        )

        results: list[RewriteResult] = []
        for strat in ordered:
            if not available.get(strat, False):
                results.append(RewriteResult(
                    strategy=strat,
                    original_query=query,
                    error=f"Strategy not available: {strat}",
                ))
                continue

            result = await self._run_strategy(strat, query, max_queries)
            results.append(result)

            # 有効なクエリが生成できたら停止
            if result.rewritten_queries and not result.error:
                logger.info(
                    "Cascade stopped at %s: generated %d queries",
                    strat, len(result.rewritten_queries),
                )
                break

        return results

    async def _run_strategy(self, strat: str, query: str, max_queries: int) -> RewriteResult:
        """単一戦略を実行する。"""
        if strat == "rule_expand":
            return self._rewrite_rule_expand(query, max_queries)
        elif strat == "flan_t5_rewrite":
            return await self._rewrite_flan_t5(query, max_queries)
        elif strat == "qwen_rewrite":
            return await self._rewrite_qwen(query, max_queries)
        elif strat == "llm_rewrite":
            return await self._rewrite_llm(query, max_queries)
        else:
            return RewriteResult(
                strategy=strat,
                original_query=query,
                error=f"Unknown strategy: {strat}",
            )

    def merge_queries(self, results: list[RewriteResult], include_original: bool = True) -> list[str]:
        """複数戦略の結果を重複除去して統合する。

        Args:
            results: rewrite() の戻り値。
            include_original: 元クエリを先頭に含めるか。

        Returns:
            統合済みクエリリスト（重複除去・順序保持）。
        """
        seen: set[str] = set()
        merged: list[str] = []

        if include_original and results:
            orig = results[0].original_query
            seen.add(orig)
            merged.append(orig)

        for r in results:
            for q in r.rewritten_queries:
                if q not in seen:
                    seen.add(q)
                    merged.append(q)

        return merged

    # ── 戦略実装 ─────────────────────────────────────────────────

    def _rewrite_rule_expand(self, query: str, max_queries: int) -> RewriteResult:
        """ルールベース展開（既存 QueryExpander を利用）。"""
        from src.rag.query_expander import QueryExpander

        try:
            expander = QueryExpander()
            expanded = expander.expand(query)
            # 元クエリと同一のものは除外
            rewritten = [q for q in expanded if q != query][:max_queries]
            return RewriteResult(
                strategy="rule_expand",
                original_query=query,
                rewritten_queries=rewritten,
            )
        except Exception as e:
            logger.exception("rule_expand failed")
            return RewriteResult(
                strategy="rule_expand",
                original_query=query,
                error=str(e),
            )

    async def _rewrite_flan_t5(self, query: str, max_queries: int) -> RewriteResult:
        """FLAN-T5-small による検索クエリ生成。"""
        if not getattr(self, "_flan_t5_available", False):
            return RewriteResult(
                strategy="flan_t5_rewrite",
                original_query=query,
                error="FLAN-T5-small model not found in data/models/flan-t5-small/",
            )

        try:
            model, tokenizer = self._load_flan_t5()

            # 複数のプロンプトバリエーションで多様なクエリを生成
            prompts = [
                f"Generate a search query for: {query}",
                f"Rewrite as a search query: {query}",
                f"What keywords should I search for: {query}",
            ]

            rewritten: list[str] = []
            seen: set[str] = set()

            for prompt in prompts[:max_queries]:
                inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    num_beams=2,
                    early_stopping=True,
                    do_sample=False,
                )
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                if decoded and decoded != query and decoded not in seen:
                    seen.add(decoded)
                    rewritten.append(decoded)

            return RewriteResult(
                strategy="flan_t5_rewrite",
                original_query=query,
                rewritten_queries=rewritten,
            )
        except Exception as e:
            logger.exception("flan_t5_rewrite failed")
            return RewriteResult(
                strategy="flan_t5_rewrite",
                original_query=query,
                error=str(e),
            )

    async def _rewrite_qwen(self, query: str, max_queries: int) -> RewriteResult:
        """Qwen2.5-0.5B-Instruct によるクエリ書き換え。"""
        if not getattr(self, "_qwen_available", False):
            return RewriteResult(
                strategy="qwen_rewrite",
                original_query=query,
                error="Qwen2.5-0.5B-Instruct model not found in data/models/Qwen2.5-0.5B-Instruct/",
            )

        try:
            model, tokenizer = self._load_qwen()

            system_msg = (
                "You are a search query optimizer. Given a user question, generate "
                "a concise, effective search query. Output ONLY the search query, nothing else."
            )

            prompts_user = [
                f"Rewrite as an optimal search query: {query}",
                f"Generate alternative search keywords for: {query}",
            ]

            rewritten: list[str] = []
            seen: set[str] = set()

            for user_prompt in prompts_user[:max_queries]:
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                )
                # 入力部分を除外してデコード
                generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
                decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                if decoded and decoded != query and decoded not in seen:
                    seen.add(decoded)
                    rewritten.append(decoded)

            return RewriteResult(
                strategy="qwen_rewrite",
                original_query=query,
                rewritten_queries=rewritten,
            )
        except Exception as e:
            logger.exception("qwen_rewrite failed")
            return RewriteResult(
                strategy="qwen_rewrite",
                original_query=query,
                error=str(e),
            )

    async def _rewrite_llm(self, query: str, max_queries: int) -> RewriteResult:
        """Teacher LLM によるクエリ書き換え。"""
        if self._gateway is None:
            return RewriteResult(
                strategy="llm_rewrite",
                original_query=query,
                error="LLM Gateway not configured",
            )

        try:
            from src.llm.gateway import LLMMessage

            system_msg = (
                "You are a search query optimizer for a RAG system. "
                "Given a user question, generate up to 3 different search queries "
                "that would retrieve the most relevant documents. "
                "Output one query per line. No numbering, no explanation."
            )
            user_msg = f"Generate search queries for: {query}"

            response = await self._gateway.complete(
                [LLMMessage(role="user", content=user_msg)],
                system=system_msg,
                max_tokens=128,
            )

            rewritten: list[str] = []
            seen: set[str] = set()
            for line in response.content.strip().split("\n"):
                line = line.strip().strip("-").strip("•").strip()
                if line and line != query and line not in seen:
                    seen.add(line)
                    rewritten.append(line)
                if len(rewritten) >= max_queries:
                    break

            return RewriteResult(
                strategy="llm_rewrite",
                original_query=query,
                rewritten_queries=rewritten,
            )
        except Exception as e:
            logger.exception("llm_rewrite failed")
            return RewriteResult(
                strategy="llm_rewrite",
                original_query=query,
                error=str(e),
            )

    # ── モデルロード (lazy) ──────────────────────────────────────

    def _load_flan_t5(self):
        """FLAN-T5 をロードする（初回のみ）。fine-tune 済みがあれば優先。"""
        if self._flan_t5_model is not None:
            return self._flan_t5_model, self._flan_t5_tokenizer

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_path = str(getattr(self, "_flan_t5_path", self._model_dir / "flan-t5-small"))
        logger.info("Loading FLAN-T5 from %s", model_path)
        self._flan_t5_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._flan_t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self._flan_t5_model.eval()
        return self._flan_t5_model, self._flan_t5_tokenizer

    def _load_qwen(self):
        """Qwen をロードする（初回のみ）。fine-tune 済みがあれば優先。"""
        if self._qwen_model is not None:
            return self._qwen_model, self._qwen_tokenizer

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = str(getattr(self, "_qwen_path", self._model_dir / "Qwen2.5-0.5B-Instruct"))
        logger.info("Loading Qwen from %s", model_path)
        self._qwen_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._qwen_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype="auto",
        )
        self._qwen_model.eval()
        return self._qwen_model, self._qwen_tokenizer
