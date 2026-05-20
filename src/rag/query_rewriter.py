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

外部プロバイダー (LMStudio 等):
- 環境変数 FLAN_T5_PROVIDER_URL / QWEN_PROVIDER_URL を設定すると
  OpenAI 互換 /v1/chat/completions を使用する。
- 10 秒以内に有効なレスポンスが得られない場合はローカルモデルへ自動フォールバック。

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

import json
import logging
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm.gateway import LLMGateway

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_MODEL_DIR = _PROJECT_ROOT / "data" / "models"
_PROVIDER_PROBE_TIMEOUT = 10  # 秒: プロバイダー疎通確認のタイムアウト


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

        # 外部プロバイダー URL (疎通確認済み)
        self._flan_t5_provider_url: str | None = None
        self._flan_t5_provider_model: str = "gguf-flan-t5-small"
        self._qwen_provider_url: str | None = None
        self._qwen_provider_model: str = "qwen2.5-0.5b-instruct"

        self._initialized = False

    async def initialize(self) -> None:
        """利用可能なモデルを検出する（ロードは遅延）。

        fine-tune 済みモデル (*-crag/) があればそちらを優先する。
        外部プロバイダーが設定されている場合は 10 秒以内に疎通確認を行い、
        失敗時はローカルモデルへフォールバックする。
        """
        from src.common.config import get_settings

        cfg = get_settings().query_rewriter

        # ── FLAN-T5 プロバイダー確認 ─────────────────────────────────────
        flan_t5_url = os.environ.get("FLAN_T5_PROVIDER_URL") or cfg.flan_t5_provider_url
        flan_t5_model = os.environ.get("FLAN_T5_PROVIDER_MODEL") or cfg.flan_t5_provider_model
        if flan_t5_url:
            if self._probe_llm_provider(flan_t5_url, flan_t5_model):
                self._flan_t5_provider_url = flan_t5_url.rstrip("/")
                self._flan_t5_provider_model = flan_t5_model
                logger.info("FLAN-T5: using external provider %s (model=%s)", self._flan_t5_provider_url, flan_t5_model)
            else:
                logger.warning(
                    "FLAN-T5 provider %s unavailable or returned empty — falling back to local model",
                    flan_t5_url,
                )

        # ── Qwen プロバイダー確認 ─────────────────────────────────────────
        qwen_url = os.environ.get("QWEN_PROVIDER_URL") or cfg.qwen_provider_url
        qwen_model = os.environ.get("QWEN_PROVIDER_MODEL") or cfg.qwen_provider_model
        if qwen_url:
            if self._probe_llm_provider(qwen_url, qwen_model):
                self._qwen_provider_url = qwen_url.rstrip("/")
                self._qwen_provider_model = qwen_model
                logger.info("Qwen: using external provider %s (model=%s)", self._qwen_provider_url, qwen_model)
            else:
                logger.warning(
                    "Qwen provider %s unavailable or returned empty — falling back to local model",
                    qwen_url,
                )

        # ── ローカルモデル検出（プロバイダーが未設定または失敗した場合） ──
        if not self._flan_t5_provider_url:
            self._flan_t5_path = self._model_dir / "flan-t5-small-crag"
            if not self._flan_t5_path.exists():
                self._flan_t5_path = self._model_dir / "flan-t5-small"
            self._flan_t5_available = self._flan_t5_path.exists()
        else:
            self._flan_t5_available = True  # プロバイダー経由で利用可能
            self._flan_t5_path = self._model_dir / "flan-t5-small"  # ダミー（未使用）

        if not self._qwen_provider_url:
            self._qwen_path = self._model_dir / "Qwen2.5-0.5B-Instruct-crag"
            if not self._qwen_path.exists():
                self._qwen_path = self._model_dir / "Qwen2.5-0.5B-Instruct"
            self._qwen_available = self._qwen_path.exists()
        else:
            self._qwen_available = True  # プロバイダー経由で利用可能
            self._qwen_path = self._model_dir / "Qwen2.5-0.5B-Instruct"  # ダミー（未使用）

        self._initialized = True
        logger.info(
            "QueryRewriter initialized: flan_t5=%s (provider=%s), qwen=%s (provider=%s), llm=%s",
            self._flan_t5_available,
            "http" if self._flan_t5_provider_url else "local",
            self._qwen_available,
            "http" if self._qwen_provider_url else "local",
            self._gateway is not None,
        )

    def _probe_llm_provider(self, url: str, model: str) -> bool:
        """プロバイダーに短いテスト補完を送って疎通・非空レスポンスを確認する (10 秒タイムアウト)。"""
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 8,
        }).encode()
        req = urllib.request.Request(
            f"{url.rstrip('/')}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=_PROVIDER_PROBE_TIMEOUT) as resp:
                data = json.loads(resp.read())
            content = data["choices"][0]["message"]["content"]
            if not content or not content.strip():
                logger.warning("Provider %s (model=%s) returned empty content", url, model)
                return False
            return True
        except Exception as exc:
            logger.warning("Provider probe failed (%s, model=%s): %s", url, model, exc)
            return False

    def _http_chat_complete(
        self,
        provider_url: str,
        model: str,
        messages: list[dict],
        max_tokens: int = 64,
    ) -> str:
        """OpenAI 互換 /v1/chat/completions へリクエストを送り、生成テキストを返す。"""
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }).encode()
        req = urllib.request.Request(
            f"{provider_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()

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
        provider: str | None = None,
        timeout: float | None = None,
    ) -> list[RewriteResult]:
        """指定された戦略でクエリを書き換える。

        Args:
            query: 元のクエリ文字列。
            strategies: 使用する戦略名のリスト (None = rule_expand のみ)。
            max_queries_per_strategy: 各戦略が返す最大クエリ数。
            mode: 実行モード。
                - "parallel": 全戦略を実行し結果を統合。
                - "cascade": コストが低い順に試し、クエリが生成できたら停止。
            provider: LLM プロバイダ名（llm_rewrite 戦略で使用）。
            timeout: リクエストタイムアウト秒数。

        Returns:
            戦略ごとの RewriteResult リスト。
        """
        if strategies is None:
            strategies = ["rule_expand"]

        if mode == "cascade":
            return await self._rewrite_cascade(query, strategies, max_queries_per_strategy, provider=provider, timeout=timeout)
        return await self._rewrite_parallel(query, strategies, max_queries_per_strategy, provider=provider, timeout=timeout)

    async def _rewrite_parallel(
        self,
        query: str,
        strategies: list[str],
        max_queries: int,
        provider: str | None = None,
        timeout: float | None = None,
    ) -> list[RewriteResult]:
        """全戦略を実行して結果を統合する。"""
        results: list[RewriteResult] = []
        for strat in strategies:
            results.append(await self._run_strategy(strat, query, max_queries, provider=provider, timeout=timeout))
        return results

    async def _rewrite_cascade(
        self,
        query: str,
        strategies: list[str],
        max_queries: int,
        provider: str | None = None,
        timeout: float | None = None,
    ) -> list[RewriteResult]:
        """コストが低い順に試し、クエリが生成できた時点で停止する。

        CASCADE_ORDER に基づいて strategies をソートし、
        利用可能 かつ 有効なクエリを返した最初の戦略で停止する。
        """
        available = self.available_strategies()
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

            result = await self._run_strategy(strat, query, max_queries, provider=provider, timeout=timeout)
            results.append(result)

            if result.rewritten_queries and not result.error:
                logger.info(
                    "Cascade stopped at %s: generated %d queries",
                    strat, len(result.rewritten_queries),
                )
                break

        return results

    async def _run_strategy(
        self, strat: str, query: str, max_queries: int,
        provider: str | None = None, timeout: float | None = None,
    ) -> RewriteResult:
        """単一戦略を実行する。"""
        if strat == "rule_expand":
            return self._rewrite_rule_expand(query, max_queries)
        elif strat == "flan_t5_rewrite":
            return await self._rewrite_flan_t5(query, max_queries)
        elif strat == "qwen_rewrite":
            return await self._rewrite_qwen(query, max_queries)
        elif strat == "llm_rewrite":
            return await self._rewrite_llm(query, max_queries, provider=provider, timeout=timeout)
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
        """FLAN-T5-small による検索クエリ生成。外部プロバイダー優先、ローカルへフォールバック。"""
        if not getattr(self, "_flan_t5_available", False):
            return RewriteResult(
                strategy="flan_t5_rewrite",
                original_query=query,
                error="FLAN-T5-small model not found in data/models/flan-t5-small/",
            )

        # 外部プロバイダー経由
        if self._flan_t5_provider_url:
            return self._http_flan_t5_rewrite(query, max_queries)

        # ローカルモデル
        try:
            model, tokenizer = self._load_flan_t5()

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

    def _http_flan_t5_rewrite(self, query: str, max_queries: int) -> RewriteResult:
        """外部プロバイダーを使った FLAN-T5 クエリ書き換え。"""
        prompts = [
            f"Generate a search query for: {query}",
            f"Rewrite as a search query: {query}",
            f"What keywords should I search for: {query}",
        ]
        rewritten: list[str] = []
        seen: set[str] = set()

        try:
            for prompt in prompts[:max_queries]:
                messages = [{"role": "user", "content": prompt}]
                decoded = self._http_chat_complete(
                    self._flan_t5_provider_url,
                    self._flan_t5_provider_model,
                    messages,
                    max_tokens=64,
                )
                if decoded and decoded != query and decoded not in seen:
                    seen.add(decoded)
                    rewritten.append(decoded)

            return RewriteResult(
                strategy="flan_t5_rewrite",
                original_query=query,
                rewritten_queries=rewritten,
            )
        except Exception as e:
            logger.exception("flan_t5_rewrite (http) failed")
            return RewriteResult(
                strategy="flan_t5_rewrite",
                original_query=query,
                error=str(e),
            )

    async def _rewrite_qwen(self, query: str, max_queries: int) -> RewriteResult:
        """Qwen2.5-0.5B-Instruct によるクエリ書き換え。外部プロバイダー優先、ローカルへフォールバック。"""
        if not getattr(self, "_qwen_available", False):
            return RewriteResult(
                strategy="qwen_rewrite",
                original_query=query,
                error="Qwen2.5-0.5B-Instruct model not found in data/models/Qwen2.5-0.5B-Instruct/",
            )

        # 外部プロバイダー経由
        if self._qwen_provider_url:
            return self._http_qwen_rewrite(query, max_queries)

        # ローカルモデル
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

    def _http_qwen_rewrite(self, query: str, max_queries: int) -> RewriteResult:
        """外部プロバイダーを使った Qwen クエリ書き換え。"""
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

        try:
            for user_prompt in prompts_user[:max_queries]:
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt},
                ]
                decoded = self._http_chat_complete(
                    self._qwen_provider_url,
                    self._qwen_provider_model,
                    messages,
                    max_tokens=64,
                )
                if decoded and decoded != query and decoded not in seen:
                    seen.add(decoded)
                    rewritten.append(decoded)

            return RewriteResult(
                strategy="qwen_rewrite",
                original_query=query,
                rewritten_queries=rewritten,
            )
        except Exception as e:
            logger.exception("qwen_rewrite (http) failed")
            return RewriteResult(
                strategy="qwen_rewrite",
                original_query=query,
                error=str(e),
            )

    async def _rewrite_llm(
        self, query: str, max_queries: int,
        provider: str | None = None, timeout: float | None = None,
    ) -> RewriteResult:
        """Teacher LLM によるクエリ書き換え。"""
        if self._gateway is None:
            return RewriteResult(
                strategy="llm_rewrite",
                original_query=query,
                error="LLM Gateway not configured",
            )

        try:
            system_msg = (
                "You are a search query optimizer for a RAG system. "
                "Given a user question, generate up to 3 different search queries "
                "that would retrieve the most relevant documents. "
                "Output one query per line. No numbering, no explanation."
            )
            user_msg = f"Generate search queries for: {query}"

            response = await self._gateway.complete(
                user_msg,
                system=system_msg,
                provider=provider,
                max_tokens=512,
                timeout=timeout,
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
