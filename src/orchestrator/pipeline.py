"""src/orchestrator/pipeline.py — MED RAG パイプライン

クエリ処理の全フローを統合する:
1. FAISS 検索（MemoryManager）
2. 外部 RAG 検索（RetrieverRouter）→ LLM 裏どり → チャンク化 → FAISS 保存
3. LLM レスポンス生成（ResponseGenerator）
4. コード実行（SandboxManager）
5. フィードバック記録

使い方:
    from src.orchestrator.pipeline import MEDPipeline

    pipeline = MEDPipeline()
    await pipeline.initialize()
    response = await pipeline.query("Python でリストをソートするには？")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.conversation.manager import ConversationManager
from src.llm.code_generator import CodeGenerator, CodeResult
from src.llm.gateway import LLMGateway, LLMMessage
from src.llm.response_generator import GeneratedResponse, ResponseGenerator
from src.memory.iterative_retrieval import IterativeRetriever
from src.memory.memory_manager import MemoryManager
from src.memory.schema import SearchResult
from src.rag.chunker import Chunker
from src.rag.query_expander import QueryExpander
from src.rag.query_rewriter import QueryRewriter
from src.rag.retriever import RetrieverRouter
from src.rag.verifier import ResultVerifier
from src.sandbox.manager import SandboxManager

logger = logging.getLogger(__name__)


@dataclass
class QueryResponse:
    """パイプラインの最終レスポンス。"""

    answer: str
    query: str
    faiss_results: list[SearchResult] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    code_result: CodeResult | None = None
    sandbox_stdout: str = ""
    sandbox_success: bool = False
    context_doc_ids: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    debug_info: dict = field(default_factory=dict)
    session_id: str | None = None       # 会話セッション ID
    user_id: str = "default"            # ユーザー識別子


class MEDPipeline:
    """MED RAG システムの統合パイプライン。

    Args:
        memory_manager: MemoryManager（省略時は自動生成）。
        gateway: LLMGateway（省略時は自動生成）。
        retriever_router: RetrieverRouter（省略時は自動生成）。
        sandbox_manager: SandboxManager（省略時は自動生成）。
        enable_external_rag: 外部 RAG 検索を有効にするか。
        enable_sandbox: コード実行を有効にするか。
        enable_iterative: Iterative Retrieval を有効にするか。
    """

    def __init__(
        self,
        memory_manager: MemoryManager | None = None,
        gateway: LLMGateway | None = None,
        retriever_router: RetrieverRouter | None = None,
        sandbox_manager: SandboxManager | None = None,
        conversation_manager: ConversationManager | None = None,
        *,
        enable_external_rag: bool = True,
        enable_sandbox: bool = True,
        enable_iterative: bool = False,
    ) -> None:
        self._mm = memory_manager or MemoryManager()
        self._gateway = gateway or LLMGateway()
        self._router = retriever_router or RetrieverRouter()
        self._sandbox = sandbox_manager or SandboxManager()
        self._conv = conversation_manager   # None = 会話履歴機能を無効化
        self._chunker = Chunker()
        self._verifier = ResultVerifier(gateway=self._gateway)
        self._response_gen = ResponseGenerator(self._gateway)
        self._code_gen = CodeGenerator(self._gateway)
        self._iterative = IterativeRetriever(self._mm, self._mm.embedder)
        self._expander = QueryExpander()
        self._rewriter = QueryRewriter(gateway=self._gateway)

        self._enable_rag = enable_external_rag
        self._enable_sandbox = enable_sandbox
        self._enable_iterative = enable_iterative
        self._initialized = False

    async def initialize(self) -> None:
        """パイプラインを初期化する。"""
        await self._mm.initialize()
        await self._rewriter.initialize()
        if self._conv is not None:
            await self._conv.initialize()
        self._initialized = True
        logger.info("MEDPipeline initialized")

    async def close(self) -> None:
        """リソースを解放する。"""
        await self._mm.close()
        if self._conv is not None:
            await self._conv.close()
        self._initialized = False

    @property
    def rewriter(self) -> QueryRewriter:
        """QueryRewriter インスタンスを返す（GUI から利用可否を取得するため）。"""
        return self._rewriter

    async def query(
        self,
        query: str,
        domain: str | None = None,
        k: int = 5,
        run_code: bool = False,
        provider: str | None = None,
        model: str | None = None,
        use_memory: bool = True,
        use_rag: bool = True,
        session_id: str | None = None,
        user_id: str = "default",
        timeout: float | None = None,
        crag_strategies: list[str] | None = None,
        crag_mode: str = "cascade",
    ) -> QueryResponse:
        """クエリを受け取り、RAG + LLM で回答を生成する。

        Args:
            query: ユーザーのクエリ。
            domain: 検索ドメイン（None = 全ドメイン）。
            k: FAISS から取得するドキュメント数。
            run_code: 生成コードを Sandbox で実行するか。
            session_id: 会話セッション ID（ConversationManager が有効な場合に使用）。
            user_id: ユーザー識別子（ユーザー FAISS の選択に使用）。

        Returns:
            QueryResponse オブジェクト。
        """
        if not self._initialized:
            raise RuntimeError("MEDPipeline.initialize() must be called before use")

        # ── Step 0: 会話履歴からコンテキストを取得 ───────
        conv_messages: list[LLMMessage] = []
        if self._conv is not None and session_id:
            turns = await self._conv.get_context_turns(session_id)
            for t in turns:
                conv_messages.append(LLMMessage(role=t.role, content=t.content))
            logger.debug(
                "Conversation context: %d turns for session=%s", len(turns), session_id
            )

        # ── Step 1: FAISS 検索（ハイブリッド: ベクトル + エイリアスキーワード） ──
        faiss_results: list[SearchResult] = []
        if use_memory:
            if self._enable_iterative:
                faiss_results = await self._iterative.retrieve(
                    query, domain=domain, max_rounds=3, k_per_round=k, strategy="vector_add"
                )
            else:
                faiss_results = await self._mm.search_hybrid(query, domain=domain, k=k)

        logger.debug("FAISS: %d results for query=%r", len(faiss_results), query[:50])

        # ── Step 2: 外部 RAG 検索 → 保存 ───────────
        rag_raw_results: list = []
        if use_rag and self._enable_rag:
            _, rag_raw_results = await self._fetch_and_store_external(
                query, domain=domain or "general", provider=provider
            )

        # ── Step 3: LLM レスポンス生成 ──────────────
        gen_response: GeneratedResponse = await self._response_gen.generate(
            query,
            context_docs=faiss_results,
            provider=provider,
            model=model,
            conversation_history=conv_messages if conv_messages else None,
            timeout=timeout,
        )

        # ── Step 3.5: CRAG リトライ ─────────────────
        # トリガー条件（いずれか）:
        #   (A) LLM 回答にネガティブシグナルが含まれる
        #   (B) FAISS 結果が閾値未満 かつ クエリが複合形（expand が新クエリを生成）
        retry_triggered = False
        expanded_queries: list[str] = []
        retry_faiss_results: list[SearchResult] = []
        retry_rag_results: list = []
        rewriter_details: list[dict] = []   # デバッグ用: 各戦略の結果

        if use_rag and self._enable_rag:
            # ── CRAG 戦略によるクエリ生成 ──
            # crag_strategies が指定されている場合、QueryRewriter で追加クエリを生成
            active_strategies = crag_strategies or []
            rewriter_queries: list[str] = []

            # QueryRewriter で全戦略を実行（cascade/parallel はモード指定に従う）
            if active_strategies:
                rw_results = await self._rewriter.rewrite(
                    query, strategies=active_strategies, mode=crag_mode,
                )
                for rr in rw_results:
                    detail = {
                        "strategy": rr.strategy,
                        "queries": rr.rewritten_queries,
                        "error": rr.error,
                    }
                    rewriter_details.append(detail)
                    rewriter_queries.extend(rr.rewritten_queries)

            # rule_expand が戦略に含まれない場合もフォールバック展開を試みる
            if "rule_expand" not in active_strategies:
                expanded = self._expander.expand(query)
                rule_new_queries = [q for q in expanded if q != query]
            else:
                rule_new_queries = []  # rule_expand は rewriter 経由で実行済み

            # 全戦略のクエリを統合（重複除去）
            all_new_queries = list(dict.fromkeys(rewriter_queries + rule_new_queries))

            is_negative = self._expander.is_negative(gen_response.answer)
            min_faiss = self._expander.crag_min_faiss
            is_sparse = min_faiss > 0 and len(faiss_results) < min_faiss

            # リトライ判定: ルールベースの新クエリ or モデルベースの新クエリがある場合
            should_retry = bool(all_new_queries) and (is_negative or is_sparse)

            if should_retry:
                retry_triggered = True
                expanded_queries = [query] + all_new_queries
                trigger_reason = "negative_signal" if is_negative else "sparse_faiss"
                logger.info(
                    "CRAG retry triggered (reason=%s, strategies=%s, faiss_count=%d) "
                    "for query=%r → %s",
                    trigger_reason, active_strategies, len(faiss_results),
                    query[:60], expanded_queries,
                )
                # 展開クエリごとに外部 RAG を追加取得
                for sub_q in all_new_queries:
                    _, sub_raw = await self._fetch_and_store_external(
                        sub_q,
                        domain=domain or "general",
                        provider=provider,
                        max_results=self._expander.retry_max_results,
                    )
                    retry_rag_results.extend(sub_raw)

                # FAISS を再検索（新規ドキュメントが追加されているので結果が変わる可能性あり）
                retry_faiss_results = await self._mm.search(query, domain=domain, k=k)

                # LLM を再実行
                gen_response = await self._response_gen.generate(
                    query, context_docs=retry_faiss_results, provider=provider, model=model,
                    timeout=timeout,
                )
                logger.debug(
                    "CRAG retry complete: faiss=%d docs, rag_new=%d docs",
                    len(retry_faiss_results), len(retry_rag_results),
                )

        # ── Step 3.6: Agentic 1-step（CRAG 後もまだ否定的なら LLM に代替クエリを提案させる）
        agentic_triggered = False
        agentic_query: str = ""
        agentic_rag_results: list = []

        if (
            use_rag and self._enable_rag
            and self._expander.is_negative(gen_response.answer)
        ):
            suggested = await self._suggest_search_query(
                query, gen_response.answer, provider=provider, model=model
            )
            if suggested and suggested != query:
                agentic_triggered = True
                agentic_query = suggested
                logger.info(
                    "Agentic retry: LLM suggested query=%r for original=%r",
                    suggested[:60], query[:60],
                )
                _, agentic_rag_results = await self._fetch_and_store_external(
                    suggested,
                    domain=domain or "general",
                    provider=provider,
                    max_results=self._expander.retry_max_results,
                )
                agentic_faiss = await self._mm.search_hybrid(query, domain=domain, k=k)
                gen_response = await self._response_gen.generate(
                    query, context_docs=agentic_faiss, provider=provider, model=model,
                    timeout=timeout,
                )
                logger.debug(
                    "Agentic retry complete: new_faiss=%d new_rag=%d",
                    len(agentic_faiss), len(agentic_rag_results),
                )

        # ── Step 4: コード生成 + Sandbox 実行 ─────
        code_result: CodeResult | None = None
        sandbox_stdout = ""
        sandbox_success = False

        if run_code and self._enable_sandbox:
            try:
                code_result = await self._code_gen.generate(task=query)
                if code_result.is_complete:
                    sandbox_run = await self._sandbox.run(code_result.code)
                    sandbox_stdout = sandbox_run.stdout
                    sandbox_success = sandbox_run.success
            except Exception:
                logger.exception("Code generation/execution failed")

        # ── Step 5: フィードバック記録 ──────────────
        for sr in faiss_results:
            try:
                await self._mm.record_retrieval(sr.document.id)
            except Exception:
                logger.warning("Failed to record retrieval for doc=%s", sr.document.id)

        # ── デバッグ情報 ──────────────────────────
        def _sr_to_dict(sr: SearchResult) -> dict:
            return {
                "id": sr.document.id,
                "content": sr.document.content[:300],
                "score": round(sr.score, 4),
                "domain": sr.document.domain,
                "source": sr.document.source.url or "",
            }

        def _raw_to_dict(r: object) -> dict:
            return {
                "title": getattr(r, "title", ""),
                "content": getattr(r, "content", "")[:300],
                "url": getattr(r, "url", ""),
                "source": getattr(r, "source", ""),
                "score": round(getattr(r, "score", 0.0), 4),
            }

        debug_info: dict = {
            "faiss_query": query,
            "faiss_results": [_sr_to_dict(sr) for sr in faiss_results],
            "rag_query": query,
            "rag_results": [_raw_to_dict(r) for r in rag_raw_results],
            "retry_triggered": retry_triggered,
            "agentic_triggered": agentic_triggered,
        }
        if retry_triggered:
            debug_info["expanded_queries"] = expanded_queries
            debug_info["retry_faiss_results"] = [_sr_to_dict(sr) for sr in retry_faiss_results]
            debug_info["retry_rag_results"] = [_raw_to_dict(r) for r in retry_rag_results]
            debug_info["crag_strategies"] = crag_strategies or []
            debug_info["crag_mode"] = crag_mode
            debug_info["rewriter_details"] = rewriter_details
        if agentic_triggered:
            debug_info["agentic_query"] = agentic_query
            debug_info["agentic_rag_results"] = [_raw_to_dict(r) for r in agentic_rag_results]

        # ── Step N: 会話履歴にターンを保存 ──────────
        if self._conv is not None and session_id:
            import asyncio
            user_turn = self._conv.make_user_turn(session_id, query)
            assistant_turn = self._conv.make_assistant_turn(
                session_id,
                gen_response.answer,
                provider=gen_response.provider,
                model=gen_response.model,
                input_tokens=gen_response.input_tokens,
                output_tokens=gen_response.output_tokens,
            )
            await self._conv.add_turn(user_turn)
            await self._conv.add_turn(assistant_turn)
            # FAISS 登録は非同期でバックグラウンド実行（レスポンス遅延を防ぐ）
            asyncio.ensure_future(
                self._conv.save_to_user_faiss(
                    assistant_turn, user_id, lambda uid: self._mm
                )
            )
            logger.debug(
                "Saved conversation turns for session=%s user=%s", session_id, user_id
            )

        return QueryResponse(
            answer=gen_response.answer,
            query=query,
            faiss_results=faiss_results,
            provider=gen_response.provider,
            model=gen_response.model,
            code_result=code_result,
            sandbox_stdout=sandbox_stdout,
            sandbox_success=sandbox_success,
            context_doc_ids=gen_response.context_doc_ids,
            input_tokens=gen_response.input_tokens,
            output_tokens=gen_response.output_tokens,
            debug_info=debug_info,
            session_id=session_id,
            user_id=user_id,
        )

    async def add_document(
        self,
        content: str,
        domain: str = "general",
        teacher_id: str | None = None,
    ) -> str:
        """単一ドキュメントをメモリに追加する（ユーティリティメソッド）。

        Args:
            content:    ドキュメントテキスト。
            domain:     ドメイン分類。
            teacher_id: Teacher モデル識別子（素性追跡用）。
        """
        return await self._mm.add_from_text(
            content=content,
            domain=domain,
            source_type="manual",
            teacher_id=teacher_id,
        )

    async def _suggest_search_query(
        self,
        original_query: str,
        failed_answer: str,
        provider: str | None = None,
        model: str | None = None,
    ) -> str:
        """LLM に代替検索クエリを提案させる（Agentic 1-step）。

        CRAG リトライ後も回答が否定的な場合、LLM に「どんなキーワードで
        再検索すれば情報が見つかるか」を尋ねる。

        Returns:
            代替クエリ文字列。生成失敗時は空文字列。
        """
        _SUGGEST_SYSTEM = (
            "You are a search query optimizer. "
            "Given a question and a failed search attempt, suggest ONE alternative "
            "search query that might find relevant documents. "
            "Output ONLY the query string, nothing else. No explanation, no quotes."
        )
        _SUGGEST_PROMPT = (
            f"Original question: {original_query}\n\n"
            f"Previous answer (insufficient): {failed_answer[:200]}\n\n"
            "Suggest one alternative search query to find better information:"
        )
        try:
            from src.llm.gateway import LLMMessage
            response = await self._gateway.complete(
                [LLMMessage(role="user", content=_SUGGEST_PROMPT)],
                system=_SUGGEST_SYSTEM,
                provider=provider,
                model=model,
                max_tokens=64,
            )
            suggested = response.content.strip().strip('"').strip("'")
            return suggested if len(suggested) >= 3 else ""
        except Exception:
            logger.debug("Agentic query suggestion failed (non-fatal)")
            return ""

    async def _fetch_and_store_external(
        self,
        query: str,
        domain: str,
        provider: str | None = None,
        max_results: int = 10,
    ) -> tuple[int, list]:
        """外部 RAG 検索 → 裏どり → FAISS 保存。

        Args:
            query:       検索クエリ。
            domain:      FAISS 保存ドメイン。
            provider:    LLM プロバイダー（Verifier 用）。
            max_results: 取得する最大件数（リトライ時は小さくする）。

        Returns:
            (stored_count, raw_results) — raw_results はデバッグ用の生検索結果。
        """
        try:
            raw_results = await self._router.search(query, max_results=max_results)
            if not raw_results:
                return 0, []

            # チャットで選択したプロバイダーを Verifier に伝播（未設定なら Verifier 側で可否判断）
            verified = await self._verifier.verify(raw_results, query=query, provider=provider)
            docs = self._chunker.chunk_results(verified, domain=domain)

            added_ids = await self._mm.add_batch(docs)

            # ── 生結果を raw_results テーブルへ保存（FAISS とは独立） ──
            # チャンク化前の原文・URL・スコアをそのまま保存するため、
            # 後から原文参照・BI 集計・FAISS 非依存のキーワード検索が可能。
            await self._store_raw_results(query, raw_results, domain, added_ids)

            logger.debug("External RAG: stored %d new docs", len(added_ids))
            return len(added_ids), raw_results
        except Exception:
            logger.exception("External RAG fetch failed")
            return 0, []

    async def _store_raw_results(
        self,
        query: str,
        raw_results: list,
        domain: str,
        chunked_doc_ids: list[str],
    ) -> None:
        """生検索結果を raw_results テーブルに保存する（ベストエフォート）。

        URL を ID として使うため同一ページの重複保存は ON CONFLICT で上書きされる。
        """
        import hashlib
        try:
            for r in raw_results:
                url = getattr(r, "url", "") or ""
                result_id = hashlib.sha256(url.encode()).hexdigest()[:32] if url else ""
                if not result_id:
                    continue
                await self._mm.store.store_raw_result(
                    result_id=result_id,
                    query=query,
                    title=getattr(r, "title", ""),
                    content=getattr(r, "content", ""),
                    url=url,
                    source=getattr(r, "source", ""),
                    score=float(getattr(r, "score", 0.0)),
                    domain=domain,
                    doc_ids=chunked_doc_ids,
                )
        except Exception:
            logger.debug("raw_results storage failed (non-fatal)")
