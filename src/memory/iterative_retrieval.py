"""src/memory/iterative_retrieval.py — マルチホップ Iterative Retrieval

複数ラウンドの検索で文脈を積み上げながら関連ドキュメントを収集する。
2 つの戦略を実装:

1. **ベクトル加算 (vector_add)**: 検索済みドキュメントの埋め込みをクエリに加算して次ラウンドへ。
   LLM 不要で高速。
2. **LLM リライト (llm_rewrite)**: 各ラウンドの結果を踏まえて LLM がクエリを再生成。
   精度が高いが LLM API コストが発生する。

HyDE (Hypothetical Document Embeddings):
   LLM に「このクエリの回答として想定されるドキュメント」を生成させ、
   その埋め込みを検索クエリとして使用する。

使い方:
    from src.memory.iterative_retrieval import IterativeRetriever

    retriever = IterativeRetriever(memory_manager, embedder)
    results = await retriever.retrieve("マルチホップが必要なクエリ", max_rounds=3, k_per_round=5)
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from src.memory.schema import SearchResult

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMCallable(Protocol):
    """LLM 呼び出しプロトコル（テスト時にモック可能）。"""

    async def complete(self, prompt: str) -> str:
        """プロンプトを送り、テキスト応答を返す。"""
        ...


class IterativeRetriever:
    """マルチホップ Iterative Retrieval。

    Args:
        memory_manager: MemoryManager インスタンス（search メソッドを持つ）。
        embedder: Embedder インスタンス（embed メソッドを持つ）。
        llm: LLM 呼び出しオブジェクト（llm_rewrite 戦略を使う場合に必要）。
        dedup: 結果の重複除去を行うか。
    """

    def __init__(
        self,
        memory_manager,  # MemoryManager (avoid circular import)
        embedder,        # Embedder
        llm: Optional[LLMCallable] = None,
        dedup: bool = True,
    ) -> None:
        self._mm = memory_manager
        self._embedder = embedder
        self._llm = llm
        self._dedup = dedup

    async def retrieve(
        self,
        query: str,
        domain: Optional[str] = None,
        max_rounds: int = 3,
        k_per_round: int = 5,
        strategy: str = "vector_add",
    ) -> list[SearchResult]:
        """Iterative Retrieval を実行する。

        Args:
            query: 初期クエリ文字列。
            domain: 検索ドメイン (None = 全ドメイン)。
            max_rounds: 最大検索ラウンド数。
            k_per_round: 1 ラウンドあたりの取得件数。
            strategy: 検索戦略。
                - "vector_add": ベクトル加算（LLM 不要、高速）
                - "llm_rewrite": LLM クエリ再生成（高精度）
                - "hyde": Hypothetical Document Embedding

        Returns:
            重複除去・スコア降順の SearchResult リスト。
        """
        if strategy == "vector_add":
            return await self._retrieve_vector_add(query, domain, max_rounds, k_per_round)
        elif strategy == "llm_rewrite":
            return await self._retrieve_llm_rewrite(query, domain, max_rounds, k_per_round)
        elif strategy == "hyde":
            return await self._retrieve_hyde(query, domain, k_per_round)
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}. Use 'vector_add', 'llm_rewrite', or 'hyde'.")

    # ------------------------------------------------------------------
    # 戦略 1: ベクトル加算
    # ------------------------------------------------------------------

    async def _retrieve_vector_add(
        self,
        query: str,
        domain: Optional[str],
        max_rounds: int,
        k_per_round: int,
    ) -> list[SearchResult]:
        """ベクトル加算マルチホップ検索。

        各ラウンドで取得したドキュメントの埋め込みをクエリに加算（累積平均）し、
        次ラウンドの検索クエリとする。
        """
        query_vec: NDArray[np.float32] = self._embedder.embed(query)
        all_results: list[SearchResult] = []
        seen_ids: set[str] = set()

        for round_idx in range(max_rounds):
            results = await self._mm.search(
                query="",  # dummy — override with vec below
                domain=domain,
                k=k_per_round,
            )
            # MemoryManager.search が文字列を受け取るため、埋め込みベクトルで直接検索
            results = await self._search_by_vec(query_vec, domain, k_per_round)

            if not results:
                logger.debug("Round %d: no results, stopping", round_idx)
                break

            # 新規ドキュメントのみ追加
            new_results = []
            for r in results:
                if r.document.id not in seen_ids:
                    seen_ids.add(r.document.id)
                    new_results.append(r)
                    all_results.append(r)

            if not new_results:
                logger.debug("Round %d: all results already seen, stopping", round_idx)
                break

            # クエリベクトルを更新: 新規ドキュメントの平均埋め込みを加算
            new_embeddings = []
            for r in new_results:
                if r.document.embedding is not None:
                    new_embeddings.append(r.document.embedding)
                else:
                    # 埋め込みがなければ content から生成
                    new_embeddings.append(self._embedder.embed(r.document.content))

            if new_embeddings:
                centroid = np.mean(new_embeddings, axis=0).astype(np.float32)
                # クエリと重心の加重平均（探索の広がりを制御）
                alpha = 0.7  # クエリの重み
                query_vec = alpha * query_vec + (1.0 - alpha) * centroid
                # L2 正規化（内積 = コサイン類似度のため）
                norm = np.linalg.norm(query_vec)
                if norm > 0:
                    query_vec = query_vec / norm

            logger.debug("Round %d: collected %d new docs (total=%d)", round_idx, len(new_results), len(all_results))

        return self._sort_and_dedup(all_results) if self._dedup else all_results

    # ------------------------------------------------------------------
    # 戦略 2: LLM リライト
    # ------------------------------------------------------------------

    async def _retrieve_llm_rewrite(
        self,
        query: str,
        domain: Optional[str],
        max_rounds: int,
        k_per_round: int,
    ) -> list[SearchResult]:
        """LLM クエリ再生成マルチホップ検索。"""
        if self._llm is None:
            logger.warning("llm_rewrite requires LLM; falling back to vector_add")
            return await self._retrieve_vector_add(query, domain, max_rounds, k_per_round)

        current_query = query
        all_results: list[SearchResult] = []
        seen_ids: set[str] = set()
        context_snippets: list[str] = []

        for round_idx in range(max_rounds):
            results = await self._mm.search(current_query, domain=domain, k=k_per_round)

            new_results = []
            for r in results:
                if r.document.id not in seen_ids:
                    seen_ids.add(r.document.id)
                    new_results.append(r)
                    all_results.append(r)
                    context_snippets.append(r.document.content[:200])

            if not new_results:
                logger.debug("Round %d: no new results", round_idx)
                break

            if round_idx < max_rounds - 1:
                # LLM にクエリを再生成させる
                context_text = "\n\n".join(context_snippets[-5:])  # 直近 5 件
                rewrite_prompt = (
                    f"Original query: {query}\n"
                    f"Retrieved context so far:\n{context_text}\n\n"
                    f"Generate an improved search query to find more relevant information "
                    f"that was not covered by the context above. "
                    f"Output only the query text, nothing else."
                )
                try:
                    current_query = (await self._llm.complete(rewrite_prompt)).strip()
                    logger.debug("Round %d: rewritten query=%r", round_idx, current_query[:80])
                except Exception:
                    logger.exception("LLM rewrite failed at round %d; stopping", round_idx)
                    break

        return self._sort_and_dedup(all_results) if self._dedup else all_results

    # ------------------------------------------------------------------
    # 戦略 3: HyDE
    # ------------------------------------------------------------------

    async def _retrieve_hyde(
        self,
        query: str,
        domain: Optional[str],
        k: int,
    ) -> list[SearchResult]:
        """Hypothetical Document Embedding 検索。

        LLM に仮想的な回答ドキュメントを生成させ、その埋め込みで検索する。
        """
        if self._llm is None:
            logger.warning("HyDE requires LLM; falling back to standard search")
            return await self._mm.search(query, domain=domain, k=k)

        hyde_prompt = (
            f"Write a detailed, informative document that would be the ideal answer "
            f"to the following question. Use technical details and specifics.\n\n"
            f"Question: {query}\n\n"
            f"Document:"
        )
        try:
            hypothetical_doc = (await self._llm.complete(hyde_prompt)).strip()
            logger.debug("HyDE generated doc length=%d", len(hypothetical_doc))
        except Exception:
            logger.exception("HyDE LLM generation failed; falling back to standard search")
            return await self._mm.search(query, domain=domain, k=k)

        # 仮想ドキュメントの埋め込みで検索
        hyde_vec = self._embedder.embed(hypothetical_doc)
        return await self._search_by_vec(hyde_vec, domain, k)

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    async def _search_by_vec(
        self,
        query_vec: NDArray[np.float32],
        domain: Optional[str],
        k: int,
    ) -> list[SearchResult]:
        """埋め込みベクトルで直接 FAISS 検索し、メタデータを付与して返す。"""
        query_2d = query_vec.reshape(1, -1)

        if domain is not None:
            raw = self._mm.faiss.search(domain, query_2d, k=k)
            pairs = raw  # list[(doc_id, score)]
        else:
            raw3 = self._mm.faiss.search_all(query_2d, k=k)
            pairs = [(doc_id, score) for doc_id, _dom, score in raw3]

        if not pairs:
            return []

        doc_ids = [doc_id for doc_id, _ in pairs]
        docs = await self._mm.store.get_batch(doc_ids)
        doc_map = {d.id: d for d in docs if d is not None}

        results: list[SearchResult] = []
        for doc_id, score in pairs:
            doc = doc_map.get(doc_id)
            if doc is not None:
                results.append(SearchResult(document=doc, score=score, query=""))

        return results

    def _sort_and_dedup(self, results: list[SearchResult]) -> list[SearchResult]:
        """スコア降順に並び替え、重複 doc_id を除去する。"""
        seen: set[str] = set()
        unique: list[SearchResult] = []
        for r in sorted(results, key=lambda x: x.score, reverse=True):
            if r.document.id not in seen:
                seen.add(r.document.id)
                unique.append(r)
        return unique
