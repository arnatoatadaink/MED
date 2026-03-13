"""src/memory/memory_manager.py — FAISS-SQLite 原子的操作マネージャー

FAISSIndexManager と MetadataStore を統合し、ドキュメントの追加・削除・検索を
原子的に管理する。KG 登録フックも備える（Phase 1.5 以降で実装）。

使い方:
    from src.memory.memory_manager import MemoryManager

    manager = MemoryManager()
    await manager.initialize()

    doc = Document(content="...", domain="code", source=SourceMeta(...))
    doc_id = await manager.add(doc)

    results = await manager.search("query text", domain="code", k=5)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.common.config import FAISSConfig, MetadataConfig, get_settings
from src.memory.embedder import Embedder
from src.memory.faiss_index import FAISSIndexManager
from src.memory.metadata_store import MetadataStore
from src.memory.schema import Document, Domain, SearchResult, SourceMeta, SourceType, UsefulnessScore

logger = logging.getLogger(__name__)


class MemoryManager:
    """FAISS + SQLite を統合した原子的メモリ操作マネージャー。

    追加・削除では FAISS → SQLite の順に操作し、SQLite が失敗した場合は
    FAISS 側をロールバックして整合性を保つ。

    Attributes:
        embedder: テキスト埋め込みモジュール。
        faiss: ドメイン別 FAISS インデックス管理。
        store: SQLite メタデータストア。
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        faiss: Optional[FAISSIndexManager] = None,
        store: Optional[MetadataStore] = None,
        *,
        faiss_config: Optional[FAISSConfig] = None,
        metadata_config: Optional[MetadataConfig] = None,
    ) -> None:
        settings = get_settings()
        self.embedder = embedder or Embedder()
        self.faiss = faiss or FAISSIndexManager(faiss_config or settings.faiss)
        self.store = store or MetadataStore(metadata_config or settings.metadata)
        self._initialized = False

    # ------------------------------------------------------------------
    # ライフサイクル
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """SQLite テーブル作成など初期化処理を実行する。"""
        await self.store.initialize()
        self._initialized = True
        logger.info("MemoryManager initialized")

    async def close(self) -> None:
        """リソースを解放する。"""
        await self.store.close()
        logger.info("MemoryManager closed")

    # ------------------------------------------------------------------
    # 追加 (FAISS → SQLite, SQLite 失敗時は FAISS ロールバック)
    # ------------------------------------------------------------------

    async def add(self, doc: Document) -> str:
        """ドキュメントを FAISS + SQLite に追加する。

        Returns:
            保存された doc.id。
        """
        self._ensure_initialized()

        # 埋め込みがなければ生成
        if doc.embedding is None:
            doc = doc.model_copy(update={"embedding": self.embedder.embed(doc.content)})

        embedding = doc.embedding
        assert embedding is not None

        domain = doc.domain if isinstance(doc.domain, str) else doc.domain.value

        # FAISS へ追加（先に行いロールバック可能にする）
        self.faiss.add(domain, [doc.id], embedding.reshape(1, -1))

        # SQLite へ保存（失敗時は FAISS からも削除）
        try:
            await self.store.save(doc)
        except Exception:
            logger.exception("SQLite save failed for doc=%s; rolling back FAISS", doc.id)
            try:
                self.faiss.remove(domain, [doc.id])
            except Exception:
                logger.exception("FAISS rollback also failed for doc=%s", doc.id)
            raise

        # KG 登録フック（Phase 1.5 で実装）
        await self._kg_register_hook(doc)

        logger.debug("Added doc=%s domain=%s", doc.id, domain)
        return doc.id

    async def add_from_text(
        self,
        content: str,
        domain: str = "general",
        source_type: str = "manual",
        source_url: Optional[str] = None,
        teacher_id: Optional[str] = None,
        teacher_provider: Optional[str] = None,
        **kwargs,
    ) -> str:
        """テキストから直接ドキュメントを追加するコンビニエンスメソッド。

        ``teacher_id`` を指定すると、``SourceMeta.extra`` に Teacher 素性を記録する。

        Args:
            content:          ドキュメントテキスト。
            domain:           ドメイン (``"code"`` / ``"academic"`` / ``"general"``)。
            source_type:      取得元種別文字列 (``SourceType`` の値)。
            source_url:       元 URL。
            teacher_id:       Teacher モデル識別子。例: ``"claude-opus-4-6"``。
            teacher_provider: Teacher プロバイダ。省略時は ``teacher_id`` から自動推定。
            **kwargs:         ``Document`` 追加フィールド (difficulty, confidence, …)。

        Returns:
            保存された doc.id。
        """
        try:
            st = SourceType(source_type)
        except ValueError:
            st = SourceType.MANUAL

        source = SourceMeta(source_type=st, url=source_url)
        if teacher_id:
            source.set_teacher(teacher_id, provider=teacher_provider)

        doc = Document(content=content, domain=domain, source=source, **kwargs)
        return await self.add(doc)

    async def add_batch(self, docs: list[Document]) -> list[str]:
        """複数ドキュメントを一括追加する。

        各ドキュメントを個別に add() するため、失敗したドキュメントだけ
        例外が発生し他は影響を受けない（部分成功）。

        Returns:
            成功した doc.id のリスト。
        """
        self._ensure_initialized()

        # バッチ埋め込み
        need_embed = [d for d in docs if d.embedding is None]
        if need_embed:
            texts = [d.content for d in need_embed]
            embeddings = self.embedder.embed_batch(texts)
            idx = 0
            docs = list(docs)  # copy to avoid mutating caller's list
            for i, d in enumerate(docs):
                if d.embedding is None:
                    docs[i] = d.model_copy(update={"embedding": embeddings[idx]})
                    idx += 1

        added: list[str] = []
        for doc in docs:
            try:
                doc_id = await self.add(doc)
                added.append(doc_id)
            except Exception:
                logger.exception("Failed to add doc=%s in batch; skipping", doc.id)

        return added

    # ------------------------------------------------------------------
    # 削除 (SQLite → FAISS, FAISS 失敗は警告のみ)
    # ------------------------------------------------------------------

    async def delete(self, doc_id: str) -> bool:
        """ドキュメントを FAISS + SQLite から削除する。

        Returns:
            ドキュメントが存在して削除できた場合 True。
        """
        self._ensure_initialized()

        doc = await self.store.get(doc_id)
        if doc is None:
            return False

        domain = doc.domain if isinstance(doc.domain, str) else doc.domain.value

        # SQLite から先に削除（失敗したら FAISS は触らない）
        deleted = await self.store.delete(doc_id)
        if not deleted:
            return False

        # FAISS から削除（失敗しても SQLite 側はロールバックしない — 次の rebuild で整合）
        try:
            self.faiss.remove(domain, [doc_id])
        except Exception:
            logger.warning("FAISS remove failed for doc=%s; will be cleaned on next rebuild", doc_id)

        logger.debug("Deleted doc=%s domain=%s", doc_id, domain)
        return True

    # ------------------------------------------------------------------
    # 検索
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        domain: Optional[str] = None,
        k: int = 5,
        min_score: float = -1.0,
    ) -> list[SearchResult]:
        """クエリテキストでベクトル検索を行い、メタデータを付与して返す。

        Args:
            query: 検索クエリ文字列。
            domain: 検索対象ドメイン (None = 全ドメイン)。
            k: 返す件数。
            min_score: この値未満のスコアは除外（内積スコアは -1〜1 の範囲）。

        Returns:
            スコア降順の SearchResult リスト。
        """
        self._ensure_initialized()

        query_vec = self.embedder.embed(query).reshape(1, -1)

        # FAISS 検索
        if domain is not None:
            # list[tuple[str, float]]
            raw: list[tuple[str, float]] = self.faiss.search(domain, query_vec, k=k)
        else:
            # list[tuple[str, str, float]] — (doc_id, domain, score)
            raw3: list[tuple[str, str, float]] = self.faiss.search_all(query_vec, k=k)
            raw = [(doc_id, score) for doc_id, _dom, score in raw3]

        if not raw:
            return []

        # スコアフィルタ
        raw = [(doc_id, score) for doc_id, score in raw if score >= min_score]

        if not raw:
            return []

        # メタデータ取得
        doc_ids = [doc_id for doc_id, _ in raw]
        docs = await self.store.get_batch(doc_ids)
        doc_map = {d.id: d for d in docs if d is not None}

        results: list[SearchResult] = []
        for doc_id, score in raw:
            doc = doc_map.get(doc_id)
            if doc is None:
                logger.warning("Doc=%s found in FAISS but not in SQLite; skipping", doc_id)
                continue
            results.append(SearchResult(document=doc, score=score, query=query))

        return results

    # ------------------------------------------------------------------
    # 検索後フィードバック
    # ------------------------------------------------------------------

    async def record_retrieval(self, doc_id: str, selected: bool = False) -> None:
        """検索ヒット・選択イベントを記録する。

        Args:
            doc_id: 対象ドキュメント ID。
            selected: ユーザーが実際に選択/利用した場合 True。
        """
        self._ensure_initialized()
        await self.store.increment_retrieval(doc_id)
        if selected:
            await self.store.increment_selection(doc_id)

    async def add_feedback(self, doc_id: str, positive: bool) -> None:
        """ドキュメントにフィードバックを追加する。

        Args:
            doc_id: 対象ドキュメント ID。
            positive: 肯定的フィードバックなら True。
        """
        self._ensure_initialized()
        await self.store.add_feedback(doc_id, positive=positive)

    # ------------------------------------------------------------------
    # 取得・統計
    # ------------------------------------------------------------------

    async def get(self, doc_id: str) -> Optional[Document]:
        """ドキュメントを ID で取得する。"""
        self._ensure_initialized()
        return await self.store.get(doc_id)

    async def exists(self, doc_id: str) -> bool:
        """ドキュメントが存在するか確認する。"""
        self._ensure_initialized()
        return await self.store.exists(doc_id)

    async def stats(self) -> dict:
        """メモリ全体の統計情報を返す。

        Returns:
            {
                "total_docs": int,
                "avg_confidence": float,
                "faiss_stats": {domain: count},
            }
        """
        self._ensure_initialized()
        total = await self.store.count()
        avg_conf = await self.store.avg_confidence()
        faiss_stats = self.faiss.domain_stats()
        return {
            "total_docs": total,
            "avg_confidence": avg_conf,
            "faiss_stats": faiss_stats,
        }

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("MemoryManager.initialize() must be called before use")

    async def _kg_register_hook(self, doc: Document) -> None:
        """KG 登録フック（Phase 1.5 で EntityExtractor を呼び出す）。"""
        # Phase 1.5 で実装: EntityExtractor.extract(doc) → KGStore.add_entity/relation
        pass
