"""src/knowledge_graph/router_bridge.py — ModelRouter と KG の連携

KG の近傍情報をクエリ処理に組み込む橋渡し層。
FAISS 検索と KG クエリを組み合わせて、関連 doc_id の候補を拡張する。

設計方針（CLAUDE.md 原則）:
- KG は橋渡しに徹する — ルーティング補助と doc_id 拡張のみ担当
- Phase 1.5: NetworkX → Phase 2 で Neo4j 移行
- 単体で答えを出さず、MemoryManager の検索を補強する

使い方:
    from src.knowledge_graph.router_bridge import KGRouterBridge

    bridge = KGRouterBridge(kg_store, extractor)
    # ドキュメント追加時に KG 自動登録
    await bridge.on_document_added(doc)
    # クエリ時に KG で関連 doc_id を拡張
    extra_ids = await bridge.expand_query(query_entities, k=10)
"""

from __future__ import annotations

import logging

from src.knowledge_graph.store import KGQueryResult, KnowledgeGraphStore
from src.memory.schema import Document

logger = logging.getLogger(__name__)


class KGRouterBridge:
    """KG とメモリマネージャーをつなぐブリッジ。

    Args:
        kg: KnowledgeGraphStore インスタンス。
        extractor: EntityExtractor インスタンス（None = 自動抽出なし）。
    """

    def __init__(
        self,
        kg: KnowledgeGraphStore,
        extractor: EntityExtractor | None = None,
    ) -> None:
        self._kg = kg
        self._extractor = extractor

    async def on_document_added(self, doc: Document) -> None:
        """ドキュメントが MemoryManager に追加された際に KG へ自動登録する。

        MemoryManager._kg_register_hook() から呼ばれる想定。
        """
        if self._extractor is None:
            logger.debug("No extractor configured; skipping KG registration for doc=%s", doc.id)
            return

        try:
            await self._extractor.extract_and_register(doc, self._kg)
        except Exception:
            logger.exception("KG registration failed for doc=%s; continuing", doc.id)

    def expand_query_with_kg(
        self,
        entity_names: list[str],
        max_depth: int = 2,
        max_doc_ids: int = 20,
    ) -> list[str]:
        """クエリに関連する Entity の近傍から doc_id を収集する。

        FAISS 検索では見つからないが KG 上で関連する文書を発見するために使用する。

        Args:
            entity_names: クエリから抽出された Entity 名のリスト。
            max_depth: KG 探索深度。
            max_doc_ids: 返す最大 doc_id 数。

        Returns:
            関連 doc_id のリスト（重複除去済み）。
        """
        all_doc_ids: set[str] = set()

        for name in entity_names:
            if not self._kg.entity_exists(name):
                continue
            result: KGQueryResult = self._kg.query_neighbors(name, max_depth=max_depth)
            all_doc_ids.update(result.doc_ids)

        doc_ids = list(all_doc_ids)[:max_doc_ids]
        logger.debug(
            "KG expanded query: %d entities → %d doc_ids",
            len(entity_names), len(doc_ids),
        )
        return doc_ids

    def get_related_entities(
        self,
        entity_names: list[str],
        max_depth: int = 1,
    ) -> list[str]:
        """指定 Entity の関連 Entity 名リストを返す（ルーティング補助）。"""
        related: set[str] = set()
        for name in entity_names:
            if self._kg.entity_exists(name):
                result = self._kg.query_neighbors(name, max_depth=max_depth)
                related.update(e.name for e in result.entities)
        related.discard("")
        for name in entity_names:
            related.discard(name)
        return list(related)

    def find_connecting_path(
        self,
        entity_a: str,
        entity_b: str,
        max_hops: int = 4,
    ) -> list[str]:
        """2 つの Entity を結ぶパスを返す（推論トレース用）。"""
        return self._kg.find_path(entity_a, entity_b, max_hops=max_hops)

    @property
    def kg(self) -> KnowledgeGraphStore:
        """内部 KG ストアへのアクセサ。"""
        return self._kg

    @property
    def stats(self) -> dict:
        """KG 統計情報。"""
        return self._kg.stats()


# ──────────────────────────────────────────────
# 型アノテーション用 (循環インポート回避)
# ──────────────────────────────────────────────
from src.knowledge_graph.extractor import EntityExtractor  # noqa: E402
