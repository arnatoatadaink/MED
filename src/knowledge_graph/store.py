"""src/knowledge_graph/store.py — KnowledgeGraphStore ABC + データクラス

FAISS（連想記憶）と SQL（宣言的記憶）の橋渡し層として機能する Knowledge Graph。
Entity 間の関係性（因果・階層・時系列）を管理し、推論品質を向上させる。

設計方針（CLAUDE.md 原則）:
- Phase 1.5: NetworkX インメモリ実装（依存ゼロ、プロトタイプ）
- Phase 2+: Neo4j 移行（永続化、本格クエリ）
- KG は橋渡しに徹する — KG 単独で答えを出さず、ルーティングと Fusion を担う

使い方:
    from src.knowledge_graph.store import KnowledgeGraphStore

    # ファクトリで生成（デフォルト: NetworkX）
    kg = KnowledgeGraphStore.create()
    kg = KnowledgeGraphStore.create(backend="neo4j", uri="bolt://localhost:7687")

    kg.add_entity("Python", entity_type="language", doc_id="doc_001")
    kg.add_relation("Python", "FAISS", relation_type="uses", weight=0.9)

    neighbors = kg.query_neighbors("Python", max_depth=2)
    path = kg.find_path("Python", "FAISS")
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# データクラス（バックエンド非依存）
# ──────────────────────────────────────────────


@dataclass
class Entity:
    """KG ノード（Entity）。"""

    name: str
    entity_type: str = "concept"  # "language", "library", "concept", "algorithm", ...
    doc_ids: list[str] = field(default_factory=list)  # 関連ドキュメント ID
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation:
    """KG エッジ（Relation）。"""

    source: str
    target: str
    relation_type: str  # "uses", "extends", "causes", "part_of", "related_to", ...
    weight: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class KGQueryResult:
    """KG クエリ結果。"""

    entities: list[Entity]
    relations: list[Relation]
    paths: list[list[str]]  # [[entity_a, entity_b, ...], ...]
    doc_ids: list[str]  # 関連する全 doc_id


# ──────────────────────────────────────────────
# 抽象基底クラス
# ──────────────────────────────────────────────


class KnowledgeGraphStore(ABC):
    """Knowledge Graph ストアの抽象基底クラス。

    バックエンド（NetworkX / Neo4j）に依存しない共通インターフェース。
    具体的な実装は networkx_store.py / neo4j_store.py を参照。
    """

    # ── ファクトリ ───────────────────────────────

    @classmethod
    def create(cls, backend: str = "networkx", **kwargs: Any) -> KnowledgeGraphStore:
        """バックエンドを指定して KG ストアを生成する。

        Args:
            backend: "networkx" (デフォルト) or "neo4j"。
            **kwargs: バックエンド固有のパラメータ。
                networkx: directed (bool)
                neo4j: uri, auth, database

        Returns:
            KnowledgeGraphStore の具象インスタンス。
        """
        if backend == "neo4j":
            from src.knowledge_graph.neo4j_store import Neo4jKnowledgeGraphStore
            return Neo4jKnowledgeGraphStore(**kwargs)
        if backend == "networkx":
            from src.knowledge_graph.networkx_store import NetworkXKnowledgeGraphStore
            return NetworkXKnowledgeGraphStore(**kwargs)
        raise ValueError(f"Unknown KG backend: {backend!r}. Use 'networkx' or 'neo4j'.")

    @classmethod
    def load(cls, path: str | Path, backend: str = "networkx") -> KnowledgeGraphStore:
        """永続化データから復元する。

        Args:
            path: 保存先パス。
            backend: バックエンド種別。

        Returns:
            復元された KnowledgeGraphStore。
        """
        if backend == "networkx":
            from src.knowledge_graph.networkx_store import NetworkXKnowledgeGraphStore
            return NetworkXKnowledgeGraphStore.load(path)
        raise ValueError(f"load() not supported for backend: {backend!r}")

    # ── Entity 管理 ─────────────────────────────

    @abstractmethod
    def add_entity(
        self,
        name: str,
        entity_type: str = "concept",
        doc_id: str | None = None,
        properties: dict | None = None,
    ) -> None:
        """Entity をグラフに追加する。既存なら doc_id を追加し properties をマージ。"""

    @abstractmethod
    def get_entity(self, name: str) -> Entity | None:
        """Entity を取得する。存在しない場合は None。"""

    @abstractmethod
    def entity_exists(self, name: str) -> bool:
        """Entity の存在を確認する。"""

    @abstractmethod
    def remove_entity(self, name: str) -> bool:
        """Entity とその関連エッジを削除する。"""

    @abstractmethod
    def all_entities(self) -> list[Entity]:
        """全 Entity のリストを返す。"""

    # ── Relation 管理 ───────────────────────────

    @abstractmethod
    def add_relation(
        self,
        source: str,
        target: str,
        relation_type: str = "related_to",
        weight: float = 1.0,
        properties: dict | None = None,
    ) -> None:
        """Relation（エッジ）を追加する。source/target が未登録なら自動追加。"""

    @abstractmethod
    def get_relations(self, source: str, target: str | None = None) -> list[Relation]:
        """source からの Relation リストを返す。target 指定で絞り込み可能。"""

    @abstractmethod
    def remove_relation(self, source: str, target: str) -> bool:
        """Relation を削除する。"""

    # ── クエリ ──────────────────────────────────

    @abstractmethod
    def query_neighbors(
        self,
        entity_name: str,
        max_depth: int = 1,
        relation_types: list[str] | None = None,
    ) -> KGQueryResult:
        """指定 Entity の近傍ノードを返す。"""

    @abstractmethod
    def find_path(
        self,
        source: str,
        target: str,
        max_hops: int = 5,
    ) -> list[str]:
        """source から target への最短パスを返す。"""

    @abstractmethod
    def get_doc_ids_for_entities(self, entity_names: list[str]) -> list[str]:
        """指定 Entity に関連する doc_id リストを返す。"""

    # ── 統計 ────────────────────────────────────

    @property
    @abstractmethod
    def entity_count(self) -> int: ...

    @property
    @abstractmethod
    def relation_count(self) -> int: ...

    @abstractmethod
    def stats(self) -> dict: ...

    # ── エイリアス ──────────────────────────────

    @abstractmethod
    def add_alias(self, canonical: str, alias: str, doc_id: str | None = None) -> None:
        """canonical と alias を alias_of エッジで登録する。"""

    @abstractmethod
    def query_by_alias(self, alias: str) -> list[str]:
        """エイリアスに対応する canonical Entity の doc_ids を返す。"""

    # ── 永続化 ──────────────────────────────────

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """グラフを永続化する。"""
