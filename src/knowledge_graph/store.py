"""src/knowledge_graph/store.py — KnowledgeGraphStore (NetworkX backend)

FAISS（連想記憶）と SQL（宣言的記憶）の橋渡し層として機能する Knowledge Graph。
Entity 間の関係性（因果・階層・時系列）を管理し、推論品質を向上させる。

設計方針（CLAUDE.md 原則）:
- Phase 1.5: NetworkX インメモリ実装（依存ゼロ、プロトタイプ）
- Phase 2+: Neo4j 移行（永続化、本格クエリ）
- KG は橋渡しに徹する — KG 単独で答えを出さず、ルーティングと Fusion を担う

使い方:
    from src.knowledge_graph.store import KnowledgeGraphStore

    kg = KnowledgeGraphStore()
    kg.add_entity("Python", entity_type="language", doc_id="doc_001")
    kg.add_relation("Python", "FAISS", relation_type="uses", weight=0.9)

    neighbors = kg.query_neighbors("Python", max_depth=2)
    path = kg.find_path("Python", "FAISS")
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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


class KnowledgeGraphStore:
    """NetworkX ベースの Knowledge Graph ストア。

    Args:
        directed: 有向グラフを使うか（False = 無向グラフ）。
    """

    def __init__(self, directed: bool = True) -> None:
        try:
            import networkx as nx
            self._nx = nx
        except ImportError:
            raise RuntimeError("networkx not installed. Run: pip install networkx")

        self._graph = nx.DiGraph() if directed else nx.Graph()
        self._directed = directed

    # ------------------------------------------------------------------
    # Entity 管理
    # ------------------------------------------------------------------

    def add_entity(
        self,
        name: str,
        entity_type: str = "concept",
        doc_id: str | None = None,
        properties: dict | None = None,
    ) -> None:
        """Entity をグラフに追加する。

        既存の Entity の場合は doc_id を追加し、properties をマージする。
        """
        if self._graph.has_node(name):
            # 既存ノードを更新
            node = self._graph.nodes[name]
            if doc_id and doc_id not in node.get("doc_ids", []):
                node.setdefault("doc_ids", []).append(doc_id)
            if properties:
                node.setdefault("properties", {}).update(properties)
        else:
            self._graph.add_node(
                name,
                entity_type=entity_type,
                doc_ids=[doc_id] if doc_id else [],
                properties=properties or {},
            )
        logger.debug("Entity added/updated: %s (%s)", name, entity_type)

    def get_entity(self, name: str) -> Entity | None:
        """Entity を取得する。存在しない場合は None。"""
        if not self._graph.has_node(name):
            return None
        node = self._graph.nodes[name]
        return Entity(
            name=name,
            entity_type=node.get("entity_type", "concept"),
            doc_ids=list(node.get("doc_ids", [])),
            properties=dict(node.get("properties", {})),
        )

    def entity_exists(self, name: str) -> bool:
        return self._graph.has_node(name)

    def remove_entity(self, name: str) -> bool:
        """Entity とその関連エッジを削除する。"""
        if not self._graph.has_node(name):
            return False
        self._graph.remove_node(name)
        return True

    def all_entities(self) -> list[Entity]:
        """全 Entity のリストを返す。"""
        result = []
        for name, data in self._graph.nodes(data=True):
            result.append(Entity(
                name=name,
                entity_type=data.get("entity_type", "concept"),
                doc_ids=list(data.get("doc_ids", [])),
                properties=dict(data.get("properties", {})),
            ))
        return result

    # ------------------------------------------------------------------
    # Relation 管理
    # ------------------------------------------------------------------

    def add_relation(
        self,
        source: str,
        target: str,
        relation_type: str = "related_to",
        weight: float = 1.0,
        properties: dict | None = None,
    ) -> None:
        """Relation（エッジ）を追加する。

        source/target が存在しない場合は自動的に Entity として追加する。
        """
        if not self._graph.has_node(source):
            self.add_entity(source)
        if not self._graph.has_node(target):
            self.add_entity(target)

        self._graph.add_edge(
            source,
            target,
            relation_type=relation_type,
            weight=weight,
            properties=properties or {},
        )
        logger.debug("Relation added: %s -[%s]-> %s", source, relation_type, target)

    def get_relations(self, source: str, target: str | None = None) -> list[Relation]:
        """source からの Relation リストを返す。target 指定で絞り込み可能。"""
        if not self._graph.has_node(source):
            return []

        relations = []
        edges = (
            [(source, target, self._graph.get_edge_data(source, target))]
            if target is not None and self._graph.has_edge(source, target)
            else self._graph.out_edges(source, data=True) if self._directed
            else self._graph.edges(source, data=True)
        )
        for src, tgt, data in edges:
            if data is not None:
                relations.append(Relation(
                    source=str(src),
                    target=str(tgt),
                    relation_type=data.get("relation_type", "related_to"),
                    weight=data.get("weight", 1.0),
                    properties=dict(data.get("properties", {})),
                ))
        return relations

    def remove_relation(self, source: str, target: str) -> bool:
        """Relation を削除する。"""
        if not self._graph.has_edge(source, target):
            return False
        self._graph.remove_edge(source, target)
        return True

    # ------------------------------------------------------------------
    # クエリ
    # ------------------------------------------------------------------

    def query_neighbors(
        self,
        entity_name: str,
        max_depth: int = 1,
        relation_types: list[str] | None = None,
    ) -> KGQueryResult:
        """指定 Entity の近傍ノードを返す。

        Args:
            entity_name: 起点となる Entity 名。
            max_depth: 探索深度。
            relation_types: フィルタするリレーション型（None = 全て）。

        Returns:
            KGQueryResult。
        """
        if not self._graph.has_node(entity_name):
            return KGQueryResult(entities=[], relations=[], paths=[], doc_ids=[])

        visited: set[str] = {entity_name}
        frontier = {entity_name}
        all_relations: list[Relation] = []

        for _ in range(max_depth):
            next_frontier: set[str] = set()
            for node in frontier:
                edges = (
                    self._graph.out_edges(node, data=True) if self._directed
                    else self._graph.edges(node, data=True)
                )
                for src, tgt, data in edges:
                    if relation_types and data.get("relation_type") not in relation_types:
                        continue
                    rel = Relation(
                        source=str(src),
                        target=str(tgt),
                        relation_type=data.get("relation_type", "related_to"),
                        weight=data.get("weight", 1.0),
                        properties=dict(data.get("properties", {})),
                    )
                    all_relations.append(rel)
                    if str(tgt) not in visited:
                        visited.add(str(tgt))
                        next_frontier.add(str(tgt))
            frontier = next_frontier
            if not frontier:
                break

        entities = [e for e in (self.get_entity(n) for n in visited) if e is not None]
        doc_ids = list({did for e in entities for did in e.doc_ids})

        return KGQueryResult(
            entities=entities,
            relations=all_relations,
            paths=[],
            doc_ids=doc_ids,
        )

    def find_path(
        self,
        source: str,
        target: str,
        max_hops: int = 5,
    ) -> list[str]:
        """source から target への最短パスを返す。

        Returns:
            ノード名のリスト（パスが存在しない場合は空リスト）。
        """
        if not (self._graph.has_node(source) and self._graph.has_node(target)):
            return []

        try:
            path = self._nx.shortest_path(
                self._graph, source, target,
                weight=lambda u, v, d: 1.0 / max(d.get("weight", 1.0), 1e-6),
            )
            if len(path) - 1 <= max_hops:
                return path
            return []
        except self._nx.NetworkXNoPath:
            return []
        except self._nx.NodeNotFound:
            return []

    def get_doc_ids_for_entities(self, entity_names: list[str]) -> list[str]:
        """指定 Entity に関連する doc_id リストを返す。"""
        doc_ids: set[str] = set()
        for name in entity_names:
            entity = self.get_entity(name)
            if entity:
                doc_ids.update(entity.doc_ids)
        return list(doc_ids)

    # ------------------------------------------------------------------
    # 統計
    # ------------------------------------------------------------------

    @property
    def entity_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def relation_count(self) -> int:
        return self._graph.number_of_edges()

    def stats(self) -> dict:
        return {
            "entity_count": self.entity_count,
            "relation_count": self.relation_count,
            "directed": self._directed,
        }

    # ------------------------------------------------------------------
    # エイリアス管理 (alias_of relation を活用)
    # ------------------------------------------------------------------

    def add_alias(self, canonical: str, alias: str, doc_id: str | None = None) -> None:
        """canonical（正式名）と alias（略称・別名）を alias_of エッジで登録する。

        例: add_alias("Learning to Reason in 13 Parameters", "TinyLoRA")
        これにより "TinyLoRA" で検索したとき canonical の doc_ids が返る。

        Args:
            canonical: 正式名 / 論文タイトル / ライブラリ名。
            alias:     通称 / 略称 / ニックネーム。
            doc_id:    関連ドキュメント ID（省略可）。
        """
        self.add_entity(canonical, entity_type="concept", doc_id=doc_id)
        self.add_entity(alias, entity_type="alias")
        self.add_relation(alias, canonical, relation_type="alias_of", weight=1.0)
        logger.debug("Alias registered: %r --[alias_of]--> %r", alias, canonical)

    def query_by_alias(self, alias: str) -> list[str]:
        """エイリアスに対応する canonical Entity の doc_ids を返す。

        alias → canonical（alias_of エッジ）→ doc_ids の順で辿る。
        alias が直接 doc_id を持つ場合もそれを含める。

        Args:
            alias: 検索するエイリアス文字列（大文字小文字を区別しない）。

        Returns:
            関連する doc_id のリスト（重複除去済み）。
        """
        # 大文字小文字を無視して一致するノードを探す
        alias_lower = alias.lower()
        matched_nodes = [
            n for n in self._graph.nodes
            if n.lower() == alias_lower
        ]

        doc_ids: list[str] = []
        for node in matched_nodes:
            # alias 自身の doc_ids
            node_data = self._graph.nodes[node]
            doc_ids.extend(node_data.get("doc_ids", []))

            # alias_of エッジを辿って canonical の doc_ids も取得
            edges = (
                self._graph.out_edges(node, data=True)
                if self._directed
                else self._graph.edges(node, data=True)
            )
            for _src, tgt, data in edges:
                if data.get("relation_type") == "alias_of" and self._graph.has_node(tgt):
                    canonical_data = self._graph.nodes[tgt]
                    doc_ids.extend(canonical_data.get("doc_ids", []))

        # 重複除去・順序保持
        return list(dict.fromkeys(doc_ids))

    # ------------------------------------------------------------------
    # 永続化
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """グラフを pickle で保存する。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._graph, f)
        logger.info("KG saved to %s (%d entities, %d relations)", path, self.entity_count, self.relation_count)

    @classmethod
    def load(cls, path: str | Path) -> KnowledgeGraphStore:
        """pickle から復元する。"""
        path = Path(path)
        with open(path, "rb") as f:
            graph = pickle.load(f)
        kg = cls.__new__(cls)
        try:
            import networkx as nx
            kg._nx = nx
        except ImportError:
            raise RuntimeError("networkx not installed")
        kg._graph = graph
        kg._directed = isinstance(graph, kg._nx.DiGraph)
        logger.info("KG loaded from %s", path)
        return kg
