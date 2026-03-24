"""src/knowledge_graph/networkx_store.py — NetworkX バックエンド

Phase 1.5 のインメモリ実装。依存は networkx のみ。
永続化は pickle を使用する。

使い方:
    from src.knowledge_graph.networkx_store import NetworkXKnowledgeGraphStore

    kg = NetworkXKnowledgeGraphStore()
    kg.add_entity("Python", entity_type="language", doc_id="doc_001")
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

from src.knowledge_graph.store import (
    Entity,
    KGQueryResult,
    KnowledgeGraphStore,
    Relation,
)

logger = logging.getLogger(__name__)


class NetworkXKnowledgeGraphStore(KnowledgeGraphStore):
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
        if self._graph.has_node(name):
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
        if not self._graph.has_node(name):
            return False
        self._graph.remove_node(name)
        return True

    def all_entities(self) -> list[Entity]:
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
        if not self._graph.has_node(source):
            return []

        relations = []
        edges: Any
        if target is not None and self._graph.has_edge(source, target):
            edges = [(source, target, self._graph.get_edge_data(source, target))]
        elif self._directed:
            edges = self._graph.out_edges(source, data=True)
        else:
            edges = self._graph.edges(source, data=True)

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
            "backend": "networkx",
        }

    # ------------------------------------------------------------------
    # エイリアス管理
    # ------------------------------------------------------------------

    def add_alias(self, canonical: str, alias: str, doc_id: str | None = None) -> None:
        self.add_entity(canonical, entity_type="concept", doc_id=doc_id)
        self.add_entity(alias, entity_type="alias")
        self.add_relation(alias, canonical, relation_type="alias_of", weight=1.0)
        logger.debug("Alias registered: %r --[alias_of]--> %r", alias, canonical)

    def query_by_alias(self, alias: str) -> list[str]:
        alias_lower = alias.lower()
        matched_nodes = [
            n for n in self._graph.nodes
            if n.lower() == alias_lower
        ]

        doc_ids: list[str] = []
        for node in matched_nodes:
            node_data = self._graph.nodes[node]
            doc_ids.extend(node_data.get("doc_ids", []))

            edges = (
                self._graph.out_edges(node, data=True)
                if self._directed
                else self._graph.edges(node, data=True)
            )
            for _src, tgt, data in edges:
                if data.get("relation_type") == "alias_of" and self._graph.has_node(tgt):
                    canonical_data = self._graph.nodes[tgt]
                    doc_ids.extend(canonical_data.get("doc_ids", []))

        return list(dict.fromkeys(doc_ids))

    # ------------------------------------------------------------------
    # 永続化
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._graph, f)
        logger.info("KG saved to %s (%d entities, %d relations)", path, self.entity_count, self.relation_count)

    @classmethod
    def load(cls, path: str | Path) -> NetworkXKnowledgeGraphStore:
        """pickle から復元する。"""
        path = Path(path)
        with open(path, "rb") as f:
            graph = pickle.load(f)  # noqa: S301
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
