"""src/knowledge_graph/neo4j_store.py — Neo4j バックエンド

Phase 2+ の永続化実装。Neo4j の Bolt プロトコルで接続する。
ACID トランザクション、Cypher クエリ、同時アクセスに対応。

前提:
    pip install neo4j
    docker run -d -p 7474:7474 -p 7687:7687 neo4j:5-community

使い方:
    from src.knowledge_graph.neo4j_store import Neo4jKnowledgeGraphStore

    kg = Neo4jKnowledgeGraphStore(uri="bolt://localhost:7687", auth=("neo4j", "password"))
    await kg.initialize()
    kg.add_entity("Python", entity_type="language", doc_id="doc_001")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.knowledge_graph.store import (
    Entity,
    KGQueryResult,
    KnowledgeGraphStore,
    Relation,
)

logger = logging.getLogger(__name__)


class Neo4jKnowledgeGraphStore(KnowledgeGraphStore):
    """Neo4j ベースの Knowledge Graph ストア。

    Args:
        uri: Bolt プロトコル URI (例: "bolt://localhost:7687")。
        auth: 認証情報 (user, password) タプル。
        database: Neo4j データベース名。
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        auth: tuple[str, str] = ("neo4j", "neo4j"),
        database: str = "neo4j",
    ) -> None:
        try:
            import neo4j
            self._neo4j = neo4j
        except ImportError:
            raise RuntimeError(
                "neo4j not installed. Run: pip install neo4j"
            )

        self._uri = uri
        self._auth = auth
        self._database = database
        self._driver = neo4j.GraphDatabase.driver(uri, auth=auth)
        logger.info("Neo4j KG store created: %s (db=%s)", uri, database)

    def close(self) -> None:
        """ドライバーを閉じる。"""
        self._driver.close()

    def _run_query(self, query: str, **params: Any) -> list[dict]:
        """同期 Cypher クエリを実行し、結果をリストで返す。"""
        with self._driver.session(database=self._database) as session:
            result = session.run(query, **params)
            return [record.data() for record in result]

    def _run_write(self, query: str, **params: Any) -> None:
        """同期 Cypher 書き込みクエリを実行する。"""
        with self._driver.session(database=self._database) as session:
            session.run(query, **params)

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
        props = properties or {}
        if doc_id:
            self._run_write(
                """
                MERGE (e:Entity {name: $name})
                ON CREATE SET e.entity_type = $entity_type,
                              e.doc_ids = [$doc_id],
                              e.properties = $properties
                ON MATCH SET e.doc_ids = CASE
                    WHEN NOT $doc_id IN e.doc_ids THEN e.doc_ids + $doc_id
                    ELSE e.doc_ids
                END
                """,
                name=name, entity_type=entity_type, doc_id=doc_id,
                properties=props,
            )
        else:
            self._run_write(
                """
                MERGE (e:Entity {name: $name})
                ON CREATE SET e.entity_type = $entity_type,
                              e.doc_ids = [],
                              e.properties = $properties
                """,
                name=name, entity_type=entity_type, properties=props,
            )
        logger.debug("Entity added/updated: %s (%s)", name, entity_type)

    def get_entity(self, name: str) -> Entity | None:
        rows = self._run_query(
            """
            MATCH (e:Entity {name: $name})
            RETURN e.name AS name, e.entity_type AS entity_type,
                   e.doc_ids AS doc_ids, e.properties AS properties
            """,
            name=name,
        )
        if not rows:
            return None
        r = rows[0]
        return Entity(
            name=r["name"],
            entity_type=r.get("entity_type", "concept"),
            doc_ids=list(r.get("doc_ids") or []),
            properties=dict(r.get("properties") or {}),
        )

    def entity_exists(self, name: str) -> bool:
        rows = self._run_query(
            "MATCH (e:Entity {name: $name}) RETURN count(e) AS cnt",
            name=name,
        )
        return rows[0]["cnt"] > 0 if rows else False

    def remove_entity(self, name: str) -> bool:
        rows = self._run_query(
            """
            MATCH (e:Entity {name: $name})
            WITH e, count(e) AS cnt
            DETACH DELETE e
            RETURN cnt
            """,
            name=name,
        )
        return rows[0]["cnt"] > 0 if rows else False

    def all_entities(self) -> list[Entity]:
        rows = self._run_query(
            """
            MATCH (e:Entity)
            RETURN e.name AS name, e.entity_type AS entity_type,
                   e.doc_ids AS doc_ids, e.properties AS properties
            """
        )
        return [
            Entity(
                name=r["name"],
                entity_type=r.get("entity_type", "concept"),
                doc_ids=list(r.get("doc_ids") or []),
                properties=dict(r.get("properties") or {}),
            )
            for r in rows
        ]

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
        props = properties or {}
        # MERGE で source/target を自動作成
        self._run_write(
            """
            MERGE (s:Entity {name: $source})
            ON CREATE SET s.entity_type = 'concept', s.doc_ids = [], s.properties = {}
            MERGE (t:Entity {name: $target})
            ON CREATE SET t.entity_type = 'concept', t.doc_ids = [], t.properties = {}
            MERGE (s)-[r:RELATION {relation_type: $relation_type}]->(t)
            SET r.weight = $weight, r.properties = $properties
            """,
            source=source, target=target, relation_type=relation_type,
            weight=weight, properties=props,
        )
        logger.debug("Relation added: %s -[%s]-> %s", source, relation_type, target)

    def get_relations(self, source: str, target: str | None = None) -> list[Relation]:
        if target is not None:
            rows = self._run_query(
                """
                MATCH (s:Entity {name: $source})-[r:RELATION]->(t:Entity {name: $target})
                RETURN s.name AS source, t.name AS target,
                       r.relation_type AS relation_type, r.weight AS weight,
                       r.properties AS properties
                """,
                source=source, target=target,
            )
        else:
            rows = self._run_query(
                """
                MATCH (s:Entity {name: $source})-[r:RELATION]->(t:Entity)
                RETURN s.name AS source, t.name AS target,
                       r.relation_type AS relation_type, r.weight AS weight,
                       r.properties AS properties
                """,
                source=source,
            )
        return [
            Relation(
                source=r["source"],
                target=r["target"],
                relation_type=r.get("relation_type", "related_to"),
                weight=r.get("weight", 1.0),
                properties=dict(r.get("properties") or {}),
            )
            for r in rows
        ]

    def remove_relation(self, source: str, target: str) -> bool:
        rows = self._run_query(
            """
            MATCH (s:Entity {name: $source})-[r:RELATION]->(t:Entity {name: $target})
            WITH r, count(r) AS cnt
            DELETE r
            RETURN cnt
            """,
            source=source, target=target,
        )
        return rows[0]["cnt"] > 0 if rows else False

    # ------------------------------------------------------------------
    # クエリ
    # ------------------------------------------------------------------

    def query_neighbors(
        self,
        entity_name: str,
        max_depth: int = 1,
        relation_types: list[str] | None = None,
    ) -> KGQueryResult:
        if not self.entity_exists(entity_name):
            return KGQueryResult(entities=[], relations=[], paths=[], doc_ids=[])

        # 可変長パスで近傍を取得
        type_filter = ""
        params: dict[str, Any] = {"name": entity_name, "max_depth": max_depth}
        if relation_types:
            type_filter = "AND r.relation_type IN $relation_types"
            params["relation_types"] = relation_types

        rows = self._run_query(
            f"""
            MATCH (start:Entity {{name: $name}})
            CALL apoc.path.expandConfig(start, {{
                maxLevel: $max_depth,
                relationshipFilter: 'RELATION>'
            }}) YIELD path
            WITH nodes(path) AS ns, relationships(path) AS rs
            UNWIND ns AS n
            WITH DISTINCT n, rs
            RETURN n.name AS name, n.entity_type AS entity_type,
                   n.doc_ids AS doc_ids, n.properties AS properties
            """,
            **params,
        )

        # APOC が利用できない場合のフォールバック: 深度ごとにクエリ
        if not rows:
            rows = self._query_neighbors_fallback(entity_name, max_depth, relation_types)

        entities = [
            Entity(
                name=r["name"],
                entity_type=r.get("entity_type", "concept"),
                doc_ids=list(r.get("doc_ids") or []),
                properties=dict(r.get("properties") or {}),
            )
            for r in rows
        ]
        doc_ids = list({did for e in entities for did in e.doc_ids})

        # リレーション取得
        all_relations: list[Relation] = []
        for e in entities:
            rels = self.get_relations(e.name)
            if relation_types:
                rels = [r for r in rels if r.relation_type in relation_types]
            all_relations.extend(rels)

        return KGQueryResult(
            entities=entities,
            relations=all_relations,
            paths=[],
            doc_ids=doc_ids,
        )

    def _query_neighbors_fallback(
        self,
        entity_name: str,
        max_depth: int,
        relation_types: list[str] | None,
    ) -> list[dict]:
        """APOC なしフォールバック: BFS を Cypher で実行。"""
        type_filter = ""
        params: dict[str, Any] = {"name": entity_name}
        if relation_types:
            type_filter = "WHERE ALL(r IN relationships(p) WHERE r.relation_type IN $relation_types)"
            params["relation_types"] = relation_types

        rows = self._run_query(
            f"""
            MATCH p = (start:Entity {{name: $name}})-[*1..{max_depth}]->(n:Entity)
            {type_filter}
            WITH DISTINCT n
            RETURN n.name AS name, n.entity_type AS entity_type,
                   n.doc_ids AS doc_ids, n.properties AS properties
            """,
            **params,
        )
        # 起点自身も含める
        start = self._run_query(
            """
            MATCH (e:Entity {name: $name})
            RETURN e.name AS name, e.entity_type AS entity_type,
                   e.doc_ids AS doc_ids, e.properties AS properties
            """,
            name=entity_name,
        )
        return start + rows

    def find_path(
        self,
        source: str,
        target: str,
        max_hops: int = 5,
    ) -> list[str]:
        if not self.entity_exists(source) or not self.entity_exists(target):
            return []

        rows = self._run_query(
            f"""
            MATCH p = shortestPath(
                (a:Entity {{name: $source}})-[*1..{max_hops}]->(b:Entity {{name: $target}})
            )
            RETURN [n IN nodes(p) | n.name] AS path
            """,
            source=source, target=target,
        )
        if rows and rows[0].get("path"):
            return rows[0]["path"]
        return []

    def get_doc_ids_for_entities(self, entity_names: list[str]) -> list[str]:
        if not entity_names:
            return []
        rows = self._run_query(
            """
            MATCH (e:Entity) WHERE e.name IN $names
            UNWIND e.doc_ids AS did
            RETURN DISTINCT did
            """,
            names=entity_names,
        )
        return [r["did"] for r in rows]

    # ------------------------------------------------------------------
    # 統計
    # ------------------------------------------------------------------

    @property
    def entity_count(self) -> int:
        rows = self._run_query("MATCH (e:Entity) RETURN count(e) AS cnt")
        return rows[0]["cnt"] if rows else 0

    @property
    def relation_count(self) -> int:
        rows = self._run_query("MATCH ()-[r:RELATION]->() RETURN count(r) AS cnt")
        return rows[0]["cnt"] if rows else 0

    def stats(self) -> dict:
        return {
            "entity_count": self.entity_count,
            "relation_count": self.relation_count,
            "directed": True,
            "backend": "neo4j",
            "uri": self._uri,
            "database": self._database,
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
        rows = self._run_query(
            """
            MATCH (a:Entity)-[r:RELATION {relation_type: 'alias_of'}]->(c:Entity)
            WHERE toLower(a.name) = toLower($alias)
            RETURN a.doc_ids AS alias_doc_ids, c.doc_ids AS canonical_doc_ids
            """,
            alias=alias,
        )
        doc_ids: list[str] = []
        for r in rows:
            doc_ids.extend(r.get("alias_doc_ids") or [])
            doc_ids.extend(r.get("canonical_doc_ids") or [])

        # alias 自体に doc_ids がある場合も取得
        alias_rows = self._run_query(
            """
            MATCH (e:Entity)
            WHERE toLower(e.name) = toLower($alias)
            RETURN e.doc_ids AS doc_ids
            """,
            alias=alias,
        )
        for r in alias_rows:
            doc_ids.extend(r.get("doc_ids") or [])

        return list(dict.fromkeys(doc_ids))

    # ------------------------------------------------------------------
    # 永続化
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Neo4j はデータベース自体が永続化するため、エクスポート操作として実装。

        Cypher の EXPORT を使ってノードとリレーションを JSON で出力する。
        """
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        entities = self.all_entities()
        relations_data: list[dict] = []
        for e in entities:
            for r in self.get_relations(e.name):
                relations_data.append({
                    "source": r.source,
                    "target": r.target,
                    "relation_type": r.relation_type,
                    "weight": r.weight,
                    "properties": r.properties,
                })

        data = {
            "backend": "neo4j",
            "entities": [
                {
                    "name": e.name,
                    "entity_type": e.entity_type,
                    "doc_ids": e.doc_ids,
                    "properties": e.properties,
                }
                for e in entities
            ],
            "relations": relations_data,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Neo4j KG exported to %s (%d entities, %d relations)",
                     path, len(entities), len(relations_data))
