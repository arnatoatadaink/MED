"""src/knowledge_graph/migration.py — NetworkX → Neo4j データ移行

既存の NetworkX pickle ファイルから Neo4j にデータを移行する。
逆方向（Neo4j → NetworkX pickle）のエクスポートにも対応。

使い方:
    # NetworkX pickle → Neo4j
    poetry run python -m src.knowledge_graph.migration \
        --source data/kg.pkl \
        --target bolt://localhost:7687 \
        --auth neo4j:password

    # Neo4j → JSON エクスポート
    poetry run python -m src.knowledge_graph.migration \
        --direction neo4j-to-json \
        --source bolt://localhost:7687 \
        --target data/kg_export.json \
        --auth neo4j:password

    # dry-run
    poetry run python -m src.knowledge_graph.migration \
        --source data/kg.pkl --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def migrate_networkx_to_neo4j(
    pickle_path: str | Path,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_auth: tuple[str, str] = ("neo4j", "neo4j"),
    neo4j_database: str = "neo4j",
    batch_size: int = 100,
    dry_run: bool = False,
) -> dict[str, int]:
    """NetworkX pickle → Neo4j に移行する。

    Args:
        pickle_path: NetworkX pickle ファイルのパス。
        neo4j_uri: Neo4j Bolt URI。
        neo4j_auth: (user, password)。
        neo4j_database: データベース名。
        batch_size: バッチ書き込みサイズ。
        dry_run: True なら読み込みのみ。

    Returns:
        {"entities": N, "relations": N} の移行件数。
    """
    from src.knowledge_graph.networkx_store import NetworkXKnowledgeGraphStore

    # 1. NetworkX から読み込み
    logger.info("Loading NetworkX KG from %s", pickle_path)
    nx_kg = NetworkXKnowledgeGraphStore.load(pickle_path)
    entities = nx_kg.all_entities()
    logger.info("Loaded %d entities, %d relations", nx_kg.entity_count, nx_kg.relation_count)

    # リレーション収集
    relations = []
    for entity in entities:
        for rel in nx_kg.get_relations(entity.name):
            relations.append(rel)

    if dry_run:
        logger.info("DRY RUN: would migrate %d entities, %d relations", len(entities), len(relations))
        for e in entities[:5]:
            logger.info("  Entity: %s (%s) doc_ids=%s", e.name, e.entity_type, e.doc_ids)
        for r in relations[:5]:
            logger.info("  Relation: %s -[%s]-> %s (w=%.2f)", r.source, r.relation_type, r.target, r.weight)
        return {"entities": len(entities), "relations": len(relations)}

    # 2. Neo4j に書き込み
    from src.knowledge_graph.neo4j_store import Neo4jKnowledgeGraphStore

    neo4j_kg = Neo4jKnowledgeGraphStore(
        uri=neo4j_uri, auth=neo4j_auth, database=neo4j_database,
    )

    # Entity をバッチ書き込み
    entity_count = 0
    for i in range(0, len(entities), batch_size):
        batch = entities[i:i + batch_size]
        for entity in batch:
            for doc_id in entity.doc_ids:
                neo4j_kg.add_entity(
                    name=entity.name,
                    entity_type=entity.entity_type,
                    doc_id=doc_id,
                    properties=entity.properties,
                )
            if not entity.doc_ids:
                neo4j_kg.add_entity(
                    name=entity.name,
                    entity_type=entity.entity_type,
                    properties=entity.properties,
                )
            entity_count += 1
        logger.info("Migrated %d/%d entities", entity_count, len(entities))

    # Relation をバッチ書き込み
    relation_count = 0
    for i in range(0, len(relations), batch_size):
        batch = relations[i:i + batch_size]
        for rel in batch:
            neo4j_kg.add_relation(
                source=rel.source,
                target=rel.target,
                relation_type=rel.relation_type,
                weight=rel.weight,
                properties=rel.properties,
            )
            relation_count += 1
        logger.info("Migrated %d/%d relations", relation_count, len(relations))

    neo4j_kg.close()
    logger.info("Migration complete: %d entities, %d relations", entity_count, relation_count)
    return {"entities": entity_count, "relations": relation_count}


def export_neo4j_to_json(
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_auth: tuple[str, str] = ("neo4j", "neo4j"),
    neo4j_database: str = "neo4j",
    output_path: str | Path = "data/kg_export.json",
) -> dict[str, int]:
    """Neo4j → JSON エクスポート。

    Args:
        neo4j_uri: Neo4j Bolt URI。
        neo4j_auth: (user, password)。
        neo4j_database: データベース名。
        output_path: 出力先 JSON パス。

    Returns:
        {"entities": N, "relations": N} のエクスポート件数。
    """
    from src.knowledge_graph.neo4j_store import Neo4jKnowledgeGraphStore

    neo4j_kg = Neo4jKnowledgeGraphStore(
        uri=neo4j_uri, auth=neo4j_auth, database=neo4j_database,
    )

    # Neo4j の save() が JSON エクスポートを行う
    neo4j_kg.save(output_path)
    stats = neo4j_kg.stats()
    neo4j_kg.close()

    return {"entities": stats["entity_count"], "relations": stats["relation_count"]}


def import_json_to_networkx(
    json_path: str | Path,
    output_pickle: str | Path,
) -> dict[str, int]:
    """JSON → NetworkX pickle にインポートする。

    Neo4j からエクスポートした JSON を NetworkX に復元するユーティリティ。

    Args:
        json_path: 入力 JSON パス。
        output_pickle: 出力 pickle パス。

    Returns:
        {"entities": N, "relations": N} のインポート件数。
    """
    from src.knowledge_graph.networkx_store import NetworkXKnowledgeGraphStore

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    kg = NetworkXKnowledgeGraphStore()

    for e in data.get("entities", []):
        for doc_id in e.get("doc_ids", []):
            kg.add_entity(
                name=e["name"],
                entity_type=e.get("entity_type", "concept"),
                doc_id=doc_id,
                properties=e.get("properties"),
            )
        if not e.get("doc_ids"):
            kg.add_entity(
                name=e["name"],
                entity_type=e.get("entity_type", "concept"),
                properties=e.get("properties"),
            )

    for r in data.get("relations", []):
        kg.add_relation(
            source=r["source"],
            target=r["target"],
            relation_type=r.get("relation_type", "related_to"),
            weight=r.get("weight", 1.0),
            properties=r.get("properties"),
        )

    kg.save(output_pickle)
    logger.info("Imported %d entities, %d relations to %s",
                kg.entity_count, kg.relation_count, output_pickle)
    return {"entities": kg.entity_count, "relations": kg.relation_count}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="KG データ移行ツール")
    parser.add_argument(
        "--direction",
        choices=["nx-to-neo4j", "neo4j-to-json", "json-to-nx"],
        default="nx-to-neo4j",
        help="移行方向 (default: nx-to-neo4j)",
    )
    parser.add_argument("--source", required=True, help="入力パス (pickle/URI/JSON)")
    parser.add_argument("--target", default=None, help="出力先 (URI/JSON/pickle)")
    parser.add_argument("--auth", default="neo4j:neo4j", help="Neo4j auth (user:password)")
    parser.add_argument("--database", default="neo4j", help="Neo4j database name")
    parser.add_argument("--batch-size", type=int, default=100, help="バッチサイズ")
    parser.add_argument("--dry-run", action="store_true", help="読み込みのみ")
    args = parser.parse_args()

    user, password = args.auth.split(":", 1)

    if args.direction == "nx-to-neo4j":
        target_uri = args.target or "bolt://localhost:7687"
        result = migrate_networkx_to_neo4j(
            pickle_path=args.source,
            neo4j_uri=target_uri,
            neo4j_auth=(user, password),
            neo4j_database=args.database,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
    elif args.direction == "neo4j-to-json":
        target_path = args.target or "data/kg_export.json"
        result = export_neo4j_to_json(
            neo4j_uri=args.source,
            neo4j_auth=(user, password),
            neo4j_database=args.database,
            output_path=target_path,
        )
    elif args.direction == "json-to-nx":
        target_pickle = args.target or "data/kg_restored.pkl"
        result = import_json_to_networkx(
            json_path=args.source,
            output_pickle=target_pickle,
        )
    else:
        print(f"Unknown direction: {args.direction}", file=sys.stderr)
        sys.exit(1)

    print(f"Result: {result}")


if __name__ == "__main__":
    main()
