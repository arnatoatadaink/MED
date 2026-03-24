"""tests/integration/test_neo4j_kg.py — Neo4j KG 統合テスト

Neo4j コンテナを使った KnowledgeGraphStore の統合テスト。
Docker が必要。CI では docker-tests ジョブで実行。

実行:
    poetry run pytest tests/integration/test_neo4j_kg.py -v --timeout=60

前提:
    pip install neo4j testcontainers
    Docker daemon が起動していること
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Neo4j テストは Docker + neo4j パッケージが必要
neo4j = pytest.importorskip("neo4j", reason="neo4j package not installed")

try:
    from testcontainers.neo4j import Neo4jContainer
    HAS_TESTCONTAINERS = True
except ImportError:
    HAS_TESTCONTAINERS = False

from src.knowledge_graph.store import KnowledgeGraphStore

pytestmark = [
    pytest.mark.docker,
    pytest.mark.slow,
    pytest.mark.skipif(not HAS_TESTCONTAINERS, reason="testcontainers not installed"),
]


@pytest.fixture(scope="module")
def neo4j_container():
    """Neo4j コンテナを起動し、テスト終了後に停止する。"""
    with Neo4jContainer("neo4j:5-community") as container:
        yield container


@pytest.fixture()
def neo4j_kg(neo4j_container):
    """テストごとにクリーンな Neo4j KG ストアを返す。"""
    from src.knowledge_graph.neo4j_store import Neo4jKnowledgeGraphStore

    uri = neo4j_container.get_connection_url()
    # testcontainers-neo4j はデフォルトパスワードを 'test' に設定
    auth = ("neo4j", "test")

    kg = Neo4jKnowledgeGraphStore(uri=uri, auth=auth)

    # テスト前にデータをクリア
    kg._run_write("MATCH (n) DETACH DELETE n")

    yield kg

    kg.close()


# ──────────────────────────────────────────────
# Entity 管理
# ──────────────────────────────────────────────


class TestNeo4jEntity:
    def test_add_and_get_entity(self, neo4j_kg) -> None:
        neo4j_kg.add_entity("Python", entity_type="language", doc_id="doc_001")
        ent = neo4j_kg.get_entity("Python")
        assert ent is not None
        assert ent.name == "Python"
        assert ent.entity_type == "language"
        assert "doc_001" in ent.doc_ids

    def test_entity_exists(self, neo4j_kg) -> None:
        neo4j_kg.add_entity("FAISS")
        assert neo4j_kg.entity_exists("FAISS")
        assert not neo4j_kg.entity_exists("NonExistent")

    def test_duplicate_merges_doc_ids(self, neo4j_kg) -> None:
        neo4j_kg.add_entity("Python", doc_id="doc_001")
        neo4j_kg.add_entity("Python", doc_id="doc_002")
        ent = neo4j_kg.get_entity("Python")
        assert ent is not None
        assert "doc_001" in ent.doc_ids
        assert "doc_002" in ent.doc_ids

    def test_remove_entity(self, neo4j_kg) -> None:
        neo4j_kg.add_entity("ToRemove")
        assert neo4j_kg.remove_entity("ToRemove")
        assert not neo4j_kg.entity_exists("ToRemove")

    def test_remove_nonexistent_returns_false(self, neo4j_kg) -> None:
        assert not neo4j_kg.remove_entity("DoesNotExist")

    def test_all_entities(self, neo4j_kg) -> None:
        neo4j_kg.add_entity("A", entity_type="language")
        neo4j_kg.add_entity("B", entity_type="library")
        entities = neo4j_kg.all_entities()
        names = {e.name for e in entities}
        assert "A" in names
        assert "B" in names

    def test_entity_count(self, neo4j_kg) -> None:
        neo4j_kg.add_entity("X")
        neo4j_kg.add_entity("Y")
        assert neo4j_kg.entity_count == 2


# ──────────────────────────────────────────────
# Relation 管理
# ──────────────────────────────────────────────


class TestNeo4jRelation:
    def test_add_relation(self, neo4j_kg) -> None:
        neo4j_kg.add_relation("Python", "FAISS", relation_type="uses", weight=0.9)
        assert neo4j_kg.entity_exists("Python")
        assert neo4j_kg.entity_exists("FAISS")
        assert neo4j_kg.relation_count == 1

    def test_get_relations(self, neo4j_kg) -> None:
        neo4j_kg.add_relation("Python", "FAISS", relation_type="uses")
        neo4j_kg.add_relation("Python", "NumPy", relation_type="uses")
        rels = neo4j_kg.get_relations("Python")
        assert len(rels) == 2

    def test_get_relations_to_target(self, neo4j_kg) -> None:
        neo4j_kg.add_relation("Python", "FAISS", relation_type="uses")
        rels = neo4j_kg.get_relations("Python", target="FAISS")
        assert len(rels) == 1
        assert rels[0].relation_type == "uses"

    def test_remove_relation(self, neo4j_kg) -> None:
        neo4j_kg.add_relation("A", "B")
        assert neo4j_kg.remove_relation("A", "B")
        assert neo4j_kg.relation_count == 0

    def test_relation_weight(self, neo4j_kg) -> None:
        neo4j_kg.add_relation("A", "B", weight=0.7)
        rels = neo4j_kg.get_relations("A")
        assert abs(rels[0].weight - 0.7) < 1e-6


# ──────────────────────────────────────────────
# クエリ
# ──────────────────────────────────────────────


class TestNeo4jQuery:
    def _seed(self, kg) -> None:
        kg.add_entity("Python", entity_type="language", doc_id="doc_py")
        kg.add_entity("FAISS", entity_type="library", doc_id="doc_faiss")
        kg.add_entity("NumPy", entity_type="library", doc_id="doc_np")
        kg.add_entity("VectorSearch", entity_type="concept", doc_id="doc_vs")
        kg.add_relation("Python", "FAISS", relation_type="uses", weight=0.9)
        kg.add_relation("Python", "NumPy", relation_type="uses", weight=0.8)
        kg.add_relation("FAISS", "VectorSearch", relation_type="implements", weight=1.0)

    def test_query_neighbors_depth1(self, neo4j_kg) -> None:
        self._seed(neo4j_kg)
        result = neo4j_kg.query_neighbors("Python", max_depth=1)
        names = {e.name for e in result.entities}
        assert "Python" in names
        assert "FAISS" in names
        assert "NumPy" in names

    def test_query_neighbors_depth2(self, neo4j_kg) -> None:
        self._seed(neo4j_kg)
        result = neo4j_kg.query_neighbors("Python", max_depth=2)
        names = {e.name for e in result.entities}
        assert "VectorSearch" in names

    def test_query_neighbors_doc_ids(self, neo4j_kg) -> None:
        self._seed(neo4j_kg)
        result = neo4j_kg.query_neighbors("Python", max_depth=1)
        assert "doc_py" in result.doc_ids

    def test_query_neighbors_nonexistent(self, neo4j_kg) -> None:
        result = neo4j_kg.query_neighbors("NonExistent")
        assert result.entities == []

    def test_find_path(self, neo4j_kg) -> None:
        self._seed(neo4j_kg)
        path = neo4j_kg.find_path("Python", "VectorSearch")
        assert len(path) > 0
        assert path[0] == "Python"
        assert path[-1] == "VectorSearch"

    def test_find_path_no_path(self, neo4j_kg) -> None:
        neo4j_kg.add_entity("Isolated_A")
        neo4j_kg.add_entity("Isolated_B")
        path = neo4j_kg.find_path("Isolated_A", "Isolated_B")
        assert path == []

    def test_get_doc_ids_for_entities(self, neo4j_kg) -> None:
        self._seed(neo4j_kg)
        ids = neo4j_kg.get_doc_ids_for_entities(["Python", "FAISS"])
        assert "doc_py" in ids
        assert "doc_faiss" in ids


# ──────────────────────────────────────────────
# エイリアス
# ──────────────────────────────────────────────


class TestNeo4jAlias:
    def test_add_and_query_alias(self, neo4j_kg) -> None:
        neo4j_kg.add_alias("Learning to Reason in 13 Parameters", "TinyLoRA", doc_id="doc_tl")
        ids = neo4j_kg.query_by_alias("TinyLoRA")
        assert "doc_tl" in ids

    def test_alias_case_insensitive(self, neo4j_kg) -> None:
        neo4j_kg.add_alias("FAISS Library", "faiss", doc_id="doc_f")
        ids = neo4j_kg.query_by_alias("FAISS")
        assert "doc_f" in ids


# ──────────────────────────────────────────────
# 統計 & 永続化
# ──────────────────────────────────────────────


class TestNeo4jStats:
    def test_stats(self, neo4j_kg) -> None:
        neo4j_kg.add_entity("A")
        neo4j_kg.add_relation("A", "B")
        s = neo4j_kg.stats()
        assert s["entity_count"] == 2
        assert s["relation_count"] == 1
        assert s["backend"] == "neo4j"

    def test_save_exports_json(self, neo4j_kg, tmp_path: Path) -> None:
        neo4j_kg.add_entity("Python", doc_id="doc_py")
        neo4j_kg.add_relation("Python", "FAISS", relation_type="uses")
        export_path = tmp_path / "kg_export.json"
        neo4j_kg.save(export_path)
        assert export_path.exists()

        import json
        with open(export_path) as f:
            data = json.load(f)
        assert data["backend"] == "neo4j"
        assert len(data["entities"]) == 2


# ──────────────────────────────────────────────
# 移行
# ──────────────────────────────────────────────


class TestMigration:
    def test_networkx_to_neo4j_migration(self, neo4j_container, tmp_path: Path) -> None:
        """NetworkX pickle → Neo4j 移行テスト。"""
        from src.knowledge_graph.migration import migrate_networkx_to_neo4j
        from src.knowledge_graph.networkx_store import NetworkXKnowledgeGraphStore

        # NetworkX で KG を作成して pickle に保存
        nx_kg = NetworkXKnowledgeGraphStore()
        nx_kg.add_entity("Python", entity_type="language", doc_id="doc_py")
        nx_kg.add_entity("FAISS", entity_type="library", doc_id="doc_faiss")
        nx_kg.add_relation("Python", "FAISS", relation_type="uses", weight=0.9)
        pickle_path = tmp_path / "test_kg.pkl"
        nx_kg.save(pickle_path)

        # Neo4j に移行
        uri = neo4j_container.get_connection_url()
        result = migrate_networkx_to_neo4j(
            pickle_path=pickle_path,
            neo4j_uri=uri,
            neo4j_auth=("neo4j", "test"),
        )
        assert result["entities"] == 2
        assert result["relations"] == 1

        # 移行先を検証
        from src.knowledge_graph.neo4j_store import Neo4jKnowledgeGraphStore
        neo4j_kg = Neo4jKnowledgeGraphStore(uri=uri, auth=("neo4j", "test"))
        assert neo4j_kg.entity_exists("Python")
        assert neo4j_kg.entity_exists("FAISS")
        rels = neo4j_kg.get_relations("Python", target="FAISS")
        assert len(rels) == 1
        assert rels[0].relation_type == "uses"
        neo4j_kg.close()

    def test_json_roundtrip(self, tmp_path: Path) -> None:
        """NetworkX → JSON → NetworkX のラウンドトリップ。"""
        from src.knowledge_graph.migration import import_json_to_networkx
        from src.knowledge_graph.networkx_store import NetworkXKnowledgeGraphStore

        # 元データ
        nx_kg = NetworkXKnowledgeGraphStore()
        nx_kg.add_entity("Python", entity_type="language", doc_id="doc_py")
        nx_kg.add_entity("FAISS", entity_type="library", doc_id="doc_faiss")
        nx_kg.add_relation("Python", "FAISS", relation_type="uses", weight=0.9)

        # JSON エクスポート (Neo4j の save 形式を模倣)
        import json
        json_data = {
            "backend": "neo4j",
            "entities": [
                {"name": e.name, "entity_type": e.entity_type, "doc_ids": e.doc_ids, "properties": e.properties}
                for e in nx_kg.all_entities()
            ],
            "relations": [
                {"source": r.source, "target": r.target, "relation_type": r.relation_type, "weight": r.weight, "properties": r.properties}
                for e in nx_kg.all_entities() for r in nx_kg.get_relations(e.name)
            ],
        }
        json_path = tmp_path / "kg.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f)

        # JSON → NetworkX pickle
        pickle_path = tmp_path / "restored.pkl"
        result = import_json_to_networkx(json_path, pickle_path)
        assert result["entities"] == 2
        assert result["relations"] == 1

        # 復元を検証
        restored = NetworkXKnowledgeGraphStore.load(pickle_path)
        assert restored.entity_exists("Python")
        assert restored.entity_exists("FAISS")
        assert restored.relation_count == 1
