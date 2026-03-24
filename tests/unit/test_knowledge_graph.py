"""tests/unit/test_knowledge_graph.py — Knowledge Graph の単体テスト"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.knowledge_graph.extractor import EntityExtractor
from src.knowledge_graph.networkx_store import NetworkXKnowledgeGraphStore
from src.knowledge_graph.router_bridge import KGRouterBridge
from src.knowledge_graph.store import KnowledgeGraphStore
from src.llm.gateway import LLMGateway, LLMResponse
from src.memory.schema import Document, SourceMeta, SourceType

# ──────────────────────────────────────────────
# フィクスチャ
# ──────────────────────────────────────────────


def _make_kg() -> KnowledgeGraphStore:
    return KnowledgeGraphStore.create(backend="networkx")


def _make_doc(content: str = "Python uses FAISS for vector search.") -> Document:
    return Document(
        content=content,
        domain="code",
        source=SourceMeta(source_type=SourceType.MANUAL),
    )


# ──────────────────────────────────────────────
# KnowledgeGraphStore — Entity 管理
# ──────────────────────────────────────────────


class TestKGEntity:
    def test_add_entity(self) -> None:
        kg = _make_kg()
        kg.add_entity("Python", entity_type="language")
        assert kg.entity_exists("Python")

    def test_add_entity_with_doc_id(self) -> None:
        kg = _make_kg()
        kg.add_entity("Python", doc_id="doc_001")
        ent = kg.get_entity("Python")
        assert ent is not None
        assert "doc_001" in ent.doc_ids

    def test_add_entity_duplicate_merges(self) -> None:
        kg = _make_kg()
        kg.add_entity("Python", doc_id="doc_001")
        kg.add_entity("Python", doc_id="doc_002")
        ent = kg.get_entity("Python")
        assert ent is not None
        assert "doc_001" in ent.doc_ids
        assert "doc_002" in ent.doc_ids

    def test_get_nonexistent_entity_returns_none(self) -> None:
        kg = _make_kg()
        assert kg.get_entity("NonExistent") is None

    def test_entity_count(self) -> None:
        kg = _make_kg()
        kg.add_entity("A")
        kg.add_entity("B")
        assert kg.entity_count == 2

    def test_remove_entity(self) -> None:
        kg = _make_kg()
        kg.add_entity("Python")
        kg.remove_entity("Python")
        assert not kg.entity_exists("Python")

    def test_remove_nonexistent_returns_false(self) -> None:
        kg = _make_kg()
        assert kg.remove_entity("NonExistent") is False

    def test_all_entities(self) -> None:
        kg = _make_kg()
        kg.add_entity("A", entity_type="language")
        kg.add_entity("B", entity_type="library")
        entities = kg.all_entities()
        names = [e.name for e in entities]
        assert "A" in names
        assert "B" in names

    def test_entity_properties(self) -> None:
        kg = _make_kg()
        kg.add_entity("Python", properties={"version": "3.11"})
        ent = kg.get_entity("Python")
        assert ent is not None
        assert ent.properties.get("version") == "3.11"


# ──────────────────────────────────────────────
# KnowledgeGraphStore — Relation 管理
# ──────────────────────────────────────────────


class TestKGRelation:
    def test_add_relation(self) -> None:
        kg = _make_kg()
        kg.add_relation("Python", "FAISS", relation_type="uses", weight=0.9)
        assert kg.entity_exists("Python")
        assert kg.entity_exists("FAISS")
        assert kg.relation_count == 1

    def test_add_relation_auto_creates_entities(self) -> None:
        kg = _make_kg()
        kg.add_relation("A", "B")
        assert kg.entity_exists("A")
        assert kg.entity_exists("B")

    def test_get_relations(self) -> None:
        kg = _make_kg()
        kg.add_relation("Python", "FAISS", relation_type="uses", weight=0.9)
        kg.add_relation("Python", "NumPy", relation_type="uses", weight=0.8)
        rels = kg.get_relations("Python")
        assert len(rels) == 2

    def test_get_relations_to_target(self) -> None:
        kg = _make_kg()
        kg.add_relation("Python", "FAISS", relation_type="uses")
        rels = kg.get_relations("Python", target="FAISS")
        assert len(rels) == 1
        assert rels[0].relation_type == "uses"

    def test_remove_relation(self) -> None:
        kg = _make_kg()
        kg.add_relation("A", "B")
        kg.remove_relation("A", "B")
        assert kg.relation_count == 0

    def test_remove_nonexistent_relation_returns_false(self) -> None:
        kg = _make_kg()
        assert kg.remove_relation("A", "B") is False

    def test_relation_weight(self) -> None:
        kg = _make_kg()
        kg.add_relation("A", "B", weight=0.7)
        rels = kg.get_relations("A")
        assert abs(rels[0].weight - 0.7) < 1e-6


# ──────────────────────────────────────────────
# KnowledgeGraphStore — クエリ
# ──────────────────────────────────────────────


class TestKGQuery:
    def _make_sample_kg(self) -> KnowledgeGraphStore:
        kg = _make_kg()
        kg.add_entity("Python", entity_type="language", doc_id="doc_py")
        kg.add_entity("FAISS", entity_type="library", doc_id="doc_faiss")
        kg.add_entity("NumPy", entity_type="library", doc_id="doc_np")
        kg.add_entity("VectorSearch", entity_type="concept", doc_id="doc_vs")
        kg.add_relation("Python", "FAISS", relation_type="uses", weight=0.9)
        kg.add_relation("Python", "NumPy", relation_type="uses", weight=0.8)
        kg.add_relation("FAISS", "VectorSearch", relation_type="implements", weight=1.0)
        return kg

    def test_query_neighbors_depth1(self) -> None:
        kg = self._make_sample_kg()
        result = kg.query_neighbors("Python", max_depth=1)
        entity_names = [e.name for e in result.entities]
        assert "Python" in entity_names
        assert "FAISS" in entity_names
        assert "NumPy" in entity_names

    def test_query_neighbors_depth2(self) -> None:
        kg = self._make_sample_kg()
        result = kg.query_neighbors("Python", max_depth=2)
        entity_names = [e.name for e in result.entities]
        assert "VectorSearch" in entity_names  # 2 ホップ先

    def test_query_neighbors_doc_ids(self) -> None:
        kg = self._make_sample_kg()
        result = kg.query_neighbors("Python", max_depth=1)
        assert "doc_py" in result.doc_ids

    def test_query_neighbors_nonexistent(self) -> None:
        kg = _make_kg()
        result = kg.query_neighbors("NonExistent")
        assert result.entities == []
        assert result.relations == []

    def test_find_path(self) -> None:
        kg = self._make_sample_kg()
        path = kg.find_path("Python", "VectorSearch")
        assert len(path) > 0
        assert path[0] == "Python"
        assert path[-1] == "VectorSearch"

    def test_find_path_no_path(self) -> None:
        kg = _make_kg()
        kg.add_entity("A")
        kg.add_entity("B")  # A と B は接続なし
        path = kg.find_path("A", "B")
        assert path == []

    def test_find_path_nonexistent_entity(self) -> None:
        kg = _make_kg()
        path = kg.find_path("NonExistent", "Also NonExistent")
        assert path == []

    def test_get_doc_ids_for_entities(self) -> None:
        kg = self._make_sample_kg()
        ids = kg.get_doc_ids_for_entities(["Python", "FAISS"])
        assert "doc_py" in ids
        assert "doc_faiss" in ids

    def test_relation_type_filter(self) -> None:
        kg = self._make_sample_kg()
        result = kg.query_neighbors("Python", max_depth=1, relation_types=["uses"])
        entity_names = [e.name for e in result.entities]
        assert "FAISS" in entity_names
        assert "NumPy" in entity_names

    def test_stats(self) -> None:
        kg = self._make_sample_kg()
        s = kg.stats()
        assert s["entity_count"] == 4
        assert s["relation_count"] == 3
        assert s["directed"] is True


# ──────────────────────────────────────────────
# KnowledgeGraphStore — 永続化
# ──────────────────────────────────────────────


class TestKGPersistence:
    def test_save_and_load(self, tmp_path: Path) -> None:
        kg = _make_kg()
        kg.add_entity("Python", entity_type="language", doc_id="doc_001")
        kg.add_relation("Python", "FAISS", relation_type="uses")

        path = tmp_path / "kg.pkl"
        kg.save(path)

        loaded = KnowledgeGraphStore.load(path)
        assert loaded.entity_exists("Python")
        assert loaded.entity_exists("FAISS")
        assert loaded.relation_count == 1

    def test_loaded_doc_ids_preserved(self, tmp_path: Path) -> None:
        kg = _make_kg()
        kg.add_entity("Python", doc_id="doc_py")
        path = tmp_path / "kg.pkl"
        kg.save(path)
        loaded = KnowledgeGraphStore.load(path)
        ent = loaded.get_entity("Python")
        assert ent is not None
        assert "doc_py" in ent.doc_ids


# ──────────────────────────────────────────────
# ファクトリ & ABC
# ──────────────────────────────────────────────


class TestKGFactory:
    def test_create_networkx(self) -> None:
        kg = KnowledgeGraphStore.create(backend="networkx")
        assert isinstance(kg, NetworkXKnowledgeGraphStore)

    def test_create_default_is_networkx(self) -> None:
        kg = KnowledgeGraphStore.create()
        assert isinstance(kg, NetworkXKnowledgeGraphStore)

    def test_create_invalid_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown KG backend"):
            KnowledgeGraphStore.create(backend="invalid")

    def test_create_neo4j_raises_without_driver(self) -> None:
        """neo4j パッケージ未インストールなら RuntimeError。"""
        # neo4j がインストールされていない環境では RuntimeError
        # インストール済みなら接続エラー (接続先がないため)
        # どちらかのエラーが出れば OK
        with pytest.raises((RuntimeError, Exception)):
            KnowledgeGraphStore.create(backend="neo4j")

    def test_load_networkx(self, tmp_path: Path) -> None:
        kg = KnowledgeGraphStore.create()
        kg.add_entity("Test", doc_id="doc_1")
        path = tmp_path / "test_kg.pkl"
        kg.save(path)
        loaded = KnowledgeGraphStore.load(path)
        assert isinstance(loaded, NetworkXKnowledgeGraphStore)
        assert loaded.entity_exists("Test")

    def test_load_invalid_backend_raises(self) -> None:
        with pytest.raises(ValueError):
            KnowledgeGraphStore.load("/tmp/fake.pkl", backend="invalid")

    def test_stats_includes_backend(self) -> None:
        kg = KnowledgeGraphStore.create()
        assert kg.stats()["backend"] == "networkx"


# ──────────────────────────────────────────────
# EntityExtractor
# ──────────────────────────────────────────────


class MockGateway(LLMGateway):
    def __init__(self, response: str) -> None:
        self._response = response
        self._providers = {}
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0

    async def complete(self, prompt, **kwargs) -> LLMResponse:
        return LLMResponse(content=self._response, provider="mock", model="mock", input_tokens=10, output_tokens=50)


_VALID_JSON = """{
  "entities": [
    {"name": "Python", "type": "language"},
    {"name": "FAISS", "type": "library"}
  ],
  "relations": [
    {"source": "Python", "target": "FAISS", "type": "uses", "weight": 0.9}
  ]
}"""


class TestEntityExtractor:
    @pytest.mark.asyncio
    async def test_extract_valid_json(self) -> None:
        gw = MockGateway(_VALID_JSON)
        extractor = EntityExtractor(gw)
        doc = _make_doc()
        result = await extractor.extract(doc)
        assert result.success
        assert len(result.entities) == 2
        assert len(result.relations) == 1

    @pytest.mark.asyncio
    async def test_extract_invalid_json_returns_empty(self) -> None:
        gw = MockGateway("not valid json at all")
        extractor = EntityExtractor(gw)
        doc = _make_doc()
        result = await extractor.extract(doc)
        # JSON パース失敗 → 空リストで success=True (LLM は呼べた)
        assert result.entities == []
        assert result.relations == []

    @pytest.mark.asyncio
    async def test_extract_and_register(self) -> None:
        gw = MockGateway(_VALID_JSON)
        extractor = EntityExtractor(gw)
        kg = _make_kg()
        doc = _make_doc()
        result = await extractor.extract_and_register(doc, kg)
        assert result.success
        assert kg.entity_exists("Python")
        assert kg.entity_exists("FAISS")
        assert kg.relation_count == 1

    @pytest.mark.asyncio
    async def test_extract_registers_doc_id(self) -> None:
        gw = MockGateway(_VALID_JSON)
        extractor = EntityExtractor(gw)
        kg = _make_kg()
        doc = _make_doc()
        await extractor.extract_and_register(doc, kg)
        ent = kg.get_entity("Python")
        assert ent is not None
        assert doc.id in ent.doc_ids

    @pytest.mark.asyncio
    async def test_failing_gateway_returns_failed_result(self) -> None:
        class FailingGateway(LLMGateway):
            def __init__(self):
                self._providers = {}
                self._total_input_tokens = 0
                self._total_output_tokens = 0
                self._call_count = 0
            async def complete(self, *args, **kwargs):
                raise RuntimeError("LLM failed")

        extractor = EntityExtractor(FailingGateway())
        doc = _make_doc()
        result = await extractor.extract(doc)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_extract_json_in_code_block(self) -> None:
        json_in_block = f"```json\n{_VALID_JSON}\n```"
        gw = MockGateway(json_in_block)
        extractor = EntityExtractor(gw)
        result = await extractor.extract(_make_doc())
        assert result.success
        assert len(result.entities) == 2


# ──────────────────────────────────────────────
# KGRouterBridge
# ──────────────────────────────────────────────


class TestKGRouterBridge:
    def _make_bridge_with_data(self) -> tuple[KGRouterBridge, KnowledgeGraphStore]:
        kg = _make_kg()
        kg.add_entity("Python", doc_id="doc_py")
        kg.add_entity("FAISS", doc_id="doc_faiss")
        kg.add_entity("NumPy", doc_id="doc_np")
        kg.add_relation("Python", "FAISS", relation_type="uses")
        kg.add_relation("Python", "NumPy", relation_type="uses")
        bridge = KGRouterBridge(kg, extractor=None)
        return bridge, kg

    def test_expand_query_returns_doc_ids(self) -> None:
        bridge, _ = self._make_bridge_with_data()
        ids = bridge.expand_query_with_kg(["Python"], max_depth=1)
        assert "doc_py" in ids
        assert "doc_faiss" in ids

    def test_expand_query_nonexistent_entity(self) -> None:
        bridge, _ = self._make_bridge_with_data()
        ids = bridge.expand_query_with_kg(["NonExistent"])
        assert ids == []

    def test_get_related_entities(self) -> None:
        bridge, _ = self._make_bridge_with_data()
        related = bridge.get_related_entities(["Python"])
        assert "FAISS" in related
        assert "NumPy" in related
        assert "Python" not in related  # 自分自身は除外

    def test_find_connecting_path(self) -> None:
        kg = _make_kg()
        kg.add_relation("A", "B")
        kg.add_relation("B", "C")
        bridge = KGRouterBridge(kg)
        path = bridge.find_connecting_path("A", "C")
        assert path == ["A", "B", "C"]

    def test_stats(self) -> None:
        bridge, _ = self._make_bridge_with_data()
        s = bridge.stats
        assert s["entity_count"] == 3
        assert s["relation_count"] == 2

    @pytest.mark.asyncio
    async def test_on_document_added_no_extractor(self) -> None:
        """extractor なし → スキップ（エラーなし）。"""
        kg = _make_kg()
        bridge = KGRouterBridge(kg, extractor=None)
        doc = _make_doc()
        await bridge.on_document_added(doc)  # should not raise
        assert kg.entity_count == 0

    @pytest.mark.asyncio
    async def test_on_document_added_with_extractor(self) -> None:
        gw = MockGateway(_VALID_JSON)
        extractor = EntityExtractor(gw)
        kg = _make_kg()
        bridge = KGRouterBridge(kg, extractor=extractor)
        doc = _make_doc()
        await bridge.on_document_added(doc)
        assert kg.entity_exists("Python")
