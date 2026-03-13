"""tests/unit/test_teacher_provenance_step1.py — Teacher 素性追跡 Step 1 テスト

テスト対象:
- src/memory/schema.py の SourceMeta teacher 素性ヘルパー
- src/memory/memory_manager.py の add_from_text() teacher_id 引数
- src/memory/maturation/seed_builder.py の teacher_id 引数
"""

from __future__ import annotations

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.memory.schema import (
    Document,
    SourceMeta,
    SourceType,
    _TEACHER_ID_KEY,
    _TEACHER_PROVIDER_KEY,
    _infer_provider,
)


# ===========================================================================
# _infer_provider
# ===========================================================================

class TestInferProvider:
    def test_claude(self):
        assert _infer_provider("claude-opus-4-6") == "anthropic"

    def test_claude_haiku(self):
        assert _infer_provider("claude-haiku-4-5") == "anthropic"

    def test_gpt4o(self):
        assert _infer_provider("gpt-4o") == "openai"

    def test_gpt4o_mini(self):
        assert _infer_provider("gpt-4o-mini") == "openai"

    def test_o1(self):
        assert _infer_provider("o1-mini") == "openai"

    def test_gemini(self):
        assert _infer_provider("gemini-pro") == "google"

    def test_llama(self):
        assert _infer_provider("llama3-8b") == "ollama"

    def test_qwen(self):
        assert _infer_provider("qwen2.5-7b") == "ollama"

    def test_unknown(self):
        assert _infer_provider("my-custom-model") is None

    def test_case_insensitive(self):
        assert _infer_provider("Claude-Opus-4-6") == "anthropic"
        assert _infer_provider("GPT-4O") == "openai"


# ===========================================================================
# SourceMeta.set_teacher
# ===========================================================================

class TestSourceMetaSetTeacher:
    def test_set_teacher_basic(self):
        m = SourceMeta()
        result = m.set_teacher("claude-opus-4-6")
        # メソッドチェーン可能（self を返す）
        assert result is m

    def test_teacher_id_stored_in_extra(self):
        m = SourceMeta()
        m.set_teacher("claude-opus-4-6")
        assert m.extra[_TEACHER_ID_KEY] == "claude-opus-4-6"

    def test_provider_auto_inferred(self):
        m = SourceMeta()
        m.set_teacher("claude-opus-4-6")
        assert m.extra[_TEACHER_PROVIDER_KEY] == "anthropic"

    def test_provider_explicit_overrides_inference(self):
        m = SourceMeta()
        m.set_teacher("claude-opus-4-6", provider="custom-provider")
        assert m.extra[_TEACHER_PROVIDER_KEY] == "custom-provider"

    def test_unknown_model_no_provider_key(self):
        m = SourceMeta()
        m.set_teacher("my-model")
        assert _TEACHER_PROVIDER_KEY not in m.extra

    def test_model_key_only_when_differs(self):
        from src.memory.schema import _TEACHER_MODEL_KEY
        m = SourceMeta()
        # teacher_id と model が同じ場合はキーなし
        m.set_teacher("gpt-4o", model="gpt-4o")
        assert _TEACHER_MODEL_KEY not in m.extra

    def test_model_key_set_when_different(self):
        from src.memory.schema import _TEACHER_MODEL_KEY
        m = SourceMeta()
        m.set_teacher("gpt-4o", model="gpt-4o-2024-05-13")
        assert m.extra[_TEACHER_MODEL_KEY] == "gpt-4o-2024-05-13"

    def test_method_chaining_with_constructor(self):
        m = SourceMeta(source_type=SourceType.TEACHER, tags=["seed"]).set_teacher("gpt-4o")
        assert m.teacher_id == "gpt-4o"
        assert m.source_type == SourceType.TEACHER
        assert "seed" in m.tags

    def test_existing_extra_preserved(self):
        m = SourceMeta(extra={"key": "value"})
        m.set_teacher("claude-opus-4-6")
        assert m.extra["key"] == "value"
        assert m.extra[_TEACHER_ID_KEY] == "claude-opus-4-6"


# ===========================================================================
# SourceMeta.teacher_id / teacher_provider プロパティ
# ===========================================================================

class TestSourceMetaProperties:
    def test_teacher_id_none_when_not_set(self):
        m = SourceMeta()
        assert m.teacher_id is None

    def test_teacher_id_returns_value(self):
        m = SourceMeta()
        m.set_teacher("gpt-4o")
        assert m.teacher_id == "gpt-4o"

    def test_teacher_provider_none_when_not_set(self):
        m = SourceMeta()
        assert m.teacher_provider is None

    def test_teacher_provider_returns_value(self):
        m = SourceMeta()
        m.set_teacher("claude-opus-4-6")
        assert m.teacher_provider == "anthropic"


# ===========================================================================
# SourceMeta.is_teacher_generated
# ===========================================================================

class TestIsTeacherGenerated:
    def test_teacher_source_type(self):
        m = SourceMeta(source_type=SourceType.TEACHER)
        assert m.is_teacher_generated is True

    def test_seed_source_type(self):
        m = SourceMeta(source_type=SourceType.SEED)
        assert m.is_teacher_generated is True

    def test_manual_with_teacher_id(self):
        # extra に teacher_id があれば source_type に関わらず True
        m = SourceMeta(source_type=SourceType.MANUAL)
        m.set_teacher("claude-opus-4-6")
        assert m.is_teacher_generated is True

    def test_github_no_teacher_id(self):
        m = SourceMeta(source_type=SourceType.GITHUB)
        assert m.is_teacher_generated is False

    def test_stackoverflow_no_teacher_id(self):
        m = SourceMeta(source_type=SourceType.STACKOVERFLOW)
        assert m.is_teacher_generated is False

    def test_manual_no_teacher_id(self):
        m = SourceMeta(source_type=SourceType.MANUAL)
        assert m.is_teacher_generated is False


# ===========================================================================
# Document に埋め込まれた状態での確認
# ===========================================================================

class TestDocumentTeacherProvenance:
    def test_document_source_teacher_id(self):
        source = SourceMeta(source_type=SourceType.TEACHER)
        source.set_teacher("claude-opus-4-6")
        doc = Document(content="test", source=source)
        assert doc.source.teacher_id == "claude-opus-4-6"
        assert doc.source.teacher_provider == "anthropic"
        assert doc.source.is_teacher_generated is True

    def test_document_without_teacher(self):
        doc = Document(content="test")
        assert doc.source.teacher_id is None
        assert doc.source.is_teacher_generated is False

    def test_teacher_id_survives_model_copy(self):
        source = SourceMeta(source_type=SourceType.TEACHER)
        source.set_teacher("gpt-4o")
        doc = Document(content="original", source=source)
        doc2 = doc.model_copy(update={"content": "updated"})
        assert doc2.source.teacher_id == "gpt-4o"


# ===========================================================================
# MemoryManager.add_from_text — teacher_id 引数
# ===========================================================================

class _StubEmbedder:
    def embed(self, text):
        import numpy as np
        return np.zeros(4, dtype="float32")

    def embed_batch(self, texts):
        import numpy as np
        return [np.zeros(4, dtype="float32") for _ in texts]


class _StubFAISS:
    def add(self, domain, ids, vecs): pass
    def remove(self, domain, ids): pass
    def search(self, domain, vec, k): return [], []
    def stats(self): return {}


class _StubStore:
    def __init__(self):
        self.saved: list[Document] = []

    async def initialize(self): pass
    async def close(self): pass

    async def save(self, doc: Document) -> str:
        self.saved.append(doc)
        return doc.id

    async def get(self, doc_id): return None
    async def delete(self, doc_id): return True
    async def search_metadata(self, **kwargs): return []


class TestAddFromText:
    def _manager(self):
        from src.memory.memory_manager import MemoryManager
        store = _StubStore()
        mgr = MemoryManager(
            embedder=_StubEmbedder(),
            faiss=_StubFAISS(),
            store=store,
        )
        mgr._initialized = True
        return mgr, store

    def test_add_from_text_no_teacher(self):
        mgr, store = self._manager()
        doc_id = asyncio.get_event_loop().run_until_complete(
            mgr.add_from_text("hello world", domain="general")
        )
        assert doc_id
        saved = store.saved[0]
        assert saved.source.teacher_id is None
        assert saved.source.is_teacher_generated is False

    def test_add_from_text_with_teacher_id(self):
        mgr, store = self._manager()
        asyncio.get_event_loop().run_until_complete(
            mgr.add_from_text(
                "hello world",
                domain="code",
                teacher_id="claude-opus-4-6",
            )
        )
        saved = store.saved[0]
        assert saved.source.teacher_id == "claude-opus-4-6"
        assert saved.source.teacher_provider == "anthropic"
        assert saved.source.is_teacher_generated is True

    def test_add_from_text_teacher_provider_explicit(self):
        mgr, store = self._manager()
        asyncio.get_event_loop().run_until_complete(
            mgr.add_from_text(
                "content",
                teacher_id="my-model",
                teacher_provider="custom",
            )
        )
        saved = store.saved[0]
        assert saved.source.teacher_id == "my-model"
        assert saved.source.teacher_provider == "custom"

    def test_add_from_text_source_url(self):
        mgr, store = self._manager()
        asyncio.get_event_loop().run_until_complete(
            mgr.add_from_text("content", source_url="https://example.com")
        )
        saved = store.saved[0]
        assert saved.source.url == "https://example.com"

    def test_add_from_text_domain_stored(self):
        from src.memory.schema import Domain
        mgr, store = self._manager()
        asyncio.get_event_loop().run_until_complete(
            mgr.add_from_text("code content", domain="code")
        )
        saved = store.saved[0]
        assert saved.domain == Domain.CODE


# ===========================================================================
# SeedBuilder — teacher_id 引数
# ===========================================================================

class _FakeGateway:
    def __init__(self, response="Generated content about the topic."):
        self._resp = response

    async def complete(self, *args, **kwargs):
        class R:
            content = "Generated content about the topic."
        return R()


class TestSeedBuilderTeacherId:
    def _builder(self, teacher_id=None):
        from src.memory.maturation.seed_builder import SeedBuilder

        store = _StubStore()
        mgr = MagicMock()
        # add() が doc を受け取って store に保存する stub
        saved_docs: list[Document] = []

        async def fake_add(doc: Document) -> str:
            saved_docs.append(doc)
            return doc.id

        mgr.add = fake_add
        gw = _FakeGateway()
        builder = SeedBuilder(gw, mgr, teacher_id=teacher_id)
        return builder, saved_docs

    def test_default_teacher_id_unknown(self):
        from src.memory.maturation.seed_builder import SeedBuilder
        store = _StubStore()
        mgr = MagicMock()
        gw = _FakeGateway()
        builder = SeedBuilder(gw, mgr)
        assert builder._teacher_id == "unknown"

    def test_teacher_id_set(self):
        from src.memory.maturation.seed_builder import SeedBuilder
        store = _StubStore()
        mgr = MagicMock()
        gw = _FakeGateway()
        builder = SeedBuilder(gw, mgr, teacher_id="claude-opus-4-6")
        assert builder._teacher_id == "claude-opus-4-6"

    def test_seed_doc_has_teacher_id(self):
        builder, saved_docs = self._builder(teacher_id="gpt-4o")

        async def run():
            from src.memory.schema import DifficultyLevel
            await builder._generate_doc("FAISS", "code", DifficultyLevel.BEGINNER, "document")

        asyncio.get_event_loop().run_until_complete(run())
        assert len(saved_docs) == 1
        doc = saved_docs[0]
        assert doc.source.teacher_id == "gpt-4o"
        assert doc.source.teacher_provider == "openai"

    def test_seed_doc_unknown_teacher(self):
        builder, saved_docs = self._builder(teacher_id=None)

        async def run():
            from src.memory.schema import DifficultyLevel
            await builder._generate_doc("FAISS", "code", DifficultyLevel.BEGINNER, "document")

        asyncio.get_event_loop().run_until_complete(run())
        assert len(saved_docs) == 1
        doc = saved_docs[0]
        assert doc.source.teacher_id == "unknown"
