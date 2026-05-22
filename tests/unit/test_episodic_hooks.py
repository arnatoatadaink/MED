"""tests/unit/test_episodic_hooks.py — Q-3b/Q-3c エピソード記憶フックのユニットテスト"""

from __future__ import annotations

import uuid
from datetime import datetime

import numpy as np
import pytest

from src.common.config import FAISSConfig, FAISSIndexConfig, MetadataConfig
from src.memory.embedder import Embedder
from src.memory.faiss_index import FAISSIndexManager
from src.memory.memory_manager import MemoryManager
from src.memory.metadata_store import MetadataStore
from src.memory.schema import (
    Domain,
    ReviewStatus,
    SourceMeta,
    SourceType,
    ThoughtLog,
)


# ──────────────────────────────────────────────
# フィクスチャ
# ──────────────────────────────────────────────


def _make_manager() -> MemoryManager:
    faiss_cfg = FAISSConfig(
        base_dir=f"/tmp/med_faiss_episodic_{uuid.uuid4().hex}",
        indices={
            "code": FAISSIndexConfig(dim=384),
            "general": FAISSIndexConfig(dim=384),
            "episodic": FAISSIndexConfig(dim=384),
        },
    )
    meta_cfg = MetadataConfig(db_path=":memory:")
    embedder = Embedder(mock=True)
    faiss_mgr = FAISSIndexManager(faiss_cfg)
    store = MetadataStore(meta_cfg)
    return MemoryManager(embedder=embedder, faiss=faiss_mgr, store=store)


@pytest.fixture
async def manager() -> MemoryManager:
    mm = _make_manager()
    await mm.initialize()
    yield mm
    await mm.close()


# ──────────────────────────────────────────────
# スキーマ: SourceType auto-zone テスト
# ──────────────────────────────────────────────


class TestEpisodicAutoZone:
    """Document の _auto_set_episodic_zone が正しく動作するか。"""

    def test_thought_log_source_auto_episodic(self):
        from src.memory.schema import Document

        doc = Document(
            content="test",
            source=SourceMeta(source_type=SourceType.THOUGHT_LOG),
        )
        assert doc.memory_zone == "episodic"

    def test_conversation_source_auto_episodic(self):
        from src.memory.schema import Document

        doc = Document(
            content="test",
            source=SourceMeta(source_type=SourceType.CONVERSATION),
        )
        assert doc.memory_zone == "episodic"

    def test_awep_source_auto_episodic(self):
        from src.memory.schema import Document

        doc = Document(
            content="test",
            source=SourceMeta(source_type=SourceType.AWEP),
        )
        assert doc.memory_zone == "episodic"

    def test_manual_source_stays_knowledge(self):
        from src.memory.schema import Document

        doc = Document(
            content="test",
            source=SourceMeta(source_type=SourceType.MANUAL),
        )
        assert doc.memory_zone == "knowledge"


# ──────────────────────────────────────────────
# Q-3b: save_thought_log
# ──────────────────────────────────────────────


class TestSaveThoughtLog:
    @pytest.mark.asyncio
    async def test_saves_to_db_and_faiss(self, manager: MemoryManager):
        log = ThoughtLog(
            input="What is FAISS?",
            output="FAISS is a vector search library.",
            reward=0.85,
            timestamp=datetime(2026, 5, 23, 10, 0, 0),
        )
        doc_id = await manager.save_thought_log(log)

        # DB に ThoughtLog が保存されたか
        saved = await manager.store.get_thought_log(log.id)
        assert saved is not None
        assert saved.input == log.input

        # FAISS に episodic ドキュメントが追加されたか
        assert doc_id != ""
        doc = await manager.store.get(doc_id)
        assert doc is not None
        assert doc.memory_zone == "episodic"
        assert doc.domain == Domain.EPISODIC
        assert doc.source.source_type == SourceType.THOUGHT_LOG
        assert doc.review_status == ReviewStatus.APPROVED

    @pytest.mark.asyncio
    async def test_content_contains_input_and_output(self, manager: MemoryManager):
        log = ThoughtLog(
            input="input text",
            output="output text",
            reward=0.7,
        )
        doc_id = await manager.save_thought_log(log)
        doc = await manager.store.get(doc_id)
        assert "input text" in doc.content
        assert "output text" in doc.content


# ──────────────────────────────────────────────
# Q-3c: save_turn_to_episodic
# ──────────────────────────────────────────────


class TestSaveTurnToEpisodic:
    def _make_turn(self, role: str = "user", content: str = "Hello FAISS!") -> object:
        from src.conversation.schema import Turn

        return Turn(
            turn_id=uuid.uuid4().hex,
            session_id="sess-test",
            role=role,
            content=content,
            timestamp=datetime(2026, 5, 23, 11, 0, 0),
        )

    @pytest.mark.asyncio
    async def test_user_turn_saved(self, manager: MemoryManager):
        turn = self._make_turn(role="user", content="How does FAISS indexing work?")
        doc_id = await manager.save_turn_to_episodic(turn)

        assert doc_id != ""
        doc = await manager.store.get(doc_id)
        assert doc is not None
        assert doc.memory_zone == "episodic"
        assert doc.domain == Domain.EPISODIC
        assert doc.source.source_type == SourceType.CONVERSATION
        assert doc.source.extra["role"] == "user"
        assert doc.source.extra["session_id"] == "sess-test"

    @pytest.mark.asyncio
    async def test_assistant_turn_saved(self, manager: MemoryManager):
        turn = self._make_turn(role="assistant", content="FAISS uses inverted index structures.")
        doc_id = await manager.save_turn_to_episodic(turn)

        assert doc_id != ""
        doc = await manager.store.get(doc_id)
        assert doc.source.extra["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_short_turn_skipped(self, manager: MemoryManager):
        turn = self._make_turn(content="ok")
        doc_id = await manager.save_turn_to_episodic(turn)
        assert doc_id == ""

    @pytest.mark.asyncio
    async def test_turn_content_preserved(self, manager: MemoryManager):
        content = "What is the difference between HNSW and IVFFlat?"
        turn = self._make_turn(content=content)
        doc_id = await manager.save_turn_to_episodic(turn)
        doc = await manager.store.get(doc_id)
        assert doc.content == content
