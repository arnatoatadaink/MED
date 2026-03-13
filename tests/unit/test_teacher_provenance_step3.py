"""tests/unit/test_teacher_provenance_step3.py — MetadataStore Teacher 素性テスト (Step 3)

テスト対象:
- src/memory/metadata_store.py の Teacher 素性関連機能
  - マイグレーション（teacher_id 列の追加）
  - save/get での teacher_id 往復
  - list_by_teacher / count_by_teacher
  - exclude_teacher
  - delete_by_teacher
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.memory.schema import Document, SourceMeta, SourceType


# ===========================================================================
# Helpers
# ===========================================================================

def _make_doc(content: str, teacher_id: str | None = None, domain: str = "general") -> Document:
    source = SourceMeta(source_type=SourceType.TEACHER if teacher_id else SourceType.MANUAL)
    if teacher_id:
        source.set_teacher(teacher_id)
    return Document(content=content, domain=domain, source=source)


async def _store(tmp_path: Path):
    from src.memory.metadata_store import MetadataStore
    store = MetadataStore(db_path=str(tmp_path / "test.db"))
    await store.initialize()
    return store


# ===========================================================================
# マイグレーション
# ===========================================================================

class TestMigration:
    def test_teacher_id_column_exists_after_init(self, tmp_path: Path):
        """initialize() 後に teacher_id 列が存在すること。"""
        import aiosqlite

        async def run():
            store = await _store(tmp_path)
            # PRAGMA で列確認
            async with aiosqlite.connect(str(tmp_path / "test.db")) as db:
                cursor = await db.execute("PRAGMA table_info(documents)")
                cols = {row[1] for row in await cursor.fetchall()}
            await store.close()
            return cols

        cols = asyncio.get_event_loop().run_until_complete(run())
        assert "teacher_id" in cols

    def test_migration_idempotent(self, tmp_path: Path):
        """initialize() を 2 回呼んでもエラーにならない。"""
        from src.memory.metadata_store import MetadataStore

        async def run():
            store = MetadataStore(db_path=str(tmp_path / "test.db"))
            await store.initialize()
            await store.close()
            # 2 回目
            store2 = MetadataStore(db_path=str(tmp_path / "test.db"))
            await store2.initialize()
            await store2.close()

        asyncio.get_event_loop().run_until_complete(run())  # no error


# ===========================================================================
# teacher_id の保存 / 復元
# ===========================================================================

class TestTeacherIdRoundtrip:
    def test_save_and_get_with_teacher_id(self, tmp_path: Path):
        async def run():
            store = await _store(tmp_path)
            doc = _make_doc("hello", teacher_id="claude-opus-4-6")
            await store.save(doc)
            retrieved = await store.get(doc.id)
            await store.close()
            return retrieved

        doc = asyncio.get_event_loop().run_until_complete(run())
        assert doc is not None
        assert doc.source.teacher_id == "claude-opus-4-6"
        assert doc.source.is_teacher_generated is True

    def test_save_and_get_without_teacher_id(self, tmp_path: Path):
        async def run():
            store = await _store(tmp_path)
            doc = _make_doc("no teacher here")
            await store.save(doc)
            retrieved = await store.get(doc.id)
            await store.close()
            return retrieved

        doc = asyncio.get_event_loop().run_until_complete(run())
        assert doc is not None
        assert doc.source.teacher_id is None

    def test_teacher_id_persists_across_reopen(self, tmp_path: Path):
        """DB を閉じて再オープンしても teacher_id が保持される。"""
        from src.memory.metadata_store import MetadataStore

        async def run():
            store = MetadataStore(db_path=str(tmp_path / "test.db"))
            await store.initialize()
            doc = _make_doc("persistent", teacher_id="gpt-4o")
            await store.save(doc)
            doc_id = doc.id
            await store.close()

            store2 = MetadataStore(db_path=str(tmp_path / "test.db"))
            await store2.initialize()
            retrieved = await store2.get(doc_id)
            await store2.close()
            return retrieved

        doc = asyncio.get_event_loop().run_until_complete(run())
        assert doc.source.teacher_id == "gpt-4o"


# ===========================================================================
# list_by_teacher
# ===========================================================================

class TestListByTeacher:
    def test_list_by_teacher_basic(self, tmp_path: Path):
        async def run():
            store = await _store(tmp_path)
            for i in range(3):
                await store.save(_make_doc(f"doc {i}", teacher_id="claude-opus-4-6"))
            await store.save(_make_doc("other", teacher_id="gpt-4o"))
            result = await store.list_by_teacher("claude-opus-4-6")
            await store.close()
            return result

        docs = asyncio.get_event_loop().run_until_complete(run())
        assert len(docs) == 3
        assert all(d.source.teacher_id == "claude-opus-4-6" for d in docs)

    def test_list_by_teacher_empty(self, tmp_path: Path):
        async def run():
            store = await _store(tmp_path)
            result = await store.list_by_teacher("nonexistent-teacher")
            await store.close()
            return result

        docs = asyncio.get_event_loop().run_until_complete(run())
        assert docs == []

    def test_list_by_teacher_with_domain(self, tmp_path: Path):
        async def run():
            store = await _store(tmp_path)
            await store.save(_make_doc("code doc", teacher_id="claude-opus-4-6", domain="code"))
            await store.save(_make_doc("general doc", teacher_id="claude-opus-4-6", domain="general"))
            result = await store.list_by_teacher("claude-opus-4-6", domain="code")
            await store.close()
            return result

        docs = asyncio.get_event_loop().run_until_complete(run())
        assert len(docs) == 1
        assert docs[0].domain.value == "code"

    def test_list_by_teacher_pagination(self, tmp_path: Path):
        async def run():
            store = await _store(tmp_path)
            for i in range(5):
                await store.save(_make_doc(f"doc {i}", teacher_id="model-a"))
            page1 = await store.list_by_teacher("model-a", limit=3, offset=0)
            page2 = await store.list_by_teacher("model-a", limit=3, offset=3)
            await store.close()
            return page1, page2

        page1, page2 = asyncio.get_event_loop().run_until_complete(run())
        assert len(page1) == 3
        assert len(page2) == 2
        # 重複なし
        ids1 = {d.id for d in page1}
        ids2 = {d.id for d in page2}
        assert ids1.isdisjoint(ids2)


# ===========================================================================
# count_by_teacher
# ===========================================================================

class TestCountByTeacher:
    def test_count_basic(self, tmp_path: Path):
        async def run():
            store = await _store(tmp_path)
            for i in range(4):
                await store.save(_make_doc(f"d{i}", teacher_id="model-x"))
            count = await store.count_by_teacher("model-x")
            await store.close()
            return count

        assert asyncio.get_event_loop().run_until_complete(run()) == 4

    def test_count_zero(self, tmp_path: Path):
        async def run():
            store = await _store(tmp_path)
            count = await store.count_by_teacher("nobody")
            await store.close()
            return count

        assert asyncio.get_event_loop().run_until_complete(run()) == 0

    def test_count_with_domain(self, tmp_path: Path):
        async def run():
            store = await _store(tmp_path)
            await store.save(_make_doc("code", teacher_id="model-y", domain="code"))
            await store.save(_make_doc("general", teacher_id="model-y", domain="general"))
            await store.save(_make_doc("code2", teacher_id="model-y", domain="code"))
            count = await store.count_by_teacher("model-y", domain="code")
            await store.close()
            return count

        assert asyncio.get_event_loop().run_until_complete(run()) == 2


# ===========================================================================
# exclude_teacher
# ===========================================================================

class TestExcludeTeacher:
    def test_exclude_single_teacher(self, tmp_path: Path):
        async def run():
            store = await _store(tmp_path)
            await store.save(_make_doc("good", teacher_id="trusted-model"))
            await store.save(_make_doc("bad", teacher_id="bad-model"))
            await store.save(_make_doc("no teacher"))
            result = await store.exclude_teacher(["bad-model"])
            await store.close()
            return result

        docs = asyncio.get_event_loop().run_until_complete(run())
        teacher_ids = [d.source.teacher_id for d in docs]
        assert "bad-model" not in teacher_ids
        assert len(docs) == 2  # trusted + no-teacher

    def test_exclude_multiple_teachers(self, tmp_path: Path):
        async def run():
            store = await _store(tmp_path)
            await store.save(_make_doc("doc1", teacher_id="model-a"))
            await store.save(_make_doc("doc2", teacher_id="model-b"))
            await store.save(_make_doc("doc3", teacher_id="model-c"))
            result = await store.exclude_teacher(["model-a", "model-b"])
            await store.close()
            return result

        docs = asyncio.get_event_loop().run_until_complete(run())
        assert len(docs) == 1
        assert docs[0].source.teacher_id == "model-c"

    def test_exclude_includes_null_teacher(self, tmp_path: Path):
        """teacher_id が NULL のドキュメントは除外されない。"""
        async def run():
            store = await _store(tmp_path)
            await store.save(_make_doc("null-teacher"))
            await store.save(_make_doc("excluded", teacher_id="bad-model"))
            result = await store.exclude_teacher(["bad-model"])
            await store.close()
            return result

        docs = asyncio.get_event_loop().run_until_complete(run())
        assert len(docs) == 1
        assert docs[0].source.teacher_id is None

    def test_exclude_empty_list(self, tmp_path: Path):
        """空リストを渡した場合は全件返す（domain=general）。"""
        async def run():
            store = await _store(tmp_path)
            for i in range(3):
                await store.save(_make_doc(f"doc {i}", teacher_id="any-model"))
            result = await store.exclude_teacher([])
            await store.close()
            return result

        docs = asyncio.get_event_loop().run_until_complete(run())
        assert len(docs) == 3

    def test_exclude_with_domain_filter(self, tmp_path: Path):
        async def run():
            store = await _store(tmp_path)
            await store.save(_make_doc("code", teacher_id="trusted", domain="code"))
            await store.save(_make_doc("general", teacher_id="trusted", domain="general"))
            await store.save(_make_doc("bad code", teacher_id="bad-model", domain="code"))
            result = await store.exclude_teacher(["bad-model"], domain="code")
            await store.close()
            return result

        docs = asyncio.get_event_loop().run_until_complete(run())
        assert len(docs) == 1
        assert docs[0].source.teacher_id == "trusted"


# ===========================================================================
# delete_by_teacher
# ===========================================================================

class TestDeleteByTeacher:
    def test_delete_basic(self, tmp_path: Path):
        async def run():
            store = await _store(tmp_path)
            for i in range(3):
                await store.save(_make_doc(f"doc {i}", teacher_id="bad-model"))
            await store.save(_make_doc("keep", teacher_id="good-model"))
            deleted = await store.delete_by_teacher("bad-model")
            remaining = await store.list_by_teacher("bad-model")
            await store.close()
            return deleted, remaining

        deleted, remaining = asyncio.get_event_loop().run_until_complete(run())
        assert deleted == 3
        assert remaining == []

    def test_delete_preserves_other_teachers(self, tmp_path: Path):
        async def run():
            store = await _store(tmp_path)
            await store.save(_make_doc("delete this", teacher_id="bad-model"))
            await store.save(_make_doc("keep this", teacher_id="good-model"))
            await store.delete_by_teacher("bad-model")
            good = await store.list_by_teacher("good-model")
            await store.close()
            return good

        docs = asyncio.get_event_loop().run_until_complete(run())
        assert len(docs) == 1

    def test_delete_nonexistent_returns_zero(self, tmp_path: Path):
        async def run():
            store = await _store(tmp_path)
            count = await store.delete_by_teacher("nobody")
            await store.close()
            return count

        assert asyncio.get_event_loop().run_until_complete(run()) == 0

    def test_delete_with_domain(self, tmp_path: Path):
        async def run():
            store = await _store(tmp_path)
            await store.save(_make_doc("code", teacher_id="bad-model", domain="code"))
            await store.save(_make_doc("general", teacher_id="bad-model", domain="general"))
            deleted = await store.delete_by_teacher("bad-model", domain="code")
            remaining_general = await store.list_by_teacher("bad-model", domain="general")
            await store.close()
            return deleted, remaining_general

        deleted, remaining = asyncio.get_event_loop().run_until_complete(run())
        assert deleted == 1
        assert len(remaining) == 1
