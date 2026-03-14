"""tests/unit/test_teacher_provenance_step2.py — TeacherRegistry Step 2 テスト

テスト対象:
- src/memory/teacher_registry.py
- MemoryManager への TeacherRegistry 統合
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.memory.teacher_registry import (
    _EWMA_ALPHA,
    _MIN_TRUST,
    _WARMUP_N,
    TeacherProfile,
    TeacherRegistry,
    _ewma_update,
)

# ===========================================================================
# _ewma_update — ユニット
# ===========================================================================

class TestEwmaUpdate:
    def test_warmup_welford(self):
        # n=1 では reward そのものになる
        result = _ewma_update(0.5, 0.9, n=1)
        assert result == pytest.approx(0.9)

    def test_warmup_moving_avg(self):
        # n=2: (0.5 + 0.9) / 2 = 0.7 ではなく Welford 法
        # old=0.9, reward=0.3, n=2 → 0.9 + (0.3 - 0.9)/2 = 0.6
        result = _ewma_update(0.9, 0.3, n=2)
        assert result == pytest.approx(0.6)

    def test_post_warmup_ewma(self):
        # n > WARMUP_N: EWMA
        old = 0.7
        reward = 1.0
        result = _ewma_update(old, reward, n=_WARMUP_N + 1)
        expected = old * (1 - _EWMA_ALPHA) + reward * _EWMA_ALPHA
        assert result == pytest.approx(expected)

    def test_warmup_boundary(self):
        # n=WARMUP_N はまだ Welford 法
        result = _ewma_update(0.5, 0.5, n=_WARMUP_N)
        expected = 0.5 + (0.5 - 0.5) / _WARMUP_N
        assert result == pytest.approx(expected)


# ===========================================================================
# TeacherProfile — ユニット
# ===========================================================================

class TestTeacherProfile:
    def _profile(self, trust_score: float = 0.8, **kwargs) -> TeacherProfile:
        from datetime import datetime
        return TeacherProfile(
            teacher_id="test-model",
            provider="test",
            trust_score=trust_score,
            total_docs=10,
            avg_reward=trust_score,
            n_feedback=5,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    def test_is_trusted_high(self):
        p = self._profile(trust_score=0.8)
        assert p.is_trusted is True

    def test_is_trusted_boundary(self):
        p = self._profile(trust_score=0.5)
        assert p.is_trusted is True

    def test_not_trusted(self):
        p = self._profile(trust_score=0.4)
        assert p.is_trusted is False

    def test_is_low_trust(self):
        p = self._profile(trust_score=0.2)
        assert p.is_low_trust is True

    def test_not_low_trust(self):
        p = self._profile(trust_score=0.3)
        assert p.is_low_trust is False

    def test_summary_contains_key_fields(self):
        p = self._profile(trust_score=0.75)
        s = p.summary()
        assert "test-model" in s
        assert "0.750" in s


# ===========================================================================
# TeacherRegistry — 統合テスト（SQLite）
# ===========================================================================

@pytest.fixture
def registry(tmp_path: Path) -> TeacherRegistry:
    reg = TeacherRegistry(tmp_path / "test.db")
    asyncio.get_event_loop().run_until_complete(reg.initialize())
    return reg


class TestTeacherRegistryEnsure:
    def test_ensure_creates_profile(self, registry: TeacherRegistry):
        profile = asyncio.get_event_loop().run_until_complete(
            registry.ensure("claude-opus-4-6")
        )
        assert profile.teacher_id == "claude-opus-4-6"
        assert profile.provider == "anthropic"  # auto-inferred
        assert profile.trust_score == pytest.approx(1.0)
        assert profile.total_docs == 0
        assert profile.n_feedback == 0

    def test_ensure_idempotent(self, registry: TeacherRegistry):
        p1 = asyncio.get_event_loop().run_until_complete(
            registry.ensure("gpt-4o", provider="openai")
        )
        p2 = asyncio.get_event_loop().run_until_complete(
            registry.ensure("gpt-4o")
        )
        assert p1.teacher_id == p2.teacher_id
        assert p1.created_at == p2.created_at  # 同じレコード

    def test_ensure_explicit_provider(self, registry: TeacherRegistry):
        p = asyncio.get_event_loop().run_until_complete(
            registry.ensure("my-model", provider="custom-cloud")
        )
        assert p.provider == "custom-cloud"

    def test_ensure_unknown_model_no_provider(self, registry: TeacherRegistry):
        p = asyncio.get_event_loop().run_until_complete(
            registry.ensure("unknown-model-xyz")
        )
        assert p.provider is None


class TestTeacherRegistryGet:
    def test_get_existing(self, registry: TeacherRegistry):
        asyncio.get_event_loop().run_until_complete(
            registry.ensure("claude-haiku-4-5")
        )
        p = asyncio.get_event_loop().run_until_complete(
            registry.get("claude-haiku-4-5")
        )
        assert p is not None
        assert p.teacher_id == "claude-haiku-4-5"

    def test_get_missing_returns_none(self, registry: TeacherRegistry):
        p = asyncio.get_event_loop().run_until_complete(
            registry.get("nonexistent")
        )
        assert p is None


class TestTeacherRegistryListAll:
    def test_list_all_empty(self, registry: TeacherRegistry):
        result = asyncio.get_event_loop().run_until_complete(registry.list_all())
        assert result == []

    def test_list_all_sorted_by_trust(self, registry: TeacherRegistry):
        async def setup():
            await registry.ensure("model-a")
            await registry.set_trust("model-a", 0.9)
            await registry.ensure("model-b")
            await registry.set_trust("model-b", 0.3)
            await registry.ensure("model-c")
            await registry.set_trust("model-c", 0.6)
            return await registry.list_all()

        profiles = asyncio.get_event_loop().run_until_complete(setup())
        assert len(profiles) == 3
        scores = [p.trust_score for p in profiles]
        assert scores == sorted(scores, reverse=True)


class TestTeacherRegistryGetLowTrust:
    def test_low_trust_filter(self, registry: TeacherRegistry):
        async def setup():
            await registry.ensure("good-model")
            await registry.set_trust("good-model", 0.8)
            await registry.ensure("bad-model")
            await registry.set_trust("bad-model", 0.1)
            return await registry.get_low_trust(threshold=0.3)

        low = asyncio.get_event_loop().run_until_complete(setup())
        assert len(low) == 1
        assert low[0].teacher_id == "bad-model"

    def test_low_trust_default_threshold(self, registry: TeacherRegistry):
        async def setup():
            await registry.ensure("borderline")
            await registry.set_trust("borderline", 0.29)
            return await registry.get_low_trust()

        low = asyncio.get_event_loop().run_until_complete(setup())
        assert any(p.teacher_id == "borderline" for p in low)


class TestTeacherRegistryRecordDoc:
    def test_record_doc_increments(self, registry: TeacherRegistry):
        async def run():
            await registry.ensure("claude-opus-4-6")
            await registry.record_doc("claude-opus-4-6")
            await registry.record_doc("claude-opus-4-6")
            return await registry.get("claude-opus-4-6")

        p = asyncio.get_event_loop().run_until_complete(run())
        assert p.total_docs == 2

    def test_record_doc_auto_ensures(self, registry: TeacherRegistry):
        # ensure を呼ばなくても record_doc で自動登録される
        async def run():
            await registry.record_doc("new-teacher")
            return await registry.get("new-teacher")

        p = asyncio.get_event_loop().run_until_complete(run())
        assert p is not None
        assert p.total_docs == 1


class TestTeacherRegistryRecordFeedback:
    def test_feedback_increases_trust_when_reward_high(self, registry: TeacherRegistry):
        async def run():
            await registry.ensure("model-x")
            # 初期 avg_reward=0.5, trust=1.0
            # 高い reward を連続投入
            for _ in range(5):
                await registry.record_feedback("model-x", reward=1.0)
            return await registry.get("model-x")

        p = asyncio.get_event_loop().run_until_complete(run())
        assert p.avg_reward > 0.5
        assert p.n_feedback == 5

    def test_feedback_decreases_trust_when_reward_low(self, registry: TeacherRegistry):
        async def run():
            await registry.ensure("model-y")
            for _ in range(15):
                await registry.record_feedback("model-y", reward=0.0)
            return await registry.get("model-y")

        p = asyncio.get_event_loop().run_until_complete(run())
        assert p.avg_reward < 0.5
        assert p.trust_score >= _MIN_TRUST  # 下限を下回らない

    def test_trust_clamped_at_min(self, registry: TeacherRegistry):
        async def run():
            await registry.ensure("bad-teacher")
            # 最悪スコアを大量に投入
            for _ in range(50):
                await registry.record_feedback("bad-teacher", reward=0.0)
            return await registry.get("bad-teacher")

        p = asyncio.get_event_loop().run_until_complete(run())
        assert p.trust_score >= _MIN_TRUST

    def test_reward_clamp_upper(self, registry: TeacherRegistry):
        async def run():
            await registry.ensure("m")
            return await registry.record_feedback("m", reward=1.5)  # 1.0 にクランプ

        p = asyncio.get_event_loop().run_until_complete(run())
        assert p.avg_reward <= 1.0

    def test_reward_clamp_lower(self, registry: TeacherRegistry):
        async def run():
            await registry.ensure("m")
            return await registry.record_feedback("m", reward=-0.5)  # 0.0 にクランプ

        p = asyncio.get_event_loop().run_until_complete(run())
        assert p.avg_reward >= 0.0


class TestTeacherRegistrySetTrust:
    def test_set_trust_direct(self, registry: TeacherRegistry):
        async def run():
            await registry.ensure("model-z")
            return await registry.set_trust("model-z", 0.42)

        p = asyncio.get_event_loop().run_until_complete(run())
        assert p.trust_score == pytest.approx(0.42)

    def test_set_trust_clamp_upper(self, registry: TeacherRegistry):
        async def run():
            await registry.ensure("m")
            return await registry.set_trust("m", 1.5)

        p = asyncio.get_event_loop().run_until_complete(run())
        assert p.trust_score == pytest.approx(1.0)

    def test_set_trust_clamp_lower(self, registry: TeacherRegistry):
        async def run():
            await registry.ensure("m")
            return await registry.set_trust("m", -0.1)

        p = asyncio.get_event_loop().run_until_complete(run())
        assert p.trust_score == pytest.approx(0.0)

    def test_reset_trust(self, registry: TeacherRegistry):
        async def run():
            await registry.ensure("m")
            await registry.set_trust("m", 0.1)
            return await registry.reset_trust("m")

        p = asyncio.get_event_loop().run_until_complete(run())
        assert p.trust_score == pytest.approx(1.0)


# ===========================================================================
# MemoryManager + TeacherRegistry 統合
# ===========================================================================

import numpy as np

from src.memory.schema import Document, SourceMeta, SourceType


class _StubEmbedder:
    def embed(self, text):
        return np.zeros(4, dtype="float32")

    def embed_batch(self, texts):
        return [np.zeros(4, dtype="float32") for _ in texts]


class _StubFAISS:
    def add(self, domain, ids, vecs): pass
    def remove(self, domain, ids): pass
    def search(self, domain, vec, k): return [], []
    def stats(self): return {}


class _StubMetadataStore:
    async def initialize(self): pass
    async def close(self): pass
    async def save(self, doc: Document) -> str: return doc.id
    async def get(self, doc_id): return None
    async def delete(self, doc_id): return True
    async def search_metadata(self, **kwargs): return []


class TestMemoryManagerTeacherRegistry:
    def _manager(self, tmp_path: Path):
        from src.memory.memory_manager import MemoryManager
        registry = TeacherRegistry(tmp_path / "reg.db")
        mgr = MemoryManager(
            embedder=_StubEmbedder(),
            faiss=_StubFAISS(),
            store=_StubMetadataStore(),
            teacher_registry=registry,
        )
        mgr._initialized = True
        return mgr, registry

    def test_teacher_registry_attribute(self, tmp_path: Path):
        from src.memory.memory_manager import MemoryManager
        mgr = MemoryManager(
            embedder=_StubEmbedder(),
            faiss=_StubFAISS(),
            store=_StubMetadataStore(),
        )
        assert mgr.teacher_registry is None

    def test_teacher_registry_set(self, tmp_path: Path):
        mgr, registry = self._manager(tmp_path)
        assert mgr.teacher_registry is registry

    def test_add_without_teacher_id_no_registry_call(self, tmp_path: Path):
        mgr, registry = self._manager(tmp_path)

        async def run():
            await registry.initialize()
            doc = Document(content="hello", source=SourceMeta())
            await mgr.add(doc)
            return await registry.list_all()

        profiles = asyncio.get_event_loop().run_until_complete(run())
        assert profiles == []  # teacher なし → registry に何も登録されない

    def test_add_with_teacher_id_registers(self, tmp_path: Path):
        mgr, registry = self._manager(tmp_path)

        async def run():
            await registry.initialize()
            source = SourceMeta(source_type=SourceType.TEACHER)
            source.set_teacher("claude-opus-4-6")
            doc = Document(content="content", source=source)
            await mgr.add(doc)
            return await registry.get("claude-opus-4-6")

        p = asyncio.get_event_loop().run_until_complete(run())
        assert p is not None
        assert p.total_docs == 1

    def test_add_multiple_docs_same_teacher(self, tmp_path: Path):
        mgr, registry = self._manager(tmp_path)

        async def run():
            await registry.initialize()
            for i in range(3):
                source = SourceMeta(source_type=SourceType.TEACHER)
                source.set_teacher("gpt-4o")
                await mgr.add(Document(content=f"doc {i}", source=source))
            return await registry.get("gpt-4o")

        p = asyncio.get_event_loop().run_until_complete(run())
        assert p.total_docs == 3

    def test_add_without_registry_does_not_crash(self, tmp_path: Path):
        from src.memory.memory_manager import MemoryManager
        mgr = MemoryManager(
            embedder=_StubEmbedder(),
            faiss=_StubFAISS(),
            store=_StubMetadataStore(),
            teacher_registry=None,
        )
        mgr._initialized = True

        async def run():
            source = SourceMeta(source_type=SourceType.TEACHER)
            source.set_teacher("claude-opus-4-6")
            return await mgr.add(Document(content="test", source=source))

        doc_id = asyncio.get_event_loop().run_until_complete(run())
        assert doc_id  # クラッシュせずに doc_id が返る

    def test_initialize_also_initializes_registry(self, tmp_path: Path):
        from src.memory.memory_manager import MemoryManager
        registry = TeacherRegistry(tmp_path / "reg2.db")
        mgr = MemoryManager(
            embedder=_StubEmbedder(),
            faiss=_StubFAISS(),
            store=_StubMetadataStore(),
            teacher_registry=registry,
        )

        async def run():
            await mgr.initialize()
            # initialize 後は Registry も使える
            await registry.ensure("model-a")
            return await registry.get("model-a")

        p = asyncio.get_event_loop().run_until_complete(run())
        assert p is not None
