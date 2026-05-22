"""tests/unit/test_reviewer_worker.py — ReviewerConfig プリセット + ReviewerSession 統合テスト"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from src.cycle.reviewer_config import ReviewerConfig, SlotConfig
from src.cycle.reviewer_worker import ReviewerSession, build_task_list
from src.memory.metadata_store import MetadataStore
from src.memory.schema import Document, SourceMeta, SourceType


# ── ヘルパー ──────────────────────────────────────────────────────


async def _make_test_db(db_path: str, n: int = 3) -> None:
    """テスト用 DB を MetadataStore で初期化し、unreviewed 文書を n 件挿入する。"""
    store = MetadataStore(db_path=db_path)
    await store.initialize()
    for i in range(n):
        doc = Document(
            content=f"test content {i}",
            source=SourceMeta(source_type=SourceType.MANUAL),
        )
        await store.save(doc)
    await store.close()


# ── ReviewerConfig プリセット ─────────────────────────────────────


class TestReviewerConfigPresets:
    def test_test_preset_has_small_values(self) -> None:
        cfg = ReviewerConfig.TEST_PRESET()
        assert cfg.limit <= 5
        assert cfg.timeout_sec <= 5
        assert cfg.lock_sleep_min_ms == 0
        assert cfg.lock_sleep_max_ms <= 1
        assert cfg.ui_poll_interval_sec == 1

    def test_prod_preset_has_default_values(self) -> None:
        cfg = ReviewerConfig.PROD_PRESET()
        assert cfg.limit == 200
        assert cfg.timeout_sec == 60
        assert cfg.lock_sleep_min_ms == 100
        assert cfg.lock_sleep_max_ms == 1000
        assert cfg.ui_poll_interval_sec == 10

    def test_test_preset_accepts_overrides(self) -> None:
        cfg = ReviewerConfig.TEST_PRESET(limit=2)
        assert cfg.limit == 2
        assert cfg.lock_sleep_max_ms <= 1  # 他のプリセット値は維持

    def test_prod_preset_accepts_overrides(self) -> None:
        cfg = ReviewerConfig.PROD_PRESET(limit=50)
        assert cfg.limit == 50
        assert cfg.lock_sleep_min_ms == 100


# ── build_task_list ───────────────────────────────────────────────


class TestBuildTaskList:
    def test_build_returns_correct_count(self, tmp_path: Any) -> None:
        db = str(tmp_path / "meta.db")
        asyncio.run(_make_test_db(db, n=3))
        cfg = ReviewerConfig.TEST_PRESET(db_path=db)
        tasks = build_task_list(cfg)
        assert len(tasks) == 3

    def test_build_respects_limit(self, tmp_path: Any) -> None:
        db = str(tmp_path / "meta.db")
        asyncio.run(_make_test_db(db, n=5))
        cfg = ReviewerConfig.TEST_PRESET(db_path=db, limit=2)
        tasks = build_task_list(cfg)
        assert len(tasks) == 2

    def test_build_returns_empty_for_missing_db(self, tmp_path: Any) -> None:
        cfg = ReviewerConfig.TEST_PRESET(db_path=str(tmp_path / "nonexistent.db"))
        tasks = build_task_list(cfg)
        assert tasks == []


# ── ReviewerSession 統合テスト ────────────────────────────────────


class TestReviewerSession:
    def test_session_completes_with_mocked_reviewer(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TEST_PRESET + モック MemoryReviewer で全タスクが done になる。"""
        db = str(tmp_path / "meta.db")
        asyncio.run(_make_test_db(db, n=3))

        async def _fake_review(self_r: Any, doc: Any) -> None:
            pass

        monkeypatch.setattr(
            "src.memory.maturation.reviewer.MemoryReviewer.review", _fake_review
        )

        cfg = ReviewerConfig.TEST_PRESET(
            db_path=db,
            slots=[SlotConfig(provider="dummy", model="dummy", personas=["auto"])],
        )
        session = ReviewerSession(cfg)
        n = session.build()
        assert n == 3

        session.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if not session.get_stats()["is_running"]:
                break
            time.sleep(0.05)
        session.stop()

        stats = session.get_stats()
        assert stats["counts"]["done"] == 3
        assert stats["counts"]["error"] == 0

    def test_session_build_returns_zero_when_empty(self, tmp_path: Any) -> None:
        db = str(tmp_path / "meta.db")
        asyncio.run(_make_test_db(db, n=0))
        cfg = ReviewerConfig.TEST_PRESET(db_path=db, slots=[])
        session = ReviewerSession(cfg)
        assert session.build() == 0
