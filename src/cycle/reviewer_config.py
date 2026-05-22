"""src/cycle/reviewer_config.py — Reviewer 設定データクラス群

ReviewTask / SlotConfig / ReviewerConfig を定義する。
ワーカーロジック（reviewer_worker.py）から分離して依存関係を軽量に保つ。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReviewTask:
    doc_id: str
    source_type: str
    domain_flag: str
    status: str = "pending"       # pending | in_progress | done | error
    assigned_to: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


@dataclass
class SlotConfig:
    provider: str
    model: str
    personas: list[str]           # このスロットが処理できるペルソナ


@dataclass
class ReviewerConfig:
    slots: list[SlotConfig] = field(default_factory=list)
    limit: int = 200
    timeout_sec: int = 60
    db_path: str = "data/metadata.db"
    include_low_quality: bool = True
    lock_sleep_min_ms: int = 100
    lock_sleep_max_ms: int = 1000
    ui_poll_interval_sec: int = 10

    @classmethod
    def TEST_PRESET(cls, **kwargs: object) -> "ReviewerConfig":
        """テスト用プリセット: スリープ最小化・件数上限小・タイムアウト短。"""
        defaults: dict[str, object] = {
            "limit": 5,
            "timeout_sec": 5,
            "lock_sleep_min_ms": 0,
            "lock_sleep_max_ms": 1,
            "ui_poll_interval_sec": 1,
        }
        defaults.update(kwargs)
        return cls(**defaults)  # type: ignore[arg-type]

    @classmethod
    def PROD_PRESET(cls, **kwargs: object) -> "ReviewerConfig":
        """本番用プリセット: デフォルト値をそのまま適用。"""
        return cls(**kwargs)  # type: ignore[arg-type]
