"""src/cycle/schema.py — サイクル管理の共通データ型"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class GapType(str, Enum):
    """ギャップの種類 — downstream action を分岐させる。"""

    SMALL_CLUSTER     = "small_cluster"      # 小さい島 → 収集
    UNREVIEWED_BACKLOG = "unreviewed_backlog" # 未レビュー多い → mature
    SOURCE_IMBALANCE  = "source_imbalance"   # ソース偏り → 別ソースで収集
    LOW_QUALITY       = "low_quality"        # 低品質クラスタ → re-mature


@dataclass
class CollectionTask:
    """1つのギャップ検出結果。

    P1b: QueryGenerator が signals を読んで keywords / query を追加
    P1c: DB に保存
    P1d: Orchestrator が gap_type で分岐
    P3/P4: WebGUI で表示・上書き

    Attributes:
        task_id:    UUID (文字列)
        created_at: ISO8601 UTC タイムスタンプ
        gap_type:   ギャップ種別
        signals:    ギャップ詳細（cluster_id, source_dist, etc.）
        priority:   0.0–1.0（高いほど優先）
        reason:     人間が読める説明
        keywords:   検索キーワード（P1b が追加、初期値 []）
        queries:    実際の検索クエリ文字列（P1b が追加、初期値 []）
    """

    gap_type:   GapType
    signals:    dict[str, Any]
    priority:   float
    reason:     str
    task_id:    str          = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str          = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    keywords:   list[str]    = field(default_factory=list)
    queries:    list[str]    = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"[{self.gap_type.value}] priority={self.priority:.2f}  {self.reason}"
        )
