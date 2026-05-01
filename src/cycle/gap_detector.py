"""src/cycle/gap_detector.py — FAISS メモリのギャップ検出

UMAP キャッシュの島分析から CollectionTask リストを生成する。
トピック文字列や検索クエリは生成しない（それは P1b QueryGenerator の仕事）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.cycle.schema import CollectionTask, GapType
from src.cycle.umap_islands import (
    CacheStaleError,
    Island,
    IslandSet,
    compute_islands,
)

logger = logging.getLogger(__name__)

# ---- 検出パラメータ (tuneable) ----------------------------------------

@dataclass
class GapDetectorConfig:
    """GapDetector のしきい値設定。"""

    # SMALL_CLUSTER: size がこれ未満かつ q_avg がこれ以上のクラスタを収集対象とする
    small_cluster_max_size:     int   = 200
    small_cluster_min_quality:  float = 0.4

    # UNREVIEWED_BACKLOG: unreviewed 比率がこれ以上 かつ size がこれ以上
    unreviewed_threshold:       float = 0.60
    unreviewed_min_size:        int   = 50

    # SOURCE_IMBALANCE: 1ソースの占有率がこれ以上 かつ size がこれ以上
    source_imbalance_threshold: float = 0.80
    source_imbalance_min_size:  int   = 100

    # LOW_QUALITY: q_avg がこれ未満 かつ size がこれ以上
    low_quality_threshold:      float = 0.30
    low_quality_min_size:       int   = 100

    # 出力上限
    max_tasks:                  int   = 20


_DEFAULT_CONFIG = GapDetectorConfig()


# ---- GapDetector ----------------------------------------------------

class GapDetector:
    """UMAP島分析に基づくギャップ検出器。

    Args:
        cache_path:        umap_cache.npz のパス。
        db_path:           metadata.db のパス。
        config:            検出パラメータ。
        raise_if_stale:    True の場合、キャッシュ未存在 or stale で例外。
    """

    def __init__(
        self,
        cache_path: str | Path = "data/umap_cache.npz",
        db_path: str = "data/metadata.db",
        config: Optional[GapDetectorConfig] = None,
        raise_if_stale: bool = False,
    ) -> None:
        self._cache_path    = Path(cache_path)
        self._db_path       = db_path
        self._cfg           = config or _DEFAULT_CONFIG
        self._raise_if_stale = raise_if_stale

    def detect(self) -> list[CollectionTask]:
        """キャッシュを読み込み CollectionTask リストを返す。

        Returns:
            priority 降順でソートされた CollectionTask リスト。

        Raises:
            CacheStaleError: raise_if_stale=True かつキャッシュが無効な場合。
        """
        island_set = compute_islands(
            cache_path=self._cache_path,
            db_path=self._db_path,
            raise_if_stale=self._raise_if_stale,
        )

        if island_set.stale and not self._raise_if_stale:
            logger.warning("UMAP cache is stale — gap detection may be inaccurate")

        if island_set.n_docs == 0:
            logger.info("No UMAP data available; returning empty task list")
            return []

        tasks: list[CollectionTask] = []
        tasks.extend(self._detect_small_clusters(island_set))
        tasks.extend(self._detect_unreviewed_backlog(island_set))
        tasks.extend(self._detect_source_imbalance(island_set))
        tasks.extend(self._detect_low_quality(island_set))

        tasks.sort(key=lambda t: t.priority, reverse=True)
        tasks = tasks[: self._cfg.max_tasks]

        logger.info(
            "Gap detection complete: %d tasks from %d islands",
            len(tasks), island_set.n_clusters,
        )
        return tasks

    # ---- 検出ロジック -----------------------------------------------

    def _detect_small_clusters(self, iset: IslandSet) -> list[CollectionTask]:
        """サイズが小さく品質が十分な島 → 収集で拡充。"""
        tasks: list[CollectionTask] = []
        cfg = self._cfg

        for island in iset.islands:
            if (
                island.size < cfg.small_cluster_max_size
                and island.q_avg >= cfg.small_cluster_min_quality
            ):
                priority = _scale(
                    cfg.small_cluster_max_size - island.size,
                    0, cfg.small_cluster_max_size,
                ) * 0.9  # max 0.9

                tasks.append(CollectionTask(
                    gap_type=GapType.SMALL_CLUSTER,
                    priority=priority,
                    reason=(
                        f"Island #{island.id} has only {island.size} docs "
                        f"(q={island.q_avg:.2f}). Needs more documents."
                    ),
                    signals=_island_signals(island, iset),
                ))

        return tasks

    def _detect_unreviewed_backlog(self, iset: IslandSet) -> list[CollectionTask]:
        """未レビュー比率が高い島 → mature バックログ。"""
        tasks: list[CollectionTask] = []
        cfg = self._cfg

        for island in iset.islands:
            if (
                island.size >= cfg.unreviewed_min_size
                and island.unreviewed_pct >= cfg.unreviewed_threshold
            ):
                priority = _scale(
                    island.unreviewed_pct - cfg.unreviewed_threshold,
                    0.0, 1.0 - cfg.unreviewed_threshold,
                ) * 0.8

                tasks.append(CollectionTask(
                    gap_type=GapType.UNREVIEWED_BACKLOG,
                    priority=priority,
                    reason=(
                        f"Island #{island.id}: {island.unreviewed_pct * 100:.0f}% unreviewed "
                        f"({island.size} docs). Prioritize maturation."
                    ),
                    signals=_island_signals(island, iset),
                ))

        return tasks

    def _detect_source_imbalance(self, iset: IslandSet) -> list[CollectionTask]:
        """1ソースに偏った中規模以上の島 → 別ソースで多様化。"""
        tasks: list[CollectionTask] = []
        cfg = self._cfg

        for island in iset.islands:
            if (
                island.size >= cfg.source_imbalance_min_size
                and island.dominant_source_pct >= cfg.source_imbalance_threshold
            ):
                priority = _scale(
                    island.dominant_source_pct - cfg.source_imbalance_threshold,
                    0.0, 1.0 - cfg.source_imbalance_threshold,
                ) * 0.7

                tasks.append(CollectionTask(
                    gap_type=GapType.SOURCE_IMBALANCE,
                    priority=priority,
                    reason=(
                        f"Island #{island.id}: {island.dominant_source_pct * 100:.0f}% "
                        f"from '{island.dominant_source}' ({island.size} docs). "
                        "Diversify with alternative sources."
                    ),
                    signals=_island_signals(island, iset),
                ))

        return tasks

    def _detect_low_quality(self, iset: IslandSet) -> list[CollectionTask]:
        """品質スコアが低い中規模以上の島 → re-mature が有効。"""
        tasks: list[CollectionTask] = []
        cfg = self._cfg

        for island in iset.islands:
            if (
                island.size >= cfg.low_quality_min_size
                and island.q_avg < cfg.low_quality_threshold
            ):
                priority = _scale(
                    cfg.low_quality_threshold - island.q_avg,
                    0.0, cfg.low_quality_threshold,
                ) * 0.6

                tasks.append(CollectionTask(
                    gap_type=GapType.LOW_QUALITY,
                    priority=priority,
                    reason=(
                        f"Island #{island.id}: q_avg={island.q_avg:.2f} "
                        f"({island.size} docs). Consider re-maturation."
                    ),
                    signals=_island_signals(island, iset),
                ))

        return tasks


# ---- ヘルパー -------------------------------------------------------

def _island_signals(island: Island, iset: IslandSet) -> dict:
    """CollectionTask.signals に格納するクラスタ情報。"""
    return {
        "cluster_id":       island.id,
        "size":             island.size,
        "q_avg":            round(island.q_avg, 3),
        "centroid":         list(island.centroid),
        "source_dist":      island.source_dist,
        "review_dist":      island.review_dist,
        "dominant_source":  island.dominant_source,
        "dominant_source_pct": round(island.dominant_source_pct, 3),
        "approved_pct":     round(island.approved_pct, 3),
        "unreviewed_pct":   round(island.unreviewed_pct, 3),
        "sample_doc_ids":   island.doc_ids[:10],
        "n_total_islands":  iset.n_clusters,
        "cache_age_h":      round(iset.cache_age_h, 1),
    }


def _scale(value: float, lo: float, hi: float) -> float:
    """[lo, hi] → [0, 1] に線形スケール（クランプ付き）。"""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))
