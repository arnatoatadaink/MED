"""src/cycle/umap_islands.py — UMAP キャッシュからの島検出共有ロジック

scripts/umap_analysis.py::island_report と
src/cycle/gap_detector.py の両方から使用する。
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_DEFAULT_CACHE = Path("data/umap_cache.npz")
_DB_PATH       = "data/metadata.db"


# ---- データ型 -------------------------------------------------------

@dataclass
class Island:
    """1つの DBSCAN クラスタ。"""

    id: int                          # 1-based
    size: int
    centroid: tuple[float, float]    # (x, y) in 2D UMAP space
    doc_ids:      list[str]
    source_dist:  dict[str, int]     # source_type → count
    review_dist:  dict[str, int]     # review_status → count
    q_avg: float

    @property
    def dominant_source(self) -> str:
        return max(self.source_dist, key=self.source_dist.__getitem__) if self.source_dist else "unknown"

    @property
    def dominant_source_pct(self) -> float:
        if not self.source_dist:
            return 0.0
        top = self.source_dist[self.dominant_source]
        return top / self.size

    @property
    def approved_pct(self) -> float:
        return self.review_dist.get("approved", 0) / self.size

    @property
    def unreviewed_pct(self) -> float:
        return self.review_dist.get("unreviewed", 0) / self.size


@dataclass
class IslandSet:
    """compute_islands の出力。"""

    islands:      list[Island]
    noise_count:  int
    eps:          float
    sigma_x:      float
    sigma_y:      float
    n_docs:       int
    cache_age_h:  float
    stale:        bool
    embedding:    np.ndarray          # (N, 2)
    doc_ids:      list[str]
    review_status: list[str]
    source_type:  list[str]
    quality:      list[float]

    @property
    def n_clusters(self) -> int:
        return len(self.islands)


# ---- 計算ロジック ---------------------------------------------------

class CacheStaleError(RuntimeError):
    """UMAP キャッシュが存在しないか古すぎる場合。"""


def compute_islands(
    cache_path: str | Path = _DEFAULT_CACHE,
    db_path: str = _DB_PATH,
    min_samples: int = 10,
    raise_if_stale: bool = False,
) -> IslandSet:
    """UMAP キャッシュを読み込み DBSCAN で島を検出する。

    Args:
        cache_path: umap_cache.npz のパス。
        db_path: metadata.db のパス（staleness 判定用）。
        min_samples: DBSCAN の min_samples。
        raise_if_stale: True の場合、キャッシュ未存在 or stale で例外。

    Returns:
        IslandSet

    Raises:
        CacheStaleError: raise_if_stale=True かつキャッシュが無効な場合。
    """
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors

    cache_path = Path(cache_path)

    if not cache_path.exists():
        if raise_if_stale:
            raise CacheStaleError(
                f"Cache not found: {cache_path}\n"
                "Run: poetry run python scripts/umap_analysis.py --save-cache"
            )
        # fallback: return empty IslandSet
        return IslandSet(
            islands=[], noise_count=0, eps=0.0,
            sigma_x=0.0, sigma_y=0.0, n_docs=0, cache_age_h=0.0, stale=True,
            embedding=np.empty((0, 2)), doc_ids=[], review_status=[],
            source_type=[], quality=[],
        )

    cache = np.load(cache_path, allow_pickle=True)
    age_h           = (time.time() - float(cache["timestamp"][0])) / 3600
    embedding       = cache["embedding"]
    doc_ids         = cache["doc_ids"].tolist()
    review_status   = cache["review_status"].tolist()
    source_type_list = cache["source_type"].tolist()
    quality         = cache["quality"].tolist()
    cached_db_total = int(cache["db_total"][0]) if "db_total" in cache else 0
    n = len(doc_ids)

    # staleness — DB 件数が10%以上増加していたら陳腐と判定
    stale = False
    try:
        current_total = sqlite3.connect(db_path).execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()[0]
        stale = cached_db_total > 0 and current_total > cached_db_total * 1.10
    except Exception:
        pass

    if raise_if_stale and stale:
        raise CacheStaleError(
            f"Cache is stale (db grew >10% since cache was built). "
            "Run: poetry run python scripts/umap_analysis.py --save-cache"
        )

    if n == 0:
        return IslandSet(
            islands=[], noise_count=0, eps=0.0,
            sigma_x=0.0, sigma_y=0.0, n_docs=0, cache_age_h=age_h, stale=stale,
            embedding=embedding, doc_ids=doc_ids, review_status=review_status,
            source_type=source_type_list, quality=quality,
        )

    # 正規化してから DBSCAN
    emb_min  = embedding.min(axis=0)
    emb_max  = embedding.max(axis=0)
    emb_norm = (embedding - emb_min) / (emb_max - emb_min + 1e-8)

    # eps auto-tune: サンプルが少ないほど percentile を緩める
    k   = min_samples
    pct = max(10, min(30, 10 + (8000 - n) // 400))
    nn  = NearestNeighbors(n_neighbors=k, n_jobs=1).fit(emb_norm)
    dists, _ = nn.kneighbors(emb_norm)
    eps = max(0.02, float(np.percentile(dists[:, -1], pct)))

    labels    = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1).fit_predict(emb_norm)
    n_clusters = int(labels.max()) + 1
    noise_count = int((labels == -1).sum())

    sigma_x = float(embedding[:, 0].std())
    sigma_y = float(embedding[:, 1].std())

    # クラスタ情報を組み立て
    islands: list[Island] = []
    for cid in range(n_clusters):
        mask = [i for i in range(n) if labels[i] == cid]
        size = len(mask)
        if size == 0:
            continue

        q_avg = float(sum(quality[i] for i in mask) / size)
        cx    = float(sum(embedding[i, 0] for i in mask) / size)
        cy    = float(sum(embedding[i, 1] for i in mask) / size)

        src_cnt: dict[str, int] = {}
        rev_cnt: dict[str, int] = {}
        ids: list[str] = []
        for i in mask:
            s = source_type_list[i]
            r = review_status[i]
            src_cnt[s] = src_cnt.get(s, 0) + 1
            rev_cnt[r] = rev_cnt.get(r, 0) + 1
            ids.append(doc_ids[i])

        islands.append(Island(
            id=cid + 1,
            size=size,
            centroid=(cx, cy),
            doc_ids=ids,
            source_dist=src_cnt,
            review_dist=rev_cnt,
            q_avg=q_avg,
        ))

    islands.sort(key=lambda c: c.size, reverse=True)

    return IslandSet(
        islands=islands,
        noise_count=noise_count,
        eps=eps,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        n_docs=n,
        cache_age_h=age_h,
        stale=stale,
        embedding=embedding,
        doc_ids=doc_ids,
        review_status=review_status,
        source_type=source_type_list,
        quality=quality,
    )


def detect_isolated_pairs(
    iset: IslandSet,
    min_dist_percentile: int = 75,
    max_pairs: int = 5,
) -> list[tuple[Island, Island, float]]:
    """空間的に離れた島ペアを検出する。

    Args:
        iset: compute_islands() が返す IslandSet。
        min_dist_percentile: 重心間距離のパーセンタイルしきい値 (0-100)。
        max_pairs: 返す最大ペア数。

    Returns:
        (island_a, island_b, distance) のリスト（距離降順）。
    """
    islands = iset.islands
    if len(islands) < 2:
        return []

    pairs: list[tuple[Island, Island, float]] = []
    for i in range(len(islands)):
        for j in range(i + 1, len(islands)):
            ax, ay = islands[i].centroid
            bx, by = islands[j].centroid
            dist = float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2))
            pairs.append((islands[i], islands[j], dist))

    if not pairs:
        return []

    dists = [p[2] for p in pairs]
    threshold = float(np.percentile(dists, min_dist_percentile))
    far_pairs = [(a, b, d) for a, b, d in pairs if d >= threshold]
    far_pairs.sort(key=lambda x: -x[2])
    return far_pairs[:max_pairs]
