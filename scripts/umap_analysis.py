#!/usr/bin/env python3
"""scripts/umap_analysis.py — FAISSメモリの UMAP 可視化 / 島分析

FAISS の埋め込みベクトルを UMAP で2次元に圧縮し、
review_status / source_type / domain_flag / quality_score で色分けして出力する。
--save-cache で DBSCAN 島分析用キャッシュ (data/umap_cache.npz) を保存し、
island_report() で check_progress.sh から呼び出せる。

Usage:
    # review_status で色分け（デフォルト）
    poetry run python scripts/umap_analysis.py

    # 4パネル一括
    poetry run python scripts/umap_analysis.py --color-by all

    # キャッシュ保存（check_progress 用）
    poetry run python scripts/umap_analysis.py --save-cache

    # 可視化 + キャッシュ保存を同時に
    poetry run python scripts/umap_analysis.py --color-by all --save-cache

    # インタラクティブ HTML（要 plotly: poetry add plotly）
    poetry run python scripts/umap_analysis.py --interactive

    # キャッシュから島レポートのみ表示
    poetry run python scripts/umap_analysis.py --report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---- カラーパレット ------------------------------------------------

_STATUS_COLORS = {
    "approved":     "#2ecc71",  # green
    "hold":         "#e67e22",  # orange
    "needs_update": "#f1c40f",  # yellow
    "unreviewed":   "#95a5a6",  # gray
    "orphaned":     "#c0392b",  # red (FAISSにありDBにない)
}

_SOURCE_COLORS = {
    "arxiv":        "#3498db",
    "github_docs":  "#2ecc71",
    "web_docs":     "#9b59b6",
    "tavily":       "#e67e22",
    "stackoverflow":"#e74c3c",
    "manual":       "#1abc9c",
    "teacher":      "#f39c12",
    "other":        "#95a5a6",
}

_DOMAIN_COLORS = {
    "on_domain":           "#3498db",
    "off_domain":          "#e74c3c",
    "practical_reference": "#2ecc71",
    "unknown":             "#95a5a6",
}

# ---- データ読み込み -----------------------------------------------

async def _load_metadata(db_path: str) -> dict[str, dict]:
    """SQLite から doc_id → メタデータ dict を返す。"""
    import aiosqlite

    meta: dict[str, dict] = {}
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT id, source_type, review_status, teacher_quality, "
            "       composite_score, difficulty, source_title, content, source_extra "
            "FROM documents"
        )
        async for row in cur:
            extra = {}
            if row["source_extra"]:
                try:
                    extra = json.loads(row["source_extra"])
                except Exception:
                    pass
            meta[row["id"]] = {
                "source_type":    row["source_type"] or "other",
                "review_status":  row["review_status"] or "unreviewed",
                "quality":        row["teacher_quality"] or 0.0,
                "composite":      row["composite_score"] or 0.0,
                "difficulty":     row["difficulty"] or "unknown",
                "title":          (row["source_title"] or "")[:80],
                "snippet":        (row["content"] or "")[:120],
                "domain_flag":    extra.get("domain_flag", "unknown"),
                "content_type":   extra.get("content_type", "unknown"),
            }
    return meta


def _load_faiss_vectors(
    n_samples: int,
    seed: int = 42,
) -> tuple[np.ndarray, list[str], int]:
    """FAISS から埋め込みベクトルと doc_ids を返す。

    Returns:
        (vectors: float32 (N, dim), doc_ids: list[str], faiss_ntotal: int)
    """
    from src.common.config import get_settings
    from src.memory.faiss_index import FAISSIndexManager

    settings = get_settings()
    fim = FAISSIndexManager(settings.faiss)
    fim.load()

    all_vecs: list[np.ndarray] = []
    all_ids: list[str] = []

    for domain_idx in fim._indices.values():
        if domain_idx.count == 0:
            continue
        # _idx_to_id のキーは削除により不連続な場合があるため、存在するキーのみ使う
        sorted_items = sorted(domain_idx._idx_to_id.items())
        internal_indices = [item[0] for item in sorted_items]
        ids = [item[1] for item in sorted_items]
        vecs = np.vstack([
            domain_idx._index.reconstruct(i) for i in internal_indices
        ]).astype(np.float32)
        all_vecs.append(vecs)
        all_ids.extend(ids)

    if not all_vecs:
        raise RuntimeError("FAISS インデックスが空です")

    vectors = np.vstack(all_vecs)
    faiss_ntotal = len(all_ids)
    logger.info("FAISS: %d vectors loaded (dim=%d)", faiss_ntotal, vectors.shape[1])

    if faiss_ntotal > n_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(faiss_ntotal, size=n_samples, replace=False)
        vectors = vectors[idx]
        all_ids = [all_ids[i] for i in idx]
        logger.info("Subsampled to %d vectors", n_samples)

    return vectors, all_ids, faiss_ntotal


# ---- キャッシュ保存 / 島レポート ----------------------------------

_DEFAULT_CACHE = Path("data/umap_cache.npz")


def _save_cache(
    embedding: np.ndarray,
    doc_ids: list[str],
    meta: dict[str, dict],
    faiss_ntotal: int,
    cache_path: Path = _DEFAULT_CACHE,
) -> None:
    """UMAP 結果と関連メタデータをキャッシュファイルに保存する。"""
    import time

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    review_status = np.array([meta.get(d, {}).get("review_status", "orphaned") for d in doc_ids])
    source_type   = np.array([meta.get(d, {}).get("source_type",   "orphaned") for d in doc_ids])
    domain_flag   = np.array([meta.get(d, {}).get("domain_flag",   "unknown")  for d in doc_ids])
    quality       = np.array([meta.get(d, {}).get("quality",       0.0)        for d in doc_ids], dtype=np.float32)
    composite     = np.array([meta.get(d, {}).get("composite",     0.0)        for d in doc_ids], dtype=np.float32)

    np.savez(
        cache_path,
        embedding=embedding.astype(np.float32),
        doc_ids=np.array(doc_ids, dtype=object),
        review_status=review_status,
        source_type=source_type,
        domain_flag=domain_flag,
        quality=quality,
        composite=composite,
        timestamp=np.array([time.time()]),
        faiss_ntotal=np.array([faiss_ntotal]),  # _idx_to_id で有効なベクトル数
        db_total=np.array([len(meta)]),          # DB ドキュメント総数（staleness 比較用）
    )
    logger.info("Cache saved: %s (%d docs)", cache_path, len(doc_ids))


def island_report(
    cache_path: str | Path = _DEFAULT_CACHE,
    top_n: int = 5,
) -> None:
    """UMAP キャッシュから DBSCAN で島を検出し、統計を標準出力に表示する。

    check_progress.sh から直接 import して呼び出す想定。
    キャッシュが存在しない場合は生成コマンドを示して終了する。
    """
    import glob
    import time

    cache_path = Path(cache_path)
    if not cache_path.exists():
        print("\n[UMAP] キャッシュなし")
        print("  → 生成: poetry run python scripts/umap_analysis.py --save-cache")
        return

    cache = np.load(cache_path, allow_pickle=True)
    age_h   = (time.time() - float(cache["timestamp"][0])) / 3600
    embedding     = cache["embedding"]              # (N, 2)
    doc_ids       = cache["doc_ids"].tolist()
    review_status = cache["review_status"].tolist()
    source_type   = cache["source_type"].tolist()
    quality       = cache["quality"].tolist()
    cached_db_total = int(cache["db_total"][0]) if "db_total" in cache else 0
    n = len(doc_ids)

    # DB 件数で staleness を判定（軽量 sqlite3 クエリ）
    stale = False
    try:
        import sqlite3
        current_db_total = sqlite3.connect("data/metadata.db").execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()[0]
        stale = cached_db_total > 0 and current_db_total > cached_db_total * 1.10
    except Exception:
        pass

    # DBSCAN — 正規化座標で eps を自動決定
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors

    emb_min  = embedding.min(axis=0)
    emb_max  = embedding.max(axis=0)
    emb_norm = (embedding - emb_min) / (emb_max - emb_min + 1e-8)

    # eps auto-tune: サンプル数が少ない場合は percentile を緩める
    k = 10
    pct = max(10, min(30, 10 + (8000 - n) // 400))  # 1000点→30%, 8000点→10%
    nn   = NearestNeighbors(n_neighbors=k, n_jobs=1).fit(emb_norm)
    dists, _ = nn.kneighbors(emb_norm)
    eps  = max(0.02, float(np.percentile(dists[:, -1], pct)))

    labels     = DBSCAN(eps=eps, min_samples=10, n_jobs=1).fit_predict(emb_norm)
    n_clusters = int(labels.max()) + 1
    noise      = int((labels == -1).sum())

    sigma_x = float(embedding[:, 0].std())
    sigma_y = float(embedding[:, 1].std())

    # クラスタ統計
    clusters: list[dict] = []
    for cid in range(n_clusters):
        mask   = [i for i in range(n) if labels[i] == cid]
        size   = len(mask)
        q_avg  = sum(quality[i] for i in mask) / size

        src_cnt: dict[str, int] = {}
        st_cnt:  dict[str, int] = {}
        for i in mask:
            src_cnt[source_type[i]]   = src_cnt.get(source_type[i], 0) + 1
            st_cnt[review_status[i]]  = st_cnt.get(review_status[i], 0) + 1

        dom_src     = max(src_cnt, key=src_cnt.__getitem__)
        dom_src_pct = src_cnt[dom_src] * 100 // size
        approved    = st_cnt.get("approved", 0) * 100 // size

        clusters.append({
            "id": cid + 1, "size": size,
            "source": dom_src, "src_pct": dom_src_pct,
            "approved_pct": approved, "q_avg": q_avg,
        })

    clusters.sort(key=lambda c: c["size"], reverse=True)
    max_size = clusters[0]["size"] if clusters else 1

    # 出力
    stale_mark = "  ⚠ STALE (FAISS +10% growth)" if stale else ""
    print(f"\n[UMAP] Memory Map  (cache: {age_h:.1f}h ago, {n:,} docs){stale_mark}")
    print(f"  Islands: {n_clusters} clusters  noise: {noise} ({noise * 100 // n}%)  eps={eps:.3f}")
    print(f"  Dispersion: σx={sigma_x:.2f}  σy={sigma_y:.2f}")

    print(f"\n  Top clusters (by size):")
    for c in clusters[:top_n]:
        bar = "█" * (c["size"] * 20 // max_size)
        print(
            f"    #{c['id']:02d}  {c['size']:>5} docs  "
            f"{c['source']:<14} {c['src_pct']:>2}%  "
            f"approved {c['approved_pct']:>2}%  "
            f"q={c['q_avg']:.2f}  {bar}"
        )
    if n_clusters > top_n:
        rest = sum(c["size"] for c in clusters[top_n:])
        print(f"    ...  {n_clusters - top_n} more clusters ({rest:,} docs)")

    print(f"\n  → Refresh: poetry run python scripts/umap_analysis.py --save-cache")


# ---- UMAP 実行 ---------------------------------------------------

def _run_umap(
    vectors: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
) -> np.ndarray:
    """UMAP で (N, 2) 配列を返す。"""
    import umap

    logger.info(
        "Running UMAP (n=%d, n_neighbors=%d, min_dist=%.2f, metric=%s)...",
        len(vectors), n_neighbors, min_dist, metric,
    )
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
        low_memory=False,
        verbose=True,
    )
    embedding = reducer.fit_transform(vectors)
    logger.info("UMAP done: shape=%s", embedding.shape)
    return embedding


# ---- matplotlib 描画 -------------------------------------------

def _scatter_matplotlib(
    ax,
    embedding: np.ndarray,
    labels: list[str],
    color_map: dict[str, str],
    title: str,
    alpha: float = 0.5,
    s: float = 4.0,
) -> None:
    """単一パネルの散布図を ax に描画する。"""
    import matplotlib.patches as mpatches

    unique_labels = sorted(set(labels))
    for label in unique_labels:
        mask = np.array([l == label for l in labels])
        color = color_map.get(label, "#aaaaaa")
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=color, s=s, alpha=alpha, label=label, linewidths=0,
        )

    patches = [
        mpatches.Patch(color=color_map.get(lb, "#aaaaaa"), label=f"{lb} ({labels.count(lb)})")
        for lb in unique_labels
    ]
    ax.legend(handles=patches, loc="best", fontsize=7, markerscale=2, framealpha=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def _quality_scatter_matplotlib(
    ax,
    embedding: np.ndarray,
    qualities: list[float],
    title: str,
    alpha: float = 0.5,
    s: float = 4.0,
) -> None:
    """quality_score の連続値で色付けする。"""
    import matplotlib.pyplot as plt

    q = np.array(qualities)
    sc = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=q, cmap="RdYlGn", vmin=0.0, vmax=1.0,
        s=s, alpha=alpha, linewidths=0,
    )
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def _save_matplotlib(
    embedding: np.ndarray,
    doc_ids: list[str],
    meta: dict[str, dict],
    color_by: str,
    output: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def get_label(doc_id: str, key: str) -> str:
        if doc_id not in meta:
            return "orphaned"
        return meta[doc_id].get(key, "unknown") or "unknown"

    n = len(doc_ids)
    alpha = max(0.15, min(0.6, 2000.0 / n))
    s = max(2.0, min(8.0, 4000.0 / n))

    if color_by == "all":
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(
            f"UMAP Analysis — {n:,} docs (FAISS 384-dim → 2D)",
            fontsize=13, fontweight="bold",
        )

        status_labels = [get_label(d, "review_status") for d in doc_ids]
        _scatter_matplotlib(axes[0, 0], embedding, status_labels, _STATUS_COLORS, "Review Status", alpha=alpha, s=s)

        source_labels = [get_label(d, "source_type") for d in doc_ids]
        _scatter_matplotlib(axes[0, 1], embedding, source_labels, _SOURCE_COLORS, "Source Type", alpha=alpha, s=s)

        domain_labels = [get_label(d, "domain_flag") for d in doc_ids]
        _scatter_matplotlib(axes[1, 0], embedding, domain_labels, _DOMAIN_COLORS, "Domain Flag", alpha=alpha, s=s)

        qualities = [meta[d]["quality"] if d in meta else 0.0 for d in doc_ids]
        _quality_scatter_matplotlib(axes[1, 1], embedding, qualities, "Quality Score", alpha=alpha, s=s)

    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        fig.suptitle(
            f"UMAP Analysis — {n:,} docs | color: {color_by}",
            fontsize=12, fontweight="bold",
        )
        if color_by == "quality_score":
            qualities = [meta[d]["quality"] if d in meta else 0.0 for d in doc_ids]
            _quality_scatter_matplotlib(ax, embedding, qualities, color_by, alpha=alpha, s=s)
        else:
            key_map = {
                "review_status": ("review_status", _STATUS_COLORS),
                "source_type":   ("source_type",   _SOURCE_COLORS),
                "domain_flag":   ("domain_flag",   _DOMAIN_COLORS),
            }
            key, cmap = key_map.get(color_by, ("review_status", _STATUS_COLORS))
            labels = [get_label(d, key) for d in doc_ids]
            _scatter_matplotlib(ax, embedding, labels, cmap, color_by, alpha=alpha, s=s)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", output)


# ---- plotly インタラクティブ出力 ---------------------------------

def _save_plotly(
    embedding: np.ndarray,
    doc_ids: list[str],
    meta: dict[str, dict],
    color_by: str,
    output: Path,
) -> None:
    try:
        import plotly.express as px
        import pandas as pd
    except ImportError:
        logger.error("plotly not installed. Run: poetry add plotly")
        sys.exit(1)

    rows = []
    for i, doc_id in enumerate(doc_ids):
        m = meta.get(doc_id, {})
        rows.append({
            "x": float(embedding[i, 0]),
            "y": float(embedding[i, 1]),
            "doc_id": doc_id[:12],
            "review_status":  m.get("review_status", "orphaned"),
            "source_type":    m.get("source_type", "orphaned"),
            "domain_flag":    m.get("domain_flag", "unknown"),
            "quality":        round(m.get("quality", 0.0), 2),
            "composite":      round(m.get("composite", 0.0), 2),
            "difficulty":     m.get("difficulty", "unknown"),
            "title":          m.get("title", ""),
            "snippet":        m.get("snippet", ""),
        })

    df = pd.DataFrame(rows)

    color_col = {
        "review_status": "review_status",
        "source_type":   "source_type",
        "domain_flag":   "domain_flag",
        "quality_score": "quality",
    }.get(color_by, "review_status")

    color_discrete = None
    if color_by == "review_status":
        color_discrete = _STATUS_COLORS
    elif color_by == "source_type":
        color_discrete = _SOURCE_COLORS
    elif color_by == "domain_flag":
        color_discrete = _DOMAIN_COLORS

    fig = px.scatter(
        df, x="x", y="y",
        color=color_col,
        color_discrete_map=color_discrete,
        hover_data={
            "x": False, "y": False,
            "doc_id": True, "review_status": True,
            "source_type": True, "domain_flag": True,
            "quality": True, "difficulty": True,
            "title": True, "snippet": True,
        },
        title=f"UMAP Analysis — {len(doc_ids):,} docs | color: {color_by}",
        template="plotly_dark",
        opacity=0.6,
    )
    fig.update_traces(marker_size=4)
    fig.update_layout(width=1200, height=900)
    fig.write_html(str(output))
    logger.info("Saved interactive HTML: %s", output)


# ---- main --------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="UMAP で FAISS メモリを可視化する")
    parser.add_argument(
        "--color-by",
        default="all",
        choices=["all", "review_status", "source_type", "domain_flag", "quality_score"],
        help="色分け軸 (default: all — 4パネル)",
    )
    parser.add_argument("--n-samples", type=int, default=8000, help="サンプル数 (default: 8000)")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors (default: 15)")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist (default: 0.1)")
    parser.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"], help="距離指標")
    parser.add_argument("--output", type=str, default=None, help="出力ファイルパス (省略時は data/umap_<color>.png)")
    parser.add_argument("--interactive", action="store_true", help="plotly でインタラクティブ HTML を出力 (要 plotly)")
    parser.add_argument("--save-cache", action="store_true", help="DBSCAN 島分析用キャッシュを data/umap_cache.npz に保存")
    parser.add_argument("--report", action="store_true", help="キャッシュから島レポートを表示して終了 (UMAP 計算なし)")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    args = parser.parse_args()

    # --report: UMAP 計算なしでキャッシュから表示
    if args.report:
        island_report()
        return

    # 出力先が必要か（--save-cache のみの場合は画像出力をスキップ）
    need_image = args.interactive or args.output or not args.save_cache

    output: Path | None = None
    if need_image:
        if args.output:
            output = Path(args.output)
        else:
            suffix = ".html" if args.interactive else ".png"
            output = _ROOT / "data" / f"umap_{args.color_by}{suffix}"
        output.parent.mkdir(parents=True, exist_ok=True)

    # メタデータ読み込み
    from src.common.config import get_settings
    settings = get_settings()
    logger.info("Loading metadata from DB...")
    meta = asyncio.run(_load_metadata(str(settings.metadata.db_path)))
    logger.info("Loaded %d document records from DB", len(meta))

    # FAISS ベクトル読み込み
    vectors, doc_ids, faiss_ntotal = _load_faiss_vectors(n_samples=args.n_samples, seed=args.seed)

    # UMAP
    embedding = _run_umap(
        vectors,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
    )

    orphaned = sum(1 for d in doc_ids if d not in meta)
    if orphaned:
        logger.info("Orphaned (FAISS only, not in DB): %d", orphaned)

    # キャッシュ保存
    if args.save_cache:
        _save_cache(embedding, doc_ids, meta, faiss_ntotal)
        island_report()

    # 画像出力
    if output is not None:
        if args.interactive:
            _save_plotly(embedding, doc_ids, meta, args.color_by if args.color_by != "all" else "review_status", output)
        else:
            _save_matplotlib(embedding, doc_ids, meta, args.color_by, output)
        print(f"\nOutput: {output}")


if __name__ == "__main__":
    main()
