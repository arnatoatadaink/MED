"""
FAISS 埋め込み分布 UMAP 可視化スクリプト

Usage:
    poetry run python scripts/visualize_faiss.py [--sample N] [--output PATH]

確認ポイント:
  1. 断絶度    : academic / code / general が空間的に分離しているか
  2. コード内部 : source_type (arxiv/github_docs/SO/tavily) でクラスターが分かれるか
  3. 孤立点    : 最近傍との距離が大きいアウトライアー文書の特定
  4. ブリッジ候補: academic と code の中間に位置する文書
"""

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path

import faiss
import matplotlib
matplotlib.use("Agg")  # WSL2 / headless 環境用
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import umap

# ── 設定 ─────────────────────────────────────────────────────────────────────

FAISS_DIR = Path("data/faiss_indices")
DB_PATH   = Path("data/metadata.db")
OUT_DIR   = Path("data/analysis")

DOMAINS = ["academic", "code", "general"]

# domain ごとのサンプル上限（UMAPの計算コスト対策）
SAMPLE_LIMITS = {
    "academic": 5000,   # 全件（11点）
    "code":     3000,   # 25k から stratified sample
    "general":  5000,   # 全件（36点）
}

# カラーパレット
DOMAIN_COLORS  = {"academic": "#e74c3c", "code": "#3498db", "general": "#2ecc71"}
SOURCE_COLORS  = {
    "arxiv":       "#e74c3c",
    "github_docs": "#3498db",
    "github":      "#2980b9",
    "stackoverflow": "#f39c12",
    "web_docs":    "#9b59b6",
    "tavily":      "#1abc9c",
    "manual":      "#e67e22",
    "teacher":     "#e91e63",
    "other":       "#95a5a6",
}
STATUS_COLORS  = {
    "approved":     "#27ae60",
    "needs_update": "#f39c12",
    "unreviewed":   "#3498db",
    "hold":         "#e74c3c",
    "rejected":     "#7f8c8d",
}

# ── データ取得 ────────────────────────────────────────────────────────────────

def load_domain(domain: str, sample_limit: int) -> tuple[np.ndarray, list[str]]:
    """FAISSインデックスからベクトルと文書IDを取得（サンプリング付き）。

    id_map.npz の構造:
      ids[j]     : j番目エントリの文書ID (str)
      indices[j] : j番目エントリのFAISS内部インデックス (int)
    IDが確認済みのエントリのみを対象にする。
    """
    idx_path = FAISS_DIR / domain / "index.faiss"
    map_path = FAISS_DIR / domain / "id_map.npz"
    if not idx_path.exists():
        return np.empty((0, 384), dtype=np.float32), []

    index   = faiss.read_index(str(idx_path))
    id_data = np.load(str(map_path), allow_pickle=True)
    doc_ids    = list(id_data["ids"])       # 文書ID (len = mapped entries)
    faiss_idxs = id_data["indices"].tolist() # FAISS内部インデックス

    m = len(doc_ids)  # IDが確認済みのエントリ数
    if m == 0:
        return np.empty((0, 384), dtype=np.float32), []

    if m <= sample_limit:
        vecs = np.vstack([index.reconstruct(int(fi)) for fi in faiss_idxs])
        return vecs, doc_ids

    # stratified sample: j (0..m-1) からランダムに sample_limit 件選ぶ
    rng      = np.random.default_rng(42)
    j_sample = np.sort(rng.choice(m, size=sample_limit, replace=False))
    vecs     = np.vstack([index.reconstruct(int(faiss_idxs[j])) for j in j_sample])
    sampled_ids = [doc_ids[j] for j in j_sample]
    return vecs, sampled_ids


def fetch_metadata(doc_ids: list[str]) -> dict[str, dict]:
    """SQLiteから文書メタデータを取得。"""
    if not doc_ids:
        return {}
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()
    placeholders = ",".join(["?"] * len(doc_ids))
    cur.execute(
        f"SELECT id, source_type, review_status, domain, source_title "
        f"FROM documents WHERE id IN ({placeholders})",
        doc_ids,
    )
    rows = {r["id"]: dict(r) for r in cur.fetchall()}
    conn.close()
    return rows

# ── UMAP 実行 ─────────────────────────────────────────────────────────────────

def run_umap(vecs: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="cosine",
        random_state=42,
        verbose=False,
    )
    return reducer.fit_transform(vecs.astype(np.float32))

# ── 可視化 ────────────────────────────────────────────────────────────────────

def _legend_patches(color_map: dict, title: str) -> tuple[list, str]:
    patches = [mpatches.Patch(color=c, label=k) for k, c in color_map.items()]
    return patches, title


def plot_all(
    xy: np.ndarray,
    all_ids: list[str],
    meta: dict,
    domain_labels: list[str],
    out_dir: Path,
    date_str: str,
) -> None:
    """3種類の散布図（domain / source_type / review_status）を1ファイルに出力。"""
    n = len(all_ids)

    def get_val(key: str, color_map: dict, default_key: str = "other") -> tuple[list, list]:
        colors = []
        labels = []
        for i, doc_id in enumerate(all_ids):
            row = meta.get(doc_id)
            if row:
                val = row.get(key, default_key) or default_key
            else:
                # id_map に存在するが DB にないケース（FAISSとDBの同期ずれ）
                val = default_key
            labels.append(val)
            colors.append(color_map.get(val, color_map.get(default_key, "#95a5a6")))
        return colors, labels

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(
        f"FAISS Embedding Distribution — UMAP  ({date_str})  n={n:,}",
        fontsize=14, y=1.01,
    )

    # ── 図1: domain ──────────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_title("by Domain", fontsize=11)
    domain_colors_used = {d: DOMAIN_COLORS.get(d, "#95a5a6") for d in DOMAINS}
    d_colors = [DOMAIN_COLORS.get(d, "#95a5a6") for d in domain_labels]
    ax.scatter(xy[:, 0], xy[:, 1], c=d_colors, s=4, alpha=0.5, linewidths=0)
    patches, ttl = _legend_patches(domain_colors_used, "Domain")
    ax.legend(handles=patches, title=ttl, loc="upper right", markerscale=2, fontsize=8)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    # academic 点にラベル
    for i, (doc_id, dl) in enumerate(zip(all_ids, domain_labels)):
        if dl == "academic":
            row = meta.get(doc_id, {})
            title = (row.get("source_title") or doc_id)[:20]
            ax.annotate(title, (xy[i, 0], xy[i, 1]), fontsize=5, alpha=0.7)

    # ── 図2: source_type ─────────────────────────────────────────────────────
    ax = axes[1]
    ax.set_title("by Source Type", fontsize=11)
    s_colors, s_labels = get_val("source_type", SOURCE_COLORS, "other")
    ax.scatter(xy[:, 0], xy[:, 1], c=s_colors, s=4, alpha=0.5, linewidths=0)
    used_sources = {k: v for k, v in SOURCE_COLORS.items() if k in set(s_labels)}
    patches, ttl = _legend_patches(used_sources, "Source Type")
    ax.legend(handles=patches, title=ttl, loc="upper right", markerscale=2, fontsize=8)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    # ── 図3: review_status ───────────────────────────────────────────────────
    ax = axes[2]
    ax.set_title("by Review Status", fontsize=11)
    r_colors, _ = get_val("review_status", STATUS_COLORS, "unreviewed")
    ax.scatter(xy[:, 0], xy[:, 1], c=r_colors, s=4, alpha=0.5, linewidths=0)
    patches, ttl = _legend_patches(STATUS_COLORS, "Review Status")
    ax.legend(handles=patches, title=ttl, loc="upper right", markerscale=2, fontsize=8)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    plt.tight_layout()
    out_path = out_dir / f"faiss_umap_{date_str}.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")

# ── 分析レポート ──────────────────────────────────────────────────────────────

def analyze_gaps(
    xy: np.ndarray,
    all_ids: list[str],
    meta: dict,
    domain_labels: list[str],
) -> None:
    """断絶度・孤立点・ブリッジ候補をテキストレポートとして出力。"""
    from scipy.spatial.distance import cdist

    academic_mask = np.array([d == "academic" for d in domain_labels])
    code_mask     = np.array([d == "code"     for d in domain_labels])

    print("\n" + "=" * 60)
    print("FAISS 埋め込み空間 分析レポート")
    print("=" * 60)

    # 1. 断絶度: academic centroid と code centroid の距離
    if academic_mask.any() and code_mask.any():
        ac_center   = xy[academic_mask].mean(axis=0)
        code_center = xy[code_mask].mean(axis=0)
        gap = np.linalg.norm(ac_center - code_center)
        print(f"\n[1] academic ↔ code 重心間距離 (UMAP空間): {gap:.3f}")
        print(f"    ※ この値が大きいほど2つのドメインが断絶している")

    # 2. 孤立点: 全点に対して最近傍距離 Top-10
    if len(xy) > 1:
        dists = cdist(xy, xy)
        np.fill_diagonal(dists, np.inf)
        min_dists = dists.min(axis=1)
        top_isolated = np.argsort(min_dists)[::-1][:10]
        print(f"\n[2] 孤立点 Top-10（最近傍距離が大きい文書）:")
        for rank, idx in enumerate(top_isolated, 1):
            doc_id = all_ids[idx]
            row    = meta.get(doc_id, {})
            title  = (row.get("source_title") or doc_id)[:50]
            print(f"    {rank:2d}. [{domain_labels[idx]:8s}] {min_dists[idx]:.3f}  {title}")

    # 3. ブリッジ候補: academicとcodeの中間点（両者の重心に近い文書）
    if academic_mask.any() and code_mask.any():
        bridge_center = (xy[academic_mask].mean(axis=0) + xy[code_mask].mean(axis=0)) / 2
        bridge_dists  = np.linalg.norm(xy - bridge_center, axis=1)
        top_bridge    = np.argsort(bridge_dists)[:10]
        print(f"\n[3] ブリッジ候補 Top-10（academic/code重心の中間に位置する文書）:")
        for rank, idx in enumerate(top_bridge, 1):
            doc_id = all_ids[idx]
            row    = meta.get(doc_id, {})
            title  = (row.get("source_title") or doc_id)[:50]
            src    = row.get("source_type", "?")
            stat   = row.get("review_status", "?")
            print(f"    {rank:2d}. [{domain_labels[idx]:8s}/{src:12s}] {title}  ({stat})")

    # 4. source_type 別の重心分布（code ドメイン内部の構造）
    if code_mask.any():
        print(f"\n[4] code ドメイン内 source_type 別重心（UMAP座標）:")
        code_ids    = [all_ids[i] for i in np.where(code_mask)[0]]
        code_meta   = {k: meta[k] for k in code_ids if k in meta}
        src_groups: dict[str, list[int]] = {}
        for i in np.where(code_mask)[0]:
            row = meta.get(all_ids[i], {})
            src = row.get("source_type", "other") or "other"
            src_groups.setdefault(src, []).append(i)
        for src, idxs in sorted(src_groups.items(), key=lambda x: -len(x[1])):
            center = xy[idxs].mean(axis=0)
            print(f"    {src:15s} n={len(idxs):5d}  center=({center[0]:+.2f}, {center[1]:+.2f})")

    print("\n" + "=" * 60)

# ── エントリポイント ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="FAISS UMAP visualization")
    parser.add_argument("--sample", type=int, default=None,
                        help="code ドメインのサンプル数（デフォルト: 3000）")
    parser.add_argument("--output", type=str, default=None,
                        help="出力ディレクトリ（デフォルト: data/analysis/）")
    parser.add_argument("--no-plot", action="store_true",
                        help="プロット生成をスキップ（分析レポートのみ）")
    args = parser.parse_args()

    if args.sample:
        SAMPLE_LIMITS["code"] = args.sample
    out_dir = Path(args.output) if args.output else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d_%H%M")

    # 各ドメインのデータ読み込み
    all_vecs:    list[np.ndarray] = []
    all_ids:     list[str]        = []
    domain_labels: list[str]      = []

    for domain in DOMAINS:
        vecs, ids = load_domain(domain, SAMPLE_LIMITS[domain])
        if len(vecs) == 0:
            print(f"[skip] {domain}: no vectors")
            continue
        all_vecs.append(vecs)
        all_ids.extend(ids)
        domain_labels.extend([domain] * len(ids))
        print(f"[load] {domain}: {len(ids):,} vectors")

    if not all_vecs:
        print("ERROR: FAISSインデックスが空です")
        return

    vecs_concat = np.vstack(all_vecs)
    print(f"[total] {len(all_ids):,} vectors → UMAP (384 → 2)")

    # メタデータ取得
    meta = fetch_metadata(all_ids)
    print(f"[meta] {len(meta):,} / {len(all_ids):,} docs found in DB")

    # UMAP 実行
    print("[umap] fitting...")
    xy = run_umap(vecs_concat)
    print("[umap] done")

    # 分析レポート（scipy が必要）
    try:
        analyze_gaps(xy, all_ids, meta, domain_labels)
    except ImportError:
        print("[skip] scipy not installed — skipping gap analysis")

    # プロット出力
    if not args.no_plot:
        plot_all(xy, all_ids, meta, domain_labels, out_dir, date_str)

    # UMAP座標を CSV に保存（後続分析用）
    import csv
    csv_path = out_dir / f"faiss_umap_{date_str}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "umap_x", "umap_y", "domain",
                         "source_type", "review_status", "title"])
        for i, doc_id in enumerate(all_ids):
            row = meta.get(doc_id, {})
            writer.writerow([
                doc_id, f"{xy[i,0]:.4f}", f"{xy[i,1]:.4f}",
                domain_labels[i],
                row.get("source_type", ""),
                row.get("review_status", ""),
                (row.get("source_title") or "")[:80],
            ])
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
