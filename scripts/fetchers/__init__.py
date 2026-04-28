"""scripts/fetchers — サイト別記事フェッチャーパッケージ

新サイト追加手順:
  1. scripts/fetchers/{site}.py に BaseFetcher サブクラスを実装
  2. REGISTRY に登録
"""

from __future__ import annotations

from scripts.fetchers.base import BaseFetcher, FetchedArticle, extract_arxiv_ids, strip_html
from scripts.fetchers.note import NoteFetcher

REGISTRY: dict[str, type[BaseFetcher]] = {
    "note": NoteFetcher,
    # 将来追加: "qiita": QiitaFetcher, "zenn": ZennFetcher, ...
}


def get_fetcher(site: str) -> BaseFetcher:
    if site not in REGISTRY:
        available = ", ".join(REGISTRY.keys())
        raise ValueError(f"Unknown site '{site}'. Available: {available}")
    return REGISTRY[site]()


__all__ = [
    "BaseFetcher",
    "FetchedArticle",
    "extract_arxiv_ids",
    "strip_html",
    "REGISTRY",
    "get_fetcher",
]
