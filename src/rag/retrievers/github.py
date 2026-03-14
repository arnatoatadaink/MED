"""src/rag/retrievers/github.py — GitHub Code Search レトリーバー"""

from __future__ import annotations

import logging
import os

from src.rag.retriever import BaseRetriever, RawResult

logger = logging.getLogger(__name__)


class GitHubRetriever(BaseRetriever):
    """GitHub Code Search API を使った検索。

    GITHUB_TOKEN 環境変数または settings から API トークンを取得する。
    """

    @property
    def source_name(self) -> str:
        return "github"

    def __init__(self) -> None:
        self._token = os.environ.get("GITHUB_TOKEN", "")

    def is_available(self) -> bool:
        return bool(self._token)

    async def search(self, query: str, max_results: int = 5) -> list[RawResult]:
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx not installed. Run: pip install httpx")

        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        params = {
            "q": query,
            "per_page": max_results,
            "sort": "best match",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                "https://api.github.com/search/code",
                headers=headers,
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()

        results: list[RawResult] = []
        for item in data.get("items", [])[:max_results]:
            repo = item.get("repository", {})
            results.append(RawResult(
                title=item.get("name", ""),
                content=item.get("path", ""),
                url=item.get("html_url", ""),
                source=self.source_name,
                score=float(item.get("score", 0)),
                metadata={
                    "repo": repo.get("full_name", ""),
                    "language": repo.get("language", ""),
                    "stars": repo.get("stargazers_count", 0),
                },
            ))

        return results
