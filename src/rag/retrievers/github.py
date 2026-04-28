"""src/rag/retrievers/github.py — GitHub Code Search レトリーバー

コード検索でファイルを特定し、Contents API で実際のファイル内容を取得する。
source="github_docs" を設定することで chunker が markdown chunker を使用する。
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from pathlib import Path

from src.rag.github_docs_fetcher import GitHubDocsFetcher
from src.rag.retriever import BaseRetriever, RawResult

logger = logging.getLogger(__name__)

_GITHUB_API = "https://api.github.com"
_RATE_SEC = 1.0


class GitHubRetriever(BaseRetriever):
    """GitHub Code Search + Contents API を使った検索。

    コード検索でファイルを特定後、Contents API で実際のファイル内容を取得する。
    source="github_docs" を返すことで chunker.py が markdown chunker を使用する。
    GITHUB_TOKEN 環境変数が必要。
    """

    @property
    def source_name(self) -> str:
        return "github"

    def __init__(self) -> None:
        self._token = os.environ.get("GITHUB_TOKEN", "")
        self._last_request: float = 0.0

    def is_available(self) -> bool:
        return bool(self._token)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def _rate_limit(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request
        if elapsed < _RATE_SEC:
            await asyncio.sleep(_RATE_SEC - elapsed)
        self._last_request = time.monotonic()

    async def _fetch_file_content(self, repo: str, path: str) -> str | None:
        """Contents API でファイル内容を取得・base64 デコードする。"""
        import httpx

        await self._rate_limit()
        url = f"{_GITHUB_API}/repos/{repo}/contents/{path}"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, headers=self._headers())
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.warning("Failed to fetch content %s/%s: %s", repo, path, e)
            return None

        if data.get("encoding") != "base64":
            logger.debug("Unsupported encoding for %s/%s", repo, path)
            return None

        raw = data.get("content", "")
        try:
            return base64.b64decode(raw).decode("utf-8", errors="replace")
        except Exception:
            return None

    async def _do_search(self, query: str, max_results: int = 5) -> list[RawResult]:
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx not installed. Run: pip install httpx")

        params = {
            "q": query,
            "per_page": max_results,
            "sort": "best match",
        }

        await self._rate_limit()
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{_GITHUB_API}/search/code",
                headers=self._headers(),
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()

        results: list[RawResult] = []
        for item in data.get("items", [])[:max_results]:
            repo_info = item.get("repository", {})
            repo_full = repo_info.get("full_name", "")
            file_path = item.get("path", "")
            html_url = item.get("html_url", "")

            if not repo_full or not file_path:
                continue

            content_raw = await self._fetch_file_content(repo_full, file_path)
            if content_raw is None:
                continue

            ext = Path(file_path).suffix.lower()
            content = GitHubDocsFetcher._clean_content(content_raw, ext)

            filename = Path(file_path).stem
            title = f"{repo_full}: {filename}"

            results.append(RawResult(
                title=title,
                content=content,
                url=html_url,
                source="github_docs",
                score=float(item.get("score", 0)),
                metadata={
                    "content_type": "code_file",
                    "repo": repo_full,
                    "file_path": file_path,
                    "language": repo_info.get("language", ""),
                    "stars": repo_info.get("stargazers_count", 0),
                    "format": ext.lstrip(".") if ext else "text",
                },
            ))

        return results
