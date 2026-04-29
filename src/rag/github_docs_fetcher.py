"""src/rag/github_docs_fetcher.py — GitHub ドキュメントリポジトリ一括取得

対象: tldr-pages/tldr, nodejs/node/doc, python/cpython/Doc, mdn/content 等
設定: data/doc_urls/github_doc_repos.yaml

使い方:
    fetcher = GitHubDocsFetcher()
    results = await fetcher.fetch_repo({
        "repo": "tldr-pages/tldr",
        "path": "pages/linux",
        "ref": "main",
        "extensions": [".md"],
        "max_files": 100,
    })
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
from pathlib import Path
from typing import Any

from src.rag.retriever import RawResult

logger = logging.getLogger(__name__)

_GITHUB_API = "https://api.github.com"
_RATE_SEC = 1.0  # GitHub Contents API: 5,000/hr = 1.4/sec、余裕を持って 1/sec

# 対応拡張子と形式
_EXT_FORMAT = {
    ".md": "markdown",
    ".rst": "rst",
    ".txt": "text",
}


class GitHubDocsFetcher:
    """GitHub リポジトリからドキュメントファイルを一括取得する。

    Args:
        token: GitHub Personal Access Token。未設定時は環境変数 GITHUB_TOKEN を使用。
        rate_sec: リクエスト間隔（秒）。デフォルト 1.0。
    """

    def __init__(
        self,
        token: str | None = None,
        rate_sec: float = _RATE_SEC,
    ) -> None:
        self._token = token or os.environ.get("GITHUB_TOKEN", "")
        self._rate_sec = rate_sec
        self._last_request: float = 0.0

    def is_available(self) -> bool:
        return bool(self._token)

    @classmethod
    def load_config(cls, config_path: Path | None = None) -> list[dict]:
        """github_doc_repos.yaml を読み込む。"""
        import yaml

        path = config_path or (
            Path(__file__).parent.parent.parent / "data" / "doc_urls" / "github_doc_repos.yaml"
        )
        if not path.exists():
            logger.warning("github_doc_repos.yaml not found: %s", path)
            return []
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("repos", [])

    async def fetch_all(
        self,
        config_path: Path | None = None,
        max_files_per_repo: int | None = None,
    ) -> list[RawResult]:
        """設定ファイルに定義された全リポジトリからドキュメントを取得する。"""
        configs = self.load_config(config_path)
        if not configs:
            return []

        all_results: list[RawResult] = []
        for repo_cfg in configs:
            if not repo_cfg.get("enabled", True):
                continue
            limit = max_files_per_repo or repo_cfg.get("max_files", 100)
            results = await self.fetch_repo(repo_cfg, max_files=limit)
            all_results.extend(results)
            logger.info(
                "Repo %s: %d files fetched",
                repo_cfg.get("repo", "?"), len(results),
            )

        return all_results

    async def fetch_repo(
        self,
        repo_cfg: dict[str, Any],
        max_files: int = 100,
    ) -> list[RawResult]:
        """リポジトリ設定からドキュメントを取得する。

        Args:
            repo_cfg: github_doc_repos.yaml の repos[] 要素。
            max_files: 取得する最大ファイル数。

        Returns:
            RawResult のリスト（source="github_docs"）。
        """
        repo = repo_cfg["repo"]
        paths = repo_cfg.get("paths", [repo_cfg.get("path", "")])
        if isinstance(paths, str):
            paths = [paths]
        ref = repo_cfg.get("ref", "HEAD")
        extensions = repo_cfg.get("extensions", [".md", ".rst"])
        label = repo_cfg.get("label", repo)

        logger.info("Fetching %s (ref=%s, paths=%s, max=%d)", repo, ref, paths, max_files)

        # ファイル一覧を収集
        file_paths: list[str] = []
        for path_prefix in paths:
            if len(file_paths) >= max_files:
                break
            remaining = max_files - len(file_paths)
            found = await self._list_files(repo, path_prefix, ref, extensions, remaining)
            file_paths.extend(found)

        logger.info("%s: %d files to fetch", repo, len(file_paths))

        cleaner_profile = repo_cfg.get("cleaner_profile", "default")

        # ファイル内容を順次取得（レート制限付き）
        results: list[RawResult] = []
        for i, file_path in enumerate(file_paths):
            content = await self._fetch_file(repo, file_path, ref)
            if content is None:
                continue

            ext = Path(file_path).suffix.lower()

            # プロファイル別クリーニング + メタ抽出
            if cleaner_profile == "nodejs_api" and ext == ".md":
                doc_meta = self._extract_nodejs_meta(content)
                clean = self._clean_nodejs_markdown(content)
            else:
                doc_meta = {}
                clean = self._clean_content(content, ext)

            if len(clean) < 100:
                continue

            filename = Path(file_path).stem
            title = f"{label}: {filename}"

            results.append(RawResult(
                title=title,
                content=clean,
                url=f"https://github.com/{repo}/blob/{ref}/{file_path}",
                source="github_docs",
                score=0.9,
                metadata={
                    "repo": repo,
                    "file_path": file_path,
                    "ref": ref,
                    "format": _EXT_FORMAT.get(ext, "text"),
                    "label": label,
                    "content_type": "documentation",
                    **doc_meta,
                },
            ))

            if (i + 1) % 20 == 0:
                logger.info("  %s: %d/%d fetched", repo, i + 1, len(file_paths))

        return results

    async def _list_files(
        self,
        repo: str,
        path: str,
        ref: str,
        extensions: list[str],
        max_files: int,
    ) -> list[str]:
        """Contents API でディレクトリ内のファイル一覧を取得（非再帰）。"""
        import httpx

        headers = self._headers()
        url = f"{_GITHUB_API}/repos/{repo}/contents/{path}"
        params = {"ref": ref}

        await self._rate_limit()
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, headers=headers, params=params)
                if resp.status_code == 404:
                    logger.warning("Path not found: %s/%s", repo, path)
                    return []
                resp.raise_for_status()
                items = resp.json()
        except Exception as e:
            logger.warning("Failed to list %s/%s: %s", repo, path, e)
            return []

        if not isinstance(items, list):
            # ファイルが直接指定された場合
            if isinstance(items, dict) and items.get("type") == "file":
                ext = Path(items["name"]).suffix.lower()
                if ext in extensions:
                    return [items["path"]]
            return []

        file_paths: list[str] = []
        for item in items:
            if len(file_paths) >= max_files:
                break
            if item["type"] == "file":
                ext = Path(item["name"]).suffix.lower()
                if ext in extensions:
                    file_paths.append(item["path"])

        return file_paths

    async def _fetch_file(self, repo: str, path: str, ref: str) -> str | None:
        """ファイル内容を取得してデコードする。"""
        import httpx

        headers = self._headers()
        url = f"{_GITHUB_API}/repos/{repo}/contents/{path}"
        params = {"ref": ref}

        await self._rate_limit()
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, headers=headers, params=params)
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.warning("Failed to fetch %s/%s: %s", repo, path, e)
            return None

        if data.get("encoding") != "base64":
            logger.debug("Unsupported encoding for %s/%s", repo, path)
            return None

        raw = data.get("content", "")
        try:
            return base64.b64decode(raw).decode("utf-8", errors="replace")
        except Exception:
            return None

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def _rate_limit(self) -> None:
        import time
        now = time.monotonic()
        elapsed = now - self._last_request
        if elapsed < self._rate_sec:
            await asyncio.sleep(self._rate_sec - elapsed)
        self._last_request = time.monotonic()

    # ── コンテンツクリーナー ──────────────────────────────────────────────

    @staticmethod
    def _extract_nodejs_meta(text: str) -> dict:
        """Node.js API Markdown から属性メタデータを抽出する（クリーニング前に呼ぶ）。

        抽出対象:
        - ``<!-- YAML\\nadded: vX.X.X\\n-->`` コメントブロック → ``added_in``
        - ``> Stability: N - Label`` ブロック引用 → ``stability_level`` / ``stability_label``

        Returns:
            source_extra に格納するキー辞書。
        """
        meta: dict = {}

        # <!-- YAML ... --> ブロックから added_in を取得（クリーニング前のみ存在）
        yaml_blocks = re.findall(r'<!--\s*YAML\s*(.*?)-->', text, re.DOTALL)
        for block in yaml_blocks:
            m = re.search(r'^added:\s*(v[\d.]+(?:-\w+)?)', block, re.MULTILINE)
            if m:
                meta.setdefault("added_in", m.group(1))
                break

        # > Stability: N - Label
        m = re.search(r'^>\s*Stability:\s*(\d)\s*[-\u2013]\s*(.+)$', text, re.MULTILINE)
        if m:
            meta["stability_level"] = int(m.group(1))
            meta["stability_label"] = m.group(2).strip()

        return meta

    @staticmethod
    def _clean_nodejs_markdown(text: str) -> str:
        """Node.js API Markdown 専用クリーナー。

        標準 ``_clean_markdown()`` に加え、Node.js 固有のメタ行を除去する:
        - ``> Stability: N - ...`` ブロック引用
        """
        text = GitHubDocsFetcher._clean_markdown(text)
        # Stability ブロック引用を除去
        text = re.sub(r'^>\s*Stability:\s*\d\s*[-\u2013]\s*.+$', '', text, flags=re.MULTILINE)
        # {TypeName} 型参照を展開: {FileHandle} → FileHandle
        text = re.sub(r'\{([A-Za-z][A-Za-z0-9_]*)\}', r'\1', text)
        # 連続空行を正規化
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def _clean_content(text: str, ext: str) -> str:
        """拡張子に応じてマークアップを除去し、FAISS 向けにクリーンなテキストを返す。"""
        if ext == ".md":
            return GitHubDocsFetcher._clean_markdown(text)
        elif ext == ".rst":
            return GitHubDocsFetcher._clean_rst(text)
        else:
            return text.strip()

    @staticmethod
    def _clean_markdown(text: str) -> str:
        """Markdown のメタデータ・装飾を除去してテキストを返す。
        コードブロックは内容を保持する（プログラミングドキュメントで重要）。
        """
        # YAML frontmatter を除去
        text = re.sub(r'^---\n.*?\n---\n?', '', text, flags=re.DOTALL)
        # TOML frontmatter を除去
        text = re.sub(r'^\+\+\+\n.*?\n\+\+\+\n?', '', text, flags=re.DOTALL)
        # HTML コメントを除去
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        # バッジ画像を除去
        text = re.sub(r'!\[.*?\]\(https?://[^)]+\)', '', text)
        # HTML タグを除去（Markdown 中の混在）
        text = re.sub(r'<[^>]+>', '', text)
        # 行頭の # ヘッダーマーカーを保持（内容はそのまま）
        # インラインコードのバッククォートを除去（内容は保持）
        text = re.sub(r'`([^`\n]+)`', r'\1', text)
        # 未解決内部リンク記法を除去: [text][ref] → text, [text][] → text
        # （Node.js API docs 等で多用される記法。属性としては extract_internal_links で取得済み）
        text = re.sub(r'\[([^\]]+)\]\[[^\]]*\]', r'\1', text)
        # 連続空行を2行に正規化
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def _clean_rst(text: str) -> str:
        """RST のディレクティブ・クロスリファレンスを除去してテキストを返す。
        コードブロック内容は保持する。
        """
        # セクション下線を除去（=== --- ~~~ 等の行）
        text = re.sub(r'^[=\-~^"#*+`]{3,}\s*$', '', text, flags=re.MULTILINE)
        # インラインマークアップ: :role:`text` → text  (:ref:`label <target>` も同様)
        text = re.sub(r':[a-zA-Z_]+:`([^`]+)`', r'\1', text)
        # :role:`text <target>` の残留 <target> を除去
        text = re.sub(r'\s*<(?!https?://|/)[^>]{1,120}>', '', text)
        # チルダ参照: ~module.attr → module.attr（短縮表示記法）
        text = re.sub(r'~([A-Za-z][A-Za-z0-9_.]*)', r'\1', text)
        # ダブルバッククォート: ``code`` → code
        text = re.sub(r'``([^`]+)``', r'\1', text)
        # ディレクティブ行を除去（.. directive:: args）
        text = re.sub(r'^\.\. [a-zA-Z_-]+::.*$', '', text, flags=re.MULTILINE)
        # versionadded/versionchanged/deprecated 注記
        text = re.sub(r'^\.\. version\w+::\s*\S+', '', text, flags=re.MULTILINE)
        # 連続空行を2行に正規化
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
