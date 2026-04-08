"""src/rag/url_list_fetcher.py — キュレーテッド URL リストの一括取得

data/doc_urls/*.txt に記載された URL を順次フェッチし、RawResult を返す。
ドメインごとのレート制限を適用して robots.txt を尊重する。

使い方:
    fetcher = UrlListFetcher()
    results = await fetcher.fetch_from_file(Path("data/doc_urls/archwiki.txt"))
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from urllib.parse import urlparse

from src.rag.retriever import RawResult

logger = logging.getLogger(__name__)

# ドメイン別レート制限（秒）
_DOMAIN_RATE_LIMITS: dict[str, float] = {
    "wiki.archlinux.org": 5.0,       # robots.txt Crawl-delay: 5
    "docs.python.org": 1.5,
    "linuxcommand.org": 3.0,
    "developer.mozilla.org": 1.5,
    "tldp.org": 2.0,
    "linuxfromscratch.org": 2.0,
    "default": 2.0,
}

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_MAX_CONTENT = 8000  # chars per page


class UrlListFetcher:
    """URL リストファイルからコンテンツを順次取得する。

    Args:
        timeout: HTTP タイムアウト秒数。
        max_content: 1ページあたりの最大文字数。
        user_agent: User-Agent ヘッダー。
    """

    def __init__(
        self,
        timeout: float = 30.0,
        max_content: int = _MAX_CONTENT,
        user_agent: str = "MED-DocBot/1.0 (Research; contact: local)",
    ) -> None:
        self._timeout = timeout
        self._max_content = max_content
        self._ua = user_agent
        self._last_by_domain: dict[str, float] = {}

    async def fetch_from_file(
        self,
        path: Path,
        limit: int | None = None,
    ) -> list[RawResult]:
        """テキストファイルに列挙された URL を順次取得する。

        ファイル形式:
            # コメント行は無視
            https://example.com/page1
            https://example.com/page2

        Args:
            path: URL リストファイルのパス。
            limit: 取得する最大 URL 数。

        Returns:
            RawResult のリスト（source="web_docs"）。
        """
        if not path.exists():
            logger.warning("URL list file not found: %s", path)
            return []

        urls = self._load_urls(path)
        if limit:
            urls = urls[:limit]

        logger.info("URL list %s: %d URLs to fetch", path.name, len(urls))
        return await self.fetch_url_list(urls)

    async def fetch_all_files(
        self,
        doc_urls_dir: Path | None = None,
        limit_per_file: int | None = None,
    ) -> list[RawResult]:
        """data/doc_urls/ 以下の全 .txt ファイルを処理する。"""
        dir_path = doc_urls_dir or (
            Path(__file__).parent.parent.parent / "data" / "doc_urls"
        )
        all_results: list[RawResult] = []
        for txt_file in sorted(dir_path.glob("*.txt")):
            results = await self.fetch_from_file(txt_file, limit=limit_per_file)
            all_results.extend(results)
            logger.info("File %s: %d results", txt_file.name, len(results))
        return all_results

    async def fetch_url_list(self, urls: list[str]) -> list[RawResult]:
        """URL リストを順次（レート制限付き）フェッチする。"""
        results: list[RawResult] = []
        for i, url in enumerate(urls):
            await self._domain_rate_limit(url)
            result = await self._fetch_one(url)
            if result is not None:
                results.append(result)
            if (i + 1) % 10 == 0:
                logger.info("  %d/%d URLs fetched (%d succeeded)", i + 1, len(urls), len(results))

        logger.info("URL list: %d/%d succeeded", len(results), len(urls))
        return results

    async def _fetch_one(self, url: str) -> RawResult | None:
        """単一 URL からコンテンツを取得する。"""
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx not installed")

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
                headers={"User-Agent": self._ua},
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
        except Exception as e:
            logger.warning("Fetch failed [%s]: %s", url[:60], e)
            return None

        content_type = resp.headers.get("content-type", "")

        if "text/html" in content_type:
            text = self._extract_html(resp.text)
            title = self._extract_title(resp.text) or url
        elif "text/plain" in content_type or "text/markdown" in content_type:
            text = resp.text
            title = url.split("/")[-1]
        else:
            logger.debug("Skipping non-text: %s (%s)", url[:60], content_type)
            return None

        text = text[:self._max_content]
        if len(text.strip()) < 80:
            logger.debug("Near-empty page skipped: %s", url[:60])
            return None

        logger.debug("Fetched: %s (%d chars)", url[:70], len(text))
        return RawResult(
            title=title[:200],
            content=text,
            url=url,
            source="web_docs",
            score=0.85,
            metadata={
                "domain": self._get_domain(url),
                "content_type": "documentation",
                "direct_fetch": True,
            },
        )

    # ── ユーティリティ ────────────────────────────────────────────────────

    @staticmethod
    def _load_urls(path: Path) -> list[str]:
        """ファイルから URL リストを読み込む（コメント・空行スキップ）。"""
        urls = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
        return urls

    @staticmethod
    def _get_domain(url: str) -> str:
        try:
            return urlparse(url).netloc
        except Exception:
            return "unknown"

    def _get_rate_sec(self, url: str) -> float:
        domain = self._get_domain(url)
        return _DOMAIN_RATE_LIMITS.get(domain, _DOMAIN_RATE_LIMITS["default"])

    async def _domain_rate_limit(self, url: str) -> None:
        domain = self._get_domain(url)
        rate_sec = _DOMAIN_RATE_LIMITS.get(domain, _DOMAIN_RATE_LIMITS["default"])
        last = self._last_by_domain.get(domain, 0.0)
        elapsed = time.monotonic() - last
        if elapsed < rate_sec:
            await asyncio.sleep(rate_sec - elapsed)
        self._last_by_domain[domain] = time.monotonic()

    @staticmethod
    def _extract_html(html: str) -> str:
        """HTML から本文テキストを抽出する。

        1. <main>/<article>/id="content" 等のコンテンツ要素を優先抽出
        2. 見つからない場合は全体から nav/header/footer を除去して抽出
        """
        # まずコンテンツ領域を特定する（正規表現で近似）
        content_html = UrlListFetcher._find_main_content(html) or html

        # script, style, nav, header, footer, aside を除去
        for tag in ("script", "style", "nav", "header", "footer", "aside", "menu"):
            content_html = re.sub(
                rf"<{tag}[^>]*>.*?</{tag}>", "", content_html,
                flags=re.DOTALL | re.IGNORECASE,
            )
        # HTML タグを除去
        text = _HTML_TAG_RE.sub(" ", content_html)
        # HTML エンティティ
        text = (text
                .replace("&nbsp;", " ")
                .replace("&amp;", "&")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&quot;", '"')
                .replace("&#39;", "'"))
        # 連続空白を正規化
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def _find_main_content(html: str) -> str | None:
        """<main>、<article>、id/class="content" 等の主要コンテンツ領域を抽出する。"""
        # 優先順位順に試す
        patterns = [
            # HTML5 semantic
            r"<main[^>]*>(.*?)</main>",
            r"<article[^>]*>(.*?)</article>",
            # MediaWiki (Arch Wiki, Wikipedia)
            r'<div[^>]+class=["\'][^"\']*mw-parser-output[^"\']*["\'][^>]*>(.*?)</div\s*>(?=\s*</div)',
            r'<div[^>]+id=["\']mw-content-text["\'][^>]*>(.*?)</div>',
            # Python docs, MDN, generic
            r'<div[^>]+id=["\'](?:main[_-]?content|content|main|post|entry|bodycontent)["\'][^>]*>(.*?)</div>',
            r'<div[^>]+class=["\'][^"\']*(?:main[_-]?content|page[_-]?content|body[_-]?content|document|rst-content)[^"\']*["\'][^>]*>(.*?)</div>',
        ]
        for pattern in patterns:
            m = re.search(pattern, html, flags=re.DOTALL | re.IGNORECASE)
            if m:
                inner = m.group(1)
                # 最低限のコンテンツがあるか確認（500文字以上）
                text_approx = _HTML_TAG_RE.sub("", inner)
                if len(text_approx.strip()) >= 500:
                    return inner
        return None

    @staticmethod
    def _extract_title(html: str) -> str:
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.DOTALL | re.IGNORECASE)
        if m:
            return _HTML_TAG_RE.sub("", m.group(1)).strip()[:200]
        return ""
