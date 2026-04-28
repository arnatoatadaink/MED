"""scripts/fetchers/base.py — 記事フェッチャー基底クラス・共通ユーティリティ"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from html.parser import HTMLParser


# ── HTML → プレーンテキスト ───────────────────────────────────────────────────

class _HTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return "\n".join(p.strip() for p in self._parts if p.strip())


def strip_html(html: str) -> str:
    """HTML タグを除去してプレーンテキストを返す。"""
    s = _HTMLStripper()
    s.feed(html)
    text = s.get_text()
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ── arXiv ID 抽出 ────────────────────────────────────────────────────────────

_ARXIV_PATTERNS = [
    # https://arxiv.org/abs/2401.12345  or /pdf/
    re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE),
    # arXiv:2401.12345  or  arXiv: 2401.12345
    re.compile(r"arXiv[:\s]+(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE),
]


def extract_arxiv_ids(text: str) -> list[str]:
    """テキストから arXiv ID を抽出して重複除去・ソートして返す。バージョンサフィックス除去。"""
    ids: set[str] = set()
    for pat in _ARXIV_PATTERNS:
        for m in pat.finditer(text):
            arxiv_id = re.sub(r"v\d+$", "", m.group(1))
            ids.add(arxiv_id)
    return sorted(ids)


# ── データクラス ─────────────────────────────────────────────────────────────

@dataclass
class FetchedArticle:
    key: str            # サイト固有の一意キー（例: note の 'n0020f11fa12d'）
    title: str
    url: str
    body_html: str
    body_text: str      # HTML タグ除去済みプレーンテキスト
    published_at: str   # ISO 8601
    is_paid: bool       # True の場合、body は空（取得不可）
    site: str           # フェッチャー識別子（例: "note"）
    account: str        # アカウント識別子
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "url": self.url,
            "body_html": self.body_html,
            "body_text": self.body_text,
            "published_at": self.published_at,
            "is_paid": self.is_paid,
            "site": self.site,
            "account": self.account,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FetchedArticle":
        return cls(**d)


# ── 基底フェッチャー ─────────────────────────────────────────────────────────

class BaseFetcher(ABC):
    """サイト別フェッチャーの基底クラス。"""

    site_name: str  # サブクラスで設定必須

    @abstractmethod
    def fetch_article_list(
        self,
        account: str,
        delay_range: tuple[float, float],
        limit: int | None = None,
    ) -> list[dict]:
        """記事メタデータ一覧を返す（ページネーション対応）。"""
        ...

    @abstractmethod
    def fetch_article(
        self,
        key: str,
        delay_range: tuple[float, float],
        account: str,
    ) -> FetchedArticle | None:
        """記事詳細（本文含む）を返す。有料記事は is_paid=True で本文空。"""
        ...

    def is_paid(self, article_meta: dict) -> bool:
        """メタデータから有料記事かどうかを判定。サブクラスでオーバーライド可。"""
        return False

    def get_key(self, article_meta: dict) -> str:
        """メタデータから記事キーを返す。サブクラスでオーバーライド可。"""
        return article_meta.get("key", "")

    def __enter__(self) -> "BaseFetcher":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        pass
