"""src/llm/prompt_cache.py — プロンプトキャッシュ

同一プロンプトへの LLM 呼び出しをキャッシュして API コストを削減する。
TTL（Time-To-Live）付きのインメモリ LRU キャッシュ。

使い方:
    from src.llm.prompt_cache import PromptCache

    cache = PromptCache(max_size=1000, ttl_seconds=3600)
    key = cache.make_key(prompt, system="...", model="...")
    if cache.get(key) is not None:
        return cache.get(key)
    response = await llm.complete(...)
    cache.set(key, response)
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """キャッシュエントリ。"""

    value: Any
    created_at: float
    ttl: float
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        if self.ttl <= 0:
            return False  # TTL=0 は永続
        return time.time() - self.created_at > self.ttl


class PromptCache:
    """TTL 付き LRU プロンプトキャッシュ。

    Args:
        max_size: 最大キャッシュエントリ数。
        ttl_seconds: TTL 秒数（0 で永続、デフォルト: 3600 = 1h）。
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 3600.0,
    ) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def make_key(
        self,
        prompt: str,
        system: str = "",
        model: str = "",
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        """キャッシュキーを生成する（SHA-256 ハッシュ）。"""
        key_content = f"{prompt}|{system}|{model}|{temperature}"
        return hashlib.sha256(key_content.encode()).hexdigest()[:32]

    def get(self, key: str) -> Optional[Any]:
        """キャッシュからエントリを取得する。

        Returns:
            キャッシュヒット時は値、ミス or 期限切れ時は None。
        """
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None

        if entry.is_expired:
            del self._cache[key]
            self._misses += 1
            logger.debug("PromptCache: expired key=%s", key[:8])
            return None

        # LRU: アクセスされたエントリを末尾に移動
        self._cache.move_to_end(key)
        entry.hit_count += 1
        self._hits += 1
        logger.debug("PromptCache: hit key=%s (count=%d)", key[:8], entry.hit_count)
        return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """キャッシュにエントリを保存する。"""
        # 既存エントリの更新
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl=ttl if ttl is not None else self._ttl,
            )
            return

        # 最大サイズ超過時は最古エントリを削除
        if len(self._cache) >= self._max_size:
            evicted = self._cache.popitem(last=False)
            logger.debug("PromptCache: evicted key=%s", evicted[0][:8])

        self._cache[key] = CacheEntry(
            value=value,
            created_at=time.time(),
            ttl=ttl if ttl is not None else self._ttl,
        )
        logger.debug("PromptCache: set key=%s (size=%d)", key[:8], len(self._cache))

    def invalidate(self, key: str) -> bool:
        """特定キーを無効化する。"""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """全キャッシュをクリアする。"""
        count = len(self._cache)
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("PromptCache: cleared %d entries", count)

    def evict_expired(self) -> int:
        """期限切れエントリを削除する。"""
        expired_keys = [k for k, v in self._cache.items() if v.is_expired]
        for k in expired_keys:
            del self._cache[k]
        if expired_keys:
            logger.debug("PromptCache: evicted %d expired entries", len(expired_keys))
        return len(expired_keys)

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
            "ttl_seconds": self._ttl,
        }
