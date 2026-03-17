"""src/rag/query_expander.py — ルールベースクエリ展開

YAML パターン設定に基づき、複合クエリを複数のサブクエリに展開する。

例:
    "TinyLoRAとLoRAの比較" → ["TinyLoRA", "LoRA", "TinyLoRA LoRA"]
    "Python vs Java"       → ["Python", "Java", "Python Java"]
    "RAGを使ったQA"        → ["RAG", "QA", "RAG QA"]

パターンにマッチしない場合は元クエリのみを返す（展開なし）。
パターンファイルは configs/query_expansion.yaml で管理し、
reload() でホットリロード可能。

使い方:
    expander = QueryExpander()
    queries = expander.expand("TinyLoRAとLoRAの比較")
    # → ["TinyLoRA", "LoRA", "TinyLoRA LoRA"]

    if expander.is_negative(llm_answer):
        # リトライをトリガー
        ...
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "configs" / "query_expansion.yaml"

# パターンに関係なく「助詞・動詞・比較語」と判断するキーワード
_PARTICLES: frozenset[str] = frozenset({
    "比較", "違い", "差", "差異", "相違", "difference", "differences",
    "vs", "versus", "対", "組み合わせ", "統合", "連携",
    "using", "with", "for", "to", "in",
    "による", "での", "を比べ", "を比較",
    "compare", "comparison",
})


class QueryExpander:
    """YAML 設定ファイルに基づくルールベースクエリ展開クラス。

    Args:
        config_path: YAML 設定ファイルパス（省略時は configs/query_expansion.yaml）。
    """

    def __init__(self, config_path: Path | None = None) -> None:
        self._config_path = config_path or _DEFAULT_CONFIG
        self._compiled: list[tuple[str, re.Pattern]] = []  # (name, pattern)
        self._negative_signals: list[str] = []
        self._max_expanded: int = 3
        self._retry_max_results: int = 5
        self._load_config()

    # ── 設定管理 ─────────────────────────────────────────────────

    def _load_config(self) -> None:
        """YAML からパターン・シグナルを読み込む。"""
        try:
            with open(self._config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            self._compiled = [
                (name, re.compile(pat, re.IGNORECASE | re.UNICODE))
                for name, pat in cfg.get("patterns", {}).items()
            ]
            self._negative_signals = [
                s.lower() for s in cfg.get("negative_signals", [])
            ]
            self._max_expanded = int(cfg.get("max_expanded", 3))
            self._retry_max_results = int(cfg.get("retry_max_results_per_query", 5))
            logger.debug(
                "QueryExpander loaded: %d patterns, %d signals",
                len(self._compiled), len(self._negative_signals),
            )
        except FileNotFoundError:
            logger.warning(
                "query_expansion.yaml not found at %s — expander disabled",
                self._config_path,
            )
        except Exception:
            logger.exception("Failed to load query_expansion.yaml")

    def reload(self) -> None:
        """設定ファイルを再読み込みする（ホットリロード用）。"""
        self._compiled = []
        self._negative_signals = []
        self._load_config()

    # ── 公開 API ─────────────────────────────────────────────────

    def expand(self, query: str) -> list[str]:
        """クエリをサブクエリリストに展開する。

        パターンにマッチした場合、各キャプチャグループ（助詞除く）と
        それらを結合したクエリを返す。

        マッチしない場合は [query] のみを返す。

        Args:
            query: 元のクエリ文字列。

        Returns:
            展開済みクエリリスト（重複除去・最大 max_expanded 件）。
        """
        q = query.strip()

        # ── パターンマッチング ──────────────────────
        for name, pattern in self._compiled:
            m = pattern.search(q)
            if not m:
                continue
            # キャプチャグループから助詞・比較語を除去
            terms = [
                g.strip()
                for g in m.groups()
                if g and g.strip() and not self._is_particle(g.strip())
            ]
            if len(terms) < 2:
                continue
            # terms + 結合クエリ（重複除去・順序保持）
            combined = " ".join(terms)
            result = list(dict.fromkeys(terms + [combined]))
            logger.debug("QueryExpander: pattern=%s → %s", name, result)
            return result[: self._max_expanded]

        # ── フォールバック展開 ──────────────────────
        return self._fallback_expand(q)[: self._max_expanded]

    def is_negative(self, answer: str) -> bool:
        """LLM 出力が情報不足（ネガティブ）かどうかを判定する。

        Args:
            answer: LLM が生成した回答テキスト。

        Returns:
            True = 情報不足と判断 → リトライ推奨。
        """
        if not self._negative_signals:
            return False
        lower = answer.lower()
        return any(sig in lower for sig in self._negative_signals)

    @property
    def retry_max_results(self) -> int:
        """リトライ時に展開クエリ1件あたり取得する RAG 結果数。"""
        return self._retry_max_results

    # ── 内部ヘルパー ─────────────────────────────────────────────

    @staticmethod
    def _is_particle(text: str) -> bool:
        """テキストが助詞・比較語・動詞かどうかを判定する。"""
        return text.lower() in _PARTICLES

    @staticmethod
    def _fallback_expand(query: str) -> list[str]:
        """パターン未マッチ時のフォールバック展開。

        日本語の「と」「や」「および」、英語の「and」で分割を試みる。
        分割できなければ元クエリのみを返す。
        """
        parts = re.split(r'[とや]|および|\band\b', query)
        parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 1]
        if len(parts) >= 2:
            combined = " ".join(parts)
            return list(dict.fromkeys(parts + [combined]))
        return [query]
