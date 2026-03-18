"""src/memory/keyword_annotator.py — ルールベースキーワードアノテーション

ドキュメント本文から技術用語・固有名詞・略称を LLM を使わず決定論的に抽出する。
抽出結果は SQLite の `keywords` 列に保存され、FTS5 全文検索と組み合わせることで
「TinyLoRA」など文書本文に明示されていない用語の検索精度を高める。

抽出する用語の種類:
  - CamelCase 識別子: TinyLoRA, PyTorch, NetworkX, LearningToReason
  - ALLCAPS 略語: RAG, FAISS, LLM, KG, BERT, GPT
  - ハイフン複合語: fine-tuning, cross-encoder, all-MiniLM
  - バージョン付き名称: GPT-4, Qwen2.5, BERT-base, v1.0
  - 数字+単位パターン: 13B, 7B, 768d
  - タイトル語: source.title のスペース区切りトークン（全て）

コスト: ゼロ（正規表現ベース、LLM 呼び出しなし）
速度: ドキュメント 1 件あたり < 1ms

使い方:
    annotator = KeywordAnnotator()
    keywords = annotator.extract(doc)
    # → ["TinyLoRA", "LoRA", "RAG", "FAISS", "13B", "fine-tuning", ...]
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.memory.schema import Document

# ── 抽出パターン ──────────────────────────────────────────────────────

# CamelCase: 大文字で始まり小文字が続く 2 ブロック以上 (TinyLoRA, PyTorch)
_RE_CAMEL = re.compile(r'\b[A-Z][a-z]+(?:[A-Z][a-z0-9]*)+\b')

# ALLCAPS 略語: 2 文字以上の全大文字 (RAG, FAISS, LLM, KG, GPT)
_RE_ALLCAPS = re.compile(r'\b[A-Z]{2,}\b')

# ハイフン複合語: 英数字-英数字 (fine-tuning, cross-encoder, all-MiniLM)
_RE_HYPHEN = re.compile(r'\b[A-Za-z][\w]*(?:-[\w]+)+\b')

# バージョン付き名称: 英字+数字ドット (GPT-4, Qwen2.5, v1.0, BERT-base-uncased)
_RE_VERSION = re.compile(r'\b[A-Za-z][\w]*[-.][\d][\w.]*\b|\bv\d+(?:\.\d+)+\b')

# 数字+単位: 7B, 13B, 768d, 512M (モデルサイズ等)
_RE_NUM_UNIT = re.compile(r'\b\d+[BbMmGgKkDd]\b')

# 除外リスト: 一般的すぎる ALLCAPS (前置詞・接続詞等)
_COMMON_ALLCAPS_EXCLUDE: frozenset[str] = frozenset({
    "A", "AN", "THE", "IN", "ON", "AT", "BY", "OF", "OR", "AND", "BUT",
    "FOR", "TO", "IS", "ARE", "WAS", "BE", "IT", "AS", "WE", "HE", "SHE",
    "IF", "DO", "US", "SO", "NO", "UP", "GO", "MY", "NEW", "OLD",
})

# 最小キーワード長（文字）
_MIN_KEYWORD_LEN = 2

# 最大キーワード数（上位のみ保持）
_MAX_KEYWORDS = 50


class KeywordAnnotator:
    """ドキュメントから技術用語を決定論的に抽出するアノテーター。

    Args:
        max_content_chars: キーワード抽出に使う本文の最大文字数。
                          大きくすると精度向上・処理時間増加（デフォルト: 3000）。
        max_keywords:      保持する最大キーワード数。
    """

    def __init__(
        self,
        max_content_chars: int = 3000,
        max_keywords: int = _MAX_KEYWORDS,
    ) -> None:
        self._max_content_chars = max_content_chars
        self._max_keywords = max_keywords

    def extract(self, doc: "Document") -> list[str]:
        """ドキュメントからキーワードリストを抽出する。

        Args:
            doc: 対象 Document。

        Returns:
            重複除去・上位 max_keywords 件のキーワードリスト。
        """
        text = doc.content[: self._max_content_chars]
        title = doc.source.title or ""

        keywords: list[str] = []

        # タイトルのトークンは全て追加（高優先度）
        if title:
            title_tokens = [t.strip(".,;:!?\"'()[]{}") for t in title.split()]
            keywords.extend(t for t in title_tokens if len(t) >= _MIN_KEYWORD_LEN)

        # 各パターンでキーワードを抽出
        keywords.extend(self._extract_patterns(text))

        # 重複除去・順序保持・上限適用
        seen: dict[str, None] = {}
        for kw in keywords:
            kw_clean = kw.strip()
            if kw_clean and kw_clean.lower() not in seen:
                seen[kw_clean.lower()] = None
        result = list(seen.keys())[: self._max_keywords]

        # 実際の値（大文字小文字保持）で返す
        lower_to_original: dict[str, str] = {}
        for kw in keywords:
            kw_clean = kw.strip()
            if kw_clean and kw_clean.lower() not in lower_to_original:
                lower_to_original[kw_clean.lower()] = kw_clean
        return [lower_to_original[k] for k in result]

    @staticmethod
    def _extract_patterns(text: str) -> list[str]:
        """正規表現パターンでテキストからキーワードを抽出する。"""
        keywords: list[str] = []

        # ALLCAPS 略語（一般語を除く）
        for m in _RE_ALLCAPS.finditer(text):
            word = m.group()
            if word not in _COMMON_ALLCAPS_EXCLUDE and len(word) >= _MIN_KEYWORD_LEN:
                keywords.append(word)

        # CamelCase 識別子
        keywords.extend(m.group() for m in _RE_CAMEL.finditer(text))

        # ハイフン複合語
        keywords.extend(m.group() for m in _RE_HYPHEN.finditer(text))

        # バージョン付き名称
        keywords.extend(m.group() for m in _RE_VERSION.finditer(text))

        # 数字+単位
        keywords.extend(m.group() for m in _RE_NUM_UNIT.finditer(text))

        return keywords

    def search_in_doc(self, doc: "Document", keyword: str) -> bool:
        """ドキュメント本文にキーワードが含まれるかを判定する（大文字小文字無視）。

        全文検索ではなく、特定用語のアノテーション判定に使用する。

        Args:
            doc:     対象ドキュメント。
            keyword: 検索キーワード。

        Returns:
            True = キーワードが本文またはタイトルに含まれる。
        """
        kw = keyword.lower()
        return (
            kw in doc.content.lower()
            or (doc.source.title is not None and kw in doc.source.title.lower())
        )
