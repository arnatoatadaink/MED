"""src/rag/chunker.py — テキストチャンク化

外部検索結果のテキストを Document に変換する際に適切なサイズに分割する。
オーバーラップによって文脈の連続性を保つ。

使い方:
    from src.rag.chunker import Chunker
    from src.rag.retriever import RawResult

    chunker = Chunker(chunk_size=512, chunk_overlap=50)
    docs = chunker.chunk_result(result, domain="code")
"""

from __future__ import annotations

import logging
from typing import Optional

from src.memory.schema import Document, Domain, SourceMeta, SourceType
from src.rag.retriever import RawResult

logger = logging.getLogger(__name__)

_SOURCE_TYPE_MAP = {
    "github": SourceType.GITHUB,
    "stackoverflow": SourceType.STACKOVERFLOW,
    "tavily": SourceType.TAVILY,
    "arxiv": SourceType.ARXIV,
}


class Chunker:
    """テキストを固定サイズのチャンクに分割し、Document リストを生成する。

    Args:
        chunk_size: 各チャンクの最大文字数。
        chunk_overlap: 隣接チャンク間のオーバーラップ文字数。
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> list[str]:
        """テキストをチャンクリストに分割する。

        Returns:
            チャンク文字列のリスト。
        """
        if not text:
            return []

        chunks: list[str] = []
        start = 0
        step = self._chunk_size - self._chunk_overlap

        while start < len(text):
            end = start + self._chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start += step

        return chunks

    def chunk_result(
        self,
        result: RawResult,
        domain: str = "general",
        parent_id: Optional[str] = None,
    ) -> list[Document]:
        """RawResult を Document リストに変換する。

        Args:
            result: 外部検索の生結果。
            domain: 対象ドメイン。
            parent_id: 親ドキュメント ID（チャンク間の関係を追跡）。

        Returns:
            チャンク化された Document のリスト。
        """
        full_text = f"{result.title}\n\n{result.content}".strip()
        chunks = self.chunk_text(full_text)

        if not chunks:
            return []

        source_type = _SOURCE_TYPE_MAP.get(result.source, SourceType.TAVILY)

        documents: list[Document] = []
        first_doc_id: Optional[str] = None

        for idx, chunk_text in enumerate(chunks):
            source_meta = SourceMeta(
                source_type=source_type,
                url=result.url,
                title=result.title,
                tags=[result.source],
                extra=result.metadata,
            )
            doc = Document(
                content=chunk_text,
                domain=domain,
                source=source_meta,
                chunk_index=idx,
                parent_id=parent_id or (first_doc_id if idx > 0 else None),
            )
            if idx == 0:
                first_doc_id = doc.id
            documents.append(doc)

        return documents

    def chunk_results(
        self,
        results: list[RawResult],
        domain: str = "general",
    ) -> list[Document]:
        """複数の RawResult を一括チャンク化する。"""
        all_docs: list[Document] = []
        for result in results:
            docs = self.chunk_result(result, domain=domain)
            all_docs.extend(docs)
        return all_docs
