"""src/rag/chunker.py — テキストチャンク化

外部検索結果のテキストを Document に変換する際に適切なサイズに分割する。
段落・文境界を優先して分割し、文の途中でのカットを避ける。

使い方:
    from src.rag.chunker import Chunker
    from src.rag.retriever import RawResult

    chunker = Chunker(chunk_size=1500, chunk_overlap=100, min_chunk_len=100)
    docs = chunker.chunk_result(result, domain="code")
"""

from __future__ import annotations

import logging
import re

from src.memory.schema import Document, SourceMeta, SourceType
from src.rag.retriever import RawResult

logger = logging.getLogger(__name__)

_SOURCE_TYPE_MAP = {
    "github": SourceType.GITHUB,
    "stackoverflow": SourceType.STACKOVERFLOW,
    "tavily": SourceType.TAVILY,
    "arxiv": SourceType.ARXIV,
    "github_docs": SourceType.GITHUB_DOCS,
    "web_docs": SourceType.WEB_DOCS,
}

# 文末境界パターン（句点・感嘆符・疑問符の後ろの空白）
_SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+')
# 段落区切り
_PARA_SPLIT_RE = re.compile(r'\n\n+')
# 行区切り（単一改行）
_LINE_SPLIT_RE = re.compile(r'\n')
# Markdown 見出し（# / ## / ###）
_HEADING_RE = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)
# 実質的な文を数えるための簡易パターン（コードブロック・見出し・空行を除く）
_MEANINGFUL_LINE_RE = re.compile(r'^(?!#|```|\s*$).{20,}', re.MULTILINE)


class Chunker:
    """テキストを段落・文境界優先でチャンク分割し、Document リストを生成する。

    固定文字数での強制カットを避け、自然な境界（段落→文→行）で分割する。
    chunk_size を超える場合のみ文字数制限にフォールバックする。

    Args:
        chunk_size: 各チャンクの目標最大文字数。
        chunk_overlap: 隣接チャンク間のオーバーラップ文字数（文字ベースフォールバック時のみ使用）。
        min_chunk_len: この文字数未満のチャンクは断片とみなし除外する。
    """

    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 100,
        min_chunk_len: int = 100,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_chunk_len = min_chunk_len

    def chunk_text(self, text: str) -> list[str]:
        """テキストを段落・文境界優先でチャンクリストに分割する。

        分割優先度: 段落境界 → 文境界 → 行境界 → 文字数フォールバック

        Returns:
            チャンク文字列のリスト。min_chunk_len 未満は除外される。
        """
        if not text:
            return []

        # テキスト全体が chunk_size 以内なら分割不要
        text = text.strip()
        if len(text) <= self._chunk_size:
            if len(text) >= self._min_chunk_len:
                return [text]
            return []

        chunks: list[str] = []
        current = ""

        # 段落単位で処理
        paragraphs = [p.strip() for p in _PARA_SPLIT_RE.split(text) if p.strip()]

        for para in paragraphs:
            # 現在バッファ + 段落が chunk_size 以内なら結合
            candidate = (current + "\n\n" + para).strip() if current else para
            if len(candidate) <= self._chunk_size:
                current = candidate
                continue

            # chunk_size を超える場合: まず現在バッファを確定
            if current:
                chunks.append(current)
                current = ""

            # 段落自体が chunk_size 以内なら新バッファに
            if len(para) <= self._chunk_size:
                current = para
                continue

            # 段落が chunk_size を超える場合: 文境界で分割
            sentences = _SENTENCE_END_RE.split(para)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                candidate = (current + " " + sent).strip() if current else sent
                if len(candidate) <= self._chunk_size:
                    current = candidate
                    continue

                # 文を追加しても超える場合: バッファ確定
                if current:
                    chunks.append(current)
                    current = ""

                # 1文が chunk_size 以内なら新バッファに
                if len(sent) <= self._chunk_size:
                    current = sent
                    continue

                # 1文が chunk_size を超える場合: 行境界で再分割
                lines = [l.strip() for l in _LINE_SPLIT_RE.split(sent) if l.strip()]
                for line in lines:
                    candidate = (current + " " + line).strip() if current else line
                    if len(candidate) <= self._chunk_size:
                        current = candidate
                        continue

                    if current:
                        chunks.append(current)
                        current = ""

                    # 1行が chunk_size を超える場合: 文字数フォールバック
                    if len(line) <= self._chunk_size:
                        current = line
                    else:
                        step = self._chunk_size - self._chunk_overlap
                        for i in range(0, len(line), step):
                            c = line[i:i + self._chunk_size].strip()
                            if c:
                                chunks.append(c)

        if current:
            chunks.append(current)

        # min_chunk_len 未満の断片を除外
        if self._min_chunk_len > 0:
            chunks = [c for c in chunks if len(c) >= self._min_chunk_len]

        return chunks

    def chunk_markdown(self, text: str, min_body_lines: int = 3) -> list[str]:
        """Markdown 見出し単位でチャンク分割する（GITHUB_DOCS 向け）。

        各 # / ## / ### セクションを基本チャンク単位とする。
        本文が短いセクションは次のセクションと結合し、chunk_size を超える場合は
        既存の chunk_text() にフォールバックする。

        Args:
            text: クリーニング済みの Markdown テキスト。
            min_body_lines: これ未満の実質行数のセクションは次と結合する。

        Returns:
            チャンク文字列のリスト。
        """
        if not text:
            return []

        # 見出し位置を特定
        heading_matches = list(_HEADING_RE.finditer(text))
        if not heading_matches:
            return self.chunk_text(text)

        # セクションテキストのリストを構築
        sections: list[str] = []
        preamble = text[:heading_matches[0].start()].strip()
        if preamble and len(preamble) >= self._min_chunk_len:
            sections.append(preamble)

        for i, m in enumerate(heading_matches):
            end = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(text)
            section = text[m.start():end].strip()
            if section:
                sections.append(section)

        # セクションを結合・分割してチャンクを生成
        chunks: list[str] = []
        buffer = ""

        for section in sections:
            # chunk_size を超えるセクション: 見出し行 + body を chunk_text() で分割
            if len(section) > self._chunk_size:
                if buffer and len(buffer) >= self._min_chunk_len:
                    chunks.append(buffer)
                    buffer = ""
                lines = section.split('\n', 1)
                heading_line = lines[0]
                body = lines[1].strip() if len(lines) > 1 else ""
                for sub in self.chunk_text(body):
                    chunks.append(f"{heading_line}\n\n{sub}")
                continue

            # 本文の実質行数が min_body_lines 未満なら次のセクションと結合
            body_lines = len(_MEANINGFUL_LINE_RE.findall(section))
            if body_lines < min_body_lines:
                buffer = (buffer + "\n\n" + section).strip() if buffer else section
                # バッファが chunk_size を超えたら確定
                if len(buffer) > self._chunk_size:
                    chunks.append(buffer)
                    buffer = ""
                continue

            # 通常: バッファに追加して chunk_size 以内なら結合
            candidate = (buffer + "\n\n" + section).strip() if buffer else section
            if len(candidate) <= self._chunk_size:
                buffer = candidate
            else:
                if buffer and len(buffer) >= self._min_chunk_len:
                    chunks.append(buffer)
                buffer = section

        if buffer and len(buffer) >= self._min_chunk_len:
            chunks.append(buffer)

        return chunks

    def chunk_result(
        self,
        result: RawResult,
        domain: str = "general",
        parent_id: str | None = None,
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
        # GITHUB_DOCS は見出し単位チャンキングを使用
        if result.source == "github_docs":
            chunks = self.chunk_markdown(full_text)
        else:
            chunks = self.chunk_text(full_text)

        if not chunks:
            return []

        source_type = _SOURCE_TYPE_MAP.get(result.source, SourceType.TAVILY)

        documents: list[Document] = []
        first_doc_id: str | None = None

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
