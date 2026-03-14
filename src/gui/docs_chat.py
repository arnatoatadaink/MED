"""src/gui/docs_chat.py — ドキュメント Q&A チャットエンジン（案C）

docs/site/ 以下の MkDocs Markdown ファイルを読み込み、
インメモリ FAISS インデックスを構築して自然言語で検索・回答する。

依存関係:
  - sentence-transformers + faiss-cpu インストール済み → ベクトル検索
  - 未インストール → キーワード検索にフォールバック

LLM 統合:
  - オーケストレーターが起動中 → POST /query で回答生成
  - 未起動 → 検索結果のチャンクをそのまま整形して返す
"""

from __future__ import annotations

import logging
import re
import textwrap
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# ドキュメントディレクトリ
_DOCS_DIR = Path(__file__).parent.parent.parent / "docs" / "site"
_ORCHESTRATOR_URL = "http://localhost:8000"
_MAX_CONTEXT_CHARS = 4000  # LLM プロンプトに渡すコンテキスト最大文字数
_CHUNK_SIZE = 400           # 1 チャンクの目標文字数
_CHUNK_OVERLAP = 80         # チャンク間のオーバーラップ文字数


# ──────────────────────────────────────────────────────────────────
# データ構造
# ──────────────────────────────────────────────────────────────────

@dataclass
class DocChunk:
    """ドキュメントの 1 チャンク。"""
    doc_id: int
    source_file: str    # 相対パス (例: "features/memory.md")
    section: str        # ## 見出しテキスト
    text: str           # チャンク本文
    score: float = 0.0  # 検索スコア


# ──────────────────────────────────────────────────────────────────
# ドキュメント読み込み & チャンキング
# ──────────────────────────────────────────────────────────────────

def _extract_title(content: str) -> str:
    """Markdown の最初の # 見出しを返す。"""
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _split_by_headings(content: str) -> list[tuple[str, str]]:
    """
    Markdown を ## 見出し単位に分割する。
    Returns list of (heading, text) pairs.
    """
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []

    for line in content.splitlines():
        if re.match(r"^#{1,3} ", line):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = line.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    return [(h, t) for h, t in sections if t]


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """テキストを固定長チャンクに分割する。"""
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def load_docs(docs_dir: Path = _DOCS_DIR) -> list[DocChunk]:
    """docs_dir 以下の全 .md ファイルを読み込んでチャンクリストを返す。"""
    chunks: list[DocChunk] = []
    doc_id = 0

    if not docs_dir.exists():
        logger.warning("Docs directory not found: %s", docs_dir)
        return chunks

    md_files = sorted(docs_dir.rglob("*.md"))
    logger.info("Loading %d markdown files from %s", len(md_files), docs_dir)

    for md_path in md_files:
        rel_path = str(md_path.relative_to(docs_dir))
        try:
            content = md_path.read_text(encoding="utf-8")
        except OSError:
            continue

        title = _extract_title(content)
        sections = _split_by_headings(content)

        for heading, section_text in sections:
            section_label = f"{title} — {heading}" if heading else title
            for chunk_text in _chunk_text(section_text, _CHUNK_SIZE, _CHUNK_OVERLAP):
                if len(chunk_text.strip()) < 30:
                    continue
                chunks.append(DocChunk(
                    doc_id=doc_id,
                    source_file=rel_path,
                    section=section_label,
                    text=chunk_text.strip(),
                ))
                doc_id += 1

    logger.info("Loaded %d chunks from docs", len(chunks))
    return chunks


# ──────────────────────────────────────────────────────────────────
# 検索エンジン（FAISS or キーワード）
# ──────────────────────────────────────────────────────────────────

class _KeywordSearcher:
    """FAISS が使えない場合のフォールバック: ASCII + 日本語対応キーワードスコア。"""

    def __init__(self, chunks: list[DocChunk]) -> None:
        self._chunks = chunks

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """ASCII 単語と日本語 2-gram を抽出する。"""
        tokens: list[str] = []
        # ASCII: 単語単位
        tokens += [t.lower() for t in re.findall(r"[a-zA-Z0-9_]+", text)]
        # 日本語・CJK: 2-gram
        cjk_chars = re.findall(r"[\u3000-\u9fff\uff00-\uffef]", text)
        tokens += [
            "".join(cjk_chars[i : i + 2])
            for i in range(len(cjk_chars) - 1)
        ]
        # 個別 CJK 文字も追加（1-gram）
        tokens += cjk_chars
        return tokens

    def search(self, query: str, top_k: int = 5) -> list[DocChunk]:
        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return []
        scored: list[tuple[float, DocChunk]] = []
        for chunk in self._chunks:
            combined = chunk.section + " " + chunk.text
            chunk_tokens = set(self._tokenize(combined))
            overlap = len(query_tokens & chunk_tokens)
            if overlap > 0:
                score = overlap / (len(query_tokens) + 1)
                scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, chunk in scored[:top_k]:
            results.append(DocChunk(
                doc_id=chunk.doc_id,
                source_file=chunk.source_file,
                section=chunk.section,
                text=chunk.text,
                score=score,
            ))
        return results


class _FAISSSearcher:
    """sentence-transformers + FAISS によるベクトル検索。"""

    def __init__(self, chunks: list[DocChunk], dim: int = 384) -> None:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self._chunks = chunks
        self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        texts = [f"{c.section}\n{c.text}" for c in chunks]
        logger.info("Encoding %d chunks for FAISS index...", len(texts))
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        actual_dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(actual_dim)
        self._index.add(embeddings)
        logger.info("FAISS index built: %d vectors (dim=%d)", len(chunks), actual_dim)

    def search(self, query: str, top_k: int = 5) -> list[DocChunk]:
        import numpy as np

        q_vec = self._model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        scores, indices = self._index.search(q_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            results.append(DocChunk(
                doc_id=chunk.doc_id,
                source_file=chunk.source_file,
                section=chunk.section,
                text=chunk.text,
                score=float(score),
            ))
        return results


# ──────────────────────────────────────────────────────────────────
# メインクラス
# ──────────────────────────────────────────────────────────────────

class DocsChatEngine:
    """MkDocs ドキュメントを検索して回答する Q&A エンジン。

    GUI 起動時に一度だけインスタンスを作成し、get_engine() 経由でアクセスする。
    """

    def __init__(self) -> None:
        self._chunks: list[DocChunk] = []
        self._searcher: _FAISSSearcher | _KeywordSearcher | None = None
        self._initialized = False

    def initialize(self) -> None:
        """ドキュメントを読み込んでインデックスを構築する（遅延初期化）。"""
        if self._initialized:
            return
        self._initialized = True

        self._chunks = load_docs()
        if not self._chunks:
            logger.warning("No doc chunks loaded. Chatbot will return empty results.")
            return

        # FAISS が使えるか試みる
        try:
            import faiss  # noqa: F401
            from sentence_transformers import SentenceTransformer  # noqa: F401
            self._searcher = _FAISSSearcher(self._chunks)
            logger.info("DocsChatEngine: using FAISS vector search")
        except ImportError:
            logger.info("DocsChatEngine: FAISS/ST not available, using keyword search")
            self._searcher = _KeywordSearcher(self._chunks)

    def search(self, query: str, top_k: int = 5) -> list[DocChunk]:
        """クエリに関連するチャンクを返す。"""
        if not self._initialized:
            self.initialize()
        if self._searcher is None:
            return []
        return self._searcher.search(query, top_k=top_k)

    def _build_context(self, chunks: list[DocChunk]) -> str:
        """LLM プロンプト用のコンテキスト文字列を構築する。"""
        parts: list[str] = []
        total = 0
        for i, chunk in enumerate(chunks, 1):
            block = f"[{i}] **{chunk.section}**\n{chunk.text}"
            if total + len(block) > _MAX_CONTEXT_CHARS:
                break
            parts.append(block)
            total += len(block)
        return "\n\n---\n\n".join(parts)

    def _format_fallback_answer(self, query: str, chunks: list[DocChunk]) -> str:
        """オーケストレーター未接続時の回答: チャンクを整形して返す。"""
        if not chunks:
            return (
                "ドキュメントに一致する情報が見つかりませんでした。\n\n"
                "別のキーワードで試すか、[📖 ガイド] のセクションを直接参照してください。"
            )
        lines = [
            f"**「{query}」** に関連するドキュメントを見つけました:\n",
        ]
        for i, chunk in enumerate(chunks[:3], 1):
            lines.append(f"### {i}. {chunk.section}")
            # 長すぎるチャンクは短縮
            preview = textwrap.shorten(chunk.text, width=300, placeholder="...")
            lines.append(preview)
            lines.append(f"_📄 `{chunk.source_file}`_\n")
        lines.append(
            "\n---\n_💡 オーケストレーターを起動すると LLM による詳細な回答が得られます:_\n"
            "```bash\nuvicorn src.orchestrator.server:app --reload --port 8000\n```"
        )
        return "\n\n".join(lines)

    def _query_orchestrator(self, query: str, context: str) -> str | None:
        """オーケストレーターに context 付きクエリを送って回答を取得する。"""
        prompt = (
            f"以下は MED システムのドキュメントです:\n\n"
            f"{context}\n\n"
            f"---\n\n"
            f"上記のドキュメントをもとに、以下の質問に日本語で回答してください:\n\n"
            f"{query}"
        )
        try:
            r = httpx.post(
                f"{_ORCHESTRATOR_URL}/query",
                json={
                    "prompt": prompt,
                    "mode": "teacher",
                    "use_memory": False,
                    "use_rag": False,
                },
                timeout=60.0,
            )
            r.raise_for_status()
            data = r.json()
            answer = data.get("answer", "")
            model = data.get("model_used", "")
            if answer:
                suffix = f"\n\n---\n_モデル: {model}_" if model else ""
                return answer + suffix
        except Exception as e:
            logger.debug("Orchestrator query failed: %s", e)
        return None

    def answer(self, query: str, top_k: int = 5) -> str:
        """クエリに対して回答文字列を返す（ストリームなし）。"""
        if not query.strip():
            return "質問を入力してください。"

        chunks = self.search(query, top_k=top_k)
        context = self._build_context(chunks)

        # オーケストレーターが起動中なら LLM で回答
        if context:
            llm_answer = self._query_orchestrator(query, context)
            if llm_answer:
                src_note = "\n\n**参照ドキュメント:** " + ", ".join(
                    {c.source_file for c in chunks}
                )
                return llm_answer + src_note

        # フォールバック: チャンクをそのまま整形
        return self._format_fallback_answer(query, chunks)

    def answer_stream(
        self, query: str, history: list, top_k: int = 5
    ) -> Generator[list, None, None]:
        """Gradio Chatbot ストリーミング用: history を更新しながら yield する。"""
        if not query.strip():
            yield history
            return

        # "考え中..." を先に表示
        _history = list(history) + [{"role": "user", "content": query}]
        _history.append({"role": "assistant", "content": "⏳ 検索中..."})
        yield _history

        full_answer = self.answer(query, top_k=top_k)
        _history[-1]["content"] = full_answer
        yield _history


# ──────────────────────────────────────────────────────────────────
# シングルトン
# ──────────────────────────────────────────────────────────────────

_engine: DocsChatEngine | None = None


def get_engine() -> DocsChatEngine:
    """DocsChatEngine のシングルトンを返す。初回呼び出し時に遅延初期化する。"""
    global _engine
    if _engine is None:
        _engine = DocsChatEngine()
    return _engine
