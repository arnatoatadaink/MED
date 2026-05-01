"""src/sandbox/code_extractor.py — ドキュメントからコードブロックを抽出

DB に格納されたドキュメントの Markdown コードブロック (```lang\n...\n```) を
解析し、サンドボックスで実行可能な CodeBlock のリストを返す。

使い方:
    from src.sandbox.code_extractor import CodeExtractor, CodeBlock

    extractor = CodeExtractor()
    blocks = await extractor.extract_from_db(languages=["python", "bash"], limit=500)
    for b in blocks:
        print(b.language, b.code[:60])
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---- 言語正規化マップ -------------------------------------------

_LANG_NORMALIZE: dict[str, str] = {
    # Python
    "python": "python", "python3": "python", "py": "python",
    # Bash / Shell
    "bash": "bash", "sh": "bash", "shell": "bash", "zsh": "bash",
    "console": "bash",    # $ プレフィックス付き CLI 出力混じりが多いが有用
    "terminal": "bash", "cmd": "bash",
    # JavaScript
    "javascript": "javascript", "js": "javascript",
    "mjs": "javascript", "cjs": "javascript", "node": "javascript",
    # TypeScript
    "typescript": "typescript", "ts": "typescript",
    # その他
    "sql": "sql", "json": "json", "yaml": "yaml", "toml": "toml",
}

# サンドボックスで実行できる言語（docker image が必要）
_EXECUTABLE: dict[str, str] = {
    "python":     "python:3.11-slim",
    "bash":       "python:3.11-slim",   # bash は python イメージにも入っている
    "javascript": "node:18-slim",
    "typescript": "node:18-slim",
}

# ---- テンプレート / プレースホルダー検出 -------------------------

_SKIP_PATTERNS: list[re.Pattern] = [
    re.compile(r'\{\{#include'),          # mdBook テンプレート (Rust docs)
    re.compile(r'\{\{.*?\}\}'),           # Handlebars / Jinja テンプレート
    re.compile(r'<your[-_]?\w+>'),        # <your-api-key> 等のプレースホルダー
    re.compile(r'\.\.\.\s*$', re.M),      # 省略記号のみの行
    re.compile(r'^#!\s*/usr/bin/env\s+(?!python|bash)', re.M),  # 未対応シェバン
]

# console ブロックで実行可能なのは $ で始まる行
_CONSOLE_PROMPT = re.compile(r'^\$\s+', re.M)

# ---- データクラス ------------------------------------------------

@dataclass
class CodeBlock:
    """1つのコードブロック。

    Attributes:
        doc_id: 元ドキュメントの ID。
        source_type: ドキュメントのソース種別。
        source_url: ドキュメントの URL。
        title: ドキュメントタイトル。
        language: 正規化後の言語名。
        raw_language: ``` の直後に書かれた元の言語タグ。
        code: コードブロックの本文。
        docker_image: 推奨 Docker イメージ。
        block_hash: コード本文の sha256 ハッシュ（重複除去用）。
        quality: 実行可能性スコア 0.0〜1.0。
    """

    doc_id: str
    source_type: str
    source_url: str
    title: str
    language: str
    raw_language: str
    code: str
    docker_image: str
    block_hash: str = field(init=False)
    quality: float = 1.0

    def __post_init__(self) -> None:
        self.block_hash = hashlib.sha256(self.code.encode()).hexdigest()[:16]

    @property
    def is_executable(self) -> bool:
        return self.language in _EXECUTABLE

    @property
    def run_command(self) -> str:
        """サンドボックスで実行するコマンド文字列を返す。"""
        if self.language == "python":
            return self.code
        if self.language == "bash":
            return self.code
        if self.language in ("javascript", "typescript"):
            return self.code
        return self.code


# ---- 抽出ロジック -----------------------------------------------

_CODE_FENCE = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)


def _normalize_lang(raw: str) -> str:
    return _LANG_NORMALIZE.get(raw.lower().strip(), raw.lower().strip() or "unknown")


def _is_skippable(code: str) -> bool:
    """テンプレート・プレースホルダーが含まれるコードを弾く。"""
    for pat in _SKIP_PATTERNS:
        if pat.search(code):
            return True
    return False


def _console_to_bash(code: str) -> str:
    """console ブロックから $ プレフィックスの行だけ抽出してシェルコードにする。"""
    lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if _CONSOLE_PROMPT.match(stripped):
            lines.append(_CONSOLE_PROMPT.sub("", stripped))
    return "\n".join(lines)


def extract_blocks(
    doc_id: str,
    source_type: str,
    source_url: str,
    title: str,
    content: str,
    languages: Optional[list[str]] = None,
    min_length: int = 10,
    max_length: int = 4000,
) -> list[CodeBlock]:
    """1ドキュメントからコードブロックを抽出する。

    Args:
        doc_id: ドキュメント ID。
        source_type: ソース種別。
        source_url: URL。
        title: タイトル。
        content: ドキュメント本文（Markdown）。
        languages: 対象言語リスト（None = 全言語）。
        min_length: コードの最小文字数。
        max_length: コードの最大文字数（長過ぎるブロックをスキップ）。

    Returns:
        CodeBlock のリスト。
    """
    blocks: list[CodeBlock] = []

    for raw_lang, code in _CODE_FENCE.findall(content):
        lang = _normalize_lang(raw_lang)

        # 言語フィルター
        if languages and lang not in languages:
            continue

        # console → bash 変換
        if raw_lang.lower() == "console":
            code = _console_to_bash(code)
            if not code:
                continue

        code = code.strip()

        # 長さフィルター
        if len(code) < min_length or len(code) > max_length:
            continue

        # テンプレート除去
        if _is_skippable(code):
            continue

        docker_image = _EXECUTABLE.get(lang, "")

        blocks.append(CodeBlock(
            doc_id=doc_id,
            source_type=source_type,
            source_url=source_url,
            title=title,
            language=lang,
            raw_language=raw_lang,
            code=code,
            docker_image=docker_image,
        ))

    return blocks


# ---- DB クエリ --------------------------------------------------

class CodeExtractor:
    """DB からドキュメントを読み込み、コードブロックを抽出する。

    Args:
        db_path: metadata.db のパス。
    """

    def __init__(self, db_path: str = "data/metadata.db") -> None:
        self._db_path = db_path

    def extract_from_db(
        self,
        languages: Optional[list[str]] = None,
        source_types: Optional[list[str]] = None,
        review_status: Optional[list[str]] = None,
        limit: int = 1000,
        deduplicate: bool = True,
        min_length: int = 10,
        max_length: int = 4000,
    ) -> list[CodeBlock]:
        """DB からドキュメントを取得しコードブロックを抽出する（同期版）。

        Args:
            languages: 対象言語（None = 全言語）。実行可能なものだけなら
                       ["python", "bash", "javascript"] を指定。
            source_types: 対象ソース種別（None = 全種別）。
            review_status: 対象レビューステータス（None = 全ステータス）。
            limit: 処理するドキュメント数の上限。
            deduplicate: 同一コードの重複を除去するか。
            min_length: コードの最小文字数。
            max_length: コードの最大文字数。

        Returns:
            CodeBlock のリスト。
        """
        conn = sqlite3.connect(self._db_path)
        try:
            return self._query_and_extract(
                conn, languages, source_types, review_status,
                limit, deduplicate, min_length, max_length,
            )
        finally:
            conn.close()

    def _query_and_extract(
        self,
        conn: sqlite3.Connection,
        languages: Optional[list[str]],
        source_types: Optional[list[str]],
        review_status: Optional[list[str]],
        limit: int,
        deduplicate: bool,
        min_length: int,
        max_length: int,
    ) -> list[CodeBlock]:
        conditions = ["content LIKE '%```%'"]
        params: list[str | int] = []

        if source_types:
            placeholders = ",".join("?" * len(source_types))
            conditions.append(f"source_type IN ({placeholders})")
            params.extend(source_types)

        if review_status:
            placeholders = ",".join("?" * len(review_status))
            conditions.append(f"review_status IN ({placeholders})")
            params.extend(review_status)

        where = " AND ".join(conditions)
        params.append(limit)

        rows = conn.execute(
            f"SELECT id, source_type, source_url, source_title, content "
            f"FROM documents WHERE {where} ORDER BY RANDOM() LIMIT ?",
            params,
        ).fetchall()

        all_blocks: list[CodeBlock] = []
        seen_hashes: set[str] = set()

        for doc_id, source_type, url, title, content in rows:
            for block in extract_blocks(
                doc_id=doc_id,
                source_type=source_type or "",
                source_url=url or "",
                title=title or "",
                content=content or "",
                languages=languages,
                min_length=min_length,
                max_length=max_length,
            ):
                if deduplicate and block.block_hash in seen_hashes:
                    continue
                seen_hashes.add(block.block_hash)
                all_blocks.append(block)

        return all_blocks

    def stats(
        self,
        languages: Optional[list[str]] = None,
        source_types: Optional[list[str]] = None,
    ) -> dict[str, int]:
        """言語 × ドキュメント数の統計を返す（軽量カウントのみ）。"""
        conn = sqlite3.connect(self._db_path)
        try:
            conditions = ["content LIKE '%```%'"]
            params: list[str] = []
            if source_types:
                placeholders = ",".join("?" * len(source_types))
                conditions.append(f"source_type IN ({placeholders})")
                params.extend(source_types)
            where = " AND ".join(conditions)
            rows = conn.execute(
                f"SELECT content FROM documents WHERE {where} LIMIT 5000", params
            ).fetchall()
        finally:
            conn.close()

        lang_counts: dict[str, int] = {}
        for (content,) in rows:
            for raw_lang in re.findall(r'```(\w*)\n', content or ""):
                lang = _normalize_lang(raw_lang)
                if languages and lang not in languages:
                    continue
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
        return dict(sorted(lang_counts.items(), key=lambda x: -x[1]))
