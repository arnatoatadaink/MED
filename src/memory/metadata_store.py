"""src/memory/metadata_store.py — SQLite メタデータストア

Document のメタデータ（有用性スコア、難易度、品質スコア等）を SQLite に保存し、
FAISS のベクトルインデックスと対になるメタデータ管理を提供する。

使い方:
    from src.memory.metadata_store import MetadataStore

    store = MetadataStore()                         # デフォルト設定
    await store.initialize()                        # テーブル作成
    await store.save(doc)                           # Document 保存
    doc = await store.get("doc_id")                 # 取得
    docs = await store.list_by_domain("code", limit=10)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

from src.common.config import MetadataConfig, get_settings
from src.memory.schema import (
    DifficultyLevel,
    Document,
    Domain,
    ReviewStatus,
    SourceMeta,
    SourceType,
    UsefulnessScore,
)

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    content_hash TEXT,
    chunk_index INTEGER DEFAULT 0,
    parent_id TEXT,

    domain TEXT NOT NULL DEFAULT 'general',

    source_type TEXT DEFAULT 'manual',
    source_url TEXT,
    source_title TEXT,
    source_author TEXT,
    source_language TEXT,
    source_tags TEXT DEFAULT '[]',
    source_retrieved_at TEXT,
    source_extra TEXT DEFAULT '{}',

    retrieval_count INTEGER DEFAULT 0,
    selection_count INTEGER DEFAULT 0,
    positive_feedback INTEGER DEFAULT 0,
    negative_feedback INTEGER DEFAULT 0,
    teacher_quality REAL DEFAULT 0.0,
    execution_success_rate REAL DEFAULT 0.0,
    freshness REAL DEFAULT 1.0,
    composite_score REAL DEFAULT 0.0,

    difficulty TEXT,
    review_status TEXT DEFAULT 'unreviewed',
    confidence REAL DEFAULT 0.5,

    is_executable INTEGER DEFAULT 0,
    execution_verified INTEGER DEFAULT 0,
    last_execution_success INTEGER,

    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    reviewed_at TEXT,

    -- Teacher 素性（Step 3）
    teacher_id TEXT DEFAULT NULL,

    -- 別名・略称・通称（エイリアス）— JSON 配列 ["TinyLoRA", "13-param LoRA"]
    aliases TEXT DEFAULT '[]'
);
"""

_CREATE_INDICES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_documents_domain ON documents(domain);",
    "CREATE INDEX IF NOT EXISTS idx_documents_review_status ON documents(review_status);",
    "CREATE INDEX IF NOT EXISTS idx_documents_difficulty ON documents(difficulty);",
    "CREATE INDEX IF NOT EXISTS idx_documents_confidence ON documents(confidence);",
    "CREATE INDEX IF NOT EXISTS idx_documents_composite ON documents(composite_score);",
    "CREATE INDEX IF NOT EXISTS idx_documents_parent_id ON documents(parent_id);",
    "CREATE INDEX IF NOT EXISTS idx_documents_teacher_id ON documents(teacher_id);",
]

# 既存 DB へのマイグレーション（列が存在しない場合のみ実行）
_MIGRATION_ADD_TEACHER_ID = (
    "ALTER TABLE documents ADD COLUMN teacher_id TEXT DEFAULT NULL;"
)
_MIGRATION_ADD_ALIASES = (
    "ALTER TABLE documents ADD COLUMN aliases TEXT DEFAULT '[]';"
)


def _doc_to_row(doc: Document) -> dict[str, Any]:
    """Document を SQLite の行データに変換する。"""
    return {
        "id": doc.id,
        "content": doc.content,
        "content_hash": doc.content_hash,
        "chunk_index": doc.chunk_index,
        "parent_id": doc.parent_id,
        "domain": doc.domain.value if isinstance(doc.domain, Domain) else doc.domain,
        "source_type": doc.source.source_type.value,
        "source_url": doc.source.url,
        "source_title": doc.source.title,
        "source_author": doc.source.author,
        "source_language": doc.source.language,
        "source_tags": json.dumps(doc.source.tags),
        "source_retrieved_at": doc.source.retrieved_at.isoformat(),
        "source_extra": json.dumps(doc.source.extra),
        "retrieval_count": doc.usefulness.retrieval_count,
        "selection_count": doc.usefulness.selection_count,
        "positive_feedback": doc.usefulness.positive_feedback,
        "negative_feedback": doc.usefulness.negative_feedback,
        "teacher_quality": doc.usefulness.teacher_quality,
        "execution_success_rate": doc.usefulness.execution_success_rate,
        "freshness": doc.usefulness.freshness,
        "composite_score": doc.usefulness.composite,
        "difficulty": doc.difficulty.value if doc.difficulty else None,
        "review_status": doc.review_status.value,
        "confidence": doc.confidence,
        "is_executable": int(doc.is_executable),
        "execution_verified": int(doc.execution_verified),
        "last_execution_success": (
            int(doc.last_execution_success) if doc.last_execution_success is not None else None
        ),
        "created_at": doc.created_at.isoformat(),
        "updated_at": doc.updated_at.isoformat(),
        "reviewed_at": doc.reviewed_at.isoformat() if doc.reviewed_at else None,
        # Teacher 素性 — source.extra から取り出して専用列に保存
        "teacher_id": doc.source.teacher_id,
        # エイリアス（JSON 配列文字列）
        "aliases": json.dumps(doc.aliases),
    }


def _build_source_meta(d: dict) -> SourceMeta:
    """行データから SourceMeta を復元する。

    teacher_id 専用列の値を extra dict にも注入することで、
    SourceMeta.teacher_id プロパティが正しく機能する。
    """
    from src.memory.schema import _TEACHER_ID_KEY

    extra: dict = json.loads(d["source_extra"]) if d["source_extra"] else {}

    # 専用列の teacher_id を extra に同期（専用列が権威ソース）
    db_teacher_id: str | None = d.get("teacher_id")
    if db_teacher_id:
        extra[_TEACHER_ID_KEY] = db_teacher_id
    elif _TEACHER_ID_KEY in extra:
        # extra だけに入っている旧データも受け入れる
        pass

    return SourceMeta(
        source_type=SourceType(d["source_type"]),
        url=d["source_url"],
        title=d["source_title"],
        author=d["source_author"],
        language=d["source_language"],
        tags=json.loads(d["source_tags"]) if d["source_tags"] else [],
        retrieved_at=datetime.fromisoformat(d["source_retrieved_at"]),
        extra=extra,
    )


def _row_to_doc(row: aiosqlite.Row) -> Document:
    """SQLite の行データを Document に復元する。"""
    d = dict(row)
    return Document(
        id=d["id"],
        content=d["content"],
        content_hash=d["content_hash"],
        chunk_index=d["chunk_index"],
        parent_id=d["parent_id"],
        domain=Domain(d["domain"]),
        embedding=None,  # メタデータストアは embedding を保持しない
        source=_build_source_meta(d),
        usefulness=UsefulnessScore(
            retrieval_count=d["retrieval_count"],
            selection_count=d["selection_count"],
            positive_feedback=d["positive_feedback"],
            negative_feedback=d["negative_feedback"],
            teacher_quality=d["teacher_quality"],
            execution_success_rate=d["execution_success_rate"],
            freshness=d["freshness"],
            composite=d["composite_score"],
        ),
        difficulty=DifficultyLevel(d["difficulty"]) if d["difficulty"] else None,
        review_status=ReviewStatus(d["review_status"]),
        confidence=d["confidence"],
        is_executable=bool(d["is_executable"]),
        execution_verified=bool(d["execution_verified"]),
        last_execution_success=(
            bool(d["last_execution_success"]) if d["last_execution_success"] is not None else None
        ),
        aliases=json.loads(d["aliases"]) if d.get("aliases") else [],
        created_at=datetime.fromisoformat(d["created_at"]),
        updated_at=datetime.fromisoformat(d["updated_at"]),
        reviewed_at=(
            datetime.fromisoformat(d["reviewed_at"]) if d["reviewed_at"] else None
        ),
    )


class MetadataStore:
    """SQLite ベースの Document メタデータストア。

    全操作は非同期 (aiosqlite)。FAISS インデックスと対で使用する。

    Args:
        config: MetadataConfig。省略時は get_settings().metadata を使用。
        db_path: DB ファイルパス。config より優先。":memory:" でインメモリ。
    """

    def __init__(
        self,
        config: MetadataConfig | None = None,
        db_path: str | None = None,
    ) -> None:
        cfg = config or get_settings().metadata
        self._db_path = db_path or str(cfg.db_path)
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """DB 接続を開き、テーブルを作成する。"""
        # ファイルモードの場合は親ディレクトリを作成
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL;")
        await self._db.execute("PRAGMA foreign_keys=ON;")
        await self._db.execute(_CREATE_TABLE_SQL)
        await self._migrate()  # 列追加を先に行い、その後インデックス作成
        for idx_sql in _CREATE_INDICES_SQL:
            await self._db.execute(idx_sql)
        await self._db.commit()
        logger.info("MetadataStore initialized: %s", self._db_path)

    async def _migrate(self) -> None:
        """既存 DB スキーマへの後方互換マイグレーションを実行する。"""
        cursor = await self._db.execute("PRAGMA table_info(documents)")
        columns = {row[1] for row in await cursor.fetchall()}
        if "teacher_id" not in columns:
            await self._db.execute(_MIGRATION_ADD_TEACHER_ID)
            logger.info("MetadataStore: migrated — added teacher_id column")
        if "aliases" not in columns:
            await self._db.execute(_MIGRATION_ADD_ALIASES)
            logger.info("MetadataStore: migrated — added aliases column")

    async def close(self) -> None:
        """DB 接続を閉じる。"""
        if self._db:
            await self._db.close()
            self._db = None

    async def save(self, doc: Document) -> None:
        """Document を保存する (UPSERT)。"""
        row = _doc_to_row(doc)
        columns = ", ".join(row.keys())
        placeholders = ", ".join(f":{k}" for k in row.keys())
        updates = ", ".join(f"{k}=excluded.{k}" for k in row.keys() if k != "id")

        sql = f"""
            INSERT INTO documents ({columns}) VALUES ({placeholders})
            ON CONFLICT(id) DO UPDATE SET {updates}
        """
        await self._db.execute(sql, row)
        await self._db.commit()

    async def save_batch(self, docs: list[Document]) -> None:
        """複数 Document を一括保存する。"""
        if not docs:
            return
        for doc in docs:
            row = _doc_to_row(doc)
            columns = ", ".join(row.keys())
            placeholders = ", ".join(f":{k}" for k in row.keys())
            updates = ", ".join(f"{k}=excluded.{k}" for k in row.keys() if k != "id")
            sql = f"""
                INSERT INTO documents ({columns}) VALUES ({placeholders})
                ON CONFLICT(id) DO UPDATE SET {updates}
            """
            await self._db.execute(sql, row)
        await self._db.commit()

    async def get(self, doc_id: str) -> Document | None:
        """ID で Document を取得する。"""
        cursor = await self._db.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_doc(row)

    async def get_batch(self, doc_ids: list[str]) -> list[Document]:
        """複数 ID で Document を取得する。見つからない ID は無視。"""
        if not doc_ids:
            return []
        placeholders = ", ".join("?" for _ in doc_ids)
        cursor = await self._db.execute(
            f"SELECT * FROM documents WHERE id IN ({placeholders})", doc_ids
        )
        rows = await cursor.fetchall()
        return [_row_to_doc(row) for row in rows]

    async def delete(self, doc_id: str) -> bool:
        """Document を削除する。"""
        cursor = await self._db.execute(
            "DELETE FROM documents WHERE id = ?", (doc_id,)
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def delete_batch(self, doc_ids: list[str]) -> int:
        """複数 Document を削除する。削除件数を返す。"""
        if not doc_ids:
            return 0
        placeholders = ", ".join("?" for _ in doc_ids)
        cursor = await self._db.execute(
            f"DELETE FROM documents WHERE id IN ({placeholders})", doc_ids
        )
        await self._db.commit()
        return cursor.rowcount

    async def exists(self, doc_id: str) -> bool:
        """Document が存在するか。"""
        cursor = await self._db.execute(
            "SELECT 1 FROM documents WHERE id = ? LIMIT 1", (doc_id,)
        )
        return await cursor.fetchone() is not None

    # ── エイリアス操作 ──────────────────────────────────────────────────

    async def update_aliases(self, doc_id: str, aliases: list[str]) -> None:
        """ドキュメントの aliases 列を更新する。

        Args:
            doc_id:  対象ドキュメント ID。
            aliases: 新しいエイリアスリスト（上書き）。
        """
        await self._db.execute(
            "UPDATE documents SET aliases = ?, updated_at = ? WHERE id = ?",
            (json.dumps(aliases), datetime.utcnow().isoformat(), doc_id),
        )
        await self._db.commit()
        logger.debug("Updated aliases for doc=%s: %s", doc_id[:8], aliases)

    async def search_by_alias(
        self,
        keyword: str,
        domain: str | None = None,
        limit: int = 10,
    ) -> list[str]:
        """エイリアス列にキーワードが含まれるドキュメントの ID を返す。

        SQLite の LIKE を使ったシンプルな全文一致。
        10K 件規模では十分な速度で動作する。

        Args:
            keyword: 検索キーワード（大文字小文字を区別しない）。
            domain:  ドメインフィルタ（None = 全ドメイン）。
            limit:   返す最大件数。

        Returns:
            マッチした doc_id のリスト。
        """
        kw = keyword.lower()
        if domain:
            cursor = await self._db.execute(
                "SELECT id FROM documents WHERE lower(aliases) LIKE ? AND domain = ? LIMIT ?",
                (f"%{kw}%", domain, limit),
            )
        else:
            cursor = await self._db.execute(
                "SELECT id FROM documents WHERE lower(aliases) LIKE ? LIMIT ?",
                (f"%{kw}%", limit),
            )
        rows = await cursor.fetchall()
        return [row[0] for row in rows]

    # ── クエリ系 ────────────────────────────────────────────────────────

    async def list_by_domain(
        self,
        domain: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        """ドメイン指定で Document を取得する。"""
        cursor = await self._db.execute(
            "SELECT * FROM documents WHERE domain = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (domain, limit, offset),
        )
        return [_row_to_doc(row) for row in await cursor.fetchall()]

    async def list_by_teacher(
        self,
        teacher_id: str,
        domain: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        """特定 Teacher が生成したドキュメントを取得する。

        Args:
            teacher_id: Teacher モデル識別子。
            domain:     ドメインで絞り込む場合に指定。
            limit:      最大取得件数。
            offset:     オフセット（ページング用）。

        Returns:
            Document リスト（作成日時降順）。
        """
        if domain:
            cursor = await self._db.execute(
                "SELECT * FROM documents WHERE teacher_id = ? AND domain = ? "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (teacher_id, domain, limit, offset),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM documents WHERE teacher_id = ? "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (teacher_id, limit, offset),
            )
        return [_row_to_doc(row) for row in await cursor.fetchall()]

    async def count_by_teacher(
        self,
        teacher_id: str,
        domain: str | None = None,
    ) -> int:
        """特定 Teacher のドキュメント数を返す。"""
        if domain:
            cursor = await self._db.execute(
                "SELECT COUNT(*) FROM documents WHERE teacher_id = ? AND domain = ?",
                (teacher_id, domain),
            )
        else:
            cursor = await self._db.execute(
                "SELECT COUNT(*) FROM documents WHERE teacher_id = ?",
                (teacher_id,),
            )
        row = await cursor.fetchone()
        return row[0]

    async def exclude_teacher(
        self,
        exclude_teacher_ids: list[str],
        domain: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        """指定した Teacher のドキュメントを除外して取得する。

        信頼度の低い Teacher のデータを排除した検索に使う。

        Args:
            exclude_teacher_ids: 除外する teacher_id のリスト。
            domain:              ドメイン絞り込み（任意）。
            limit:               最大取得件数。
            offset:              オフセット。

        Returns:
            Document リスト（composite_score 降順）。
        """
        if not exclude_teacher_ids:
            return await self.list_by_domain(domain or "general", limit=limit, offset=offset)

        placeholders = ", ".join("?" for _ in exclude_teacher_ids)
        params: list[Any] = list(exclude_teacher_ids)

        if domain:
            where = f"(teacher_id IS NULL OR teacher_id NOT IN ({placeholders})) AND domain = ?"
            params.append(domain)
        else:
            where = f"(teacher_id IS NULL OR teacher_id NOT IN ({placeholders}))"

        params += [limit, offset]
        cursor = await self._db.execute(
            f"SELECT * FROM documents WHERE {where} "
            f"ORDER BY composite_score DESC LIMIT ? OFFSET ?",
            params,
        )
        return [_row_to_doc(row) for row in await cursor.fetchall()]

    async def delete_by_teacher(
        self,
        teacher_id: str,
        domain: str | None = None,
    ) -> int:
        """特定 Teacher のドキュメントを一括削除する。

        信頼度が極めて低い Teacher のデータを完全排除する場合に使用。
        FAISS 側の削除は呼び出し元 (MemoryManager) が担う。

        Returns:
            削除件数。
        """
        if domain:
            cursor = await self._db.execute(
                "DELETE FROM documents WHERE teacher_id = ? AND domain = ?",
                (teacher_id, domain),
            )
        else:
            cursor = await self._db.execute(
                "DELETE FROM documents WHERE teacher_id = ?",
                (teacher_id,),
            )
        await self._db.commit()
        deleted = cursor.rowcount
        logger.info(
            "MetadataStore.delete_by_teacher: teacher=%s domain=%s deleted=%d",
            teacher_id, domain, deleted,
        )
        return deleted

    async def get_unreviewed(
        self,
        domain: str | None = None,
        limit: int = 100,
    ) -> list[Document]:
        """未レビューの Document を取得する (maturation/reviewer 用)。"""
        if domain:
            cursor = await self._db.execute(
                "SELECT * FROM documents WHERE review_status = 'unreviewed' AND domain = ? "
                "ORDER BY created_at ASC LIMIT ?",
                (domain, limit),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM documents WHERE review_status = 'unreviewed' "
                "ORDER BY created_at ASC LIMIT ?",
                (limit,),
            )
        return [_row_to_doc(row) for row in await cursor.fetchall()]

    async def get_low_confidence(
        self,
        threshold: float = 0.3,
        limit: int = 100,
    ) -> list[Document]:
        """信頼度が閾値以下の Document を取得する (品質管理用)。"""
        cursor = await self._db.execute(
            "SELECT * FROM documents WHERE confidence <= ? ORDER BY confidence ASC LIMIT ?",
            (threshold, limit),
        )
        return [_row_to_doc(row) for row in await cursor.fetchall()]

    # ── 有用性スコア更新 ────────────────────────────────────────────────

    async def increment_retrieval(self, doc_id: str) -> None:
        """検索ヒット回数をインクリメントする。"""
        await self._db.execute(
            "UPDATE documents SET retrieval_count = retrieval_count + 1, "
            "updated_at = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), doc_id),
        )
        await self._db.commit()

    async def increment_selection(self, doc_id: str) -> None:
        """選択回数をインクリメントする。"""
        await self._db.execute(
            "UPDATE documents SET selection_count = selection_count + 1, "
            "updated_at = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), doc_id),
        )
        await self._db.commit()

    async def add_feedback(self, doc_id: str, positive: bool) -> None:
        """フィードバックを記録する。"""
        col = "positive_feedback" if positive else "negative_feedback"
        await self._db.execute(
            f"UPDATE documents SET {col} = {col} + 1, updated_at = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), doc_id),
        )
        await self._db.commit()

    async def update_quality(
        self,
        doc_id: str,
        teacher_quality: float | None = None,
        difficulty: str | None = None,
        review_status: str | None = None,
        confidence: float | None = None,
        composite_score: float | None = None,
    ) -> None:
        """品質関連フィールドを更新する (maturation 用)。"""
        updates: list[str] = []
        params: list[Any] = []

        if teacher_quality is not None:
            updates.append("teacher_quality = ?")
            params.append(teacher_quality)
        if difficulty is not None:
            updates.append("difficulty = ?")
            params.append(difficulty)
        if review_status is not None:
            updates.append("review_status = ?")
            params.append(review_status)
            if review_status in ("approved", "rejected", "needs_update"):
                updates.append("reviewed_at = ?")
                params.append(datetime.utcnow().isoformat())
        if confidence is not None:
            updates.append("confidence = ?")
            params.append(confidence)
        if composite_score is not None:
            updates.append("composite_score = ?")
            params.append(composite_score)

        if not updates:
            return

        updates.append("updated_at = ?")
        params.append(datetime.utcnow().isoformat())
        params.append(doc_id)

        sql = f"UPDATE documents SET {', '.join(updates)} WHERE id = ?"
        await self._db.execute(sql, params)
        await self._db.commit()

    # ── 統計 ────────────────────────────────────────────────────────────

    async def count(self, domain: str | None = None) -> int:
        """Document 数を返す。"""
        if domain:
            cursor = await self._db.execute(
                "SELECT COUNT(*) FROM documents WHERE domain = ?", (domain,)
            )
        else:
            cursor = await self._db.execute("SELECT COUNT(*) FROM documents")
        row = await cursor.fetchone()
        return row[0]

    async def avg_confidence(self, domain: str | None = None) -> float:
        """平均信頼度を返す。"""
        if domain:
            cursor = await self._db.execute(
                "SELECT AVG(confidence) FROM documents WHERE domain = ?", (domain,)
            )
        else:
            cursor = await self._db.execute("SELECT AVG(confidence) FROM documents")
        row = await cursor.fetchone()
        return row[0] or 0.0

    async def get_stats(self, domain: str | None = None) -> dict:
        """品質メトリクス用の集計統計を返す。

        QualityMetrics.compute() から呼び出される。

        Args:
            domain: ドメイン絞り込み（None で全ドメイン）。

        Returns:
            total_docs / approved_docs / rejected_docs / pending_docs /
            avg_confidence / avg_teacher_quality / avg_composite_score /
            exec_success_rate / avg_retrieval_count / avg_selection_count /
            difficulty_distribution を含む dict。
        """
        where = "WHERE domain = ?" if domain else ""
        params: tuple = (domain,) if domain else ()

        # 全体集計（1クエリ）
        agg_sql = f"""
            SELECT
                COUNT(*)                          AS total_docs,
                SUM(review_status = 'approved')   AS approved_docs,
                SUM(review_status = 'rejected')   AS rejected_docs,
                SUM(review_status = 'unreviewed') AS pending_docs,
                AVG(confidence)                   AS avg_confidence,
                AVG(teacher_quality)              AS avg_teacher_quality,
                AVG(composite_score)              AS avg_composite_score,
                AVG(execution_success_rate)       AS exec_success_rate,
                AVG(retrieval_count)              AS avg_retrieval_count,
                AVG(selection_count)              AS avg_selection_count
            FROM documents {where}
        """
        cursor = await self._db.execute(agg_sql, params)
        row = dict(await cursor.fetchone())

        # 難易度分布
        diff_sql = f"""
            SELECT difficulty, COUNT(*) AS cnt
            FROM documents {where}
            GROUP BY difficulty
        """
        cursor = await self._db.execute(diff_sql, params)
        difficulty_distribution = {
            (r["difficulty"] or "unknown"): r["cnt"]
            for r in await cursor.fetchall()
        }

        return {
            "total_docs":            row["total_docs"] or 0,
            "approved_docs":         row["approved_docs"] or 0,
            "rejected_docs":         row["rejected_docs"] or 0,
            "pending_docs":          row["pending_docs"] or 0,
            "avg_confidence":        row["avg_confidence"] or 0.0,
            "avg_teacher_quality":   row["avg_teacher_quality"] or 0.0,
            "avg_composite_score":   row["avg_composite_score"] or 0.0,
            "exec_success_rate":     row["exec_success_rate"] or 0.0,
            "avg_retrieval_count":   row["avg_retrieval_count"] or 0.0,
            "avg_selection_count":   row["avg_selection_count"] or 0.0,
            "difficulty_distribution": difficulty_distribution,
        }
