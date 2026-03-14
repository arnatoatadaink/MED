"""src/memory/teacher_registry.py — Teacher プロファイルレジストリ

どの Teacher モデルがどれだけ信頼できるかを SQLite で管理する。
ドキュメント投入時に teacher_id を自動登録し、フィードバックを受け取るたびに
EWMA（指数加重移動平均）で trust_score を更新する。

テーブル: teacher_profiles
  teacher_id   TEXT PRIMARY KEY  例: "claude-opus-4-6"
  provider     TEXT              例: "anthropic"
  trust_score  REAL              0.0〜1.0（高いほど検索で優遇）
  total_docs   INTEGER           このTeacher由来のドキュメント数
  avg_reward   REAL              フィードバック報酬の EWMA
  n_feedback   INTEGER           受け取ったフィードバック件数
  created_at   TEXT
  updated_at   TEXT

信頼度更新アルゴリズム:
  フィードバックが少ない段階（n < WARMUP）は Welford 法（真の平均）で更新し、
  WARMUP 以降は固定 α の EWMA に切り替える。これにより、初期は素直に平均を取り、
  実績が積み上がった後は最近のフィードバックを重視する。

使い方:
    from src.memory.teacher_registry import TeacherRegistry

    registry = TeacherRegistry("data/metadata.db")
    await registry.initialize()

    await registry.ensure("claude-opus-4-6", provider="anthropic")
    await registry.record_doc("claude-opus-4-6")
    await registry.record_feedback("claude-opus-4-6", reward=0.9)

    profile = await registry.get("claude-opus-4-6")
    print(profile.trust_score)   # 0.9...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

# EWMA パラメータ
_EWMA_ALPHA = 0.05       # WARMUP 後の固定学習率
_WARMUP_N = 10           # これ未満は Welford 法（真の平均）
_TRUST_DECAY = 0.9       # trust_score をリセットするときの減衰率
_MIN_TRUST = 0.05        # trust_score の下限

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS teacher_profiles (
    teacher_id   TEXT PRIMARY KEY,
    provider     TEXT,
    trust_score  REAL    NOT NULL DEFAULT 1.0,
    total_docs   INTEGER NOT NULL DEFAULT 0,
    avg_reward   REAL    NOT NULL DEFAULT 0.5,
    n_feedback   INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT    NOT NULL,
    updated_at   TEXT    NOT NULL
);
"""

_CREATE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_teacher_trust "
    "ON teacher_profiles(trust_score);"
)


@dataclass
class TeacherProfile:
    """Teacher モデルの信頼プロファイル。"""

    teacher_id: str
    provider: str | None
    trust_score: float
    total_docs: int
    avg_reward: float
    n_feedback: int
    created_at: datetime
    updated_at: datetime

    @property
    def is_trusted(self) -> bool:
        """trust_score が 0.5 以上か。"""
        return self.trust_score >= 0.5

    @property
    def is_low_trust(self) -> bool:
        """trust_score が 0.3 未満か（要注意ライン）。"""
        return self.trust_score < 0.3

    def summary(self) -> str:
        return (
            f"Teacher({self.teacher_id!r} provider={self.provider} "
            f"trust={self.trust_score:.3f} docs={self.total_docs} "
            f"avg_reward={self.avg_reward:.3f} n_fb={self.n_feedback})"
        )


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _ewma_update(old_avg: float, reward: float, n: int) -> float:
    """EWMA または Welford 法で平均を更新する。

    Args:
        old_avg: 現在の avg_reward。
        reward:  新しい報酬値 (0.0〜1.0)。
        n:       更新後の n_feedback。

    Returns:
        更新後の平均値。
    """
    if n <= _WARMUP_N:
        # Welford: 真の移動平均
        return old_avg + (reward - old_avg) / n
    else:
        # 固定 alpha の EWMA
        return old_avg * (1.0 - _EWMA_ALPHA) + reward * _EWMA_ALPHA


class TeacherRegistry:
    """Teacher プロファイルを SQLite で管理するレジストリ。

    MetadataStore と同じ SQLite ファイルを共有するか、
    別ファイルを指定することができる。

    Args:
        db_path: SQLite ファイルパス。MetadataStore と同じパスを推奨。
    """

    def __init__(self, db_path: str | Path = "data/metadata.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # ライフサイクル
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """テーブルとインデックスを作成する。"""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(_CREATE_TABLE_SQL)
            await db.execute(_CREATE_INDEX_SQL)
            await db.commit()
        logger.info("TeacherRegistry initialized at %s", self._db_path)

    # ------------------------------------------------------------------
    # 登録 / 取得
    # ------------------------------------------------------------------

    async def ensure(
        self,
        teacher_id: str,
        provider: str | None = None,
    ) -> TeacherProfile:
        """teacher_id を登録する。既存ならそのまま返す（upsert）。

        Args:
            teacher_id: Teacher モデル識別子。
            provider:   プロバイダ名。未指定なら推定を試みる。

        Returns:
            TeacherProfile。
        """
        from src.memory.schema import _infer_provider
        resolved = provider or _infer_provider(teacher_id)
        now = _now_iso()

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute(
                """
                INSERT INTO teacher_profiles
                    (teacher_id, provider, trust_score, total_docs,
                     avg_reward, n_feedback, created_at, updated_at)
                VALUES (?, ?, 1.0, 0, 0.5, 0, ?, ?)
                ON CONFLICT(teacher_id) DO NOTHING
                """,
                (teacher_id, resolved, now, now),
            )
            await db.commit()

        profile = await self.get(teacher_id)
        assert profile is not None
        logger.debug("TeacherRegistry.ensure: %s", profile.summary())
        return profile

    async def get(self, teacher_id: str) -> TeacherProfile | None:
        """teacher_id のプロファイルを返す。未登録なら None。"""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM teacher_profiles WHERE teacher_id = ?",
                (teacher_id,),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_profile(dict(row))

    async def list_all(self) -> list[TeacherProfile]:
        """全プロファイルを trust_score 降順で返す。"""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM teacher_profiles ORDER BY trust_score DESC"
            ) as cursor:
                rows = await cursor.fetchall()
        return [_row_to_profile(dict(r)) for r in rows]

    async def get_low_trust(self, threshold: float = 0.3) -> list[TeacherProfile]:
        """trust_score が threshold 未満のプロファイルを返す。"""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM teacher_profiles WHERE trust_score < ? ORDER BY trust_score ASC",
                (threshold,),
            ) as cursor:
                rows = await cursor.fetchall()
        return [_row_to_profile(dict(r)) for r in rows]

    # ------------------------------------------------------------------
    # ドキュメントカウント
    # ------------------------------------------------------------------

    async def record_doc(self, teacher_id: str) -> None:
        """このTeacher由来のドキュメントが1件追加されたことを記録する。

        teacher_id が未登録の場合は自動的に ensure() を呼ぶ。
        """
        await self.ensure(teacher_id)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                UPDATE teacher_profiles
                SET total_docs = total_docs + 1,
                    updated_at = ?
                WHERE teacher_id = ?
                """,
                (_now_iso(), teacher_id),
            )
            await db.commit()

    # ------------------------------------------------------------------
    # 信頼度更新
    # ------------------------------------------------------------------

    async def record_feedback(
        self,
        teacher_id: str,
        reward: float,
    ) -> TeacherProfile:
        """フィードバック報酬を受け取り、avg_reward と trust_score を更新する。

        trust_score は avg_reward と同値にする（単純化）。
        下限 _MIN_TRUST でクランプする。

        Args:
            teacher_id: Teacher モデル識別子。
            reward:     報酬値 (0.0〜1.0)。

        Returns:
            更新後の TeacherProfile。
        """
        reward = max(0.0, min(1.0, reward))
        profile = await self.ensure(teacher_id)

        new_n = profile.n_feedback + 1
        new_avg = _ewma_update(profile.avg_reward, reward, new_n)
        new_trust = max(_MIN_TRUST, new_avg)

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                UPDATE teacher_profiles
                SET avg_reward  = ?,
                    trust_score = ?,
                    n_feedback  = ?,
                    updated_at  = ?
                WHERE teacher_id = ?
                """,
                (new_avg, new_trust, new_n, _now_iso(), teacher_id),
            )
            await db.commit()

        updated = await self.get(teacher_id)
        assert updated is not None
        logger.debug(
            "TeacherRegistry.record_feedback: %s reward=%.3f → trust=%.3f",
            teacher_id, reward, updated.trust_score,
        )
        return updated

    async def set_trust(
        self,
        teacher_id: str,
        trust_score: float,
    ) -> TeacherProfile:
        """trust_score を直接上書きする（管理者操作用）。

        Args:
            teacher_id:  Teacher モデル識別子。
            trust_score: 新しい信頼スコア (0.0〜1.0)。

        Returns:
            更新後の TeacherProfile。
        """
        trust_score = max(0.0, min(1.0, trust_score))
        await self.ensure(teacher_id)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                UPDATE teacher_profiles
                SET trust_score = ?,
                    updated_at  = ?
                WHERE teacher_id = ?
                """,
                (trust_score, _now_iso(), teacher_id),
            )
            await db.commit()
        updated = await self.get(teacher_id)
        assert updated is not None
        logger.info(
            "TeacherRegistry.set_trust: %s → %.3f", teacher_id, trust_score,
        )
        return updated

    async def reset_trust(self, teacher_id: str) -> TeacherProfile:
        """trust_score を初期値 1.0 にリセットする。"""
        return await self.set_trust(teacher_id, 1.0)


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _row_to_profile(d: dict) -> TeacherProfile:
    return TeacherProfile(
        teacher_id=d["teacher_id"],
        provider=d.get("provider"),
        trust_score=float(d["trust_score"]),
        total_docs=int(d["total_docs"]),
        avg_reward=float(d["avg_reward"]),
        n_feedback=int(d["n_feedback"]),
        created_at=datetime.fromisoformat(d["created_at"]),
        updated_at=datetime.fromisoformat(d["updated_at"]),
    )
