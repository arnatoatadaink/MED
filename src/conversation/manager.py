"""src/conversation/manager.py — 会話履歴管理（ビジネスロジック層）。

責務:
  - セッションのライフサイクル管理（作成・取得・一覧・削除）
  - ターンの追加とトークン数推定
  - LLM コンテキスト用メッセージリスト生成（トークン上限内）
  - assistant ターンのユーザー FAISS への自動登録
  - セッション上限超過時の古いセッション自動削除
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Callable

from src.conversation.schema import Session, Turn
from src.conversation.store import ConversationStore

if TYPE_CHECKING:
    from src.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

_AUTO_TITLE_MAX_CHARS = 20


def _estimate_tokens(text: str) -> int:
    """簡易トークン数推定: 文字数 // 4（英語基準の近似）。最小 1 を保証。"""
    return max(1, len(text) // 4)


def _auto_title(query: str) -> str:
    """クエリ先頭から自動タイトルを生成する。"""
    title = query.strip().replace("\n", " ")
    if len(title) > _AUTO_TITLE_MAX_CHARS:
        title = title[:_AUTO_TITLE_MAX_CHARS] + "…"
    return title or "新しい会話"


class ConversationManager:
    """会話履歴の管理を担う中心クラス。

    Args:
        store: ConversationStore（依存注入）。
        max_sessions_per_user: ユーザー毎のセッション上限数。超過時に古いものを自動削除。
        context_window_tokens: LLM コンテキストに渡すトークン数上限。
        auto_save_to_faiss: True の場合 assistant ターンをユーザー FAISS に登録する。
    """

    def __init__(
        self,
        store: ConversationStore,
        *,
        max_sessions_per_user: int = 50,
        context_window_tokens: int = 2048,
        auto_save_to_faiss: bool = True,
    ) -> None:
        self._store = store
        self._max_sessions = max_sessions_per_user
        self._context_tokens = context_window_tokens
        self._auto_faiss = auto_save_to_faiss

    async def initialize(self) -> None:
        await self._store.initialize()

    async def close(self) -> None:
        await self._store.close()

    # ── セッション ────────────────────────────────────────────

    async def create_session(
        self,
        user_id: str,
        first_query: str,
        domain: str = "general",
    ) -> Session:
        """新規セッションを作成する。first_query からタイトルを自動生成。"""
        now = datetime.utcnow()
        session = Session(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            title=_auto_title(first_query),
            domain=domain,
            created_at=now,
            updated_at=now,
        )
        await self._store.save_session(session)

        # セッション上限チェック（古いものを削除）
        count = await self._store.count_sessions(user_id)
        if count > self._max_sessions:
            deleted = await self._store.delete_oldest_sessions(
                user_id, keep=self._max_sessions
            )
            if deleted:
                logger.info(
                    "Auto-deleted %d old session(s) for user=%s", deleted, user_id
                )
        return session

    async def get_session(self, session_id: str) -> Session | None:
        return await self._store.get_session(session_id)

    async def list_sessions(
        self, user_id: str, limit: int = 30
    ) -> list[Session]:
        return await self._store.list_sessions(user_id, limit=limit)

    async def delete_session(self, session_id: str) -> bool:
        return await self._store.delete_session(session_id)

    # ── ターン ────────────────────────────────────────────────

    async def add_turn(self, turn: Turn) -> None:
        """ターンを保存してセッションの updated_at を更新する。"""
        if turn.token_count == 0:
            turn.token_count = _estimate_tokens(turn.content)
        await self._store.save_turn(turn)
        await self._store.touch_session(turn.session_id)

    async def get_context_turns(
        self,
        session_id: str,
        max_tokens: int | None = None,
    ) -> list[Turn]:
        """LLM コンテキスト用ターンをトークン上限内で返す（時系列順）。"""
        limit = max_tokens if max_tokens is not None else self._context_tokens
        return await self._store.get_recent_turns_within_tokens(
            session_id, max_tokens=limit
        )

    async def get_all_turns(self, session_id: str) -> list[Turn]:
        """セッション全ターンを時系列順で返す（GUI 復元用）。"""
        return await self._store.get_turns(session_id)

    # ── LLM メッセージ変換 ────────────────────────────────────

    def to_messages(self, turns: list[Turn]) -> list[dict[str, str]]:
        """Turn リストを LLM messages 形式（list[dict]）に変換する。"""
        return [t.to_message() for t in turns]

    # ── FAISS 連携 ────────────────────────────────────────────

    async def save_to_user_faiss(
        self,
        turn: Turn,
        user_id: str,
        get_user_mm: Callable[[str], "MemoryManager"],
    ) -> str | None:
        """assistant ターンをユーザー専用 FAISS に登録する。

        Args:
            turn: assistant ターン。
            user_id: ユーザー識別子（FAISS パスの決定に使用）。
            get_user_mm: user_id → MemoryManager のファクトリ関数。

        Returns:
            登録された faiss_doc_id。失敗時は None。
        """
        if not self._auto_faiss or turn.role != "assistant":
            return None
        try:
            mm = get_user_mm(user_id)
            doc_id = await mm.add_from_text(
                content=turn.content,
                domain="general",
                source_type="manual",
            )
            # ターンに doc_id を紐付け
            await self._store.update_turn_faiss_doc_id(turn.turn_id, doc_id)
            logger.debug(
                "Saved assistant turn to user FAISS: user=%s doc_id=%s", user_id, doc_id
            )
            return doc_id
        except Exception as e:
            logger.warning("Failed to save turn to user FAISS: %s", e)
            return None

    # ── ユーティリティ ────────────────────────────────────────

    def make_user_turn(
        self,
        session_id: str,
        content: str,
    ) -> Turn:
        now = datetime.utcnow()
        return Turn(
            turn_id=str(uuid.uuid4()),
            session_id=session_id,
            role="user",
            content=content,
            timestamp=now,
            token_count=_estimate_tokens(content),
        )

    def make_assistant_turn(
        self,
        session_id: str,
        content: str,
        provider: str = "",
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> Turn:
        now = datetime.utcnow()
        return Turn(
            turn_id=str(uuid.uuid4()),
            session_id=session_id,
            role="assistant",
            content=content,
            timestamp=now,
            token_count=_estimate_tokens(content),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
