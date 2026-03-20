"""tests/unit/test_conversation.py — 会話履歴モジュールの単体テスト。"""

from __future__ import annotations

import pytest

from src.conversation.manager import ConversationManager, _auto_title, _estimate_tokens
from src.conversation.schema import Session, Turn
from src.conversation.store import ConversationStore


@pytest.fixture
async def store():
    s = ConversationStore(db_path=":memory:")
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
async def manager(store):
    mgr = ConversationManager(
        store=store,
        max_sessions_per_user=5,
        context_window_tokens=200,
        auto_save_to_faiss=False,  # FAISS 依存を排除
    )
    # store は既に initialize 済み
    return mgr


# ── ユーティリティ ────────────────────────────────────────────────


class TestUtils:
    def test_auto_title_short(self):
        assert _auto_title("こんにちは") == "こんにちは"

    def test_auto_title_long(self):
        title = _auto_title("A" * 50)
        assert len(title) <= 21  # 20 + "…"
        assert title.endswith("…")

    def test_auto_title_empty(self):
        assert _auto_title("") == "新しい会話"

    def test_estimate_tokens(self):
        assert _estimate_tokens("") == 1
        assert _estimate_tokens("a" * 100) == 25


# ── セッション操作 ────────────────────────────────────────────────


class TestSessionCRUD:
    @pytest.mark.asyncio
    async def test_create_session(self, manager):
        session = await manager.create_session("user1", "Python のリストをソートするには？")
        assert session.session_id
        assert session.user_id == "user1"
        assert "Python" in session.title or len(session.title) <= 21

    @pytest.mark.asyncio
    async def test_get_session(self, manager):
        session = await manager.create_session("user1", "テスト")
        fetched = await manager.get_session(session.session_id)
        assert fetched is not None
        assert fetched.session_id == session.session_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, manager):
        fetched = await manager.get_session("no-such-id")
        assert fetched is None

    @pytest.mark.asyncio
    async def test_list_sessions(self, manager):
        for i in range(3):
            await manager.create_session("user1", f"クエリ {i}")
        sessions = await manager.list_sessions("user1")
        assert len(sessions) == 3

    @pytest.mark.asyncio
    async def test_list_sessions_user_isolation(self, manager):
        await manager.create_session("user1", "クエリ A")
        await manager.create_session("user2", "クエリ B")
        assert len(await manager.list_sessions("user1")) == 1
        assert len(await manager.list_sessions("user2")) == 1

    @pytest.mark.asyncio
    async def test_delete_session(self, manager):
        session = await manager.create_session("user1", "削除テスト")
        deleted = await manager.delete_session(session.session_id)
        assert deleted
        assert await manager.get_session(session.session_id) is None

    @pytest.mark.asyncio
    async def test_auto_delete_old_sessions(self, manager):
        """max_sessions_per_user=5 を超えたら古いセッションが削除される。"""
        for i in range(6):  # 6 個作成 → 1 個削除されるはず
            await manager.create_session("user1", f"クエリ {i}")
        sessions = await manager.list_sessions("user1", limit=100)
        assert len(sessions) <= 5


# ── ターン操作 ────────────────────────────────────────────────────


class TestTurnCRUD:
    @pytest.mark.asyncio
    async def test_add_and_get_turns(self, manager):
        session = await manager.create_session("user1", "テスト")
        user_turn = manager.make_user_turn(session.session_id, "こんにちは")
        asst_turn = manager.make_assistant_turn(session.session_id, "こんにちは！")
        await manager.add_turn(user_turn)
        await manager.add_turn(asst_turn)

        turns = await manager.get_all_turns(session.session_id)
        assert len(turns) == 2
        assert turns[0].role == "user"
        assert turns[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_turn_count_updated(self, manager, store):
        session = await manager.create_session("user1", "テスト")
        turn = manager.make_user_turn(session.session_id, "hello")
        await manager.add_turn(turn)

        fetched = await store.get_session(session.session_id)
        assert fetched is not None
        assert fetched.turn_count == 1

    @pytest.mark.asyncio
    async def test_token_count_estimated(self, manager):
        session = await manager.create_session("user1", "テスト")
        turn = manager.make_user_turn(session.session_id, "A" * 40)
        assert turn.token_count == 10  # 40 // 4

    @pytest.mark.asyncio
    async def test_get_context_turns_within_tokens(self, manager):
        """トークン上限（200）内に収まるターン数だけ返す。"""
        session = await manager.create_session("user1", "テスト")
        # 各ターン ~50 トークン（200文字）× 6 個 → 300 token 分追加
        for i in range(6):
            t = manager.make_user_turn(session.session_id, "A" * 200)
            await manager.add_turn(t)

        context = await manager.get_context_turns(session.session_id)
        # 上限 200 トークン以内に収まること
        total = sum(t.token_count for t in context)
        assert total <= 200

    @pytest.mark.asyncio
    async def test_context_turns_in_chronological_order(self, manager):
        """コンテキストターンは古い順に並ぶ。"""
        session = await manager.create_session("user1", "テスト")
        for i in range(3):
            t = manager.make_user_turn(session.session_id, f"メッセージ {i}")
            await manager.add_turn(t)

        context = await manager.get_context_turns(session.session_id)
        contents = [t.content for t in context]
        assert contents == ["メッセージ 0", "メッセージ 1", "メッセージ 2"]


# ── LLM メッセージ変換 ────────────────────────────────────────────


class TestToMessages:
    @pytest.mark.asyncio
    async def test_to_messages_format(self, manager):
        session = await manager.create_session("user1", "テスト")
        u = manager.make_user_turn(session.session_id, "質問")
        a = manager.make_assistant_turn(session.session_id, "回答")
        await manager.add_turn(u)
        await manager.add_turn(a)

        turns = await manager.get_all_turns(session.session_id)
        messages = manager.to_messages(turns)
        assert messages == [
            {"role": "user", "content": "質問"},
            {"role": "assistant", "content": "回答"},
        ]


# ── ConversationStore 直接テスト ──────────────────────────────────


class TestConversationStore:
    @pytest.mark.asyncio
    async def test_get_recent_turns_within_tokens_order(self, store):
        """get_recent_turns_within_tokens は時系列順で返す。"""
        import uuid
        from datetime import datetime

        from src.conversation.schema import Session, Turn

        sid = str(uuid.uuid4())
        now = datetime.utcnow()
        session = Session(
            session_id=sid, user_id="u1", title="t",
            domain="general", created_at=now, updated_at=now,
        )
        await store.save_session(session)

        for i in range(3):
            t = Turn(
                turn_id=str(uuid.uuid4()),
                session_id=sid,
                role="user",
                content=f"msg{i}",
                timestamp=datetime.utcnow(),
                token_count=10,
            )
            await store.save_turn(t)

        turns = await store.get_recent_turns_within_tokens(sid, max_tokens=1000)
        contents = [t.content for t in turns]
        assert contents == ["msg0", "msg1", "msg2"]

    @pytest.mark.asyncio
    async def test_cascade_delete(self, store):
        """セッション削除でターンも CASCADE 削除される。"""
        import uuid
        from datetime import datetime

        from src.conversation.schema import Session, Turn

        sid = str(uuid.uuid4())
        now = datetime.utcnow()
        session = Session(
            session_id=sid, user_id="u1", title="t",
            domain="general", created_at=now, updated_at=now,
        )
        await store.save_session(session)

        t = Turn(
            turn_id=str(uuid.uuid4()),
            session_id=sid,
            role="user",
            content="hello",
            timestamp=now,
            token_count=5,
        )
        await store.save_turn(t)

        await store.delete_session(sid)
        turns = await store.get_turns(sid)
        assert len(turns) == 0
