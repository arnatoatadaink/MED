"""tests/unit/test_auth.py — 認証モジュールの単体テスト。"""

from __future__ import annotations

import pytest

from src.auth.service import AuthService
from src.auth.store import UserStore

_SECRET = "test-secret-key"


@pytest.fixture
async def store():
    s = UserStore(db_path=":memory:")
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
async def svc(store):
    return AuthService(
        store=store,
        secret_key=_SECRET,
        algorithm="HS256",
        expire_days=7,
        allow_test_token=True,
    )


# ── ユーザー登録 ──────────────────────────────────────────────────


class TestRegister:
    @pytest.mark.asyncio
    async def test_register_normal_user(self, svc):
        user = await svc.register("alice", "password123")
        assert user.username == "alice"
        assert user.hashed_password is not None
        assert not user.is_test
        assert user.is_active

    @pytest.mark.asyncio
    async def test_register_duplicate_raises(self, svc):
        await svc.register("alice", "password123")
        with pytest.raises(ValueError, match="already exists"):
            await svc.register("alice", "other")

    @pytest.mark.asyncio
    async def test_register_test_user(self, svc):
        user = await svc.register_test_user("test_alice")
        assert user.is_test
        assert user.hashed_password is None

    @pytest.mark.asyncio
    async def test_register_test_user_idempotent(self, svc):
        """テストユーザーを重複登録しても既存ユーザーを返す。"""
        u1 = await svc.register_test_user("test_alice")
        u2 = await svc.register_test_user("test_alice")
        assert u1.user_id == u2.user_id

    @pytest.mark.asyncio
    async def test_register_admin(self, svc):
        user = await svc.register("admin", "adminpass", is_admin=True)
        assert user.is_admin


# ── ログイン / JWT ──────────────────────────────────────────────


class TestLogin:
    @pytest.mark.asyncio
    async def test_login_success(self, svc):
        await svc.register("alice", "password123")
        resp = await svc.login("alice", "password123")
        assert resp.access_token
        assert resp.username == "alice"

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, svc):
        await svc.register("alice", "password123")
        with pytest.raises(ValueError):
            await svc.login("alice", "wrongpass")

    @pytest.mark.asyncio
    async def test_login_nonexistent_user(self, svc):
        with pytest.raises(ValueError):
            await svc.login("nobody", "pass")

    @pytest.mark.asyncio
    async def test_test_user_cannot_use_login(self, svc):
        """テストユーザーは通常ログインを使えない。"""
        await svc.register_test_user("test_alice")
        with pytest.raises(ValueError, match="test token endpoint"):
            await svc.login("test_alice", "any")


# ── テストトークン ────────────────────────────────────────────────


class TestTestToken:
    @pytest.mark.asyncio
    async def test_issue_test_token(self, svc):
        await svc.register_test_user("test_alice")
        resp = await svc.issue_test_token("test_alice")
        assert resp.access_token
        assert resp.username == "test_alice"

    @pytest.mark.asyncio
    async def test_test_token_disabled(self, store):
        svc = AuthService(store=store, secret_key=_SECRET, allow_test_token=False)
        await svc.register_test_user("test_alice")
        with pytest.raises(PermissionError):
            await svc.issue_test_token("test_alice")

    @pytest.mark.asyncio
    async def test_normal_user_cannot_get_test_token(self, svc):
        await svc.register("alice", "password123")
        with pytest.raises(ValueError, match="not a test user"):
            await svc.issue_test_token("alice")


# ── JWT 検証 ──────────────────────────────────────────────────────


class TestJWT:
    @pytest.mark.asyncio
    async def test_decode_valid_token(self, svc):
        user = await svc.register("alice", "password123")
        token = svc.create_token(user)
        payload = svc.decode_token(token)
        assert payload.sub == user.user_id
        assert payload.username == "alice"

    def test_decode_invalid_token(self, svc):
        with pytest.raises(ValueError, match="Invalid token"):
            svc.decode_token("not.a.token")

    @pytest.mark.asyncio
    async def test_get_user_by_token(self, svc):
        user = await svc.register("alice", "password123")
        token = svc.create_token(user)
        fetched = await svc.get_user_by_token(token)
        assert fetched.user_id == user.user_id


# ── UserStore CRUD ────────────────────────────────────────────────


class TestUserStore:
    @pytest.mark.asyncio
    async def test_save_and_get(self, store):
        svc = AuthService(store=store, secret_key=_SECRET)
        user = await svc.register("alice", "password123")
        fetched = await store.get_by_id(user.user_id)
        assert fetched is not None
        assert fetched.username == "alice"

    @pytest.mark.asyncio
    async def test_get_by_username(self, store):
        svc = AuthService(store=store, secret_key=_SECRET)
        user = await svc.register("alice", "password123")
        fetched = await store.get_by_username("alice")
        assert fetched is not None
        assert fetched.user_id == user.user_id

    @pytest.mark.asyncio
    async def test_delete_user(self, store):
        svc = AuthService(store=store, secret_key=_SECRET)
        user = await svc.register("alice", "password123")
        deleted = await store.delete(user.user_id)
        assert deleted
        assert await store.get_by_id(user.user_id) is None

    @pytest.mark.asyncio
    async def test_set_inactive(self, store):
        svc = AuthService(store=store, secret_key=_SECRET)
        user = await svc.register("alice", "password123")
        await store.set_active(user.user_id, False)
        fetched = await store.get_by_id(user.user_id)
        assert fetched is not None
        assert not fetched.is_active

    @pytest.mark.asyncio
    async def test_list_all(self, store):
        svc = AuthService(store=store, secret_key=_SECRET)
        await svc.register("alice", "pw1")
        await svc.register("bob", "pw2")
        users = await store.list_all()
        usernames = {u.username for u in users}
        assert {"alice", "bob"} == usernames
