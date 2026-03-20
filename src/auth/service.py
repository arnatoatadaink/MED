"""src/auth/service.py — 認証サービス（登録・ログイン・JWT）。

使用ライブラリ:
  - python-jose[cryptography] : JWT 生成・検証
  - passlib[bcrypt]           : パスワードハッシュ
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta

from jose import JWTError, jwt
from passlib.context import CryptContext

from src.auth.schema import TokenPayload, TokenResponse, User
from src.auth.store import UserStore

logger = logging.getLogger(__name__)

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """ユーザー登録・ログイン・JWT 発行・検証を担う。

    Args:
        store: UserStore インスタンス。
        secret_key: JWT 署名キー（環境変数 JWT_SECRET_KEY を推奨）。
        algorithm: JWT アルゴリズム（デフォルト HS256）。
        expire_days: アクセストークン有効期間（日数）。
        allow_test_token: テストユーザートークン発行を許可するか。
    """

    def __init__(
        self,
        store: UserStore,
        secret_key: str,
        algorithm: str = "HS256",
        expire_days: int = 7,
        allow_test_token: bool = True,
    ) -> None:
        self._store = store
        self._secret = secret_key
        self._algorithm = algorithm
        self._expire_days = expire_days
        self._allow_test = allow_test_token

    # ── パスワード ────────────────────────────────────────────

    def hash_password(self, plain: str) -> str:
        return _pwd_ctx.hash(plain)

    def verify_password(self, plain: str, hashed: str) -> bool:
        return _pwd_ctx.verify(plain, hashed)

    # ── JWT ──────────────────────────────────────────────────

    def create_token(self, user: User) -> str:
        expire = datetime.utcnow() + timedelta(days=self._expire_days)
        payload = {
            "sub": user.user_id,
            "username": user.username,
            "is_admin": user.is_admin,
            "is_test": user.is_test,
            "exp": expire,
        }
        return jwt.encode(payload, self._secret, algorithm=self._algorithm)

    def decode_token(self, token: str) -> TokenPayload:
        """JWT を検証してペイロードを返す。失敗時は ValueError を送出。"""
        try:
            data = jwt.decode(token, self._secret, algorithms=[self._algorithm])
            return TokenPayload(
                sub=data["sub"],
                username=data["username"],
                is_admin=data.get("is_admin", False),
                is_test=data.get("is_test", False),
            )
        except JWTError as e:
            raise ValueError(f"Invalid token: {e}") from e

    # ── ユーザー操作 ──────────────────────────────────────────

    async def register(
        self,
        username: str,
        password: str,
        is_admin: bool = False,
    ) -> User:
        """通常ユーザーを登録する。ユーザー名が既存の場合は ValueError。"""
        if await self._store.exists_username(username):
            raise ValueError(f"Username '{username}' already exists")
        user = User(
            user_id=str(uuid.uuid4()),
            username=username,
            hashed_password=self.hash_password(password),
            is_admin=is_admin,
        )
        await self._store.save(user)
        logger.info("Registered user: %s (admin=%s)", username, is_admin)
        return user

    async def register_test_user(self, username: str) -> User:
        """テストユーザー（パスワードなし）を登録する。"""
        if await self._store.exists_username(username):
            # 既存のテストユーザーをそのまま返す
            existing = await self._store.get_by_username(username)
            if existing and existing.is_test:
                return existing
            raise ValueError(f"Username '{username}' already exists as non-test user")
        user = User(
            user_id=str(uuid.uuid4()),
            username=username,
            hashed_password=None,
            is_test=True,
        )
        await self._store.save(user)
        logger.info("Registered test user: %s", username)
        return user

    async def login(self, username: str, password: str) -> TokenResponse:
        """ユーザー名・パスワードを検証してトークンを返す。失敗時は ValueError。"""
        user = await self._store.get_by_username(username)
        if not user or not user.is_active:
            raise ValueError("Invalid username or password")
        if user.is_test:
            raise ValueError("Test users must use the test token endpoint")
        if not user.hashed_password or not self.verify_password(password, user.hashed_password):
            raise ValueError("Invalid username or password")

        await self._store.update_last_login(user.user_id, datetime.utcnow())
        token = self.create_token(user)
        return TokenResponse(
            access_token=token,
            user_id=user.user_id,
            username=user.username,
            is_admin=user.is_admin,
        )

    async def issue_test_token(self, username: str) -> TokenResponse:
        """テストユーザーにパスワードなしでトークンを発行する。"""
        if not self._allow_test:
            raise PermissionError("Test token issuance is disabled")
        user = await self._store.get_by_username(username)
        if not user:
            raise ValueError(f"Test user '{username}' not found")
        if not user.is_test:
            raise ValueError(f"User '{username}' is not a test user")
        if not user.is_active:
            raise ValueError(f"User '{username}' is inactive")

        await self._store.update_last_login(user.user_id, datetime.utcnow())
        token = self.create_token(user)
        return TokenResponse(
            access_token=token,
            user_id=user.user_id,
            username=user.username,
            is_admin=user.is_admin,
        )

    async def get_user_by_token(self, token: str) -> User:
        """JWT を検証してユーザーを返す。失敗時は ValueError。"""
        payload = self.decode_token(token)
        user = await self._store.get_by_id(payload.sub)
        if not user or not user.is_active:
            raise ValueError("User not found or inactive")
        return user
