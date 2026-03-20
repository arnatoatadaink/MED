"""src/auth/schema.py — 認証モジュールのデータモデル。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class User:
    """ユーザーエンティティ。"""

    user_id: str                    # UUID
    username: str
    hashed_password: str | None     # None = テストユーザー
    is_test: bool = False
    is_admin: bool = False
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: datetime | None = None


@dataclass
class TokenPayload:
    """JWT ペイロード。"""

    sub: str        # user_id
    username: str
    is_admin: bool = False
    is_test: bool = False


@dataclass
class LoginRequest:
    username: str
    password: str


@dataclass
class RegisterRequest:
    username: str
    password: str
    is_admin: bool = False


@dataclass
class TestTokenRequest:
    """テストユーザー専用トークン発行リクエスト（パスワード不要）。"""
    username: str


@dataclass
class TokenResponse:
    access_token: str
    token_type: str = "bearer"
    user_id: str = ""
    username: str = ""
    is_admin: bool = False
