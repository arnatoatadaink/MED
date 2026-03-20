"""src/auth/deps.py — FastAPI 依存注入（認証・ローカル制限）。"""

from __future__ import annotations

import logging

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.auth.schema import User
from src.auth.service import AuthService

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)

# アプリ起動後に server.py から注入される
_auth_service: AuthService | None = None


def set_auth_service(svc: AuthService) -> None:
    """サーバー起動時に AuthService を登録する。"""
    global _auth_service
    _auth_service = svc


def get_auth_service() -> AuthService:
    if _auth_service is None:
        raise RuntimeError("AuthService not initialized")
    return _auth_service


# ── localhost 制限 ────────────────────────────────────────────


def require_localhost(request: Request) -> None:
    """ローカルホスト以外からのアクセスを 403 で拒否する。

    テストトークンエンドポイント等、開発専用エンドポイントに使用する。
    """
    if request.client is None:
        raise HTTPException(status_code=403, detail="Cannot determine client address")
    allowed = {"127.0.0.1", "::1", "localhost"}
    if request.client.host not in allowed:
        raise HTTPException(
            status_code=403,
            detail="This endpoint is only accessible from localhost",
        )


# ── JWT 認証依存 ──────────────────────────────────────────────


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    svc: AuthService = Depends(get_auth_service),
) -> User:
    """Bearer トークンを検証してユーザーを返す。未認証は 401。"""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        return await svc.get_user_by_token(credentials.credentials)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e


async def get_current_admin(
    user: User = Depends(get_current_user),
) -> User:
    """管理者ユーザーのみ許可する。"""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin privilege required")
    return user


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    svc: AuthService = Depends(get_auth_service),
) -> User | None:
    """トークンがあれば検証して返す。なければ None（認証なしも許容するエンドポイント用）。"""
    if credentials is None:
        return None
    try:
        return await svc.get_user_by_token(credentials.credentials)
    except ValueError:
        return None
