#!/usr/bin/env python3
"""scripts/seed_test_users.py — テストユーザーの初期登録スクリプト。

テスト・開発用のパスワードなしユーザーを users.db に登録する。
本番環境では実行しないこと（`allow_test_token: false` で制御）。

使い方:
    python scripts/seed_test_users.py
    python scripts/seed_test_users.py --db data/users.db
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.auth.service import AuthService
from src.auth.store import UserStore
from src.common.config import get_settings

# ── 登録するテストユーザー ────────────────────────────────────
TEST_USERS = [
    {"username": "test_alice",  "description": "一般テストユーザー A"},
    {"username": "test_bob",    "description": "一般テストユーザー B"},
    {"username": "test_system", "description": "自動テスト・シードデータ生成用"},
]


async def main(db_path: str | None = None) -> None:
    cfg = get_settings()
    path = db_path or str(cfg.auth.users_db_path)

    print(f"[seed_test_users] DB: {path}")

    store = UserStore(db_path=path)
    await store.initialize()

    svc = AuthService(
        store=store,
        secret_key=cfg.auth.jwt_secret_key,
        algorithm=cfg.auth.jwt_algorithm,
    )

    results = []
    for u in TEST_USERS:
        try:
            user = await svc.register_test_user(u["username"])
            results.append(("created", user.username, user.user_id))
        except ValueError as e:
            results.append(("skipped", u["username"], str(e)))

    await store.close()

    print("\n結果:")
    for status, name, detail in results:
        mark = "✓" if status == "created" else "—"
        print(f"  {mark} {name}: {detail}")

    print("\nテストトークン取得例 (localhost のみ):")
    for u in TEST_USERS:
        uname = u["username"]
        print(
            f"  curl -X POST http://localhost:8000/auth/token/test"
            f" -H 'Content-Type: application/json'"
            f" -d '{{\"username\": \"{uname}\"}}'"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="テストユーザー登録")
    parser.add_argument("--db", default=None, help="users.db のパス（省略時は設定ファイル値）")
    args = parser.parse_args()
    asyncio.run(main(db_path=args.db))
