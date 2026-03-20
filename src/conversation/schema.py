"""src/conversation/schema.py — 会話履歴のデータモデル。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass
class Turn:
    """会話の1往復（ユーザー発話 or アシスタント返答）。"""

    turn_id: str
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime
    token_count: int = 0            # コンテキストウィンドウ計算用（文字数 // 4 推定）
    provider: str = ""
    model: str = ""
    faiss_doc_id: str | None = None # FAISS 登録済みの場合の doc_id
    input_tokens: int = 0
    output_tokens: int = 0

    def to_message(self) -> dict[str, str]:
        """LLM messages 形式に変換する。"""
        return {"role": self.role, "content": self.content}


@dataclass
class Session:
    """会話スレッド（セッション）。"""

    session_id: str
    user_id: str
    title: str                      # 自動生成タイトル（最初のクエリ先頭20字 + "…"）
    domain: str
    created_at: datetime
    updated_at: datetime
    turn_count: int = 0

    def display_title(self) -> str:
        """ドロップダウン表示用: "タイトル (日付)" 形式。"""
        date_str = self.updated_at.strftime("%m/%d %H:%M")
        return f"{self.title} ({date_str})"
