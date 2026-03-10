"""src/common/logger.py — 構造化ロギング設定

プロジェクト全体で統一されたロギングを提供する。
JSON 形式 or テキスト形式を環境変数で切り替え可能。

使い方:
    from src.common.logger import get_logger, setup_logging

    setup_logging(level="INFO", json_format=False)
    logger = get_logger(__name__)
    logger.info("Processing query", extra={"query": "hello"})
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

# デフォルトフォーマット
_TEXT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    stream=None,
) -> None:
    """ルートロガーを設定する。

    Args:
        level: ログレベル文字列 ("DEBUG", "INFO", "WARNING", "ERROR")。
        json_format: True のとき JSON Lines 形式で出力する。
        stream: 出力ストリーム（省略時は sys.stderr）。
    """
    global _configured

    numeric = getattr(logging, level.upper(), logging.INFO)
    stream = stream or sys.stderr

    handler: logging.Handler
    if json_format:
        handler = _JsonStreamHandler(stream)
    else:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(numeric)

    # 既存ハンドラを置き換え（二重登録防止）
    root.handlers.clear()
    root.addHandler(handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """名前付きロガーを取得する。

    未設定の場合は自動的に INFO / テキスト形式で設定する。
    """
    if not _configured:
        setup_logging()
    return logging.getLogger(name)


class _JsonStreamHandler(logging.StreamHandler):
    """JSON Lines 形式のストリームハンドラ。"""

    def format(self, record: logging.LogRecord) -> str:
        import json
        import traceback

        _fmt = logging.Formatter(datefmt=_DATE_FORMAT)
        data: dict = {
            "timestamp": _fmt.formatTime(record, _DATE_FORMAT),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            data["exception"] = "".join(traceback.format_exception(*record.exc_info))

        # extra フィールドを取り込む
        skip = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "taskName",
        }
        for key, val in record.__dict__.items():
            if key not in skip:
                data[key] = val

        return json.dumps(data, ensure_ascii=False, default=str)
