#!/usr/bin/env python3
"""scripts/launch_gui.py — MED Gradio Web GUI 起動スクリプト。

使い方:
    python scripts/launch_gui.py
    python scripts/launch_gui.py --port 7861
    python scripts/launch_gui.py --share
    python scripts/launch_gui.py --host 127.0.0.1 --port 7860 --debug
"""

from __future__ import annotations

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from src.gui.app import _parse_args, launch

if __name__ == "__main__":
    args = _parse_args()
    print(f"[MED GUI] Starting Gradio at http://{args.host}:{args.port}")
    launch(host=args.host, port=args.port, share=args.share, debug=args.debug)
