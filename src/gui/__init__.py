"""src/gui — Gradio Web GUI パッケージ。

エントリーポイント:
    python -m src.gui.app
    # または
    python scripts/launch_gui.py
"""

from src.gui.app import build_app, launch

__all__ = ["build_app", "launch"]
