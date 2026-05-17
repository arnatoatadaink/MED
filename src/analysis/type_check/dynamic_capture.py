"""pytest plugin: capture TypeCheckError events to runtime/dynamic_check_events.jsonl.

Registration: add ``pytest_plugins = ["src.analysis.type_check.dynamic_capture"]``
to ``tests/conftest.py``.  Only active when typeguard is installed; safe to
import without --typeguard-packages (no errors emitted when typeguard is absent).

On-demand workflow:
  pytest tests/unit/ --typeguard-packages=src   # writes JSONL
  python -m src.analysis.type_check.dynamic_checker   # reads JSONL, emits report
"""

from __future__ import annotations

import json
import re
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

_OUTPUT = Path("runtime/dynamic_check_events.jsonl")

# Matches: argument "param" (ActualType) is not an instance of ExpectedType
_ARG_PATTERN = re.compile(
    r'^argument "(?P<param>[^"]+)" \((?P<actual>[^)]+)\) is not an instance of (?P<expected>.+)$'
)
# Matches: return value (ActualType) is not an instance of ExpectedType
_RET_PATTERN = re.compile(
    r"^return value \((?P<actual>[^)]+)\) is not an instance of (?P<expected>.+)$"
)

_TYPEGUARD_MARKER = "typeguard"
_VENV_MARKER = ".venv"


def parse_typeguard_message(message: str) -> dict[str, str]:
    """Extract structured fields from a TypeCheckError message string.

    Returns a dict with keys: kind, param_name (if argument), actual_type, expected_type.
    Falls back to ``kind='unknown'`` when the format is unrecognized.
    """
    m = _ARG_PATTERN.match(message)
    if m:
        return {
            "kind": "argument",
            "param_name": m.group("param"),
            "actual_type": m.group("actual"),
            "expected_type": m.group("expected"),
        }
    m = _RET_PATTERN.match(message)
    if m:
        return {
            "kind": "return",
            "param_name": "",
            "actual_type": m.group("actual"),
            "expected_type": m.group("expected"),
        }
    return {"kind": "unknown", "param_name": "", "actual_type": "", "expected_type": ""}


def extract_user_frames(tb: types.TracebackType | None) -> list[dict[str, str | int]]:
    """Walk a traceback and return frames not inside typeguard or .venv.

    Each entry: {file, line, func}.  Ordered from outermost (call site) inward.
    """
    frames: list[dict[str, str | int]] = []
    while tb is not None:
        fname = tb.tb_frame.f_code.co_filename
        if _TYPEGUARD_MARKER not in fname and _VENV_MARKER not in fname:
            frames.append(
                {
                    "file": fname,
                    "line": tb.tb_lineno,
                    "func": tb.tb_frame.f_code.co_name,
                }
            )
        tb = tb.tb_next
    return frames


def _capture(excinfo: pytest.ExceptionInfo) -> None:  # type: ignore[type-arg]
    """Serialize one TypeCheckError event and append it to the JSONL output file."""
    message = str(excinfo.value)
    parsed = parse_typeguard_message(message)
    frames = extract_user_frames(excinfo.tb)

    call_site = frames[0] if frames else {}
    func_frame = frames[1] if len(frames) > 1 else {}

    event: dict[str, str | int] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": message,
        "call_site_file": str(call_site.get("file", "")),
        "call_site_line": int(call_site.get("line", 0)),
        "call_site_func": str(call_site.get("func", "")),
        "func_file": str(func_frame.get("file", "")),
        "func_line": int(func_frame.get("line", 0)),
        "func_name": str(func_frame.get("func", "")),
        **parsed,
    }

    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with _OUTPUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> None:  # type: ignore[type-arg]
    """Intercept test failures caused by TypeCheckError and capture them."""
    yield
    if call.when != "call" or call.excinfo is None:
        return
    try:
        from typeguard import TypeCheckError
    except ImportError:
        return
    if issubclass(call.excinfo.type, TypeCheckError):
        _capture(call.excinfo)
