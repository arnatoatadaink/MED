"""Offline dynamic type-mismatch analyzer.

Reads ``runtime/dynamic_check_events.jsonl`` written by the pytest capture
plugin, cross-references with AST + call-graph data, and emits a report.

Usage:
  # Step 1 — generate events (instruments src/ at runtime):
  pytest tests/unit/ --typeguard-packages=src

  # Step 2 — analyze and report:
  python -m src.analysis.type_check.dynamic_checker [src_dir] [events_file]
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .ast_extractor import FunctionInfo, extract_from_directory
from .call_graph import CallEdge, build_call_graph, build_reverse_call_graph
from .type_mismatch import TypeMismatch, detect_mismatches

_DEFAULT_EVENTS = Path("runtime/dynamic_check_events.jsonl")
_DEFAULT_REPORT_JSON = Path("runtime/dynamic_check_report.json")
_DEFAULT_REPORT_MD = Path("runtime/dynamic_check_report.md")


@dataclass
class DynamicCheckEvent:
    """One TypeCheckError captured during a typeguard-instrumented test run."""

    timestamp: str
    message: str
    kind: str
    param_name: str
    actual_type: str
    expected_type: str
    call_site_file: str
    call_site_line: int
    call_site_func: str
    func_file: str
    func_line: int
    func_name: str


@dataclass
class EnrichedEvent:
    """A dynamic event annotated with AST and call-graph context."""

    event: DynamicCheckEvent
    func_info: FunctionInfo | None
    callers: list[CallEdge] = field(default_factory=list)
    static_match: TypeMismatch | None = None


def load_events(events_file: Path = _DEFAULT_EVENTS) -> list[DynamicCheckEvent]:
    """Read and deserialize all events from the JSONL capture file."""
    if not events_file.exists():
        return []
    events: list[DynamicCheckEvent] = []
    for line in events_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        events.append(
            DynamicCheckEvent(
                timestamp=data.get("timestamp", ""),
                message=data.get("message", ""),
                kind=data.get("kind", "unknown"),
                param_name=data.get("param_name", ""),
                actual_type=data.get("actual_type", ""),
                expected_type=data.get("expected_type", ""),
                call_site_file=data.get("call_site_file", ""),
                call_site_line=int(data.get("call_site_line", 0)),
                call_site_func=data.get("call_site_func", ""),
                func_file=data.get("func_file", ""),
                func_line=int(data.get("func_line", 0)),
                func_name=data.get("func_name", ""),
            )
        )
    return events


def enrich_events(
    events: list[DynamicCheckEvent],
    functions: list[FunctionInfo],
    edges: list[CallEdge],
    static_mismatches: list[TypeMismatch],
) -> list[EnrichedEvent]:
    """Annotate each event with AST + reverse-CGA + static mismatch data."""
    func_by_name: dict[str, FunctionInfo] = {f.name: f for f in functions}
    reverse_cga = build_reverse_call_graph(edges)

    static_index: dict[tuple[str, str], TypeMismatch] = {
        (m.function_name, m.param_name): m for m in static_mismatches
    }

    enriched: list[EnrichedEvent] = []
    for ev in events:
        func_info = func_by_name.get(ev.func_name)
        qname = func_info.qualified_name if func_info else ev.func_name
        callers = reverse_cga.get(qname, [])
        static_match = static_index.get((qname, ev.param_name))
        enriched.append(
            EnrichedEvent(event=ev, func_info=func_info, callers=callers, static_match=static_match)
        )
    return enriched


def generate_report_json(
    enriched: list[EnrichedEvent], output: Path = _DEFAULT_REPORT_JSON
) -> None:
    """Write enriched events to a JSON report file."""
    records = []
    for e in enriched:
        ev = e.event
        records.append(
            {
                "timestamp": ev.timestamp,
                "message": ev.message,
                "kind": ev.kind,
                "param_name": ev.param_name,
                "actual_type": ev.actual_type,
                "expected_type": ev.expected_type,
                "call_site": f"{ev.call_site_file}:{ev.call_site_line} in {ev.call_site_func}",
                "func": {
                    "name": ev.func_name,
                    "file": ev.func_file,
                    "line": ev.func_line,
                    "qualified_name": e.func_info.qualified_name if e.func_info else ev.func_name,
                },
                "callers": [
                    {
                        "caller_func": c.caller,
                        "file": c.call_site.file,
                        "line": c.call_site.line,
                    }
                    for c in e.callers
                ],
                "static_mismatch_also_detected": e.static_match is not None,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_report_md(enriched: list[EnrichedEvent], output: Path = _DEFAULT_REPORT_MD) -> None:
    """Write enriched events to a Markdown report file."""
    lines = ["# Dynamic Type Mismatch Report\n"]
    if not enriched:
        lines.append("_No TypeCheckErrors captured._\n")
    for i, e in enumerate(enriched, 1):
        ev = e.event
        qname = e.func_info.qualified_name if e.func_info else ev.func_name
        static_tag = " *(also detected statically)*" if e.static_match else ""
        lines.append(f"## {i}. `{qname}` — {ev.kind}{static_tag}\n")
        lines.append(f"- **Message**: `{ev.message}`\n")
        cs = f"{ev.call_site_file}:{ev.call_site_line}"
        lines.append(f"- **Call site**: `{cs}` in `{ev.call_site_func}`\n")
        lines.append(f"- **Function**: `{ev.func_file}:{ev.func_line}`\n")
        if e.callers:
            lines.append("- **Known callers (CGA)**:\n")
            for c in e.callers:
                lines.append(f"  - `{c.caller}` at `{c.call_site.file}:{c.call_site.line}`\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(lines), encoding="utf-8")


def run_check(
    src_dir: Path | None = None,
    events_file: Path = _DEFAULT_EVENTS,
) -> list[EnrichedEvent]:
    """Full pipeline: load events, build AST/CGA, enrich, write reports.

    Returns the list of enriched events (empty if no events file found).
    """
    root = src_dir or Path("src")
    events = load_events(events_file)
    if not events:
        return []

    functions, call_sites = extract_from_directory(root)
    edges = build_call_graph(functions, call_sites)
    static_mismatches = detect_mismatches(edges)
    enriched = enrich_events(events, functions, edges, static_mismatches)

    generate_report_json(enriched)
    generate_report_md(enriched)
    return enriched


if __name__ == "__main__":
    src_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    events_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else _DEFAULT_EVENTS
    result = run_check(src_dir=src_arg, events_file=events_arg)
    print(f"Enriched {len(result)} TypeCheckError event(s).")
    print(f"Reports: {_DEFAULT_REPORT_JSON} / {_DEFAULT_REPORT_MD}")
