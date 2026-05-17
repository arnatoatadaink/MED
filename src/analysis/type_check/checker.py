"""Main orchestrator: AST + Call Graph Analysis type mismatch checker."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from .ast_extractor import CallSiteInfo, FunctionInfo, extract_from_directory
from .call_graph import CallEdge, build_call_graph
from .type_mismatch import TypeMismatch, detect_mismatches


@dataclass
class CheckResult:
    """Aggregated output of a type mismatch check over a directory."""

    functions: list[FunctionInfo] = field(default_factory=list)
    call_sites: list[CallSiteInfo] = field(default_factory=list)
    edges: list[CallEdge] = field(default_factory=list)
    mismatches: list[TypeMismatch] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Return True when at least one mismatch was found."""
        return bool(self.mismatches)

    @property
    def summary(self) -> str:
        """Human-readable summary of all detected mismatches."""
        if not self.mismatches:
            return "No type mismatches found."
        lines = [f"{len(self.mismatches)} type mismatch(es) detected:"]
        for m in self.mismatches:
            lines.append(
                f"  {m.call_site_file}:{m.call_site_line}"
                f" → {m.function_name}({m.param_name}: {m.expected})"
                f" called with {m.actual}"
            )
        return "\n".join(lines)


def check_directory(path: Path) -> CheckResult:
    """Run AST + CGA type mismatch detection over all Python files in path."""
    functions, call_sites = extract_from_directory(path)
    edges = build_call_graph(functions, call_sites)
    mismatches = detect_mismatches(edges)
    return CheckResult(
        functions=functions,
        call_sites=call_sites,
        edges=edges,
        mismatches=mismatches,
    )


def _main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("src")
    if not target.exists():
        print(f"Error: {target} does not exist.", file=sys.stderr)
        sys.exit(1)
    result = check_directory(target)
    print(
        f"Scanned {len(result.functions)} functions, "
        f"{len(result.call_sites)} call sites, "
        f"{len(result.edges)} resolved edges."
    )
    print(result.summary)
    sys.exit(1 if result.has_errors else 0)


if __name__ == "__main__":
    _main()
