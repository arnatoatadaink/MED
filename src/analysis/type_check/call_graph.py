"""Call graph construction from extracted AST data."""

from dataclasses import dataclass

from .ast_extractor import CallSiteInfo, FunctionInfo


@dataclass
class CallEdge:
    """A resolved edge in the call graph: caller site → callee definition."""

    caller: str
    callee: FunctionInfo
    call_site: CallSiteInfo


def build_reverse_call_graph(edges: list[CallEdge]) -> dict[str, list[CallEdge]]:
    """Return a mapping from callee qualified_name to all CallEdges that target it."""
    reverse: dict[str, list[CallEdge]] = {}
    for edge in edges:
        reverse.setdefault(edge.callee.qualified_name, []).append(edge)
    return reverse


def build_call_graph(
    functions: list[FunctionInfo],
    call_sites: list[CallSiteInfo],
) -> list[CallEdge]:
    """Resolve call sites to FunctionInfo entries and return call graph edges.

    Resolution uses simple name matching (callee_name == func.name).
    When multiple functions share a name, the first definition is used.
    Unresolved call sites (e.g., builtins, external calls) are skipped.
    """
    name_index: dict[str, FunctionInfo] = {}
    for func in functions:
        name_index.setdefault(func.name, func)

    edges: list[CallEdge] = []
    for site in call_sites:
        target = name_index.get(site.callee_name)
        if target is None:
            continue
        edges.append(CallEdge(caller=site.caller_func, callee=target, call_site=site))
    return edges
