"""Type mismatch detection from call graph edges."""

from dataclasses import dataclass

from .call_graph import CallEdge

# Key = actual (inferred) type; value = expected annotations it satisfies.
# bool is a subtype of int; int widens to float.
_COMPATIBLE: dict[str, frozenset[str]] = {
    "bool": frozenset({"bool", "int"}),
    "int": frozenset({"int", "float"}),
    "float": frozenset({"float"}),
    "str": frozenset({"str"}),
    "bytes": frozenset({"bytes"}),
    "None": frozenset({"None"}),
    "list": frozenset({"list"}),
    "dict": frozenset({"dict"}),
    "tuple": frozenset({"tuple"}),
    "set": frozenset({"set"}),
}

# Only these bare annotations trigger mismatch checks.
# Custom class annotations (e.g. FAISSIndex, MetadataStore) are skipped.
_PRIMITIVE_ANNOTATIONS: frozenset[str] = frozenset(_COMPATIBLE.keys())


@dataclass
class TypeMismatch:
    """Describes a detected type mismatch at a call site."""

    function_name: str
    param_name: str
    expected: str
    actual: str
    call_site_file: str
    call_site_line: int


def _is_compatible(expected: str, actual: str) -> bool:
    """Return True if actual is an acceptable type for the expected annotation.

    Only bare simple annotations are checked (str, int, float, …).
    Generic annotations (list[X], Optional[X], Union[X,Y], X|Y) are skipped
    to avoid false positives.
    """
    bare = expected.strip()
    if "[" in bare or "|" in bare:
        return True
    if bare not in _PRIMITIVE_ANNOTATIONS:
        return True  # custom / unknown annotation; skip to avoid false positives
    acceptable = _COMPATIBLE.get(actual, frozenset({actual}))
    return bare in acceptable


def detect_mismatches(edges: list[CallEdge]) -> list[TypeMismatch]:
    """Detect argument type mismatches at each resolved call site.

    A mismatch is reported only when both the parameter annotation and the
    inferred argument type are known and incompatible.
    """
    mismatches: list[TypeMismatch] = []
    for edge in edges:
        func = edge.callee
        site = edge.call_site
        param_names = list(func.params.keys())
        for i, actual_type in enumerate(site.arg_type_hints):
            if i >= len(param_names):
                break
            if actual_type is None:
                continue
            param_name = param_names[i]
            expected = func.params[param_name]
            if expected is None:
                continue
            if not _is_compatible(expected, actual_type):
                mismatches.append(
                    TypeMismatch(
                        function_name=func.qualified_name,
                        param_name=param_name,
                        expected=expected,
                        actual=actual_type,
                        call_site_file=site.file,
                        call_site_line=site.line,
                    )
                )
    return mismatches
