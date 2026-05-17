"""AST-based extractor for function definitions and call sites."""

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FunctionInfo:
    """Extracted metadata for a function or method definition."""

    name: str
    qualified_name: str
    file: str
    line: int
    params: dict[str, str | None]
    return_annotation: str | None


@dataclass
class CallSiteInfo:
    """Extracted metadata for a function call site."""

    caller_func: str
    callee_name: str
    arg_type_hints: list[str | None]
    file: str
    line: int


def _annotation_to_str(node: ast.expr | None) -> str | None:
    """Convert an AST annotation node to its string representation."""
    if node is None:
        return None
    return ast.unparse(node)


def _infer_type_from_node(node: ast.expr) -> str | None:
    """Infer a primitive type label from an AST expression node.

    Returns None for non-literal expressions (Name, Call, etc.).
    """
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool):
            return "bool"
        if isinstance(v, int):
            return "int"
        if isinstance(v, float):
            return "float"
        if isinstance(v, str):
            return "str"
        if isinstance(v, bytes):
            return "bytes"
        if v is None:
            return "None"
    elif isinstance(node, ast.List):
        return "list"
    elif isinstance(node, ast.Dict):
        return "dict"
    elif isinstance(node, ast.Tuple):
        return "tuple"
    elif isinstance(node, ast.Set):
        return "set"
    return None


class _Extractor(ast.NodeVisitor):
    """AST visitor that collects FunctionInfo and CallSiteInfo."""

    def __init__(self, file_path: str) -> None:
        self._file = file_path
        self._scope_stack: list[str] = []
        self.functions: list[FunctionInfo] = []
        self.call_sites: list[CallSiteInfo] = []

    def _current_scope(self) -> str:
        return ".".join(self._scope_stack) if self._scope_stack else "<module>"

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track class scope for qualified method names."""
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Extract FunctionInfo and recurse into the function body."""
        qualified = f"{self._scope_stack[-1]}.{node.name}" if self._scope_stack else node.name
        params: dict[str, str | None] = {}
        all_args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        for arg in all_args:
            if arg.arg in ("self", "cls"):
                continue
            params[arg.arg] = _annotation_to_str(arg.annotation)

        self.functions.append(
            FunctionInfo(
                name=node.name,
                qualified_name=qualified,
                file=self._file,
                line=node.lineno,
                params=params,
                return_annotation=_annotation_to_str(node.returns),
            )
        )
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Call(self, node: ast.Call) -> None:
        """Record a call site with inferred argument types."""
        callee_name: str | None = None
        if isinstance(node.func, ast.Name):
            callee_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            callee_name = node.func.attr

        if callee_name is not None:
            self.call_sites.append(
                CallSiteInfo(
                    caller_func=self._current_scope(),
                    callee_name=callee_name,
                    arg_type_hints=[_infer_type_from_node(a) for a in node.args],
                    file=self._file,
                    line=node.lineno,
                )
            )
        self.generic_visit(node)


def extract_from_source(
    source: str, file_path: str
) -> tuple[list[FunctionInfo], list[CallSiteInfo]]:
    """Parse Python source text and return function definitions and call sites."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], []
    visitor = _Extractor(file_path)
    visitor.visit(tree)
    return visitor.functions, visitor.call_sites


def extract_from_directory(
    directory: Path,
) -> tuple[list[FunctionInfo], list[CallSiteInfo]]:
    """Walk all .py files under directory and aggregate extraction results."""
    all_functions: list[FunctionInfo] = []
    all_call_sites: list[CallSiteInfo] = []
    for py_file in sorted(directory.rglob("*.py")):
        source = py_file.read_text(encoding="utf-8")
        funcs, calls = extract_from_source(source, str(py_file))
        all_functions.extend(funcs)
        all_call_sites.extend(calls)
    return all_functions, all_call_sites
