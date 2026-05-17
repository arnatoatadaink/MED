"""Tests for src/analysis/type_check — AST extractor, call graph, type mismatch detector."""

from pathlib import Path

from src.analysis.type_check.ast_extractor import (
    CallSiteInfo,
    FunctionInfo,
    extract_from_source,
)
from src.analysis.type_check.call_graph import CallEdge, build_call_graph
from src.analysis.type_check.checker import CheckResult, check_directory
from src.analysis.type_check.type_mismatch import detect_mismatches


def _parse(src: str) -> tuple[list[FunctionInfo], list[CallSiteInfo]]:
    return extract_from_source(src, "<test>")


# ---------------------------------------------------------------------------
# Phase 1: AST Extractor
# ---------------------------------------------------------------------------


class TestExtractFromSource:
    def test_extract_annotated_function_params_and_return(self) -> None:
        src = "def greet(name: str, count: int) -> str:\n    pass\n"
        funcs, _ = _parse(src)
        assert len(funcs) == 1
        f = funcs[0]
        assert f.name == "greet"
        assert f.qualified_name == "greet"
        assert f.params == {"name": "str", "count": "int"}
        assert f.return_annotation == "str"

    def test_extract_unannotated_params_returns_none(self) -> None:
        src = "def func(a, b):\n    pass\n"
        funcs, _ = _parse(src)
        assert funcs[0].params == {"a": None, "b": None}

    def test_extract_method_skips_self(self) -> None:
        src = "class Foo:\n    def bar(self, x: int) -> None:\n        pass\n"
        funcs, _ = _parse(src)
        method = next(f for f in funcs if f.name == "bar")
        assert "self" not in method.params
        assert method.params == {"x": "int"}

    def test_extract_method_qualified_name_includes_class(self) -> None:
        src = "class Foo:\n    def bar(self) -> None:\n        pass\n"
        funcs, _ = _parse(src)
        method = next(f for f in funcs if f.name == "bar")
        assert method.qualified_name == "Foo.bar"

    def test_extract_call_site_with_literal_args(self) -> None:
        src = "def f(x: int) -> None:\n    pass\nf(42)\n"
        _, calls = _parse(src)
        site = next(c for c in calls if c.callee_name == "f")
        assert site.arg_type_hints == ["int"]

    def test_extract_call_site_string_literal(self) -> None:
        src = "greet('hello')\n"
        _, calls = _parse(src)
        assert calls[0].arg_type_hints == ["str"]

    def test_extract_call_site_list_literal(self) -> None:
        src = "func([1, 2, 3])\n"
        _, calls = _parse(src)
        assert calls[0].arg_type_hints == ["list"]

    def test_extract_call_site_dict_literal(self) -> None:
        src = "func({'a': 1})\n"
        _, calls = _parse(src)
        assert calls[0].arg_type_hints == ["dict"]

    def test_extract_call_site_name_arg_returns_none(self) -> None:
        src = "x = 1\nfunc(x)\n"
        _, calls = _parse(src)
        assert calls[0].arg_type_hints == [None]

    def test_extract_call_site_call_arg_returns_none(self) -> None:
        src = "func(other())\n"
        _, calls = _parse(src)
        site = next(c for c in calls if c.callee_name == "func")
        assert site.arg_type_hints == [None]

    def test_syntax_error_returns_empty(self) -> None:
        funcs, calls = extract_from_source("def (broken:", "<test>")
        assert funcs == []
        assert calls == []

    def test_async_function_extracted(self) -> None:
        src = "async def fetch(url: str) -> bytes:\n    ...\n"
        funcs, _ = _parse(src)
        assert len(funcs) == 1
        assert funcs[0].name == "fetch"
        assert funcs[0].params == {"url": "str"}

    def test_bool_literal_inferred_as_bool(self) -> None:
        src = "func(True)\n"
        _, calls = _parse(src)
        assert calls[0].arg_type_hints == ["bool"]

    def test_none_literal_inferred_as_none(self) -> None:
        src = "func(None)\n"
        _, calls = _parse(src)
        assert calls[0].arg_type_hints == ["None"]

    def test_float_literal_inferred_as_float(self) -> None:
        src = "func(3.14)\n"
        _, calls = _parse(src)
        assert calls[0].arg_type_hints == ["float"]


# ---------------------------------------------------------------------------
# Phase 2: Call Graph
# ---------------------------------------------------------------------------


class TestBuildCallGraph:
    def test_resolves_matching_call_site(self) -> None:
        src = "def process(value: int) -> None:\n    pass\nprocess(1)\n"
        funcs, calls = _parse(src)
        edges = build_call_graph(funcs, calls)
        resolved = [e for e in edges if e.callee.name == "process"]
        assert len(resolved) >= 1

    def test_skips_unresolved_builtins(self) -> None:
        src = "print('hello')\nlen([1, 2])\n"
        funcs, calls = _parse(src)
        edges = build_call_graph(funcs, calls)
        assert edges == []

    def test_edge_contains_call_site_info(self) -> None:
        src = "def add(a: int, b: int) -> int:\n    return a + b\nadd(1, 2)\n"
        funcs, calls = _parse(src)
        edges = build_call_graph(funcs, calls)
        edge = next(e for e in edges if e.callee.name == "add")
        assert edge.call_site.arg_type_hints == ["int", "int"]

    def test_attribute_call_resolved_by_method_name(self) -> None:
        src = (
            "class Foo:\n"
            "    def run(self, n: int) -> None:\n"
            "        pass\n"
            "obj = Foo()\n"
            "obj.run(5)\n"
        )
        funcs, calls = _parse(src)
        edges = build_call_graph(funcs, calls)
        resolved = [e for e in edges if e.callee.name == "run"]
        assert len(resolved) >= 1


# ---------------------------------------------------------------------------
# Phase 3: Type Mismatch Detection
# ---------------------------------------------------------------------------


class TestDetectMismatches:
    def _edges_for(self, src: str) -> list[CallEdge]:
        funcs, calls = _parse(src)
        return build_call_graph(funcs, calls)

    def test_detects_str_param_called_with_int(self) -> None:
        src = "def greet(name: str) -> None:\n    pass\ngreet(42)\n"
        edges = self._edges_for(src)
        mismatches = detect_mismatches(edges)
        assert len(mismatches) == 1
        m = mismatches[0]
        assert m.param_name == "name"
        assert m.expected == "str"
        assert m.actual == "int"

    def test_detects_int_param_called_with_str(self) -> None:
        src = "def count(n: int) -> None:\n    pass\ncount('five')\n"
        mismatches = detect_mismatches(self._edges_for(src))
        assert any(m.expected == "int" and m.actual == "str" for m in mismatches)

    def test_no_mismatch_for_matching_types(self) -> None:
        src = "def greet(name: str) -> None:\n    pass\ngreet('hello')\n"
        mismatches = detect_mismatches(self._edges_for(src))
        assert mismatches == []

    def test_no_mismatch_for_unknown_actual_type(self) -> None:
        src = "def greet(name: str) -> None:\n    pass\ngreet(some_var)\n"
        mismatches = detect_mismatches(self._edges_for(src))
        assert mismatches == []

    def test_no_mismatch_for_unannotated_param(self) -> None:
        src = "def func(x):\n    pass\nfunc(42)\n"
        mismatches = detect_mismatches(self._edges_for(src))
        assert mismatches == []

    def test_no_mismatch_for_generic_annotation(self) -> None:
        src = "def items(xs: list[int]) -> None:\n    pass\nitems([1, 2])\n"
        mismatches = detect_mismatches(self._edges_for(src))
        assert mismatches == []

    def test_bool_compatible_with_int(self) -> None:
        src = "def f(flag: int) -> None:\n    pass\nf(True)\n"
        mismatches = detect_mismatches(self._edges_for(src))
        assert mismatches == []

    def test_int_compatible_with_float(self) -> None:
        src = "def f(x: float) -> None:\n    pass\nf(1)\n"
        mismatches = detect_mismatches(self._edges_for(src))
        assert mismatches == []

    def test_mismatch_has_correct_location(self) -> None:
        src = "def greet(name: str) -> None:\n    pass\ngreet(99)\n"
        mismatches = detect_mismatches(self._edges_for(src))
        assert mismatches[0].call_site_line == 3
        assert mismatches[0].call_site_file == "<test>"

    def test_skips_excess_positional_args(self) -> None:
        src = "def f(a: str) -> None:\n    pass\nf('ok', 'extra', 99)\n"
        mismatches = detect_mismatches(self._edges_for(src))
        assert mismatches == []

    def test_multiple_params_detects_second_mismatch(self) -> None:
        src = "def f(a: str, b: int) -> None:\n    pass\nf('ok', 'wrong')\n"
        mismatches = detect_mismatches(self._edges_for(src))
        assert len(mismatches) == 1
        assert mismatches[0].param_name == "b"


# ---------------------------------------------------------------------------
# Phase 4: Integration
# ---------------------------------------------------------------------------


class TestCheckDirectory:
    def test_check_directory_returns_check_result(self, tmp_path: Path) -> None:
        (tmp_path / "sample.py").write_text("def f(x: int) -> None:\n    pass\nf(1)\n")
        result = check_directory(tmp_path)
        assert isinstance(result, CheckResult)
        assert len(result.functions) >= 1

    def test_check_directory_detects_mismatch(self, tmp_path: Path) -> None:
        (tmp_path / "bad.py").write_text("def f(x: str) -> None:\n    pass\nf(42)\n")
        result = check_directory(tmp_path)
        assert result.has_errors
        assert "f" in result.summary

    def test_check_directory_clean_on_matching_types(self, tmp_path: Path) -> None:
        (tmp_path / "good.py").write_text("def f(x: str) -> None:\n    pass\nf('hello')\n")
        result = check_directory(tmp_path)
        assert not result.has_errors

    def test_check_directory_empty_dir_returns_empty_result(self, tmp_path: Path) -> None:
        result = check_directory(tmp_path)
        assert result.functions == []
        assert result.mismatches == []

    def test_no_type_mismatches_in_src_analysis(self) -> None:
        """Guard: src/analysis/ must have zero type mismatches detected by the checker.

        Scoped to src/analysis/ rather than all of src/ because the simple name-based
        CGA resolver produces false positives on common method names (execute, max, min)
        that exist in both domain code and standard library / SQLite cursor calls.
        """
        result = check_directory(Path("src/analysis"))
        assert not result.has_errors, result.summary
