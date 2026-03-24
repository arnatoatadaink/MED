"""tests/unit/test_query_rewriter.py — QueryRewriter ユニットテスト

テスト対象:
- RewriteResult データクラス
- QueryRewriter 初期化・戦略検出
- rule_expand 戦略
- モデル未配置時のエラーハンドリング
- merge_queries 重複除去・統合
- cascade / parallel モード
- llm_rewrite (gateway モック)
- unknown strategy エラー
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag.query_rewriter import QueryRewriter, RewriteResult


# ── ヘルパー ──────────────────────────────────────────────────


def _run(coro):
    """asyncio.run のショートカット。"""
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def tmp_model_dir(tmp_path: Path) -> Path:
    """モデルが存在しないテスト用ディレクトリ。"""
    return tmp_path / "models"


@pytest.fixture
def fake_model_dir(tmp_path: Path) -> Path:
    """モデルディレクトリのダミーを作成する。"""
    d = tmp_path / "models"
    (d / "flan-t5-small").mkdir(parents=True)
    (d / "Qwen2.5-0.5B-Instruct").mkdir(parents=True)
    return d


# ── RewriteResult ─────────────────────────────────────────────


class TestRewriteResult:
    def test_default_fields(self):
        r = RewriteResult(strategy="test", original_query="q")
        assert r.strategy == "test"
        assert r.original_query == "q"
        assert r.rewritten_queries == []
        assert r.error is None

    def test_with_queries(self):
        r = RewriteResult(
            strategy="rule_expand",
            original_query="A vs B",
            rewritten_queries=["A", "B", "A B"],
        )
        assert len(r.rewritten_queries) == 3

    def test_with_error(self):
        r = RewriteResult(
            strategy="flan_t5_rewrite",
            original_query="q",
            error="Model not found",
        )
        assert r.error == "Model not found"
        assert r.rewritten_queries == []


# ── 初期化・戦略検出 ─────────────────────────────────────────


class TestInitialization:
    def test_available_strategies_no_models(self, tmp_model_dir: Path):
        qr = QueryRewriter(model_dir=tmp_model_dir)
        _run(qr.initialize())
        avail = qr.available_strategies()
        assert avail["rule_expand"] is True
        assert avail["flan_t5_rewrite"] is False
        assert avail["qwen_rewrite"] is False
        assert avail["llm_rewrite"] is False

    def test_available_strategies_with_model_dirs(self, fake_model_dir: Path):
        qr = QueryRewriter(model_dir=fake_model_dir)
        _run(qr.initialize())
        avail = qr.available_strategies()
        assert avail["flan_t5_rewrite"] is True
        assert avail["qwen_rewrite"] is True

    def test_available_strategies_with_gateway(self, tmp_model_dir: Path):
        mock_gw = MagicMock()
        qr = QueryRewriter(model_dir=tmp_model_dir, gateway=mock_gw)
        _run(qr.initialize())
        assert qr.available_strategies()["llm_rewrite"] is True

    def test_cascade_order_defined(self):
        assert QueryRewriter.CASCADE_ORDER == [
            "rule_expand", "flan_t5_rewrite", "qwen_rewrite", "llm_rewrite",
        ]


# ── rule_expand 戦略 ──────────────────────────────────────────


class TestRuleExpand:
    def test_compound_query_expanded(self, tmp_model_dir: Path):
        qr = QueryRewriter(model_dir=tmp_model_dir)
        _run(qr.initialize())
        results = _run(qr.rewrite("TinyLoRAとLoRAの比較", strategies=["rule_expand"]))
        assert len(results) == 1
        r = results[0]
        assert r.strategy == "rule_expand"
        assert r.error is None
        assert len(r.rewritten_queries) > 0

    def test_simple_query_no_expansion(self, tmp_model_dir: Path):
        qr = QueryRewriter(model_dir=tmp_model_dir)
        _run(qr.initialize())
        results = _run(qr.rewrite("FAISS", strategies=["rule_expand"]))
        r = results[0]
        # 単純クエリは展開不可 → 空リスト
        assert r.error is None
        assert r.rewritten_queries == []

    def test_default_strategies_is_rule_expand(self, tmp_model_dir: Path):
        qr = QueryRewriter(model_dir=tmp_model_dir)
        _run(qr.initialize())
        results = _run(qr.rewrite("test query"))
        assert len(results) == 1
        assert results[0].strategy == "rule_expand"


# ── モデル未配置時のエラー ────────────────────────────────────


class TestModelNotFound:
    def test_flan_t5_not_available(self, tmp_model_dir: Path):
        qr = QueryRewriter(model_dir=tmp_model_dir)
        _run(qr.initialize())
        results = _run(qr.rewrite("test", strategies=["flan_t5_rewrite"]))
        assert results[0].error is not None
        assert "not found" in results[0].error

    def test_qwen_not_available(self, tmp_model_dir: Path):
        qr = QueryRewriter(model_dir=tmp_model_dir)
        _run(qr.initialize())
        results = _run(qr.rewrite("test", strategies=["qwen_rewrite"]))
        assert results[0].error is not None
        assert "not found" in results[0].error

    def test_llm_no_gateway(self, tmp_model_dir: Path):
        qr = QueryRewriter(model_dir=tmp_model_dir)
        _run(qr.initialize())
        results = _run(qr.rewrite("test", strategies=["llm_rewrite"]))
        assert results[0].error is not None
        assert "not configured" in results[0].error


# ── merge_queries ─────────────────────────────────────────────


class TestMergeQueries:
    def test_merge_with_original(self, tmp_model_dir: Path):
        qr = QueryRewriter(model_dir=tmp_model_dir)
        results = [
            RewriteResult(strategy="a", original_query="orig", rewritten_queries=["q1", "q2"]),
            RewriteResult(strategy="b", original_query="orig", rewritten_queries=["q2", "q3"]),
        ]
        merged = qr.merge_queries(results, include_original=True)
        assert merged[0] == "orig"
        assert "q1" in merged
        assert "q2" in merged
        assert "q3" in merged
        # 重複除去: q2 は1回のみ
        assert merged.count("q2") == 1

    def test_merge_without_original(self, tmp_model_dir: Path):
        qr = QueryRewriter(model_dir=tmp_model_dir)
        results = [
            RewriteResult(strategy="a", original_query="orig", rewritten_queries=["q1"]),
        ]
        merged = qr.merge_queries(results, include_original=False)
        assert "orig" not in merged
        assert merged == ["q1"]

    def test_merge_empty_results(self, tmp_model_dir: Path):
        qr = QueryRewriter(model_dir=tmp_model_dir)
        merged = qr.merge_queries([], include_original=True)
        assert merged == []

    def test_merge_preserves_order(self, tmp_model_dir: Path):
        qr = QueryRewriter(model_dir=tmp_model_dir)
        results = [
            RewriteResult(strategy="a", original_query="orig", rewritten_queries=["c", "b", "a"]),
        ]
        merged = qr.merge_queries(results)
        assert merged == ["orig", "c", "b", "a"]


# ── cascade モード ────────────────────────────────────────────


class TestCascadeMode:
    def test_cascade_stops_on_first_success(self, tmp_model_dir: Path):
        """rule_expand が成功したら後続戦略はスキップされる。"""
        qr = QueryRewriter(model_dir=tmp_model_dir)
        _run(qr.initialize())
        results = _run(qr.rewrite(
            "TinyLoRAとLoRAの比較",
            strategies=["rule_expand", "flan_t5_rewrite", "qwen_rewrite"],
            mode="cascade",
        ))
        # rule_expand が成功するので1件のみ
        assert len(results) == 1
        assert results[0].strategy == "rule_expand"
        assert results[0].rewritten_queries

    def test_cascade_falls_through_on_no_expansion(self, tmp_model_dir: Path):
        """rule_expand で展開不可 → 次の戦略に進む（モデル未配置でエラー付き）。"""
        qr = QueryRewriter(model_dir=tmp_model_dir)
        _run(qr.initialize())
        results = _run(qr.rewrite(
            "FAISS",
            strategies=["rule_expand", "flan_t5_rewrite"],
            mode="cascade",
        ))
        # rule_expand は展開なし → flan_t5 に進むがモデル未配置でエラー
        assert len(results) == 2
        assert results[0].strategy == "rule_expand"
        assert results[0].rewritten_queries == []
        assert results[1].strategy == "flan_t5_rewrite"
        assert results[1].error is not None

    def test_cascade_respects_order(self, tmp_model_dir: Path):
        """逆順で渡しても CASCADE_ORDER でソートされる。"""
        qr = QueryRewriter(model_dir=tmp_model_dir)
        _run(qr.initialize())
        results = _run(qr.rewrite(
            "TinyLoRAとLoRAの違い",
            strategies=["qwen_rewrite", "flan_t5_rewrite", "rule_expand"],
            mode="cascade",
        ))
        # rule_expand が最初に実行されて成功
        assert results[0].strategy == "rule_expand"

    def test_cascade_skips_unavailable(self, tmp_model_dir: Path):
        """利用不可の戦略はスキップ（エラー付き）。"""
        qr = QueryRewriter(model_dir=tmp_model_dir)
        _run(qr.initialize())
        results = _run(qr.rewrite(
            "FAISS",
            strategies=["flan_t5_rewrite"],
            mode="cascade",
        ))
        assert len(results) == 1
        assert results[0].error is not None
        assert "not available" in results[0].error


# ── parallel モード ───────────────────────────────────────────


class TestParallelMode:
    def test_parallel_runs_all_strategies(self, tmp_model_dir: Path):
        qr = QueryRewriter(model_dir=tmp_model_dir)
        _run(qr.initialize())
        results = _run(qr.rewrite(
            "TinyLoRAとLoRAの比較",
            strategies=["rule_expand", "flan_t5_rewrite"],
            mode="parallel",
        ))
        assert len(results) == 2
        assert results[0].strategy == "rule_expand"
        assert results[1].strategy == "flan_t5_rewrite"

    def test_parallel_collects_errors(self, tmp_model_dir: Path):
        qr = QueryRewriter(model_dir=tmp_model_dir)
        _run(qr.initialize())
        results = _run(qr.rewrite(
            "test",
            strategies=["flan_t5_rewrite", "qwen_rewrite", "llm_rewrite"],
            mode="parallel",
        ))
        assert len(results) == 3
        for r in results:
            assert r.error is not None


# ── llm_rewrite (モック) ──────────────────────────────────────


class TestLLMRewrite:
    def test_llm_rewrite_with_mock_gateway(self, tmp_model_dir: Path):
        mock_response = MagicMock()
        mock_response.content = "FAISS vector similarity search\nFAISS Python tutorial\nfaiss-cpu usage guide"

        mock_gw = MagicMock()
        mock_gw.complete = AsyncMock(return_value=mock_response)

        qr = QueryRewriter(model_dir=tmp_model_dir, gateway=mock_gw)
        _run(qr.initialize())
        results = _run(qr.rewrite("FAISSの使い方", strategies=["llm_rewrite"]))

        assert len(results) == 1
        r = results[0]
        assert r.strategy == "llm_rewrite"
        assert r.error is None
        assert len(r.rewritten_queries) == 3
        assert "FAISS vector similarity search" in r.rewritten_queries

    def test_llm_rewrite_max_queries_limit(self, tmp_model_dir: Path):
        mock_response = MagicMock()
        mock_response.content = "q1\nq2\nq3\nq4\nq5"

        mock_gw = MagicMock()
        mock_gw.complete = AsyncMock(return_value=mock_response)

        qr = QueryRewriter(model_dir=tmp_model_dir, gateway=mock_gw)
        _run(qr.initialize())
        results = _run(qr.rewrite("test", strategies=["llm_rewrite"], max_queries_per_strategy=2))

        assert len(results[0].rewritten_queries) <= 2

    def test_llm_rewrite_deduplicates_original(self, tmp_model_dir: Path):
        """元クエリと同じ行は除外される。"""
        mock_response = MagicMock()
        mock_response.content = "test query\nalternative query"

        mock_gw = MagicMock()
        mock_gw.complete = AsyncMock(return_value=mock_response)

        qr = QueryRewriter(model_dir=tmp_model_dir, gateway=mock_gw)
        _run(qr.initialize())
        results = _run(qr.rewrite("test query", strategies=["llm_rewrite"]))

        assert "test query" not in results[0].rewritten_queries
        assert "alternative query" in results[0].rewritten_queries

    def test_llm_rewrite_strips_bullets(self, tmp_model_dir: Path):
        """箇条書き記号が除去される。"""
        mock_response = MagicMock()
        mock_response.content = "- query one\n• query two\n  query three  "

        mock_gw = MagicMock()
        mock_gw.complete = AsyncMock(return_value=mock_response)

        qr = QueryRewriter(model_dir=tmp_model_dir, gateway=mock_gw)
        _run(qr.initialize())
        results = _run(qr.rewrite("test", strategies=["llm_rewrite"]))

        assert "query one" in results[0].rewritten_queries
        assert "query two" in results[0].rewritten_queries
        assert "query three" in results[0].rewritten_queries

    def test_llm_rewrite_handles_exception(self, tmp_model_dir: Path):
        mock_gw = MagicMock()
        mock_gw.complete = AsyncMock(side_effect=RuntimeError("API error"))

        qr = QueryRewriter(model_dir=tmp_model_dir, gateway=mock_gw)
        _run(qr.initialize())
        results = _run(qr.rewrite("test", strategies=["llm_rewrite"]))

        assert results[0].error is not None
        assert "API error" in results[0].error


# ── unknown strategy ──────────────────────────────────────────


class TestUnknownStrategy:
    def test_unknown_strategy_returns_error(self, tmp_model_dir: Path):
        qr = QueryRewriter(model_dir=tmp_model_dir)
        _run(qr.initialize())
        results = _run(qr.rewrite("test", strategies=["nonexistent"]))
        assert len(results) == 1
        assert results[0].error is not None
        assert "Unknown strategy" in results[0].error

    def test_unknown_mixed_with_valid(self, tmp_model_dir: Path):
        qr = QueryRewriter(model_dir=tmp_model_dir)
        _run(qr.initialize())
        results = _run(qr.rewrite(
            "TinyLoRAとLoRAの比較",
            strategies=["rule_expand", "nonexistent"],
            mode="parallel",
        ))
        assert len(results) == 2
        assert results[0].error is None
        assert results[1].error is not None


# ── cascade + llm_rewrite 統合 ────────────────────────────────


class TestCascadeWithLLM:
    def test_cascade_stops_at_llm_if_rule_fails(self, tmp_model_dir: Path):
        """rule_expand 展開なし → llm_rewrite で成功。"""
        mock_response = MagicMock()
        mock_response.content = "optimized search query"

        mock_gw = MagicMock()
        mock_gw.complete = AsyncMock(return_value=mock_response)

        qr = QueryRewriter(model_dir=tmp_model_dir, gateway=mock_gw)
        _run(qr.initialize())
        results = _run(qr.rewrite(
            "FAISS",
            strategies=["rule_expand", "llm_rewrite"],
            mode="cascade",
        ))
        # rule_expand は展開なし → llm_rewrite に到達して成功
        assert len(results) == 2
        assert results[0].strategy == "rule_expand"
        assert results[0].rewritten_queries == []
        assert results[1].strategy == "llm_rewrite"
        assert results[1].rewritten_queries == ["optimized search query"]
