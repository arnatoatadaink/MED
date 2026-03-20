"""tests/unit/test_query_expander.py — QueryExpander 単体テスト"""

from __future__ import annotations

from pathlib import Path

from src.rag.query_expander import QueryExpander

# configs/query_expansion.yaml を使用（プロジェクトルートから）
_CONFIG = Path(__file__).parent.parent.parent / "configs" / "query_expansion.yaml"


class TestQueryExpanderInit:
    def test_loads_config(self):
        exp = QueryExpander(_CONFIG)
        # コンパイル済みパターンが存在する
        assert len(exp._compiled) > 0

    def test_missing_config_does_not_raise(self, tmp_path):
        missing = tmp_path / "nonexistent.yaml"
        exp = QueryExpander(missing)
        # パターンなしで初期化される
        assert exp._compiled == []

    def test_reload(self):
        exp = QueryExpander(_CONFIG)
        n = len(exp._compiled)
        exp.reload()
        assert len(exp._compiled) == n


class TestQueryExpanderExpand:
    def setup_method(self):
        self.exp = QueryExpander(_CONFIG)

    # ── 日本語比較クエリ ────────────────────────────────────────────

    def test_comparison_ja_basic(self):
        result = self.exp.expand("TinyLoRAとLoRAの比較")
        assert len(result) >= 2
        assert any("TinyLoRA" in q for q in result)
        assert any("LoRA" in q for q in result)

    def test_comparison_ja_difference(self):
        result = self.exp.expand("TransformerとCNNの違い")
        assert any("Transformer" in q for q in result)
        assert any("CNN" in q for q in result)

    def test_versus_ja(self):
        result = self.exp.expand("RAG対FAISS")
        assert any("RAG" in q for q in result)
        assert any("FAISS" in q for q in result)

    def test_toha_difference_ja(self):
        result = self.exp.expand("TinyLoRAとはどのような技術ですか？通常のLoRAとの違いを教えてください")
        assert any("TinyLoRA" in q for q in result)
        assert any("LoRA" in q for q in result)

    # ── 英語比較クエリ ────────────────────────────────────────────

    def test_versus_en(self):
        result = self.exp.expand("Python vs Java")
        assert any("Python" in q for q in result)
        assert any("Java" in q for q in result)

    def test_comparison_en(self):
        result = self.exp.expand("FAISS and Chroma comparison")
        assert any("FAISS" in q for q in result)
        assert any("Chroma" in q for q in result)

    # ── フォールバック展開 ────────────────────────────────────────

    def test_fallback_with_to_particle_ja(self):
        result = self.exp.expand("RAGとFAISS")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_fallback_with_and_en(self):
        result = self.exp.expand("TinyLoRA and LoRA")
        assert any("TinyLoRA" in q for q in result)

    def test_no_expansion_simple_query(self):
        result = self.exp.expand("What is FAISS?")
        assert result == ["What is FAISS?"]

    def test_empty_query(self):
        result = self.exp.expand("  ")
        assert isinstance(result, list)

    # ── 最大展開数 ────────────────────────────────────────────────

    def test_max_expanded_respected(self):
        result = self.exp.expand("TinyLoRAとLoRAの比較")
        assert len(result) <= self.exp._max_expanded

    # ── 重複除去 ─────────────────────────────────────────────────

    def test_no_duplicate_in_result(self):
        result = self.exp.expand("Python vs Java")
        assert len(result) == len(set(result))


class TestQueryExpanderIsNegative:
    def setup_method(self):
        self.exp = QueryExpander(_CONFIG)

    def test_negative_signal_japanese(self):
        assert self.exp.is_negative("その情報は見つかりません。") is True

    def test_negative_signal_english(self):
        assert self.exp.is_negative("I couldn't find relevant information.") is True

    def test_positive_answer(self):
        assert self.exp.is_negative("FAISS is a library for efficient similarity search.") is False

    def test_empty_string(self):
        assert self.exp.is_negative("") is False

    def test_case_insensitive_signal(self):
        assert self.exp.is_negative("COULD NOT FIND any documents.") is True

    def test_no_signals_when_disabled(self, tmp_path):
        # YAML なし → シグナルなし → always False
        exp = QueryExpander(tmp_path / "missing.yaml")
        assert exp.is_negative("見つかりません") is False


class TestQueryExpanderProperties:
    def setup_method(self):
        self.exp = QueryExpander(_CONFIG)

    def test_retry_max_results_positive(self):
        assert self.exp.retry_max_results > 0

    def test_crag_min_faiss_nonnegative(self):
        assert self.exp.crag_min_faiss >= 0


class TestQueryExpanderCleanTerm:
    def test_strips_japanese_punctuation(self):
        assert QueryExpander._clean_term("、CNN") == "CNN"
        assert QueryExpander._clean_term("LoRA。") == "LoRA"

    def test_strips_brackets(self):
        assert QueryExpander._clean_term("[FAISS]") == "FAISS"

    def test_no_change_for_clean_term(self):
        assert QueryExpander._clean_term("TinyLoRA") == "TinyLoRA"


class TestQueryExpanderIsParticle:
    def test_known_particles(self):
        for p in ["vs", "versus", "比較", "difference", "using", "with"]:
            assert QueryExpander._is_particle(p) is True

    def test_non_particle(self):
        assert QueryExpander._is_particle("TinyLoRA") is False
        assert QueryExpander._is_particle("FAISS") is False
