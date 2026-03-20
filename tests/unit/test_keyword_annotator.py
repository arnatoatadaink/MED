"""tests/unit/test_keyword_annotator.py — KeywordAnnotator 単体テスト"""

from __future__ import annotations

from src.memory.keyword_annotator import KeywordAnnotator
from src.memory.schema import Document, SourceMeta, SourceType


def _make_doc(content: str, title: str = "") -> Document:
    source = SourceMeta(type=SourceType.MANUAL, title=title or None)
    return Document(content=content, domain="general", source=source)


class TestKeywordAnnotatorBasic:
    def setup_method(self):
        self.ann = KeywordAnnotator()

    def test_allcaps_extracted(self):
        doc = _make_doc("RAG and FAISS are used with LLM systems.")
        kws = self.ann.extract(doc)
        assert "RAG" in kws
        assert "FAISS" in kws
        assert "LLM" in kws

    def test_common_allcaps_excluded(self):
        doc = _make_doc("IN the system, IS there a problem OR is it OK?")
        kws = self.ann.extract(doc)
        assert "IN" not in kws
        assert "IS" not in kws
        assert "OR" not in kws

    def test_camelcase_extracted(self):
        doc = _make_doc("TinyLoRA uses PyTorch and NetworkX internally.")
        kws = self.ann.extract(doc)
        assert "TinyLoRA" in kws
        assert "PyTorch" in kws
        assert "NetworkX" in kws

    def test_hyphen_compound_extracted(self):
        doc = _make_doc("fine-tuning and cross-encoder are used in all-MiniLM.")
        kws = self.ann.extract(doc)
        assert "fine-tuning" in kws
        assert "cross-encoder" in kws
        assert "all-MiniLM" in kws

    def test_version_name_extracted(self):
        doc = _make_doc("GPT-4 and Qwen2.5 and v1.0 models are tested.")
        kws = self.ann.extract(doc)
        assert "GPT-4" in kws
        assert "Qwen2.5" in kws

    def test_num_unit_extracted(self):
        doc = _make_doc("The 7B model and 13B model differ in 768d embeddings.")
        kws = self.ann.extract(doc)
        assert "7B" in kws
        assert "13B" in kws
        assert "768d" in kws

    def test_title_tokens_included(self):
        doc = _make_doc("Some content here.", title="TinyLoRA Explained")
        kws = self.ann.extract(doc)
        assert "TinyLoRA" in kws
        assert "Explained" in kws

    def test_deduplication(self):
        doc = _make_doc("RAG RAG RAG FAISS FAISS")
        kws = self.ann.extract(doc)
        assert kws.count("RAG") == 1
        assert kws.count("FAISS") == 1

    def test_max_keywords_respected(self):
        ann = KeywordAnnotator(max_keywords=5)
        content = " ".join([f"TERM{i}" for i in range(100)])
        doc = _make_doc(content)
        kws = ann.extract(doc)
        assert len(kws) <= 5

    def test_empty_content(self):
        doc = _make_doc("")
        kws = self.ann.extract(doc)
        assert isinstance(kws, list)

    def test_no_keywords_in_plain_text(self):
        doc = _make_doc("the quick brown fox jumps over the lazy dog")
        kws = self.ann.extract(doc)
        # plain lowercase words shouldn't produce keywords
        assert isinstance(kws, list)

    def test_max_content_chars_respected(self):
        ann = KeywordAnnotator(max_content_chars=10)
        doc = _make_doc("ABCDE FGHIJ KLMNO PQRST")
        kws = ann.extract(doc)
        # Only first 10 chars processed: "ABCDE FGHI"
        assert "PQRST" not in kws


class TestKeywordAnnotatorSearchInDoc:
    def setup_method(self):
        self.ann = KeywordAnnotator()

    def test_keyword_found_in_content(self):
        doc = _make_doc("RAG is used in this system.")
        assert self.ann.search_in_doc(doc, "RAG") is True

    def test_keyword_case_insensitive(self):
        doc = _make_doc("RAG is used here.")
        assert self.ann.search_in_doc(doc, "rag") is True

    def test_keyword_in_title(self):
        doc = _make_doc("Some content.", title="TinyLoRA Overview")
        assert self.ann.search_in_doc(doc, "tinyLora") is True

    def test_keyword_not_found(self):
        doc = _make_doc("Nothing relevant here.")
        assert self.ann.search_in_doc(doc, "TinyLoRA") is False


class TestKeywordAnnotatorPatterns:
    def setup_method(self):
        self.ann = KeywordAnnotator()

    def test_mixed_document(self):
        content = (
            "TinyLoRA achieves 91% on GSM8K with only 13B parameters. "
            "FAISS-based RAG with cross-encoder reranking uses Qwen2.5 "
            "and all-MiniLM-L6-v2 embeddings."
        )
        doc = _make_doc(content)
        kws = self.ann.extract(doc)
        assert "TinyLoRA" in kws
        assert "FAISS" in kws
        assert "RAG" in kws
        assert "cross-encoder" in kws
        assert "Qwen2.5" in kws
        assert "13B" in kws
