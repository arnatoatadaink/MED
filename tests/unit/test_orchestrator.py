"""tests/unit/test_orchestrator.py — Pipeline / Server の単体テスト"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.common.config import FAISSConfig, FAISSIndexConfig, MetadataConfig
from src.llm.gateway import LLMGateway, LLMResponse
from src.memory.embedder import Embedder
from src.memory.faiss_index import FAISSIndexManager
from src.memory.memory_manager import MemoryManager
from src.memory.metadata_store import MetadataStore
from src.orchestrator.pipeline import MEDPipeline

# ──────────────────────────────────────────────
# モック
# ──────────────────────────────────────────────


class MockGateway(LLMGateway):
    def __init__(self, response: str = "mock answer") -> None:
        self._response = response
        self._providers = {}
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0

    async def complete(self, prompt, **kwargs) -> LLMResponse:
        self._call_count += 1
        return LLMResponse(
            content=self._response,
            provider="mock",
            model="mock-model",
            input_tokens=10,
            output_tokens=20,
        )

    async def complete_messages(self, messages, **kwargs) -> LLMResponse:
        self._call_count += 1
        return LLMResponse(
            content=self._response,
            provider="mock",
            model="mock-model",
            input_tokens=10,
            output_tokens=20,
        )


def _make_pipeline(gateway_response: str = "Test answer.") -> MEDPipeline:
    faiss_cfg = FAISSConfig(
        base_dir="data/faiss_test",
        domains={
            "code": FAISSIndexConfig(dim=768),
            "general": FAISSIndexConfig(dim=768),
        },
    )
    meta_cfg = MetadataConfig(db_path=":memory:")
    embedder = Embedder(mock=True)
    faiss_mgr = FAISSIndexManager(faiss_cfg)
    store = MetadataStore(meta_cfg)
    mm = MemoryManager(embedder=embedder, faiss=faiss_mgr, store=store)
    gateway = MockGateway(gateway_response)

    from src.rag.retriever import RetrieverRouter
    from src.sandbox.manager import SandboxManager

    class EmptyRouter(RetrieverRouter):
        def __init__(self):
            self._retrievers = {}
            self._timeout = 30.0
            self._max_results = 5

    pipeline = MEDPipeline(
        memory_manager=mm,
        gateway=gateway,
        retriever_router=EmptyRouter(),
        sandbox_manager=SandboxManager(use_docker=False),
        enable_external_rag=False,  # テストでは外部 RAG を無効化
        enable_sandbox=True,
    )
    return pipeline


# ──────────────────────────────────────────────
# MEDPipeline テスト
# ──────────────────────────────────────────────


class TestMEDPipeline:
    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        pipeline = _make_pipeline()
        await pipeline.initialize()
        assert pipeline._initialized
        await pipeline.close()

    @pytest.mark.asyncio
    async def test_not_initialized_raises(self) -> None:
        pipeline = _make_pipeline()
        with pytest.raises(RuntimeError, match="initialize"):
            await pipeline.query("test")

    @pytest.mark.asyncio
    async def test_query_empty_memory(self) -> None:
        pipeline = _make_pipeline("The answer is 42.")
        await pipeline.initialize()
        result = await pipeline.query("What is the answer?")
        assert result.answer == "The answer is 42."
        assert result.query == "What is the answer?"
        await pipeline.close()

    @pytest.mark.asyncio
    async def test_query_with_memory(self) -> None:
        pipeline = _make_pipeline()
        await pipeline.initialize()
        await pipeline.add_document("Python uses Timsort for sorting.", domain="code")
        result = await pipeline.query("How does Python sort?", domain="code")
        assert result.answer is not None
        assert len(result.faiss_results) > 0
        await pipeline.close()

    @pytest.mark.asyncio
    async def test_add_document(self) -> None:
        pipeline = _make_pipeline()
        await pipeline.initialize()
        doc_id = await pipeline.add_document("test content")
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0
        await pipeline.close()

    @pytest.mark.asyncio
    async def test_query_with_code_execution(self) -> None:
        pipeline = _make_pipeline("```python\nprint('hello')\n```\nThis prints hello.")
        await pipeline.initialize()
        result = await pipeline.query("Print hello", run_code=True)
        assert result.answer is not None
        # コード実行結果は sandbox_stdout に入る
        await pipeline.close()

    @pytest.mark.asyncio
    async def test_query_returns_token_counts(self) -> None:
        pipeline = _make_pipeline()
        await pipeline.initialize()
        result = await pipeline.query("test")
        assert result.input_tokens >= 0
        assert result.output_tokens >= 0
        await pipeline.close()

    @pytest.mark.asyncio
    async def test_close_without_initialize(self) -> None:
        """初期化なしのクローズはエラーにならない。"""
        pipeline = _make_pipeline()
        await pipeline.close()  # should not raise


# ──────────────────────────────────────────────
# FastAPI サーバーテスト
# ──────────────────────────────────────────────


class TestServer:
    def _make_client(self, gateway_response: str = "Server answer.") -> TestClient:
        """パイプラインをモックした TestClient を生成する。"""
        import asyncio

        from src.orchestrator import server as server_module

        pipeline = _make_pipeline(gateway_response)
        asyncio.get_event_loop().run_until_complete(pipeline.initialize())

        # lifespan の代わりにモジュールレベル変数を直接上書き
        # （TestClient が lifespan を呼ぶ前にセットしておく）
        original = server_module._pipeline
        server_module._pipeline = pipeline

        # lifespan をパッチして何もしないコンテキストマネージャーに差し替える
        from contextlib import asynccontextmanager
        original_lifespan = server_module.app.router.lifespan_context

        @asynccontextmanager
        async def _noop_lifespan(app):
            yield  # 何もせず yield だけ

        server_module.app.router.lifespan_context = _noop_lifespan
        client = TestClient(server_module.app)

        # 元の lifespan を戻す（テスト後のクリーンアップ）
        server_module.app.router.lifespan_context = original_lifespan
        return client

    def test_health_check(self) -> None:
        client = self._make_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_query_endpoint(self) -> None:
        client = self._make_client("Test response from API.")
        resp = client.post("/query", json={"query": "test question"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert data["answer"] == "Test response from API."

    def test_add_document_endpoint(self) -> None:
        client = self._make_client()
        resp = client.post("/add", json={"content": "new document", "domain": "general"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "doc_id" in data

    def test_stats_endpoint(self) -> None:
        client = self._make_client()
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_docs" in data
        assert "faiss_stats" in data

    def test_delete_nonexistent_doc(self) -> None:
        client = self._make_client()
        resp = client.delete("/doc/nonexistent_id")
        assert resp.status_code == 404

    def test_query_empty_string_rejected(self) -> None:
        client = self._make_client()
        resp = client.post("/query", json={"query": ""})
        assert resp.status_code == 422  # Validation error

    def test_add_then_delete(self) -> None:
        client = self._make_client()
        add_resp = client.post("/add", json={"content": "to delete", "domain": "general"})
        assert add_resp.status_code == 200
        doc_id = add_resp.json()["doc_id"]

        del_resp = client.delete(f"/doc/{doc_id}")
        assert del_resp.status_code == 200
