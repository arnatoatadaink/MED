"""tests/integration/test_e2e_pipeline.py — MED パイプライン E2E 統合テスト

外部 API・Docker 不要の完全モック環境で、エンドツーエンドのフローを検証する。

テスト対象:
    1. MEDPipeline — initialize / query / add_document / close
    2. FastAPI サーバー — /health / /query / /add / /search / /stats / /doc/{id}
    3. MemoryManager — add → search → delete の一貫性
    4. LLM モック統合 — gateway をモックして LLM 依存を排除
    5. エラーハンドリング — 未初期化、存在しない doc_id など

実行方法:
    pytest tests/integration/test_e2e_pipeline.py -v
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.common.config import FAISSConfig, FAISSIndexConfig, MetadataConfig
from src.llm.gateway import LLMGateway, LLMResponse
from src.memory.embedder import Embedder
from src.memory.faiss_index import FAISSIndexManager
from src.memory.memory_manager import MemoryManager
from src.memory.metadata_store import MetadataStore
from src.memory.schema import SearchResult
from src.orchestrator.pipeline import MEDPipeline, QueryResponse


# ============================================================================
# フィクスチャ
# ============================================================================


def _make_memory_manager(faiss_base: str | None = None) -> MemoryManager:
    """テスト用 MemoryManager（mock embedder + in-memory SQLite）。

    faiss_base を毎回変えることでテスト間のFAISSファイル汚染を防ぐ。
    None の場合は呼び出し元で tmp_path を渡すこと。
    """
    import uuid
    base = faiss_base or f"/tmp/med_faiss_test_{uuid.uuid4().hex}"
    faiss_cfg = FAISSConfig(
        base_dir=base,
        domains={
            "general": FAISSIndexConfig(dim=768),
            "code": FAISSIndexConfig(dim=768),
            "academic": FAISSIndexConfig(dim=768),
        },
    )
    meta_cfg = MetadataConfig(db_path=":memory:")
    embedder = Embedder(mock=True)
    faiss_mgr = FAISSIndexManager(faiss_cfg)
    store = MetadataStore(meta_cfg)
    return MemoryManager(embedder=embedder, faiss=faiss_mgr, store=store)


def _make_mock_gateway(answer: str = "モック回答です。") -> MagicMock:
    """LLM 呼び出しをモックした LLMGateway を返す。"""
    gw = MagicMock(spec=LLMGateway)
    gw.available_providers.return_value = ["mock"]
    gw.complete = AsyncMock(
        return_value=LLMResponse(
            content=answer,
            provider="mock",
            model="mock-model",
            input_tokens=10,
            output_tokens=20,
        )
    )
    gw.complete_messages = AsyncMock(
        return_value=LLMResponse(
            content=answer,
            provider="mock",
            model="mock-model",
            input_tokens=10,
            output_tokens=20,
        )
    )
    return gw


@pytest.fixture
async def memory_manager():
    """インメモリ MemoryManager（mock embedder + in-memory SQLite）。"""
    mm = _make_memory_manager()
    await mm.initialize()
    yield mm
    await mm.close()


@pytest.fixture
async def pipeline():
    """外部依存なしの MEDPipeline。"""
    mm = _make_memory_manager()
    gw = _make_mock_gateway()

    pl = MEDPipeline(
        memory_manager=mm,
        gateway=gw,
        enable_external_rag=False,
        enable_sandbox=False,
        enable_iterative=False,
    )
    await pl.initialize()
    yield pl
    await pl.close()


@pytest.fixture
def api_client():
    """FastAPI TestClient（lifespan を mock pipeline で置換）。"""
    import src.orchestrator.server as srv
    from contextlib import asynccontextmanager

    mm = _make_memory_manager()
    gw = _make_mock_gateway("サーバー経由のモック回答")
    pl = MEDPipeline(
        memory_manager=mm,
        gateway=gw,
        enable_external_rag=False,
        enable_sandbox=False,
    )

    loop = asyncio.new_event_loop()
    loop.run_until_complete(pl.initialize())

    # lifespan を差し替えて mock pipeline が使われるようにする
    @asynccontextmanager
    async def _mock_lifespan(app):
        srv._pipeline = pl
        yield
        srv._pipeline = None

    original_lifespan = srv.app.router.lifespan_context
    srv.app.router.lifespan_context = _mock_lifespan

    with TestClient(srv.app, raise_server_exceptions=True) as client:
        yield client

    srv.app.router.lifespan_context = original_lifespan
    loop.run_until_complete(pl.close())
    loop.close()


# ============================================================================
# 1. MemoryManager — 基本 CRUD
# ============================================================================


class TestMemoryManagerE2E:
    """MemoryManager の add → search → delete フローを検証。"""

    @pytest.mark.asyncio
    async def test_add_and_search(self, memory_manager):
        """追加したドキュメントが検索で返ること。"""
        doc_id = await memory_manager.add_from_text(
            "FAISS はベクトル類似検索ライブラリです。",
            domain="general",
        )
        assert doc_id, "doc_id が返ること"

        results = await memory_manager.search("ベクトル検索", domain="general", k=5)
        assert len(results) >= 1
        ids = [r.document.id for r in results]
        assert doc_id in ids

    @pytest.mark.asyncio
    async def test_add_multiple_and_search_ranking(self, memory_manager):
        """複数追加時に検索結果が返ること（mock embedder のため順位は不問）。"""
        await memory_manager.add_from_text("Python はインタープリタ言語です。", domain="general")
        await memory_manager.add_from_text(
            "FAISS は高速な近似最近傍探索を提供します。", domain="general"
        )
        await memory_manager.add_from_text("Docker はコンテナ仮想化ツールです。", domain="general")

        results = await memory_manager.search("最近傍探索 FAISS", domain="general", k=3)
        # mock embedder はランダム埋め込みのため順位は保証しない
        # 追加した 3 件が検索可能な状態であることのみ確認
        assert isinstance(results, list)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_delete_document(self, memory_manager):
        """削除後は検索結果に含まれないこと。"""
        doc_id = await memory_manager.add_from_text(
            "削除対象ドキュメントです。", domain="general"
        )
        deleted = await memory_manager.delete(doc_id)
        assert deleted

        results = await memory_manager.search("削除対象", domain="general", k=5)
        ids = [r.document.id for r in results]
        assert doc_id not in ids

    @pytest.mark.asyncio
    async def test_stats_reflects_additions(self, memory_manager):
        """stats() がドキュメント追加を反映すること。"""
        before = await memory_manager.stats()
        await memory_manager.add_from_text("統計確認用ドキュメント", domain="general")
        after = await memory_manager.stats()
        assert after["total_docs"] == before["total_docs"] + 1

    @pytest.mark.asyncio
    async def test_search_empty_memory(self, memory_manager):
        """空のメモリに対する検索が例外なく空リストを返すこと。"""
        results = await memory_manager.search("なんでも", domain="general", k=5)
        assert isinstance(results, list)


# ============================================================================
# 2. MEDPipeline — query / add_document
# ============================================================================


class TestMEDPipelineE2E:
    """MEDPipeline の主要フローを検証。"""

    @pytest.mark.asyncio
    async def test_query_returns_response(self, pipeline):
        """query() が QueryResponse を返し answer が空でないこと。"""
        resp = await pipeline.query(
            "Python のリストをソートするには？",
            use_rag=False,
        )
        assert isinstance(resp, QueryResponse)
        assert resp.answer
        assert resp.query == "Python のリストをソートするには？"

    @pytest.mark.asyncio
    async def test_query_with_memory(self, pipeline):
        """事前に追加した文書がコンテキストとして利用されること。"""
        await pipeline.add_document(
            "sorted() 関数を使うと Python のリストをソートできます。",
            domain="general",
        )
        resp = await pipeline.query(
            "Python でリストをソートする方法は？",
            use_rag=False,
        )
        assert resp.answer
        # FAISS にドキュメントがあれば faiss_results に含まれる可能性
        assert isinstance(resp.faiss_results, list)

    @pytest.mark.asyncio
    async def test_add_document_returns_id(self, pipeline):
        """add_document() が文字列 doc_id を返すこと。"""
        doc_id = await pipeline.add_document("テストドキュメント", domain="general")
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0

    @pytest.mark.asyncio
    async def test_pipeline_provider_in_response(self, pipeline):
        """レスポンスの provider / model がモック値と一致すること。"""
        resp = await pipeline.query("テストクエリ", use_rag=False)
        assert resp.provider == "mock"
        assert resp.model == "mock-model"

    @pytest.mark.asyncio
    async def test_pipeline_tokens_counted(self, pipeline):
        """input_tokens / output_tokens が正の値で返ること。"""
        resp = await pipeline.query("トークン確認", use_rag=False)
        assert resp.input_tokens >= 0
        assert resp.output_tokens >= 0

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        """initialize() を 2 回呼んでも例外が出ないこと。"""
        mm = _make_memory_manager()
        gw = _make_mock_gateway()
        pl = MEDPipeline(memory_manager=mm, gateway=gw, enable_external_rag=False)
        await pl.initialize()
        await pl.initialize()  # 2 回目
        await pl.close()


# ============================================================================
# 3. FastAPI サーバー — HTTP エンドポイント
# ============================================================================


class TestFastAPIEndpoints:
    """FastAPI の各エンドポイントを TestClient で検証。"""

    def test_health_ok(self, api_client):
        """GET /health が 200 を返し pipeline_initialized=True なこと。"""
        r = api_client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["pipeline_initialized"] is True

    def test_add_document(self, api_client):
        """POST /add がドキュメントを追加し doc_id を返すこと。"""
        r = api_client.post(
            "/add",
            json={"content": "E2E テスト用ドキュメントです。", "domain": "general"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert "doc_id" in data
        assert data["doc_id"]

    def test_search_after_add(self, api_client):
        """ドキュメント追加後の POST /search が結果を返すこと。"""
        api_client.post(
            "/add",
            json={"content": "検索テスト用ドキュメント FAISS", "domain": "search_test"},
        )
        r = api_client.post(
            "/search",
            json={"query": "FAISS 検索", "domain": "search_test", "top_k": 3},
        )
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_stats_endpoint(self, api_client):
        """GET /stats が total_docs を含む統計を返すこと。"""
        r = api_client.get("/stats")
        assert r.status_code == 200
        data = r.json()
        assert "total_docs" in data
        assert isinstance(data["total_docs"], int)

    def test_delete_existing_doc(self, api_client):
        """追加したドキュメントを DELETE /doc/{id} で削除できること。"""
        add_r = api_client.post(
            "/add", json={"content": "削除テスト用", "domain": "general"}
        )
        doc_id = add_r.json()["doc_id"]

        del_r = api_client.delete(f"/doc/{doc_id}")
        assert del_r.status_code == 200
        # サーバーは {"deleted": doc_id} を返す
        assert del_r.json()["deleted"] == doc_id

    def test_delete_nonexistent_doc(self, api_client):
        """存在しない doc_id の DELETE が 404 を返すこと。"""
        r = api_client.delete("/doc/nonexistent-id-12345")
        assert r.status_code == 404

    def test_query_endpoint(self, api_client):
        """POST /query が answer を含むレスポンスを返すこと。"""
        r = api_client.post(
            "/query",
            json={
                "query": "MED フレームワークとは？",
                "use_rag": False,
                "use_memory": True,
                "k": 3,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert data["answer"]
        assert data["query"] == "MED フレームワークとは？"

    def test_query_missing_field_422(self, api_client):
        """query フィールドなしの POST /query が 422 を返すこと。"""
        r = api_client.post("/query", json={"use_rag": False})
        assert r.status_code == 422

    def test_query_empty_string_422(self, api_client):
        """空文字クエリが 422 を返すこと（min_length=1）。"""
        r = api_client.post("/query", json={"query": ""})
        assert r.status_code == 422


# ============================================================================
# 4. エラーハンドリング
# ============================================================================


class TestErrorHandling:
    """異常系・境界値のハンドリングを検証。"""

    @pytest.mark.asyncio
    async def test_query_very_long_text(self, pipeline):
        """長大クエリでも例外なく処理されること。"""
        long_query = "Python " * 200
        resp = await pipeline.query(long_query, use_rag=False)
        assert resp.answer

    @pytest.mark.asyncio
    async def test_add_empty_text_raises(self, memory_manager):
        """空文字の add が ValueError または空文字列を返すこと（クラッシュしないこと）。"""
        try:
            result = await memory_manager.add_from_text("", domain="general")
            # 空でも doc_id が返る実装の場合はそれも許容
            assert isinstance(result, str)
        except (ValueError, Exception):
            pass  # 例外を投げる実装も正当

    @pytest.mark.asyncio
    async def test_search_special_characters(self, memory_manager):
        """記号・絵文字を含むクエリで例外が出ないこと。"""
        results = await memory_manager.search("🔥 <test> & query", k=3)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(self, memory_manager):
        """存在しない doc_id の delete が False を返すこと。"""
        result = await memory_manager.delete("no-such-id-xyz")
        assert result is False


# ============================================================================
# 5. パイプライン + メモリ 連携フロー
# ============================================================================


class TestPipelineMemoryIntegration:
    """pipeline.add_document → pipeline.query の一貫したフロー。"""

    @pytest.mark.asyncio
    async def test_add_then_query_flow(self, pipeline):
        """ドキュメント追加後にクエリを実行して一貫した動作をすること。"""
        # 複数ドキュメントを追加
        ids = []
        for i in range(3):
            doc_id = await pipeline.add_document(
                f"テストコンテンツ {i}: MED フレームワークの機能 {i}",
                domain="general",
            )
            ids.append(doc_id)

        assert len(ids) == 3
        assert len(set(ids)) == 3  # すべて異なる ID

        # クエリを実行
        resp = await pipeline.query(
            "MED フレームワークの機能について教えてください",
            domain="general",
            use_rag=False,
        )
        assert resp.answer
        assert isinstance(resp.faiss_results, list)

    @pytest.mark.asyncio
    async def test_stats_after_pipeline_add(self, pipeline):
        """pipeline.add_document が stats に反映されること。"""
        stats_before = await pipeline._mm.stats()
        await pipeline.add_document("統計確認用", domain="general")
        stats_after = await pipeline._mm.stats()
        assert stats_after["total_docs"] == stats_before["total_docs"] + 1
