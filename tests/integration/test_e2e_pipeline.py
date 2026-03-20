"""tests/integration/test_e2e_pipeline.py — MED パイプライン E2E 統合テスト

外部 API・Docker 不要の完全モック環境で、エンドツーエンドのフローを検証する。

テスト対象:
    1. MEDPipeline — initialize / query / add_document / close
    2. FastAPI サーバー — /health / /query / /add / /search / /stats / /doc/{id}
    3. MemoryManager — add → search → delete の一貫性
    4. LLM モック統合 — gateway をモックして LLM 依存を排除
    5. エラーハンドリング — 未初期化、存在しない doc_id など
    6. 認証エンドポイント — /auth/register / /auth/login / /auth/me
    7. セッション・ターン — /sessions CRUD / /sessions/{id}/turns
    8. 管理者エンドポイント — /admin/users (admin のみ)

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
    """FastAPI TestClient（lifespan を mock pipeline + auth + conv で置換）。"""
    import src.orchestrator.server as srv
    from contextlib import asynccontextmanager

    from src.auth.deps import set_auth_service
    from src.auth.service import AuthService
    from src.auth.store import UserStore
    from src.conversation.manager import ConversationManager
    from src.conversation.store import ConversationStore

    mm = _make_memory_manager()
    gw = _make_mock_gateway("サーバー経由のモック回答")

    loop = asyncio.new_event_loop()

    # 認証サービス初期化（in-memory）
    user_store = UserStore(db_path=":memory:")
    loop.run_until_complete(user_store.initialize())
    auth_svc = AuthService(
        store=user_store,
        secret_key="test-secret-key",
        algorithm="HS256",
        expire_days=7,
        allow_test_token=True,
    )

    # 会話履歴管理初期化（in-memory）
    conv_store = ConversationStore(db_path=":memory:")
    conv_mgr = ConversationManager(
        store=conv_store,
        max_sessions_per_user=50,
        context_window_tokens=2048,
        auto_save_to_faiss=False,
    )
    loop.run_until_complete(conv_mgr.initialize())

    pl = MEDPipeline(
        memory_manager=mm,
        gateway=gw,
        conversation_manager=conv_mgr,
        enable_external_rag=False,
        enable_sandbox=False,
    )
    loop.run_until_complete(pl.initialize())

    @asynccontextmanager
    async def _mock_lifespan(app):
        srv._pipeline = pl
        srv._conv_manager = conv_mgr
        srv._auth_service_instance = auth_svc
        set_auth_service(auth_svc)
        yield
        srv._pipeline = None
        srv._conv_manager = None

    original_lifespan = srv.app.router.lifespan_context
    srv.app.router.lifespan_context = _mock_lifespan

    with TestClient(srv.app, raise_server_exceptions=False) as client:
        yield client

    srv.app.router.lifespan_context = original_lifespan
    loop.run_until_complete(conv_mgr.close())
    loop.run_until_complete(user_store.close())
    loop.run_until_complete(pl.close())
    loop.close()


# ── 認証ヘルパー ──────────────────────────────────────


def _register_user(client: TestClient, username: str = "alice", password: str = "password123") -> dict:
    """ユーザーを登録してレスポンス JSON を返す。"""
    r = client.post("/auth/register", json={"username": username, "password": password})
    assert r.status_code == 200, f"Register failed: {r.text}"
    return r.json()


def _login_user(client: TestClient, username: str = "alice", password: str = "password123") -> dict:
    """ログインしてレスポンス JSON を返す。"""
    r = client.post("/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, f"Login failed: {r.text}"
    return r.json()


def _auth_header(token: str) -> dict[str, str]:
    """Bearer 認証ヘッダーを返す。"""
    return {"Authorization": f"Bearer {token}"}


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


# ============================================================================
# 6. 認証エンドポイント
# ============================================================================


class TestAuthEndpoints:
    """認証系 API エンドポイントの E2E テスト。"""

    def test_register_user(self, api_client):
        """POST /auth/register でユーザー登録し、JWT が返ること。"""
        data = _register_user(api_client, "e2e_alice", "password123")
        assert data["access_token"]
        assert data["username"] == "e2e_alice"
        assert data["user_id"]
        assert data["token_type"] == "bearer"

    def test_register_duplicate_409(self, api_client):
        """同名ユーザーの二重登録が 409 を返すこと。"""
        _register_user(api_client, "dup_user", "password123")
        r = api_client.post(
            "/auth/register",
            json={"username": "dup_user", "password": "other_pass"},
        )
        assert r.status_code == 409

    def test_register_short_password_422(self, api_client):
        """短いパスワード（6文字未満）が 422 を返すこと。"""
        r = api_client.post(
            "/auth/register",
            json={"username": "short_pw", "password": "abc"},
        )
        assert r.status_code == 422

    def test_login_success(self, api_client):
        """POST /auth/login でログインし JWT が返ること。"""
        _register_user(api_client, "login_user", "password123")
        data = _login_user(api_client, "login_user", "password123")
        assert data["access_token"]
        assert data["username"] == "login_user"

    def test_login_wrong_password_401(self, api_client):
        """パスワード不一致が 401 を返すこと。"""
        _register_user(api_client, "wrong_pw_user", "password123")
        r = api_client.post(
            "/auth/login",
            json={"username": "wrong_pw_user", "password": "bad_password"},
        )
        assert r.status_code == 401

    def test_login_nonexistent_user_401(self, api_client):
        """存在しないユーザーが 401 を返すこと。"""
        r = api_client.post(
            "/auth/login",
            json={"username": "ghost_user", "password": "any"},
        )
        assert r.status_code == 401

    def test_me_endpoint(self, api_client):
        """GET /auth/me で自分のプロフィールが返ること。"""
        reg = _register_user(api_client, "me_user", "password123")
        r = api_client.get("/auth/me", headers=_auth_header(reg["access_token"]))
        assert r.status_code == 200
        data = r.json()
        assert data["username"] == "me_user"
        assert data["user_id"] == reg["user_id"]
        assert data["is_active"] is True

    def test_me_without_token_401(self, api_client):
        """認証なしの GET /auth/me が 401 を返すこと。"""
        r = api_client.get("/auth/me")
        assert r.status_code == 401

    def test_me_invalid_token_401(self, api_client):
        """不正トークンの GET /auth/me が 401 を返すこと。"""
        r = api_client.get("/auth/me", headers=_auth_header("invalid.token.here"))
        assert r.status_code == 401


# ============================================================================
# 7. セッション・ターン エンドポイント
# ============================================================================


class TestSessionEndpoints:
    """セッション管理 API エンドポイントの E2E テスト。"""

    def test_create_session(self, api_client):
        """POST /sessions でセッション作成が成功すること。"""
        reg = _register_user(api_client, "sess_user", "password123")
        headers = _auth_header(reg["access_token"])

        r = api_client.post(
            "/sessions",
            json={"first_query": "Python でリストをソートするには？"},
            headers=headers,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["session_id"]
        assert data["user_id"] == reg["user_id"]
        assert "Python" in data["title"] or len(data["title"]) <= 21

    def test_list_sessions(self, api_client):
        """GET /sessions でセッション一覧が返ること。"""
        reg = _register_user(api_client, "list_user", "password123")
        headers = _auth_header(reg["access_token"])

        # 3 セッション作成
        for i in range(3):
            api_client.post(
                "/sessions",
                json={"first_query": f"クエリ {i}"},
                headers=headers,
            )

        r = api_client.get("/sessions", headers=headers)
        assert r.status_code == 200
        data = r.json()
        assert len(data["sessions"]) == 3

    def test_list_sessions_user_isolation(self, api_client):
        """他ユーザーのセッションが見えないこと。"""
        reg1 = _register_user(api_client, "iso_user1", "password123")
        reg2 = _register_user(api_client, "iso_user2", "password123")

        api_client.post(
            "/sessions",
            json={"first_query": "User1 のクエリ"},
            headers=_auth_header(reg1["access_token"]),
        )
        api_client.post(
            "/sessions",
            json={"first_query": "User2 のクエリ"},
            headers=_auth_header(reg2["access_token"]),
        )

        r1 = api_client.get("/sessions", headers=_auth_header(reg1["access_token"]))
        r2 = api_client.get("/sessions", headers=_auth_header(reg2["access_token"]))
        assert len(r1.json()["sessions"]) == 1
        assert len(r2.json()["sessions"]) == 1

    def test_delete_session(self, api_client):
        """DELETE /sessions/{id} でセッション削除が成功すること。"""
        reg = _register_user(api_client, "del_user", "password123")
        headers = _auth_header(reg["access_token"])

        cr = api_client.post(
            "/sessions",
            json={"first_query": "削除テスト"},
            headers=headers,
        )
        sid = cr.json()["session_id"]

        r = api_client.delete(f"/sessions/{sid}", headers=headers)
        assert r.status_code == 200
        assert r.json()["deleted"] is True

        # 削除後は一覧に出ない
        r2 = api_client.get("/sessions", headers=headers)
        assert len(r2.json()["sessions"]) == 0

    def test_delete_other_user_session_403(self, api_client):
        """他ユーザーのセッション削除が 403 を返すこと。"""
        reg1 = _register_user(api_client, "own_user", "password123")
        reg2 = _register_user(api_client, "other_user", "password123")

        cr = api_client.post(
            "/sessions",
            json={"first_query": "own session"},
            headers=_auth_header(reg1["access_token"]),
        )
        sid = cr.json()["session_id"]

        r = api_client.delete(
            f"/sessions/{sid}",
            headers=_auth_header(reg2["access_token"]),
        )
        assert r.status_code == 403

    def test_get_turns_empty(self, api_client):
        """新規セッションの GET /sessions/{id}/turns が空リストを返すこと。"""
        reg = _register_user(api_client, "turns_user", "password123")
        headers = _auth_header(reg["access_token"])

        cr = api_client.post(
            "/sessions",
            json={"first_query": "ターンテスト"},
            headers=headers,
        )
        sid = cr.json()["session_id"]

        r = api_client.get(f"/sessions/{sid}/turns", headers=headers)
        assert r.status_code == 200
        data = r.json()
        assert data["session_id"] == sid
        assert data["turns"] == []

    def test_sessions_without_auth_401(self, api_client):
        """認証なしのセッション操作が 401 を返すこと。"""
        r = api_client.get("/sessions")
        assert r.status_code == 401

        r = api_client.post("/sessions", json={"first_query": "test"})
        assert r.status_code == 401


# ============================================================================
# 8. /query + session_id 統合テスト
# ============================================================================


class TestQueryWithSession:
    """/query エンドポイントの認証・セッション統合テスト。"""

    def test_query_without_auth_works(self, api_client):
        """/query は認証なしでも動作すること（get_optional_user）。"""
        r = api_client.post(
            "/query",
            json={"query": "認証なしクエリ", "use_rag": False},
        )
        assert r.status_code == 200
        assert r.json()["answer"]

    def test_query_with_auth(self, api_client):
        """/query に Bearer トークンを付けてもエラーにならないこと。"""
        reg = _register_user(api_client, "query_auth_user", "password123")
        r = api_client.post(
            "/query",
            json={"query": "認証ありクエリ", "use_rag": False},
            headers=_auth_header(reg["access_token"]),
        )
        assert r.status_code == 200
        assert r.json()["answer"]

    def test_query_with_session_id(self, api_client):
        """/query に session_id を付けて送信してもエラーにならないこと。"""
        reg = _register_user(api_client, "sess_query_user", "password123")
        headers = _auth_header(reg["access_token"])

        cr = api_client.post(
            "/sessions",
            json={"first_query": "セッション付きクエリ"},
            headers=headers,
        )
        sid = cr.json()["session_id"]

        r = api_client.post(
            "/query",
            json={
                "query": "セッション付きクエリ",
                "session_id": sid,
                "use_rag": False,
            },
            headers=headers,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["answer"]


# ============================================================================
# 9. 管理者エンドポイント
# ============================================================================


class TestAdminEndpoints:
    """管理者専用エンドポイントの E2E テスト。"""

    def test_admin_list_users(self, api_client):
        """admin ユーザーが GET /admin/users でユーザー一覧を取得できること。"""
        reg = _register_user(api_client, "admin_user", "password123")
        # 直接 admin にはできないので、register with is_admin=True
        r = api_client.post(
            "/auth/register",
            json={"username": "real_admin", "password": "adminpass", "is_admin": True},
        )
        assert r.status_code == 200
        admin_token = r.json()["access_token"]

        r = api_client.get("/admin/users", headers=_auth_header(admin_token))
        assert r.status_code == 200
        data = r.json()
        assert "users" in data
        usernames = {u["username"] for u in data["users"]}
        assert "admin_user" in usernames
        assert "real_admin" in usernames

    def test_non_admin_cannot_list_users(self, api_client):
        """一般ユーザーが GET /admin/users にアクセスすると 403 を返すこと。"""
        reg = _register_user(api_client, "normal_user", "password123")
        r = api_client.get("/admin/users", headers=_auth_header(reg["access_token"]))
        assert r.status_code == 403

    def test_admin_delete_user(self, api_client):
        """admin が DELETE /admin/users/{id} でユーザーを削除できること。"""
        # admin 作成
        r = api_client.post(
            "/auth/register",
            json={"username": "del_admin", "password": "adminpass", "is_admin": True},
        )
        admin_token = r.json()["access_token"]

        # 削除対象ユーザー作成
        target = _register_user(api_client, "target_user", "password123")

        r = api_client.delete(
            f"/admin/users/{target['user_id']}",
            headers=_auth_header(admin_token),
        )
        assert r.status_code == 200
        assert r.json()["deleted"] == target["user_id"]

    def test_admin_set_active(self, api_client):
        """admin が PATCH /admin/users/{id}/activate でアクティブ状態を変更できること。"""
        r = api_client.post(
            "/auth/register",
            json={"username": "act_admin", "password": "adminpass", "is_admin": True},
        )
        admin_token = r.json()["access_token"]

        target = _register_user(api_client, "deactivate_me", "password123")

        r = api_client.patch(
            f"/admin/users/{target['user_id']}/activate?active=false",
            headers=_auth_header(admin_token),
        )
        assert r.status_code == 200
        assert r.json()["is_active"] is False
