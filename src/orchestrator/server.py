"""src/orchestrator/server.py — FastAPI オーケストレーターサーバー

MED システムの REST API エンドポイントを提供する。

エンドポイント:
    POST /query              — クエリを受け取り、RAG + LLM で回答を生成
    POST /add                — ドキュメントをメモリに追加
    DELETE /doc/{id}         — ドキュメントを削除
    GET  /stats              — メモリ統計情報
    GET  /health             — ヘルスチェック
    POST /auth/register      — ユーザー登録
    POST /auth/login         — ログイン → JWT 返却
    POST /auth/token/test    — テストユーザー用トークン発行（localhost 限定）
    GET  /auth/me            — 自分のプロフィール確認
    GET  /sessions           — セッション一覧
    POST /sessions           — セッション作成
    DELETE /sessions/{id}    — セッション削除
    GET  /sessions/{id}/turns — ターン一覧（GUI 復元用）
    GET  /admin/users        — ユーザー一覧（admin のみ）
    DELETE /admin/users/{id} — ユーザー削除（admin のみ）

使い方:
    uvicorn src.orchestrator.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.auth.deps import (
    get_auth_service,
    get_current_admin,
    get_current_user,
    get_optional_user,
    require_localhost,
    set_auth_service,
)
from src.auth.schema import User
from src.auth.service import AuthService
from src.auth.store import UserStore
from src.common.config import get_settings
from src.conversation.manager import ConversationManager
from src.conversation.schema import Session
from src.conversation.store import ConversationStore
from src.orchestrator.pipeline import MEDPipeline, QueryResponse

logger = logging.getLogger(__name__)

# ── シングルトン ──────────────────────────────────
_pipeline: MEDPipeline | None = None
_conv_manager: ConversationManager | None = None
_auth_service_instance: AuthService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI ライフサイクル管理。"""
    global _pipeline, _conv_manager, _auth_service_instance

    cfg = get_settings()

    # 認証サービス初期化
    user_store = UserStore(db_path=str(cfg.auth.users_db_path))
    await user_store.initialize()
    _auth_service_instance = AuthService(
        store=user_store,
        secret_key=cfg.auth.jwt_secret_key,
        algorithm=cfg.auth.jwt_algorithm,
        expire_days=cfg.auth.access_token_expire_days,
        allow_test_token=cfg.auth.allow_test_token,
    )
    set_auth_service(_auth_service_instance)
    logger.info("AuthService initialized")

    # 会話履歴管理初期化
    conv_store = ConversationStore(db_path=str(cfg.conversation.db_path))
    _conv_manager = ConversationManager(
        store=conv_store,
        max_sessions_per_user=cfg.conversation.max_sessions_per_user,
        context_window_tokens=cfg.conversation.context_window_tokens,
        auto_save_to_faiss=cfg.conversation.auto_save_to_faiss,
    )
    await _conv_manager.initialize()
    logger.info("ConversationManager initialized")

    # パイプライン初期化
    _pipeline = MEDPipeline(conversation_manager=_conv_manager)
    await _pipeline.initialize()
    logger.info("MED Pipeline started")

    yield

    if _pipeline:
        await _pipeline.close()
    logger.info("MED Pipeline stopped")


app = FastAPI(
    title="MED — Memory Environment Distillation",
    description="RAG × FAISS × LLM × TinyLoRA システム",
    version="1.0.0",
    lifespan=lifespan,
)


# ── リクエスト / レスポンスモデル ─────────────────


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    domain: str | None = None
    k: int = Field(default=5, ge=1, le=20)
    run_code: bool = False
    mode: str | None = None
    use_memory: bool = True
    use_rag: bool = True
    provider: str | None = None
    model: str | None = None
    timeout_seconds: int = Field(default=300, ge=5, le=86400)
    session_id: str | None = None   # 会話セッション ID
    crag_strategies: list[str] | None = None  # CRAG 戦略リスト
    crag_mode: str = "cascade"  # "cascade" or "parallel"


class QueryResponseModel(BaseModel):
    answer: str
    query: str
    provider: str
    model: str
    context_doc_count: int
    input_tokens: int
    output_tokens: int
    sandbox_success: bool | None = None
    sandbox_stdout: str | None = None
    debug_info: dict | None = None
    session_id: str | None = None


class AddDocumentRequest(BaseModel):
    content: str = Field(..., min_length=1)
    domain: str = "general"


class AddDocumentResponse(BaseModel):
    doc_id: str
    success: bool


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    domain: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class StatsResponse(BaseModel):
    total_docs: int
    avg_confidence: float
    faiss_stats: dict


# ── Auth リクエスト / レスポンスモデル ──────────────


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=6)
    is_admin: bool = False


class LoginRequest(BaseModel):
    username: str
    password: str


class TestTokenRequest(BaseModel):
    username: str


class TokenResponseModel(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    username: str
    is_admin: bool


class UserModel(BaseModel):
    user_id: str
    username: str
    is_test: bool
    is_admin: bool
    is_active: bool
    created_at: str
    last_login: str | None


# ── セッション リクエスト / レスポンスモデル ──────────


class CreateSessionRequest(BaseModel):
    first_query: str = Field(..., min_length=1)
    domain: str = "general"


class SessionModel(BaseModel):
    session_id: str
    user_id: str
    title: str
    domain: str
    created_at: str
    updated_at: str
    turn_count: int
    display_title: str


class TurnModel(BaseModel):
    turn_id: str
    role: str
    content: str
    timestamp: str
    provider: str
    model: str
    token_count: int
    faiss_doc_id: str | None


# ============================================================================
# 基本エンドポイント
# ============================================================================


@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_initialized": _pipeline is not None}


@app.post("/query", response_model=QueryResponseModel)
async def query(
    request: QueryRequest,
    current_user: User | None = Depends(get_optional_user),
):
    """クエリを処理し、RAG + LLM で回答を生成する。"""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    user_id = current_user.user_id if current_user else "default"

    try:
        coro = _pipeline.query(
            query=request.query,
            domain=request.domain,
            k=request.k,
            run_code=request.run_code,
            provider=request.provider,
            model=request.model,
            use_memory=request.use_memory,
            use_rag=request.use_rag,
            session_id=request.session_id,
            user_id=user_id,
            timeout=float(request.timeout_seconds),
            crag_strategies=request.crag_strategies,
            crag_mode=request.crag_mode,
        )
        result: QueryResponse = await asyncio.wait_for(
            coro, timeout=float(request.timeout_seconds)
        )
    except TimeoutError:
        logger.warning("Query timed out after %ds", request.timeout_seconds)
        raise HTTPException(
            status_code=504,
            detail=f"クエリがタイムアウトしました ({request.timeout_seconds}秒)。",
        )
    except Exception as exc:
        logger.exception("Query failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return QueryResponseModel(
        answer=result.answer,
        query=result.query,
        provider=result.provider,
        model=result.model,
        context_doc_count=len(result.context_doc_ids),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        sandbox_success=result.sandbox_success if request.run_code else None,
        sandbox_stdout=result.sandbox_stdout if request.run_code else None,
        debug_info=result.debug_info,
        session_id=result.session_id,
    )


@app.get("/crag/strategies")
async def crag_strategies():
    """CRAG 戦略の一覧と利用可否を返す。"""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    from src.rag.query_rewriter import QueryRewriter
    available = _pipeline.rewriter.available_strategies()
    return {
        "strategies": [
            {"key": k, "label": QueryRewriter.STRATEGIES.get(k, k), "available": v}
            for k, v in available.items()
        ]
    }


@app.post("/add", response_model=AddDocumentResponse)
async def add_document(request: AddDocumentRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    try:
        doc_id = await _pipeline.add_document(request.content, domain=request.domain)
        return AddDocumentResponse(doc_id=doc_id, success=True)
    except Exception as exc:
        logger.exception("Add document failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/doc/{doc_id}")
async def delete_document(doc_id: str):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    deleted = await _pipeline._mm.delete(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return {"deleted": doc_id}


@app.post("/search")
async def search_memory(request: SearchRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    try:
        results = await _pipeline._mm.search(
            request.query, domain=request.domain, k=request.top_k
        )
    except Exception as exc:
        logger.exception("Search failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        "results": [
            {
                "id": sr.document.id,
                "content": sr.document.content[:500],
                "score": round(sr.score, 4),
                "domain": sr.document.domain,
                "source": sr.document.source_url or "",
            }
            for sr in results
        ]
    }


@app.get("/stats", response_model=StatsResponse)
async def stats():
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    s = await _pipeline._mm.stats()
    return StatsResponse(
        total_docs=s["total_docs"],
        avg_confidence=s["avg_confidence"],
        faiss_stats=s["faiss_stats"],
    )


# ============================================================================
# 認証エンドポイント
# ============================================================================


@app.post("/auth/register", response_model=TokenResponseModel)
async def register(
    req: RegisterRequest,
    svc: AuthService = Depends(get_auth_service),
):
    """ユーザー登録してトークンを返す。"""
    cfg = get_settings()
    if not cfg.auth.allow_registration:
        raise HTTPException(status_code=403, detail="Registration is disabled")
    try:
        user = await svc.register(req.username, req.password, is_admin=req.is_admin)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    token = svc.create_token(user)
    return TokenResponseModel(
        access_token=token,
        user_id=user.user_id,
        username=user.username,
        is_admin=user.is_admin,
    )


@app.post("/auth/login", response_model=TokenResponseModel)
async def login(
    req: LoginRequest,
    svc: AuthService = Depends(get_auth_service),
):
    """ログインしてトークンを返す。"""
    try:
        resp = await svc.login(req.username, req.password)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    return TokenResponseModel(
        access_token=resp.access_token,
        user_id=resp.user_id,
        username=resp.username,
        is_admin=resp.is_admin,
    )


@app.post("/auth/token/test", response_model=TokenResponseModel)
async def test_token(
    req: TestTokenRequest,
    request: Request,
    svc: AuthService = Depends(get_auth_service),
):
    """テストユーザー用トークン発行（localhost 限定）。"""
    cfg = get_settings()
    if cfg.auth.test_token_localhost_only:
        require_localhost(request)
    try:
        resp = await svc.issue_test_token(req.username)
    except (ValueError, PermissionError) as e:
        raise HTTPException(status_code=403, detail=str(e))
    return TokenResponseModel(
        access_token=resp.access_token,
        user_id=resp.user_id,
        username=resp.username,
        is_admin=resp.is_admin,
    )


@app.get("/auth/me", response_model=UserModel)
async def me(current_user: User = Depends(get_current_user)):
    """自分のプロフィールを返す。"""
    return _user_to_model(current_user)


# ============================================================================
# セッション エンドポイント
# ============================================================================


def _session_to_model(s: Session) -> SessionModel:
    return SessionModel(
        session_id=s.session_id,
        user_id=s.user_id,
        title=s.title,
        domain=s.domain,
        created_at=s.created_at.isoformat(),
        updated_at=s.updated_at.isoformat(),
        turn_count=s.turn_count,
        display_title=s.display_title(),
    )


@app.get("/sessions")
async def list_sessions(
    limit: int = 30,
    current_user: User = Depends(get_current_user),
):
    """ユーザーのセッション一覧を返す。"""
    if _conv_manager is None:
        raise HTTPException(status_code=503, detail="ConversationManager not initialized")
    sessions = await _conv_manager.list_sessions(current_user.user_id, limit=limit)
    return {"sessions": [_session_to_model(s) for s in sessions]}


@app.post("/sessions")
async def create_session(
    req: CreateSessionRequest,
    current_user: User = Depends(get_current_user),
):
    """新規セッションを作成して返す。"""
    if _conv_manager is None:
        raise HTTPException(status_code=503, detail="ConversationManager not initialized")
    session = await _conv_manager.create_session(
        user_id=current_user.user_id,
        first_query=req.first_query,
        domain=req.domain,
    )
    return _session_to_model(session)


@app.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    """セッションを削除する（自分のセッションのみ）。"""
    if _conv_manager is None:
        raise HTTPException(status_code=503, detail="ConversationManager not initialized")
    session = await _conv_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id != current_user.user_id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Cannot delete other user's session")
    deleted = await _conv_manager.delete_session(session_id)
    return {"deleted": deleted, "session_id": session_id}


@app.get("/sessions/{session_id}/turns")
async def get_turns(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    """セッションの全ターンを返す（GUI 復元用）。"""
    if _conv_manager is None:
        raise HTTPException(status_code=503, detail="ConversationManager not initialized")
    session = await _conv_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id != current_user.user_id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")
    turns = await _conv_manager.get_all_turns(session_id)
    return {
        "session_id": session_id,
        "turns": [
            TurnModel(
                turn_id=t.turn_id,
                role=t.role,
                content=t.content,
                timestamp=t.timestamp.isoformat(),
                provider=t.provider,
                model=t.model,
                token_count=t.token_count,
                faiss_doc_id=t.faiss_doc_id,
            )
            for t in turns
        ],
    }


# ============================================================================
# 管理エンドポイント（admin のみ）
# ============================================================================


def _user_to_model(u: User) -> UserModel:
    return UserModel(
        user_id=u.user_id,
        username=u.username,
        is_test=u.is_test,
        is_admin=u.is_admin,
        is_active=u.is_active,
        created_at=u.created_at.isoformat(),
        last_login=u.last_login.isoformat() if u.last_login else None,
    )


@app.get("/admin/users")
async def admin_list_users(
    _: User = Depends(get_current_admin),
    svc: AuthService = Depends(get_auth_service),
):
    users = await svc._store.list_all(include_inactive=True)
    return {"users": [_user_to_model(u) for u in users]}


@app.delete("/admin/users/{user_id}")
async def admin_delete_user(
    user_id: str,
    _: User = Depends(get_current_admin),
    svc: AuthService = Depends(get_auth_service),
):
    deleted = await svc._store.delete(user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="User not found")
    return {"deleted": user_id}


@app.patch("/admin/users/{user_id}/activate")
async def admin_set_active(
    user_id: str,
    active: bool = True,
    _: User = Depends(get_current_admin),
    svc: AuthService = Depends(get_auth_service),
):
    await svc._store.set_active(user_id, active)
    return {"user_id": user_id, "is_active": active}


# ============================================================================
# Phase 2 成熟管理エンドポイント
# ============================================================================


@app.get("/maturation/quality")
async def maturation_quality(domain: str | None = None):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    from src.memory.maturation.quality_metrics import QualityMetrics
    qm = QualityMetrics(_pipeline._mm.store)
    try:
        report = await qm.compute(domain=domain)
    except Exception as exc:
        logger.exception("Quality metrics failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        "total_docs": report.total_docs,
        "approved_docs": report.approved_docs,
        "rejected_docs": report.rejected_docs,
        "pending_docs": report.pending_docs,
        "avg_confidence": report.avg_confidence,
        "avg_teacher_quality": report.avg_teacher_quality,
        "avg_composite_score": report.avg_composite_score,
        "exec_success_rate": report.exec_success_rate,
        "avg_retrieval_count": report.avg_retrieval_count,
        "difficulty_distribution": report.difficulty_distribution,
        "approval_rate": report.approval_rate,
        "meets_phase2_goal": report.meets_phase2_goal,
        "phase2_progress": report.phase2_progress,
        "doc_target": report.doc_target,
        "confidence_target": report.confidence_target,
        "exec_success_target": report.exec_success_target,
    }


@app.get("/maturation/teachers")
async def maturation_teachers():
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    registry = _pipeline._mm.teacher_registry
    if registry is None:
        return {"teachers": []}
    try:
        profiles = await registry.list_all()
    except Exception as exc:
        logger.exception("Teacher registry list_all failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        "teachers": [
            {
                "teacher_id": p.teacher_id,
                "provider": p.provider,
                "trust_score": round(p.trust_score, 4),
                "total_docs": p.total_docs,
                "avg_reward": round(p.avg_reward, 4),
                "n_feedback": p.n_feedback,
                "created_at": p.created_at,
                "updated_at": p.updated_at,
            }
            for p in profiles
        ]
    }


class ReviewRequest(BaseModel):
    limit: int = Field(default=50, ge=1, le=500)
    concurrency: int = Field(default=5, ge=1, le=20)


@app.post("/maturation/review")
async def maturation_review(request: ReviewRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    from src.memory.maturation.reviewer import MemoryReviewer
    reviewer = MemoryReviewer(
        _pipeline._gateway,
        _pipeline._mm.store,
        max_text_length=1200,
        approval_threshold=0.6,
    )
    try:
        results = await reviewer.review_unreviewed(limit=request.limit)
    except Exception as exc:
        logger.exception("Batch review failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    approved = sum(1 for r in results if r.approved)
    rejected = len(results) - approved
    return {
        "reviewed": len(results),
        "approved": approved,
        "rejected": rejected,
        "avg_quality": (
            sum(r.quality_score for r in results) / len(results) if results else 0.0
        ),
        "avg_confidence": (
            sum(r.confidence for r in results) / len(results) if results else 0.0
        ),
    }
