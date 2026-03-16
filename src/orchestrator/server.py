"""src/orchestrator/server.py — FastAPI オーケストレーターサーバー

MED システムの REST API エンドポイントを提供する。

エンドポイント:
    POST /query      — クエリを受け取り、RAG + LLM で回答を生成
    POST /add        — ドキュメントをメモリに追加
    DELETE /doc/{id} — ドキュメントを削除
    GET  /stats      — メモリ統計情報
    GET  /health     — ヘルスチェック

使い方:
    uvicorn src.orchestrator.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.orchestrator.pipeline import MEDPipeline, QueryResponse

logger = logging.getLogger(__name__)

# ── パイプラインシングルトン ──────────────────
_pipeline: MEDPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI ライフサイクル管理。"""
    global _pipeline
    _pipeline = MEDPipeline()
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


# ── リクエスト / レスポンスモデル ─────────────


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    domain: str | None = None
    k: int = Field(default=5, ge=1, le=20)
    run_code: bool = False
    mode: str | None = None        # "auto" | "student" | "teacher" (ルーティングヒント)
    use_memory: bool = True        # FAISS メモリ検索を使用するか
    use_rag: bool = True           # 外部 RAG 検索を使用するか
    provider: str | None = None    # LLM プロバイダー上書き (例: "anthropic", "openai")
    model: str | None = None       # モデル名上書き (例: "claude-opus-4-6", "gpt-4o")


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


class AddDocumentRequest(BaseModel):
    content: str = Field(..., min_length=1)
    domain: str = "general"


class AddDocumentResponse(BaseModel):
    doc_id: str
    success: bool


class StatsResponse(BaseModel):
    total_docs: int
    avg_confidence: float
    faiss_stats: dict


# ── エンドポイント ────────────────────────────


@app.get("/health")
async def health():
    """ヘルスチェック。"""
    return {"status": "ok", "pipeline_initialized": _pipeline is not None}


@app.post("/query", response_model=QueryResponseModel)
async def query(request: QueryRequest):
    """クエリを処理し、RAG + LLM で回答を生成する。"""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        result: QueryResponse = await _pipeline.query(
            query=request.query,
            domain=request.domain,
            k=request.k,
            run_code=request.run_code,
            provider=request.provider,
            model=request.model,
            use_memory=request.use_memory,
            use_rag=request.use_rag,
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
    )


@app.post("/add", response_model=AddDocumentResponse)
async def add_document(request: AddDocumentRequest):
    """ドキュメントをメモリに追加する。"""
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
    """ドキュメントを削除する。"""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    deleted = await _pipeline._mm.delete(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return {"deleted": doc_id}


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """メモリ統計情報を返す。"""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    s = await _pipeline._mm.stats()
    return StatsResponse(
        total_docs=s["total_docs"],
        avg_confidence=s["avg_confidence"],
        faiss_stats=s["faiss_stats"],
    )


# ── Phase 2 成熟管理エンドポイント ────────────────────────────────


@app.get("/maturation/quality")
async def maturation_quality(domain: str | None = None):
    """Phase 2 品質レポートを返す。

    QualityReport には以下が含まれる:
    - 総ドキュメント数 / 承認数 / 却下数 / 未審査数
    - 平均信頼度 / 平均品質スコア / 平均複合スコア
    - 実行成功率 / 難易度分布
    - Phase 2 目標達成フラグ (docs≥10000, confidence≥0.7, exec_success≥0.8)
    """
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
        "total_docs":             report.total_docs,
        "approved_docs":          report.approved_docs,
        "rejected_docs":          report.rejected_docs,
        "pending_docs":           report.pending_docs,
        "avg_confidence":         report.avg_confidence,
        "avg_teacher_quality":    report.avg_teacher_quality,
        "avg_composite_score":    report.avg_composite_score,
        "exec_success_rate":      report.exec_success_rate,
        "avg_retrieval_count":    report.avg_retrieval_count,
        "difficulty_distribution": report.difficulty_distribution,
        "approval_rate":          report.approval_rate,
        "meets_phase2_goal":      report.meets_phase2_goal,
        "phase2_progress":        report.phase2_progress,
        "doc_target":             report.doc_target,
        "confidence_target":      report.confidence_target,
        "exec_success_target":    report.exec_success_target,
    }


@app.get("/maturation/teachers")
async def maturation_teachers():
    """Teacher 信頼度プロファイル一覧を返す。

    TeacherRegistry が未初期化の場合は空リストを返す。
    """
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
                "teacher_id":  p.teacher_id,
                "provider":    p.provider,
                "trust_score": round(p.trust_score, 4),
                "total_docs":  p.total_docs,
                "avg_reward":  round(p.avg_reward, 4),
                "n_feedback":  p.n_feedback,
                "created_at":  p.created_at,
                "updated_at":  p.updated_at,
            }
            for p in profiles
        ]
    }


class ReviewRequest(BaseModel):
    limit: int = Field(default=50, ge=1, le=500)
    concurrency: int = Field(default=5, ge=1, le=20)


@app.post("/maturation/review")
async def maturation_review(request: ReviewRequest):
    """未審査ドキュメントを一括審査する。

    Teacher LLM が各ドキュメントを評価し、
    quality_score / confidence / review_status を更新する。

    Args:
        limit: 最大審査件数（デフォルト 50）。
        concurrency: 並列審査数（デフォルト 5）。
    """
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
        "reviewed":  len(results),
        "approved":  approved,
        "rejected":  rejected,
        "avg_quality": (
            sum(r.quality_score for r in results) / len(results)
            if results else 0.0
        ),
        "avg_confidence": (
            sum(r.confidence for r in results) / len(results)
            if results else 0.0
        ),
    }
