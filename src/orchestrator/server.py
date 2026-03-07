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
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.orchestrator.pipeline import MEDPipeline, QueryResponse

logger = logging.getLogger(__name__)

# ── パイプラインシングルトン ──────────────────
_pipeline: Optional[MEDPipeline] = None


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
    domain: Optional[str] = None
    k: int = Field(default=5, ge=1, le=20)
    run_code: bool = False


class QueryResponseModel(BaseModel):
    answer: str
    query: str
    provider: str
    model: str
    context_doc_count: int
    input_tokens: int
    output_tokens: int
    sandbox_success: Optional[bool] = None
    sandbox_stdout: Optional[str] = None


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
