"""
FastAPI Routes
──────────────
POST /query          — Run query through the agent pipeline
POST /query/stream   — SSE streaming query
POST /ingest         — Upload and ingest a document
GET  /ingest/status  — Document store stats
GET  /health         — Health check (Ollama connectivity + store stats)
DELETE /collection   — Wipe the vector store (careful!)
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

import httpx
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from app.agents.pipeline import run_query
from app.api.schemas import (
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    StatsResponse,
)
from app.config import get_settings
from app.core.logging import get_logger
from app.core.vector_store import get_vector_store
from app.ingestion.pipeline import get_store_stats, ingest_file

logger = get_logger(__name__)
router = APIRouter()


# ── Query ──────────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, summary="Ask a question")
async def query_endpoint(request: QueryRequest):
    """Run the full multi-agent RAG pipeline and return a structured answer."""
    logger.info("api_query", query=request.query[:80])

    try:
        state = await run_query(request.query)
    except Exception as exc:
        logger.error("api_query_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {exc}",
        )

    answer = state.get("final_answer") or state.get("draft_answer") or "No answer generated."

    return QueryResponse(
        query=request.query,
        answer=answer,
        sources=state.get("sources_used", []),
        confidence_score=state.get("confidence_score", 0.0),
        critique=state.get("critique", ""),
        retrieval_loops=state.get("retrieval_loop_count", 0),
        agent_trace=state.get("agent_trace") if request.include_trace else None,
    )


@router.post("/query/stream", summary="Stream query progress via SSE")
async def query_stream_endpoint(request: QueryRequest):
    """
    Server-Sent Events stream. Emits agent trace events as they complete,
    then a final 'result' event with the full response.
    """
    async def event_generator():
        try:
            # We run the pipeline and emit trace steps
            state = await run_query(request.query)

            # Emit trace steps
            for step in state.get("agent_trace", []):
                yield {"event": "trace", "data": step}
                await asyncio.sleep(0.05)

            answer = state.get("final_answer") or state.get("draft_answer") or ""

            import json
            yield {
                "event": "result",
                "data": json.dumps({
                    "answer": answer,
                    "sources": state.get("sources_used", []),
                    "confidence_score": state.get("confidence_score", 0.0),
                    "retrieval_loops": state.get("retrieval_loop_count", 0),
                }),
            }
        except Exception as exc:
            yield {"event": "error", "data": str(exc)}

    return EventSourceResponse(event_generator())


# ── Ingestion ──────────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse, summary="Upload and ingest a document")
async def ingest_endpoint(file: UploadFile = File(...)):
    settings = get_settings()

    # Validate extension
    allowed = {".pdf", ".csv", ".html", ".htm", ".txt", ".md"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed}",
        )

    # Validate size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_upload_size_mb}MB limit ({size_mb:.1f}MB)",
        )

    # Save to uploads dir
    os.makedirs(settings.upload_dir, exist_ok=True)
    dest = Path(settings.upload_dir) / file.filename

    with open(dest, "wb") as f:
        f.write(contents)

    logger.info("file_saved", path=str(dest), size_mb=round(size_mb, 2))

    try:
        result = ingest_file(dest)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        logger.error("ingest_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        )

    return IngestResponse(**result)


@router.get("/ingest/status", response_model=StatsResponse, summary="Document store stats")
def ingest_status():
    return get_store_stats()


# ── Health ─────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check():
    settings = get_settings()
    ollama_ok = False

    async with httpx.AsyncClient(timeout=5) as client:
        try:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            ollama_ok = resp.status_code == 200
        except Exception:
            pass

    try:
        stats = get_store_stats()
        doc_count = stats["document_count"]
    except Exception:
        doc_count = -1

    return HealthResponse(
        status="ok" if ollama_ok else "degraded",
        ollama_reachable=ollama_ok,
        document_count=doc_count,
        model=settings.ollama_model,
        embed_model=settings.embed_model,
    )


# ── Admin ──────────────────────────────────────────────────────────────────────

@router.delete(
    "/collection",
    summary="Wipe vector store",
    status_code=status.HTTP_200_OK,
)
def delete_collection():
    """Permanently deletes all ingested documents. Use with caution."""
    store = get_vector_store()
    store.delete_collection()
    return {"message": "Collection deleted successfully"}
