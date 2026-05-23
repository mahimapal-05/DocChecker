from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, description="The question to answer")
    include_trace: bool = Field(False, description="Include agent execution trace in response")


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]
    confidence_score: float
    critique: str
    retrieval_loops: int
    agent_trace: Optional[list[str]] = None


# ── Ingestion ─────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    file: str
    type: str
    pages_loaded: int
    chunks_stored: int
    message: str = "Ingestion successful"


# ── Health / Stats ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    document_count: int
    model: str
    embed_model: str


class StatsResponse(BaseModel):
    document_count: int
    collection: str
