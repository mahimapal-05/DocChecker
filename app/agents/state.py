"""
Shared state schema flowing through the LangGraph agent graph.
All agents read from and write to this TypedDict.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
from langchain.schema import Document
import operator


class RetrievedChunk(TypedDict):
    content: str
    source: str
    score: float
    metadata: dict[str, Any]


class AgentState(TypedDict):
    # ── Input ────────────────────────────────────────────────────────
    query: str
    query_intent: str                        # Orchestrator fills this

    # ── Retrieval ────────────────────────────────────────────────────
    retrieval_queries: list[str]             # Expanded queries from Orchestrator
    retrieved_chunks: list[RetrievedChunk]  # Retriever fills this
    retrieval_loop_count: int               # How many times we've looped

    # ── Analysis ─────────────────────────────────────────────────────
    draft_answer: str                        # Analyst fills this
    sources_used: list[str]                 # Source filenames cited

    # ── Critic ───────────────────────────────────────────────────────
    confidence_score: float                  # 0.0 – 1.0
    critique: str                            # Critic's reasoning
    needs_retry: bool                        # True → loop back to Retriever
    retry_guidance: str                      # What the Retriever should do differently

    # ── Final output ─────────────────────────────────────────────────
    final_answer: str
    agent_trace: Annotated[list[str], operator.add]  # Audit trail
    error: Optional[str]
