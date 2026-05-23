"""
Retriever Agent
───────────────
Responsibilities:
  1. Run all expanded queries against ChromaDB
  2. Merge, de-duplicate results across queries
  3. Re-rank with a simple MMR-style diversity score
  4. On retry loops, apply guidance from the Critic
"""

from __future__ import annotations

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from app.agents.state import AgentState, RetrievedChunk
from app.core.vector_store import get_vector_store
from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _deduplicate(
    chunks: list[tuple[str, str, float, dict]],
) -> list[tuple[str, str, float, dict]]:
    """Remove near-duplicate chunks by content fingerprint."""
    seen: set[str] = set()
    unique = []
    for content, source, score, meta in chunks:
        fingerprint = content[:120].strip().lower()
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append((content, source, score, meta))
    return unique


def _mmr_rerank(
    chunks: list[tuple[str, str, float, dict]],
    top_k: int,
    lambda_param: float = 0.6,
) -> list[tuple[str, str, float, dict]]:
    """
    Maximal Marginal Relevance re-ranking.
    Balances relevance (score) vs diversity (cosine distance from already-selected chunks).
    lambda_param=1.0 → pure relevance, 0.0 → pure diversity.
    """
    if len(chunks) <= top_k:
        return chunks

    # Use content length as a proxy embedding when we don't have vectors here.
    # Simple word-overlap Jaccard diversity instead of full vector MMR.
    def word_set(text: str) -> set[str]:
        return set(text.lower().split())

    selected: list[tuple[str, str, float, dict]] = []
    remaining = list(chunks)

    while len(selected) < top_k and remaining:
        if not selected:
            # First pick: highest score
            best = max(remaining, key=lambda x: x[2])
        else:
            selected_sets = [word_set(s[0]) for s in selected]

            def mmr_score(chunk):
                rel = chunk[2]
                cset = word_set(chunk[0])
                max_sim = max(
                    len(cset & ss) / max(len(cset | ss), 1)
                    for ss in selected_sets
                )
                return lambda_param * rel - (1 - lambda_param) * max_sim

            best = max(remaining, key=mmr_score)

        selected.append(best)
        remaining.remove(best)

    return selected


def retriever_node(state: AgentState) -> AgentState:
    settings = get_settings()
    store = get_vector_store()
    queries = state["retrieval_queries"]
    loop = state["retrieval_loop_count"]

    # On retry: prepend guidance to queries
    if loop > 0 and state.get("retry_guidance"):
        guidance = state["retry_guidance"]
        queries = [f"{guidance} {q}" for q in queries]
        logger.info("retriever_retry", loop=loop, guidance=guidance)

    all_chunks: list[tuple[str, str, float, dict]] = []

    for q in queries:
        try:
            results = store.similarity_search_with_score(
                q, k=settings.retrieval_expansion_k
            )
            for doc, score in results:
                all_chunks.append((
                    doc.page_content,
                    doc.metadata.get("source", "unknown"),
                    float(score),
                    doc.metadata,
                ))
        except Exception as exc:
            logger.warning("retriever_query_failed", query=q, error=str(exc))

    if not all_chunks:
        logger.warning("retriever_no_results")
        return {
            **state,
            "retrieved_chunks": [],
            "agent_trace": [f"[Retriever] Loop {loop}: no documents found"],
        }

    # De-duplicate then re-rank
    unique = _deduplicate(all_chunks)
    reranked = _mmr_rerank(unique, top_k=settings.top_k_results)

    chunks: list[RetrievedChunk] = [
        RetrievedChunk(content=c, source=s, score=sc, metadata=m)
        for c, s, sc, m in reranked
    ]

    logger.info(
        "retriever_done",
        loop=loop,
        raw=len(all_chunks),
        after_dedup=len(unique),
        final=len(chunks),
    )

    return {
        **state,
        "retrieved_chunks": chunks,
        "agent_trace": [
            f"[Retriever] Loop {loop}: {len(chunks)} chunks "
            f"(from {len(all_chunks)} raw, {len(unique)} unique)"
        ],
    }
