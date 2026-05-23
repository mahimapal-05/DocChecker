"""
Critic Agent
────────────
Responsibilities:
  1. Score the Analyst's draft answer on confidence (0.0–1.0)
  2. Identify specific deficiencies: missing info, hallucinations, unsupported claims
  3. If confidence < threshold AND loops remaining → set needs_retry=True + guidance
  4. Otherwise → promote draft_answer to final_answer
"""

from __future__ import annotations

import json
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate

from app.agents.state import AgentState
from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_SYSTEM = """You are a rigorous Critic in a document intelligence system.
Your job is to evaluate an Analyst's draft answer against the original query and the retrieved context.

Respond ONLY with valid JSON matching this exact schema:
{
  "confidence_score": <float 0.0 to 1.0>,
  "critique": "<one or two sentences identifying specific issues, or 'Answer is complete and well-supported'>",
  "needs_retry": <true or false>,
  "retry_guidance": "<specific instruction for the next retrieval attempt, e.g. 'focus on financial data from Q3', or empty string if needs_retry is false>"
}

Scoring rubric:
  0.9–1.0: Answer fully addresses the query with clear source support
  0.7–0.9: Answer mostly correct but minor gaps or weak citations
  0.5–0.7: Answer partially addresses query; important aspects missing
  0.0–0.5: Answer is vague, hallucinated, or misses the core question

Set needs_retry=true ONLY if confidence_score < threshold AND the retrieved context clearly lacks the needed information (not just a bad answer — more retrieval might help).
"""

_HUMAN = """Original query: {query}

Retrieved context (summary):
{context_summary}

Analyst's draft answer:
{draft_answer}

Evaluate now."""


def _summarise_context(chunks) -> str:
    if not chunks:
        return "No chunks retrieved."
    sources = list({c["source"] for c in chunks})
    avg_score = sum(c["score"] for c in chunks) / len(chunks)
    sample = chunks[0]["content"][:200] if chunks else ""
    return (
        f"{len(chunks)} chunks from {len(sources)} source(s): {', '.join(sources[:5])}.\n"
        f"Avg relevance score: {avg_score:.2f}.\n"
        f"Top chunk preview: {sample}..."
    )


def critic_node(state: AgentState) -> AgentState:
    settings = get_settings()
    loop = state["retrieval_loop_count"]

    # If no answer was produced, fail fast
    if not state.get("draft_answer") or state.get("error"):
        return {
            **state,
            "confidence_score": 0.0,
            "critique": "No answer was produced by the Analyst.",
            "needs_retry": loop < settings.max_retrieval_loops,
            "retry_guidance": "Try broader search terms to find relevant documents.",
            "final_answer": state.get("draft_answer", ""),
            "agent_trace": ["[Critic] No draft answer — flagging for retry"],
        }

    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0.0,
        format="json",
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        ("human", _HUMAN),
    ])
    chain = prompt | llm

    context_summary = _summarise_context(state.get("retrieved_chunks", []))

    logger.info("critic_start", loop=loop)

    try:
        response = chain.invoke({
            "query": state["query"],
            "context_summary": context_summary,
            "draft_answer": state["draft_answer"],
        })
        parsed = json.loads(response.content)

        confidence = float(parsed.get("confidence_score", 0.5))
        critique = parsed.get("critique", "")
        needs_retry = bool(parsed.get("needs_retry", False))
        retry_guidance = parsed.get("retry_guidance", "")

        # Hard cap: don't retry if we've hit the loop limit
        if loop >= settings.max_retrieval_loops:
            needs_retry = False
            logger.info("critic_max_loops_reached", loop=loop)

        # Also don't retry if already above threshold
        if confidence >= settings.confidence_threshold:
            needs_retry = False

        logger.info(
            "critic_done",
            confidence=confidence,
            needs_retry=needs_retry,
            loop=loop,
        )

        final_answer = "" if needs_retry else state["draft_answer"]

        return {
            **state,
            "confidence_score": confidence,
            "critique": critique,
            "needs_retry": needs_retry,
            "retry_guidance": retry_guidance,
            "retrieval_loop_count": loop + 1,
            "final_answer": final_answer,
            "agent_trace": [
                f"[Critic] Loop {loop}: confidence={confidence:.2f}, "
                f"needs_retry={needs_retry}. {critique}"
            ],
        }

    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("critic_parse_error", error=str(exc))
        # Parse failed → accept the answer as-is, conservative confidence
        return {
            **state,
            "confidence_score": 0.6,
            "critique": "Critic evaluation failed; accepting answer with moderate confidence.",
            "needs_retry": False,
            "retry_guidance": "",
            "final_answer": state["draft_answer"],
            "agent_trace": [f"[Critic] JSON parse error — accepting draft. Error: {exc}"],
        }
