"""
Orchestrator Agent
─────────────────
Responsibilities:
  1. Parse the user's query intent (factual / analytical / comparative / summarisation)
  2. Generate 2-3 expanded sub-queries to improve retrieval recall
  3. Initialise loop counters and state bookkeeping
"""

from __future__ import annotations

import json
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate

from app.agents.state import AgentState
from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_SYSTEM = """You are the Orchestrator in a multi-agent document intelligence system.
Your job is to analyse a user query and prepare it for retrieval.

Respond ONLY with valid JSON matching this exact schema:
{
  "intent": "<one of: factual | analytical | comparative | summarisation>",
  "retrieval_queries": ["<query 1>", "<query 2>", "<query 3>"]
}

Rules for retrieval_queries:
- Always include the original query verbatim as the first entry
- Add 1-2 rephrased or decomposed variants that approach the same information need differently
- Keep each query under 20 words
- Do NOT answer the question — only prepare search queries
"""

_HUMAN = "User query: {query}"


def orchestrator_node(state: AgentState) -> AgentState:
    settings = get_settings()
    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0.1,
        format="json",
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        ("human", _HUMAN),
    ])

    chain = prompt | llm
    logger.info("orchestrator_start", query=state["query"])

    try:
        response = chain.invoke({"query": state["query"]})
        parsed = json.loads(response.content)

        intent = parsed.get("intent", "analytical")
        retrieval_queries = parsed.get("retrieval_queries", [state["query"]])

        # Ensure original query is always present
        if state["query"] not in retrieval_queries:
            retrieval_queries.insert(0, state["query"])

        logger.info(
            "orchestrator_done",
            intent=intent,
            num_queries=len(retrieval_queries),
        )

        return {
            **state,
            "query_intent": intent,
            "retrieval_queries": retrieval_queries[:3],  # cap at 3
            "retrieval_loop_count": 0,
            "agent_trace": [f"[Orchestrator] Intent={intent}, expanded to {len(retrieval_queries)} queries"],
        }

    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("orchestrator_parse_error", error=str(exc))
        # Graceful fallback — don't crash the pipeline
        return {
            **state,
            "query_intent": "analytical",
            "retrieval_queries": [state["query"]],
            "retrieval_loop_count": 0,
            "agent_trace": ["[Orchestrator] Fallback: used raw query (JSON parse failed)"],
        }
