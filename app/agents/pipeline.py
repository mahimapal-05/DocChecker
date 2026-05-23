"""
LangGraph Pipeline
──────────────────
Wires the 4 agents into a stateful directed graph with conditional edges.

Graph topology:
  START
    │
    ▼
  orchestrator
    │
    ▼
  retriever  ◄──────────────────────────────────┐
    │                                            │
    ▼                                            │
  analyst                                        │
    │                                            │
    ▼                                            │
  critic ──── [needs_retry=True] ──── [loop back]
    │
    └── [needs_retry=False or max_loops] ──► END
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.agents.state import AgentState
from app.agents.orchestrator import orchestrator_node
from app.agents.retriever import retriever_node
from app.agents.analyst import analyst_node
from app.agents.critic import critic_node
from app.core.logging import get_logger

logger = get_logger(__name__)


def _should_retry(state: AgentState) -> str:
    """Conditional edge: route back to retriever or exit."""
    if state.get("needs_retry", False):
        logger.info(
            "graph_retry",
            loop=state["retrieval_loop_count"],
            confidence=state.get("confidence_score"),
        )
        return "retry"
    return "done"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("critic", critic_node)

    # Linear edges
    graph.set_entry_point("orchestrator")
    graph.add_edge("orchestrator", "retriever")
    graph.add_edge("retriever", "analyst")
    graph.add_edge("analyst", "critic")

    # Conditional edge from Critic
    graph.add_conditional_edges(
        "critic",
        _should_retry,
        {
            "retry": "retriever",   # loop back
            "done": END,
        },
    )

    return graph.compile()


# Module-level compiled graph (lazy initialised)
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
        logger.info("graph_compiled")
    return _graph


async def run_query(query: str) -> AgentState:
    """
    Run a query through the full agent pipeline.
    Returns the final AgentState with all fields populated.
    """
    graph = get_graph()

    initial_state: AgentState = {
        "query": query,
        "query_intent": "",
        "retrieval_queries": [],
        "retrieved_chunks": [],
        "retrieval_loop_count": 0,
        "draft_answer": "",
        "sources_used": [],
        "confidence_score": 0.0,
        "critique": "",
        "needs_retry": False,
        "retry_guidance": "",
        "final_answer": "",
        "agent_trace": [],
        "error": None,
    }

    logger.info("pipeline_start", query=query[:80])
    result = await graph.ainvoke(initial_state)
    logger.info(
        "pipeline_done",
        confidence=result.get("confidence_score"),
        loops=result.get("retrieval_loop_count"),
        sources=result.get("sources_used"),
    )
    return result
