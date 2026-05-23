"""
Analyst Agent
─────────────
Responsibilities:
  1. Receive retrieved chunks from the Retriever
  2. Synthesise a coherent, grounded answer
  3. Cite sources; do NOT hallucinate beyond the provided context
"""

from __future__ import annotations

from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate

from app.agents.state import AgentState
from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_SYSTEM = """You are an expert Analyst in a document intelligence system.
You will be given:
  - A user query
  - A set of retrieved document chunks with source labels

Your task: write a clear, accurate, well-structured answer grounded ONLY in the provided chunks.

Rules:
- Cite sources using [Source: filename] inline when referencing specific information
- If the chunks do not contain enough information to answer, explicitly say so — do NOT fabricate
- Be concise but complete; use bullet points only if the answer genuinely requires listing
- Do not repeat the query back; go straight to the answer
"""

_HUMAN = """Query: {query}
Intent: {intent}

Retrieved Context:
{context}

Write the answer now."""


def _format_context(chunks) -> str:
    if not chunks:
        return "No relevant documents were retrieved."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        score_pct = int(chunk["score"] * 100)
        parts.append(
            f"[{i}] Source: {chunk['source']} (relevance: {score_pct}%)\n"
            f"{chunk['content'].strip()}"
        )
    return "\n\n".join(parts)


def analyst_node(state: AgentState) -> AgentState:
    settings = get_settings()

    if not state.get("retrieved_chunks"):
        return {
            **state,
            "draft_answer": (
                "I was unable to find relevant information in the document store "
                "to answer your query. Please ensure relevant documents have been ingested."
            ),
            "sources_used": [],
            "agent_trace": ["[Analyst] No chunks available — returned fallback answer"],
        }

    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0.2,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        ("human", _HUMAN),
    ])

    chain = prompt | llm
    context = _format_context(state["retrieved_chunks"])
    sources = list({c["source"] for c in state["retrieved_chunks"]})

    logger.info("analyst_start", num_chunks=len(state["retrieved_chunks"]))

    try:
        response = chain.invoke({
            "query": state["query"],
            "intent": state.get("query_intent", "analytical"),
            "context": context,
        })
        draft = response.content.strip()
        logger.info("analyst_done", answer_len=len(draft))

        return {
            **state,
            "draft_answer": draft,
            "sources_used": sources,
            "agent_trace": [
                f"[Analyst] Generated answer ({len(draft)} chars) from {len(sources)} sources"
            ],
        }

    except Exception as exc:
        logger.error("analyst_error", error=str(exc))
        return {
            **state,
            "draft_answer": f"Analysis failed due to an internal error: {exc}",
            "sources_used": sources,
            "error": str(exc),
            "agent_trace": [f"[Analyst] Error: {exc}"],
        }
