"""LangGraph node functions for BrainCache agent flow."""

from app.agent.state import AgentState
from app.services import embeddings, pinecone


async def retrieve(state: AgentState) -> AgentState:
    """Retrieve candidate context chunks from Pinecone."""
    query_vector = await embeddings.embed_query(state["query"])
    results = await pinecone.query(vector=query_vector, top_k=state.get("top_k", 5))
    return {
        **state,
        "search_results": results,
        "reasoning_steps": [*state.get("reasoning_steps", []), "Retrieved candidate chunks"],
    }


def grade_documents(state: AgentState) -> AgentState:
    """Grade retrieval quality to decide whether to rewrite query."""
    needs_more_context = len(state.get("search_results", [])) == 0
    return {
        **state,
        "needs_more_context": needs_more_context,
        "reasoning_steps": [
            *state.get("reasoning_steps", []),
            "Graded retrieval relevance",
        ],
    }


def generate(state: AgentState) -> AgentState:
    """Generate a placeholder answer from available context."""
    if state.get("search_results"):
        answer = "Found relevant context. Detailed answer generation will be added in Phase 2."
    else:
        answer = "No relevant context found yet."
    return {
        **state,
        "answer": answer,
        "reasoning_steps": [*state.get("reasoning_steps", []), "Generated response"],
    }


def rewrite_query(state: AgentState) -> AgentState:
    """Rewrite query to improve recall when no context was found."""
    rewritten = f'{state["query"]} (with broader context)'
    return {
        **state,
        "query": rewritten,
        "reasoning_steps": [*state.get("reasoning_steps", []), "Rewrote query"],
    }
