"""Conditional routing logic for the BrainCache LangGraph."""

from app.agent.state import AgentState


def route_after_grading(state: AgentState) -> str:
    """Route to generate or rewrite based on quality and retry budget."""
    if state.get("all_docs_relevant", False):
        return "generate"
    if state.get("retrieval_attempts", 0) >= state.get("max_attempts", 3):
        return "generate"
    return "rewrite_query"
