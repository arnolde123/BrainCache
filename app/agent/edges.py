"""Conditional routing logic for the BrainCache LangGraph."""

from app.agent.state import AgentState


def route_after_grading(state: AgentState) -> str:
    """Route to rewrite when context is missing, otherwise generate."""
    return "rewrite_query" if state.get("needs_more_context", False) else "generate"
