"""State contract shared across LangGraph nodes."""

from typing import TypedDict


class AgentState(TypedDict):
    """Typed state passed between graph nodes."""

    query: str
    top_k: int
    search_results: list[dict]
    answer: str
    needs_more_context: bool
    reasoning_steps: list[str]
