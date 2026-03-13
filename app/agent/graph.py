"""Graph assembly for BrainCache LangGraph agent."""

from langgraph.graph import END, StateGraph

from app.agent.edges import route_after_grading
from app.agent.nodes import generate, grade_documents, retrieve, rewrite_query
from app.agent.state import AgentState

workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {
        "rewrite_query": "rewrite_query",
        "generate": "generate",
    },
)
workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("generate", END)

compiled_graph = workflow.compile()
