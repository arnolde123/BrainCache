"""Integration test for compiled LangGraph workflow."""

import asyncio

from app.agent.graph import compiled_graph
from app.services import embeddings, pinecone


def test_compiled_graph_returns_expected_shape(monkeypatch):
    async def fake_embed_query(_query: str):
        return [0.01, 0.02]

    async def fake_pinecone_query(vector: list[float], top_k: int = 5):
        assert vector == [0.01, 0.02]
        return [{"id": "doc-1", "score": 0.88, "metadata": {"original_text": "sample"}}]

    monkeypatch.setattr(embeddings, "embed_query", fake_embed_query)
    monkeypatch.setattr(pinecone, "query", fake_pinecone_query)

    result = asyncio.run(
        compiled_graph.ainvoke(
            {
                "query": "sample question",
                "top_k": 5,
                "search_results": [],
                "answer": "",
                "needs_more_context": False,
                "reasoning_steps": [],
            }
        )
    )

    assert result["query"] == "sample question"
    assert isinstance(result["search_results"], list)
    assert isinstance(result["answer"], str)
    assert isinstance(result["reasoning_steps"], list)
