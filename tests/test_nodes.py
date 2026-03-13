"""Unit tests for LangGraph node functions."""

import asyncio

from app.agent import nodes


def test_retrieve_node_populates_search_results(monkeypatch):
    """Retrieve node should call embedding + Pinecone services."""

    async def fake_embed_query(_query: str):
        return [0.1, 0.2]

    async def fake_pinecone_query(vector: list[float], top_k: int = 5):
        assert vector == [0.1, 0.2]
        assert top_k == 3
        return [{"id": "doc-1", "score": 0.9, "metadata": {"original_text": "hello"}}]

    monkeypatch.setattr(nodes.embeddings, "embed_query", fake_embed_query)
    monkeypatch.setattr(nodes.pinecone, "query", fake_pinecone_query)

    state = {
        "query": "hello",
        "top_k": 3,
        "search_results": [],
        "answer": "",
        "needs_more_context": False,
        "reasoning_steps": [],
    }
    out = asyncio.run(nodes.retrieve(state))

    assert len(out["search_results"]) == 1
    assert "Retrieved candidate chunks" in out["reasoning_steps"]


def test_grade_documents_sets_needs_more_context():
    state = {
        "query": "hello",
        "top_k": 5,
        "search_results": [],
        "answer": "",
        "needs_more_context": False,
        "reasoning_steps": [],
    }
    out = nodes.grade_documents(state)
    assert out["needs_more_context"] is True


def test_generate_sets_placeholder_answer():
    state = {
        "query": "hello",
        "top_k": 5,
        "search_results": [{"id": "doc-1", "score": 0.9, "metadata": {}}],
        "answer": "",
        "needs_more_context": False,
        "reasoning_steps": [],
    }
    out = nodes.generate(state)
    assert out["answer"] != ""
    assert "Generated response" in out["reasoning_steps"]


def test_rewrite_query_appends_broader_context():
    state = {
        "query": "capital of france",
        "top_k": 5,
        "search_results": [],
        "answer": "",
        "needs_more_context": True,
        "reasoning_steps": [],
    }
    out = nodes.rewrite_query(state)
    assert out["query"].endswith("(with broader context)")
