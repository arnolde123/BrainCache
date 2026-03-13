"""Integration test for compiled LangGraph workflow."""

import os
import sys
import asyncio
from types import SimpleNamespace

# Ensure project root (containing the `app` package) is on sys.path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.agent import nodes
from app.agent.graph import compiled_graph
from app.services import embeddings, pinecone


def test_compiled_graph_returns_expected_shape(monkeypatch):
    class FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            self._schema = None

        def with_structured_output(self, schema):
            self._schema = schema
            return self

        def invoke(self, payload):
            if self._schema is not None:
                return self._schema(score="yes")
            return SimpleNamespace(content="Answer from context.\nSources: [note-1, note-2]")

    class FakePrompt:
        def __or__(self, other):
            return other

    async def fake_embed_query(_question: str):
        return [0.01, 0.02]

    async def fake_pinecone_query(vector: list[float], top_k: int = 5):
        assert vector == [0.01, 0.02]
        assert top_k == 5
        return [
            {
                "id": "doc-1",
                "score": 0.88,
                "metadata": {"original_text": "sample a", "source_id": "note-1"},
            },
            {
                "id": "doc-2",
                "score": 0.87,
                "metadata": {"original_text": "sample b", "source_id": "note-2"},
            },
        ]

    monkeypatch.setattr(embeddings, "embed_query", fake_embed_query)
    monkeypatch.setattr(pinecone, "query", fake_pinecone_query)
    monkeypatch.setattr(nodes, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(nodes.ChatPromptTemplate, "from_template", lambda *_: FakePrompt())

    result = asyncio.run(
        compiled_graph.ainvoke(
            {
                "question": "sample question",
                "original_question": "sample question",
                "documents": [],
                "all_docs_relevant": False,
                "retrieval_attempts": 0,
                "max_attempts": 3,
                "answer": None,
                "sources": [],
                "was_query_rewritten": False,
                "rewritten_questions": [],
            }
        )
    )

    assert result["question"] == "sample question"
    assert isinstance(result["documents"], list)
    assert isinstance(result["answer"], str)
    assert result["sources"] == ["note-1", "note-2"]
    assert result["was_query_rewritten"] is False
