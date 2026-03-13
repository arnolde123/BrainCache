"""Unit tests for LangGraph node functions."""

import os
import sys
import asyncio
from types import SimpleNamespace

# Ensure project root (containing the `app` package) is on sys.path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.agent import nodes


class _FakePrompt:
    def __or__(self, other):
        return other


def test_retrieve_node_builds_documents_and_increments_attempts(monkeypatch):
    """Retrieve node should map Pinecone results into LangChain Documents."""

    async def fake_embed_query(_query: str):
        return [0.1, 0.2]

    async def fake_pinecone_query(vector: list[float], top_k: int):
        assert vector == [0.1, 0.2]
        assert top_k == 5
        return [
            {
                "id": "doc-1",
                "score": 0.9,
                "metadata": {
                    "original_text": "hello",
                    "source_id": "note-1",
                    "s3_key": "k1",
                    "chunk_index": 2,
                },
            }
        ]

    monkeypatch.setattr(nodes.embeddings, "embed_query", fake_embed_query)
    monkeypatch.setattr(nodes.pinecone, "query", fake_pinecone_query)

    state = {
        "question": "hello",
        "original_question": "hello",
        "documents": [],
        "all_docs_relevant": False,
        "retrieval_attempts": 0,
        "max_attempts": 3,
        "answer": None,
        "sources": [],
        "was_query_rewritten": False,
        "rewritten_questions": [],
    }
    out = asyncio.run(nodes.retrieve(state))

    assert len(out["documents"]) == 1
    assert out["documents"][0].page_content == "hello"
    assert out["documents"][0].metadata["source"] == "note-1"
    assert out["retrieval_attempts"] == 1


def test_grade_documents_filters_irrelevant(monkeypatch):
    class FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            self._schema = None

        def with_structured_output(self, schema):
            self._schema = schema
            return self

        def invoke(self, payload):
            score = "yes" if "relevant" in payload["document"] else "no"
            return self._schema(score=score)

    monkeypatch.setattr(nodes, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(nodes.ChatPromptTemplate, "from_template", lambda *_: _FakePrompt())

    state = {
        "question": "What is RAG?",
        "original_question": "What is RAG?",
        "documents": [
            nodes.Document(page_content="this is relevant context", metadata={"source": "a"}),
            nodes.Document(page_content="completely unrelated text", metadata={"source": "b"}),
            nodes.Document(page_content="another relevant passage", metadata={"source": "c"}),
        ],
        "all_docs_relevant": False,
        "retrieval_attempts": 1,
        "max_attempts": 3,
        "answer": None,
        "sources": [],
        "was_query_rewritten": False,
        "rewritten_questions": [],
    }

    out = nodes.grade_documents(state)
    assert len(out["documents"]) == 2
    assert out["all_docs_relevant"] is True


def test_rewrite_query_updates_question_and_history(monkeypatch):
    class FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, _payload):
            return SimpleNamespace(content="best retrieval query")

    monkeypatch.setattr(nodes, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(nodes.ChatPromptTemplate, "from_template", lambda *_: _FakePrompt())

    state = {
        "question": "initial failed query",
        "original_question": "What is vector search?",
        "documents": [],
        "all_docs_relevant": False,
        "retrieval_attempts": 2,
        "max_attempts": 3,
        "answer": None,
        "sources": [],
        "was_query_rewritten": False,
        "rewritten_questions": ["vector embeddings meaning"],
    }

    out = nodes.rewrite_query(state)
    assert out["question"] == "best retrieval query"
    assert out["was_query_rewritten"] is True
    assert out["rewritten_questions"][-1] == "best retrieval query"


def test_generate_returns_answer_and_unique_sources(monkeypatch):
    class FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, _payload):
            return SimpleNamespace(content="Grounded answer.\nSources: [note-1, note-2]")

    monkeypatch.setattr(nodes, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(nodes.ChatPromptTemplate, "from_template", lambda *_: _FakePrompt())

    state = {
        "question": "question",
        "original_question": "question",
        "documents": [
            nodes.Document(page_content="doc1", metadata={"source": "note-1"}),
            nodes.Document(page_content="doc2", metadata={"source": "note-2"}),
            nodes.Document(page_content="doc3", metadata={"source": "note-1"}),
        ],
        "all_docs_relevant": True,
        "retrieval_attempts": 1,
        "max_attempts": 3,
        "answer": None,
        "sources": [],
        "was_query_rewritten": False,
        "rewritten_questions": [],
    }

    out = nodes.generate(state)
    assert out["answer"].startswith("Grounded answer")
    assert out["sources"] == ["note-1", "note-2"]
