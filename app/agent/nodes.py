"""LangGraph node functions for BrainCache agent flow."""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.agent.prompts import GENERATE_PROMPT, GRADE_DOCUMENTS_PROMPT, REWRITE_QUERY_PROMPT
from app.agent.state import AgentState
from app.services import embeddings, pinecone


class GradeScore(BaseModel):
    """Structured score output from the grading model."""

    score: str = Field(description="Relevance score: 'yes' or 'no'")


async def retrieve(state: AgentState) -> dict:
    """
    Query Pinecone for relevant docs and convert matches into Documents.

    Returns only modified keys so LangGraph can merge state safely.
    """
    question = state["question"]
    embedding = await embeddings.embed_query(question)
    raw_results = await pinecone.query(vector=embedding, top_k=5)

    documents = [
        Document(
            page_content=(result.get("metadata") or {}).get("original_text", ""),
            metadata={
                "source": (result.get("metadata") or {}).get("source_id", "unknown"),
                "s3_key": (result.get("metadata") or {}).get("s3_key", ""),
                "score": result.get("score", 0.0),
                "chunk_index": (result.get("metadata") or {}).get("chunk_index", 0),
            },
        )
        for result in raw_results
    ]

    return {
        "documents": documents,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }


def grade_documents(state: AgentState) -> dict:
    """
    Grade each retrieved document for question relevance.

    Keep only relevant docs and set gating flag for downstream routing.
    """
    question = state["question"]
    documents = state.get("documents", [])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(GradeScore)
    prompt = ChatPromptTemplate.from_template(GRADE_DOCUMENTS_PROMPT)
    grader_chain = prompt | structured_llm

    relevant_docs = []
    for doc in documents:
        result = grader_chain.invoke(
            {
                "document": doc.page_content,
                "question": question,
            }
        )
        if result.score.lower() == "yes":
            relevant_docs.append(doc)

    return {
        "documents": relevant_docs,
        "all_docs_relevant": len(relevant_docs) >= 2,
    }


def rewrite_query(state: AgentState) -> dict:
    """Rewrite the query when retrieval quality is insufficient."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    prompt = ChatPromptTemplate.from_template(REWRITE_QUERY_PROMPT)
    previous_rewrites = state.get("rewritten_questions", [])

    rewritten = (prompt | llm).invoke(
        {
            "original_question": state["original_question"],
            "current_question": state["question"],
            "attempt": state["retrieval_attempts"],
            "previous_rewrites": ", ".join(previous_rewrites) if previous_rewrites else "none",
        }
    )
    new_question = rewritten.content.strip()

    return {
        "question": new_question,
        "was_query_rewritten": True,
        "rewritten_questions": [*previous_rewrites, new_question],
    }


def generate(state: AgentState) -> dict:
    """Generate a grounded final answer from relevant context documents."""
    question = state["question"]
    documents = state.get("documents", [])

    context = "\n\n".join(
        [
            f"[Document {i + 1}] (Source: {doc.metadata.get('source', 'unknown')})\n{doc.page_content}"
            for i, doc in enumerate(documents)
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_template(GENERATE_PROMPT)
    response = (prompt | llm).invoke({"context": context, "question": question})

    sources: list[str] = []
    seen = set()
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        if source != "unknown" and source not in seen:
            seen.add(source)
            sources.append(source)

    return {
        "answer": response.content,
        "sources": sources,
    }
