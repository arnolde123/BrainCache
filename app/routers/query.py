"""Query endpoint: semantic search over indexed content."""
from fastapi import APIRouter, HTTPException

from app.agent import compiled_graph
from app.models import AgentQueryResponse, QueryRequest, QueryResponse, SearchResult
from app.services import embeddings, pinecone

router = APIRouter()


def _match_to_search_result(match: dict) -> SearchResult:
    """Map Pinecone match (id, score, metadata) to SearchResult."""
    meta = match.get("metadata") or {}
    text = meta.get("original_text") or meta.get("text") or ""
    source = meta.get("source_id") or meta.get("source")
    return SearchResult(
        id=match.get("id", ""),
        text=text,
        score=float(match.get("score") or 0.0),
        source=source,
    )


@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Run a semantic search query against the vector index.
    """
    try:
        query_vector = await embeddings.embed_query(request.query)
        raw_results = await pinecone.query(
            vector=query_vector,
            top_k=request.top_k,
        )
        results = [_match_to_search_result(m) for m in raw_results]
        return QueryResponse(
            query=request.query,
            results=results,
            total_found=len(results),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/agent", response_model=AgentQueryResponse)
async def agent_query(request: QueryRequest):
    """
    Run query through the LangGraph agent workflow.
    """
    try:
        final_state = await compiled_graph.ainvoke(
            {
                "question": request.query,
                "original_question": request.query,
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

        return AgentQueryResponse(
            query=request.query,
            answer=final_state.get("answer") or "",
            sources=final_state.get("sources", []),
            was_query_rewritten=final_state.get("was_query_rewritten", False),
            rewritten_questions=final_state.get("rewritten_questions", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
