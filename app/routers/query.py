"""Query endpoint: semantic search over indexed content."""
from fastapi import APIRouter, HTTPException

from app.models import QueryRequest, QueryResponse, SearchResult
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
