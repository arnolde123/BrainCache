"""Query endpoint: semantic search over indexed content."""
from fastapi import APIRouter, HTTPException

from app.models import QueryRequest, QueryResponse
from app.services import embeddings, pinecone

router = APIRouter()


@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Run a semantic search query against the vector index.
    """
    try:
        query_vector = await embeddings.embed_query(request.query)
        results = await pinecone.query(
            vector=query_vector,
            top_k=request.top_k,
        )
        return QueryResponse(
            query=request.query,
            results=results,
            count=len(results),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
