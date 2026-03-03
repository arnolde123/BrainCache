"""Ingest endpoint: upload and index documents."""
from fastapi import APIRouter, HTTPException

from app.models import IngestRequest, IngestResponse
from app.services import embeddings, pinecone, s3

router = APIRouter()


@router.post("", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """
    Ingest a document: store in S3, chunk, embed, and index in Pinecone.
    """
    try:
        # Store raw content in S3
        await s3.upload_content(request.source_id, request.content, request.metadata or {})

        # Chunk and embed (placeholder: single chunk for now)
        chunks = [request.content]
        vectors = await embeddings.embed_texts(chunks)

        # Upsert to Pinecone
        chunk_count = await pinecone.upsert(request.source_id, chunks, vectors, request.metadata or {})

        return IngestResponse(
            source_id=request.source_id,
            chunks_ingested=chunk_count,
            message="Ingestion complete",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
