"""Ingest endpoint: upload and index documents."""
from fastapi import APIRouter, HTTPException

from app.models import IngestRequest, IngestResponse
from app.services import chunking, embeddings, pinecone, s3

router = APIRouter()

# Pinecone metadata size limit; truncate chunk text for metadata
METADATA_TEXT_MAX_LEN = 1000


@router.post("", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """
    Ingest a document: store in S3, chunk with overlap, embed in one batch, and index in Pinecone.
    """
    try:
        # 1. Store raw content in S3; keep key for Pinecone metadata (re-index, audit)
        s3_key = await s3.upload_content(
            request.source_id, request.content, request.metadata or {}
        )

        # 2. Chunk with overlap (e.g. 300 words, 50-word overlap, sentence-aware)
        chunks = chunking.chunk_text(
            request.content,
            chunk_size=300,
            overlap=50,
            respect_sentences=True,
        )
        if not chunks:
            return IngestResponse(
                source_id=request.source_id,
                chunks_stored=0,
                message="No chunks produced from content",
            )

        # 3. Embed all chunks in one batch request
        vectors = await embeddings.embed_texts(chunks)

        # 4. Build records: (id, vector, metadata) with stable IDs and rich metadata
        extra = request.metadata or {}
        records = []
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            record_id = f"{request.source_id}#{i}"
            metadata = {
                "source_id": request.source_id,
                "original_text": chunk[:METADATA_TEXT_MAX_LEN],
                "chunk_index": i,
                "s3_key": s3_key,
                **{k: str(v) for k, v in extra.items()},
            }
            records.append((record_id, vec, metadata))

        # 5. Upsert to Pinecone (batched inside pinecone.upsert)
        chunk_count = await pinecone.upsert(records)

        return IngestResponse(
            source_id=request.source_id,
            chunks_stored=chunk_count,
            message="Ingestion complete",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
