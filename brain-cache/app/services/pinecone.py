"""Vector DB logic using Pinecone."""
import os
from uuid import uuid4

from pinecone import Pinecone

_index = None


def _get_index():
    global _index
    if _index is None:
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME", "second-brain")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not set")
        pc = Pinecone(api_key=api_key)
        _index = pc.Index(index_name)
    return _index


async def upsert(
    source_id: str,
    chunks: list[str],
    vectors: list[list[float]],
    metadata: dict,
) -> int:
    """Upsert chunk vectors into the index."""
    index = _get_index()
    records = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        records.append({
            "id": str(uuid4()),
            "values": vec,
            "metadata": {
                "source_id": source_id,
                "chunk_index": i,
                "text": chunk[:1000],  # Pinecone metadata size limit
                **{k: str(v) for k, v in metadata.items()},
            },
        })
    index.upsert(vectors=records)
    return len(records)


async def query(vector: list[float], top_k: int = 5) -> list[dict]:
    """Query the index by vector; return matches with metadata."""
    index = _get_index()
    result = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return [
        {
            "id": m.id,
            "score": m.score,
            "metadata": m.metadata or {},
        }
        for m in (result.matches or [])
    ]
