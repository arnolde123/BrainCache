"""Vector DB logic using Pinecone."""
import os

from pinecone import Pinecone

# Type: (id, vector, metadata). Metadata should include source_id, original_text, chunk_index.
UpsertRecord = tuple[str, list[float], dict]

_index = None
_BATCH_SIZE = 100  # Pinecone-friendly batch size


def _get_index():
    global _index
    if _index is None:
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME", "brain-cache")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not set")
        pc = Pinecone(api_key=api_key)
        _index = pc.Index(index_name)
    return _index


def _sanitize_metadata(metadata: dict) -> dict:
    """Ensure metadata values are Pinecone-compatible (str, int, float, bool)."""
    out: dict = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


async def upsert(records: list[UpsertRecord]) -> int:
    """
    Upsert vectors into the index.

    Each record is (id, vector, metadata). Metadata should include at least
    source_id, original_text (chunk text, truncated if needed), and chunk_index.
    Records are sent in batches to respect API limits.
    """
    if not records:
        return 0
    index = _get_index()
    total = 0
    for i in range(0, len(records), _BATCH_SIZE):
        batch = records[i : i + _BATCH_SIZE]
        payload = [
            {
                "id": rid,
                "values": vec,
                "metadata": _sanitize_metadata(meta),
            }
            for rid, vec, meta in batch
        ]
        index.upsert(vectors=payload)
        total += len(payload)
    return total


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
