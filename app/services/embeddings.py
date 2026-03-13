"""OpenAI embedding calls."""
import os

# Use OpenAI-compatible client; install openai and set OPENAI_API_KEY in .env
from openai import AsyncOpenAI

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        _client = AsyncOpenAI(api_key=api_key)
    return _client


async def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Get embeddings for a list of texts."""
    client = _get_client()
    resp = await client.embeddings.create(input=texts, model=model)
    return [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]


async def embed_query(query: str, model: str = "text-embedding-3-small") -> list[float]:
    """Get a single embedding for a query string."""
    vectors = await embed_texts([query], model=model)
    return vectors[0]
