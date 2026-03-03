"""Pydantic schemas for request/response models."""
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Request body for document ingestion."""

    source_id: str = Field(..., description="Unique identifier for the source")
    content: str = Field(..., description="Text content to ingest")
    metadata: dict | None = Field(default=None, description="Optional metadata")


class IngestResponse(BaseModel):
    """Response after successful ingestion."""

    source_id: str
    chunks_ingested: int
    message: str = "Ingestion complete"


class QueryRequest(BaseModel):
    """Request body for semantic search query."""

    query: str = Field(..., description="Natural language query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")


class QueryResponse(BaseModel):
    """Response with retrieved chunks and scores."""

    query: str
    results: list[dict]
    count: int
