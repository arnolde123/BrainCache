from pydantic import BaseModel, Field

# Define a Search Result model 
class SearchResult(BaseModel):
    id: str
    text: str
    score: float
    source: str | None = None

# Define the main Request/Response models
class IngestRequest(BaseModel):
    """Schema for ingesting new documents."""
    # Content is the text of the document
    content: str = Field(..., min_length=1, max_length=10000)
    source_id: str = Field(..., description="Unique identifier for the document")
    tags: list[str] = Field(default_factory=list)
    metadata: dict | None = None

class IngestResponse(BaseModel):
    """Schema for ingestion response."""
    source_id: str
    chunks_stored: int
    message: str = "Ingestion successful"

class QueryRequest(BaseModel):
    """Schema for natural language search."""
    query: str = Field(..., min_length=1)
    # top k is the number of results to return
    top_k: int = Field(default=5, ge=1, le=20)

class QueryResponse(BaseModel):
    """Schema for query response."""
    query: str
    results: list[SearchResult]
    total_found: int


class AgentQueryResponse(BaseModel):
    """Schema for LangGraph-powered query response."""

    query: str
    answer: str
    sources: list[str]
    was_query_rewritten: bool = False
    rewritten_questions: list[str] = Field(default_factory=list)
