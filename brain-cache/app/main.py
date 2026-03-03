"""FastAPI app entry point."""
from fastapi import FastAPI

from app.routers import ingest, query

app = FastAPI(title="Brain Cache API", version="0.1.0")

app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(query.router, prefix="/query", tags=["query"])


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}
