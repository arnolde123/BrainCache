"""FastAPI app entry point."""
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI

from app.config import get_secrets
from app.routers import ingest, query


@asynccontextmanager
async def lifespan(app: FastAPI):
    secrets = get_secrets()
    app.state.secrets = secrets
    yield


app = FastAPI(title="Brain Cache API", version="0.1.0", lifespan=lifespan)

app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(query.router, prefix="/query", tags=["query"])


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}
