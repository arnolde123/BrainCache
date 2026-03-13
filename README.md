# BrainCache

A personal knowledge base project that ingests documents, stores them in a vector database, and answers questions using a LangGraph retrieval-augmented generation agent. I'm building this with FastAPI, LangGraph, OpenAI, Pinecone, and AWS S3 & EC2.

---

## Project Status

BrainCache is currently being developed. The ingestion pipeline and semantic search endpoints are fully operational. The LangGraph RAG agent (retrieve, grade, rewrite, generate) is implemented and wired to the API. 

A Next.js frontend, observability/tracing layer, and more accepted document types are planned for future iterations.

---

## Installation and Setup Instructions

### Prerequisites

- Python 3.11 or higher
- An OpenAI API key
- A Pinecone account and index
- An AWS account with an S3 bucket (and optionally AWS SSM Parameter Store for production secrets)

### Clone the repository

```bash
git clone https://github.com/your-username/BrainCache.git
cd BrainCache
```

### Configure environment variables

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

Open `.env` and set these values:

```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=brain-cache
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
S3_BUCKET=...
```

For local development, also set:

```
ENV=local
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### To run the test suite

```bash
pytest
```

Or with less output:

```bash
pytest -q
```

### To start the server locally

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

Interactive API docs are at `http://localhost:8000/docs`.

### To start with Docker

```bash
docker-compose up --build
```

This runs the API at `http://localhost:8000` with hot reload enabled via a volume mount.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/ingest` | Ingest a document (chunk, embed, store in S3 and Pinecone) |
| POST | `/query` | Semantic search (direct vector similarity) |
| POST | `/query/agent` | Agentic RAG query (retrieve, grade, rewrite, generate) |

---

## Reflection

I started this project because I hate forgetting documents I saved or bookmarked for later, so I'm building a "Second Brain" to store all this info and allow me to query it when I have questions. 

Implementing a RAG system from scratc has been fun and helped me learn how each layer of the stack works. The general flow is text gets turned into vectors, which are then stored and retrived, and then an LLM can reason over the retrieved context. 

Instead of blindly passing in chucks of data into an LLM, I chose to implement LangChain. This way, I can retrive documents, grade them individually for relevance, and rewrite the query if no related info can be found. This loop will happen until it hits the retry budget (3) to prevent infinite loops. This is why I chose LangGraph over several LangChain chains. 

I chose Pinecone for vector storage and OpenAI's `text-embedding-3-small` model because of its accuracy-to-cost ratio. S3 stores the raw document source alongside the Pinecone metadata so documents can be re-indexed or audited without data loss.

The project is structured so that the agent internals (`app/agent/`) are fully isolated from the routing and service layers. This means the LangGraph graph can be upgraded, replaced, or traced independently without touching the API surface that a frontend will eventually consume.
