# DocMind

RAG-based document Q&A application. Upload PDFs, ask questions in natural language, get cited answers with source references.

## Architecture

> Architecture diagram will be added before v1.0 ships.

**Components:**
- **FastAPI Backend** — REST API for document upload and querying
- **ChromaDB** — Vector database for document embeddings
- **sentence-transformers** — Local embedding generation (all-MiniLM-L6-v2)
- **Ollama** — Local LLM backend for answer generation

## Quick Start

```bash
cp .env.example .env
docker compose up --build
```

- API: http://localhost:8000
- Health check: http://localhost:8000/health
- API docs (Swagger): http://localhost:8000/docs
- ChromaDB: http://localhost:8001

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -v
ruff check src/ tests/
```

## Tech Stack

| Tool | Purpose |
|---|---|
| **FastAPI** | Async REST API framework |
| **Pydantic v2** | Request/response validation |
| **ChromaDB** | Vector database for semantic search |
| **sentence-transformers** | Embedding generation |
| **Ollama** | Local LLM inference |
| **structlog** | Structured JSON logging |
| **Docker Compose** | Container orchestration |
| **GitHub Actions** | CI pipeline (lint + test + security audit) |

## Project Status

Under active development (Sprint 1).

- [x] Project scaffold + CI pipeline
- [ ] PDF upload with file validation
- [ ] Document chunking + embedding pipeline
- [ ] Semantic search retrieval
- [ ] LLM-powered Q&A with citations
- [ ] React frontend

## License

MIT
