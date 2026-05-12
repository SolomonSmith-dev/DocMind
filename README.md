# DocMind

![CI](https://github.com/SolomonSmith-dev/DocMind/actions/workflows/ci.yml/badge.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Python](https://img.shields.io/badge/python-3.11+-blue.svg)

Intelligent document Q&A system built on Retrieval-Augmented Generation (RAG). Upload PDFs, ask questions in natural language, get cited answers grounded in your actual documents.

Not a ChatGPT wrapper. A real pipeline: parsing, chunking, embedding, vector search, LLM synthesis with page-level citations and prompt injection defense.

## Architecture

```
                         ┌─────────────┐
                         │   Client    │
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │   FastAPI   │
                         │  (uvicorn)  │
                         └──┬───┬───┬──┘
                            │   │   │
              ┌─────────────┘   │   └─────────────┐
              │                 │                 │
     ┌────────▼─────┐  ┌───────▼───────┐  ┌──────▼──────┐
     │   Upload +   │  │   Semantic    │  │     LLM     │
     │  Ingestion   │  │    Search     │  │  Synthesis   │
     │              │  │              │  │              │
     │ • Validate   │  │ • Embed query│  │ • Build prompt│
     │ • Extract    │  │ • ChromaDB   │  │ • Inject ctx │
     │ • Chunk      │  │   top-k      │  │ • Call Ollama│
     │ • Embed      │  │ • Score +    │  │ • Map cites  │
     │ • Store      │  │   rank       │  │              │
     └──────┬───────┘  └───────┬──────┘  └──────┬──────┘
            │                  │                 │
     ┌──────▼──────────────────▼─────────────────▼──────┐
     │                    ChromaDB                       │
     │           (vector store + metadata)               │
     └───────────────────────────────────────────────────┘
```

## Features

- **PDF Upload** with security validation (magic bytes, MIME check, size limit, filename sanitization, UUID storage)
- **Intelligent Chunking** with configurable overlap, preferring paragraph/sentence boundaries
- **Semantic Search** via sentence-transformers (all-MiniLM-L6-v2) + ChromaDB
- **LLM Answer Synthesis** with page-level citations via Ollama
- **Prompt Injection Defense**: document chunks are untrusted input, wrapped in delimiters with instruction pattern filtering
- **Graceful Degradation**: returns raw chunks with citations when LLM is unavailable
- **Structured JSON Logging** with request ID propagation
- **Rate Limiting** on upload (10/min) and ask (30/min) endpoints

## Quick Start

```bash
git clone https://github.com/SolomonSmith-dev/DocMind.git
cd DocMind
cp .env.example .env
docker compose up --build
```

- API docs (Swagger): http://localhost:8000/docs
- Health check: http://localhost:8000/health
- ChromaDB: http://localhost:8001

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service status |
| `POST` | `/api/upload` | Upload a PDF (multipart form) |
| `GET` | `/api/documents` | List all uploaded documents |
| `POST` | `/api/query` | Semantic search. Returns ranked chunks with scores |
| `POST` | `/api/ask` | Full Q&A. Returns cited answer from LLM |

### POST /api/upload

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@document.pdf"
```

### POST /api/ask

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What were the Q3 results?"}'
```

Response:
```json
{
  "answer": "According to the document, Q3 revenue was $10M (page 5).",
  "citations": [
    {"page_number": 5, "source_filename": "report.pdf", "text_snippet": "Revenue was $10M..."}
  ],
  "model": "llama3.2",
  "chunks_used": 3
}
```

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# macOS requires libmagic for file type validation
brew install libmagic

pytest -v           # run tests (48 tests)
ruff check src/     # lint
ruff format src/    # format
```

## Tech Stack

| Tool | Purpose |
|---|---|
| **FastAPI** | Async REST API with auto-generated Swagger docs |
| **Pydantic v2** | Request/response validation on every endpoint |
| **sentence-transformers** | Local embedding generation (all-MiniLM-L6-v2, 384-dim) |
| **ChromaDB** | Vector database for semantic similarity search |
| **Ollama** | Local LLM inference (no API keys needed) |
| **PyMuPDF** | PDF text extraction with page metadata |
| **python-magic** | Magic byte file type validation |
| **slowapi** | Rate limiting per endpoint |
| **structlog** | Structured JSON logging with request IDs |
| **Docker Compose** | One-command startup (API + ChromaDB + Ollama) |
| **GitHub Actions** | CI: lint + test + security audit (pip-audit) |

## Security

- **File upload validation**: Magic byte verification (not just Content-Type header), MIME allowlist, 25MB size limit, UUID storage paths, filename sanitization against path traversal
- **Prompt injection defense**: Document content is untrusted. Retrieved chunks are delimiter-wrapped, instruction patterns are filtered, suspicious content is logged
- **Secrets management**: All config via `.env` with `DOCMIND_` prefix, `.env` in `.gitignore` from first commit
- **Input validation**: Pydantic models on every endpoint. No raw user input reaches filesystem or database
- **Rate limiting**: `/api/upload` (10/min), `/api/ask` (30/min) via slowapi
- **CORS**: Explicit origin allowlist, no wildcard
- **Error handling**: Safe messages to clients, detailed context in structured logs, no stack traces in responses

## Roadmap

Tracked in [open enhancement issues](https://github.com/SolomonSmith-dev/DocMind/issues?q=is%3Aissue+is%3Aopen+label%3Aenhancement).

## License

MIT
