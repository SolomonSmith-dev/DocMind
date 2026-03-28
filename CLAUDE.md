# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG-based document Q&A. Upload PDFs, ask questions, get cited answers with page references. Full pipeline: ingestion → embedding → vector search → LLM synthesis.

## Stack

Python 3.11, FastAPI, Pydantic v2, sentence-transformers (all-MiniLM-L6-v2), ChromaDB, Ollama, PyMuPDF, slowapi, structlog, Docker Compose.

## Architecture

```
src/docmind/
  app.py              # App factory — routers, middleware, CORS, rate limiting
  config.py           # Pydantic Settings, all config via DOCMIND_ env vars
  logging.py          # structlog JSON setup
  middleware.py        # Request ID middleware
  api/
    health.py          # GET /health
    upload.py          # POST /api/upload, GET /api/documents
    query.py           # POST /api/query (semantic search)
    ask.py             # POST /api/ask (full RAG Q&A)
  core/
    security.py        # File validation: magic bytes, MIME, size, sanitization
    ingest.py          # PyMuPDF extraction + chunking with overlap
    embedding.py       # sentence-transformers model loading + encoding
    retrieval.py       # ChromaDB store/query with cosine similarity
    chat.py            # Prompt building, injection defense, Ollama calls
    store.py           # In-memory document metadata store
tests/                 # 48 pytest tests
```

## Commands

- **Install:** `pip install -e ".[dev]"`
- **Tests:** `pytest -v`
- **Lint:** `ruff check src/ tests/`
- **Format:** `ruff format src/ tests/`
- **Docker:** `docker compose up --build`
- **Dev server:** `uvicorn docmind.app:app --reload`

## Environment

All config via `.env` with `DOCMIND_` prefix. Key vars: `DOCMIND_CHROMA_HOST`, `DOCMIND_OLLAMA_BASE_URL`, `DOCMIND_OLLAMA_MODEL`.

## Rules

- No print() — use structlog
- No secrets in code — .env with DOCMIND_ prefix
- All endpoints validated with Pydantic models
- File uploads: magic bytes + MIME + size + filename sanitization
- Document chunks are untrusted — delimiter-wrapped before LLM injection
