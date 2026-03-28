# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG-based document Q&A application. Upload PDFs, ask questions in natural language, get cited answers with source references.

## Stack

- Python 3.11, FastAPI, Pydantic v2
- ChromaDB (vector store, client/server mode via Docker)
- sentence-transformers (all-MiniLM-L6-v2)
- Ollama (local LLM backend)
- Docker Compose for orchestration
- structlog for JSON logging

## Architecture

- `src/docmind/` — src-layout Python package
- `src/docmind/api/` — FastAPI routers (health, upload, query)
- `src/docmind/core/` — Business logic (embedding, retrieval, chat, ingest)
- `src/docmind/app.py` — App factory, wires routers + middleware
- `src/docmind/config.py` — Pydantic Settings, all config via DOCMIND_ env vars
- `tests/` — pytest suite

## Commands

- **Install:** `pip install -e ".[dev]"`
- **Run tests:** `pytest -v`
- **Lint:** `ruff check src/ tests/`
- **Format:** `ruff format src/ tests/`
- **Docker:** `docker compose up --build`
- **Local dev server:** `uvicorn docmind.app:app --reload`

## Rules

- No print() — use structlog
- No secrets in code — use .env with DOCMIND_ prefix
- All endpoints validated with Pydantic models
- File uploads validated: MIME type, magic bytes, size, filename
