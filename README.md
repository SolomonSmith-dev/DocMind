# DocMind

[![CI](https://github.com/SolomonSmith-dev/DocMind/actions/workflows/ci.yml/badge.svg)](https://github.com/SolomonSmith-dev/DocMind/actions/workflows/ci.yml) [![E2E](https://github.com/SolomonSmith-dev/DocMind/actions/workflows/e2e.yml/badge.svg)](https://github.com/SolomonSmith-dev/DocMind/actions/workflows/e2e.yml) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Python](https://img.shields.io/badge/python-3.11+-blue.svg)

Upload a PDF, ask a question, get an answer that names the page it came from. Runs on your own machine: Ollama for the model, ChromaDB for the vectors, no API key.

Most RAG demos stop at "retrieve, then prompt." This one treats the document as hostile input and the answer as something a reader has to be able to check:

- **Page-level citations.** Every chunk keeps its 1-indexed page number from PyMuPDF through ChromaDB to the response; `/api/ask` returns `citations[].page_number` alongside the answer (`src/docmind/core/ingest.py`, `src/docmind/core/chat.py`).
- **Prompt injection defense.** Retrieved chunks are untrusted. Each is wrapped in `<DOCUMENT_CHUNK>` delimiters, six instruction patterns ("ignore previous instructions", "system prompt:", `<system>`...) are rewritten to `[FILTERED]` and logged, and the system prompt refuses instructions found inside chunks. Prior user turns in a session get the same treatment (`src/docmind/core/chat.py`).
- **Rate limiting and auth.** slowapi caps `/api/upload` at 10/min and `/api/ask` at 30/min per client IP. Every `/api/*` route except `/health` requires an HS256 JWT; passwords are bcrypt-hashed (`src/docmind/api/ask.py`, `src/docmind/api/upload.py`, `src/docmind/core/auth.py`).
- **Confidence score.** Each answer carries a weighted score (0.5 top-chunk similarity, 0.3 retrieval coverage, 0.2 LLM self-report) and a `low_confidence` flag below 0.4. If Ollama is down, the endpoint returns the cited passages instead of a 500.

## Evidence

| Claim | Check |
|---|---|
| 113 tests, all passing | `pytest -q` prints `113 passed` (Python 3.11, run 2026-11-03). Files under `tests/`: ask 14, auth 11, confidence 22, formats 8, health 2, ingest 6, rerank 9, retrieval 8, security 13, session 15, upload 5 |
| CI on every push and PR | `.github/workflows/ci.yml`: `ruff check` + `ruff format --check`, `pytest -v`, `pip-audit` |
| Browser E2E | `.github/workflows/e2e.yml` (manual dispatch): ChromaDB service + live API + Playwright, 2 specs in `web/e2e/upload-query.spec.ts` (login; upload PDF then ask) |
| Not measured | Retrieval quality. There is no eval set yet; nothing here claims an accuracy number |

## How to run

Everything runs locally. Two paths.

**Docker Compose** (API + ChromaDB + Ollama + React UI):

```bash
git clone https://github.com/SolomonSmith-dev/DocMind.git && cd DocMind
cp .env.example .env
docker compose up --build
docker compose exec ollama ollama pull llama3.2   # first run only
```

Open http://localhost:3000, sign in as `demo` / `demo1234` (hardcoded demo user in `src/docmind/core/auth.py`). Swagger lives at http://localhost:8000/docs.

**Without Docker** (macOS shown; same shape on Linux):

```bash
brew install libmagic ollama
ollama serve &
ollama pull llama3.2
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
chroma run --path ./chroma_data --port 8001 &          # optional, see note
DOCMIND_CHROMA_PORT=8001 uvicorn docmind.app:app --reload
cd web && npm install && npm run dev                   # UI at http://localhost:5173
```

If no ChromaDB server answers, the API logs a warning and falls back to an in-process ephemeral store, so you can try the pipeline with nothing but Ollama running. Data in that mode is gone on restart.

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

Stack: Python 3.11, FastAPI, Pydantic v2, sentence-transformers (all-MiniLM-L6-v2, 384-dim), ChromaDB, Ollama, PyMuPDF, python-docx, python-magic, slowapi, PyJWT + bcrypt, structlog. React 18 + Vite frontend in `web/`. Optional cross-encoder re-ranking (`DOCMIND_ENABLE_RERANK=true`) and multi-turn sessions (`session_id` on `/api/ask`).

## API

All `/api/*` routes need `Authorization: Bearer <token>`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service status (no auth) |
| `POST` | `/api/auth/token` | Exchange username/password for a JWT (60 min) |
| `POST` | `/api/upload` | Upload a PDF, DOCX or TXT (multipart form, 25 MB max) |
| `GET` | `/api/documents` | List uploaded documents |
| `POST` | `/api/query` | Semantic search only. Returns ranked chunks with scores and page numbers |
| `POST` | `/api/ask` | Full Q&A. Returns cited answer, confidence, session id |

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"demo1234"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')

curl -X POST http://localhost:8000/api/upload -H "Authorization: Bearer $TOKEN" -F "file=@document.pdf"

curl -X POST http://localhost:8000/api/ask -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d '{"question": "What were the Q3 results?"}'
```

Response shape for `/api/ask`:

```json
{
  "answer": "According to the document, Q3 revenue was $10M (page 5).",
  "citations": [
    {"page_number": 5, "source_filename": "report.pdf", "text_snippet": "Revenue was $10M..."}
  ],
  "model": "llama3.2",
  "chunks_used": 3,
  "confidence": 0.78,
  "low_confidence": false,
  "session_id": "..."
}
```

## Development

```bash
pip install -e ".[dev]"
brew install libmagic        # macOS only, for python-magic
pytest -q                    # 113 tests
ruff check src/ tests/
ruff format --check src/ tests/
cd web && npm run test:e2e   # needs the API on :8000
```

## Security notes

- **Uploads**: magic-byte detection (not the Content-Type header), MIME allowlist, DOCX verified as a ZIP containing `word/document.xml`, 25 MB cap, filename sanitization, UUID storage paths (`src/docmind/core/security.py`).
- **Config**: everything via `DOCMIND_`-prefixed env vars; `.env` has been gitignored since the first commit. The default JWT secret is for development; set `DOCMIND_JWT_SECRET_KEY` anywhere else.
- **Known gaps**: in-memory user and session stores, so a restart logs everyone out. Rate limits are keyed by client IP. The injection filter is regex-based and will not catch everything; it raises the cost of an attack, it does not end it.

## Roadmap

Tracked in [open enhancement issues](https://github.com/SolomonSmith-dev/DocMind/issues?q=is%3Aissue+is%3Aopen+label%3Aenhancement).

## License

MIT
