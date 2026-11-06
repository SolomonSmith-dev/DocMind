# DocMind

RAG document Q&A: upload PDFs, ask natural-language questions, get cited answers.

## Task Routing

| Task Type | Go To | Description |
|-----------|-------|-------------|
| Ingest / index docs | `src/` | Chunking, embeddings, store wiring |
| Answer path | `src/` | Retrieval + generate_answer |
| Web UI | `web/` | Front-end for upload and chat |
| Tests | `tests/` | Unit and integration |
| Run stack | `docker-compose.yml` | Local Docker services |
| Graph Q&A | `graphify query` | Codebase navigation |

## Shared Resources

| Resource | Location | Contains |
|----------|----------|----------|
| Deps | `pyproject.toml` | Python package + tooling |
| Graph | `graphify-out/` | Code graph |
| Ops | `~/Projects/personal-tech-ops` | Homelab / Dual Develop |
