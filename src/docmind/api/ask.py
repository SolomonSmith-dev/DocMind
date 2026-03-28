import structlog
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from docmind.core.chat import generate_answer
from docmind.core.retrieval import retrieve_chunks

router = APIRouter()
logger = structlog.get_logger()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    document_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class CitationInfo(BaseModel):
    page_number: int
    source_filename: str
    text_snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationInfo]
    model: str
    chunks_used: int


@router.post("/ask", response_model=AskResponse)
async def ask_question(body: AskRequest, request: Request):
    """Ask a question about uploaded documents.

    Full RAG pipeline:
    1. Retrieve relevant chunks via semantic search
    2. Build prompt with delimiter-wrapped context
    3. Send to Ollama LLM for answer synthesis
    4. Return answer with page citations

    If Ollama is unavailable, returns retrieved chunks directly
    instead of failing (graceful degradation).
    """
    settings = request.app.state.settings

    # Step 1: Retrieve relevant chunks
    chunks = retrieve_chunks(
        query=body.question,
        collection_name=settings.chroma_collection,
        top_k=body.top_k,
        doc_id=body.document_id,
        host=settings.chroma_host,
        port=settings.chroma_port,
    )

    # Step 2+3: Generate answer with LLM
    result = generate_answer(
        question=body.question,
        chunks=chunks,
        ollama_base_url=settings.ollama_base_url,
        model=settings.ollama_model,
    )

    logger.info(
        "ask_completed",
        question_length=len(body.question),
        model=result.model,
        chunks_used=result.chunks_used,
        citations=len(result.citations),
    )

    return AskResponse(
        answer=result.answer,
        citations=[
            CitationInfo(
                page_number=c.page_number,
                source_filename=c.source_filename,
                text_snippet=c.text_snippet,
            )
            for c in result.citations
        ],
        model=result.model,
        chunks_used=result.chunks_used,
    )
