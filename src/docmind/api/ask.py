import structlog
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from docmind.core.auth import get_current_user_id
from docmind.core.chat import generate_answer
from docmind.core.rerank import rerank
from docmind.core.retrieval import retrieve_chunks
from docmind.core.session import session_store

router = APIRouter()
logger = structlog.get_logger()
limiter = Limiter(key_func=get_remote_address)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    document_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    session_id: str | None = None


class CitationInfo(BaseModel):
    page_number: int
    source_filename: str
    text_snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationInfo]
    model: str
    chunks_used: int
    # Weighted confidence in [0, 1].
    # Factors: top similarity (0.5) + coverage (0.3) + LLM self-report (0.2).
    confidence: float
    # True when confidence falls below DOCMIND_LOW_CONFIDENCE_THRESHOLD.
    low_confidence: bool
    session_id: str | None = None


@router.post("/ask", response_model=AskResponse)
@limiter.limit("30/minute")
async def ask_question(
    request: Request,
    body: AskRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Ask a question about uploaded documents.

    Full RAG pipeline:
    1. Retrieve relevant chunks via semantic search
    2. Build prompt with delimiter-wrapped context
    3. Send to Ollama LLM for answer synthesis
    4. Return answer with page citations and confidence score

    If Ollama is unavailable, returns retrieved chunks directly
    instead of failing (graceful degradation).
    """
    settings = request.app.state.settings

    # When re-ranking is enabled, fetch a larger candidate set then narrow it.
    retrieval_k = settings.rerank_candidates if settings.enable_rerank else body.top_k
    # Resolve session -- creates one if session_id is unknown or None
    session = session_store.get_or_create(body.session_id)
    prior_turns = session.recent_turns(settings.session_window)

    chunks = retrieve_chunks(
        query=body.question,
        collection_name=settings.chroma_collection,
        top_k=retrieval_k,
        doc_id=body.document_id,
        host=settings.chroma_host,
        port=settings.chroma_port,
        user_id=user_id,
    )

    if settings.enable_rerank and chunks:
        chunks = rerank(query=body.question, chunks=chunks, top_k=body.top_k)

    result = generate_answer(
        question=body.question,
        chunks=chunks,
        ollama_base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        top_k_requested=body.top_k,
        low_confidence_threshold=settings.low_confidence_threshold,
        prior_turns=prior_turns if prior_turns else None,
    )

    # Record this turn for future context
    session.add_turn(question=body.question, answer=result.answer)

    logger.info(
        "ask_completed",
        question_length=len(body.question),
        model=result.model,
        chunks_used=result.chunks_used,
        citations=len(result.citations),
        confidence=result.confidence,
        low_confidence=result.low_confidence,
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
        confidence=result.confidence,
        low_confidence=result.low_confidence,
        session_id=session.id,
    )
