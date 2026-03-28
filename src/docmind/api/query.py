import structlog
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from docmind.core.retrieval import retrieve_chunks

router = APIRouter()
logger = structlog.get_logger()


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    document_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    text: str
    page_number: int
    score: float
    document_id: str
    source_filename: str


class QueryResponse(BaseModel):
    query: str
    results: list[Source]


@router.post("/query", response_model=QueryResponse)
async def query_documents(body: QueryRequest, request: Request):
    """Search uploaded documents using semantic similarity.

    Embeds the question, searches ChromaDB for the most relevant
    chunks, and returns them ranked by similarity score with
    page numbers for citation.
    """
    settings = request.app.state.settings

    results = retrieve_chunks(
        query=body.question,
        collection_name=settings.chroma_collection,
        top_k=body.top_k,
        doc_id=body.document_id,
        host=settings.chroma_host,
        port=settings.chroma_port,
    )

    sources = [
        Source(
            text=r.text,
            page_number=r.page_number,
            score=r.score,
            document_id=r.doc_id,
            source_filename=r.source_filename,
        )
        for r in results
    ]

    logger.info(
        "query_executed",
        question_length=len(body.question),
        results_returned=len(sources),
    )

    return QueryResponse(query=body.question, results=sources)
