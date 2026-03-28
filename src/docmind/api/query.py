from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    document_id: str | None = None


class Source(BaseModel):
    document: str
    chunk: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    raise HTTPException(status_code=501, detail="Query not yet implemented")
