from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks: int


@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_document():
    raise HTTPException(status_code=501, detail="Upload not yet implemented")
