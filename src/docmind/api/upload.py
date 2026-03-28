import structlog
from fastapi import APIRouter, HTTPException, Request, UploadFile
from pydantic import BaseModel

from docmind.core.ingest import ingest_pdf
from docmind.core.security import validate_and_prepare_upload
from docmind.core.store import document_store

router = APIRouter()
logger = structlog.get_logger()


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int
    status: str


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int
    uploaded_at: str


@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_document(file: UploadFile, request: Request):
    """Upload a PDF document for processing.

    The file goes through security validation (magic bytes, size, MIME type),
    then gets extracted and chunked with page metadata preserved.
    """
    settings = request.app.state.settings

    # Read file content
    content = await file.read()

    # Validate and prepare (raises ValueError on failure)
    try:
        doc_id, storage_path, safe_name = validate_and_prepare_upload(
            content=content,
            original_filename=file.filename or "unnamed.pdf",
            upload_dir=settings.upload_dir,
        )
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e)) from e

    # Ensure upload directory exists and save the file
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_path.write_bytes(content)

    # Ingest: extract text + chunk
    try:
        ingested = ingest_pdf(
            file_path=storage_path,
            doc_id=doc_id,
            filename=safe_name,
        )
    except Exception as e:
        # Clean up the saved file on ingestion failure
        storage_path.unlink(missing_ok=True)
        logger.error("ingestion_failed", doc_id=doc_id, error=str(e))
        raise HTTPException(status_code=422, detail="Failed to process PDF") from e

    # Store metadata and chunks
    document_store.save(ingested)

    logger.info(
        "document_uploaded",
        doc_id=doc_id,
        filename=safe_name,
        pages=ingested.total_pages,
        chunks=ingested.chunk_count,
    )

    return UploadResponse(
        document_id=doc_id,
        filename=safe_name,
        pages=ingested.total_pages,
        chunks=ingested.chunk_count,
        status="processed",
    )


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents():
    """List all uploaded documents with metadata."""
    docs = document_store.list_all()
    return [
        DocumentInfo(
            document_id=d.doc_id,
            filename=d.filename,
            pages=d.total_pages,
            chunks=d.chunk_count,
            uploaded_at=d.uploaded_at,
        )
        for d in docs
    ]
