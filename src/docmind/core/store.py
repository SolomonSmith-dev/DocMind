"""In-memory document store.

Holds ingested documents and their chunks for retrieval.
This gets replaced by ChromaDB for vector storage in the embedding step,
but the metadata store stays for listing documents.
"""

from dataclasses import dataclass
from datetime import UTC, datetime

from docmind.core.ingest import IngestedDocument


@dataclass
class DocumentRecord:
    """Metadata record for an uploaded document."""

    doc_id: str
    filename: str
    total_pages: int
    chunk_count: int
    uploaded_at: str  # ISO format
    owner_id: str = ""


class DocumentStore:
    """Simple in-memory store for document metadata and chunks."""

    def __init__(self):
        self._documents: dict[str, DocumentRecord] = {}
        self._ingested: dict[str, IngestedDocument] = {}

    def save(self, doc: IngestedDocument, owner_id: str = "") -> DocumentRecord:
        record = DocumentRecord(
            doc_id=doc.doc_id,
            filename=doc.filename,
            total_pages=doc.total_pages,
            chunk_count=doc.chunk_count,
            uploaded_at=datetime.now(UTC).isoformat(),
            owner_id=owner_id,
        )
        self._documents[doc.doc_id] = record
        self._ingested[doc.doc_id] = doc
        return record

    def get(self, doc_id: str) -> DocumentRecord | None:
        return self._documents.get(doc_id)

    def get_ingested(self, doc_id: str) -> IngestedDocument | None:
        return self._ingested.get(doc_id)

    def list_all(self) -> list[DocumentRecord]:
        return list(self._documents.values())

    def list_for_owner(self, owner_id: str) -> list[DocumentRecord]:
        return [r for r in self._documents.values() if r.owner_id == owner_id]


# Singleton — shared across the app lifetime
document_store = DocumentStore()
