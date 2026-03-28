"""PDF ingestion pipeline: extract text, chunk with overlap, preserve page metadata.

The chunking strategy matters more than most people think. Naive splitting
(every 500 characters) destroys context — a sentence about "the defendant"
gets separated from the sentence naming who the defendant is. Our approach:

1. Extract text per page (preserving page numbers for citations)
2. Split at paragraph boundaries first, sentence boundaries second
3. Use overlapping windows so context isn't lost at chunk edges
4. Tag every chunk with its source page for citation mapping later
"""

from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import structlog

logger = structlog.get_logger()


@dataclass
class Chunk:
    """A single chunk of document text with source metadata."""

    text: str
    doc_id: str
    page_number: int  # 1-indexed for human-readable citations
    chunk_index: int
    source_filename: str


@dataclass
class IngestedDocument:
    """Result of ingesting a PDF: metadata + all chunks."""

    doc_id: str
    filename: str
    total_pages: int
    chunks: list[Chunk] = field(default_factory=list)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


def extract_pages(file_path: Path) -> list[tuple[int, str]]:
    """Extract text from each page of a PDF.

    Returns list of (page_number, text) tuples.
    Page numbers are 1-indexed for human-readable citations.
    """
    doc = fitz.open(str(file_path))
    pages = []

    for page in doc:
        text = page.get_text().strip()
        if text:
            pages.append((page.number + 1, text))  # 1-indexed

    doc.close()

    logger.info("pdf_extracted", path=str(file_path), pages_with_text=len(pages))
    return pages


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph/sentence boundaries.

    Why overlap? Without it, a question about something mentioned at the end
    of chunk N and the beginning of chunk N+1 would match neither chunk well.
    Overlap ensures context spans chunk boundaries.

    Why prefer paragraph boundaries? Splitting mid-sentence creates chunks
    that are harder to embed meaningfully and produce worse retrieval results.
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If we're not at the end of the text, try to find a clean break point
        if end < len(text):
            # Look for paragraph break (double newline) near the end
            para_break = text.rfind("\n\n", start + chunk_size // 2, end)
            if para_break != -1:
                end = para_break + 2  # Include the newlines

            else:
                # Fall back to sentence boundary (period + space)
                sentence_break = text.rfind(". ", start + chunk_size // 2, end)
                if sentence_break != -1:
                    end = sentence_break + 2  # Include period and space

                else:
                    # Fall back to any whitespace
                    space_break = text.rfind(" ", start + chunk_size // 2, end)
                    if space_break != -1:
                        end = space_break + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start forward, accounting for overlap
        start = end - chunk_overlap if end < len(text) else end

    return chunks


def ingest_pdf(
    file_path: Path,
    doc_id: str,
    filename: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> IngestedDocument:
    """Full ingestion pipeline: extract pages → chunk text → build metadata.

    This is the entry point. Call this with a validated, safely-stored PDF
    and get back an IngestedDocument with all chunks tagged with page numbers.
    """
    pages = extract_pages(file_path)
    total_pages = len(pages)

    all_chunks: list[Chunk] = []
    chunk_index = 0

    for page_number, page_text in pages:
        text_chunks = chunk_text(page_text, chunk_size, chunk_overlap)

        for text in text_chunks:
            all_chunks.append(
                Chunk(
                    text=text,
                    doc_id=doc_id,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    source_filename=filename,
                )
            )
            chunk_index += 1

    result = IngestedDocument(
        doc_id=doc_id,
        filename=filename,
        total_pages=total_pages,
        chunks=all_chunks,
    )

    logger.info(
        "pdf_ingested",
        doc_id=doc_id,
        filename=filename,
        total_pages=total_pages,
        total_chunks=result.chunk_count,
    )

    return result
