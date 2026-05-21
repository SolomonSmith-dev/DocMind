"""Document ingestion pipeline: extract, chunk, and preserve page metadata.

Supported formats: PDF, DOCX, TXT.

The chunking strategy matters more than most people think. Naive splitting
(every 500 characters) destroys context -- a sentence about "the defendant"
gets separated from the sentence naming who the defendant is. Our approach:

1. Extract text per page (preserving page numbers for citations)
2. Split at paragraph boundaries first, sentence boundaries second
3. Use overlapping windows so context isn't lost at chunk edges
4. Tag every chunk with its source page for citation mapping later
"""

from dataclasses import dataclass, field
from pathlib import Path

import docx as python_docx
import fitz  # PyMuPDF
import structlog

logger = structlog.get_logger()

_DOCX_PARAGRAPHS_PER_PAGE = 30
_TXT_LINES_PER_PAGE = 60


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


def extract_pages_docx(file_path: Path) -> list[tuple[int, str]]:
    """Extract text from a DOCX file, grouping content into pseudo-pages.

    DOCX has no inherent page concept at the paragraph level, so we group
    paragraphs into batches of DOCX_PARAGRAPHS_PER_PAGE to create citation
    anchors. Table cells are treated as paragraph text and included inline.

    Returns list of (page_number, text) tuples. Page numbers are 1-indexed.
    """
    wml_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    doc = python_docx.Document(str(file_path))

    # Collect all text blocks: paragraphs and table cells
    blocks: list[str] = []
    for element in doc.element.body:
        tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag
        if tag == "p":
            text = "".join(
                node.text or "" for node in element.iter() if hasattr(node, "text")
            )
            if text.strip():
                blocks.append(text.strip())
        elif tag == "tbl":
            for row in element.iter(f"{{{wml_ns}}}tr"):
                cell_texts = []
                for cell in row.iter(f"{{{wml_ns}}}tc"):
                    cell_text = "".join(
                        node.text or ""
                        for node in cell.iter()
                        if hasattr(node, "text")
                    )
                    if cell_text.strip():
                        cell_texts.append(cell_text.strip())
                if cell_texts:
                    blocks.append(" | ".join(cell_texts))

    if not blocks:
        return []

    pages: list[tuple[int, str]] = []
    for i in range(0, len(blocks), _DOCX_PARAGRAPHS_PER_PAGE):
        page_blocks = blocks[i : i + _DOCX_PARAGRAPHS_PER_PAGE]
        page_text = "\n\n".join(page_blocks)
        pages.append((len(pages) + 1, page_text))

    logger.info("docx_extracted", path=str(file_path), pseudo_pages=len(pages))
    return pages


def extract_pages_txt(file_path: Path) -> list[tuple[int, str]]:
    """Extract text from a plain-text file with encoding auto-detection.

    Tries UTF-8 first, then latin-1 (which never fails on arbitrary bytes),
    with a cp1252 fallback for Windows-authored files. Groups lines into
    pseudo-pages using the same heuristic as DOCX extraction.

    Returns list of (page_number, text) tuples.
    """
    raw = file_path.read_bytes()
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        # latin-1 never raises, so this branch is unreachable, but be safe
        text = raw.decode("latin-1", errors="replace")

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    pages: list[tuple[int, str]] = []
    for i in range(0, len(lines), _TXT_LINES_PER_PAGE):
        page_lines = lines[i : i + _TXT_LINES_PER_PAGE]
        pages.append((len(pages) + 1, "\n".join(page_lines)))

    logger.info("txt_extracted", path=str(file_path), pseudo_pages=len(pages))
    return pages


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


def _ingest_pages(
    pages: list[tuple[int, str]],
    doc_id: str,
    filename: str,
    fmt: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> IngestedDocument:
    """Shared chunking logic for pre-extracted pages (DOCX, TXT)."""
    all_chunks: list[Chunk] = []
    chunk_index = 0

    for page_number, page_text in pages:
        for text in chunk_text(page_text, chunk_size, chunk_overlap):
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
        total_pages=len(pages),
        chunks=all_chunks,
    )

    logger.info(
        f"{fmt}_ingested",
        doc_id=doc_id,
        filename=filename,
        total_pages=result.total_pages,
        total_chunks=result.chunk_count,
    )

    return result


def ingest_docx(
    file_path: Path,
    doc_id: str,
    filename: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> IngestedDocument:
    """Ingest a DOCX file: extract paragraphs + tables, chunk, build metadata."""
    pages = extract_pages_docx(file_path)
    return _ingest_pages(pages, doc_id, filename, "docx", chunk_size, chunk_overlap)


def ingest_txt(
    file_path: Path,
    doc_id: str,
    filename: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> IngestedDocument:
    """Ingest a plain-text file with UTF-8/cp1252/latin-1 encoding detection."""
    pages = extract_pages_txt(file_path)
    return _ingest_pages(pages, doc_id, filename, "txt", chunk_size, chunk_overlap)


def ingest_document(
    file_path: Path,
    doc_id: str,
    filename: str,
    mime_type: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> IngestedDocument:
    """Format-aware ingestion dispatcher.

    Routes to the correct ingestor based on the MIME type returned
    by security.validate_magic_bytes().
    """
    if mime_type == "application/pdf":
        return ingest_pdf(file_path, doc_id, filename, chunk_size, chunk_overlap)
    if mime_type == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        return ingest_docx(file_path, doc_id, filename, chunk_size, chunk_overlap)
    if mime_type == "text/plain":
        return ingest_txt(file_path, doc_id, filename, chunk_size, chunk_overlap)

    raise ValueError(f"Unsupported MIME type for ingestion: {mime_type}")
