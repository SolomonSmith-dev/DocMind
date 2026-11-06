import io

import docx as python_docx
import fitz  # PyMuPDF
import pytest
from fastapi.testclient import TestClient

from docmind.api import ask as ask_router
from docmind.api import auth as auth_router
from docmind.api import upload as upload_router
from docmind.app import create_app
from docmind.core.auth import create_access_token
from docmind.core.retrieval import reset_client
from docmind.core.session import session_store
from docmind.core.store import document_store


@pytest.fixture
def client(tmp_path):
    """Test client with isolated upload dir, fresh store, and ephemeral ChromaDB."""
    app = create_app()
    # Point uploads to a temp directory
    app.state.settings.upload_dir = str(tmp_path / "uploads")
    # Use localhost (will fall back to ephemeral ChromaDB in tests)
    app.state.settings.chroma_host = "localhost"
    app.state.settings.chroma_port = 0  # force ephemeral fallback
    # Disable rate limits in tests to prevent cross-test 429s
    app.state.settings.upload_rate_limit = "10000/minute"
    app.state.settings.ask_rate_limit = "10000/minute"
    # Reset state between tests
    document_store._documents = {}
    document_store._ingested = {}
    reset_client()
    # Reset module-level rate limiter storage to prevent cross-test 429s
    ask_router.limiter._storage.reset()
    auth_router.limiter._storage.reset()
    upload_router.limiter._storage.reset()
    # Reset session store to prevent cross-test session bleed
    session_store._sessions = {}
    # Generate a test JWT so all protected endpoints work without extra headers.
    test_token = create_access_token(
        subject="test-user",
        secret_key=app.state.settings.jwt_secret_key,
        algorithm=app.state.settings.jwt_algorithm,
        expire_minutes=60,
    )
    with TestClient(app, headers={"Authorization": f"Bearer {test_token}"}) as c:
        yield c


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Generate a minimal real PDF with text content for testing.

    Uses PyMuPDF to create a valid PDF in memory — this has correct
    magic bytes and structure, so it passes magic byte validation.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "This is a test document for DocMind.\n\n"
        "It contains multiple paragraphs to test chunking behavior.\n\n"
        "The quick brown fox jumps over the lazy dog. "
        "This sentence exists to add more content for the chunker "
        "to work with. "
        "Semantic search requires enough text to generate "
        "meaningful embeddings.",
    )
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def sample_docx_bytes() -> bytes:
    """Generate a minimal valid DOCX in memory using python-docx."""
    buf = io.BytesIO()
    doc = python_docx.Document()
    doc.add_heading("DocMind DOCX Test", 0)
    doc.add_paragraph(
        "This is a test document in DOCX format for DocMind. "
        "It contains enough text to produce at least one chunk "
        "during the ingestion pipeline."
    )
    doc.add_paragraph(
        "Second paragraph: the quick brown fox jumps over the lazy dog. "
        "Semantic search requires meaningful text content."
    )
    doc.save(buf)
    return buf.getvalue()


@pytest.fixture
def sample_txt_bytes() -> bytes:
    """Generate plain UTF-8 text bytes for upload testing."""
    content = (
        "DocMind TXT test document.\n\n"
        "This file contains plain text with multiple lines.\n"
        "It should be ingested and chunked by the TXT pipeline.\n\n"
        "The quick brown fox jumps over the lazy dog.\n"
        "Semantic search requires enough content to embed meaningfully.\n"
    )
    return content.encode("utf-8")
