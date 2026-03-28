import fitz  # PyMuPDF
import pytest
from fastapi.testclient import TestClient

from docmind.app import create_app
from docmind.core.retrieval import reset_client
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
    # Reset state between tests
    document_store._documents = {}
    document_store._ingested = {}
    reset_client()
    with TestClient(app) as c:
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
