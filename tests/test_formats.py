"""Tests for DOCX and TXT format support (Issue #6)."""

import io
import zipfile

from docmind.core.security import validate_magic_bytes

_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class TestDocxUpload:
    def test_upload_valid_docx(self, client, sample_docx_bytes):
        resp = client.post(
            "/api/upload",
            files={"file": ("report.docx", sample_docx_bytes, _DOCX_MIME)},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["pages"] >= 1
        assert data["chunks"] >= 1
        assert data["status"] == "processed"

    def test_upload_docx_produces_queryable_content(self, client, sample_docx_bytes):
        """Uploaded DOCX content must be retrievable via /api/ask."""
        client.post(
            "/api/upload",
            files={"file": ("report.docx", sample_docx_bytes, _DOCX_MIME)},
        )
        resp = client.post(
            "/api/ask", json={"question": "What is this document about?"}
        )
        assert resp.status_code == 200
        assert resp.json()["chunks_used"] > 0

    def test_malformed_zip_rejected(self, client):
        """A ZIP that lacks word/document.xml must be rejected."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("not_a_word_doc.xml", "<root/>")
        bad_zip = buf.getvalue()

        resp = client.post(
            "/api/upload",
            files={"file": ("fake.docx", bad_zip, "application/zip")},
        )
        assert resp.status_code == 415

    def test_docx_magic_bytes_accepted_by_security(self, sample_docx_bytes):
        mime = validate_magic_bytes(sample_docx_bytes)
        assert "wordprocessingml" in mime


class TestTxtUpload:
    def test_upload_valid_txt(self, client, sample_txt_bytes):
        resp = client.post(
            "/api/upload",
            files={"file": ("notes.txt", sample_txt_bytes, "text/plain")},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["pages"] >= 1
        assert data["chunks"] >= 1
        assert data["status"] == "processed"

    def test_upload_txt_latin1_encoding(self, client):
        """latin-1 encoded TXT file should be accepted and ingested."""
        content = "Caf\xe9 menu for the conf\xe9rence.\n" * 10
        txt_bytes = content.encode("latin-1")
        resp = client.post(
            "/api/upload",
            files={"file": ("menu.txt", txt_bytes, "text/plain")},
        )
        assert resp.status_code == 201

    def test_txt_produces_queryable_content(self, client, sample_txt_bytes):
        client.post(
            "/api/upload",
            files={"file": ("notes.txt", sample_txt_bytes, "text/plain")},
        )
        resp = client.post("/api/ask", json={"question": "What is in this document?"})
        assert resp.status_code == 200
        assert resp.json()["chunks_used"] > 0

    def test_non_text_binary_rejected(self, client):
        """Random binary data should not be accepted."""
        binary = bytes(range(256)) * 4
        resp = client.post(
            "/api/upload",
            files={"file": ("binary.txt", binary, "text/plain")},
        )
        assert resp.status_code == 415
