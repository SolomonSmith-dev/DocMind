class TestUploadEndpoint:
    def test_upload_valid_pdf(self, client, sample_pdf_bytes):
        response = client.post(
            "/api/upload",
            files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["filename"] == "test.pdf"
        assert data["pages"] >= 1
        assert data["chunks"] >= 1
        assert data["status"] == "processed"
        assert "document_id" in data

    def test_upload_rejects_html_file(self, client):
        # HTML is not an accepted format; rejected regardless of filename.
        response = client.post(
            "/api/upload",
            files={
                "file": (
                    "report.pdf",
                    b"<html><body>not a doc</body></html>",
                    "text/html",
                )
            },
        )
        assert response.status_code == 415

    def test_upload_rejects_oversized(self, client, sample_pdf_bytes):
        # Append data to exceed limit — use a small limit via the test
        huge = sample_pdf_bytes + b"\x00" * (26 * 1024 * 1024)
        response = client.post(
            "/api/upload",
            files={"file": ("big.pdf", huge, "application/pdf")},
        )
        assert response.status_code == 415

    def test_list_documents_empty(self, client):
        response = client.get("/api/documents")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_list_documents_after_upload(self, client, sample_pdf_bytes):
        # Upload first
        client.post(
            "/api/upload",
            files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
        )
        response = client.get("/api/documents")
        assert response.status_code == 200
        docs = response.json()
        assert len(docs) >= 1
        assert docs[-1]["filename"] == "test.pdf"
