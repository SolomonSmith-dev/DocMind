"""Tests for JWT authentication (Issue #4).

Covers:
- POST /api/auth/token issues a valid JWT
- Invalid credentials return 401
- Protected endpoints reject requests without a token
- Protected endpoints reject expired tokens
- Cross-user document isolation: user A cannot retrieve user B's chunks
"""

import pytest
from fastapi.testclient import TestClient
from jose import jwt

from docmind.app import create_app
from docmind.core.auth import create_access_token
from docmind.core.retrieval import reset_client
from docmind.core.session import session_store
from docmind.core.store import document_store


@pytest.fixture
def app_settings(tmp_path):
    """Bare app + settings, no default auth headers."""
    app = create_app()
    app.state.settings.upload_dir = str(tmp_path / "uploads")
    app.state.settings.chroma_host = "localhost"
    app.state.settings.chroma_port = 0
    app.state.settings.upload_rate_limit = "10000/minute"
    app.state.settings.ask_rate_limit = "10000/minute"
    document_store._documents = {}
    document_store._ingested = {}
    reset_client()
    session_store._sessions = {}
    return app


def _token(app, username: str, expire_minutes: int = 60) -> str:
    settings = app.state.settings
    return create_access_token(
        subject=username,
        secret_key=settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
        expire_minutes=expire_minutes,
    )


class TestTokenEndpoint:
    def test_valid_credentials_return_token(self, app_settings):
        with TestClient(app_settings) as c:
            resp = c.post(
                "/api/auth/token",
                json={"username": "admin", "password": "docmind-admin"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_invalid_password_returns_401(self, app_settings):
        with TestClient(app_settings) as c:
            resp = c.post(
                "/api/auth/token",
                json={"username": "admin", "password": "wrong"},
            )
        assert resp.status_code == 401

    def test_unknown_user_returns_401(self, app_settings):
        with TestClient(app_settings) as c:
            resp = c.post(
                "/api/auth/token",
                json={"username": "nobody", "password": "whatever"},
            )
        assert resp.status_code == 401

    def test_issued_token_is_valid_jwt(self, app_settings):
        settings = app_settings.state.settings
        with TestClient(app_settings) as c:
            resp = c.post(
                "/api/auth/token",
                json={"username": "demo", "password": "demo1234"},
            )
        token = resp.json()["access_token"]
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
        assert payload["sub"] == "demo"


class TestProtectedEndpoints:
    def test_upload_requires_token(self, app_settings, sample_pdf_bytes):
        with TestClient(app_settings) as c:
            resp = c.post(
                "/api/upload",
                files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
            )
        assert resp.status_code == 401

    def test_query_requires_token(self, app_settings):
        with TestClient(app_settings) as c:
            resp = c.post("/api/query", json={"question": "test"})
        assert resp.status_code == 401

    def test_ask_requires_token(self, app_settings):
        with TestClient(app_settings) as c:
            resp = c.post("/api/ask", json={"question": "test"})
        assert resp.status_code == 401

    def test_health_does_not_require_token(self, app_settings):
        with TestClient(app_settings) as c:
            resp = c.get("/health")
        assert resp.status_code == 200

    def test_expired_token_rejected(self, app_settings, sample_pdf_bytes):
        token = _token(app_settings, "admin", expire_minutes=-1)
        with TestClient(
            app_settings, headers={"Authorization": f"Bearer {token}"}
        ) as c:
            resp = c.post(
                "/api/upload",
                files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
            )
        assert resp.status_code == 401

    def test_valid_token_allows_upload(self, app_settings, sample_pdf_bytes):
        token = _token(app_settings, "admin")
        with TestClient(
            app_settings, headers={"Authorization": f"Bearer {token}"}
        ) as c:
            resp = c.post(
                "/api/upload",
                files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
            )
        assert resp.status_code == 201


class TestUserScopedDocumentIsolation:
    def test_user_a_cannot_retrieve_user_b_documents(
        self, app_settings, sample_pdf_bytes
    ):
        """User A uploads a doc; user B's query should return no chunks."""
        token_a = _token(app_settings, "admin")
        token_b = _token(app_settings, "demo")

        with TestClient(
            app_settings, headers={"Authorization": f"Bearer {token_a}"}
        ) as c_a:
            upload_resp = c_a.post(
                "/api/upload",
                files={"file": ("a_doc.pdf", sample_pdf_bytes, "application/pdf")},
            )
            assert upload_resp.status_code == 201

        with TestClient(
            app_settings, headers={"Authorization": f"Bearer {token_b}"}
        ) as c_b:
            ask_resp = c_b.post(
                "/api/ask", json={"question": "What is this document about?"}
            )
            assert ask_resp.status_code == 200
            # User B should find no chunks from user A's document
            data = ask_resp.json()
            assert data["chunks_used"] == 0 or "No relevant documents" in data["answer"]
