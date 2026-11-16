"""Security hardening: authenticated, owner-scoped document listing and
fail-closed boot when default secrets are in use (kanban t_bfbb317b)."""

import pytest
from fastapi.testclient import TestClient

from docmind.api import upload as upload_router
from docmind.app import create_app
from docmind.config import Settings
from docmind.core.auth import create_access_token
from docmind.core.retrieval import reset_client
from docmind.core.store import document_store

SECRET_ENV = "DOCMIND_JWT_SECRET_KEY"


def _client_for(app, user: str) -> TestClient:
    token = create_access_token(
        subject=user,
        secret_key=app.state.settings.jwt_secret_key,
        algorithm=app.state.settings.jwt_algorithm,
        expire_minutes=60,
    )
    return TestClient(app, headers={"Authorization": f"Bearer {token}"})


@pytest.fixture
def app(tmp_path):
    app = create_app()
    app.state.settings.upload_dir = str(tmp_path / "uploads")
    app.state.settings.chroma_port = 0
    app.state.settings.upload_rate_limit = "10000/minute"
    document_store._documents = {}
    document_store._ingested = {}
    reset_client()
    upload_router.limiter._storage.reset()
    return app


def test_list_documents_requires_auth(app):
    with TestClient(app) as anon:
        assert anon.get("/api/documents").status_code in (401, 403)


def test_list_documents_is_owner_scoped(app, sample_txt_bytes):
    files = {"file": ("a.txt", sample_txt_bytes, "text/plain")}
    with _client_for(app, "alice") as alice, _client_for(app, "bob") as bob:
        up = alice.post("/api/upload", files=files)
        assert up.status_code == 201, up.text
        doc_id = up.json()["document_id"]
        assert [d["document_id"] for d in alice.get("/api/documents").json()] == [
            doc_id
        ]
        assert bob.get("/api/documents").json() == []


def test_boot_refuses_default_jwt_secret(monkeypatch):
    monkeypatch.delenv("DOCMIND_ALLOW_INSECURE_DEFAULTS", raising=False)
    monkeypatch.delenv(SECRET_ENV, raising=False)
    monkeypatch.setenv("DOCMIND_ADMIN_PASSWORD", "a-strong-admin-pass-1")
    monkeypatch.setenv("DOCMIND_DEMO_PASSWORD", "a-strong-demo-pass-1")
    with pytest.raises(RuntimeError, match="default"):
        create_app()


def test_boot_refuses_default_passwords(monkeypatch):
    monkeypatch.delenv("DOCMIND_ALLOW_INSECURE_DEFAULTS", raising=False)
    monkeypatch.setenv(SECRET_ENV, "x" * 48)
    monkeypatch.delenv("DOCMIND_ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("DOCMIND_DEMO_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="default"):
        create_app()


def test_boot_ok_with_configured_secrets(monkeypatch):
    monkeypatch.delenv("DOCMIND_ALLOW_INSECURE_DEFAULTS", raising=False)
    monkeypatch.setenv(SECRET_ENV, "x" * 48)
    monkeypatch.setenv("DOCMIND_ADMIN_PASSWORD", "a-strong-admin-pass-1")
    monkeypatch.setenv("DOCMIND_DEMO_PASSWORD", "a-strong-demo-pass-1")
    assert create_app() is not None


def test_explicit_opt_in_allows_defaults(monkeypatch):
    monkeypatch.setenv("DOCMIND_ALLOW_INSECURE_DEFAULTS", "true")
    monkeypatch.delenv(SECRET_ENV, raising=False)
    assert Settings().allow_insecure_defaults is True
    assert create_app() is not None
