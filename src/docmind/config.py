from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "DocMind"
    debug: bool = False
    upload_dir: str = "data/uploads"
    max_upload_size_mb: int = 20
    allowed_mime_types: list[str] = ["application/pdf"]
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "docmind"
    embedding_model: str = "all-MiniLM-L6-v2"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    upload_rate_limit: str = "10/minute"
    ask_rate_limit: str = "30/minute"
    # Confidence below this value sets low_confidence=True in AskResponse.
    low_confidence_threshold: float = 0.4
    # Cross-encoder re-ranking: retrieve this many candidates, rerank to top_k.
    enable_rerank: bool = False
    rerank_candidates: int = 20
    # How many prior turns to inject into the prompt for multi-turn sessions.
    session_window: int = 5
    # JWT settings -- override in production with a long random secret.
    jwt_secret_key: str = "docmind-dev-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    # Override DOCMIND_ADMIN_PASSWORD / DOCMIND_DEMO_PASSWORD in production.
    admin_password: str = "docmind-admin"
    demo_password: str = "demo1234"

    # Set DOCMIND_ALLOW_INSECURE_DEFAULTS=true only for local dev and tests.
    # Otherwise boot refuses the built-in secrets and passwords.
    allow_insecure_defaults: bool = False

    model_config = {"env_file": ".env", "env_prefix": "DOCMIND_"}


DEFAULT_JWT_SECRET = Settings.model_fields["jwt_secret_key"].default
DEFAULT_ADMIN_PASSWORD = Settings.model_fields["admin_password"].default
DEFAULT_DEMO_PASSWORD = Settings.model_fields["demo_password"].default


def check_boot_secrets(settings: Settings) -> None:
    """Refuse to start with the built-in JWT secret or default passwords."""
    if settings.allow_insecure_defaults:
        return
    problems = []
    if settings.jwt_secret_key == DEFAULT_JWT_SECRET:
        problems.append("DOCMIND_JWT_SECRET_KEY")
    if settings.admin_password == DEFAULT_ADMIN_PASSWORD:
        problems.append("DOCMIND_ADMIN_PASSWORD")
    if settings.demo_password == DEFAULT_DEMO_PASSWORD:
        problems.append("DOCMIND_DEMO_PASSWORD")
    if problems:
        raise RuntimeError(
            "Refusing to boot with default secrets: set "
            + ", ".join(problems)
            + " (or DOCMIND_ALLOW_INSECURE_DEFAULTS=true for local dev only)."
        )
