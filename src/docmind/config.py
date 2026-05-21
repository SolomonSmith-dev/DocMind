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
    cors_origins: list[str] = ["http://localhost:3000"]

    model_config = {"env_file": ".env", "env_prefix": "DOCMIND_"}
