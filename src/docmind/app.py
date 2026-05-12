from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from docmind.api import ask, health, query, upload
from docmind.config import Settings
from docmind.logging import setup_logging
from docmind.middleware import RequestIDMiddleware


def create_app() -> FastAPI:
    settings = Settings()
    setup_logging(debug=settings.debug)

    # Rate limiter — keyed by client IP
    limiter = Limiter(key_func=get_remote_address)

    app = FastAPI(title=settings.app_name, version="0.1.0")
    app.state.settings = settings
    app.state.limiter = limiter

    # Exception handler for rate limit exceeded
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS — explicit origins, no wildcard
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.add_middleware(RequestIDMiddleware)

    app.include_router(health.router, tags=["health"])
    app.include_router(upload.router, prefix="/api", tags=["upload"])
    app.include_router(query.router, prefix="/api", tags=["query"])
    app.include_router(ask.router, prefix="/api", tags=["ask"])

    return app


app = create_app()
