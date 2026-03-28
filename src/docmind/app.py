from fastapi import FastAPI

from docmind.api import ask, health, query, upload
from docmind.config import Settings
from docmind.logging import setup_logging
from docmind.middleware import RequestIDMiddleware


def create_app() -> FastAPI:
    settings = Settings()
    setup_logging(debug=settings.debug)

    app = FastAPI(title=settings.app_name, version="0.1.0")
    app.state.settings = settings

    app.add_middleware(RequestIDMiddleware)

    app.include_router(health.router, tags=["health"])
    app.include_router(upload.router, prefix="/api", tags=["upload"])
    app.include_router(query.router, prefix="/api", tags=["query"])
    app.include_router(ask.router, prefix="/api", tags=["ask"])

    return app


app = create_app()
