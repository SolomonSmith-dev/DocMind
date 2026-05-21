"""POST /api/auth/token -- issue a JWT access token."""

import structlog
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from docmind.core.auth import authenticate_user, create_access_token

router = APIRouter()
logger = structlog.get_logger()


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/auth/token", response_model=TokenResponse)
async def issue_token(request: Request, body: TokenRequest):
    """Issue a JWT access token for a valid username/password pair.

    The token must be sent as `Authorization: Bearer <token>` on all
    subsequent /api/* requests (except /health).
    """
    settings = request.app.state.settings

    if not authenticate_user(body.username, body.password):
        logger.warning("login_failed", username=body.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(
        subject=body.username,
        secret_key=settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
        expire_minutes=settings.jwt_expire_minutes,
    )

    logger.info("token_issued", username=body.username)
    return TokenResponse(access_token=token)
