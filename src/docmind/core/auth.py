"""JWT authentication and user identity.

Provides:
- POST /api/auth/token: issue a JWT for a username/password pair
- get_current_user_id(): FastAPI dependency that validates Bearer tokens

User store: in-memory dict for this iteration. Production deployments
should replace this with a database-backed user table.

Token format: HS256 JWT with {"sub": username, "exp": expiry}.
"""

from datetime import UTC, datetime, timedelta
from typing import Annotated

import bcrypt
import jwt
import structlog
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt.exceptions import InvalidTokenError

logger = structlog.get_logger()

_bearer = HTTPBearer()


def build_users(admin_password: str, demo_password: str) -> dict[str, bytes]:
    """Hash configured plaintext passwords and return the users dict.

    Called once at app startup; result stored in app.state.users.
    Only adds a user if the password is non-empty, so unconfigured
    accounts are simply absent (no login possible).
    """
    users: dict[str, bytes] = {}
    if admin_password:
        users["admin"] = bcrypt.hashpw(admin_password.encode(), bcrypt.gensalt())
    if demo_password:
        users["demo"] = bcrypt.hashpw(demo_password.encode(), bcrypt.gensalt())
    return users


def _verify_password(plain: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed)


def authenticate_user(username: str, password: str, users: dict[str, bytes]) -> bool:
    hashed = users.get(username)
    if hashed is None:
        return False
    return _verify_password(password, hashed)


def create_access_token(
    subject: str,
    secret_key: str,
    algorithm: str,
    expire_minutes: int,
) -> str:
    """Issue a signed JWT that expires in `expire_minutes` minutes."""
    expires_at = datetime.now(UTC) + timedelta(minutes=expire_minutes)
    payload = {"sub": subject, "exp": expires_at}
    return jwt.encode(payload, secret_key, algorithm=algorithm)


def decode_token(token: str, secret_key: str, algorithm: str) -> str:
    """Validate a JWT and return the subject (username).

    Raises HTTPException 401 on any validation failure.
    """
    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        subject: str | None = payload.get("sub")
        if subject is None:
            raise InvalidTokenError("missing sub claim")
        return subject
    except InvalidTokenError as exc:
        logger.warning("jwt_validation_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


async def get_current_user_id(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer)],
) -> str:
    """FastAPI dependency: validate Bearer token and return the username.

    Reads JWT settings from request.app.state.settings so no factory is needed.
    """
    settings = request.app.state.settings
    return decode_token(
        credentials.credentials,
        settings.jwt_secret_key,
        settings.jwt_algorithm,
    )
