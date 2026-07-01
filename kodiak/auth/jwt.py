"""
kodiak/auth/jwt.py

JWT issuance and verification for user authentication (access + refresh
tokens). API-key auth lives separately in kodiak.auth.api_keys.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import ExpiredSignatureError, JWTError, jwt
from pydantic import BaseModel, ValidationError

try:
    from kodiak.config.settings import settings

    SECRET_KEY: str = settings.SECRET_KEY
    ALGORITHM: str = settings.ALGORITHM
    ACCESS_TOKEN_EXPIRE_MINUTES: int = settings.ACCESS_TOKEN_EXPIRE_MINUTES
    REFRESH_TOKEN_EXPIRE_DAYS: int = settings.REFRESH_TOKEN_EXPIRE_DAYS
except (ImportError, AttributeError):
    SECRET_KEY = "change-me"
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    REFRESH_TOKEN_EXPIRE_DAYS = 30

_TOKEN_TYPE_ACCESS = "access"
_TOKEN_TYPE_REFRESH = "refresh"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)


class TokenClaims(BaseModel):
    sub: str
    exp: datetime
    iat: datetime
    token_type: str
    roles: list[str] = []
    scopes: list[str] = []


def _encode(
    *,
    user_id: str,
    token_type: str,
    expires_delta: timedelta,
    roles: list[str] | None = None,
    scopes: list[str] | None = None,
) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + expires_delta,
        "token_type": token_type,
        "roles": roles or [],
        "scopes": scopes or [],
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_access_token(
    user_id: str,
    *,
    roles: list[str] | None = None,
    scopes: list[str] | None = None,
) -> str:
    return _encode(
        user_id=user_id,
        token_type=_TOKEN_TYPE_ACCESS,
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        roles=roles,
        scopes=scopes,
    )


def create_refresh_token(user_id: str) -> str:
    return _encode(
        user_id=user_id,
        token_type=_TOKEN_TYPE_REFRESH,
        expires_delta=timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
    )


def _decode(token: str) -> TokenClaims:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except ExpiredSignatureError as exc:
        raise ValueError("Token has expired") from exc
    except JWTError as exc:
        raise ValueError("Invalid token") from exc

    try:
        return TokenClaims(**payload)
    except ValidationError as exc:
        raise ValueError("Malformed token claims") from exc


def verify_access_token(token: str) -> TokenClaims:
    claims = _decode(token)
    if claims.token_type != _TOKEN_TYPE_ACCESS:
        raise ValueError("Expected an access token")
    return claims


def verify_refresh_token(token: str) -> str:
    claims = _decode(token)
    if claims.token_type != _TOKEN_TYPE_REFRESH:
        raise ValueError("Expected a refresh token")
    return claims.sub


async def get_current_claims(
    token: Annotated[str | None, Depends(oauth2_scheme)] = None,
) -> TokenClaims:
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return verify_access_token(token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc