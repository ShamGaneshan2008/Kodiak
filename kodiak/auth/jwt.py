"""
kodiak/auth/jwt.py

JSON Web Token issuance and verification for Kodiak's API layer.

Handles:
    - Short-lived access tokens
    - Long-lived refresh tokens (rotated on use)
    - Token revocation via a Redis-backed denylist
    - Claims used downstream by permissions.py for RBAC checks
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import ExpiredSignatureError, JWTError, jwt
from pydantic import BaseModel

from kodiak.config.settings import get_settings
from kodiak.db.session import get_redis

settings = get_settings()

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)

ALGORITHM = "HS256"
ACCESS_TOKEN_TTL = timedelta(minutes=15)
REFRESH_TOKEN_TTL = timedelta(days=14)
DENYLIST_PREFIX = "jwt:denylist:"


class TokenType(str, Enum):
    ACCESS = "access"
    REFRESH = "refresh"


class TokenClaims(BaseModel):
    """Decoded, validated claims for an authenticated request."""

    sub: str  # user id
    org_id: str | None = None
    roles: list[str] = []
    scopes: list[str] = []
    token_type: TokenType
    jti: str
    exp: datetime
    iat: datetime


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class AuthError(HTTPException):
    def __init__(self, detail: str = "Could not validate credentials") -> None:
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


def _encode(
    *,
    subject: str,
    token_type: TokenType,
    ttl: timedelta,
    org_id: str | None = None,
    roles: list[str] | None = None,
    scopes: list[str] | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> tuple[str, str, datetime]:
    now = datetime.now(timezone.utc)
    expires_at = now + ttl
    jti = str(uuid.uuid4())

    payload: dict[str, Any] = {
        "sub": subject,
        "org_id": org_id,
        "roles": roles or [],
        "scopes": scopes or [],
        "token_type": token_type.value,
        "jti": jti,
        "iat": now,
        "exp": expires_at,
        "iss": "kodiak",
    }
    if extra_claims:
        payload.update(extra_claims)

    token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=ALGORITHM)
    return token, jti, expires_at


def issue_token_pair(
    *,
    user_id: str,
    org_id: str | None = None,
    roles: list[str] | None = None,
    scopes: list[str] | None = None,
) -> TokenPair:
    """Issue a fresh access/refresh pair for a successfully authenticated user."""
    access_token, _, _ = _encode(
        subject=user_id,
        token_type=TokenType.ACCESS,
        ttl=ACCESS_TOKEN_TTL,
        org_id=org_id,
        roles=roles,
        scopes=scopes,
    )
    refresh_token, _, _ = _encode(
        subject=user_id,
        token_type=TokenType.REFRESH,
        ttl=REFRESH_TOKEN_TTL,
        org_id=org_id,
        roles=roles,
        scopes=scopes,
    )
    return TokenPair(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=int(ACCESS_TOKEN_TTL.total_seconds()),
    )


async def decode_token(token: str, *, expected_type: TokenType | None = None) -> TokenClaims:
    """Validate signature/expiry, check the denylist, and return typed claims."""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[ALGORITHM])
    except ExpiredSignatureError as exc:
        raise AuthError("Token has expired") from exc
    except JWTError as exc:
        raise AuthError("Invalid token") from exc

    claims = TokenClaims(**payload)

    if expected_type and claims.token_type != expected_type:
        raise AuthError(f"Expected a {expected_type.value} token")

    redis = await get_redis()
    if await redis.exists(f"{DENYLIST_PREFIX}{claims.jti}"):
        raise AuthError("Token has been revoked")

    return claims


async def revoke_token(claims: TokenClaims) -> None:
    """Add a token's jti to the Redis denylist until its natural expiry."""
    redis = await get_redis()
    ttl_seconds = max(int((claims.exp - datetime.now(timezone.utc)).total_seconds()), 0)
    await redis.set(f"{DENYLIST_PREFIX}{claims.jti}", "1", ex=ttl_seconds or 1)


async def rotate_refresh_token(refresh_token: str) -> TokenPair:
    """Verify a refresh token, revoke it, and issue a brand-new pair."""
    claims = await decode_token(refresh_token, expected_type=TokenType.REFRESH)
    await revoke_token(claims)
    return issue_token_pair(
        user_id=claims.sub,
        org_id=claims.org_id,
        roles=claims.roles,
        scopes=claims.scopes,
    )


async def get_current_claims(
    token: str | None = Depends(_oauth2_scheme),
) -> TokenClaims:
    """FastAPI dependency: resolve the bearer token on the request to claims."""
    if not token:
        raise AuthError("Not authenticated")
    return await decode_token(token, expected_type=TokenType.ACCESS)