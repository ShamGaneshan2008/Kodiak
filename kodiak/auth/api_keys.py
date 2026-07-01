"""
kodiak/auth/api_key.py

Long-lived API keys for programmatic access (CI pipelines, CLI, the GitHub
Actions integration, third-party plugins). Unlike JWTs, these are meant to be
pasted into config files and CI secrets, so they:

    - Are shown to the user exactly once, at creation time
    - Are stored only as a salted hash (never recoverable)
    - Carry an explicit, narrow set of scopes (no implicit role inheritance)
    - Can be revoked instantly without waiting for expiry
    - Use a recognizable prefix (`kdk_live_`, `kdk_test_`) for secret-scanning
      tooling such as Gitleaks/GitHub secret scanning partner alerts
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kodiak.auth.permissions import Scope
from kodiak.db.session import get_db
from kodiak.db.models.api_key import APIKey  # ORM model: id, org_id, name, key_hash,
                                              # prefix, scopes, last_used_at, expires_at,
                                              # revoked_at, created_by

_API_KEY_HEADER = APIKeyHeader(name="X-Kodiak-Api-Key", auto_error=False)

KEY_PREFIX_LIVE = "kdk_live_"
KEY_PREFIX_TEST = "kdk_test_"
SECRET_BYTES = 32


@dataclass
class IssuedKey:
    """Returned exactly once at creation; the plaintext is never stored."""

    id: str
    plaintext: str
    prefix: str
    scopes: list[Scope]


@dataclass
class APIKeyPrincipal:
    """Resolved identity for a validated API key request."""

    key_id: str
    org_id: str
    scopes: set[Scope]
    created_by: str


def _hash_secret(secret_value: str) -> str:
    """SHA-256 of the raw key. A long, high-entropy random secret does not
    need a slow KDF like bcrypt -- it is not a user-chosen, low-entropy password."""
    return hashlib.sha256(secret_value.encode("utf-8")).hexdigest()


def _generate_secret() -> str:
    return secrets.token_urlsafe(SECRET_BYTES)


async def create_api_key(
    db: AsyncSession,
    *,
    org_id: str,
    name: str,
    scopes: list[Scope],
    created_by: str,
    live: bool = True,
    expires_at: datetime | None = None,
) -> IssuedKey:
    """Generate, persist (hashed), and return a brand-new API key."""
    prefix = KEY_PREFIX_LIVE if live else KEY_PREFIX_TEST
    secret_value = _generate_secret()
    plaintext = f"{prefix}{secret_value}"

    record = APIKey(
        org_id=org_id,
        name=name,
        key_hash=_hash_secret(plaintext),
        prefix=prefix,
        scopes=[s.value for s in scopes],
        created_by=created_by,
        expires_at=expires_at,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)

    return IssuedKey(id=str(record.id), plaintext=plaintext, prefix=prefix, scopes=scopes)


async def revoke_api_key(db: AsyncSession, *, key_id: str, org_id: str) -> None:
    stmt = select(APIKey).where(APIKey.id == key_id, APIKey.org_id == org_id)
    result = await db.execute(stmt)
    record = result.scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")

    record.revoked_at = datetime.now(timezone.utc)
    await db.commit()


async def _resolve_api_key(
    raw_key: str | None,
    db: AsyncSession,
) -> APIKeyPrincipal:
    if not raw_key or not (raw_key.startswith(KEY_PREFIX_LIVE) or raw_key.startswith(KEY_PREFIX_TEST)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed API key",
        )

    key_hash = _hash_secret(raw_key)
    stmt = select(APIKey).where(APIKey.key_hash == key_hash)
    result = await db.execute(stmt)
    record = result.scalar_one_or_none()

    if record is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    now = datetime.now(timezone.utc)
    if record.revoked_at is not None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key has been revoked")
    if record.expires_at is not None and record.expires_at < now:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key has expired")

    record.last_used_at = now
    await db.commit()

    return APIKeyPrincipal(
        key_id=str(record.id),
        org_id=str(record.org_id),
        scopes={Scope(s) for s in record.scopes if s in Scope._value2member_map_},
        created_by=str(record.created_by),
    )


async def get_api_key_principal(
    raw_key: str | None = Security(_API_KEY_HEADER),
    db: AsyncSession = Depends(get_db),
) -> APIKeyPrincipal:
    """FastAPI dependency for routes/CI integrations authenticating via `X-Kodiak-Api-Key`."""
    return await _resolve_api_key(raw_key, db)


def require_api_key_scope(scope: Scope):
    """Dependency factory mirroring permissions.require_scope, for API-key-only routes."""

    async def dependency(
        principal: APIKeyPrincipal = Depends(get_api_key_principal),
    ) -> APIKeyPrincipal:
        if scope not in principal.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"API key is missing required scope: {scope.value}",
            )
        return principal

    return dependency

async def lookup_api_key(
    session: AsyncSession,
    raw_key: str,
) -> APIKey | None:
    """
    Resolve a raw API key to its corresponding active APIKey record.

    Returns:
        APIKey if found and active, otherwise None.
    """
    if not raw_key:
        return None

    key_hash = _hash_secret(raw_key)

    stmt = select(APIKey).where(
        APIKey.key_hash == key_hash,
        APIKey.revoked_at.is_(None),
    )

    result = await session.execute(stmt)
    record = result.scalar_one_or_none()

    if record is None:
        return None

    now = datetime.now(timezone.utc)

    if record.expires_at is not None and record.expires_at < now:
        return None

    record.last_used_at = now
    await session.commit()

    return record