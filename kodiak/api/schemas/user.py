"""
kodiak/api/schemas/user.py

Pydantic schemas for the User/auth resources, derived from the auth router
(kodiak/api/routers/auth.py) as the source of truth.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator


class UserCreate(BaseModel):
    email: str
    password: str
    username: str | None = None
    display_name: str | None = None

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        if "@" not in normalized or normalized.startswith("@") or normalized.endswith("@"):
            raise ValueError("Invalid email address")
        return normalized


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    username: str
    display_name: str | None
    avatar_url: str | None
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime
