"""Regression coverage for JWT configuration and token-type enforcement."""

from __future__ import annotations

import pytest

from kodiak.auth import jwt
from kodiak.config.settings import settings


def test_jwt_uses_canonical_settings_without_public_fallback() -> None:
    assert jwt.SECRET_KEY == settings.SECRET_KEY
    assert jwt.ALGORITHM == settings.JWT_ALGORITHM
    assert jwt.SECRET_KEY != "change-me"


def test_access_and_refresh_tokens_are_not_interchangeable() -> None:
    access = jwt.create_access_token("user-1")
    refresh = jwt.create_refresh_token("user-1")

    assert jwt.verify_access_token(access).sub == "user-1"
    assert jwt.verify_refresh_token(refresh) == "user-1"
    with pytest.raises(ValueError, match="access token"):
        jwt.verify_access_token(refresh)
    with pytest.raises(ValueError, match="refresh token"):
        jwt.verify_refresh_token(access)
