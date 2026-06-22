"""
kodiak/auth/oauth.py

GitHub OAuth and GitHub App authentication flows.

Two distinct identities are handled here:
    1. User OAuth  - a human signing into the Kodiak dashboard with their
                     GitHub account ("Login with GitHub").
    2. App auth    - the Kodiak GitHub App authenticating itself to act on
                     a repository (issue comments, PR creation, checks).

GitHub App installation tokens are short-lived (1 hour) and cached in Redis
to avoid hammering GitHub's token endpoint on every webhook/event.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx
import jwt as pyjwt
from fastapi import HTTPException, status

from kodiak.config.settings import get_settings
from kodiak.db.session import get_redis

settings = get_settings()

GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"
GITHUB_INSTALLATION_TOKEN_URL = "https://api.github.com/app/installations/{installation_id}/access_tokens"

INSTALLATION_TOKEN_CACHE_PREFIX = "github:installation_token:"
INSTALLATION_TOKEN_SAFETY_MARGIN_SECONDS = 60


@dataclass
class GitHubUser:
    id: int
    login: str
    name: str | None
    email: str | None
    avatar_url: str | None


@dataclass
class InstallationToken:
    token: str
    expires_at: int  # unix timestamp


def build_authorize_url(*, state: str, scopes: tuple[str, ...] = ("read:user", "user:email")) -> str:
    """Construct the GitHub OAuth consent URL for the dashboard login flow."""
    params = {
        "client_id": settings.GITHUB_OAUTH_CLIENT_ID,
        "redirect_uri": settings.GITHUB_OAUTH_REDIRECT_URI,
        "scope": " ".join(scopes),
        "state": state,
        "allow_signup": "false",
    }
    query = "&".join(f"{k}={httpx.QueryParams({k: v})[k]}" for k, v in params.items())
    return f"{GITHUB_AUTHORIZE_URL}?{query}"


async def exchange_code_for_user(code: str) -> GitHubUser:
    """Exchange a one-time OAuth code for an access token, then fetch the profile."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        token_resp = await client.post(
            GITHUB_TOKEN_URL,
            headers={"Accept": "application/json"},
            data={
                "client_id": settings.GITHUB_OAUTH_CLIENT_ID,
                "client_secret": settings.GITHUB_OAUTH_CLIENT_SECRET,
                "code": code,
                "redirect_uri": settings.GITHUB_OAUTH_REDIRECT_URI,
            },
        )
        token_resp.raise_for_status()
        token_data = token_resp.json()

        if "error" in token_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"GitHub OAuth error: {token_data.get('error_description', token_data['error'])}",
            )

        access_token = token_data["access_token"]

        user_resp = await client.get(
            GITHUB_USER_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/vnd.github+json",
            },
        )
        user_resp.raise_for_status()
        profile = user_resp.json()

    return GitHubUser(
        id=profile["id"],
        login=profile["login"],
        name=profile.get("name"),
        email=profile.get("email"),
        avatar_url=profile.get("avatar_url"),
    )


def _generate_app_jwt() -> str:
    """Sign a short-lived JWT as the GitHub App itself (used to mint installation tokens)."""
    now = int(time.time())
    payload = {
        "iat": now - 60,
        "exp": now + (9 * 60),
        "iss": settings.GITHUB_APP_ID,
    }
    return pyjwt.encode(payload, settings.GITHUB_APP_PRIVATE_KEY, algorithm="RS256")


async def get_installation_token(installation_id: int) -> InstallationToken:
    """
    Return a valid installation access token for a given GitHub App installation,
    reusing a cached token from Redis when it has not yet expired.
    """
    redis = await get_redis()
    cache_key = f"{INSTALLATION_TOKEN_CACHE_PREFIX}{installation_id}"

    cached = await redis.hgetall(cache_key)
    if cached and int(cached.get("expires_at", 0)) - INSTALLATION_TOKEN_SAFETY_MARGIN_SECONDS > int(time.time()):
        return InstallationToken(token=cached["token"], expires_at=int(cached["expires_at"]))

    app_jwt = _generate_app_jwt()
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            GITHUB_INSTALLATION_TOKEN_URL.format(installation_id=installation_id),
            headers={
                "Authorization": f"Bearer {app_jwt}",
                "Accept": "application/vnd.github+json",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    expires_at = int(time.mktime(time.strptime(data["expires_at"], "%Y-%m-%dT%H:%M:%SZ")))
    token = InstallationToken(token=data["token"], expires_at=expires_at)

    await redis.hset(cache_key, mapping={"token": token.token, "expires_at": token.expires_at})
    await redis.expireat(cache_key, expires_at)

    return token


async def revoke_installation_token(installation_id: int) -> None:
    """Drop a cached installation token, e.g. on app uninstall or permission change."""
    redis = await get_redis()
    await redis.delete(f"{INSTALLATION_TOKEN_CACHE_PREFIX}{installation_id}")