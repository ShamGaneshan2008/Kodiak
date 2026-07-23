"""
kodiak/auth/oauth.py

GitHub OAuth integration: exchanges an authorization code for an access
token, then resolves the authenticated GitHub user's profile and primary
verified email. Used by the auth router during the login/callback flow.
"""

from __future__ import annotations

import httpx

from kodiak.config.settings import settings

_GITHUB_OAUTH_TOKEN_URL = "https://github.com/login/oauth/access_token"
_GITHUB_USER_URL = "https://api.github.com/user"
_GITHUB_USER_EMAILS_URL = "https://api.github.com/user/emails"

_REQUEST_TIMEOUT = 10.0


class OAuthError(Exception):
    """Raised when the GitHub OAuth flow or a subsequent API call fails."""


async def exchange_code(code: str) -> dict:
    """Exchange a GitHub OAuth authorization code for an access token."""
    payload = {
        "client_id": settings.GITHUB_CLIENT_ID,
        "client_secret": settings.GITHUB_CLIENT_SECRET,
        "code": code,
        "redirect_uri": settings.GITHUB_REDIRECT_URI,
    }
    headers = {"Accept": "application/json"}

    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        try:
            response = await client.post(_GITHUB_OAUTH_TOKEN_URL, data=payload, headers=headers)
        except httpx.RequestError as exc:
            raise OAuthError(f"Network error while exchanging OAuth code: {exc}") from exc

    if response.status_code != 200:
        raise OAuthError(
            f"GitHub token exchange failed with status {response.status_code}: {response.text}"
        )

    data = response.json()

    if "error" in data:
        description = data.get("error_description", data["error"])
        raise OAuthError(f"GitHub OAuth error: {description}")

    if "access_token" not in data:
        raise OAuthError(f"GitHub token exchange response missing access_token: {data}")

    return data


async def fetch_github_user(access_token: str) -> dict:
    """Fetch the authenticated GitHub user's profile."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/vnd.github+json",
    }

    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        try:
            response = await client.get(_GITHUB_USER_URL, headers=headers)
        except httpx.RequestError as exc:
            raise OAuthError(f"Network error while fetching GitHub user profile: {exc}") from exc

    if response.status_code != 200:
        raise OAuthError(
            f"GitHub user profile fetch failed with status {response.status_code}: {response.text}"
        )

    return response.json()


async def fetch_primary_email(access_token: str) -> str:
    """Fetch the user's primary verified email from GitHub."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/vnd.github+json",
    }

    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        try:
            response = await client.get(_GITHUB_USER_EMAILS_URL, headers=headers)
        except httpx.RequestError as exc:
            raise OAuthError(f"Network error while fetching GitHub user emails: {exc}") from exc

    if response.status_code != 200:
        raise OAuthError(
            f"GitHub email fetch failed with status {response.status_code}: {response.text}"
        )

    emails = response.json()

    for entry in emails:
        if entry.get("primary") and entry.get("verified"):
            return entry["email"]

    for entry in emails:
        if entry.get("verified"):
            return entry["email"]

    raise OAuthError("No verified email found on GitHub account")
