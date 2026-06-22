from __future__ import annotations

import secrets
import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kodiak.api.dependencies import CurrentUser, get_db_session
from kodiak.api.schemas.user import RefreshRequest, TokenResponse, UserCreate, UserResponse
from kodiak.auth.jwt import create_access_token, create_refresh_token, verify_refresh_token
from kodiak.auth.oauth import exchange_code, fetch_github_user, fetch_primary_email
from kodiak.db.models.user import User

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])
_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    body: UserCreate,
    session: AsyncSession = Depends(get_db_session),
) -> User:
    existing = await session.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status.HTTP_409_CONFLICT, detail="Email already registered")

    user = User(
        email=body.email,
        display_name=body.display_name,
        hashed_password=_pwd.hash(body.password),
        is_active=True,
    )
    session.add(user)
    await session.flush()
    return user


@router.post("/login", response_model=TokenResponse)
async def login(
    body: UserCreate,
    session: AsyncSession = Depends(get_db_session),
) -> TokenResponse:
    result = await session.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()

    if user is None or not user.hashed_password or not _pwd.verify(body.password, user.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Account disabled")

    return TokenResponse(
        access_token=create_access_token(str(user.id)),
        refresh_token=create_refresh_token(str(user.id)),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    body: RefreshRequest,
    session: AsyncSession = Depends(get_db_session),
) -> TokenResponse:
    try:
        user_id_str = verify_refresh_token(body.refresh_token)
    except ValueError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc

    result = await session.execute(
        select(User).where(User.id == uuid.UUID(user_id_str), User.is_active.is_(True))
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return TokenResponse(
        access_token=create_access_token(str(user.id)),
        refresh_token=create_refresh_token(str(user.id)),
    )


@router.get("/github/callback", response_model=TokenResponse)
async def github_callback(
    code: str,
    state: str,
    session: AsyncSession = Depends(get_db_session),
) -> TokenResponse:
    try:
        token_data = await exchange_code(code)
        access_token = token_data["access_token"]
        gh_user = await fetch_github_user(access_token)
        email = await fetch_primary_email(access_token) or gh_user.get("email", "")
    except Exception as exc:
        logger.warning("github_oauth.failed", error=str(exc))
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="GitHub OAuth failed") from exc

    result = await session.execute(
        select(User).where(User.github_user_id == gh_user["id"])
    )
    user = result.scalar_one_or_none()

    if user is None:
        user = User(
            email=email,
            display_name=gh_user.get("name") or gh_user.get("login"),
            avatar_url=gh_user.get("avatar_url"),
            github_user_id=gh_user["id"],
            github_username=gh_user.get("login"),
            github_access_token=access_token,
            is_active=True,
            is_verified=True,
        )
        session.add(user)
        await session.flush()
    else:
        user.github_access_token = access_token
        user.avatar_url = gh_user.get("avatar_url")

    return TokenResponse(
        access_token=create_access_token(str(user.id)),
        refresh_token=create_refresh_token(str(user.id)),
    )


@router.get("/me", response_model=UserResponse)
async def me(current_user: CurrentUser) -> User:
    return current_user