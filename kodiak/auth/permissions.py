"""
kodiak/auth/permissions.py

Role-based access control for Kodiak.

Kodiak's permission model is intentionally coarse at the role level and
fine-grained at the scope level, because the same human role (e.g. "developer")
should not automatically be allowed to approve Kodiak's own merges into
protected branches -- that requires the explicit `approvals:write` scope,
typically granted only to "maintainer" and "owner" roles.

Used heavily by orchestration/approval_gate.py, which checks `approvals:write`
before letting an agent-authored PR merge autonomously.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from enum import StrEnum
from functools import wraps
from typing import Any

from fastapi import Depends, HTTPException, status

from kodiak.auth.jwt import TokenClaims, get_current_claims


class Role(StrEnum):
    OWNER = "owner"  # full control over an org's Kodiak instance
    MAINTAINER = "maintainer"  # can approve/merge, manage plugins & policies
    DEVELOPER = "developer"  # can trigger runs, review diffs, comment
    VIEWER = "viewer"  # read-only dashboard access
    SERVICE = "service"  # machine identity, e.g. webhook relays, CI bots


class Scope(StrEnum):
    RUNS_READ = "runs:read"
    RUNS_WRITE = "runs:write"  # trigger a new agent run
    RUNS_CANCEL = "runs:cancel"
    APPROVALS_READ = "approvals:read"
    APPROVALS_WRITE = "approvals:write"  # approve a PR/diff for autonomous merge
    PLUGINS_MANAGE = "plugins:manage"
    POLICIES_MANAGE = "policies:manage"
    MEMORY_READ = "memory:read"
    MEMORY_MANAGE = "memory:manage"  # purge/edit long-term memory
    BILLING_MANAGE = "billing:manage"
    ORG_MANAGE = "org:manage"  # invite/remove members, change roles


# Default scopes granted per role. Orgs may further restrict via
# policies/role_overrides without code changes (see policy.py).
ROLE_SCOPES: dict[Role, frozenset[Scope]] = {
    Role.OWNER: frozenset(Scope),  # everything
    Role.MAINTAINER: frozenset(
        {
            Scope.RUNS_READ,
            Scope.RUNS_WRITE,
            Scope.RUNS_CANCEL,
            Scope.APPROVALS_READ,
            Scope.APPROVALS_WRITE,
            Scope.PLUGINS_MANAGE,
            Scope.POLICIES_MANAGE,
            Scope.MEMORY_READ,
            Scope.MEMORY_MANAGE,
        }
    ),
    Role.DEVELOPER: frozenset(
        {
            Scope.RUNS_READ,
            Scope.RUNS_WRITE,
            Scope.RUNS_CANCEL,
            Scope.APPROVALS_READ,
            Scope.MEMORY_READ,
        }
    ),
    Role.VIEWER: frozenset({Scope.RUNS_READ, Scope.APPROVALS_READ, Scope.MEMORY_READ}),
    Role.SERVICE: frozenset({Scope.RUNS_READ, Scope.RUNS_WRITE, Scope.APPROVALS_READ}),
}


class PermissionDenied(HTTPException):
    def __init__(self, required: Scope) -> None:
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Missing required scope: {required.value}",
        )


def effective_scopes(roles: list[str]) -> set[Scope]:
    """Union of scopes granted by every role a principal holds."""
    scopes: set[Scope] = set()
    for raw_role in roles:
        try:
            role = Role(raw_role)
        except ValueError:
            continue
        scopes |= ROLE_SCOPES.get(role, frozenset())
    return scopes


def has_scope(claims: TokenClaims, scope: Scope) -> bool:
    if scope.value in claims.scopes:
        return True
    return scope in effective_scopes(claims.roles)


def require_scope(scope: Scope) -> Callable[[TokenClaims], Coroutine[Any, Any, TokenClaims]]:
    """
    FastAPI dependency factory.

    Usage:
        @router.post("/runs/{run_id}/approve")
        async def approve_run(
            run_id: str,
            claims: TokenClaims = Depends(require_scope(Scope.APPROVALS_WRITE)),
        ):
            ...
    """

    async def dependency(claims: TokenClaims = Depends(get_current_claims)) -> TokenClaims:
        if not has_scope(claims, scope):
            raise PermissionDenied(scope)
        return claims

    return dependency


def require_role(*allowed: Role) -> Callable[[TokenClaims], Coroutine[Any, Any, TokenClaims]]:
    """Coarser dependency for endpoints gated purely on role membership (e.g. org settings)."""

    async def dependency(claims: TokenClaims = Depends(get_current_claims)) -> TokenClaims:
        principal_roles = {r for r in claims.roles if r in {role.value for role in allowed}}
        if not principal_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of roles: {[r.value for r in allowed]}",
            )
        return claims

    return dependency


def guard(scope: Scope) -> Callable:
    """
    Decorator alternative to `require_scope` for non-FastAPI-route callables,
    e.g. agent tools invoked from orchestration/tool_router.py that still need
    to respect the human's permission boundary (`approvals:write` before an
    agent calls github.pr_manager.merge()).
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, claims: TokenClaims, **kwargs):
            if not has_scope(claims, scope):
                raise PermissionDenied(scope)
            return await func(*args, claims=claims, **kwargs)

        return wrapper

    return decorator
