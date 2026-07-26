#!/usr/bin/env python3
"""
RBAC enforcement and user-context dependencies for Opti-Oignon.

Provides reusable FastAPI dependencies for:
- Extracting authenticated user from JWT (cookie or Bearer header)
- Role-based access control (admin, user, viewer)
- User-ID extraction for data isolation
- Admin-only route protection
- Per-user data filtering helpers

All API routes should use these dependencies instead of duplicating
auth logic. The auth middleware handles deny-by-default; these
dependencies provide fine-grained per-route control.

Usage in routes::

    from opti_oignon.rbac_enforcement import (
        get_current_user,
        get_user_id,
        require_admin,
        require_role,
    )

    @router.get("/my-data")
    async def my_data(user_id: str = Depends(get_user_id)):
        # user_id is guaranteed to be the authenticated user's ID
        return fetch_data(user_id=user_id)

    @router.delete("/admin/purge")
    async def purge(admin: dict = Depends(require_admin)):
        ...
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Depends, Header, HTTPException, Request

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cookie config (must match routes_auth.py)
# ---------------------------------------------------------------------------

_ACCESS_COOKIE = "oo_access_token"


def _get_auth_manager() -> Any:
    """Lazy-load the AuthManager singleton."""
    try:
        from opti_oignon.auth import auth_manager
        return auth_manager
    except Exception:
        return None


def _is_bulbe() -> bool:
    """Check whether Bulbe mode is active."""
    try:
        from opti_oignon.security_mode import is_bulbe
        return is_bulbe()
    except Exception:
        return False


def _is_cookie_mode() -> bool:
    """Check whether cookie-based auth is enabled."""
    try:
        from opti_oignon.api.routes_auth import _is_cookie_mode as _cm
        return _cm()
    except Exception:
        return True  # Default to cookie mode


# ---------------------------------------------------------------------------
# Core dependency: get_current_user
# ---------------------------------------------------------------------------


def get_current_user(
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Extract and validate the current user from JWT.

    In single-user mode (non-Bulbe), returns a synthetic local admin user.
    Otherwise, validates the JWT from cookie or Bearer header.

    Returns a dict with at least: sub, username, role, type.

    Raises:
        HTTPException 401: If no valid token is found.
    """
    mgr = _get_auth_manager()

    bulbe_active = _is_bulbe()

    # Single-user mode bypass (except in Bulbe)
    if mgr is not None and mgr.single_user_mode and not bulbe_active:
        return {
            "sub": "local",
            "username": "local",
            "role": "admin",
            "type": "access",
        }

    # Extract token: cookie first, then header
    token = None
    if _is_cookie_mode():
        token = request.cookies.get(_ACCESS_COOKIE)

    if not token and authorization:
        parts = authorization.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
        )

    if mgr is None:
        raise HTTPException(
            status_code=503,
            detail="Authentication service unavailable.",
        )

    payload = mgr.validate_token(token)
    if payload is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token.",
        )

    return payload


# ---------------------------------------------------------------------------
# Derived dependencies
# ---------------------------------------------------------------------------


def get_user_id(current_user: dict[str, Any] = Depends(get_current_user)) -> str:
    """Extract the user_id (sub claim) from the authenticated user.

    This is the primary dependency for data-isolation: any route that
    accesses user-scoped data should depend on this.

    Returns:
        The user_id string (e.g. "local", or a UUID).
    """
    user_id = current_user.get("sub", "")
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Token missing user identifier.",
        )
    return user_id


def get_user_role(current_user: dict[str, Any] = Depends(get_current_user)) -> str:
    """Extract the role from the authenticated user.

    Returns:
        The role string (e.g. "admin", "user", "viewer").
    """
    return current_user.get("role", "viewer")


def require_admin(
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Require the current user to be an admin.

    Raises:
        HTTPException 403: If the user is not an admin.

    Returns:
        The current user dict.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin access required.",
        )
    return current_user


def require_role(*allowed_roles: str):
    """Factory for role-based access control dependency.

    Usage::

        @router.get("/edit")
        async def edit(user=Depends(require_role("admin", "user"))):
            ...

    Args:
        *allowed_roles: Roles that are allowed to access the endpoint.

    Returns:
        A FastAPI dependency function.
    """
    roles = frozenset(allowed_roles)

    def _check_role(
        current_user: dict[str, Any] = Depends(get_current_user),
    ) -> dict[str, Any]:
        user_role = current_user.get("role", "viewer")
        if user_role not in roles:
            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions. Required role: {}".format(
                    " or ".join(sorted(roles))
                ),
            )
        return current_user

    return _check_role


# ---------------------------------------------------------------------------
# Data isolation helpers
# ---------------------------------------------------------------------------


def enforce_user_ownership(
    resource_user_id: str,
    current_user: dict[str, Any],
) -> None:
    """Verify that the current user owns a resource.

    Admins can access any resource. Non-admins can only access their own.

    Args:
        resource_user_id: The user_id that owns the resource.
        current_user: The authenticated user dict.

    Raises:
        HTTPException 403: If the user does not own the resource.
    """
    if current_user.get("role") == "admin":
        return
    if current_user.get("sub") != resource_user_id:
        raise HTTPException(
            status_code=403,
            detail="Access denied: resource belongs to another user.",
        )


def get_effective_user_id(
    target_user_id: str | None,
    current_user: dict[str, Any],
) -> str:
    """Resolve the effective user_id for data queries.

    - If target_user_id is provided and the current user is admin,
      returns target_user_id (admin can query any user's data).
    - If target_user_id is provided but doesn't match current user,
      raises 403.
    - If target_user_id is None, returns the current user's ID.

    Args:
        target_user_id: Optional explicit user_id to query.
        current_user: The authenticated user dict.

    Returns:
        The user_id to use for data filtering.
    """
    caller_id = current_user.get("sub", "")

    if target_user_id is None:
        return caller_id

    if target_user_id == caller_id:
        return caller_id

    # Different user requested — admin only
    if current_user.get("role") == "admin":
        return target_user_id

    raise HTTPException(
        status_code=403,
        detail="Cannot access another user's data.",
    )


def is_admin(current_user: dict[str, Any]) -> bool:
    """Check if the current user is an admin (non-raising)."""
    return current_user.get("role") == "admin"


# ---------------------------------------------------------------------------
# Admin audit helper
# ---------------------------------------------------------------------------


def log_admin_action(
    admin_user: dict[str, Any],
    action: str,
    target_type: str,
    target_id: str,
    details: str = "",
) -> None:
    """Log an admin action to the audit trail.

    This is a convenience wrapper that delegates to the admin_audit module.

    Args:
        admin_user: The admin user dict.
        action: Action performed (e.g. "delete_user_data", "export_user_data").
        target_type: Type of target (e.g. "user", "conversation").
        target_id: ID of the target.
        details: Optional additional details.
    """
    try:
        from opti_oignon.admin_audit import log_admin_event
        log_admin_event(
            admin_id=admin_user.get("sub", "unknown"),
            action=action,
            target_type=target_type,
            target_id=target_id,
            details=details,
        )
    except ImportError:
        logger.warning("admin_audit module not available, skipping audit log")
    except Exception as e:
        logger.error("Failed to log admin action: %s", e)


# ---------------------------------------------------------------------------
# Module availability flag
# ---------------------------------------------------------------------------

RBAC_ENFORCEMENT_AVAILABLE = True
