#!/usr/bin/env python3
"""
API routes for multi-user data isolation.

Endpoints:
  GET    /api/users/{user_id}/export    — Export all user data (GDPR)
  DELETE /api/users/{user_id}/data      — Cascade delete all user data
  GET    /api/admin/audit               — Query admin audit log
  GET    /api/admin/audit/count         — Count admin audit events
  GET    /api/users/me/key-status       — Per-user encryption key status
  POST   /api/users/me/derive-key       — Derive and cache user encryption key
  DELETE /api/users/me/key-cache        — Wipe cached user encryption key
  GET    /api/users/{user_id}/plugins   — Get per-user plugin configs
  PUT    /api/users/{user_id}/plugins/{plugin} — Set per-user plugin config
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["users"])


# ---------------------------------------------------------------------------
# Rate limiting (SA-155-051)
# ---------------------------------------------------------------------------

def _get_client_ip(request: Request) -> str:
    """Extract client IP from request, considering proxy headers."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _check_user_mgmt_rate(request: Request) -> None:
    """Rate limit check for user management endpoints.

    Raises HTTPException 429 if rate limit exceeded.
    """
    try:
        from opti_oignon.rate_limiter import rate_limit_check
    except ImportError:
        return  # Graceful degradation if module not available

    client_ip = _get_client_ip(request)
    allowed, info = rate_limit_check("user_management", key=client_ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=info["message"],
            headers={"Retry-After": str(int(info["retry_after"] + 1))},
        )


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class DeriveKeyRequest(BaseModel):
    """Request body for key derivation."""
    password: str = Field(..., min_length=1, description="User password for key derivation")


class PluginConfigRequest(BaseModel):
    """Request body for per-user plugin config."""
    enabled: bool | None = Field(default=None, description="Enable/disable plugin for user")
    preferences: dict[str, Any] | None = Field(default=None, description="Plugin preferences")


class DeleteDataResponse(BaseModel):
    """Response for user data deletion."""
    user_id: str
    deleted_at: float
    conversations: int = 0
    memories: int = 0
    rag_collections: int = 0
    plugin_configs: int = 0
    settings: bool = False
    encryption_keys: bool = False
    # REV-2: identity-bound plugin reviews join the cascade.
    plugin_reviews: int = 0
    # UD-03: stores the per-user wipe cannot cover today
    # (single-user / unscoped); never empty until the scoping cycle.
    not_covered: list[str] = []
    # UD-04: stores deliberately retained on wipe (audit trails);
    # surfaced so a wipe never silently implies their erasure.
    retained_by_design: list[str] = []


# ---------------------------------------------------------------------------
# Dependencies (lazy imports)
# ---------------------------------------------------------------------------


def _get_rbac():
    """Lazy-load RBAC functions."""
    try:
        from opti_oignon.rbac_enforcement import (
            enforce_user_ownership,
            get_current_user,
            get_effective_user_id,
            get_user_id,
            is_admin,
            log_admin_action,
            require_admin,
        )
        return {
            "get_current_user": get_current_user,
            "get_user_id": get_user_id,
            "require_admin": require_admin,
            "enforce_user_ownership": enforce_user_ownership,
            "get_effective_user_id": get_effective_user_id,
            "is_admin": is_admin,
            "log_admin_action": log_admin_action,
        }
    except ImportError:
        return None


# We need to define these at module level for Depends() to work
try:
    from opti_oignon.rbac_enforcement import (
        get_current_user,
        require_admin,
    )
    from opti_oignon.rbac_enforcement import (
        get_user_id as _get_user_id,
    )
except ImportError:
    # Stubs for when module is loaded standalone
    def get_current_user() -> dict:  # type: ignore[misc]
        raise HTTPException(503, "RBAC module unavailable")

    def _get_user_id() -> str:  # type: ignore[misc]
        raise HTTPException(503, "RBAC module unavailable")

    def require_admin() -> dict:  # type: ignore[misc]
        raise HTTPException(503, "RBAC module unavailable")


# ---------------------------------------------------------------------------
# User data export (GDPR)
# ---------------------------------------------------------------------------


@router.get("/api/users/{user_id}/export")
async def export_user_data(
    user_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Export all data for a user (GDPR data portability).

    - Users can export their own data.
    - Admins can export any user's data.
    """
    _check_user_mgmt_rate(request)

    # Authorization: user can export own data, admin can export any
    caller_id = current_user.get("sub", "")
    if caller_id != user_id and current_user.get("role") != "admin":
        raise HTTPException(403, "Cannot export another user's data.")

    try:
        from opti_oignon.user_data_manager import get_user_data_exporter
        exporter = get_user_data_exporter()
        data = exporter.export(user_id)

        # Log admin action if admin is exporting another user's data
        if caller_id != user_id:
            try:
                from opti_oignon.admin_audit import log_admin_event
                log_admin_event(
                    admin_id=caller_id,
                    action="export_user_data",
                    target_type="user",
                    target_id=user_id,
                )
            except ImportError:
                pass

        return data
    except Exception as e:
        logger.error("Failed to export data for user %s: %s", user_id, e)
        raise HTTPException(500, "Failed to export user data.")


# ---------------------------------------------------------------------------
# User data deletion (cascade)
# ---------------------------------------------------------------------------


@router.delete("/api/users/{user_id}/data")
async def delete_user_data(
    user_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
) -> DeleteDataResponse:
    """Cascade delete all data for a user (GDPR right to erasure).

    - Users can delete their own data.
    - Admins can delete any user's data.
    """
    _check_user_mgmt_rate(request)

    caller_id = current_user.get("sub", "")
    if caller_id != user_id and current_user.get("role") != "admin":
        raise HTTPException(403, "Cannot delete another user's data.")

    try:
        from opti_oignon.user_data_manager import get_user_data_deleter
        deleter = get_user_data_deleter()
        admin_id = caller_id if caller_id != user_id else None
        result = deleter.delete_all(user_id=user_id, admin_id=admin_id)
        return DeleteDataResponse(**result)
    except Exception as e:
        logger.error("Failed to delete data for user %s: %s", user_id, e)
        raise HTTPException(500, "Failed to delete user data.")


# ---------------------------------------------------------------------------
# Admin audit log
# ---------------------------------------------------------------------------


@router.get("/api/admin/audit")
async def get_admin_audit(
    admin: dict = Depends(require_admin),
    admin_id: str | None = Query(default=None, description="Filter by admin ID"),
    target_type: str | None = Query(default=None, description="Filter by target type"),
    target_id: str | None = Query(default=None, description="Filter by target ID"),
    since: float | None = Query(default=None, description="Events after this UNIX timestamp"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Query admin audit log (admin only)."""
    try:
        from opti_oignon.admin_audit import get_admin_audit_store
        store = get_admin_audit_store()
        events = store.get_events(
            admin_id=admin_id,
            target_type=target_type,
            target_id=target_id,
            since=since,
            limit=limit,
            offset=offset,
        )
        total = store.count_events(admin_id=admin_id, target_type=target_type)
        return {"events": events, "total": total, "limit": limit, "offset": offset}
    except ImportError:
        raise HTTPException(503, "Admin audit module unavailable.")


@router.get("/api/admin/audit/count")
async def count_admin_audit(
    admin: dict = Depends(require_admin),
    admin_id: str | None = Query(default=None),
    target_type: str | None = Query(default=None),
) -> dict[str, int]:
    """Count admin audit events (admin only)."""
    try:
        from opti_oignon.admin_audit import get_admin_audit_store
        store = get_admin_audit_store()
        count = store.count_events(admin_id=admin_id, target_type=target_type)
        return {"count": count}
    except ImportError:
        raise HTTPException(503, "Admin audit module unavailable.")


# ---------------------------------------------------------------------------
# Per-user encryption key management
# ---------------------------------------------------------------------------


@router.get("/api/users/me/key-status")
async def get_key_status(
    user_id: str = Depends(_get_user_id),
) -> dict[str, Any]:
    """Get per-user encryption key status."""
    try:
        from opti_oignon.user_key_manager import get_user_key_manager
        mgr = get_user_key_manager()
        has_salt = mgr.salt_store.has_salt(user_id)
        is_cached = mgr.is_key_cached(user_id)
        status = mgr.get_status()
        return {
            "user_id": user_id,
            "has_salt": has_salt,
            "key_cached": is_cached,
            **status,
        }
    except ImportError:
        raise HTTPException(503, "User key manager unavailable.")


@router.post("/api/users/me/derive-key")
async def derive_user_key(
    req: DeriveKeyRequest,
    request: Request,
    user_id: str = Depends(_get_user_id),
) -> dict[str, Any]:
    """Derive and cache per-user encryption key from password."""
    _check_user_mgmt_rate(request)

    try:
        from opti_oignon.user_key_manager import get_user_key_manager
        mgr = get_user_key_manager()
        success = mgr.derive_and_cache(user_id, req.password)
        if not success:
            raise HTTPException(500, "Key derivation failed.")
        return {"status": "ok", "user_id": user_id, "key_cached": True}
    except ImportError:
        raise HTTPException(503, "User key manager unavailable.")


@router.delete("/api/users/me/key-cache")
async def wipe_key_cache(
    user_id: str = Depends(_get_user_id),
) -> dict[str, Any]:
    """Wipe cached per-user encryption key (logout)."""
    try:
        from opti_oignon.user_key_manager import get_user_key_manager
        mgr = get_user_key_manager()
        wiped = mgr.wipe_user_key(user_id)
        return {"status": "ok", "user_id": user_id, "wiped": wiped}
    except ImportError:
        raise HTTPException(503, "User key manager unavailable.")


# ---------------------------------------------------------------------------
# Per-user plugin configurations
# ---------------------------------------------------------------------------


@router.get("/api/users/{user_id}/plugins")
async def get_user_plugins(
    user_id: str,
    current_user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Get per-user plugin configurations."""
    caller_id = current_user.get("sub", "")
    if caller_id != user_id and current_user.get("role") != "admin":
        raise HTTPException(403, "Cannot access another user's plugin configs.")

    try:
        from opti_oignon.plugin_user_config import get_plugin_user_config_store
        store = get_plugin_user_config_store()
        configs = store.get_all_configs(user_id)
        return {"user_id": user_id, "plugins": configs}
    except ImportError:
        raise HTTPException(503, "Plugin user config module unavailable.")


@router.put("/api/users/{user_id}/plugins/{plugin_name}")
async def set_user_plugin(
    user_id: str,
    plugin_name: str,
    req: PluginConfigRequest,
    current_user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Set per-user plugin configuration."""
    caller_id = current_user.get("sub", "")
    if caller_id != user_id and current_user.get("role") != "admin":
        raise HTTPException(403, "Cannot modify another user's plugin configs.")

    try:
        from opti_oignon.plugin_user_config import get_plugin_user_config_store
        store = get_plugin_user_config_store()
        config = store.set_config(
            user_id=user_id,
            plugin_name=plugin_name,
            enabled=req.enabled,
            preferences=req.preferences,
        )
        return config
    except ImportError:
        raise HTTPException(503, "Plugin user config module unavailable.")


# ---------------------------------------------------------------------------
# Module flag
# ---------------------------------------------------------------------------

ROUTES_USERS_AVAILABLE = True
