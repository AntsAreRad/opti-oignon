#!/usr/bin/env python3
"""Tests for RBAC enforcement (rbac_enforcement) — the authorization decisions.

These are the access-control predicates that protect user data isolation on a
multi-user deployment: who is an admin, which roles may reach an endpoint, and
crucially whether a caller may touch a resource (or query data) that belongs to
another user. Admins bypass ownership; everyone else is confined to their own
``sub``.

The functions are FastAPI dependencies whose ``current_user`` argument is
normally injected via ``Depends(get_current_user)``. The tests bypass that by
calling each function directly with an explicit ``current_user`` dict, which
isolates the pure authorization logic. Denials surface as ``HTTPException`` and
are asserted by status code (401 = unidentified, 403 = forbidden).
"""

import pytest
from fastapi import HTTPException

from opti_oignon.rbac_enforcement import (
    enforce_user_ownership,
    get_effective_user_id,
    get_user_id,
    get_user_role,
    is_admin,
    require_admin,
    require_role,
)

ADMIN = {"sub": "admin-1", "role": "admin"}
USER = {"sub": "u1", "role": "user"}
OTHER = {"sub": "u2", "role": "user"}


def _status(fn, *args):
    with pytest.raises(HTTPException) as exc:
        fn(*args)
    return exc.value.status_code


# ===========================================================================
# is_admin (non-raising predicate)
# ===========================================================================

def test_is_admin_true():
    assert is_admin(ADMIN) is True


def test_is_admin_false_for_user_and_missing_role():
    assert is_admin(USER) is False
    assert is_admin({}) is False


# ===========================================================================
# get_user_id / get_user_role
# ===========================================================================

def test_get_user_id_returns_sub():
    assert get_user_id({"sub": "u123"}) == "u123"


def test_get_user_id_missing_sub_is_401():
    assert _status(get_user_id, {}) == 401
    assert _status(get_user_id, {"sub": ""}) == 401


def test_get_user_role_returns_role_or_viewer_default():
    assert get_user_role({"role": "user"}) == "user"
    assert get_user_role({}) == "viewer"


# ===========================================================================
# require_admin
# ===========================================================================

def test_require_admin_allows_admin():
    assert require_admin(ADMIN) == ADMIN


def test_require_admin_rejects_non_admin_403():
    assert _status(require_admin, USER) == 403
    assert _status(require_admin, {}) == 403


# ===========================================================================
# require_role factory
# ===========================================================================

def test_require_role_allows_listed_role():
    dep = require_role("admin", "user")
    assert dep(USER) == USER


def test_require_role_rejects_unlisted_role_403():
    dep = require_role("admin", "user")
    assert _status(dep, {"sub": "v", "role": "viewer"}) == 403


def test_require_role_uses_viewer_default_for_missing_role():
    dep = require_role("admin")          # viewer not allowed
    assert _status(dep, {"sub": "x"}) == 403


# ===========================================================================
# enforce_user_ownership (data isolation)
# ===========================================================================

def test_ownership_owner_is_allowed():
    assert enforce_user_ownership("u1", USER) is None


def test_ownership_admin_bypasses():
    # An admin may reach a resource owned by someone else.
    assert enforce_user_ownership("u1", ADMIN) is None


def test_ownership_other_user_denied_403():
    assert _status(enforce_user_ownership, "u1", OTHER) == 403


# ===========================================================================
# get_effective_user_id (cross-user query prevention)
# ===========================================================================

def test_effective_none_returns_caller():
    assert get_effective_user_id(None, USER) == "u1"


def test_effective_self_returns_self():
    assert get_effective_user_id("u1", USER) == "u1"


def test_effective_admin_may_target_another_user():
    assert get_effective_user_id("u2", ADMIN) == "u2"


def test_effective_non_admin_targeting_other_is_denied_403():
    assert _status(get_effective_user_id, "u2", USER) == 403
