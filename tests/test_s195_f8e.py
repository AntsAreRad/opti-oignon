#!/usr/bin/env python3
"""
Tests for S195 F8e -- plugin marketplace routes.

Coverage:
- MKT-01: the marketplace router carries the per-router auth dependency
          (defense-in-depth parity with routes_plugins / S136)
- auth coverage invariant: marketplace/reviews/template/install paths are
  NOT in the AuthMiddleware public allowlist (so the deny-by-default
  middleware enforces auth on them)
- BMK-11 not reachable: browse/search/reviews bound limit with ge=1
- route ordering: no bare GET /{name} single-segment route that could
  shadow the /marketplace literals

The marketplace router is a thin route layer; the substantive plugin
behavior was audited and fixed upstream in F8a..F8d. These tests lock the
verified properties.

Loader idiom: spec_from_file_location with sys.modules registration BEFORE
exec_module; package stubs (opti_oignon, opti_oignon.api) with real
__path__; added modules cleaned at module teardown (S194 hardening).
"""

import importlib.util
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent
MARKETPLACE_SRC = (
    ROOT / "opti_oignon" / "api" / "routes_plugin_marketplace.py"
).read_text()

_ADDED_MODULES: list[str] = []


def _seed_stub(name: str, mod: ModuleType) -> None:
    if name not in sys.modules:
        sys.modules[name] = mod
        _ADDED_MODULES.append(name)


def _load_auth_middleware() -> ModuleType:
    name = "opti_oignon.api.auth_middleware"
    if name in sys.modules:
        return sys.modules[name]
    if "opti_oignon" not in sys.modules:
        pkg = ModuleType("opti_oignon")
        pkg.__path__ = [str(ROOT / "opti_oignon")]
        _seed_stub("opti_oignon", pkg)
    if "opti_oignon.api" not in sys.modules:
        apipkg = ModuleType("opti_oignon.api")
        apipkg.__path__ = [str(ROOT / "opti_oignon" / "api")]
        _seed_stub("opti_oignon.api", apipkg)
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "opti_oignon" / "api" / "auth_middleware.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    _ADDED_MODULES.append(name)
    spec.loader.exec_module(mod)
    return mod


auth_mw = _load_auth_middleware()


@pytest.fixture(scope="module", autouse=True)
def _cleanup_added_modules():
    yield
    for name in _ADDED_MODULES:
        sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# MKT-01 -- auth dependency parity
# ---------------------------------------------------------------------------

class TestMKT01AuthParity:
    def test_router_declares_auth_dependency(self):
        assert "dependencies=_auth_dep" in MARKETPLACE_SRC
        assert "from .routes_auth import _get_current_user" in MARKETPLACE_SRC


# ---------------------------------------------------------------------------
# Auth coverage invariant -- marketplace paths are not public
# ---------------------------------------------------------------------------

class TestMarketplaceNotPublic:
    @pytest.mark.parametrize("path", [
        "/api/plugins/marketplace",
        "/api/plugins/marketplace/search",
        "/api/plugins/marketplace/install",
        "/api/plugins/marketplace/template",
        "/api/plugins/some-plugin/reviews",
    ])
    def test_marketplace_paths_require_auth(self, path):
        assert auth_mw._is_public_path(path) is False

    def test_public_allowlist_sanity(self):
        # Genuinely public paths still resolve as public.
        assert auth_mw._is_public_path("/api/health") is True
        assert auth_mw._is_public_path("/api/auth/login") is True


# ---------------------------------------------------------------------------
# BMK-11 not reachable -- limit bounded at the API layer
# ---------------------------------------------------------------------------

class TestLimitBounds:
    def test_browse_search_reviews_limit_ge_1(self):
        # Each list endpoint bounds limit with ge=1, so limit=0 (SQL LIMIT 0,
        # the BMK-11 empty-result class) cannot be requested.
        assert MARKETPLACE_SRC.count('Query(50, ge=1, le=200') >= 2
        # reviews endpoint also bounds limit
        idx = MARKETPLACE_SRC.index("def get_plugin_reviews")
        chunk = MARKETPLACE_SRC[idx:idx + 600]
        assert "ge=1" in chunk


# ---------------------------------------------------------------------------
# Route ordering -- no bare GET /{name} that could shadow /marketplace
# ---------------------------------------------------------------------------

class TestRouteOrdering:
    def test_no_bare_single_segment_name_route(self):
        # A GET "/{name}" (single segment) would shadow GET "/marketplace".
        # Only parametric routes with a second segment are allowed here.
        assert '@router.get("/{name}")' not in MARKETPLACE_SRC
        # The parametric reviews route keeps its second literal segment.
        assert '@router.get("/{name}/reviews"' in MARKETPLACE_SRC
