#!/usr/bin/env python3
"""Contract for fail-closed authentication on the security router.

Every security endpoint hangs off a router-level authentication
dependency. When the auth subsystem cannot be imported (degraded mode),
the dependency list must NOT collapse to empty -- an empty list would
serve the whole security surface (mode switch, kill switch, red team
launch, scheduler trigger) without any authentication. An undetermined
auth state is treated as untrusted: the fallback installs a dependency
that refuses with HTTP 503.

  * Contract 1 -- with the auth routes forced unimportable, the router's
    auth dependency is non-empty and its callable raises HTTP 503.

Local-only (the public distribution ships no tests). Runs under pytest
or the __main__ runner. routes_security is loaded in isolation under a
stub package; the auth import is forced to fail deterministically.
"""

import asyncio
import importlib.util
import sys
import traceback
import types
from pathlib import Path

from fastapi import HTTPException

_REPO = Path(__file__).resolve().parent.parent
_ROUTES = _REPO / "opti_oignon" / "api" / "routes_security.py"

_NAMES = (
    "opti_oignon",
    "opti_oignon.api",
    "opti_oignon.api.routes_auth",
    "opti_oignon.api.routes_security",
)


def _load_routes_security_degraded():
    """Load routes_security with the auth routes forced unimportable."""
    saved = {name: sys.modules.get(name) for name in _NAMES}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg
    api = types.ModuleType("opti_oignon.api")
    api.__path__ = []
    sys.modules["opti_oignon.api"] = api
    pkg.api = api
    # A None entry makes "from .routes_auth import ..." raise ImportError.
    sys.modules["opti_oignon.api.routes_auth"] = None

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.api.routes_security", _ROUTES,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.api.routes_security"] = mod
    spec.loader.exec_module(mod)
    return mod, saved


def _restore(saved):
    for name, value in saved.items():
        if value is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = value


def test_c1_degraded_auth_is_fail_closed():
    mod, saved = _load_routes_security_degraded()
    try:
        dep = mod._auth_dep
        assert dep, "auth dependency must not be empty in degraded mode"
        callable_dep = dep[0].dependency
        raised_status = None
        try:
            asyncio.run(callable_dep())
        except HTTPException as exc:
            raised_status = exc.status_code
        assert raised_status == 503, (
            "degraded-mode auth dependency must refuse with HTTP 503"
        )
    finally:
        _restore(saved)


_TESTS = [test_c1_degraded_auth_is_fail_closed]


def _main():
    passed = 0
    for test in _TESTS:
        try:
            test()
        except Exception:
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
        else:
            print(f"PASS {test.__name__}")
            passed += 1
    print(f"{passed}/{len(_TESTS)} passed")
    return 0 if passed == len(_TESTS) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
