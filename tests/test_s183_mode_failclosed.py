#!/usr/bin/env python3
"""
S183 / M-01: the security-mode middleware must fail closed.

When the mode cannot be determined (the security_mode module is unavailable),
the middleware must apply Bulbe enforcement -- the most restrictive -- instead
of passing the request through, matching the network bind guard and the Veilid
gate.

Source-level checks run everywhere. The behavioral checks need starlette and are
skipped where it is not installed (they run in CI).
"""

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_MW_PATH = ROOT / "opti_oignon" / "api" / "security_mode_middleware.py"
SRC = _MW_PATH.read_text(encoding="utf-8")


class TestFailClosedSource:
    def test_maps_undeterminable_mode_to_bulbe(self):
        assert 'mode, policy = "bulbe", None' in SRC

    def test_no_legacy_fail_open_passthrough(self):
        assert "Graceful degradation" not in SRC
        assert "fail-open" not in SRC

    def test_strict_getattr_defaults(self):
        assert 'getattr(policy, "bearer_auth_allowed", False)' in SRC
        assert 'getattr(policy, "plugin_allowlist_required", True)' in SRC
        assert 'getattr(policy, "cookie_samesite", "Strict")' in SRC


@pytest.fixture
def mw_mod():
    pytest.importorskip("starlette")
    name = "opti_oignon.api.security_mode_middleware"
    spec = importlib.util.spec_from_file_location(name, str(_MW_PATH))
    mod = importlib.util.module_from_spec(spec)
    for pkg in ("opti_oignon", "opti_oignon.api"):
        sys.modules.setdefault(pkg, types.ModuleType(pkg))
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_request(path, method="GET", headers=None, client=("127.0.0.1", 1111)):
    from starlette.requests import Request
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [
            (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()
        ],
        "client": client,
        "scheme": "http",
        "server": ("testserver", 80),
    }
    return Request(scope)


def _dispatch(mw_mod, request):
    from starlette.responses import Response
    calls = []

    async def call_next(req):
        calls.append(req)
        return Response("ok", status_code=200)

    mw = mw_mod.SecurityModeMiddleware(app=lambda *a, **k: None)
    resp = asyncio.run(mw.dispatch(request, call_next))
    return resp, calls


class TestFailClosedBehavior:
    def test_bearer_rejected_when_mode_undeterminable(self, mw_mod, monkeypatch):
        monkeypatch.setattr(mw_mod, "_get_security_mode", lambda: (None, None))
        req = _make_request(
            "/api/models",
            headers={"authorization": "Bearer abc.def.ghi"},
            client=("127.0.0.1", 5000),
        )
        resp, calls = _dispatch(mw_mod, req)
        assert resp.status_code == 403
        assert json.loads(bytes(resp.body))["restriction"] == "bearer_rejected"
        assert not calls  # request was not forwarded

    def test_non_local_rejected_when_mode_undeterminable(self, mw_mod, monkeypatch):
        monkeypatch.setattr(mw_mod, "_get_security_mode", lambda: (None, None))
        req = _make_request("/api/models", client=("10.1.2.3", 5000))
        resp, calls = _dispatch(mw_mod, req)
        assert resp.status_code == 403
        assert json.loads(bytes(resp.body))["restriction"] == "non_local_rejected"
        assert not calls

    def test_daily_still_passes_through(self, mw_mod, monkeypatch):
        # A determinable Daily mode must keep passing requests through.
        monkeypatch.setattr(
            mw_mod, "_get_security_mode", lambda: ("daily", object()),
        )
        req = _make_request(
            "/api/models",
            headers={"authorization": "Bearer abc.def.ghi"},
            client=("127.0.0.1", 5000),
        )
        resp, calls = _dispatch(mw_mod, req)
        assert resp.status_code == 200
        assert len(calls) == 1
