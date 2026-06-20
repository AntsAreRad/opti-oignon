#!/usr/bin/env python3
"""
S183 / RA-01: client-certificate revocation must be enforced, not just recorded.

Verified behaviors:
- the session guard can revoke a fingerprint, and validate_session_binding then
  rejects a request bound to it (cert_revoked) even when the binding matches;
- the security-mode middleware denies (403) any request that presents a revoked
  client certificate, before any other processing (behavioral, needs starlette);
- the wiring is in place: revoke_client_cert kills live sessions by fingerprint,
  and the middleware consults is_cert_revoked on the request path.
"""

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module(rel_path, name):
    fpath = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, str(fpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("opti_oignon", types.ModuleType("opti_oignon"))
    sys.modules.setdefault("opti_oignon.api", types.ModuleType("opti_oignon.api"))
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Guard unit tests (no starlette)
# ---------------------------------------------------------------------------

guard_mod = _load_module(
    "opti_oignon/remote_session_guard.py", "opti_oignon.remote_session_guard",
)


class TestGuardFingerprintRevocation:
    def test_validate_passes_when_not_revoked(self):
        g = guard_mod.RemoteSessionGuard()
        ok, reason = g.validate_session_binding("jti1", "FP", "FP")
        assert ok and reason == ""

    def test_revoke_then_validate_rejects(self):
        g = guard_mod.RemoteSessionGuard()
        g.revoke_fingerprint("FP")
        assert g.is_fingerprint_revoked("FP")
        # Binding matches and JTI is not revoked, but the cert is revoked.
        ok, reason = g.validate_session_binding("jti1", "FP", "FP")
        assert not ok
        assert reason == "cert_revoked"

    def test_other_fingerprint_unaffected(self):
        g = guard_mod.RemoteSessionGuard()
        g.revoke_fingerprint("FP")
        ok, reason = g.validate_session_binding("jti2", "OTHER", "OTHER")
        assert ok and reason == ""

    def test_reset_clears_revocations(self):
        guard_mod.remote_session_guard.revoke_fingerprint("Z")
        assert guard_mod.remote_session_guard.is_fingerprint_revoked("Z")
        guard_mod.reset_remote_session_guard()
        assert not guard_mod.remote_session_guard.is_fingerprint_revoked("Z")


# ---------------------------------------------------------------------------
# Source-level wiring (no deps)
# ---------------------------------------------------------------------------

MW_SRC = (ROOT / "opti_oignon" / "api" / "security_mode_middleware.py").read_text()
TLS_SRC = (ROOT / "opti_oignon" / "tls_manager.py").read_text()
GUARD_SRC = (ROOT / "opti_oignon" / "remote_session_guard.py").read_text()


class TestWiring:
    def test_middleware_calls_revocation_check_in_dispatch(self):
        assert "_reject_if_cert_revoked(request)" in MW_SRC
        assert "is_cert_revoked" in MW_SRC
        assert "cert_revoked" in MW_SRC

    def test_revoke_client_cert_kills_live_sessions(self):
        assert "revoke_fingerprint(fp)" in TLS_SRC

    def test_guard_has_revocation_api(self):
        assert "def revoke_fingerprint" in GUARD_SRC
        assert "_revoked_fingerprints" in GUARD_SRC
        assert 'return False, "cert_revoked"' in GUARD_SRC


# ---------------------------------------------------------------------------
# Middleware behavioral tests (need starlette)
# ---------------------------------------------------------------------------

@pytest.fixture
def mw_with_stubs():
    pytest.importorskip("starlette")
    # Stub the two modules the revocation check lazily imports.
    rsg = types.ModuleType("opti_oignon.remote_session_guard")
    rsg.extract_cert_fingerprint_from_request = (
        lambda req: req.headers.get("x-test-fp") or None
    )
    tlm = types.ModuleType("opti_oignon.tls_manager")
    tlm.is_cert_revoked = lambda fp: fp == "REVOKED"
    saved = {
        k: sys.modules.get(k)
        for k in ("opti_oignon.remote_session_guard", "opti_oignon.tls_manager")
    }
    sys.modules["opti_oignon.remote_session_guard"] = rsg
    sys.modules["opti_oignon.tls_manager"] = tlm
    mod = _load_module(
        "opti_oignon/api/security_mode_middleware.py",
        "opti_oignon.api.security_mode_middleware",
    )
    yield mod, tlm
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


def _make_request(path="/api/models", headers=None, client=("127.0.0.1", 1)):
    from starlette.requests import Request
    scope = {
        "type": "http",
        "method": "GET",
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


def _dispatch(mod, request):
    from starlette.responses import Response
    calls = []

    async def call_next(req):
        calls.append(req)
        return Response("ok", status_code=200)

    mw = mod.SecurityModeMiddleware(app=lambda *a, **k: None)
    resp = asyncio.run(mw.dispatch(request, call_next))
    return resp, calls


class TestMiddlewareRevocationBehavior:
    def test_revoked_cert_is_rejected(self, mw_with_stubs):
        mod, _tlm = mw_with_stubs
        req = _make_request(headers={"x-test-fp": "REVOKED"})
        resp, calls = _dispatch(mod, req)
        assert resp.status_code == 403
        assert json.loads(bytes(resp.body))["restriction"] == "cert_revoked"
        assert not calls

    def test_non_revoked_cert_passes(self, mw_with_stubs, monkeypatch):
        mod, _tlm = mw_with_stubs
        monkeypatch.setattr(mod, "_get_security_mode", lambda: ("daily", object()))
        req = _make_request(headers={"x-test-fp": "GOODFP"})
        resp, calls = _dispatch(mod, req)
        assert resp.status_code == 200
        assert len(calls) == 1

    def test_no_cert_local_request_passes(self, mw_with_stubs, monkeypatch):
        mod, _tlm = mw_with_stubs
        monkeypatch.setattr(mod, "_get_security_mode", lambda: ("daily", object()))
        req = _make_request()  # no x-test-fp -> no client cert
        resp, calls = _dispatch(mod, req)
        assert resp.status_code == 200
        assert len(calls) == 1

    def test_fails_closed_when_revocation_check_errors(self, mw_with_stubs):
        mod, tlm = mw_with_stubs

        def _boom(fp):
            raise RuntimeError("crl unreadable")

        tlm.is_cert_revoked = _boom
        req = _make_request(headers={"x-test-fp": "SOMEFP"})
        resp, calls = _dispatch(mod, req)
        assert resp.status_code == 403  # cert present + undeterminable -> deny
        assert not calls
