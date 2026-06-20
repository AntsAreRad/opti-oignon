"""S197 F10e -- LIGHT functional pass on the security core (inventory 8).

Confirmation only: the six core security features exist, parse, are wired into
the app (middleware registered, routers mounted), and the S183 fixes are
intact. This is NOT a security re-audit (S183 + S184 own that), and auth.py /
auth_2fa.py stay edit-free. Measured perf (per-request middleware cost, crypto,
PQC, live TLS) is shakedown territory; here we confirm there is no obvious
structural perf trap (cached mode lookup, tail-only audit append).
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
APP = OO / "api" / "app.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


CORE_MODULES = [
    "security_mode", "api/security_mode_middleware", "network_bind_guard",
    "db_encryption", "encryption", "secure_bytes", "user_key_manager",
    "pqc_signatures", "signed_audit_log", "audit_anchor_export",
    "auth", "auth_2fa", "tls_manager", "remote_session_guard",
]


def test_core_modules_present_and_parse():
    for m in CORE_MODULES:
        f = OO / f"{m}.py"
        assert f.is_file(), f"missing security-core module {m}"
        ast.parse(_read(f))  # raises on syntax error


def test_security_middleware_and_routers_wired():
    src = _read(APP)
    for mw in ("SecurityHeadersMiddleware", "_CSPMiddleware",
               "SecurityModeMiddleware", "CSRFMiddleware", "AuthMiddleware"):
        assert f"app.add_middleware({mw})" in src, f"{mw} not registered"
    for router in ("auth_router", "security_router", "users_router",
                   "network_router"):
        assert f"app.include_router({router})" in src, f"{router} not mounted"


def test_bulbe_mode_fail_secure_intact():
    # M-01 (S183): undeterminable mode -> Bulbe (most restrictive).
    src = _read(OO / "security_mode.py")
    assert "fail-secure" in src or "fail_secure" in src
    assert "_cached_mode" in src  # mode lookup is cached -> O(1) per request


def test_audit_chain_secret_keyed_and_tail_append():
    # A-01 (S183): anchor keyed on a secret, not the DB path; append reads only
    # the last hash (no full-chain rescan -> O(1) append).
    src = _read(OO / "signed_audit_log.py")
    assert "anchor_key" in src
    assert "ORDER BY id DESC LIMIT 1" in src


def test_mtls_revocation_enforced_intact():
    # RA-01 (S183): revocation is enforced and live sessions are killed.
    tls = _read(OO / "tls_manager.py")
    guard = _read(OO / "remote_session_guard.py")
    assert "def is_cert_revoked(" in tls
    assert "def revoke_client_cert(" in tls
    assert "revoke_client_cert" in guard  # the cross-call that kills sessions


def test_auth_core_present():
    # The auth core stays edit-free; confirm presence only.
    assert (OO / "auth.py").is_file()
    assert (OO / "auth_2fa.py").is_file()
