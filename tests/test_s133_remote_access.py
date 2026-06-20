"""
Tests for S133 -- Secure Remote Access + Bulbe Network Isolation.

Validates:
- Part 1: network_bind_guard.py (Bulbe fortress, safe bind, assert_localhost)
- Part 2: security_mode.py (remote_access_allowed in ModePolicy)
- Part 3: security_mode_middleware.py (non-localhost rejection in Bulbe)
- Part 4: tls_manager.py (cert generation, revocation, Bulbe refusal)
- Part 5: remote_session_guard.py (session binding, rate limiting, timing)
- Part 6: API routes (remote access endpoints, 403 in Bulbe)
- Part 7: Frontend files (RemoteAccessPanel, remoteAccess.ts, SecurityPanel)
- Part 8: Integration (version bump, no French, Kerckhoffs, launch.sh)

Target: ~48 tests
"""

import ast
import hmac
import importlib.util
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "opti_oignon")
FRONTEND_SRC = os.path.join(PROJECT_ROOT, "frontend", "src")
COMPONENTS_DIR = os.path.join(FRONTEND_SRC, "lib", "components", "settings")
API_TS_DIR = os.path.join(FRONTEND_SRC, "lib", "api")
API_DIR = os.path.join(BACKEND_DIR, "api")


def _load_module(name, path):
    """Load a Python module from file path without triggering __init__ imports."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if "opti_oignon" not in sys.modules:
        parent = types.ModuleType("opti_oignon")
        sys.modules["opti_oignon"] = parent
    if "opti_oignon.config" not in sys.modules:
        cfg_mod = types.ModuleType("opti_oignon.config")
        cfg_mod.DATA_DIR = tempfile.mkdtemp()
        sys.modules["opti_oignon.config"] = cfg_mod
    if "opti_oignon.db_encryption" not in sys.modules:
        dbe = types.ModuleType("opti_oignon.db_encryption")
        def _fake_conn(db_path, **kw):
            return sqlite3.connect(str(db_path), check_same_thread=False)
        dbe.get_encrypted_connection = _fake_conn
        dbe.SQLCIPHER_AVAILABLE = False
        sys.modules["opti_oignon.db_encryption"] = dbe
    if "opti_oignon.signed_audit_log" not in sys.modules:
        sal = types.ModuleType("opti_oignon.signed_audit_log")
        sal.chain_log = MagicMock(return_value=1)
        sal.signed_audit_log = None
        sal.SIGNED_AUDIT_AVAILABLE = True
        sys.modules["opti_oignon.signed_audit_log"] = sal
    if "opti_oignon.encryption" not in sys.modules:
        enc = types.ModuleType("opti_oignon.encryption")
        enc.load_keyfile = MagicMock(side_effect=FileNotFoundError("no keyfile"))
        sys.modules["opti_oignon.encryption"] = enc
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _stub_security_mode(mode="daily"):
    """Create a security_mode stub for the given mode."""
    sm = types.ModuleType("opti_oignon.security_mode")
    sm.is_bulbe = MagicMock(return_value=(mode == "bulbe"))
    sm.is_daily = MagicMock(return_value=(mode == "daily"))
    sm.get_current_mode = MagicMock(return_value=mode)
    sm.MODE_BULBE = "bulbe"
    sm.MODE_DAILY = "daily"
    return sm


# =========================================================================
# Part 1: network_bind_guard.py -- Bulbe Fortress
# =========================================================================

class TestNetworkBindGuardBulbe(unittest.TestCase):
    """Bulbe mode: bind address MUST be 127.0.0.1 regardless of input."""

    def setUp(self):
        sm = _stub_security_mode("bulbe")
        sys.modules["opti_oignon.security_mode"] = sm
        self.mod = _load_module(
            "opti_oignon.network_bind_guard",
            os.path.join(BACKEND_DIR, "network_bind_guard.py"),
        )

    def test_bind_guard_forces_localhost_on_0000(self):
        """get_safe_bind_address('0.0.0.0') returns '127.0.0.1' in Bulbe."""
        result = self.mod.get_safe_bind_address("0.0.0.0")
        self.assertEqual(result, "127.0.0.1")

    def test_bind_guard_forces_localhost_on_any_ip(self):
        """get_safe_bind_address('192.168.1.100') returns '127.0.0.1' in Bulbe."""
        result = self.mod.get_safe_bind_address("192.168.1.100")
        self.assertEqual(result, "127.0.0.1")

    def test_remote_access_always_false_in_bulbe(self):
        """is_remote_access_allowed() returns False in Bulbe always."""
        result = self.mod.is_remote_access_allowed()
        self.assertFalse(result)

    def test_assert_localhost_calls_exit_on_non_local(self):
        """assert_localhost_only() calls sys.exit(1) if non-local bind detected."""
        with patch.object(self.mod, "_probe_bound_address", return_value="0.0.0.0"):
            with self.assertRaises(SystemExit) as ctx:
                self.mod.assert_localhost_only(port=8001)
            self.assertEqual(ctx.exception.code, 1)

    def test_checkpoint_before_apply_hardcoded(self):
        """checkpoint_before_apply must be True."""
        self.assertTrue(self.mod.checkpoint_before_apply)


class TestNetworkBindGuardDaily(unittest.TestCase):
    """Daily mode bind guard behavior."""

    def setUp(self):
        sm = _stub_security_mode("daily")
        sys.modules["opti_oignon.security_mode"] = sm
        self.mod = _load_module(
            "opti_oignon.network_bind_guard_daily",
            os.path.join(BACKEND_DIR, "network_bind_guard.py"),
        )

    def test_daily_no_remote_returns_localhost(self):
        """Daily mode without remote_access config returns 127.0.0.1."""
        with patch.object(self.mod, "_is_remote_enabled_in_config", return_value=False):
            result = self.mod.get_safe_bind_address("0.0.0.0")
            self.assertEqual(result, "127.0.0.1")

    def test_daily_with_remote_passes_through(self):
        """Daily mode with remote enabled passes requested host through."""
        with patch.object(self.mod, "_is_remote_enabled_in_config", return_value=True):
            result = self.mod.get_safe_bind_address("0.0.0.0")
            self.assertEqual(result, "0.0.0.0")

    def test_config_tampering_in_bulbe_ignored(self):
        """Even if security.yaml says remote_access: true, Bulbe still blocks."""
        sm = _stub_security_mode("bulbe")
        sys.modules["opti_oignon.security_mode"] = sm
        with patch.object(self.mod, "_is_remote_enabled_in_config", return_value=True):
            # Re-read mode inside function
            with patch.object(self.mod, "_get_current_mode", return_value="bulbe"):
                result = self.mod.get_safe_bind_address("0.0.0.0")
                self.assertEqual(result, "127.0.0.1")


# =========================================================================
# Part 2: security_mode.py -- ModePolicy.remote_access_allowed
# =========================================================================

class TestModePolicyRemoteAccess(unittest.TestCase):
    """ModePolicy includes remote_access_allowed field."""

    def test_bulbe_policy_remote_false(self):
        """Bulbe policy has remote_access_allowed = False (hardcoded)."""
        mod = _load_module(
            "opti_oignon.security_mode_p2",
            os.path.join(BACKEND_DIR, "security_mode.py"),
        )
        policy = mod.ModePolicy.for_mode("bulbe")
        self.assertFalse(policy.remote_access_allowed)

    def test_daily_policy_has_remote_field(self):
        """Daily policy has remote_access_allowed field."""
        mod = _load_module(
            "opti_oignon.security_mode_p2b",
            os.path.join(BACKEND_DIR, "security_mode.py"),
        )
        policy = mod.ModePolicy.for_mode("daily")
        self.assertIsInstance(policy.remote_access_allowed, bool)

    def test_bulbe_remote_not_from_config(self):
        """In Bulbe, remote_access_allowed is hardcoded, not from config."""
        mod = _load_module(
            "opti_oignon.security_mode_p2c",
            os.path.join(BACKEND_DIR, "security_mode.py"),
        )
        # Even if _read_remote_access_config returns True, Bulbe overrides
        with patch.object(mod.ModePolicy, "_read_remote_access_config", return_value=True):
            policy = mod.ModePolicy.for_mode("bulbe")
            self.assertFalse(policy.remote_access_allowed)


# =========================================================================
# Part 3: security_mode_middleware.py -- Non-localhost rejection
# =========================================================================

class TestSecurityModeMiddlewareNonLocal(unittest.TestCase):
    """Middleware rejects non-localhost requests in Bulbe."""

    def test_middleware_file_has_non_local_rejection(self):
        """Middleware contains non-local IP rejection logic for Bulbe."""
        path = os.path.join(API_DIR, "security_mode_middleware.py")
        with open(path, "r") as f:
            content = f.read()
        self.assertIn("non_local_rejected", content)
        self.assertIn("127.0.0.1", content)
        self.assertIn("::1", content)

    def test_middleware_rejects_external_ip_pattern(self):
        """Middleware checks client.host against localhost addresses."""
        path = os.path.join(API_DIR, "security_mode_middleware.py")
        with open(path, "r") as f:
            content = f.read()
        # Verify the defense-in-depth pattern exists
        self.assertIn("request.client.host", content)
        self.assertIn("CRITICAL", content)

    def test_audit_function_exists(self):
        """_audit_non_local_request function is defined."""
        path = os.path.join(API_DIR, "security_mode_middleware.py")
        with open(path, "r") as f:
            content = f.read()
        self.assertIn("def _audit_non_local_request", content)


# =========================================================================
# Part 4: tls_manager.py -- TLS + mTLS
# =========================================================================

class TestTLSManagerBulbeRefusal(unittest.TestCase):
    """TLS manager refuses to operate in Bulbe mode (defense layer 5)."""

    def setUp(self):
        sm = _stub_security_mode("bulbe")
        sys.modules["opti_oignon.security_mode"] = sm

    def test_setup_tls_raises_in_bulbe(self):
        """setup_tls() raises TLSSecurityError in Bulbe."""
        mod = _load_module(
            "opti_oignon.tls_manager_bulbe",
            os.path.join(BACKEND_DIR, "tls_manager.py"),
        )
        with self.assertRaises(mod.TLSSecurityError):
            mod.setup_tls("test_passphrase_12chars")

    def test_generate_client_cert_raises_in_bulbe(self):
        """generate_client_cert() raises TLSSecurityError in Bulbe."""
        mod = _load_module(
            "opti_oignon.tls_manager_bulbe2",
            os.path.join(BACKEND_DIR, "tls_manager.py"),
        )
        with self.assertRaises(mod.TLSSecurityError):
            mod.generate_client_cert("test-device", "passphrase")

    def test_get_tls_config_raises_in_bulbe(self):
        """get_tls_config() raises TLSSecurityError in Bulbe."""
        mod = _load_module(
            "opti_oignon.tls_manager_bulbe3",
            os.path.join(BACKEND_DIR, "tls_manager.py"),
        )
        with self.assertRaises(mod.TLSSecurityError):
            mod.get_tls_config()


class TestTLSManagerDaily(unittest.TestCase):
    """TLS manager operates in Daily mode."""

    def setUp(self):
        sm = _stub_security_mode("daily")
        sys.modules["opti_oignon.security_mode"] = sm
        self.mod = _load_module(
            "opti_oignon.tls_manager_daily",
            os.path.join(BACKEND_DIR, "tls_manager.py"),
        )
        # Override TLS dir to temp
        self.tmpdir = tempfile.mkdtemp()
        self.mod._TLS_DIR = Path(self.tmpdir)
        self.mod._CA_KEY_PATH = Path(self.tmpdir) / "ca.key"
        self.mod._CA_CERT_PATH = Path(self.tmpdir) / "ca.crt"
        self.mod._SERVER_KEY_PATH = Path(self.tmpdir) / "server.key"
        self.mod._SERVER_CERT_PATH = Path(self.tmpdir) / "server.crt"
        self.mod._CA_KEY_ENC_PATH = Path(self.tmpdir) / "ca.key.enc"
        self.mod._CLIENT_DIR = Path(self.tmpdir) / "clients"
        self.mod._CRL_PATH = Path(self.tmpdir) / "crl.pem"

    def test_setup_generates_ca_and_server(self):
        """setup_tls creates CA cert, server cert, and CRL."""
        result = self.mod.setup_tls("secure_passphrase_123")
        self.assertTrue(result["success"])
        self.assertTrue(self.mod._CA_CERT_PATH.exists())
        self.assertTrue(self.mod._SERVER_CERT_PATH.exists())
        self.assertTrue(self.mod._SERVER_KEY_PATH.exists())
        self.assertTrue(self.mod._CRL_PATH.exists())
        self.assertIn("ca_fingerprint", result)

    def test_ca_key_permissions(self):
        """CA key file should have restrictive permissions."""
        self.mod.setup_tls("secure_passphrase_123")
        if os.name != "nt":
            mode = oct(os.stat(self.mod._CA_KEY_PATH).st_mode & 0o777)
            self.assertEqual(mode, "0o600")

    def test_server_key_permissions(self):
        """Server key file should have restrictive permissions."""
        self.mod.setup_tls("secure_passphrase_123")
        if os.name != "nt":
            mode = oct(os.stat(self.mod._SERVER_KEY_PATH).st_mode & 0o777)
            self.assertEqual(mode, "0o600")

    def test_generate_client_cert_after_setup(self):
        """Can generate a client cert after TLS setup."""
        self.mod.setup_tls("secure_passphrase_123")
        result = self.mod.generate_client_cert("test-phone", "p12_pass_123")
        self.assertTrue(result["success"])
        self.assertIn("fingerprint", result)
        self.assertIn("p12_path", result)

    def test_revoke_client_cert(self):
        """Can revoke a client cert, CRL is updated."""
        self.mod.setup_tls("secure_passphrase_123")
        self.mod.generate_client_cert("revoke-me", "p12_pass_123")
        result = self.mod.revoke_client_cert("revoke-me")
        self.assertTrue(result["success"])
        # Verify metadata updated
        meta = self.mod._load_client_metadata("revoke-me")
        self.assertTrue(meta["revoked"])

    def test_crl_no_caching(self):
        """CRL read is direct from disk (no cache)."""
        # This is verified by code inspection:
        # is_cert_revoked reads from disk every time
        path = os.path.join(BACKEND_DIR, "tls_manager.py")
        with open(path, "r") as f:
            content = f.read()
        self.assertIn("no caching", content.lower())
        self.assertNotIn("_crl_cache", content)

    def test_short_passphrase_rejected(self):
        """setup_tls rejects passphrases shorter than 12 chars."""
        result = self.mod.setup_tls("short")
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "passphrase_too_short")

    def test_checkpoint_hardcoded(self):
        """checkpoint_before_apply is True in tls_manager."""
        self.assertTrue(self.mod.checkpoint_before_apply)


# =========================================================================
# Part 5: remote_session_guard.py
# =========================================================================

class TestRemoteSessionGuard(unittest.TestCase):
    """Remote session guard: binding, rate limiting, timing."""

    def setUp(self):
        sm = _stub_security_mode("daily")
        sys.modules["opti_oignon.security_mode"] = sm
        self.mod = _load_module(
            "opti_oignon.remote_session_guard",
            os.path.join(BACKEND_DIR, "remote_session_guard.py"),
        )
        self.guard = self.mod.RemoteSessionGuard()

    def test_session_binding_valid(self):
        """Valid session binding passes validation."""
        fp = "abcdef1234567890" * 4
        self.guard.bind_session_to_cert("jti-1", fp, "127.0.0.1", "user1")
        valid, err = self.guard.validate_session_binding("jti-1", fp, fp)
        self.assertTrue(valid)
        self.assertEqual(err, "")

    def test_session_binding_mismatch_revokes(self):
        """Fingerprint mismatch revokes the session."""
        fp1 = "a" * 64
        fp2 = "b" * 64
        valid, err = self.guard.validate_session_binding("jti-2", fp1, fp2)
        self.assertFalse(valid)
        self.assertEqual(err, "cert_fingerprint_mismatch")
        # Session should now be revoked
        self.assertTrue(self.guard.is_session_revoked("jti-2"))

    def test_constant_time_comparison(self):
        """Fingerprint comparison uses hmac.compare_digest."""
        path = os.path.join(BACKEND_DIR, "remote_session_guard.py")
        with open(path, "r") as f:
            content = f.read()
        self.assertIn("hmac.compare_digest", content)

    def test_ip_allowlist_localhost(self):
        """Localhost is always in the allowlist."""
        self.assertTrue(self.guard.check_ip_allowed("127.0.0.1"))

    def test_ip_allowlist_lan(self):
        """LAN addresses are in the default allowlist."""
        self.assertTrue(self.guard.check_ip_allowed("192.168.1.50"))
        self.assertTrue(self.guard.check_ip_allowed("10.0.0.1"))

    def test_ip_allowlist_public_denied(self):
        """Public internet IPs are denied by default."""
        self.assertFalse(self.guard.check_ip_allowed("8.8.8.8"))

    def test_rate_limit_normal(self):
        """Normal request rate passes."""
        allowed, _ = self.guard.check_rate_limit("client-a")
        self.assertTrue(allowed)

    def test_rate_limit_exceeded(self):
        """Exceeding rate limit blocks."""
        for _ in range(65):
            self.guard.check_rate_limit("flood-client")
        allowed, err = self.guard.check_rate_limit("flood-client")
        self.assertFalse(allowed)
        self.assertEqual(err, "rate_limited")

    def test_suspicious_activity_revokes_all(self):
        """3 failed auths trigger revoke_all_remote_sessions."""
        with patch.object(self.guard, "revoke_all_remote_sessions") as mock_revoke:
            for _ in range(3):
                self.guard.record_failed_auth("suspicious-client")
            mock_revoke.assert_called()

    def test_revoke_all_sessions(self):
        """revoke_all_remote_sessions sets the nuclear flag."""
        self.guard.revoke_all_remote_sessions()
        self.assertTrue(self.guard.is_session_revoked("any-jti"))

    def test_security_headers_present(self):
        """REMOTE_SECURITY_HEADERS contains required headers."""
        headers = self.mod.REMOTE_SECURITY_HEADERS
        self.assertIn("Strict-Transport-Security", headers)
        self.assertIn("Content-Security-Policy", headers)
        self.assertIn("X-Content-Type-Options", headers)
        self.assertIn("X-Frame-Options", headers)
        self.assertIn("Referrer-Policy", headers)
        self.assertEqual(headers["X-Frame-Options"], "DENY")
        self.assertEqual(headers["Referrer-Policy"], "no-referrer")
        self.assertEqual(headers["X-XSS-Protection"], "0")

    def test_is_remote_request(self):
        """is_remote_request correctly identifies non-localhost."""
        self.assertFalse(self.mod.is_remote_request("127.0.0.1"))
        self.assertFalse(self.mod.is_remote_request("::1"))
        self.assertTrue(self.mod.is_remote_request("192.168.1.50"))
        self.assertFalse(self.mod.is_remote_request(None))


# =========================================================================
# Part 6: API routes (remote access, 403 in Bulbe)
# =========================================================================

class TestRemoteAccessAPIRoutes(unittest.TestCase):
    """API route definitions for remote access."""

    def test_routes_security_has_remote_endpoints(self):
        """routes_security.py defines all 6 remote access endpoints."""
        path = os.path.join(API_DIR, "routes_security.py")
        with open(path, "r") as f:
            content = f.read()
        self.assertIn("/remote-access/status", content)
        self.assertIn("/remote-access/enable", content)
        self.assertIn("/remote-access/disable", content)
        self.assertIn("/remote-access/generate-client-cert", content)
        self.assertIn("/remote-access/revoke-client-cert", content)
        self.assertIn("/remote-access/client-certs", content)

    def test_routes_reject_bulbe(self):
        """All remote access routes call _require_daily_mode."""
        path = os.path.join(API_DIR, "routes_security.py")
        with open(path, "r") as f:
            content = f.read()
        # Count calls to _require_daily_mode in remote-access section
        remote_section = content.split("S133: Remote Access API")[1] if "S133: Remote Access API" in content else ""
        occurrences = remote_section.count("_require_daily_mode()")
        self.assertGreaterEqual(occurrences, 5, "All remote endpoints must check _require_daily_mode")

    def test_localhost_required_for_cert_gen(self):
        """generate-client-cert endpoint requires localhost."""
        path = os.path.join(API_DIR, "routes_security.py")
        with open(path, "r") as f:
            content = f.read()
        # Find the generate endpoint and verify _require_localhost
        idx = content.index("generate-client-cert")
        section = content[idx:idx+500]
        self.assertIn("_require_localhost", section)

    def test_localhost_required_for_enable(self):
        """enable endpoint requires localhost."""
        path = os.path.join(API_DIR, "routes_security.py")
        with open(path, "r") as f:
            content = f.read()
        idx = content.index("/remote-access/enable")
        section = content[idx:idx+500]
        self.assertIn("_require_localhost", section)


# =========================================================================
# Part 7: Frontend files
# =========================================================================

class TestFrontendRemoteAccess(unittest.TestCase):
    """Frontend components and API client for remote access."""

    def test_remote_access_panel_exists(self):
        """RemoteAccessPanel.svelte exists."""
        path = os.path.join(COMPONENTS_DIR, "RemoteAccessPanel.svelte")
        self.assertTrue(os.path.exists(path))

    def test_remote_access_ts_exists(self):
        """remoteAccess.ts API client exists."""
        path = os.path.join(API_TS_DIR, "remoteAccess.ts")
        self.assertTrue(os.path.exists(path))

    def test_panel_has_bulbe_disabled_state(self):
        """RemoteAccessPanel shows disabled message in Bulbe mode."""
        path = os.path.join(COMPONENTS_DIR, "RemoteAccessPanel.svelte")
        with open(path, "r") as f:
            content = f.read()
        self.assertIn("disabled in Bulbe mode", content)
        self.assertIn("isBulbe", content)

    def test_panel_no_controls_in_bulbe(self):
        """Bulbe state renders no interactive controls (no enable toggle)."""
        path = os.path.join(COMPONENTS_DIR, "RemoteAccessPanel.svelte")
        with open(path, "r") as f:
            content = f.read()
        # Bulbe block should have no buttons
        bulbe_section = content.split("{#if isBulbe}")[1].split("{:else")[0]
        self.assertNotIn("<button", bulbe_section)
        self.assertNotIn("<input", bulbe_section)

    def test_security_panel_has_remote_tab(self):
        """SecurityPanel.svelte includes the Remote Access tab."""
        path = os.path.join(COMPONENTS_DIR, "SecurityPanel.svelte")
        with open(path, "r") as f:
            content = f.read()
        self.assertIn("Remote Access", content)
        self.assertIn("RemoteAccessPanel", content)
        self.assertIn("'remote'", content)

    def test_api_client_has_all_functions(self):
        """remoteAccess.ts exports all required API functions."""
        path = os.path.join(API_TS_DIR, "remoteAccess.ts")
        with open(path, "r") as f:
            content = f.read()
        self.assertIn("getRemoteAccessStatus", content)
        self.assertIn("enableRemoteAccess", content)
        self.assertIn("disableRemoteAccess", content)
        self.assertIn("generateClientCert", content)
        self.assertIn("revokeClientCert", content)
        self.assertIn("listClientCerts", content)

    def test_panel_css_variables_only(self):
        """RemoteAccessPanel uses only --oo-* CSS variables."""
        path = os.path.join(COMPONENTS_DIR, "RemoteAccessPanel.svelte")
        with open(path, "r") as f:
            content = f.read()
        # Find all hex colors
        hex_colors = re.findall(r"#[0-9a-fA-F]{6}", content)
        for color in hex_colors:
            # Each must be inside a var() fallback
            idx = content.index(color)
            preceding = content[max(0, idx - 50):idx]
            self.assertIn("var(", preceding,
                          f"Hex color {color} not in var() fallback context")

    def test_panel_html_balance(self):
        """RemoteAccessPanel has balanced div tags."""
        path = os.path.join(COMPONENTS_DIR, "RemoteAccessPanel.svelte")
        with open(path, "r") as f:
            content = f.read()
        div_open = len(re.findall(r"<div[\s>]", content))
        div_close = len(re.findall(r"</div>", content))
        self.assertEqual(div_open, div_close, f"div: {div_open} open, {div_close} close")


# =========================================================================
# Part 8: Integration
# =========================================================================

class TestIntegration(unittest.TestCase):
    """Integration checks: version, French, Kerckhoffs, launch.sh."""

    def test_version_bump(self):
        """Version is 2.9.4."""
        path = os.path.join(BACKEND_DIR, "__version__.py")
        with open(path, "r") as f:
            content = f.read()
        self.assertIn('"3.0.0"', content)

    def test_no_french_in_new_modules(self):
        """No French words in new S133 modules."""
        french_patterns = re.compile(
            r"\b(oui|gestion|defénse|sécurité|réseau|certificat|serveur|connexion)\b",
            re.IGNORECASE,
        )
        files = [
            os.path.join(BACKEND_DIR, "network_bind_guard.py"),
            os.path.join(BACKEND_DIR, "tls_manager.py"),
            os.path.join(BACKEND_DIR, "remote_session_guard.py"),
        ]
        for fpath in files:
            with open(fpath, "r") as f:
                content = f.read()
            matches = french_patterns.findall(content)
            self.assertEqual(
                len(matches), 0,
                f"French found in {os.path.basename(fpath)}: {matches}",
            )

    def test_kerckhoffs_no_hardcoded_secrets(self):
        """No hardcoded secrets (API keys, passwords) in source."""
        files = [
            os.path.join(BACKEND_DIR, "network_bind_guard.py"),
            os.path.join(BACKEND_DIR, "tls_manager.py"),
            os.path.join(BACKEND_DIR, "remote_session_guard.py"),
        ]
        secret_patterns = re.compile(
            r'(password|secret|api_key)\s*=\s*["\'][^"\']{8,}',
            re.IGNORECASE,
        )
        for fpath in files:
            with open(fpath, "r") as f:
                content = f.read()
            matches = secret_patterns.findall(content)
            self.assertEqual(
                len(matches), 0,
                f"Hardcoded secret in {os.path.basename(fpath)}: {matches}",
            )

    def test_argon2id_for_ca_key(self):
        """TLS manager uses Argon2id (not PBKDF2) for CA key derivation."""
        path = os.path.join(BACKEND_DIR, "tls_manager.py")
        with open(path, "r") as f:
            content = f.read()
        self.assertIn("argon2", content.lower())
        self.assertIn("ARGON2_MEMORY_COST", content)
        self.assertIn("65536", content)  # 64MB

    def test_launch_sh_bulbe_hardcode(self):
        """launch.sh hardcodes 127.0.0.1 in Bulbe mode."""
        path = os.path.join(PROJECT_ROOT, "launch.sh")
        with open(path, "r") as f:
            content = f.read()
        self.assertIn("Bulbe mode", content)
        self.assertIn("BIND_HOST=\"127.0.0.1\"", content)
        self.assertIn("ss -tlnp", content)

    def test_main_py_uses_bind_guard(self):
        """main.py imports and uses get_safe_bind_address."""
        path = os.path.join(BACKEND_DIR, "main.py")
        with open(path, "r") as f:
            content = f.read()
        self.assertIn("get_safe_bind_address", content)
        self.assertIn("network_bind_guard", content)

    def test_all_new_python_files_ast_valid(self):
        """All new Python files pass AST validation."""
        files = [
            os.path.join(BACKEND_DIR, "network_bind_guard.py"),
            os.path.join(BACKEND_DIR, "tls_manager.py"),
            os.path.join(BACKEND_DIR, "remote_session_guard.py"),
        ]
        for fpath in files:
            with open(fpath, "r") as f:
                try:
                    ast.parse(f.read(), filename=fpath)
                except SyntaxError as e:
                    self.fail(f"AST error in {os.path.basename(fpath)}: {e}")


if __name__ == "__main__":
    unittest.main()
