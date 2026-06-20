#!/usr/bin/env python3
"""
Tests for S136 — Security Audit + Documentation + v3.0.0 Release.

Covers:
  - SECURITY.md existence and required sections
  - CHANGELOG.md existence and version milestones
  - README.md references SECURITY.md and v3.0.0
  - Version 3.0.0 consistent across all sources
  - CSRFMiddleware exists and is registered
  - AuthMiddleware exists and is registered
  - db_utils.safe_connect exists
  - Audit chain anchor file mechanism
  - JWT algorithm enforcement
  - Rate limiter XFF hardening
  - Sandbox dest_dir validation
  - Model path confinement
  - SSRF protection on downloads
  - Recovery code HMAC hardening
  - checkpoint_before_apply hardcoded in new files
  - No French in new S136 files
  - AST validation on all new/modified files
"""

import ast
import importlib.util
import os
import re
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
_BACKEND_DIR = _PROJECT_ROOT / "opti_oignon"
_API_DIR = _BACKEND_DIR / "api"
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"


def _load_module(name: str, path: str):
    """Load a module without triggering the full __init__ chain."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# =========================================================================
# SECURITY.md Tests
# =========================================================================

class TestSecurityMD(unittest.TestCase):
    """Verify SECURITY.md exists with required sections."""

    def setUp(self):
        self.path = _PROJECT_ROOT / "SECURITY.md"
        self.assertTrue(self.path.exists(), "SECURITY.md must exist at project root")
        self.content = _read(self.path)

    def test_required_sections_present(self):
        """SECURITY.md must contain all critical sections."""
        required = [
            "Security Philosophy",
            "Assets Inventory",
            "Threat Actors",
            "Mitigation Matrix",
            "Known Limitations",
            "Responsible Disclosure",
            "Deployment Hardening Guide",
            "S136 Security Audit Results",
        ]
        for section in required:
            self.assertIn(section, self.content, f"Missing section: {section}")

    def test_threat_actors_enumerated(self):
        """At least 8 threat actors defined."""
        count = len(re.findall(r"### T\d+", self.content))
        self.assertGreaterEqual(count, 8, f"Only {count} threat actors (need >= 8)")

    def test_mitigation_matrix_present(self):
        """Mitigation matrix table exists."""
        self.assertIn("| Threat |", self.content)
        self.assertIn("MITIGATES", self.content)

    def test_audit_results_table(self):
        """S136 audit results include finding tables."""
        self.assertIn("CRITICAL", self.content)
        self.assertIn("| Fix |", self.content)
        self.assertIn("Round 1", self.content)
        self.assertIn("Round 4", self.content)

    def test_no_french(self):
        """SECURITY.md must be in English."""
        french_markers = ["sécurité", "menace", "utilisateur", "clé de chiffrement"]
        for marker in french_markers:
            self.assertNotIn(marker.lower(), self.content.lower(),
                             f"French detected: {marker}")


# =========================================================================
# CHANGELOG.md Tests
# =========================================================================

class TestChangelogMD(unittest.TestCase):
    """Verify CHANGELOG.md exists with version milestones."""

    def setUp(self):
        self.path = _PROJECT_ROOT / "CHANGELOG.md"
        self.assertTrue(self.path.exists(), "CHANGELOG.md must exist")
        self.content = _read(self.path)

    def test_contains_version_milestones(self):
        """Must reference key version milestones."""
        for ver in ["v3.0.0", "v2.9.0", "v2.8.0", "v2.7.0", "v2.6.0"]:
            self.assertIn(ver, self.content, f"Missing milestone: {ver}")

    def test_security_tags(self):
        """Security-relevant changes flagged with [SECURITY]."""
        count = self.content.count("[SECURITY]")
        self.assertGreaterEqual(count, 10, f"Only {count} [SECURITY] tags")


# =========================================================================
# README.md Tests
# =========================================================================

class TestReadmeMD(unittest.TestCase):
    """Verify README.md references SECURITY.md and v3.0.0."""

    def setUp(self):
        self.content = _read(_PROJECT_ROOT / "README.md")

    def test_references_security_md(self):
        self.assertIn("SECURITY.md", self.content)

    def test_version_300(self):
        self.assertIn("3.0.0", self.content)

    def test_security_section(self):
        self.assertIn("## Security", self.content)


# =========================================================================
# Version Consistency Tests
# =========================================================================

class TestVersionConsistency(unittest.TestCase):
    """Version 3.0.0 must be consistent across all sources."""

    def test_version_py(self):
        mod = _load_module("ver", str(_BACKEND_DIR / "__version__.py"))
        self.assertEqual(mod.__version__, "3.0.0")

    def test_version_in_pyproject(self):
        content = _read(_PROJECT_ROOT / "pyproject.toml")
        self.assertIn("opti_oignon.__version__.__version__", content)

    def test_version_in_security_md(self):
        content = _read(_PROJECT_ROOT / "SECURITY.md")
        self.assertIn("v3.0.0", content)

    def test_version_in_changelog(self):
        content = _read(_PROJECT_ROOT / "CHANGELOG.md")
        self.assertIn("v3.0.0", content)


# =========================================================================
# CSRF Middleware Tests
# =========================================================================

class TestCSRFMiddleware(unittest.TestCase):
    """CSRFMiddleware must exist and be registered."""

    def test_module_exists(self):
        path = _API_DIR / "csrf_middleware.py"
        self.assertTrue(path.exists())

    def test_ast_valid(self):
        content = _read(_API_DIR / "csrf_middleware.py")
        ast.parse(content)

    def test_registered_in_app(self):
        content = _read(_API_DIR / "app.py")
        self.assertIn("CSRFMiddleware", content)
        self.assertIn("add_middleware", content)

    def test_checkpoint_hardcoded(self):
        content = _read(_API_DIR / "csrf_middleware.py")
        self.assertIn("checkpoint_before_apply = True", content)


# =========================================================================
# Auth Middleware Tests
# =========================================================================

class TestAuthMiddleware(unittest.TestCase):
    """AuthMiddleware must exist and enforce deny-by-default."""

    def test_module_exists(self):
        path = _API_DIR / "auth_middleware.py"
        self.assertTrue(path.exists())

    def test_ast_valid(self):
        content = _read(_API_DIR / "auth_middleware.py")
        ast.parse(content)

    def test_registered_in_app(self):
        content = _read(_API_DIR / "app.py")
        self.assertIn("AuthMiddleware", content)

    def test_deny_by_default(self):
        content = _read(_API_DIR / "auth_middleware.py")
        self.assertIn("deny-by-default", content.lower())

    def test_public_allowlist_exists(self):
        content = _read(_API_DIR / "auth_middleware.py")
        self.assertIn("/api/auth/login", content)
        self.assertIn("/api/health", content)

    def test_checkpoint_hardcoded(self):
        content = _read(_API_DIR / "auth_middleware.py")
        self.assertIn("checkpoint_before_apply = True", content)


# =========================================================================
# db_utils Tests
# =========================================================================

class TestDbUtils(unittest.TestCase):
    """db_utils.safe_connect must exist."""

    def test_module_exists(self):
        path = _BACKEND_DIR / "db_utils.py"
        self.assertTrue(path.exists())

    def test_ast_valid(self):
        content = _read(_BACKEND_DIR / "db_utils.py")
        ast.parse(content)

    def test_safe_connect_defined(self):
        content = _read(_BACKEND_DIR / "db_utils.py")
        self.assertIn("def safe_connect", content)

    def test_checkpoint_hardcoded(self):
        content = _read(_BACKEND_DIR / "db_utils.py")
        self.assertIn("checkpoint_before_apply = True", content)


# =========================================================================
# JWT Algorithm Enforcement Tests
# =========================================================================

class TestJWTAlgEnforcement(unittest.TestCase):
    """JWT decoder must enforce server-side algorithm."""

    def test_no_header_trust(self):
        content = _read(_BACKEND_DIR / "auth.py")
        self.assertIn("algorithm mismatch", content)
        self.assertIn("server-side", content.lower())

    def test_dummy_hash_for_timing(self):
        content = _read(_BACKEND_DIR / "auth.py")
        self.assertIn("_dummy_hash", content)


# =========================================================================
# Rate Limiter XFF Hardening Tests
# =========================================================================

class TestRateLimiterXFF(unittest.TestCase):
    """Rate limiter must only trust XFF from localhost."""

    def test_xff_localhost_only(self):
        content = _read(_API_DIR / "routes_auth.py")
        self.assertIn("127.0.0.1", content)
        self.assertIn("::1", content)
        self.assertIn("Only trust X-Forwarded-For from localhost", content)


# =========================================================================
# Sandbox dest_dir Restriction Tests
# =========================================================================

class TestSandboxDestDir(unittest.TestCase):
    """Sandbox copy-out dest_dir must be restricted."""

    def test_validate_dest_dir_exists(self):
        content = _read(_API_DIR / "routes_sandbox.py")
        self.assertIn("_validate_dest_dir", content)

    def test_restricts_to_data(self):
        content = _read(_API_DIR / "routes_sandbox.py")
        self.assertIn("data", content.lower())
        self.assertIn("outside", content.lower())


# =========================================================================
# Model Path Confinement Tests
# =========================================================================

class TestModelPathConfinement(unittest.TestCase):
    """Model loading must be confined to model_dirs."""

    def test_rejects_absolute_paths(self):
        content = _read(_BACKEND_DIR / "inference_backend.py")
        self.assertIn("isabs", content)

    def test_rejects_traversal(self):
        content = _read(_BACKEND_DIR / "inference_backend.py")
        self.assertIn('".."', content)


# =========================================================================
# SSRF Protection Tests
# =========================================================================

class TestSSRFProtection(unittest.TestCase):
    """Model download must validate URLs."""

    def test_url_validation_exists(self):
        content = _read(_BACKEND_DIR / "model_manager.py")
        self.assertIn("_validate_download_url", content)

    def test_private_ip_check(self):
        content = _read(_BACKEND_DIR / "model_manager.py")
        self.assertIn("is_private", content)
        self.assertIn("is_loopback", content)


# =========================================================================
# Recovery Code Hardening Tests
# =========================================================================

class TestRecoveryCodeHardening(unittest.TestCase):
    """Recovery codes must use HMAC and sufficient entropy."""

    def test_hmac_used(self):
        content = _read(_BACKEND_DIR / "auth_2fa.py")
        self.assertIn("_hmac.new", content)

    def test_64bit_entropy(self):
        content = _read(_BACKEND_DIR / "auth_2fa.py")
        self.assertIn("RECOVERY_CODE_LENGTH = 16", content)


# =========================================================================
# Audit Chain Anchor Tests
# =========================================================================

class TestAuditChainAnchor(unittest.TestCase):
    """Audit chain must have truncation detection via anchor file."""

    def test_anchor_methods_exist(self):
        content = _read(_BACKEND_DIR / "signed_audit_log.py")
        self.assertIn("_save_anchor", content)
        self.assertIn("_check_anchor", content)
        self.assertIn("audit_chain_anchor", content)


# =========================================================================
# WebSocket Auth Tests
# =========================================================================

class TestWebSocketAuth(unittest.TestCase):
    """All WebSocket endpoints must require auth."""

    def test_chat_stream_has_auth(self):
        content = _read(_API_DIR / "routes_chat.py")
        self.assertIn("authenticate_websocket", content)

    def test_coding_ws_has_auth(self):
        content = _read(_API_DIR / "routes_coding.py")
        self.assertIn("authenticate_websocket", content)

    def test_benchmark_ws_has_auth(self):
        content = _read(_API_DIR / "routes_benchmark.py")
        self.assertIn("authenticate_websocket", content)

    def test_metrics_ws_has_auth(self):
        content = _read(_API_DIR / "routes_live_metrics.py")
        self.assertIn("authenticate_websocket", content)

    def test_origin_check(self):
        content = _read(_API_DIR / "routes_auth.py")
        self.assertIn("CSWSH", content)
        self.assertIn("_ALLOWED_WS_HOSTS", content)


# =========================================================================
# AST Validation (all new/modified S136 files)
# =========================================================================

class TestASTValidation(unittest.TestCase):
    """All new S136 files must be valid Python."""

    S136_FILES = [
        _API_DIR / "csrf_middleware.py",
        _API_DIR / "auth_middleware.py",
        _BACKEND_DIR / "db_utils.py",
        _BACKEND_DIR / "auth.py",
        _BACKEND_DIR / "auth_2fa.py",
        _BACKEND_DIR / "signed_audit_log.py",
        _BACKEND_DIR / "inference_backend.py",
        _BACKEND_DIR / "model_manager.py",
        _API_DIR / "app.py",
        _API_DIR / "routes_auth.py",
        _API_DIR / "routes_chat.py",
        _API_DIR / "routes_coding.py",
        _API_DIR / "routes_benchmark.py",
        _API_DIR / "routes_live_metrics.py",
        _API_DIR / "routes_sandbox.py",
        _API_DIR / "routes_security.py",
        _SCRIPTS_DIR / "security_scan.py",
    ]

    def test_all_s136_files_parse(self):
        for fpath in self.S136_FILES:
            if fpath.exists():
                try:
                    ast.parse(_read(fpath))
                except SyntaxError as e:
                    self.fail(f"SyntaxError in {fpath.name}: {e}")


# =========================================================================
# Audit Script Existence Tests
# =========================================================================

class TestAuditScripts(unittest.TestCase):
    """Audit scripts must exist and be executable."""

    def test_security_scan_exists(self):
        path = _SCRIPTS_DIR / "security_scan.py"
        self.assertTrue(path.exists())

    def test_accessibility_audit_exists(self):
        path = _SCRIPTS_DIR / "audit_accessibility.py"
        self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
