"""
tests/test_s155_security_audit.py -- S155 full security audit tests.

Verifies:
- Security scan script extended checks (unsafe yaml, path traversal, SSRF,
  rate limiting, cookie security, insecure random, frontend secrets)
- Security scan detects seeded bad patterns
- CSP middleware: config, nonce generation, header building, report store,
  violation parsing, report endpoint, middleware dispatch
- audit_deps.sh exists and is executable
- Audit report document exists
- Version bump check (3.2.4)
"""

import importlib.util
import json
import os
import re
import stat
import sys
import tempfile
import time
import types

# -- Isolation stubs (standard pattern) --
for mod_name in [
    "opti_oignon",
    "opti_oignon.db_utils",
    "opti_oignon.config",
    "opti_oignon.auth",
    "opti_oignon.middleware",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSION_PATH = os.path.join(PROJECT_ROOT, "opti_oignon", "__version__.py")
CSP_PATH = os.path.join(PROJECT_ROOT, "opti_oignon", "middleware", "csp.py")
SCANNER_PATH = os.path.join(PROJECT_ROOT, "scripts", "security_scan.py")
AUDIT_DEPS_PATH = os.path.join(PROJECT_ROOT, "scripts", "audit_deps.sh")
AUDIT_REPORT_PATH = os.path.join(PROJECT_ROOT, "docs", "SECURITY_AUDIT_S155.md")
APP_PATH = os.path.join(PROJECT_ROOT, "opti_oignon", "api", "app.py")
MIDDLEWARE_INIT_PATH = os.path.join(
    PROJECT_ROOT, "opti_oignon", "middleware", "__init__.py"
)


# ---------------------------------------------------------------------------
# Module loaders
# ---------------------------------------------------------------------------

def _load_module(name, path):
    """Load a Python module by file path with isolation."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def csp_mod():
    """Load the CSP middleware module."""
    return _load_module("test_csp", CSP_PATH)


@pytest.fixture(scope="module")
def scanner_mod():
    """Load the security scanner module."""
    return _load_module("test_scanner", SCANNER_PATH)


# =========================================================================
# Class 1: CSP Config
# =========================================================================

class TestCSPConfig:
    """Tests for CSPConfig dataclass."""

    def test_default_config_report_only(self, csp_mod):
        cfg = csp_mod.CSPConfig.default()
        assert cfg.report_only is True

    def test_default_config_enabled(self, csp_mod):
        cfg = csp_mod.CSPConfig.default()
        assert cfg.enabled is True

    def test_default_config_nonce_length(self, csp_mod):
        cfg = csp_mod.CSPConfig.default()
        assert cfg.nonce_length == 24

    def test_default_config_report_uri(self, csp_mod):
        cfg = csp_mod.CSPConfig.default()
        assert cfg.report_uri == "/api/csp-report"

    def test_default_config_directives_present(self, csp_mod):
        cfg = csp_mod.CSPConfig.default()
        assert "default-src" in cfg.directives
        assert "script-src" in cfg.directives
        assert "connect-src" in cfg.directives
        assert "frame-ancestors" in cfg.directives

    def test_default_connect_src_localhost_only(self, csp_mod):
        cfg = csp_mod.CSPConfig.default()
        connect = cfg.directives["connect-src"]
        assert "localhost" in connect
        assert "127.0.0.1" in connect

    def test_default_frame_ancestors_none(self, csp_mod):
        cfg = csp_mod.CSPConfig.default()
        assert cfg.directives["frame-ancestors"] == "'none'"

    def test_default_object_src_none(self, csp_mod):
        cfg = csp_mod.CSPConfig.default()
        assert cfg.directives["object-src"] == "'none'"

    def test_from_dict_override(self, csp_mod):
        cfg = csp_mod.CSPConfig.from_dict({
            "report_only": False,
            "nonce_length": 32,
        })
        assert cfg.report_only is False
        assert cfg.nonce_length == 32

    def test_from_dict_merges_directives(self, csp_mod):
        cfg = csp_mod.CSPConfig.from_dict({
            "directives": {"img-src": "'self' https:"},
        })
        assert cfg.directives["img-src"] == "'self' https:"
        # Other defaults preserved
        assert "default-src" in cfg.directives

    def test_nonce_length_min_clamp(self, csp_mod):
        cfg = csp_mod.CSPConfig(nonce_length=4)
        assert cfg.nonce_length >= 16

    def test_nonce_length_max_clamp(self, csp_mod):
        cfg = csp_mod.CSPConfig(nonce_length=200)
        assert cfg.nonce_length <= 64

    def test_max_reports_clamp_negative(self, csp_mod):
        cfg = csp_mod.CSPConfig(max_stored_reports=-5)
        assert cfg.max_stored_reports >= 0

    def test_max_reports_clamp_excessive(self, csp_mod):
        cfg = csp_mod.CSPConfig(max_stored_reports=999999)
        assert cfg.max_stored_reports <= 10000


# =========================================================================
# Class 2: Nonce Generation
# =========================================================================

class TestNonceGeneration:
    """Tests for CSP nonce generation."""

    def test_nonce_not_empty(self, csp_mod):
        nonce = csp_mod.generate_nonce()
        assert len(nonce) > 0

    def test_nonce_minimum_length(self, csp_mod):
        nonce = csp_mod.generate_nonce(24)
        assert len(nonce) >= 16

    def test_nonces_unique(self, csp_mod):
        nonces = {csp_mod.generate_nonce() for _ in range(100)}
        assert len(nonces) == 100

    def test_nonce_url_safe_chars(self, csp_mod):
        nonce = csp_mod.generate_nonce()
        # secrets.token_urlsafe produces [A-Za-z0-9_-]
        assert re.match(r"^[A-Za-z0-9_\-]+$", nonce)

    def test_nonce_custom_length(self, csp_mod):
        short = csp_mod.generate_nonce(16)
        long = csp_mod.generate_nonce(48)
        assert len(long) > len(short)


# =========================================================================
# Class 3: CSP Header Building
# =========================================================================

class TestCSPHeaderBuilding:
    """Tests for build_csp_header."""

    def test_header_contains_nonce(self, csp_mod):
        cfg = csp_mod.CSPConfig.default()
        nonce = "test-nonce-abc123"
        header = csp_mod.build_csp_header(cfg.directives, nonce)
        assert f"'nonce-{nonce}'" in header

    def test_header_contains_default_src(self, csp_mod):
        cfg = csp_mod.CSPConfig.default()
        header = csp_mod.build_csp_header(cfg.directives, "x")
        assert "default-src 'self'" in header

    def test_header_contains_frame_ancestors(self, csp_mod):
        cfg = csp_mod.CSPConfig.default()
        header = csp_mod.build_csp_header(cfg.directives, "x")
        assert "frame-ancestors 'none'" in header

    def test_header_no_eval(self, csp_mod):
        cfg = csp_mod.CSPConfig.default()
        header = csp_mod.build_csp_header(cfg.directives, "x")
        assert "'unsafe-eval'" not in header

    def test_header_with_report_uri(self, csp_mod):
        cfg = csp_mod.CSPConfig.default()
        header = csp_mod.build_csp_header(
            cfg.directives, "x", report_uri="/api/csp-report"
        )
        assert "report-uri /api/csp-report" in header

    def test_header_without_report_uri(self, csp_mod):
        cfg = csp_mod.CSPConfig.default()
        header = csp_mod.build_csp_header(cfg.directives, "x")
        assert "report-uri" not in header

    def test_header_semicolon_separated(self, csp_mod):
        cfg = csp_mod.CSPConfig.default()
        header = csp_mod.build_csp_header(cfg.directives, "x")
        parts = header.split("; ")
        assert len(parts) >= 5

    def test_nonce_only_in_script_src(self, csp_mod):
        cfg = csp_mod.CSPConfig.default()
        nonce = "unique-nonce-xyz"
        header = csp_mod.build_csp_header(cfg.directives, nonce)
        # Nonce should appear exactly once (in script-src)
        assert header.count(f"'nonce-{nonce}'") == 1
        # Verify it is in the script-src directive
        for part in header.split("; "):
            if f"nonce-{nonce}" in part:
                assert part.startswith("script-src")


# =========================================================================
# Class 4: CSP Report Store
# =========================================================================

class TestCSPReportStore:
    """Tests for CSPReportStore."""

    def test_empty_store(self, csp_mod):
        store = csp_mod.CSPReportStore(max_reports=10)
        assert store.stored_count == 0
        assert store.total_received == 0

    def test_add_report(self, csp_mod):
        store = csp_mod.CSPReportStore(max_reports=10)
        report = csp_mod.CSPViolationReport(
            timestamp=time.time(),
            violated_directive="script-src",
        )
        store.add(report)
        assert store.stored_count == 1
        assert store.total_received == 1

    def test_max_reports_eviction(self, csp_mod):
        store = csp_mod.CSPReportStore(max_reports=5)
        for i in range(10):
            store.add(csp_mod.CSPViolationReport(
                timestamp=time.time(),
                violated_directive=f"test-{i}",
            ))
        assert store.stored_count == 5
        assert store.total_received == 10

    def test_get_all(self, csp_mod):
        store = csp_mod.CSPReportStore(max_reports=10)
        store.add(csp_mod.CSPViolationReport(
            timestamp=1.0, violated_directive="a",
        ))
        store.add(csp_mod.CSPViolationReport(
            timestamp=2.0, violated_directive="b",
        ))
        all_reports = store.get_all()
        assert len(all_reports) == 2
        assert all_reports[0]["violated_directive"] == "a"

    def test_get_recent(self, csp_mod):
        store = csp_mod.CSPReportStore(max_reports=20)
        for i in range(15):
            store.add(csp_mod.CSPViolationReport(
                timestamp=float(i), violated_directive=f"d-{i}",
            ))
        recent = store.get_recent(5)
        assert len(recent) == 5
        assert recent[-1]["violated_directive"] == "d-14"

    def test_clear(self, csp_mod):
        store = csp_mod.CSPReportStore(max_reports=10)
        for i in range(5):
            store.add(csp_mod.CSPViolationReport(
                timestamp=float(i), violated_directive="x",
            ))
        cleared = store.clear()
        assert cleared == 5
        assert store.stored_count == 0
        # total_received is not reset
        assert store.total_received == 5

    def test_report_to_dict(self, csp_mod):
        report = csp_mod.CSPViolationReport(
            timestamp=123.456,
            document_uri="http://localhost:8001/",
            violated_directive="script-src",
            blocked_uri="inline",
            line_number=42,
        )
        d = report.to_dict()
        assert d["timestamp"] == 123.456
        assert d["document_uri"] == "http://localhost:8001/"
        assert d["violated_directive"] == "script-src"
        assert d["blocked_uri"] == "inline"
        assert d["line_number"] == 42


# =========================================================================
# Class 5: CSP Violation Report Parsing
# =========================================================================

class TestCSPReportParsing:
    """Tests for parse_csp_report."""

    def test_parse_valid_report(self, csp_mod):
        body = json.dumps({"csp-report": {
            "document-uri": "http://localhost:8001/chat",
            "violated-directive": "script-src 'self'",
            "blocked-uri": "inline",
            "source-file": "app.js",
            "line-number": 15,
        }}).encode()
        report = csp_mod.parse_csp_report(body)
        assert report is not None
        assert report.violated_directive == "script-src 'self'"
        assert report.blocked_uri == "inline"
        assert report.source_file == "app.js"
        assert report.line_number == 15

    def test_parse_flat_report(self, csp_mod):
        """Some browsers send without the csp-report wrapper."""
        body = json.dumps({
            "document-uri": "http://localhost/",
            "violated-directive": "style-src",
        }).encode()
        report = csp_mod.parse_csp_report(body)
        assert report is not None
        assert report.violated_directive == "style-src"

    def test_parse_invalid_json(self, csp_mod):
        report = csp_mod.parse_csp_report(b"not json at all")
        assert report is None

    def test_parse_empty_body(self, csp_mod):
        report = csp_mod.parse_csp_report(b"")
        assert report is None

    def test_parse_report_has_timestamp(self, csp_mod):
        body = json.dumps({"csp-report": {
            "violated-directive": "default-src",
        }}).encode()
        report = csp_mod.parse_csp_report(body)
        assert report is not None
        assert report.timestamp > 0


# =========================================================================
# Class 6: CSP Module Structure
# =========================================================================

class TestCSPModuleStructure:
    """Tests for CSP module attributes and constants."""

    def test_checkpoint_before_apply(self, csp_mod):
        assert hasattr(csp_mod, "checkpoint_before_apply")
        assert csp_mod.checkpoint_before_apply is True

    def test_csp_available_attribute(self, csp_mod):
        assert hasattr(csp_mod, "CSP_AVAILABLE")

    def test_default_csp_config_constant(self, csp_mod):
        assert hasattr(csp_mod, "_DEFAULT_CSP_CONFIG")
        cfg = csp_mod._DEFAULT_CSP_CONFIG
        assert "directives" in cfg
        assert "enabled" in cfg

    def test_get_report_store_singleton(self, csp_mod):
        store1 = csp_mod.get_report_store()
        store2 = csp_mod.get_report_store()
        assert store1 is store2

    def test_load_csp_config_returns_config(self, csp_mod):
        cfg = csp_mod.load_csp_config()
        assert isinstance(cfg, csp_mod.CSPConfig)


# =========================================================================
# Class 7: Security Scanner -- New Checks
# =========================================================================

class TestScannerNewChecks:
    """Tests for S155 additions to security_scan.py."""

    def test_scanner_has_18_checks(self, scanner_mod):
        report = scanner_mod.run_all_checks()
        assert report["total"] == 18

    def test_check_unsafe_yaml_exists(self, scanner_mod):
        assert hasattr(scanner_mod, "check_no_unsafe_yaml")

    def test_check_path_traversal_exists(self, scanner_mod):
        assert hasattr(scanner_mod, "check_path_traversal")

    def test_check_ssrf_vectors_exists(self, scanner_mod):
        assert hasattr(scanner_mod, "check_ssrf_vectors")

    def test_check_rate_limiting_exists(self, scanner_mod):
        assert hasattr(scanner_mod, "check_rate_limiting")

    def test_check_cookie_security_exists(self, scanner_mod):
        assert hasattr(scanner_mod, "check_cookie_security")

    def test_check_insecure_random_exists(self, scanner_mod):
        assert hasattr(scanner_mod, "check_insecure_random")

    def test_check_frontend_secrets_exists(self, scanner_mod):
        assert hasattr(scanner_mod, "check_frontend_secrets")


# =========================================================================
# Class 8: Scanner Seeded Pattern Detection
# =========================================================================

class TestScannerPatternDetection:
    """Tests that the scanner detects known bad patterns in seeded files."""

    @pytest.fixture(autouse=True)
    def setup_scanner(self, scanner_mod):
        self.scanner = scanner_mod
        # Create temp files under project root so relative_to works
        self.tmpdir = os.path.join(PROJECT_ROOT, "_test_s155_tmp")
        os.makedirs(self.tmpdir, exist_ok=True)
        yield
        # Cleanup
        import shutil
        if os.path.isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def _write_file(self, name, content):
        path = os.path.join(self.tmpdir, name)
        with open(path, "w") as f:
            f.write(content)
        from pathlib import Path
        return Path(path)

    def test_detect_unsafe_yaml_load(self):
        f = self._write_file("bad_yaml.py", 'data = yaml.load(open("f.yml"))\n')
        result = self.scanner.check_no_unsafe_yaml([f])
        assert not result.passed
        assert len(result.violations) >= 1

    def test_allow_yaml_safe_load(self):
        f = self._write_file("ok_yaml.py",
            'data = yaml.load(open("f.yml"), Loader=yaml.SafeLoader)\n')
        result = self.scanner.check_no_unsafe_yaml([f])
        assert result.passed

    def test_detect_path_traversal_os_join(self):
        f = self._write_file("bad_path.py",
            'filepath = os.path.join(base, filename)\n')
        result = self.scanner.check_path_traversal([f])
        assert not result.passed

    def test_detect_ssrf_requests_get(self):
        f = self._write_file("bad_ssrf.py",
            'resp = requests.get(f"http://{host}/api")\n')
        result = self.scanner.check_ssrf_vectors([f])
        assert not result.passed

    def test_detect_ssrf_urlopen(self):
        f = self._write_file("bad_urlopen.py",
            'with urllib.request.urlopen(user_url) as r: pass\n')
        result = self.scanner.check_ssrf_vectors([f])
        assert not result.passed

    def test_detect_insecure_random_in_security(self):
        f = self._write_file("auth_session.py",
            'token = random.randint(0, 999999)\n')
        result = self.scanner.check_insecure_random([f])
        assert not result.passed

    def test_allow_random_in_non_security(self):
        f = self._write_file("analytics.py",
            'sample = random.choice(items)\n')
        result = self.scanner.check_insecure_random([f])
        assert result.passed

    def test_detect_cookie_missing_flags(self):
        f = self._write_file("bad_cookie.py",
            'response.set_cookie("session", value="abc")\n')
        result = self.scanner.check_cookie_security([f])
        assert not result.passed
        assert any("httponly" in v["detail"] for v in result.violations)

    def test_allow_cookie_with_all_flags(self):
        f = self._write_file("ok_cookie.py",
            'response.set_cookie("session", value="abc", '
            'httponly=True, secure=True, samesite="Strict")\n')
        result = self.scanner.check_cookie_security([f])
        assert result.passed

    def test_detect_hardcoded_secret_patterns(self):
        f = self._write_file("bad_secrets.py",
            'API_KEY = "sk-abcdefghij1234567890"\n')
        result = self.scanner.check_no_hardcoded_secrets([f])
        assert not result.passed
        assert len(result.violations) >= 1

    def test_detect_frontend_secret(self):
        from pathlib import Path
        f = self._write_file("bad_frontend.ts",
            'const apiKey = "sk-1234567890abcdef12345678";\n')
        result = self.scanner.check_frontend_secrets([], [Path(f)])
        assert not result.passed

    def test_allow_frontend_env_reference(self):
        from pathlib import Path
        f = self._write_file("ok_frontend.ts",
            'const apiKey = import.meta.env.VITE_API_KEY;\n')
        result = self.scanner.check_frontend_secrets([], [Path(f)])
        assert result.passed


# =========================================================================
# Class 9: Scanner CheckResult
# =========================================================================

class TestCheckResult:
    """Tests for CheckResult data structure."""

    def test_initial_state_passed(self, scanner_mod):
        r = scanner_mod.CheckResult("test", "Test check")
        assert r.passed is True
        assert len(r.violations) == 0

    def test_add_violation_sets_failed(self, scanner_mod):
        r = scanner_mod.CheckResult("test", "Test check")
        r.add_violation("file.py", 10, "bad pattern")
        assert r.passed is False
        assert len(r.violations) == 1

    def test_to_dict(self, scanner_mod):
        r = scanner_mod.CheckResult("test", "Test check")
        r.add_violation("file.py", 10, "detail")
        d = r.to_dict()
        assert d["name"] == "test"
        assert d["passed"] is False
        assert d["violation_count"] == 1

    def test_json_output(self, scanner_mod):
        report = scanner_mod.run_all_checks()
        serialized = json.dumps(report)
        parsed = json.loads(serialized)
        assert "checks" in parsed
        assert "total" in parsed


# =========================================================================
# Class 10: File Existence and Structure
# =========================================================================

class TestFileExistence:
    """Tests that all S155 files exist and have correct structure."""

    def test_csp_module_exists(self):
        assert os.path.isfile(CSP_PATH)

    def test_middleware_init_exists(self):
        assert os.path.isfile(MIDDLEWARE_INIT_PATH)

    def test_audit_deps_script_exists(self):
        assert os.path.isfile(AUDIT_DEPS_PATH)

    def test_audit_deps_is_executable(self):
        mode = os.stat(AUDIT_DEPS_PATH).st_mode
        assert mode & stat.S_IXUSR, "audit_deps.sh must be executable"

    def test_audit_report_exists(self):
        assert os.path.isfile(AUDIT_REPORT_PATH)

    def test_security_scan_exists(self):
        assert os.path.isfile(SCANNER_PATH)

    def test_audit_report_has_executive_summary(self):
        with open(AUDIT_REPORT_PATH) as f:
            content = f.read()
        assert "Executive Summary" in content

    def test_audit_report_has_findings_table(self):
        with open(AUDIT_REPORT_PATH) as f:
            content = f.read()
        assert "SA-155-" in content

    def test_audit_report_has_remediation_plan(self):
        with open(AUDIT_REPORT_PATH) as f:
            content = f.read()
        assert "Remediation Plan" in content
        assert "S156" in content

    def test_audit_report_has_methodology(self):
        with open(AUDIT_REPORT_PATH) as f:
            content = f.read()
        assert "Methodology" in content
        assert "pip-audit" in content
        assert "bandit" in content

    def test_audit_report_has_dependency_summary(self):
        with open(AUDIT_REPORT_PATH) as f:
            content = f.read()
        assert "Dependency Inventory" in content


# =========================================================================
# Class 11: App.py Integration
# =========================================================================

class TestAppIntegration:
    """Tests that CSP middleware is registered in app.py."""

    @pytest.fixture(autouse=True)
    def load_app_source(self):
        with open(APP_PATH) as f:
            self.app_source = f.read()

    def test_csp_middleware_import(self):
        assert "CSPMiddleware" in self.app_source

    def test_csp_middleware_registered(self):
        assert "add_middleware" in self.app_source
        # Check that CSPMiddleware is used with add_middleware
        assert "app.add_middleware(_CSPMiddleware)" in self.app_source

    def test_csp_router_import(self):
        assert "csp_router" in self.app_source

    def test_csp_router_included(self):
        assert "include_router(_csp_router)" in self.app_source

    def test_csp_graceful_degradation(self):
        assert "Failed to register CSP middleware" in self.app_source


# =========================================================================
# Class 12: Middleware __init__.py
# =========================================================================

class TestMiddlewareInit:
    """Tests for middleware package init."""

    def test_checkpoint_in_init(self):
        with open(MIDDLEWARE_INIT_PATH) as f:
            content = f.read()
        assert "checkpoint_before_apply = True" in content


# =========================================================================
# Class 13: Version Bump
# =========================================================================

class TestVersionBump:
    """Tests for version bump to 3.2.4."""

    def test_version_is_3_2_4(self):
        spec = importlib.util.spec_from_file_location("ver", VERSION_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.__version__ == "3.2.4"
