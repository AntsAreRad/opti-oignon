"""
tests/test_s156_security_remediation.py -- S156 security remediation tests.

Verifies:
- Version bump to 3.2.5
- Dependency version constraints updated in pyproject.toml
- SQL query hardening in analytics.py (no f-string SQL, _build_where allowlist)
- SQL query hardening in rag_sanitizer.py (no f-string SQL)
- Rate limiter module (sliding window, key isolation, Bulbe mode, reset, status)
- Rate limiting integration in routes_files.py and routes_users.py
- Path traversal fix in artifacts.py (os.path.basename)
- Path traversal fix in routes_rag.py (.. rejection, symlink validation)
- MD5 usedforsecurity=False flag across all modules
- No French in modified files
- checkpoint_before_apply sentinel in new modules
"""

import importlib.util
import os
import re
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
    "opti_oignon.security_mode",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSION_PATH = os.path.join(PROJECT_ROOT, "opti_oignon", "__version__.py")
ANALYTICS_PATH = os.path.join(PROJECT_ROOT, "opti_oignon", "analytics.py")
RAG_SANITIZER_PATH = os.path.join(PROJECT_ROOT, "opti_oignon", "rag_sanitizer.py")
RATE_LIMITER_PATH = os.path.join(PROJECT_ROOT, "opti_oignon", "rate_limiter.py")
ARTIFACTS_PATH = os.path.join(PROJECT_ROOT, "opti_oignon", "artifacts.py")
ROUTES_FILES_PATH = os.path.join(PROJECT_ROOT, "opti_oignon", "api", "routes_files.py")
ROUTES_USERS_PATH = os.path.join(PROJECT_ROOT, "opti_oignon", "api", "routes_users.py")
ROUTES_RAG_PATH = os.path.join(PROJECT_ROOT, "opti_oignon", "api", "routes_rag.py")
PYPROJECT_PATH = os.path.join(PROJECT_ROOT, "pyproject.toml")
CHANGELOG_PATH = os.path.join(PROJECT_ROOT, "CHANGELOG.md")

# Modules with MD5 calls
MD5_MODULES = [
    os.path.join(PROJECT_ROOT, "opti_oignon", "model_manager.py"),
    os.path.join(PROJECT_ROOT, "opti_oignon", "rag", "embeddings.py"),
    os.path.join(PROJECT_ROOT, "opti_oignon", "rag", "indexer.py"),
    os.path.join(PROJECT_ROOT, "opti_oignon", "session_fingerprint.py"),
    os.path.join(PROJECT_ROOT, "opti_oignon", "web_search.py"),
]


def _load_module(name, path):
    """Load a Python module by file path with isolation."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rate_limiter_mod():
    """Load the rate limiter module."""
    return _load_module("test_rate_limiter", RATE_LIMITER_PATH)


# ===================================================================
# 1. Version and Dependency Tests
# ===================================================================


class TestVersionBump:
    """Verify version bump to 3.2.5."""

    def test_version_is_3_2_5(self):
        source = open(VERSION_PATH).read()
        assert '__version__ = "3.2.5"' in source

    def test_changelog_has_v3_2_5(self):
        source = open(CHANGELOG_PATH).read()
        assert "## v3.2.5" in source

    def test_changelog_mentions_s156(self):
        source = open(CHANGELOG_PATH).read()
        assert "(S156)" in source


class TestDependencyVersions:
    """Verify pyproject.toml dependency version floors (SA-155-001 through SA-155-009)."""

    @pytest.fixture(autouse=True)
    def _load_pyproject(self):
        self.source = open(PYPROJECT_PATH).read()

    def test_requests_version_floor(self):
        assert 'requests>=2.33.0' in self.source

    def test_pypdf_version_floor(self):
        assert 'pypdf>=6.0.0' in self.source

    def test_setuptools_version_floor(self):
        assert 'setuptools>=78.1.1' in self.source

    def test_wheel_version_floor(self):
        assert 'wheel>=0.46.2' in self.source

    def test_pyjwt_version_floor(self):
        assert 'PyJWT>=2.12.0' in self.source

    def test_flask_version_floor(self):
        assert 'flask>=3.1.3' in self.source

    def test_werkzeug_version_floor(self):
        assert 'werkzeug>=3.1.6' in self.source


# ===================================================================
# 2. SQL Query Hardening Tests
# ===================================================================


class TestAnalyticsSQLHardening:
    """Verify SQL hardening in analytics.py (SA-155-020)."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = open(ANALYTICS_PATH).read()

    def test_no_fstring_sql_in_execute(self):
        """No f-string interpolation in .execute() SQL calls."""
        lines = self.source.split("\n")
        fstring_sql_lines = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Skip error messages and non-SQL f-strings
            if "raise ValueError" in stripped or "logger." in stripped:
                continue
            # Look for f" that precede SQL keywords
            if 'f"' in stripped or "f'" in stripped:
                rest = stripped[stripped.index("f") + 2:]
                if any(kw in rest.upper() for kw in ["SELECT", "INSERT", "DELETE", "UPDATE", "FROM performance"]):
                    fstring_sql_lines.append(i)
        assert fstring_sql_lines == [], (
            f"f-string SQL found at lines: {fstring_sql_lines}"
        )

    def test_no_format_sql(self):
        """No .format() in SQL query construction."""
        lines = self.source.split("\n")
        for i, line in enumerate(lines, 1):
            if ".format(" in line and ("SELECT" in line.upper() or "FROM" in line.upper()):
                pytest.fail(f"format() SQL at line {i}: {line.strip()}")

    def test_build_where_exists(self):
        assert "def _build_where(" in self.source

    def test_allowed_conditions_frozenset(self):
        assert "_ALLOWED_CONDITIONS: frozenset" in self.source

    def test_build_where_validates_conditions(self):
        """_build_where rejects unknown condition fragments."""
        mod = _load_module("test_analytics_sql", ANALYTICS_PATH)
        with pytest.raises(ValueError, match="Disallowed SQL condition"):
            mod._build_where(["timestamp >= ?", "evil = 'DROP TABLE'"])

    def test_build_where_empty(self):
        mod = _load_module("test_analytics_sql2", ANALYTICS_PATH)
        result = mod._build_where([])
        assert result == ""

    def test_build_where_single(self):
        mod = _load_module("test_analytics_sql3", ANALYTICS_PATH)
        result = mod._build_where(["timestamp >= ?"])
        assert result == "WHERE timestamp >= ?"

    def test_build_where_multiple(self):
        mod = _load_module("test_analytics_sql4", ANALYTICS_PATH)
        result = mod._build_where(["timestamp >= ?", "model_used = ?"])
        assert "WHERE" in result
        assert "timestamp >= ?" in result
        assert "model_used = ?" in result
        assert " AND " in result

    def test_allowed_conditions_contains_all_used(self):
        """All condition fragments used in analytics.py are in the allowlist."""
        mod = _load_module("test_analytics_sql5", ANALYTICS_PATH)
        required = {
            "timestamp >= ?", "timestamp <= ?", "model_used = ?",
            "pipeline_used = ?", "pipeline_used != ''", "model_used != ''",
            "task_type != ''", "was_routed = 1", "was_routed = 0",
        }
        assert required.issubset(mod._ALLOWED_CONDITIONS)


class TestRagSanitizerSQLHardening:
    """Verify SQL hardening in rag_sanitizer.py (SA-155-021)."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = open(RAG_SANITIZER_PATH).read()

    def test_no_fstring_sql_in_query_log(self):
        """The query_log method has no f-string SQL."""
        # Find the query_log method source
        in_method = False
        fstring_found = False
        for line in self.source.split("\n"):
            if "def query_log(" in line:
                in_method = True
                continue
            if in_method:
                if line.strip().startswith("def ") and "query_log" not in line:
                    break
                stripped = line.strip()
                if ('f"' in stripped or "f'" in stripped) and "Disallowed" not in stripped:
                    if any(kw in stripped.upper() for kw in ["SELECT", "FROM", "WHERE"]):
                        fstring_found = True
        assert not fstring_found, "f-string SQL still present in query_log()"

    def test_query_log_has_allowlist_validation(self):
        assert "Disallowed SQL condition" in self.source


# ===================================================================
# 3. Rate Limiter Tests
# ===================================================================


class TestRateLimiterModule:
    """Verify rate_limiter.py exists and has correct structure."""

    def test_module_exists(self):
        assert os.path.isfile(RATE_LIMITER_PATH)

    def test_checkpoint_sentinel(self):
        source = open(RATE_LIMITER_PATH).read()
        assert "checkpoint_before_apply = True" in source

    def test_no_french(self):
        source = open(RATE_LIMITER_PATH).read()
        french_pattern = re.compile(r"[àâäéèêëïîôùûüÿçœæ]", re.IGNORECASE)
        matches = french_pattern.findall(source)
        assert matches == [], f"French characters found: {matches}"


class TestRateLimiterSlidingWindow:
    """Test sliding window behavior."""

    def test_allows_under_limit(self, rate_limiter_mod):
        rl = rate_limiter_mod.RateLimiter()
        rl.configure("test_ep", max_requests=3, window_seconds=60)
        t = 1000.0
        for i in range(3):
            allowed, info = rl.check("test_ep", "key1", now=t + i)
            assert allowed, f"Request {i+1} should be allowed"
            assert info["allowed"] is True

    def test_denies_over_limit(self, rate_limiter_mod):
        rl = rate_limiter_mod.RateLimiter()
        rl.configure("test_ep2", max_requests=2, window_seconds=60)
        t = 2000.0
        rl.check("test_ep2", "key1", now=t)
        rl.check("test_ep2", "key1", now=t + 1)
        allowed, info = rl.check("test_ep2", "key1", now=t + 2)
        assert not allowed
        assert info["allowed"] is False
        assert info["remaining"] == 0
        assert info["retry_after"] > 0

    def test_window_expiry(self, rate_limiter_mod):
        rl = rate_limiter_mod.RateLimiter()
        rl.configure("test_ep3", max_requests=2, window_seconds=10)
        t = 3000.0
        rl.check("test_ep3", "key1", now=t)
        rl.check("test_ep3", "key1", now=t + 1)
        # Should be denied now
        allowed, _ = rl.check("test_ep3", "key1", now=t + 2)
        assert not allowed
        # After window expires, should be allowed again
        allowed, info = rl.check("test_ep3", "key1", now=t + 11)
        assert allowed
        assert info["remaining"] >= 0

    def test_key_isolation(self, rate_limiter_mod):
        rl = rate_limiter_mod.RateLimiter()
        rl.configure("test_ep4", max_requests=1, window_seconds=60)
        t = 4000.0
        rl.check("test_ep4", "user_a", now=t)
        # user_a is now at limit, user_b should still be allowed
        allowed, _ = rl.check("test_ep4", "user_b", now=t + 1)
        assert allowed

    def test_unconfigured_endpoint_allows(self, rate_limiter_mod):
        rl = rate_limiter_mod.RateLimiter()
        allowed, info = rl.check("nonexistent_endpoint", "key1", now=5000.0)
        assert allowed
        assert info["remaining"] == -1


class TestRateLimiterBulbeMode:
    """Test Bulbe mode stricter limits."""

    def test_effective_limit_daily(self, rate_limiter_mod):
        el = rate_limiter_mod.EndpointLimit(
            name="test", max_requests=10, window_seconds=60,
            bulbe_max_requests=3,
        )
        mr, ws = el.effective_limit(bulbe=False)
        assert mr == 10
        assert ws == 60

    def test_effective_limit_bulbe_explicit(self, rate_limiter_mod):
        el = rate_limiter_mod.EndpointLimit(
            name="test", max_requests=10, window_seconds=60,
            bulbe_max_requests=3, bulbe_window_seconds=30,
        )
        mr, ws = el.effective_limit(bulbe=True)
        assert mr == 3
        assert ws == 30

    def test_effective_limit_bulbe_default_half(self, rate_limiter_mod):
        """Without explicit bulbe_max_requests, defaults to half."""
        el = rate_limiter_mod.EndpointLimit(
            name="test", max_requests=10, window_seconds=60,
        )
        mr, ws = el.effective_limit(bulbe=True)
        assert mr == 5  # max(1, 10 // 2)
        assert ws == 60

    def test_effective_limit_bulbe_minimum_one(self, rate_limiter_mod):
        """Bulbe default never goes below 1."""
        el = rate_limiter_mod.EndpointLimit(
            name="test", max_requests=1, window_seconds=60,
        )
        mr, _ = el.effective_limit(bulbe=True)
        assert mr >= 1


class TestRateLimiterReset:
    """Test reset and cleanup functionality."""

    def test_reset_all(self, rate_limiter_mod):
        rl = rate_limiter_mod.RateLimiter()
        rl.configure("ep_a", max_requests=5, window_seconds=60)
        rl.configure("ep_b", max_requests=5, window_seconds=60)
        rl.check("ep_a", "k1", now=6000.0)
        rl.check("ep_b", "k2", now=6000.0)
        cleared = rl.reset()
        assert cleared >= 2

    def test_reset_by_endpoint(self, rate_limiter_mod):
        rl = rate_limiter_mod.RateLimiter()
        rl.configure("ep_c", max_requests=5, window_seconds=60)
        rl.configure("ep_d", max_requests=5, window_seconds=60)
        t = 7000.0
        rl.check("ep_c", "k1", now=t)
        rl.check("ep_d", "k1", now=t)
        cleared = rl.reset(endpoint="ep_c")
        assert cleared == 1
        # ep_d still has data
        status = rl.get_status("ep_d", "k1", now=t + 1)
        assert status["current_count"] == 1

    def test_reset_by_key(self, rate_limiter_mod):
        rl = rate_limiter_mod.RateLimiter()
        rl.configure("ep_e", max_requests=5, window_seconds=60)
        t = 8000.0
        rl.check("ep_e", "k1", now=t)
        rl.check("ep_e", "k2", now=t)
        cleared = rl.reset(endpoint="ep_e", key="k1")
        assert cleared == 1
        status = rl.get_status("ep_e", "k2", now=t + 1)
        assert status["current_count"] == 1

    def test_cleanup_expired(self, rate_limiter_mod):
        rl = rate_limiter_mod.RateLimiter()
        rl.configure("ep_f", max_requests=5, window_seconds=10)
        rl.check("ep_f", "k1", now=9000.0)
        # All entries expired
        removed = rl.cleanup_expired()
        assert removed >= 1


class TestRateLimiterStatus:
    """Test get_status and configured_endpoints."""

    def test_get_status_empty(self, rate_limiter_mod):
        rl = rate_limiter_mod.RateLimiter()
        rl.configure("ep_g", max_requests=5, window_seconds=60)
        status = rl.get_status("ep_g", "k1")
        assert status["configured"] is True
        assert status["current_count"] == 0
        assert status["remaining"] == 5

    def test_get_status_after_requests(self, rate_limiter_mod):
        rl = rate_limiter_mod.RateLimiter()
        rl.configure("ep_h", max_requests=5, window_seconds=60)
        t = 10000.0
        rl.check("ep_h", "k1", now=t)
        rl.check("ep_h", "k1", now=t + 1)
        status = rl.get_status("ep_h", "k1", now=t + 2)
        assert status["current_count"] == 2
        assert status["remaining"] == 3

    def test_get_status_unconfigured(self, rate_limiter_mod):
        rl = rate_limiter_mod.RateLimiter()
        status = rl.get_status("nonexistent", "k1")
        assert status["configured"] is False

    def test_configured_endpoints(self, rate_limiter_mod):
        rl = rate_limiter_mod.RateLimiter()
        eps = rl.configured_endpoints
        assert "file_upload" in eps
        assert "user_management" in eps


class TestRateLimiterDefaults:
    """Verify default endpoint configurations."""

    def test_file_upload_defaults(self, rate_limiter_mod):
        defaults = rate_limiter_mod._DEFAULT_LIMITS
        assert "file_upload" in defaults
        fu = defaults["file_upload"]
        assert fu.max_requests == 10
        assert fu.window_seconds == 60
        assert fu.bulbe_max_requests == 5

    def test_user_management_defaults(self, rate_limiter_mod):
        defaults = rate_limiter_mod._DEFAULT_LIMITS
        assert "user_management" in defaults
        um = defaults["user_management"]
        assert um.max_requests == 5
        assert um.window_seconds == 60
        assert um.bulbe_max_requests == 2


class TestRateLimiterConvenience:
    """Test module-level convenience function."""

    def test_rate_limit_check_function(self, rate_limiter_mod):
        assert callable(rate_limiter_mod.rate_limit_check)

    def test_rate_limiter_singleton(self, rate_limiter_mod):
        assert hasattr(rate_limiter_mod, "rate_limiter")
        assert isinstance(rate_limiter_mod.rate_limiter, rate_limiter_mod.RateLimiter)

    def test_info_dict_keys(self, rate_limiter_mod):
        rl = rate_limiter_mod.RateLimiter()
        rl.configure("test_keys", max_requests=5, window_seconds=60)
        _, info = rl.check("test_keys", "k1", now=11000.0)
        expected_keys = {"allowed", "remaining", "limit", "window_seconds", "retry_after", "message"}
        assert expected_keys == set(info.keys())

    def test_denied_info_has_retry_after(self, rate_limiter_mod):
        rl = rate_limiter_mod.RateLimiter()
        rl.configure("test_retry", max_requests=1, window_seconds=30)
        rl.check("test_retry", "k1", now=12000.0)
        _, info = rl.check("test_retry", "k1", now=12001.0)
        assert info["retry_after"] > 0
        assert "Rate limit exceeded" in info["message"]


# ===================================================================
# 4. Rate Limiting Integration Tests
# ===================================================================


class TestRateLimitingIntegration:
    """Verify rate limiting is wired into route files."""

    def test_routes_files_imports_rate_limiter(self):
        source = open(ROUTES_FILES_PATH).read()
        assert "rate_limit_check" in source

    def test_routes_files_has_check_function(self):
        source = open(ROUTES_FILES_PATH).read()
        assert "def _check_upload_rate(" in source

    def test_routes_files_upload_calls_rate_check(self):
        source = open(ROUTES_FILES_PATH).read()
        assert "_check_upload_rate(request)" in source

    def test_routes_files_upload_image_calls_rate_check(self):
        source = open(ROUTES_FILES_PATH).read()
        # Find upload_image and verify it calls rate check
        in_upload_image = False
        found = False
        for line in source.split("\n"):
            if "async def upload_image(" in line:
                in_upload_image = True
            if in_upload_image and "_check_upload_rate(request)" in line:
                found = True
                break
        assert found, "upload_image does not call _check_upload_rate"

    def test_routes_files_returns_429(self):
        source = open(ROUTES_FILES_PATH).read()
        assert "status_code=429" in source

    def test_routes_files_has_retry_after_header(self):
        source = open(ROUTES_FILES_PATH).read()
        assert "Retry-After" in source

    def test_routes_users_imports_rate_limiter(self):
        source = open(ROUTES_USERS_PATH).read()
        assert "rate_limit_check" in source

    def test_routes_users_has_check_function(self):
        source = open(ROUTES_USERS_PATH).read()
        assert "def _check_user_mgmt_rate(" in source

    def test_routes_users_export_calls_rate_check(self):
        source = open(ROUTES_USERS_PATH).read()
        in_export = False
        found = False
        for line in source.split("\n"):
            if "async def export_user_data(" in line:
                in_export = True
            if in_export and "_check_user_mgmt_rate(request)" in line:
                found = True
                break
        assert found, "export_user_data does not call rate check"

    def test_routes_users_delete_calls_rate_check(self):
        source = open(ROUTES_USERS_PATH).read()
        in_delete = False
        found = False
        for line in source.split("\n"):
            if "async def delete_user_data(" in line:
                in_delete = True
            if in_delete and "_check_user_mgmt_rate(request)" in line:
                found = True
                break
        assert found, "delete_user_data does not call rate check"

    def test_routes_users_derive_key_calls_rate_check(self):
        source = open(ROUTES_USERS_PATH).read()
        in_derive = False
        found = False
        for line in source.split("\n"):
            if "async def derive_user_key(" in line:
                in_derive = True
            if in_derive and "_check_user_mgmt_rate(request)" in line:
                found = True
                break
        assert found, "derive_user_key does not call rate check"


# ===================================================================
# 5. Path Validation Tests
# ===================================================================


class TestArtifactsPathSanitization:
    """Verify path traversal fix in artifacts.py (SA-155-040)."""

    def test_export_to_file_uses_basename(self):
        source = open(ARTIFACTS_PATH).read()
        assert "os.path.basename(artifact.filename)" in source

    def test_export_all_uses_basename(self):
        source = open(ARTIFACTS_PATH).read()
        assert "os.path.basename(a.filename)" in source

    def test_no_raw_filename_in_join(self):
        """No os.path.join with unsanitized filename."""
        source = open(ARTIFACTS_PATH).read()
        lines = source.split("\n")
        for i, line in enumerate(lines, 1):
            if "os.path.join(output_dir," in line:
                if "os.path.basename" not in line and "fname" not in line:
                    pytest.fail(
                        f"Line {i}: os.path.join without basename: {line.strip()}"
                    )


class TestRoutesRagPathValidation:
    """Verify path traversal fix in routes_rag.py (SA-155-042)."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = open(ROUTES_RAG_PATH).read()

    def test_rejects_dotdot(self):
        assert '".."' in self.source or "'..' " in self.source

    def test_uses_resolve(self):
        assert ".resolve()" in self.source

    def test_checks_symlink(self):
        assert "is_symlink()" in self.source

    def test_ingest_folder_has_validation(self):
        """ingest_folder has directory validation before is_dir check."""
        in_method = False
        found_validation = False
        found_is_dir = False
        for line in self.source.split("\n"):
            if "def ingest_folder(" in line:
                in_method = True
            if in_method:
                if '".."' in line:
                    found_validation = True
                if ".is_dir()" in line:
                    found_is_dir = True
                    break
        assert found_validation, "No '..' validation in ingest_folder"
        assert found_is_dir, "No is_dir check in ingest_folder"


# ===================================================================
# 6. MD5 usedforsecurity=False Tests
# ===================================================================


class TestMD5SecurityFlag:
    """Verify usedforsecurity=False on all MD5 calls (SA-155-064)."""

    @pytest.mark.parametrize("module_path", MD5_MODULES)
    def test_md5_has_usedforsecurity_false(self, module_path):
        source = open(module_path).read()
        md5_lines = [
            (i, line.strip())
            for i, line in enumerate(source.split("\n"), 1)
            if "hashlib.md5(" in line
        ]
        assert len(md5_lines) > 0, f"No hashlib.md5 calls in {module_path}"
        for lineno, line in md5_lines:
            assert "usedforsecurity=False" in line, (
                f"{module_path}:{lineno}: MD5 call missing usedforsecurity=False: {line}"
            )

    def test_no_unflagged_md5_in_codebase(self):
        """Scan entire opti_oignon for md5 calls without the flag."""
        bad = []
        for root, dirs, files in os.walk(os.path.join(PROJECT_ROOT, "opti_oignon")):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                if not f.endswith(".py"):
                    continue
                path = os.path.join(root, f)
                for i, line in enumerate(open(path).readlines(), 1):
                    if "hashlib.md5(" in line and "usedforsecurity=False" not in line:
                        bad.append(f"{path}:{i}")
        assert bad == [], f"MD5 calls without usedforsecurity=False: {bad}"


# ===================================================================
# 7. No French in Modified Files
# ===================================================================


class TestNoFrenchInModifiedFiles:
    """Verify no French text in files modified during S156."""

    MODIFIED_FILES = [
        RATE_LIMITER_PATH,
        ROUTES_FILES_PATH,
    ]

    @pytest.mark.parametrize("filepath", MODIFIED_FILES)
    def test_no_french_characters(self, filepath):
        source = open(filepath).read()
        french_chars = re.findall(r"[àâäéèêëïîôùûüÿçœæ]", source, re.IGNORECASE)
        assert french_chars == [], (
            f"French characters in {os.path.basename(filepath)}: {french_chars}"
        )

    def test_routes_files_no_french_words(self):
        source = open(ROUTES_FILES_PATH).read()
        french_words = [
            "autorisee", "fichier", "taille", "encodage",
            "validation de", "retournant", "attachement",
        ]
        for word in french_words:
            assert word.lower() not in source.lower(), (
                f"French word '{word}' found in routes_files.py"
            )

    def test_rate_limiter_no_french_words(self):
        source = open(RATE_LIMITER_PATH).read()
        # Use word-boundary matching to avoid false positives
        # (e.g., "limite" matching inside "limiter")
        french_patterns = [
            r"\bconfigurer\b", r"\bfenetre\b", r"\brequete\b",
            r"\blimite\b",  # word boundary avoids matching "limiter"
        ]
        for pattern in french_patterns:
            matches = re.findall(pattern, source, re.IGNORECASE)
            assert matches == [], (
                f"French pattern '{pattern}' found in rate_limiter.py"
            )


# ===================================================================
# 8. Requests Timeout Verification
# ===================================================================


class TestRequestsTimeout:
    """Verify all requests.*() calls have timeout parameter (SA-155-066)."""

    def test_no_requests_without_timeout(self):
        """AST-based check for requests calls missing timeout."""
        import ast

        bad = []
        for root, dirs, files in os.walk(os.path.join(PROJECT_ROOT, "opti_oignon")):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                if not f.endswith(".py"):
                    continue
                path = os.path.join(root, f)
                try:
                    source = open(path).read()
                    tree = ast.parse(source)
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        func = node.func
                        if (isinstance(func, ast.Attribute)
                                and func.attr in ("post", "get", "put", "delete", "patch", "head")
                                and isinstance(func.value, ast.Name)
                                and func.value.id == "requests"):
                            has_timeout = any(kw.arg == "timeout" for kw in node.keywords)
                            if not has_timeout:
                                bad.append(f"{path}:{node.lineno}")
        assert bad == [], f"requests calls without timeout: {bad}"


# ===================================================================
# 9. Checkpoint Sentinel Tests
# ===================================================================


class TestCheckpointSentinel:
    """Verify checkpoint_before_apply in new modules."""

    def test_rate_limiter_sentinel(self):
        source = open(RATE_LIMITER_PATH).read()
        assert "checkpoint_before_apply = True" in source

    def test_rate_limiter_sentinel_is_hardcoded(self):
        """The sentinel must be a literal True, not a variable."""
        mod = _load_module("test_sentinel_rl", RATE_LIMITER_PATH)
        assert mod.checkpoint_before_apply is True
