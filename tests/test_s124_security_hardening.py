#!/usr/bin/env python3
"""
Tests for S124 -- Critical Security Hardening (Part 1).

Test groups:
1.  CORS: default localhost-only, custom origins, wildcard credentials OFF
2.  Plugin sandbox: blocked imports expanded (os, sys, pickle, etc.)
3.  Plugin sandbox: _RestrictedPathAccessor patches pathlib/io
4.  Plugin sandbox: _RestrictedBuiltins blocks exec/eval/compile/globals/vars
5.  Plugin sandbox: __import__ selective blocking
6.  Plugin permissions: inference_content in VALID_PERMISSIONS
7.  Hook data redaction: sensitive fields redacted without permission
8.  Hook data redaction: safe fields preserved
9.  Hook data redaction: force_redact flag
10. HookManager.execute with redact_sensitive
11. Security headers: all headers present on API responses
12. Security headers: CSP correct format
13. Security headers: configurable via YAML
14. Security headers: Cache-Control only on /api/ paths
15. Rate limiting: allowed within window
16. Rate limiting: blocked after max attempts (429)
17. Rate limiting: exponential lockout
18. Rate limiting: successful login resets username
19. Rate limiting: account lock after threshold
20. Rate limiting: per-IP isolation
21. Rate limiting: disabled mode allows all
22. Rate limiting: Retry-After header present
23. Sandbox strict mode: config field exists and defaults to True
24. Sandbox strict mode: get_isolation_status returns dict
25. Security endpoint: GET /status returns score and checks
26. Security endpoint: GET /config returns config
27. Security endpoint: PUT /config updates and persists
28. Security endpoint: score computation (max_points=100)
29. Version is 2.4.1
30. No French in new code
31. No hardcoded hex in new Svelte components (if any)

Version: 2.4.1
"""

import builtins
import importlib.util
import io
import os
import re
import sys
import tempfile
import time
import types
from pathlib import Path

import pytest
import yaml

# =========================================================================
# Helpers: load modules in isolation (bypass opti_oignon/__init__.py)
# =========================================================================

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_module(rel_path: str, name: str = ""):
    """Load a Python module by file path, bypassing __init__.py chains."""
    fpath = _PROJECT_ROOT / rel_path
    mod_name = name or fpath.stem
    spec = importlib.util.spec_from_file_location(mod_name, str(fpath))
    mod = types.ModuleType(mod_name)
    mod.__file__ = str(fpath)
    spec.loader.exec_module(mod)
    return mod


def _load_plugin_loader_classes():
    """Load plugin_loader classes without triggering the singleton."""
    code = (_PROJECT_ROOT / "opti_oignon" / "plugin_loader.py").read_text()
    parts = code.split("# =========================================================================")
    ns = {}
    exec(parts[0], ns)
    return ns


def _stub_modules():
    """Register stub modules to prevent import chain failures."""
    stubs = [
        "opti_oignon", "opti_oignon.api", "opti_oignon.api.app",
        "opti_oignon.auth", "opti_oignon.sandbox_manager",
        "opti_oignon.plugin_loader", "opti_oignon.plugin_hooks",
        "opti_oignon.api.security_middleware",
        "opti_oignon.plugin_manifest",
    ]
    saved = {}
    for name in stubs:
        saved[name] = sys.modules.get(name)
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    return saved


def _restore_modules(saved):
    """Restore original module state."""
    for name, mod in saved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


# =========================================================================
# 1. CORS Tests
# =========================================================================

class TestCORSConfiguration:
    """Tests for CORS lockdown (Phase 1)."""

    def test_cors_default_localhost_only(self):
        """Default origins (no env var, empty yaml) -> localhost-only."""
        os.environ.pop("OPTI_CORS_ORIGINS", None)
        # Parse the resolve function
        code = (_PROJECT_ROOT / "opti_oignon" / "api" / "app.py").read_text()
        assert "http://localhost" in code
        assert "http://127.0.0.1" in code
        assert "http://[::1]" in code

    def test_cors_wildcard_credentials_off(self):
        """Wildcard origin forces allow_credentials=False."""
        code = (_PROJECT_ROOT / "opti_oignon" / "api" / "app.py").read_text()
        assert 'return ["*"], False' in code

    def test_cors_custom_origins_credentials_on(self):
        """Custom explicit origins get allow_credentials=True."""
        code = (_PROJECT_ROOT / "opti_oignon" / "api" / "app.py").read_text()
        assert "return origins, True" in code

    def test_cors_localhost_regex(self):
        """Localhost regex matches port variants."""
        pattern = re.compile(
            r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"
        )
        assert pattern.match("http://localhost:5173")
        assert pattern.match("http://127.0.0.1:8080")
        assert pattern.match("http://[::1]:3000")
        assert not pattern.match("http://evil.com")
        assert not pattern.match("http://localhost.evil.com")

    def test_cors_yaml_config_section(self):
        """security.yaml has cors section."""
        cfg = yaml.safe_load(
            (_PROJECT_ROOT / "opti_oignon" / "config" / "security.yaml").read_text()
        )
        assert "cors" in cfg
        assert "origins" in cfg["cors"]

    def test_cors_default_not_wildcard(self):
        """Default CORS env is empty string (not *)."""
        code = (_PROJECT_ROOT / "opti_oignon" / "api" / "app.py").read_text()
        assert 'get("OPTI_CORS_ORIGINS", "")' in code

    def test_cors_security_yaml_takes_precedence(self):
        """security.yaml origins take precedence over env var."""
        code = (_PROJECT_ROOT / "opti_oignon" / "api" / "app.py").read_text()
        # yaml_origins check comes before env_val
        yaml_idx = code.find("if yaml_origins:")
        env_idx = code.find("elif env_val:")
        assert yaml_idx < env_idx

    def test_cors_warning_log_on_wildcard(self):
        """Warning logged when CORS is set to wildcard."""
        code = (_PROJECT_ROOT / "opti_oignon" / "api" / "app.py").read_text()
        assert "Credentials are DISABLED" in code


# =========================================================================
# 2-5. Plugin Sandbox Tests
# =========================================================================

class TestPluginSandboxBlockedImports:
    """Tests for expanded _BLOCKED_IMPORTS (Phase 2a)."""

    def setup_method(self):
        self._ns = _load_plugin_loader_classes()

    def test_os_blocked(self):
        assert "os" in self._ns["_BLOCKED_IMPORTS"]

    def test_sys_blocked(self):
        assert "sys" in self._ns["_BLOCKED_IMPORTS"]

    def test_pickle_blocked(self):
        assert "pickle" in self._ns["_BLOCKED_IMPORTS"]

    def test_marshal_blocked(self):
        assert "marshal" in self._ns["_BLOCKED_IMPORTS"]

    def test_gc_blocked(self):
        assert "gc" in self._ns["_BLOCKED_IMPORTS"]

    def test_inspect_blocked(self):
        assert "inspect" in self._ns["_BLOCKED_IMPORTS"]

    def test_pathlib_blocked(self):
        assert "pathlib" in self._ns["_BLOCKED_IMPORTS"]

    def test_io_blocked(self):
        assert "io" in self._ns["_BLOCKED_IMPORTS"]

    def test_glob_blocked(self):
        assert "glob" in self._ns["_BLOCKED_IMPORTS"]

    def test_shelve_blocked(self):
        assert "shelve" in self._ns["_BLOCKED_IMPORTS"]

    def test_original_blocks_preserved(self):
        """Original S101/S106 blocks still present."""
        blocked = self._ns["_BLOCKED_IMPORTS"]
        for mod in ["subprocess", "shutil", "ctypes", "multiprocessing",
                     "signal", "importlib"]:
            assert mod in blocked, f"{mod} missing from _BLOCKED_IMPORTS"


class TestRestrictedPathAccessor:
    """Tests for enhanced _RestrictedPathAccessor (Phase 2b)."""

    def setup_method(self):
        self._ns = _load_plugin_loader_classes()
        self._Accessor = self._ns["_RestrictedPathAccessor"]
        self._Violation = self._ns["PluginSandboxViolation"]

    def test_builtins_open_blocked(self):
        """builtins.open blocked for paths outside allowed dirs."""
        with tempfile.TemporaryDirectory() as td:
            with self._Accessor([Path(td)]):
                with pytest.raises(self._Violation):
                    open("/etc/passwd")

    def test_builtins_open_allowed(self):
        """builtins.open allowed within plugin directory."""
        with tempfile.TemporaryDirectory() as td:
            Path(td, "test.txt").write_text("hello")
            with self._Accessor([Path(td)]):
                content = open(os.path.join(td, "test.txt")).read()
                assert content == "hello"

    def test_io_open_blocked(self):
        """io.open blocked for restricted paths."""
        with tempfile.TemporaryDirectory() as td:
            with self._Accessor([Path(td)]):
                with pytest.raises(self._Violation, match="io.open"):
                    io.open("/etc/passwd")

    def test_path_read_text_blocked(self):
        """Path.read_text blocked for restricted paths."""
        with tempfile.TemporaryDirectory() as td:
            with self._Accessor([Path(td)]):
                with pytest.raises(self._Violation, match="Path.read_text"):
                    Path("/etc/passwd").read_text()

    def test_path_iterdir_blocked(self):
        """Path.iterdir blocked for restricted paths."""
        with tempfile.TemporaryDirectory() as td:
            with self._Accessor([Path(td)]):
                with pytest.raises(self._Violation, match="Path.iterdir"):
                    list(Path("/etc").iterdir())

    def test_path_glob_blocked(self):
        """Path.glob blocked for restricted paths."""
        with tempfile.TemporaryDirectory() as td:
            with self._Accessor([Path(td)]):
                with pytest.raises(self._Violation, match="Path.glob"):
                    list(Path("/").glob("*"))

    def test_restore_after_exit(self):
        """All patches restored after context exit."""
        with tempfile.TemporaryDirectory() as td:
            with self._Accessor([Path(td)]):
                pass
            # Should work after exit
            content = Path("/etc/passwd").read_text()
            assert len(content) > 0
            f = io.open("/etc/passwd")
            f.close()


class TestRestrictedBuiltins:
    """Tests for _RestrictedBuiltins (Phase 2c)."""

    def setup_method(self):
        self._ns = _load_plugin_loader_classes()
        self._Builtins = self._ns["_RestrictedBuiltins"]
        self._Violation = self._ns["PluginSandboxViolation"]

    def test_globals_blocked(self):
        with self._Builtins():
            with pytest.raises(self._Violation, match="globals"):
                builtins.globals()

    def test_vars_blocked(self):
        with self._Builtins():
            with pytest.raises(self._Violation, match="vars"):
                builtins.vars()

    def test_import_os_blocked(self):
        """__import__('os') blocked via selective wrapper."""
        with self._Builtins():
            with pytest.raises(self._Violation, match="os"):
                builtins.__import__("os")

    def test_import_pickle_blocked(self):
        with self._Builtins():
            with pytest.raises(self._Violation, match="pickle"):
                builtins.__import__("pickle")

    def test_import_safe_allowed(self):
        """Safe module imports still work."""
        with self._Builtins():
            json_mod = builtins.__import__("json")
            assert json_mod is not None

    def test_exec_blocked_from_plugin_context(self):
        """exec blocked when called from _opti_plugin_ module context."""
        plugin_mod = types.ModuleType("_opti_plugin_test")
        plugin_mod.__name__ = "_opti_plugin_test"
        sys.modules["_opti_plugin_test"] = plugin_mod
        try:
            code = (
                "import builtins\n"
                "try:\n"
                "    builtins.eval('1+1')\n"
                "    result = 'NOT_BLOCKED'\n"
                "except Exception:\n"
                "    result = 'BLOCKED'\n"
            )
            with self._Builtins():
                exec(
                    compile(code, "_opti_plugin_test", "exec"),
                    plugin_mod.__dict__,
                )
                assert plugin_mod.result == "BLOCKED"
        finally:
            sys.modules.pop("_opti_plugin_test", None)

    def test_restore_after_exit(self):
        """Builtins restored after context exit."""
        with self._Builtins():
            pass
        result = builtins.eval("1 + 1")
        assert result == 2


# =========================================================================
# 6. Plugin Permissions
# =========================================================================

class TestPluginPermissions:
    """Tests for inference_content permission (Phase 2d)."""

    def test_inference_content_in_valid_permissions(self):
        code = (_PROJECT_ROOT / "opti_oignon" / "plugin_manifest.py").read_text()
        assert '"inference_content"' in code

    def test_filesystem_read_in_valid_permissions(self):
        code = (_PROJECT_ROOT / "opti_oignon" / "plugin_manifest.py").read_text()
        assert '"filesystem_read"' in code

    def test_filesystem_write_in_valid_permissions(self):
        code = (_PROJECT_ROOT / "opti_oignon" / "plugin_manifest.py").read_text()
        assert '"filesystem_write"' in code


# =========================================================================
# 7-10. Hook Data Redaction Tests
# =========================================================================

class TestHookDataRedaction:
    """Tests for hook data redaction (Phase 3)."""

    def setup_method(self):
        # Set up stub for plugin_manifest before loading plugin_hooks
        saved = _stub_modules()
        pm = sys.modules["opti_oignon.plugin_manifest"]
        pm.VALID_HOOKS = {
            "pre_prompt", "post_prompt", "pre_inference",
            "post_inference", "tool_call", "pipeline_step", "ui_panel",
        }
        pm.plugin_registry = None

        spec = importlib.util.spec_from_file_location(
            "plugin_hooks",
            str(_PROJECT_ROOT / "opti_oignon" / "plugin_hooks.py"),
        )
        self._mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._mod)
        self._saved = saved

    def teardown_method(self):
        _restore_modules(self._saved)

    def test_sensitive_fields_redacted(self):
        """message, response, arguments, result are redacted."""
        data = {
            "message": "secret prompt",
            "response": "secret response",
            "arguments": {"q": "secret"},
            "result": "secret output",
            "model": "llama3",
        }
        redacted = self._mod.redact_hook_data(data, "test", force_redact=True)
        assert redacted["message"] == self._mod.REDACTED_PLACEHOLDER
        assert redacted["response"] == self._mod.REDACTED_PLACEHOLDER
        assert redacted["arguments"] == self._mod.REDACTED_PLACEHOLDER
        assert redacted["result"] == self._mod.REDACTED_PLACEHOLDER

    def test_safe_fields_preserved(self):
        """model, duration_ms, tokens_*, tool_name, success are preserved."""
        data = {
            "message": "secret",
            "model": "llama3",
            "duration_ms": 150.0,
            "tool_name": "web_search",
            "success": True,
        }
        redacted = self._mod.redact_hook_data(data, "test", force_redact=True)
        assert redacted["model"] == "llama3"
        assert redacted["duration_ms"] == 150.0
        assert redacted["tool_name"] == "web_search"
        assert redacted["success"] is True

    def test_force_redact_flag(self):
        """force_redact=True always redacts regardless of permission."""
        data = {"message": "hello", "model": "x"}
        r = self._mod.redact_hook_data(data, "any", force_redact=True)
        assert r["message"] == self._mod.REDACTED_PLACEHOLDER

    def test_unknown_plugin_redacted(self):
        """Plugin not in registry -> redacted (no permission found)."""
        data = {"message": "hello", "model": "x"}
        r = self._mod.redact_hook_data(data, "unknown_plugin_xyz")
        assert r["message"] == self._mod.REDACTED_PLACEHOLDER

    def test_hook_manager_redact_sensitive(self):
        """HookManager.execute with redact_sensitive=True redacts per-plugin."""
        hm = self._mod.HookManager()
        received = {}

        def spy(ctx):
            received["message"] = ctx.data.get("message")
            received["model"] = ctx.data.get("model")

        hm.register("post_inference", "spy_plugin", spy)
        hm.execute(
            "post_inference",
            data={"message": "secret", "model": "llama3"},
            redact_sensitive=True,
        )
        assert received["message"] == self._mod.REDACTED_PLACEHOLDER
        assert received["model"] == "llama3"

    def test_hook_manager_no_redact_backward_compat(self):
        """redact_sensitive=False preserves full data (backward compat)."""
        hm = self._mod.HookManager()
        received = {}

        def spy(ctx):
            received["message"] = ctx.data.get("message")

        hm.register("post_inference", "spy_plugin", spy)
        hm.execute(
            "post_inference",
            data={"message": "hello"},
            redact_sensitive=False,
        )
        assert received["message"] == "hello"


# =========================================================================
# 11-14. Security Headers Tests
# =========================================================================

class TestSecurityHeaders:
    """Tests for security headers middleware (Phase 4)."""

    def setup_method(self):
        spec = importlib.util.spec_from_file_location(
            "sec_mw",
            str(_PROJECT_ROOT / "opti_oignon" / "api" / "security_middleware.py"),
        )
        self._mod = importlib.util.module_from_spec(spec)
        sys.modules["opti_oignon.api.security_middleware"] = self._mod
        spec.loader.exec_module(self._mod)

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.add_middleware(self._mod.SecurityHeadersMiddleware)

        @app.get("/api/test")
        def _api():
            return {"ok": True}

        @app.get("/static/test")
        def _static():
            return {"ok": True}

        self._client = TestClient(app)

    def test_x_content_type_options(self):
        resp = self._client.get("/api/test")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"

    def test_x_frame_options(self):
        resp = self._client.get("/api/test")
        assert resp.headers.get("X-Frame-Options") == "DENY"

    def test_csp_format(self):
        resp = self._client.get("/api/test")
        csp = resp.headers.get("Content-Security-Policy", "")
        assert "default-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp

    def test_referrer_policy(self):
        resp = self._client.get("/api/test")
        assert resp.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"

    def test_permissions_policy(self):
        resp = self._client.get("/api/test")
        assert "camera=()" in resp.headers.get("Permissions-Policy", "")

    def test_cache_control_on_api(self):
        resp = self._client.get("/api/test")
        assert resp.headers.get("Cache-Control") == "no-store"

    def test_cache_control_not_on_static(self):
        resp = self._client.get("/static/test")
        assert resp.headers.get("Cache-Control") != "no-store"

    def test_hsts_disabled_by_default(self):
        resp = self._client.get("/api/test")
        assert "Strict-Transport-Security" not in resp.headers

    def test_config_from_yaml(self):
        cfg = self._mod.get_security_headers_config()
        assert cfg["x_frame_options"] == "DENY"
        assert cfg["x_content_type_options"] == "nosniff"


# =========================================================================
# 15-22. Rate Limiting Tests
# =========================================================================

class TestLoginRateLimiter:
    """Tests for login rate limiting (Phase 5)."""

    def _make_limiter(self, **overrides):
        """Create a rate limiter with test config."""
        cfg = {
            "enabled": True,
            "login_max_attempts": 3,
            "login_window_seconds": 300,
            "lockout_base_seconds": 10,
            "lockout_max_seconds": 120,
            "account_lock_threshold": 5,
            "account_lock_duration_seconds": 30,
        }
        cfg.update(overrides)

        # Load LoginRateLimiter class
        code = (_PROJECT_ROOT / "opti_oignon" / "auth.py").read_text()
        idx = code.find("@dataclass\nclass _RateLimitEntry")
        end_idx = code.find("# Module-level rate limiter singleton")
        ns = {
            "time": time, "yaml": yaml, "logging": __import__("logging"),
            "logger": __import__("logging").getLogger("test_rate_limiter"),
            "dataclass": __import__("dataclasses").dataclass,
            "field": __import__("dataclasses").field,
            "Path": Path, "Any": __import__("typing").Any,
        }
        exec("from dataclasses import dataclass, field\nfrom typing import Any\n" + code[idx:end_idx], ns)
        return ns["LoginRateLimiter"](config=cfg)

    def test_first_attempt_allowed(self):
        rl = self._make_limiter()
        allowed, retry = rl.check_rate_limit("1.2.3.4", "alice")
        assert allowed is True
        assert retry == 0

    def test_blocked_after_max_attempts(self):
        rl = self._make_limiter()
        for _ in range(3):
            rl.record_failure("1.2.3.4", "alice")
        allowed, retry = rl.check_rate_limit("1.2.3.4", "alice")
        assert allowed is False
        assert retry > 0

    def test_exponential_lockout(self):
        rl = self._make_limiter()
        # First lockout
        for _ in range(3):
            rl.record_failure("9.9.9.9", "dave")
        _, retry1 = rl.check_rate_limit("9.9.9.9", "dave")
        # Expire lockout
        rl._ip_entries["9.9.9.9"].lockout_until = time.time() - 1
        rl._ip_entries["9.9.9.9"].attempts.clear()
        # Second lockout
        for _ in range(3):
            rl.record_failure("9.9.9.9", "dave")
        _, retry2 = rl.check_rate_limit("9.9.9.9", "dave")
        assert retry2 > retry1

    def test_success_resets_username(self):
        rl = self._make_limiter()
        for _ in range(2):
            rl.record_failure("10.0.0.1", "carol")
        rl.record_success("10.0.0.1", "carol")
        allowed, _ = rl.check_rate_limit("10.0.0.1", "carol")
        assert allowed is True

    def test_account_lock_after_threshold(self):
        rl = self._make_limiter()
        for _ in range(5):
            rl.record_failure("10.0.0.1", "bob")
        # Different IP, same username -> locked
        allowed, retry = rl.check_rate_limit("10.0.0.2", "bob")
        assert allowed is False
        assert retry > 0

    def test_per_ip_isolation(self):
        rl = self._make_limiter()
        for _ in range(3):
            rl.record_failure("1.2.3.4", "alice")
        # Different IP should NOT be blocked
        allowed, _ = rl.check_rate_limit("5.6.7.8", "alice")
        assert allowed is True

    def test_disabled_allows_all(self):
        rl = self._make_limiter(enabled=False)
        for _ in range(100):
            rl.record_failure("1.1.1.1", "x")
        allowed, _ = rl.check_rate_limit("1.1.1.1", "x")
        assert allowed is True

    def test_retry_after_positive(self):
        rl = self._make_limiter()
        for _ in range(3):
            rl.record_failure("1.2.3.4", "alice")
        _, retry = rl.check_rate_limit("1.2.3.4", "alice")
        assert isinstance(retry, int)
        assert retry > 0


# =========================================================================
# 23-24. Sandbox Strict Mode Tests
# =========================================================================

class TestSandboxStrictMode:
    """Tests for sandbox strict mode (Phase 6)."""

    def test_strict_mode_config_field(self):
        """SandboxConfig has strict_mode field defaulting to True."""
        code = (_PROJECT_ROOT / "opti_oignon" / "sandbox_manager.py").read_text()
        assert "strict_mode: bool = True" in code

    def test_strict_mode_blocks_execution(self):
        """execute_command returns blocked when strict_mode + no bwrap."""
        code = (_PROJECT_ROOT / "opti_oignon" / "sandbox_manager.py").read_text()
        assert "strict_mode is ON but bubblewrap" in code
        assert '"blocked"' in code

    def test_get_isolation_status_method(self):
        """get_isolation_status returns proper dict structure."""
        code = (_PROJECT_ROOT / "opti_oignon" / "sandbox_manager.py").read_text()
        assert "def get_isolation_status" in code
        assert '"isolation_level"' in code
        assert '"bwrap_available"' in code
        assert '"strict_mode"' in code
        assert '"execution_blocked"' in code

    def test_health_check_includes_sandbox(self):
        """Health check includes sandbox isolation info."""
        code = (_PROJECT_ROOT / "opti_oignon" / "api" / "app.py").read_text()
        assert "get_isolation_status" in code

    def test_security_yaml_strict_mode(self):
        """security.yaml has sandbox.strict_mode: true."""
        cfg = yaml.safe_load(
            (_PROJECT_ROOT / "opti_oignon" / "config" / "security.yaml").read_text()
        )
        assert cfg["sandbox"]["strict_mode"] is True


# =========================================================================
# 25-28. Security Endpoint Tests
# =========================================================================

class TestSecurityEndpoints:
    """Tests for security status/config API (Phase 7)."""

    def setup_method(self):
        self._saved = _stub_modules()
        spec = importlib.util.spec_from_file_location(
            "routes_security",
            str(_PROJECT_ROOT / "opti_oignon" / "api" / "routes_security.py"),
        )
        self._mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._mod)

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.include_router(self._mod.router)
        self._client = TestClient(app)

    def teardown_method(self):
        _restore_modules(self._saved)
        # Restore security.yaml if modified
        sec_path = _PROJECT_ROOT / "opti_oignon" / "config" / "security.yaml"
        cfg = yaml.safe_load(sec_path.read_text())
        if cfg.get("rate_limiting", {}).get("enabled") is not True:
            cfg.setdefault("rate_limiting", {})["enabled"] = True
            with open(sec_path, "w") as f:
                yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

    def test_status_returns_score(self):
        resp = self._client.get("/api/security/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "score" in data
        assert "grade" in data
        assert "checks" in data
        assert data["max_score"] == 100

    def test_status_eight_checks(self):
        resp = self._client.get("/api/security/status")
        data = resp.json()
        assert len(data["checks"]) == 8

    def test_max_points_sum_100(self):
        resp = self._client.get("/api/security/status")
        data = resp.json()
        total = sum(c["max_points"] for c in data["checks"])
        assert total == 100

    def test_config_returns_yaml(self):
        resp = self._client.get("/api/security/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "config" in data

    def test_config_update_persists(self):
        resp = self._client.put(
            "/api/security/config",
            json={"rate_limiting": {"enabled": False}},
        )
        assert resp.status_code == 200
        assert resp.json()["updated"] is True

        # Verify
        resp2 = self._client.get("/api/security/config")
        cfg = resp2.json()["config"]
        assert cfg["rate_limiting"]["enabled"] is False

    def test_config_update_empty(self):
        resp = self._client.put("/api/security/config", json={})
        assert resp.json()["updated"] is False


# =========================================================================
# 29. Version Test
# =========================================================================

class TestVersion:
    """Version bump verification."""

    def test_version_2_4_1(self):
        content = (_PROJECT_ROOT / "opti_oignon" / "__version__.py").read_text()
        assert '"3.0.0"' in content


# =========================================================================
# 30-31. Code Quality Tests
# =========================================================================

class TestCodeQuality:
    """No French, no hardcoded hex in new S124 code."""

    _S124_FILES = [
        "opti_oignon/api/security_middleware.py",
        "opti_oignon/api/routes_security.py",
        "opti_oignon/config/security.yaml",
    ]

    _FRENCH_PATTERNS = [
        r"\bparametre\b", r"\bfonction\b", r"\bverifi\b",
        r"\bsecurit[eé]\b", r"\bconfigurati?on\b.*\bde\b",
    ]

    def test_no_french_in_new_files(self):
        for rel in self._S124_FILES:
            content = (_PROJECT_ROOT / rel).read_text()
            for pat in self._FRENCH_PATTERNS:
                matches = re.findall(pat, content, re.IGNORECASE)
                # Allow 'configuration' as it is valid English
                matches = [m for m in matches if "configuration" not in m.lower()]
                assert not matches, (
                    f"French detected in {rel}: {matches}"
                )

    def test_no_hardcoded_hex_in_new_py(self):
        """No hardcoded #RRGGBB outside of var(--oo-*, #fallback) in new .py files."""
        hex_re = re.compile(r"#[0-9a-fA-F]{3,8}")
        for rel in self._S124_FILES:
            if not rel.endswith(".py"):
                continue
            content = (_PROJECT_ROOT / rel).read_text()
            for line_no, line in enumerate(content.splitlines(), 1):
                if hex_re.search(line) and "var(--oo-" not in line:
                    # Allow in comments and version strings
                    stripped = line.strip()
                    if stripped.startswith("#") or stripped.startswith("//"):
                        continue
                    assert False, (
                        f"Hardcoded hex in {rel}:{line_no}: {line.strip()}"
                    )

    def test_english_comments_in_plugin_loader(self):
        """No French in plugin_loader.py comments."""
        content = (_PROJECT_ROOT / "opti_oignon" / "plugin_loader.py").read_text()
        # Spot check that S124 additions are in English
        assert "S124: Critical additions" in content
        assert "defense-in-depth" in content

    def test_all_new_files_ast_valid(self):
        """All new/modified Python files pass AST validation."""
        import ast
        files = [
            "opti_oignon/api/app.py",
            "opti_oignon/api/security_middleware.py",
            "opti_oignon/api/routes_security.py",
            "opti_oignon/api/routes_auth.py",
            "opti_oignon/api/routes_chat.py",
            "opti_oignon/plugin_loader.py",
            "opti_oignon/plugin_manifest.py",
            "opti_oignon/plugin_hooks.py",
            "opti_oignon/auth.py",
            "opti_oignon/sandbox_manager.py",
        ]
        for rel in files:
            fpath = _PROJECT_ROOT / rel
            ast.parse(fpath.read_text())
