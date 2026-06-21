#!/usr/bin/env python3
"""
Tests for S138 — safe_connect Migration + SQL Parameterization + CSS Fix.

Covers:
  - No raw sqlite3.connect() in core modules (exclude plugins/, db_utils.py, db_encryption.py)
  - No f-string SQL in any core module
  - No bare #fff/#000 outside var() in Svelte files
  - safe_connect() accepts timeout and check_same_thread kwargs
  - security_scan.py passes no_raw_sqlite and no_fstring_sql checks
  - Allowlist constants exist in hardened modules
  - Version bump to 3.0.1
  - AST validation on all modified files
"""

import ast
import importlib.util
import os
import re
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_DIR = _PROJECT_ROOT / "opti_oignon"
_FRONTEND_DIR = _PROJECT_ROOT / "frontend" / "src"


def _load_module(name: str, path: Path):
    """Load a module by file path without triggering __init__ chain."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _py_files(root: Path = _BACKEND_DIR) -> list[Path]:
    """Collect all Python files under a directory."""
    return [
        f for f in root.rglob("*.py")
        if "__pycache__" not in str(f)
    ]


def _svelte_files() -> list[Path]:
    """Collect all Svelte files."""
    if not _FRONTEND_DIR.exists():
        return []
    return [
        f for f in _FRONTEND_DIR.rglob("*.svelte")
        if "node_modules" not in str(f)
    ]


# ============================================================================
# Goal 1: No raw sqlite3.connect() in core modules
# ============================================================================

class TestNoRawSqliteConnect(unittest.TestCase):
    """Ensure all core modules use safe_connect() instead of sqlite3.connect()."""

    _SQLITE_RE = re.compile(r"\bsqlite3\.connect\s*\(")
    _ALLOWED_FILES = {"db_utils.py", "db_encryption.py"}

    def test_no_raw_sqlite3_connect_in_core(self):
        """No raw sqlite3.connect() outside db_utils/db_encryption."""
        violations = []
        for fpath in _py_files():
            if fpath.name in self._ALLOWED_FILES:
                continue
            if "/plugins/" in str(fpath) or os.sep + "plugins" + os.sep in str(fpath):
                continue
            text = fpath.read_text(encoding="utf-8", errors="replace")
            for i, line in enumerate(text.splitlines(), 1):
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                if self._SQLITE_RE.search(line):
                    rel = fpath.relative_to(_PROJECT_ROOT)
                    violations.append(f"  {rel}:{i} -> {stripped[:80]}")
        self.assertEqual(
            violations, [],
            "Found raw sqlite3.connect() calls:\n" + "\n".join(violations),
        )

    def test_conversation_uses_safe_connect(self):
        """conversation.py imports safe_connect."""
        text = (_BACKEND_DIR / "conversation.py").read_text()
        self.assertIn("from opti_oignon.db_utils import safe_connect", text)
        # Should not import get_encrypted_connection directly
        self.assertNotIn("from opti_oignon.db_encryption import get_encrypted_connection", text)

    def test_memory_uses_safe_connect(self):
        """memory.py imports safe_connect."""
        text = (_BACKEND_DIR / "memory.py").read_text()
        self.assertIn("from opti_oignon.db_utils import safe_connect", text)

    def test_auth_uses_safe_connect(self):
        """auth.py imports safe_connect."""
        text = (_BACKEND_DIR / "auth.py").read_text()
        self.assertIn("from opti_oignon.db_utils import safe_connect", text)

    def test_auth_2fa_uses_safe_connect(self):
        """auth_2fa.py imports safe_connect."""
        text = (_BACKEND_DIR / "auth_2fa.py").read_text()
        self.assertIn("from opti_oignon.db_utils import safe_connect", text)

    def test_conversation_branches_uses_safe_connect(self):
        """conversation_branches.py imports safe_connect."""
        text = (_BACKEND_DIR / "conversation_branches.py").read_text()
        self.assertIn("from opti_oignon.db_utils import safe_connect", text)

    def test_signed_audit_log_uses_safe_connect(self):
        """signed_audit_log.py imports safe_connect."""
        text = (_BACKEND_DIR / "signed_audit_log.py").read_text()
        self.assertIn("from opti_oignon.db_utils import safe_connect", text)

    def test_sandbox_manager_uses_safe_connect(self):
        """sandbox_manager.py imports safe_connect."""
        text = (_BACKEND_DIR / "sandbox_manager.py").read_text()
        self.assertIn("from opti_oignon.db_utils import safe_connect", text)

    def test_coding_history_uses_safe_connect(self):
        """coding_history.py uses _safe_connect."""
        text = (_BACKEND_DIR / "coding_history.py").read_text()
        self.assertIn("_safe_connect", text)
        self.assertNotIn("sqlite3.connect(self._db_path", text)

    def test_performance_monitor_uses_safe_connect(self):
        """performance_monitor.py uses _safe_connect."""
        text = (_BACKEND_DIR / "performance_monitor.py").read_text()
        self.assertIn("_safe_connect", text)
        self.assertNotIn("sqlite3.connect(self._db_path", text)

    def test_telemetry_history_uses_safe_connect(self):
        """telemetry_history.py uses _safe_connect."""
        text = (_BACKEND_DIR / "telemetry_history.py").read_text()
        self.assertIn("_safe_connect", text)
        self.assertNotIn("sqlite3.connect(self._db_path", text)


# ============================================================================
# Goal 1b: safe_connect() accepts required kwargs
# ============================================================================

class TestSafeConnectSignature(unittest.TestCase):
    """Verify safe_connect() supports all kwargs used across the codebase."""

    def test_safe_connect_exists(self):
        """db_utils.safe_connect is importable."""
        mod = _load_module("db_utils", _BACKEND_DIR / "db_utils.py")
        self.assertTrue(hasattr(mod, "safe_connect"))

    def test_safe_connect_accepts_timeout(self):
        """safe_connect() has a timeout parameter."""
        import inspect
        mod = _load_module("db_utils", _BACKEND_DIR / "db_utils.py")
        sig = inspect.signature(mod.safe_connect)
        self.assertIn("timeout", sig.parameters)

    def test_safe_connect_accepts_check_same_thread(self):
        """safe_connect() has a check_same_thread parameter."""
        import inspect
        mod = _load_module("db_utils", _BACKEND_DIR / "db_utils.py")
        sig = inspect.signature(mod.safe_connect)
        self.assertIn("check_same_thread", sig.parameters)

    def test_safe_connect_returns_connection(self):
        """safe_connect() returns a sqlite3.Connection."""
        import sqlite3
        import tempfile
        mod = _load_module("db_utils", _BACKEND_DIR / "db_utils.py")
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            conn = mod.safe_connect(tmp.name, timeout=5.0, check_same_thread=False)
            self.assertIsInstance(conn, sqlite3.Connection)
            conn.close()


# ============================================================================
# Goal 2: No f-string SQL in core modules
# ============================================================================

class TestNoFstringSql(unittest.TestCase):
    """Ensure no f-string SQL literals remain in core production code."""

    # Matches actual SQL statements in f-strings (not logger messages)
    _FSTRING_SQL_RE = re.compile(
        r"""\bf(["'])"""
        r"""(?=.*\b("""
        r"""SELECT\s+.+?\s+FROM"""
        r"""|INSERT\s+INTO"""
        r"""|UPDATE\s+\w+\s+SET"""
        r"""|DELETE\s+FROM"""
        r"""|ALTER\s+TABLE"""
        r"""|CREATE\s+(TABLE|INDEX)"""
        r""")\b)""",
        re.IGNORECASE,
    )

    _SKIP_FILES = {"security_scan.py"}

    def test_no_fstring_sql_in_core(self):
        """No f-string SQL literals in production Python files."""
        violations = []
        for fpath in _py_files():
            if fpath.name.startswith("test_"):
                continue
            if fpath.name in self._SKIP_FILES:
                continue
            if "/plugins/" in str(fpath):
                continue
            for i, line in enumerate(fpath.read_text(errors="replace").splitlines(), 1):
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                if self._FSTRING_SQL_RE.search(line):
                    rel = fpath.relative_to(_PROJECT_ROOT)
                    violations.append(f"  {rel}:{i} -> {stripped[:80]}")
        self.assertEqual(
            violations, [],
            "Found f-string SQL:\n" + "\n".join(violations),
        )

    def test_migrated_files_use_format(self):
        """Key migrated files use .format() for dynamic SQL, not f-strings."""
        files_to_check = [
            "conversation.py", "conversation_branches.py", "auth.py",
            "memory.py", "fine_tune_tracker.py", "benchmark_runner.py",
            "analytics.py", "performance_monitor.py", "user_isolation.py",
            "plugin_index.py", "plugin_reviews.py", "semantic_cache.py",
            "signed_audit_log.py",
        ]
        for fname in files_to_check:
            text = (_BACKEND_DIR / fname).read_text()
            matches = self._FSTRING_SQL_RE.findall(text)
            self.assertEqual(
                len(matches), 0,
                f"{fname} still contains f-string SQL: {matches}",
            )


# ============================================================================
# Goal 2b: Allowlist constants exist
# ============================================================================

class TestAllowlistConstants(unittest.TestCase):
    """Verify allowlist frozensets exist in hardened modules."""

    def test_conversation_has_update_cols(self):
        text = (_BACKEND_DIR / "conversation.py").read_text()
        self.assertIn("_CONV_UPDATE_COLS", text)
        self.assertIn("frozenset", text.split("_CONV_UPDATE_COLS")[1][:50])

    def test_conversation_branches_has_update_cols(self):
        text = (_BACKEND_DIR / "conversation_branches.py").read_text()
        self.assertIn("_BRANCH_UPDATE_COLS", text)
        self.assertIn("frozenset", text.split("_BRANCH_UPDATE_COLS")[1][:50])

    def test_user_isolation_has_allowed_tables(self):
        text = (_BACKEND_DIR / "user_isolation.py").read_text()
        self.assertIn("_ALLOWED_TABLES", text)
        self.assertIn("frozenset", text.split("_ALLOWED_TABLES")[1][:50])

    def test_performance_monitor_has_metric_cols(self):
        text = (_BACKEND_DIR / "performance_monitor.py").read_text()
        self.assertIn("_METRIC_COLS", text)
        self.assertIn("frozenset", text.split("_METRIC_COLS")[1][:80])


# ============================================================================
# Goal 3: No bare hex colors outside var() in Svelte
# ============================================================================

class TestNoBareHexColors(unittest.TestCase):
    """No hardcoded #fff/#000 outside var(--oo-*, fallback) in Svelte."""

    _HEX_RE = re.compile(r"#[0-9a-fA-F]{3,8}\b")
    _FALLBACK_RE = re.compile(r"var\(--oo-[\w-]+,\s*#[0-9a-fA-F]{3,8}\)")
    _HTML_ENTITY_RE = re.compile(r"&#[0-9a-fA-F]{1,8};")

    def test_no_bare_hex_in_svelte(self):
        """No bare hex colors in any Svelte file."""
        violations = []
        for fpath in _svelte_files():
            for i, line in enumerate(fpath.read_text(errors="replace").splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("<!--") or stripped.startswith("//"):
                    continue
                if not self._HEX_RE.search(line):
                    continue
                cleaned = self._FALLBACK_RE.sub("", line)
                cleaned = self._HTML_ENTITY_RE.sub("", cleaned)
                remaining = self._HEX_RE.findall(cleaned)
                if remaining:
                    rel = fpath.relative_to(_PROJECT_ROOT)
                    violations.append(f"  {rel}:{i} -> {remaining} in: {stripped[:80]}")
        self.assertEqual(
            violations, [],
            "Found bare hex colors:\n" + "\n".join(violations),
        )

    def test_totp_setup_uses_css_var(self):
        """TOTPSetup.svelte uses var(--oo-fg-on-accent) not bare #fff."""
        text = (_FRONTEND_DIR / "lib/components/settings/TOTPSetup.svelte").read_text()
        # The error button line should now use the CSS variable
        self.assertNotIn("color: #fff", text)

    def test_webauthn_setup_uses_css_var(self):
        """WebAuthnSetup.svelte uses var(--oo-fg-on-accent) not bare #fff."""
        text = (_FRONTEND_DIR / "lib/components/settings/WebAuthnSetup.svelte").read_text()
        self.assertNotIn("color: #fff", text)


# ============================================================================
# Goal 4: security_scan.py checks pass
# ============================================================================

class TestSecurityScanChecks(unittest.TestCase):
    """Verify security_scan.py checks pass for S138 targets."""

    def test_security_scan_has_no_fstring_sql_check(self):
        """security_scan.py includes check_no_fstring_sql."""
        text = (_PROJECT_ROOT / "scripts" / "security_scan.py").read_text()
        self.assertIn("def check_no_fstring_sql", text)
        self.assertIn("check_no_fstring_sql(py_files)", text)

    def test_security_scan_tightened_sqlite_check(self):
        """security_scan.py sqlite allowed list is db_utils + db_encryption only."""
        text = (_PROJECT_ROOT / "scripts" / "security_scan.py").read_text()
        # Should allow only db_utils.py and db_encryption.py
        self.assertIn('"db_utils.py"', text)
        self.assertIn('"db_encryption.py"', text)
        # Should NOT have the old signed_audit_log bypass
        self.assertNotIn('"signed_audit_log.py"', text.split("_SQLITE_ALLOWED")[1][:200])


# ============================================================================
# Goal 5: Version bump
# ============================================================================

class TestVersionBump(unittest.TestCase):
    """Version must be at least 3.0.1."""

    def test_version_file(self):
        mod = _load_module("__version__", _BACKEND_DIR / "__version__.py")
        self.assertGreaterEqual(mod.__version__, "3.0.1")


# ============================================================================
# AST validation on all modified files
# ============================================================================

class TestASTValidation(unittest.TestCase):
    """All modified Python files must parse without errors."""

    _MODIFIED_FILES = [
        "conversation.py",
        "memory.py",
        "auth.py",
        "auth_2fa.py",
        "coding_history.py",
        "conversation_branches.py",
        "performance_monitor.py",
        "telemetry_history.py",
        "signed_audit_log.py",
        "sandbox_manager.py",
        "fine_tune_tracker.py",
        "benchmark_runner.py",
        "analytics.py",
        "user_isolation.py",
        "plugin_index.py",
        "plugin_reviews.py",
        "semantic_cache.py",
        "db_utils.py",
    ]

    def test_ast_parse_all_modified(self):
        """Every modified file passes AST parsing."""
        failures = []
        for fname in self._MODIFIED_FILES:
            fpath = _BACKEND_DIR / fname
            if not fpath.exists():
                failures.append(f"{fname}: file not found")
                continue
            try:
                ast.parse(fpath.read_text())
            except SyntaxError as e:
                failures.append(f"{fname}: {e}")
        self.assertEqual(failures, [], "\n".join(failures))

    def test_ast_parse_security_scan(self):
        """security_scan.py passes AST parsing."""
        fpath = _PROJECT_ROOT / "scripts" / "security_scan.py"
        ast.parse(fpath.read_text())  # raises on failure


if __name__ == "__main__":
    unittest.main()
