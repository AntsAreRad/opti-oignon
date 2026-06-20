"""
Tests for S130 -- Hash-Chain Signed Audit Log + Static Security Scanner.

Validates:
- Part 1: signed_audit_log.py (append, verify, tamper detection, query, export, genesis)
- Part 2: security_scan.py (each check has positive + negative tests)
- Part 3: API routes (audit-chain endpoints exist and return correct structure)
- Part 4: Integration (chain_log forwarding in security_mode, auth, tool_call_approval)
- Part 5: Frontend files (auditChain.ts, AuditChainPanel.svelte, SecurityPanel.svelte)
- Part 6: Version bump 2.9.0 -> 2.9.1
- Zero regressions

Target: ~65 tests
"""

import ast
import csv
import importlib.util
import io
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
from unittest.mock import MagicMock, patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "opti_oignon")
FRONTEND_SRC = os.path.join(PROJECT_ROOT, "frontend", "src")
COMPONENTS_DIR = os.path.join(FRONTEND_SRC, "lib", "components", "settings")
API_TS_DIR = os.path.join(FRONTEND_SRC, "lib", "api")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
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
    # Provide db_encryption stub
    if "opti_oignon.db_encryption" not in sys.modules:
        dbe = types.ModuleType("opti_oignon.db_encryption")
        def _fake_conn(db_path, **kw):
            return sqlite3.connect(str(db_path), check_same_thread=False)
        dbe.get_encrypted_connection = _fake_conn
        dbe.SQLCIPHER_AVAILABLE = False
        sys.modules["opti_oignon.db_encryption"] = dbe
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _get_audit_log(db_path=None):
    """Create a fresh SignedAuditLog pointing to a temp DB."""
    mod = _load_module(
        "opti_oignon.signed_audit_log",
        os.path.join(BACKEND_DIR, "signed_audit_log.py"),
    )
    if db_path is None:
        db_path = os.path.join(tempfile.mkdtemp(), "test_audit.db")
    return mod.SignedAuditLog(db_path=db_path), mod, db_path


# =========================================================================
# Part 1: signed_audit_log.py
# =========================================================================

class TestSignedAuditLogAST(unittest.TestCase):
    """AST and structural validation."""

    def test_ast_valid(self):
        """signed_audit_log.py passes AST validation."""
        path = os.path.join(BACKEND_DIR, "signed_audit_log.py")
        with open(path) as f:
            tree = ast.parse(f.read())
        self.assertIsNotNone(tree)

    def test_has_required_classes_and_functions(self):
        """Module exposes required API surface."""
        path = os.path.join(BACKEND_DIR, "signed_audit_log.py")
        with open(path) as f:
            tree = ast.parse(f.read())
        names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        }
        for required in [
            "SignedAuditLog", "AuditEntry",
            "append_event", "verify_chain", "get_events",
            "export_chain_csv", "chain_log", "_compute_entry_hash",
        ]:
            self.assertIn(required, names, f"Missing: {required}")

    def test_feature_flag_exists(self):
        """SIGNED_AUDIT_AVAILABLE feature flag defined."""
        path = os.path.join(BACKEND_DIR, "signed_audit_log.py")
        content = open(path).read()
        self.assertIn("SIGNED_AUDIT_AVAILABLE", content)


class TestSignedAuditLogGenesis(unittest.TestCase):
    """Genesis entry and empty-chain behavior."""

    def test_empty_chain_is_valid(self):
        """An empty chain should verify as valid with 0 entries."""
        log, mod, _ = _get_audit_log()
        valid, broken, total = log.verify_chain()
        self.assertTrue(valid)
        self.assertIsNone(broken)
        self.assertEqual(total, 0)

    def test_first_entry_uses_genesis_prev_hash(self):
        """First entry should have prev_hash = '0' * 128."""
        log, mod, db_path = _get_audit_log()
        log.append_event("test", source="test", action="init")
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT prev_hash FROM audit_chain WHERE id=1"
        ).fetchone()
        conn.close()
        self.assertEqual(row[0], "0" * 128)

    def test_entry_count_starts_at_zero(self):
        """Fresh chain has 0 entries."""
        log, _, _ = _get_audit_log()
        self.assertEqual(log.entry_count(), 0)


class TestSignedAuditLogAppend(unittest.TestCase):
    """Append operations."""

    def test_append_returns_sequential_ids(self):
        """IDs increment from 1."""
        log, _, _ = _get_audit_log()
        id1 = log.append_event("evt1")
        id2 = log.append_event("evt2")
        id3 = log.append_event("evt3")
        self.assertEqual(id1, 1)
        self.assertEqual(id2, 2)
        self.assertEqual(id3, 3)

    def test_append_stores_all_fields(self):
        """All fields are correctly stored."""
        log, _, db_path = _get_audit_log()
        log.append_event(
            "login_success",
            source="auth",
            action="user logged in",
            severity="INFO",
            details={"user": "alice"},
        )
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT event_type, source, action, severity, details_json "
            "FROM audit_chain WHERE id=1"
        ).fetchone()
        conn.close()
        self.assertEqual(row[0], "login_success")
        self.assertEqual(row[1], "auth")
        self.assertEqual(row[2], "user logged in")
        self.assertEqual(row[3], "INFO")
        details = json.loads(row[4])
        self.assertEqual(details["user"], "alice")

    def test_hash_chain_links(self):
        """Each entry's prev_hash matches the previous entry's entry_hash."""
        log, _, db_path = _get_audit_log()
        log.append_event("a")
        log.append_event("b")
        log.append_event("c")
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT id, prev_hash, entry_hash FROM audit_chain ORDER BY id"
        ).fetchall()
        conn.close()
        self.assertEqual(rows[0][1], "0" * 128)  # genesis
        self.assertEqual(rows[1][1], rows[0][2])  # chain link
        self.assertEqual(rows[2][1], rows[1][2])  # chain link

    def test_entry_count_after_append(self):
        """entry_count() returns correct count."""
        log, _, _ = _get_audit_log()
        log.append_event("a")
        log.append_event("b")
        self.assertEqual(log.entry_count(), 2)


class TestSignedAuditLogVerify(unittest.TestCase):
    """Chain verification and tamper detection."""

    def test_valid_chain_passes(self):
        """Unmodified chain should verify successfully."""
        log, _, _ = _get_audit_log()
        for i in range(5):
            log.append_event(f"event_{i}")
        valid, broken, total = log.verify_chain()
        self.assertTrue(valid)
        self.assertIsNone(broken)
        self.assertEqual(total, 5)

    def test_tampered_action_detected(self):
        """Modifying an action field breaks the chain."""
        log, _, db_path = _get_audit_log()
        log.append_event("original")
        log.append_event("second")
        conn = sqlite3.connect(db_path)
        conn.execute("UPDATE audit_chain SET action='hacked' WHERE id=1")
        conn.commit()
        conn.close()
        valid, broken, total = log.verify_chain()
        self.assertFalse(valid)
        self.assertEqual(broken, 1)

    def test_tampered_hash_detected(self):
        """Modifying entry_hash directly is detected."""
        log, _, db_path = _get_audit_log()
        log.append_event("test")
        conn = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE audit_chain SET entry_hash='bad' WHERE id=1"
        )
        conn.commit()
        conn.close()
        valid, broken, _ = log.verify_chain()
        self.assertFalse(valid)
        self.assertEqual(broken, 1)

    def test_tampered_prev_hash_detected(self):
        """Modifying prev_hash breaks the chain at that entry."""
        log, _, db_path = _get_audit_log()
        log.append_event("a")
        log.append_event("b")
        conn = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE audit_chain SET prev_hash='wrong' WHERE id=2"
        )
        conn.commit()
        conn.close()
        valid, broken, _ = log.verify_chain()
        self.assertFalse(valid)
        self.assertEqual(broken, 2)

    def test_deleted_entry_detected(self):
        """Deleting an entry breaks the chain."""
        log, _, db_path = _get_audit_log()
        log.append_event("a")
        log.append_event("b")
        log.append_event("c")
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM audit_chain WHERE id=2")
        conn.commit()
        conn.close()
        valid, broken, _ = log.verify_chain()
        self.assertFalse(valid)
        # Entry 3's prev_hash won't match entry 1's hash
        self.assertEqual(broken, 3)


class TestSignedAuditLogQuery(unittest.TestCase):
    """Event querying and filtering."""

    def test_get_events_default(self):
        """get_events returns recent events in descending order."""
        log, _, _ = _get_audit_log()
        log.append_event("first")
        log.append_event("second")
        events = log.get_events(limit=10)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["event_type"], "second")  # DESC
        self.assertEqual(events[1]["event_type"], "first")

    def test_filter_by_event_type(self):
        """Filtering by event_type returns only matching events."""
        log, _, _ = _get_audit_log()
        log.append_event("login")
        log.append_event("logout")
        log.append_event("login")
        events = log.get_events(event_type="login")
        self.assertEqual(len(events), 2)
        for e in events:
            self.assertEqual(e["event_type"], "login")

    def test_filter_by_severity(self):
        """Filtering by severity works."""
        log, _, _ = _get_audit_log()
        log.append_event("a", severity="INFO")
        log.append_event("b", severity="WARNING")
        log.append_event("c", severity="CRITICAL")
        events = log.get_events(severity="WARNING")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["severity"], "WARNING")

    def test_pagination(self):
        """Offset and limit paginate correctly."""
        log, _, _ = _get_audit_log()
        for i in range(10):
            log.append_event(f"evt_{i}")
        page1 = log.get_events(limit=3, offset=0)
        page2 = log.get_events(limit=3, offset=3)
        self.assertEqual(len(page1), 3)
        self.assertEqual(len(page2), 3)
        # No overlap (DESC order so page1 has highest ids)
        ids1 = {e["id"] for e in page1}
        ids2 = {e["id"] for e in page2}
        self.assertEqual(len(ids1 & ids2), 0)

    def test_event_details_parsed(self):
        """Details are returned as parsed dict."""
        log, _, _ = _get_audit_log()
        log.append_event("test", details={"key": "value", "num": 42})
        events = log.get_events()
        self.assertEqual(events[0]["details"]["key"], "value")
        self.assertEqual(events[0]["details"]["num"], 42)


class TestSignedAuditLogExport(unittest.TestCase):
    """CSV export."""

    def test_export_csv_header(self):
        """CSV export includes correct header row."""
        log, _, _ = _get_audit_log()
        log.append_event("test")
        csv_text = log.export_chain_csv()
        reader = csv.reader(io.StringIO(csv_text))
        header = next(reader)
        expected = [
            "id", "timestamp", "event_type", "source", "action",
            "severity", "details_json", "prev_hash", "entry_hash",
        ]
        self.assertEqual(header, expected)

    def test_export_csv_data_rows(self):
        """CSV export contains all entries."""
        log, _, _ = _get_audit_log()
        log.append_event("a")
        log.append_event("b")
        csv_text = log.export_chain_csv()
        reader = csv.reader(io.StringIO(csv_text))
        rows = list(reader)
        self.assertEqual(len(rows), 3)  # header + 2 data rows


class TestSignedAuditLogStatus(unittest.TestCase):
    """Status endpoint data."""

    def test_status_empty_chain(self):
        """Status for empty chain."""
        log, _, _ = _get_audit_log()
        s = log.get_status()
        self.assertEqual(s["total_entries"], 0)
        self.assertIsNone(s["last_entry"])
        self.assertTrue(s["chain_valid"])

    def test_status_with_entries(self):
        """Status shows last entry and total."""
        log, _, _ = _get_audit_log()
        log.append_event("a")
        log.append_event("b")
        s = log.get_status()
        self.assertEqual(s["total_entries"], 2)
        self.assertIsNotNone(s["last_entry"])
        self.assertEqual(s["last_entry"]["id"], 2)
        self.assertTrue(s["chain_valid"])


class TestChainLogConvenience(unittest.TestCase):
    """chain_log() convenience wrapper."""

    def test_chain_log_returns_id(self):
        """chain_log() returns entry id when chain is available."""
        log, mod, _ = _get_audit_log()
        # Replace module-level singleton
        mod.signed_audit_log = log
        result = mod.chain_log("test_event", source="test")
        self.assertEqual(result, 1)

    def test_chain_log_none_when_unavailable(self):
        """chain_log() returns None when chain is None."""
        _, mod, _ = _get_audit_log()
        mod.signed_audit_log = None
        result = mod.chain_log("test_event")
        self.assertIsNone(result)


# =========================================================================
# Part 2: security_scan.py
# =========================================================================

class TestSecurityScanAST(unittest.TestCase):
    """AST validation for security_scan.py."""

    def test_ast_valid(self):
        """security_scan.py passes AST validation."""
        path = os.path.join(SCRIPTS_DIR, "security_scan.py")
        with open(path) as f:
            tree = ast.parse(f.read())
        self.assertIsNotNone(tree)

    def test_has_all_check_functions(self):
        """All 10 check functions exist."""
        path = os.path.join(SCRIPTS_DIR, "security_scan.py")
        with open(path) as f:
            tree = ast.parse(f.read())
        func_names = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        expected = [
            "check_no_raw_sqlite",
            "check_no_eval_exec",
            "check_no_hardcoded_secrets",
            "check_no_pickle",
            "check_sql_parameterized",
            "check_no_shell_true",
            "check_csrf_protection",
            "check_no_hardcoded_colors",
            "check_checkpoint_hardcoded",
            "check_no_french",
        ]
        for fn in expected:
            self.assertIn(fn, func_names, f"Missing check: {fn}")

    def test_has_run_all_checks(self):
        """run_all_checks() and main() exist."""
        path = os.path.join(SCRIPTS_DIR, "security_scan.py")
        with open(path) as f:
            tree = ast.parse(f.read())
        func_names = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("run_all_checks", func_names)
        self.assertIn("main", func_names)


class TestSecurityScanChecks(unittest.TestCase):
    """Individual check positive/negative tests using temp files."""

    def _run_check(self, check_func, content, suffix=".py"):
        """Write content to a temp file and run a check function on it."""
        mod = _load_module(
            "security_scan",
            os.path.join(SCRIPTS_DIR, "security_scan.py"),
        )
        tmpdir = tempfile.mkdtemp()
        # Use a name without 'test_' or 'sandbox' to avoid skip patterns
        fpath = os.path.join(tmpdir, f"scan_target{suffix}")
        with open(fpath, "w") as f:
            f.write(content)
        # Monkey-patch PROJECT_ROOT for relative path computation
        mod._PROJECT_ROOT = Path(tmpdir)
        func = getattr(mod, check_func)
        return func([Path(fpath)])

    def test_raw_sqlite_positive(self):
        """Detects direct sqlite3.connect()."""
        result = self._run_check(
            "check_no_raw_sqlite",
            "import sqlite3\nconn = sqlite3.connect('test.db')\n",
        )
        self.assertFalse(result.passed)

    def test_raw_sqlite_negative(self):
        """Clean code passes."""
        result = self._run_check(
            "check_no_raw_sqlite",
            "from db_encryption import get_encrypted_connection\nconn = get_encrypted_connection('test.db')\n",
        )
        self.assertTrue(result.passed)

    def test_eval_positive(self):
        """Detects eval()."""
        result = self._run_check(
            "check_no_eval_exec",
            "result = eval(user_input)\n",
        )
        self.assertFalse(result.passed)

    def test_eval_negative(self):
        """Code without eval passes."""
        result = self._run_check(
            "check_no_eval_exec",
            "result = int(user_input)\n",
        )
        self.assertTrue(result.passed)

    def test_hardcoded_secret_positive(self):
        """Detects hardcoded password."""
        result = self._run_check(
            "check_no_hardcoded_secrets",
            'db_password = "SuperSecretP@ss123"\n',
        )
        self.assertFalse(result.passed)

    def test_hardcoded_secret_negative(self):
        """Environment variable usage passes."""
        result = self._run_check(
            "check_no_hardcoded_secrets",
            'password = os.environ.get("DB_PASSWORD")\n',
        )
        self.assertTrue(result.passed)

    def test_pickle_positive(self):
        """Detects pickle.loads()."""
        result = self._run_check(
            "check_no_pickle",
            "import pickle\ndata = pickle.loads(raw_bytes)\n",
        )
        self.assertFalse(result.passed)

    def test_pickle_negative(self):
        """Code without pickle passes."""
        result = self._run_check(
            "check_no_pickle",
            "import json\ndata = json.loads(raw_text)\n",
        )
        self.assertTrue(result.passed)

    def test_sql_fstring_positive(self):
        """Detects f-string in .execute()."""
        result = self._run_check(
            "check_sql_parameterized",
            'conn.execute(f"SELECT * FROM users WHERE id={uid}")\n',
        )
        self.assertFalse(result.passed)

    def test_sql_parameterized_negative(self):
        """Parameterized query passes."""
        result = self._run_check(
            "check_sql_parameterized",
            'conn.execute("SELECT * FROM users WHERE id=?", (uid,))\n',
        )
        self.assertTrue(result.passed)

    def test_shell_true_positive(self):
        """Detects shell=True."""
        result = self._run_check(
            "check_no_shell_true",
            'subprocess.run(cmd, shell=True)\n',
        )
        self.assertFalse(result.passed)

    def test_shell_true_negative(self):
        """subprocess without shell=True passes."""
        result = self._run_check(
            "check_no_shell_true",
            'subprocess.run(["ls", "-la"])\n',
        )
        self.assertTrue(result.passed)

    def test_checkpoint_positive(self):
        """Detects checkpoint_before_apply = False."""
        result = self._run_check(
            "check_checkpoint_hardcoded",
            "checkpoint_before_apply = False\n",
        )
        self.assertFalse(result.passed)

    def test_checkpoint_negative(self):
        """checkpoint_before_apply = True passes."""
        result = self._run_check(
            "check_checkpoint_hardcoded",
            "checkpoint_before_apply = True\n",
        )
        self.assertTrue(result.passed)


class TestSecurityScanJsonOutput(unittest.TestCase):
    """JSON report format."""

    def test_json_report_structure(self):
        """run_all_checks() returns correct structure."""
        mod = _load_module(
            "security_scan",
            os.path.join(SCRIPTS_DIR, "security_scan.py"),
        )
        report = mod.run_all_checks()
        self.assertIn("checks", report)
        self.assertIn("passed", report)
        self.assertIn("failed", report)
        self.assertIn("total", report)
        self.assertIn("all_passed", report)
        self.assertEqual(report["total"], 10)
        self.assertEqual(
            report["passed"] + report["failed"],
            report["total"],
        )

    def test_each_check_has_required_fields(self):
        """Each check result has name, description, passed, violations."""
        mod = _load_module(
            "security_scan",
            os.path.join(SCRIPTS_DIR, "security_scan.py"),
        )
        report = mod.run_all_checks()
        for check in report["checks"]:
            self.assertIn("name", check)
            self.assertIn("description", check)
            self.assertIn("passed", check)
            self.assertIn("violations", check)
            self.assertIn("violation_count", check)


# =========================================================================
# Part 3: API routes
# =========================================================================

class TestAuditChainRoutes(unittest.TestCase):
    """Verify audit-chain endpoints exist in routes_security.py."""

    def test_ast_valid(self):
        """routes_security.py passes AST validation."""
        path = os.path.join(API_DIR, "routes_security.py")
        with open(path) as f:
            tree = ast.parse(f.read())
        self.assertIsNotNone(tree)

    def test_audit_chain_status_endpoint(self):
        """GET /audit-chain/status endpoint defined."""
        path = os.path.join(API_DIR, "routes_security.py")
        content = open(path).read()
        self.assertIn("/audit-chain/status", content)
        self.assertIn("audit_chain_status", content)

    def test_audit_chain_events_endpoint(self):
        """GET /audit-chain/events endpoint defined."""
        path = os.path.join(API_DIR, "routes_security.py")
        content = open(path).read()
        self.assertIn("/audit-chain/events", content)
        self.assertIn("audit_chain_events", content)

    def test_audit_chain_verify_endpoint(self):
        """POST /audit-chain/verify endpoint defined."""
        path = os.path.join(API_DIR, "routes_security.py")
        content = open(path).read()
        self.assertIn("/audit-chain/verify", content)
        self.assertIn("audit_chain_verify", content)

    def test_audit_chain_export_endpoint(self):
        """GET /audit-chain/export endpoint defined."""
        path = os.path.join(API_DIR, "routes_security.py")
        content = open(path).read()
        self.assertIn("/audit-chain/export", content)
        self.assertIn("audit_chain_export", content)

    def test_csv_response_type(self):
        """Export endpoint uses PlainTextResponse with CSV media type."""
        path = os.path.join(API_DIR, "routes_security.py")
        content = open(path).read()
        self.assertIn("text/csv", content)
        self.assertIn("PlainTextResponse", content)

    def test_query_parameters(self):
        """Events endpoint accepts filter query parameters."""
        path = os.path.join(API_DIR, "routes_security.py")
        content = open(path).read()
        for param in ["event_type", "severity", "after", "before", "limit", "offset"]:
            self.assertIn(param, content)

    def test_require_chain_helper(self):
        """_require_chain helper raises 503 when unavailable."""
        path = os.path.join(API_DIR, "routes_security.py")
        content = open(path).read()
        self.assertIn("_require_chain", content)
        self.assertIn("503", content)


# =========================================================================
# Part 4: Integration -- chain_log forwarding
# =========================================================================

class TestChainLogIntegration(unittest.TestCase):
    """Verify chain_log is called from security_mode, auth, tool_call_approval."""

    def test_security_mode_calls_chain_log(self):
        """security_mode._audit_log contains chain_log forwarding."""
        path = os.path.join(BACKEND_DIR, "security_mode.py")
        content = open(path).read()
        self.assertIn("from opti_oignon.signed_audit_log import chain_log", content)
        self.assertIn("chain_log(", content)

    def test_auth_calls_chain_log(self):
        """auth._log_audit contains chain_log forwarding."""
        path = os.path.join(BACKEND_DIR, "auth.py")
        content = open(path).read()
        self.assertIn("from opti_oignon.signed_audit_log import chain_log", content)
        self.assertIn("chain_log(", content)

    def test_tool_call_approval_calls_chain_log(self):
        """tool_call_approval._resolve contains chain_log forwarding."""
        path = os.path.join(BACKEND_DIR, "tool_call_approval.py")
        content = open(path).read()
        self.assertIn("from opti_oignon.signed_audit_log import chain_log", content)
        self.assertIn("chain_log(", content)

    def test_killswitch_inherits_chain(self):
        """search_killswitch.py calls _audit_log which now chains."""
        path = os.path.join(BACKEND_DIR, "search_killswitch.py")
        content = open(path).read()
        self.assertIn("from opti_oignon.security_mode import _audit_log", content)

    def test_plugin_allowlist_inherits_chain(self):
        """plugin_allowlist.py calls _audit_log which now chains."""
        path = os.path.join(BACKEND_DIR, "plugin_allowlist.py")
        content = open(path).read()
        self.assertIn("from opti_oignon.security_mode import _audit_log", content)


# =========================================================================
# Part 5: Frontend files
# =========================================================================

class TestAuditChainTS(unittest.TestCase):
    """Validate auditChain.ts API client."""

    def test_file_exists(self):
        """auditChain.ts exists."""
        path = os.path.join(API_TS_DIR, "auditChain.ts")
        self.assertTrue(os.path.isfile(path))

    def test_exports_required_functions(self):
        """All API functions are exported."""
        path = os.path.join(API_TS_DIR, "auditChain.ts")
        content = open(path).read()
        for fn in [
            "getAuditChainStatus",
            "getAuditChainEvents",
            "verifyAuditChain",
            "exportAuditChainCsv",
        ]:
            self.assertIn(f"export async function {fn}", content)

    def test_exports_required_interfaces(self):
        """TypeScript interfaces are exported."""
        path = os.path.join(API_TS_DIR, "auditChain.ts")
        content = open(path).read()
        for iface in [
            "AuditChainStatus",
            "AuditEvent",
            "AuditEventsResponse",
            "AuditChainVerifyResult",
        ]:
            self.assertIn(f"export interface {iface}", content)

    def test_imports_from_client(self):
        """Uses apiGet/apiPost from client.ts."""
        path = os.path.join(API_TS_DIR, "auditChain.ts")
        content = open(path).read()
        self.assertIn("from './client'", content)
        self.assertIn("apiGet", content)
        self.assertIn("apiPost", content)


class TestAuditChainPanelSvelte(unittest.TestCase):
    """Validate AuditChainPanel.svelte."""

    def test_file_exists(self):
        """AuditChainPanel.svelte exists."""
        path = os.path.join(COMPONENTS_DIR, "AuditChainPanel.svelte")
        self.assertTrue(os.path.isfile(path))

    def test_imports_api(self):
        """Imports from auditChain API client."""
        path = os.path.join(COMPONENTS_DIR, "AuditChainPanel.svelte")
        content = open(path).read()
        self.assertIn("from '../api/auditChain'", content)

    def test_html_balanced(self):
        """HTML tags are balanced."""
        path = os.path.join(COMPONENTS_DIR, "AuditChainPanel.svelte")
        content = open(path).read()
        # Extract template
        parts = content.split("</script>")
        template = parts[-1] if len(parts) > 1 else content
        # Remove Svelte blocks and self-closing tags
        template = re.sub(r"\{[#/:][^}]+\}", "", template)
        template = re.sub(r"<\w+[^>]*/>", "", template)
        template = re.sub(r"<!--.*?-->", "", template, flags=re.DOTALL)
        from collections import Counter
        opens = Counter(re.findall(r"<(\w+)[\s>]", template))
        closes = Counter(re.findall(r"</(\w+)>", template))
        void_tags = {"br", "hr", "input", "img", "meta", "link", "col"}
        for tag in set(list(opens.keys()) + list(closes.keys())):
            if tag in void_tags:
                continue
            self.assertEqual(
                opens[tag], closes[tag],
                f"Unbalanced <{tag}>: opened {opens[tag]}, closed {closes[tag]}",
            )

    def test_no_hardcoded_hex(self):
        """No hardcoded hex colors outside var(--oo-*) fallbacks."""
        path = os.path.join(COMPONENTS_DIR, "AuditChainPanel.svelte")
        content = open(path).read()
        cleaned = re.sub(r"var\(--oo-[\w-]+,\s*#[0-9a-fA-F]{3,8}\)", "", content)
        cleaned = re.sub(r"&#[0-9a-fA-F]{1,8};", "", cleaned)
        matches = re.findall(r"#[0-9a-fA-F]{3,8}\b", cleaned)
        self.assertEqual(matches, [], f"Hardcoded hex colors found: {matches}")

    def test_uses_oo_css_variables(self):
        """Uses --oo-* CSS variables."""
        path = os.path.join(COMPONENTS_DIR, "AuditChainPanel.svelte")
        content = open(path).read()
        oo_vars = re.findall(r"var\(--oo-[\w-]+", content)
        self.assertGreater(len(oo_vars), 5, "Should use multiple --oo-* CSS vars")

    def test_has_verify_button(self):
        """Has a Verify Chain button."""
        path = os.path.join(COMPONENTS_DIR, "AuditChainPanel.svelte")
        content = open(path).read()
        self.assertIn("Verify Chain", content)

    def test_has_export_button(self):
        """Has an Export CSV button."""
        path = os.path.join(COMPONENTS_DIR, "AuditChainPanel.svelte")
        content = open(path).read()
        self.assertIn("Export CSV", content)


class TestSecurityPanelIntegration(unittest.TestCase):
    """SecurityPanel.svelte includes Audit Log tab."""

    def test_imports_audit_chain_panel(self):
        """SecurityPanel imports AuditChainPanel."""
        path = os.path.join(COMPONENTS_DIR, "SecurityPanel.svelte")
        content = open(path).read()
        self.assertIn("import AuditChainPanel", content)

    def test_audit_log_tab_exists(self):
        """Audit Log tab defined in tab list."""
        path = os.path.join(COMPONENTS_DIR, "SecurityPanel.svelte")
        content = open(path).read()
        self.assertIn("auditlog", content)
        self.assertIn("Audit Log", content)

    def test_renders_audit_chain_panel(self):
        """AuditChainPanel is rendered conditionally."""
        path = os.path.join(COMPONENTS_DIR, "SecurityPanel.svelte")
        content = open(path).read()
        self.assertIn("<AuditChainPanel", content)

    def test_html_balanced(self):
        """SecurityPanel.svelte HTML tags remain balanced."""
        path = os.path.join(COMPONENTS_DIR, "SecurityPanel.svelte")
        content = open(path).read()
        parts = content.split("</script>")
        template = parts[-1] if len(parts) > 1 else content
        template = re.sub(r"\{[#/:][^}]+\}", "", template)
        template = re.sub(r"<\w+[^>]*/>", "", template)
        template = re.sub(r"<!--.*?-->", "", template, flags=re.DOTALL)
        from collections import Counter
        opens = Counter(re.findall(r"<(\w+)[\s>]", template))
        closes = Counter(re.findall(r"</(\w+)>", template))
        void_tags = {"br", "hr", "input", "img", "meta", "link", "col"}
        for tag in set(list(opens.keys()) + list(closes.keys())):
            if tag in void_tags:
                continue
            self.assertEqual(
                opens[tag], closes[tag],
                f"Unbalanced <{tag}>: opened {opens[tag]}, closed {closes[tag]}",
            )


# =========================================================================
# Part 6: Version bump
# =========================================================================

class TestVersionBump(unittest.TestCase):
    """Version should be 2.9.1."""

    def test_version_file(self):
        """__version__.py has 2.9.1."""
        path = os.path.join(BACKEND_DIR, "__version__.py")
        content = open(path).read()
        self.assertIn('"2.9.3"', content)


# =========================================================================
# Part 7: No French in new S130 files
# =========================================================================

class TestNoFrenchInS130Files(unittest.TestCase):
    """New S130 files must have English-only comments."""

    def _check_no_french(self, path):
        french_re = re.compile(
            r"\b(parametre|sauvegarde|suppression|utilisateur|recherche|bienvenue)\b",
            re.IGNORECASE,
        )
        lines = open(path).readlines()
        violations = []
        for i, line in enumerate(lines, 1):
            # Skip regex pattern definitions (the scanner itself defines French words)
            if "re.compile" in line or "r\"\"\"" in line or "r'''" in line:
                continue
            # Skip lines inside a multi-line regex (heuristic: heavy pipe usage)
            stripped = line.strip()
            if stripped.startswith("|") or stripped.endswith("|"):
                continue
            # Skip lines that are part of the French detection regex
            if "FRENCH_WORDS" in line:
                continue
            matches = french_re.findall(line)
            violations.extend(matches)
        self.assertEqual(
            violations, [],
            f"French words found in {path}: {violations}",
        )

    def test_signed_audit_log_english(self):
        self._check_no_french(os.path.join(BACKEND_DIR, "signed_audit_log.py"))

    def test_security_scan_english(self):
        self._check_no_french(os.path.join(SCRIPTS_DIR, "security_scan.py"))


if __name__ == "__main__":
    unittest.main()
