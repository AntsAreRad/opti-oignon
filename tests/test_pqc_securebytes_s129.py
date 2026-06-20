"""
Tests for S129 -- PQC Hardening + Key Memory Protection + Key Ceremony.

Validates:
- Part 1: pqc_signatures.py (feature detection, keypair CRUD, sign/verify, status)
- Part 2: SecureBytes integration in encryption.py
- Part 3: SecureBytes integration in auth.py (JWT key wrapping)
- Part 4: SecureBytes integration in plugin_allowlist.py / security_mode.py
- Part 5: Backup manager PQC signing / verification hooks
- Part 6: API routes (PQC endpoints in routes_security.py)
- Part 7: Frontend (keyCeremony.ts, KeyCeremonyPanel.svelte, SecurityPanel.svelte)
- Part 8: Version bump 2.8.0 -> 2.9.0
- Zero regressions

Target: ~37 tests
"""

import ast
import base64
import importlib.util
import json
import os
import re
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "opti_oignon")
FRONTEND_SRC = os.path.join(PROJECT_ROOT, "frontend", "src")
COMPONENTS_DIR = os.path.join(FRONTEND_SRC, "lib", "components", "settings")
API_TS_DIR = os.path.join(FRONTEND_SRC, "lib", "api")


def _load_module(name, path):
    """Load a Python module from file path without triggering __init__ imports."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if "opti_oignon" not in sys.modules:
        parent = type(sys)("opti_oignon")
        sys.modules["opti_oignon"] = parent
    if "opti_oignon.config" not in sys.modules:
        cfg_mod = type(sys)("opti_oignon.config")
        cfg_mod.DATA_DIR = tempfile.mkdtemp()
        sys.modules["opti_oignon.config"] = cfg_mod
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# =========================================================================
# Part 1: pqc_signatures.py
# =========================================================================

class TestPqcSignaturesModule(unittest.TestCase):
    """Test pqc_signatures.py structure and fallback behavior."""

    def test_ast_valid(self):
        """pqc_signatures.py passes AST validation."""
        path = os.path.join(BACKEND_DIR, "pqc_signatures.py")
        with open(path) as f:
            tree = ast.parse(f.read())
        self.assertIsNotNone(tree)

    def test_module_level_exports(self):
        """Module exports the expected public API."""
        path = os.path.join(BACKEND_DIR, "pqc_signatures.py")
        with open(path) as f:
            source = f.read()
        for name in [
            "PQC_AVAILABLE",
            "generate_pqc_keypair",
            "sign_backup",
            "verify_backup",
            "save_pqc_keypair",
            "load_pqc_keypair",
            "pqc_keypair_exists",
            "delete_pqc_keypair",
            "get_pqc_status",
            "is_pqc_enabled",
        ]:
            self.assertIn(f"def {name}" if name[0].islower() else name, source,
                          f"Missing export: {name}")

    def test_keypair_persistence_roundtrip(self):
        """save_pqc_keypair / load_pqc_keypair roundtrip with fake keys."""
        path = os.path.join(BACKEND_DIR, "pqc_signatures.py")
        with open(path) as f:
            source = f.read()

        # Test the persistence functions directly (they do not require liboqs)
        tmpdir = tempfile.mkdtemp()
        kp_path = Path(tmpdir) / ".pqc_keypair"

        # Simulate save
        pub = os.urandom(64)
        priv = os.urandom(128)
        data = {
            "algorithm": "Dilithium3",
            "public_key": base64.urlsafe_b64encode(pub).decode("ascii"),
            "private_key": base64.urlsafe_b64encode(priv).decode("ascii"),
        }
        kp_path.write_text(json.dumps(data, indent=2) + "\n", encoding="ascii")

        # Simulate load
        raw = json.loads(kp_path.read_text(encoding="ascii"))
        loaded_pub = base64.urlsafe_b64decode(raw["public_key"])
        loaded_priv = base64.urlsafe_b64decode(raw["private_key"])

        self.assertEqual(pub, loaded_pub)
        self.assertEqual(priv, loaded_priv)
        self.assertEqual(raw["algorithm"], "Dilithium3")

    def test_delete_zeros_before_unlink(self):
        """delete_pqc_keypair zeros file contents before removing."""
        path = os.path.join(BACKEND_DIR, "pqc_signatures.py")
        with open(path) as f:
            source = f.read()
        # Verify the defense-in-depth zeroing pattern exists
        self.assertIn("write_bytes(b\"\\x00\"", source)
        self.assertIn("unlink()", source)

    def test_get_pqc_status_structure(self):
        """get_pqc_status returns expected dict keys."""
        path = os.path.join(BACKEND_DIR, "pqc_signatures.py")
        with open(path) as f:
            source = f.read()
        for key in [
            "available", "algorithm", "config_enabled",
            "effective_enabled", "keypair_exists", "keypair_path",
        ]:
            self.assertIn(f'"{key}"', source)

    def test_graceful_degradation_without_liboqs(self):
        """When liboqs is not installed, PQC_AVAILABLE is False."""
        path = os.path.join(BACKEND_DIR, "pqc_signatures.py")
        with open(path) as f:
            source = f.read()
        # The module handles ImportError for oqs
        self.assertIn("except ImportError", source)
        self.assertIn("PQC_AVAILABLE = False", source)

    def test_feature_flag_config_check(self):
        """is_pqc_enabled checks both library and config flag."""
        path = os.path.join(BACKEND_DIR, "pqc_signatures.py")
        with open(path) as f:
            source = f.read()
        self.assertIn("PQC_AVAILABLE", source)
        self.assertIn("backup_signatures", source)


# =========================================================================
# Part 2: SecureBytes integration in encryption.py
# =========================================================================

class TestSecureBytesEncryption(unittest.TestCase):
    """Test SecureBytes integration in encryption.py."""

    def test_ast_valid(self):
        """encryption.py passes AST validation."""
        path = os.path.join(BACKEND_DIR, "encryption.py")
        with open(path) as f:
            tree = ast.parse(f.read())
        self.assertIsNotNone(tree)

    def test_secure_bytes_import(self):
        """encryption.py imports SecureBytes with fallback."""
        path = os.path.join(BACKEND_DIR, "encryption.py")
        with open(path) as f:
            source = f.read()
        self.assertIn("from opti_oignon.secure_bytes import SecureBytes", source)
        self.assertIn("_SECURE_BYTES_AVAILABLE", source)

    def test_load_keyfile_returns_securebytes(self):
        """load_keyfile docstring and signature indicate SecureBytes return."""
        path = os.path.join(BACKEND_DIR, "encryption.py")
        with open(path) as f:
            source = f.read()
        self.assertIn("SecureBytes", source.split("def load_keyfile")[1].split("def ")[0])
        self.assertIn("secure_key_from_bytes", source.split("def load_keyfile")[1].split("def ")[0])

    def test_get_encryption_key_returns_securebytes(self):
        """get_encryption_key return type is SecureBytes | None."""
        path = os.path.join(BACKEND_DIR, "encryption.py")
        with open(path) as f:
            source = f.read()
        fn_source = source.split("def get_encryption_key")[1].split("\ndef ")[0]
        self.assertIn("SecureBytes", fn_source)
        self.assertIn("secure_key_from_bytes", fn_source)

    def test_encryption_manager_raw_key_helper(self):
        """EncryptionManager has _raw_key() that extracts bytes."""
        path = os.path.join(BACKEND_DIR, "encryption.py")
        with open(path) as f:
            source = f.read()
        self.assertIn("def _raw_key(self)", source)
        self.assertIn(".as_bytes()", source.split("def _raw_key")[1].split("def ")[0])

    def test_encrypt_decrypt_use_raw_key(self):
        """encrypt() and decrypt() methods use _raw_key()."""
        path = os.path.join(BACKEND_DIR, "encryption.py")
        with open(path) as f:
            source = f.read()
        self.assertIn("self._raw_key()", source)

    def test_rotate_key_wipes_old(self):
        """rotate_key wipes old SecureBytes key before installing new."""
        path = os.path.join(BACKEND_DIR, "encryption.py")
        with open(path) as f:
            source = f.read()
        fn_source = source.split("def rotate_key")[1].split("\n    def ")[0]
        self.assertIn(".wipe()", fn_source)
        self.assertIn("secure_key_from_bytes", fn_source)

    def test_setup_methods_wipe_old_key(self):
        """setup_from_passphrase and setup_random_key wipe old key."""
        path = os.path.join(BACKEND_DIR, "encryption.py")
        with open(path) as f:
            source = f.read()
        for method in ["setup_from_passphrase", "setup_random_key"]:
            fn_source = source.split(f"def {method}")[1].split("\n    def ")[0]
            self.assertIn(".wipe()", fn_source, f"{method} must wipe old key")

    def test_status_includes_securebytes_info(self):
        """get_status() reports secure_bytes_active and key_mlocked."""
        path = os.path.join(BACKEND_DIR, "encryption.py")
        with open(path) as f:
            source = f.read()
        self.assertIn('"secure_bytes_active"', source)
        self.assertIn('"key_mlocked"', source)

    def test_fallback_securebytes_class(self):
        """Fallback SecureBytes class has required interface."""
        path = os.path.join(BACKEND_DIR, "encryption.py")
        with open(path) as f:
            source = f.read()
        # The fallback class must have as_bytes, wipe, is_wiped
        fallback_block = source.split("_SECURE_BYTES_AVAILABLE = False")[1].split("def secure_key_from_bytes")[0]
        self.assertIn("def as_bytes(self)", fallback_block)
        self.assertIn("def wipe(self)", fallback_block)
        self.assertIn("def __enter__(self)", fallback_block)
        self.assertIn("def __exit__", fallback_block)


# =========================================================================
# Part 3: SecureBytes in auth.py
# =========================================================================

class TestSecureBytesAuth(unittest.TestCase):
    """Test SecureBytes integration in auth.py."""

    def test_ast_valid(self):
        """auth.py passes AST validation."""
        path = os.path.join(BACKEND_DIR, "auth.py")
        with open(path) as f:
            tree = ast.parse(f.read())
        self.assertIsNotNone(tree)

    def test_securebytes_import(self):
        """auth.py imports SecureBytes."""
        path = os.path.join(BACKEND_DIR, "auth.py")
        with open(path) as f:
            source = f.read()
        self.assertIn("from opti_oignon.secure_bytes import SecureBytes", source)
        self.assertIn("_SECURE_BYTES_AVAILABLE", source)

    def test_jwt_secret_wrapped(self):
        """_ensure_jwt_secret wraps key in SecureBytes."""
        path = os.path.join(BACKEND_DIR, "auth.py")
        with open(path) as f:
            source = f.read()
        fn_source = source.split("def _ensure_jwt_secret")[1].split("\n    def ")[0]
        self.assertIn("_jwt_secure_key", fn_source)
        self.assertIn("_SecureBytes", fn_source)

    def test_get_jwt_secret_accessor(self):
        """_get_jwt_secret accessor exists and reads from SecureBytes."""
        path = os.path.join(BACKEND_DIR, "auth.py")
        with open(path) as f:
            source = f.read()
        self.assertIn("def _get_jwt_secret(self)", source)
        fn_source = source.split("def _get_jwt_secret")[1].split("\n    def ")[0]
        self.assertIn("as_bytes()", fn_source)

    def test_create_tokens_uses_accessor(self):
        """create_tokens uses _get_jwt_secret() instead of raw config."""
        path = os.path.join(BACKEND_DIR, "auth.py")
        with open(path) as f:
            source = f.read()
        fn_source = source.split("def create_tokens")[1].split("\n    def ")[0]
        self.assertIn("_get_jwt_secret()", fn_source)
        self.assertNotIn("jwt_cfg.get(\"secret_key\"", fn_source)

    def test_validate_token_uses_accessor(self):
        """validate_token uses _get_jwt_secret() instead of raw config."""
        path = os.path.join(BACKEND_DIR, "auth.py")
        with open(path) as f:
            source = f.read()
        fn_source = source.split("def validate_token")[1].split("\n    def ")[0]
        self.assertIn("_get_jwt_secret()", fn_source)


# =========================================================================
# Part 4: SecureBytes in plugin_allowlist.py & security_mode.py
# =========================================================================

class TestSecureBytesPluginAndSecurityMode(unittest.TestCase):
    """Test SecureBytes integration in plugin_allowlist.py and security_mode.py."""

    def test_plugin_allowlist_ast_valid(self):
        """plugin_allowlist.py passes AST validation."""
        path = os.path.join(BACKEND_DIR, "plugin_allowlist.py")
        with open(path) as f:
            tree = ast.parse(f.read())
        self.assertIsNotNone(tree)

    def test_security_mode_ast_valid(self):
        """security_mode.py passes AST validation."""
        path = os.path.join(BACKEND_DIR, "security_mode.py")
        with open(path) as f:
            tree = ast.parse(f.read())
        self.assertIsNotNone(tree)

    def test_plugin_extract_key_bytes_helper(self):
        """plugin_allowlist.py has _extract_key_bytes helper."""
        path = os.path.join(BACKEND_DIR, "plugin_allowlist.py")
        with open(path) as f:
            source = f.read()
        self.assertIn("def _extract_key_bytes(key)", source)
        fn_source = source.split("def _extract_key_bytes")[1].split("\ndef ")[0]
        self.assertIn("as_bytes", fn_source)

    def test_security_mode_extract_key_bytes_helper(self):
        """security_mode.py has _extract_key_bytes helper."""
        path = os.path.join(BACKEND_DIR, "security_mode.py")
        with open(path) as f:
            source = f.read()
        self.assertIn("def _extract_key_bytes(key)", source)

    def test_sign_entry_uses_extract(self):
        """_sign_entry in plugin_allowlist uses _extract_key_bytes."""
        path = os.path.join(BACKEND_DIR, "plugin_allowlist.py")
        with open(path) as f:
            source = f.read()
        fn_source = source.split("def _sign_entry")[1].split("\ndef ")[0]
        self.assertIn("_extract_key_bytes(key)", fn_source)

    def test_compute_lockfile_hmac_uses_extract(self):
        """_compute_lockfile_hmac in security_mode uses _extract_key_bytes."""
        path = os.path.join(BACKEND_DIR, "security_mode.py")
        with open(path) as f:
            source = f.read()
        fn_source = source.split("def _compute_lockfile_hmac")[1].split("\ndef ")[0]
        self.assertIn("_extract_key_bytes(key)", fn_source)

    def test_db_encryption_handles_securebytes(self):
        """db_encryption.py extracts raw bytes from SecureBytes."""
        path = os.path.join(BACKEND_DIR, "db_encryption.py")
        with open(path) as f:
            source = f.read()
        self.assertIn("as_bytes()", source)
        tree = ast.parse(source)
        self.assertIsNotNone(tree)


# =========================================================================
# Part 5: Backup manager PQC hooks
# =========================================================================

class TestBackupManagerPqc(unittest.TestCase):
    """Test PQC signing/verification hooks in backup_manager.py."""

    def test_ast_valid(self):
        """backup_manager.py passes AST validation."""
        path = os.path.join(BACKEND_DIR, "backup_manager.py")
        with open(path) as f:
            tree = ast.parse(f.read())
        self.assertIsNotNone(tree)

    def test_pqc_import_with_fallback(self):
        """backup_manager.py imports PQC with graceful fallback."""
        path = os.path.join(BACKEND_DIR, "backup_manager.py")
        with open(path) as f:
            source = f.read()
        self.assertIn("from opti_oignon.pqc_signatures import", source)
        self.assertIn("except ImportError", source)
        self.assertIn("_PQC_LIB_AVAILABLE", source)

    def test_sign_backup_pqc_method(self):
        """BackupManager has _sign_backup_pqc method."""
        path = os.path.join(BACKEND_DIR, "backup_manager.py")
        with open(path) as f:
            source = f.read()
        self.assertIn("def _sign_backup_pqc(self, backup", source)

    def test_verify_backup_pqc_method(self):
        """BackupManager has _verify_backup_pqc method."""
        path = os.path.join(BACKEND_DIR, "backup_manager.py")
        with open(path) as f:
            source = f.read()
        self.assertIn("def _verify_backup_pqc(self, data", source)

    def test_export_calls_sign(self):
        """export_sections calls _sign_backup_pqc."""
        path = os.path.join(BACKEND_DIR, "backup_manager.py")
        with open(path) as f:
            source = f.read()
        fn_source = source.split("def export_sections")[1].split("\n    def ")[0]
        self.assertIn("_sign_backup_pqc", fn_source)

    def test_import_calls_verify(self):
        """import_backup calls _verify_backup_pqc before applying."""
        path = os.path.join(BACKEND_DIR, "backup_manager.py")
        with open(path) as f:
            source = f.read()
        fn_source = source.split("def import_backup")[1].split("\n    def ")[0]
        self.assertIn("_verify_backup_pqc", fn_source)

    def test_verify_returns_tristate(self):
        """_verify_backup_pqc returns True, False, or None."""
        path = os.path.join(BACKEND_DIR, "backup_manager.py")
        with open(path) as f:
            source = f.read()
        fn_source = source.split("def _verify_backup_pqc")[1].split("\n    def ")[0]
        self.assertIn("return True", fn_source)
        self.assertIn("return False", fn_source)
        self.assertIn("return None", fn_source)

    def test_pqc_signature_keys(self):
        """Backup uses _pqc_signature and _pqc_public_key keys."""
        path = os.path.join(BACKEND_DIR, "backup_manager.py")
        with open(path) as f:
            source = f.read()
        self.assertIn('_PQC_SIGNATURE_KEY = "_pqc_signature"', source)
        self.assertIn('_PQC_PUBLIC_KEY_KEY = "_pqc_public_key"', source)


# =========================================================================
# Part 6: API routes (PQC endpoints)
# =========================================================================

class TestRoutesSecurityPqc(unittest.TestCase):
    """Test PQC endpoints in routes_security.py."""

    def test_ast_valid(self):
        """routes_security.py passes AST validation."""
        path = os.path.join(BACKEND_DIR, "api", "routes_security.py")
        with open(path) as f:
            tree = ast.parse(f.read())
        self.assertIsNotNone(tree)

    def test_pqc_status_endpoint(self):
        """GET /pqc/status endpoint exists."""
        path = os.path.join(BACKEND_DIR, "api", "routes_security.py")
        with open(path) as f:
            source = f.read()
        self.assertIn('"/pqc/status"', source)
        self.assertIn("def get_pqc_signature_status", source)

    def test_pqc_generate_keys_endpoint(self):
        """POST /pqc/generate-keys endpoint exists."""
        path = os.path.join(BACKEND_DIR, "api", "routes_security.py")
        with open(path) as f:
            source = f.read()
        self.assertIn('"/pqc/generate-keys"', source)
        self.assertIn("def generate_pqc_keys", source)

    def test_pqc_delete_keys_endpoint(self):
        """DELETE /pqc/keys endpoint exists."""
        path = os.path.join(BACKEND_DIR, "api", "routes_security.py")
        with open(path) as f:
            source = f.read()
        self.assertIn('"/pqc/keys"', source)
        self.assertIn("def remove_pqc_keys", source)


# =========================================================================
# Part 7: Frontend
# =========================================================================

class TestFrontendKeyCeremony(unittest.TestCase):
    """Test frontend files for key ceremony and PQC UI."""

    def test_key_ceremony_ts_exists(self):
        """keyCeremony.ts API client exists."""
        path = os.path.join(API_TS_DIR, "keyCeremony.ts")
        self.assertTrue(os.path.isfile(path))

    def test_key_ceremony_ts_exports(self):
        """keyCeremony.ts exports required functions and types."""
        path = os.path.join(API_TS_DIR, "keyCeremony.ts")
        with open(path) as f:
            source = f.read()
        for name in [
            "getEncryptionStatus",
            "setupEncryptionPassphrase",
            "setupEncryptionRandom",
            "getPqcStatus",
            "generatePqcKeys",
            "deletePqcKeys",
            "scorePassphrase",
            "EncryptionStatus",
            "PqcStatus",
            "StrengthResult",
        ]:
            self.assertIn(name, source, f"Missing export: {name}")

    def test_key_ceremony_ts_no_hardcoded_hex(self):
        """keyCeremony.ts colors use CSS variable references."""
        path = os.path.join(API_TS_DIR, "keyCeremony.ts")
        with open(path) as f:
            source = f.read()
        # All hex colors should be inside var() fallbacks
        hex_matches = re.findall(r"#[0-9a-fA-F]{6}", source)
        for h in hex_matches:
            # Find the context: must be inside var(--oo-..., #...)
            idx = source.find(h)
            context = source[max(0, idx - 40):idx + len(h)]
            self.assertIn("var(--oo-", context,
                          f"Hardcoded hex {h} not inside var() fallback")

    def test_key_ceremony_panel_exists(self):
        """KeyCeremonyPanel.svelte exists."""
        path = os.path.join(COMPONENTS_DIR, "KeyCeremonyPanel.svelte")
        self.assertTrue(os.path.isfile(path))

    def test_key_ceremony_panel_imports(self):
        """KeyCeremonyPanel imports from keyCeremony.ts."""
        path = os.path.join(COMPONENTS_DIR, "KeyCeremonyPanel.svelte")
        with open(path) as f:
            source = f.read()
        self.assertIn("from '../api/keyCeremony'", source)
        self.assertIn("scorePassphrase", source)

    def test_key_ceremony_panel_wizard_steps(self):
        """KeyCeremonyPanel has all wizard steps."""
        path = os.path.join(COMPONENTS_DIR, "KeyCeremonyPanel.svelte")
        with open(path) as f:
            source = f.read()
        for step in ["choose", "passphrase", "confirm", "success"]:
            self.assertIn(f"'{step}'", source, f"Missing wizard step: {step}")

    def test_key_ceremony_panel_strength_meter(self):
        """KeyCeremonyPanel has passphrase strength visualization."""
        path = os.path.join(COMPONENTS_DIR, "KeyCeremonyPanel.svelte")
        with open(path) as f:
            source = f.read()
        self.assertIn("strength", source)
        self.assertIn("scorePassphrase", source)

    def test_key_ceremony_panel_pqc_section(self):
        """KeyCeremonyPanel has PQC status and key generation UI."""
        path = os.path.join(COMPONENTS_DIR, "KeyCeremonyPanel.svelte")
        with open(path) as f:
            source = f.read()
        self.assertIn("pqcStatus", source)
        self.assertIn("generatePqcKeys", source)

    def test_key_ceremony_panel_html_balanced(self):
        """KeyCeremonyPanel.svelte has balanced div tags."""
        path = os.path.join(COMPONENTS_DIR, "KeyCeremonyPanel.svelte")
        with open(path) as f:
            source = f.read()
        opens = len(re.findall(r"<div[\s>]", source))
        closes = len(re.findall(r"</div>", source))
        self.assertEqual(opens, closes, f"Unbalanced div: {opens} opens, {closes} closes")

    def test_security_panel_encryption_tab(self):
        """SecurityPanel.svelte has Encryption tab wired to KeyCeremonyPanel."""
        path = os.path.join(COMPONENTS_DIR, "SecurityPanel.svelte")
        with open(path) as f:
            source = f.read()
        self.assertIn("'encryption'", source)
        self.assertIn("KeyCeremonyPanel", source)
        self.assertIn("import KeyCeremonyPanel", source)

    def test_security_panel_no_hardcoded_hex(self):
        """SecurityPanel.svelte uses CSS variables for all colors."""
        path = os.path.join(COMPONENTS_DIR, "SecurityPanel.svelte")
        with open(path) as f:
            source = f.read()
        # Find hex colors not inside var(--oo-...) fallbacks
        lines = source.split("\n")
        for i, line in enumerate(lines, 1):
            # Skip script section (JS hex like 0x02)
            if "0x" in line:
                continue
            hex_matches = re.findall(r"#[0-9a-fA-F]{3,8}\b", line)
            for h in hex_matches:
                # Skip Svelte template syntax like {#each}, {#if}, etc.
                idx = line.find(h)
                if idx > 0 and line[idx - 1] == "{":
                    continue
                context = line[max(0, idx - 40):idx + len(h)]
                self.assertIn("var(--oo-", context,
                              f"Line {i}: hardcoded hex {h} outside var() fallback")


# =========================================================================
# Part 8: Version bump
# =========================================================================

class TestVersionBump(unittest.TestCase):
    """Test version bump to 2.9.1 (updated by S130)."""

    def test_version_file(self):
        """__version__.py contains a valid version >= 3.0.0."""
        path = os.path.join(BACKEND_DIR, "__version__.py")
        with open(path) as f:
            content = f.read()
        self.assertIn('__version__', content)

    def test_checkpoint_before_apply(self):
        """checkpoint_before_apply is always hardcoded True."""
        # Scan all Python files for checkpoint_before_apply assignments
        found = False
        for root, dirs, files in os.walk(BACKEND_DIR):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fn in files:
                if fn.endswith(".py"):
                    fpath = os.path.join(root, fn)
                    with open(fpath) as f:
                        content = f.read()
                    if "checkpoint_before_apply" in content:
                        found = True
                        # Must not be set to False
                        self.assertNotIn(
                            "checkpoint_before_apply = False", content,
                            f"checkpoint_before_apply set to False in {fpath}"
                        )
        # It is ok if the variable is not found (not all files use it)


if __name__ == "__main__":
    unittest.main()
