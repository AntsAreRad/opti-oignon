#!/usr/bin/env python3
"""S142 — Multi-User Data Isolation Hardening tests.

Verifies:
- Part 1: Per-user encryption keys (Argon2id / PBKDF2 derivation)
- Part 2: UserKeyManager lifecycle (init, cache, wipe, rotate, expire)
- Part 3: UserKeySaltStore persistence
- Part 4: RBAC enforcement module (dependencies, role checks)
- Part 5: Admin audit logging (store, query, count)
- Part 6: Per-user plugin configurations
- Part 7: Per-user RAG collection namespacing
- Part 8: User data export (GDPR)
- Part 9: User data deletion (cascade)
- Part 10: API routes file structure
- Part 11: Cross-user data isolation
- Part 12: Version bump (3.1.2)

Target: ~80 tests
"""

import ast
import importlib.util
import json
import os
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PKG = os.path.join(_PROJECT_ROOT, "opti_oignon")
_API = os.path.join(_PKG, "api")


def _load_module(name: str, path: str):
    """Load a module without triggering __init__ import chain."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Ensure parent package is in sys.modules for relative imports
    parent = name.rsplit(".", 1)[0] if "." in name else None
    if parent and parent not in sys.modules:
        sys.modules[parent] = MagicMock()
    spec.loader.exec_module(mod)
    return mod


def _ensure_deps():
    """Ensure db_utils and secure_bytes are loadable."""
    if "opti_oignon.db_utils" not in sys.modules:
        try:
            _load_module("opti_oignon.db_utils", os.path.join(_PKG, "db_utils.py"))
            sys.modules["opti_oignon.db_utils"] = _load_module(
                "opti_oignon.db_utils", os.path.join(_PKG, "db_utils.py")
            )
        except Exception:
            # Stub safe_connect
            mock_mod = MagicMock()
            mock_mod.safe_connect = lambda p, **kw: sqlite3.connect(str(p), **kw)
            sys.modules["opti_oignon.db_utils"] = mock_mod

    if "opti_oignon.secure_bytes" not in sys.modules:
        try:
            mod = _load_module(
                "opti_oignon.secure_bytes",
                os.path.join(_PKG, "secure_bytes.py"),
            )
            sys.modules["opti_oignon.secure_bytes"] = mod
        except Exception:
            pass


_ensure_deps()


def _load_user_key_manager():
    mod = _load_module(
        "opti_oignon.user_key_manager",
        os.path.join(_PKG, "user_key_manager.py"),
    )
    sys.modules["opti_oignon.user_key_manager"] = mod
    return mod


def _load_admin_audit():
    mod = _load_module(
        "opti_oignon.admin_audit",
        os.path.join(_PKG, "admin_audit.py"),
    )
    sys.modules["opti_oignon.admin_audit"] = mod
    return mod


def _load_plugin_user_config():
    mod = _load_module(
        "opti_oignon.plugin_user_config",
        os.path.join(_PKG, "plugin_user_config.py"),
    )
    sys.modules["opti_oignon.plugin_user_config"] = mod
    return mod


def _load_user_data_manager():
    mod = _load_module(
        "opti_oignon.user_data_manager",
        os.path.join(_PKG, "user_data_manager.py"),
    )
    sys.modules["opti_oignon.user_data_manager"] = mod
    return mod


def _load_rbac():
    """Load rbac_enforcement with stubs for fastapi."""
    # Ensure fastapi stubs exist
    if "fastapi" not in sys.modules:
        mock_fastapi = MagicMock()
        mock_fastapi.Depends = lambda x: x
        mock_fastapi.Header = MagicMock(return_value=None)
        mock_fastapi.HTTPException = type("HTTPException", (Exception,), {
            "__init__": lambda self, status_code=500, detail="": (
                setattr(self, "status_code", status_code) or
                setattr(self, "detail", detail)
            )
        })
        mock_fastapi.Request = MagicMock
        sys.modules["fastapi"] = mock_fastapi
    mod = _load_module(
        "opti_oignon.rbac_enforcement",
        os.path.join(_PKG, "rbac_enforcement.py"),
    )
    sys.modules["opti_oignon.rbac_enforcement"] = mod
    return mod


# ============================================================================
# Part 1: Per-user encryption key derivation
# ============================================================================


class TestDeriveUserSubkey(unittest.TestCase):
    """Test derive_user_subkey function."""

    def setUp(self):
        self.mod = _load_user_key_manager()

    def test_derive_returns_32_byte_key(self):
        key, salt, kdf = self.mod.derive_user_subkey("password123")
        self.assertEqual(len(key), 32)

    def test_derive_returns_salt(self):
        key, salt, kdf = self.mod.derive_user_subkey("password123")
        self.assertIsInstance(salt, bytes)
        self.assertGreater(len(salt), 0)

    def test_derive_returns_kdf_name(self):
        key, salt, kdf = self.mod.derive_user_subkey("password123")
        self.assertIn(kdf, ("argon2id", "pbkdf2"))

    def test_derive_same_password_same_salt_same_key(self):
        key1, salt1, _ = self.mod.derive_user_subkey("password123")
        key2, _, _ = self.mod.derive_user_subkey("password123", salt=salt1)
        self.assertEqual(key1, key2)

    def test_derive_different_passwords_different_keys(self):
        key1, salt, _ = self.mod.derive_user_subkey("password1")
        key2, _, _ = self.mod.derive_user_subkey("password2", salt=salt)
        self.assertNotEqual(key1, key2)

    def test_derive_different_salts_different_keys(self):
        key1, salt1, _ = self.mod.derive_user_subkey("password")
        key2, salt2, _ = self.mod.derive_user_subkey("password")
        # Random salts should differ
        if salt1 != salt2:
            self.assertNotEqual(key1, key2)

    def test_force_pbkdf2(self):
        key, salt, kdf = self.mod.derive_user_subkey("password", force_pbkdf2=True)
        self.assertEqual(kdf, "pbkdf2")
        self.assertEqual(len(key), 32)

    def test_provided_salt_used(self):
        custom_salt = b"0123456789abcdef"
        key, salt, _ = self.mod.derive_user_subkey("password", salt=custom_salt)
        self.assertEqual(salt, custom_salt)


# ============================================================================
# Part 2: UserKeyManager lifecycle
# ============================================================================


class TestUserKeyManager(unittest.TestCase):
    """Test UserKeyManager cache lifecycle."""

    def setUp(self):
        self.mod = _load_user_key_manager()
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, "auth.db")
        self.mgr = self.mod.UserKeyManager(db_path=self.db_path, key_ttl=3600)

    def test_initialize_user_key(self):
        result = self.mgr.initialize_user_key("user1", "password1")
        self.assertTrue(result)

    def test_key_cached_after_init(self):
        self.mgr.initialize_user_key("user1", "password1")
        self.assertTrue(self.mgr.is_key_cached("user1"))

    def test_get_user_key_bytes(self):
        self.mgr.initialize_user_key("user1", "password1")
        key_bytes = self.mgr.get_user_key_bytes("user1")
        self.assertIsNotNone(key_bytes)
        self.assertEqual(len(key_bytes), 32)

    def test_derive_and_cache(self):
        # First init creates salt
        self.mgr.initialize_user_key("user1", "password1")
        key1 = self.mgr.get_user_key_bytes("user1")

        # Wipe and re-derive
        self.mgr.wipe_user_key("user1")
        self.assertFalse(self.mgr.is_key_cached("user1"))

        result = self.mgr.derive_and_cache("user1", "password1")
        self.assertTrue(result)
        key2 = self.mgr.get_user_key_bytes("user1")
        self.assertEqual(key1, key2)  # Same password + salt = same key

    def test_wipe_user_key(self):
        self.mgr.initialize_user_key("user1", "password1")
        self.assertTrue(self.mgr.is_key_cached("user1"))
        wiped = self.mgr.wipe_user_key("user1")
        self.assertTrue(wiped)
        self.assertFalse(self.mgr.is_key_cached("user1"))

    def test_wipe_nonexistent(self):
        wiped = self.mgr.wipe_user_key("nobody")
        self.assertFalse(wiped)

    def test_wipe_all(self):
        self.mgr.initialize_user_key("user1", "p1")
        self.mgr.initialize_user_key("user2", "p2")
        count = self.mgr.wipe_all()
        self.assertEqual(count, 2)
        self.assertFalse(self.mgr.is_key_cached("user1"))
        self.assertFalse(self.mgr.is_key_cached("user2"))

    def test_expired_key_not_returned(self):
        mgr = self.mod.UserKeyManager(db_path=self.db_path, key_ttl=0)
        mgr.initialize_user_key("user1", "password1")
        time.sleep(0.05)
        self.assertIsNone(mgr.get_user_key("user1"))

    def test_cleanup_expired(self):
        mgr = self.mod.UserKeyManager(db_path=self.db_path, key_ttl=0)
        mgr.initialize_user_key("user1", "p1")
        mgr.initialize_user_key("user2", "p2")
        time.sleep(0.05)
        count = mgr.cleanup_expired()
        self.assertEqual(count, 2)

    def test_rotate_user_key(self):
        self.mgr.initialize_user_key("user1", "old_password")
        old_key = self.mgr.get_user_key_bytes("user1")
        old_ret, new_ret = self.mgr.rotate_user_key("user1", "new_password")
        self.assertEqual(old_ret, old_key)
        self.assertIsNotNone(new_ret)
        self.assertNotEqual(old_ret, new_ret)
        # New key is cached
        self.assertEqual(self.mgr.get_user_key_bytes("user1"), new_ret)

    def test_delete_user_keys(self):
        self.mgr.initialize_user_key("user1", "password1")
        self.assertTrue(self.mgr.salt_store.has_salt("user1"))
        result = self.mgr.delete_user_keys("user1")
        self.assertTrue(result)
        self.assertFalse(self.mgr.is_key_cached("user1"))
        self.assertFalse(self.mgr.salt_store.has_salt("user1"))

    def test_get_status(self):
        self.mgr.initialize_user_key("user1", "p1")
        status = self.mgr.get_status()
        self.assertEqual(status["cached_keys"], 1)
        self.assertIn("kdf", status)
        self.assertIn("key_ttl_seconds", status)

    def test_two_users_different_keys(self):
        self.mgr.initialize_user_key("alice", "alice_pw")
        self.mgr.initialize_user_key("bob", "bob_pw")
        k_alice = self.mgr.get_user_key_bytes("alice")
        k_bob = self.mgr.get_user_key_bytes("bob")
        self.assertNotEqual(k_alice, k_bob)

    def test_derive_and_cache_no_existing_salt(self):
        """derive_and_cache with no prior salt should initialize."""
        result = self.mgr.derive_and_cache("new_user", "password")
        self.assertTrue(result)
        self.assertTrue(self.mgr.is_key_cached("new_user"))
        self.assertTrue(self.mgr.salt_store.has_salt("new_user"))


# ============================================================================
# Part 3: UserKeySaltStore persistence
# ============================================================================


class TestUserKeySaltStore(unittest.TestCase):
    """Test salt storage persistence."""

    def setUp(self):
        self.mod = _load_user_key_manager()
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, "salt_test.db")
        self.store = self.mod.UserKeySaltStore(self.db_path)

    def test_store_and_retrieve_salt(self):
        salt = b"test_salt_16byte"
        self.store.store_salt("user1", salt, "argon2id")
        result = self.store.get_salt("user1")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], salt)
        self.assertEqual(result[1], "argon2id")

    def test_get_nonexistent_salt(self):
        result = self.store.get_salt("nobody")
        self.assertIsNone(result)

    def test_has_salt(self):
        self.assertFalse(self.store.has_salt("user1"))
        self.store.store_salt("user1", b"salt", "pbkdf2")
        self.assertTrue(self.store.has_salt("user1"))

    def test_delete_salt(self):
        self.store.store_salt("user1", b"salt", "argon2id")
        self.assertTrue(self.store.delete_salt("user1"))
        self.assertFalse(self.store.has_salt("user1"))

    def test_delete_nonexistent_salt(self):
        self.assertFalse(self.store.delete_salt("nobody"))

    def test_upsert_salt(self):
        """store_salt should update on conflict."""
        self.store.store_salt("user1", b"salt_v1", "argon2id")
        self.store.store_salt("user1", b"salt_v2", "pbkdf2")
        result = self.store.get_salt("user1")
        self.assertEqual(result[0], b"salt_v2")
        self.assertEqual(result[1], "pbkdf2")


# ============================================================================
# Part 4: RBAC enforcement module
# ============================================================================


class TestRBACEnforcement(unittest.TestCase):
    """Test RBAC enforcement module structure and helpers."""

    def test_module_exists(self):
        path = os.path.join(_PKG, "rbac_enforcement.py")
        self.assertTrue(os.path.isfile(path))

    def test_ast_valid(self):
        path = os.path.join(_PKG, "rbac_enforcement.py")
        with open(path) as fh:
            ast.parse(fh.read())

    def test_has_key_functions(self):
        mod = _load_rbac()
        for name in [
            "get_current_user", "get_user_id", "get_user_role",
            "require_admin", "require_role",
            "enforce_user_ownership", "get_effective_user_id",
            "is_admin", "log_admin_action",
        ]:
            self.assertTrue(hasattr(mod, name), f"Missing: {name}")

    def test_enforce_user_ownership_admin_passes(self):
        mod = _load_rbac()
        admin = {"sub": "admin1", "role": "admin"}
        # Should not raise
        mod.enforce_user_ownership("other_user", admin)

    def test_enforce_user_ownership_owner_passes(self):
        mod = _load_rbac()
        user = {"sub": "user1", "role": "user"}
        mod.enforce_user_ownership("user1", user)

    def test_enforce_user_ownership_non_owner_fails(self):
        mod = _load_rbac()
        user = {"sub": "user1", "role": "user"}
        with self.assertRaises(Exception):
            mod.enforce_user_ownership("user2", user)

    def test_get_effective_user_id_self(self):
        mod = _load_rbac()
        user = {"sub": "user1", "role": "user"}
        result = mod.get_effective_user_id(None, user)
        self.assertEqual(result, "user1")

    def test_get_effective_user_id_admin_override(self):
        mod = _load_rbac()
        admin = {"sub": "admin1", "role": "admin"}
        result = mod.get_effective_user_id("target_user", admin)
        self.assertEqual(result, "target_user")

    def test_get_effective_user_id_non_admin_blocked(self):
        mod = _load_rbac()
        user = {"sub": "user1", "role": "user"}
        with self.assertRaises(Exception):
            mod.get_effective_user_id("other_user", user)

    def test_is_admin(self):
        mod = _load_rbac()
        self.assertTrue(mod.is_admin({"role": "admin"}))
        self.assertFalse(mod.is_admin({"role": "user"}))
        self.assertFalse(mod.is_admin({"role": "viewer"}))
        self.assertFalse(mod.is_admin({}))

    def test_availability_flag(self):
        mod = _load_rbac()
        self.assertTrue(mod.RBAC_ENFORCEMENT_AVAILABLE)


# ============================================================================
# Part 5: Admin audit logging
# ============================================================================


class TestAdminAudit(unittest.TestCase):
    """Test AdminAuditStore."""

    def setUp(self):
        self.mod = _load_admin_audit()
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, "admin_audit.db")
        self.store = self.mod.AdminAuditStore(db_path=self.db_path)

    def test_log_event(self):
        entry_id = self.store.log_event(
            admin_id="admin1",
            action="delete_user_data",
            target_type="user",
            target_id="user1",
        )
        self.assertGreater(entry_id, 0)

    def test_get_events(self):
        self.store.log_event("admin1", "export", "user", "u1")
        self.store.log_event("admin1", "delete", "user", "u2")
        events = self.store.get_events()
        self.assertEqual(len(events), 2)

    def test_filter_by_admin(self):
        self.store.log_event("admin1", "export", "user", "u1")
        self.store.log_event("admin2", "delete", "user", "u2")
        events = self.store.get_events(admin_id="admin1")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["admin_id"], "admin1")

    def test_filter_by_target_type(self):
        self.store.log_event("admin1", "delete", "user", "u1")
        self.store.log_event("admin1", "delete", "conversation", "c1")
        events = self.store.get_events(target_type="conversation")
        self.assertEqual(len(events), 1)

    def test_filter_by_target_id(self):
        self.store.log_event("admin1", "export", "user", "u1")
        self.store.log_event("admin1", "export", "user", "u2")
        events = self.store.get_events(target_id="u1")
        self.assertEqual(len(events), 1)

    def test_filter_since(self):
        self.store.log_event("admin1", "a1", "user", "u1")
        cutoff = time.time()
        time.sleep(0.05)
        self.store.log_event("admin1", "a2", "user", "u2")
        events = self.store.get_events(since=cutoff)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["action"], "a2")

    def test_count_events(self):
        self.store.log_event("admin1", "a", "user", "u1")
        self.store.log_event("admin1", "b", "user", "u2")
        self.store.log_event("admin2", "c", "user", "u3")
        self.assertEqual(self.store.count_events(), 3)
        self.assertEqual(self.store.count_events(admin_id="admin1"), 2)

    def test_pagination(self):
        for i in range(5):
            self.store.log_event("admin1", f"action_{i}", "user", f"u{i}")
        events = self.store.get_events(limit=2, offset=0)
        self.assertEqual(len(events), 2)
        events2 = self.store.get_events(limit=2, offset=2)
        self.assertEqual(len(events2), 2)

    def test_details_stored(self):
        self.store.log_event("admin1", "delete", "user", "u1", details='{"count": 5}')
        events = self.store.get_events()
        self.assertEqual(events[0]["details"], '{"count": 5}')

    def test_ip_address_stored(self):
        self.store.log_event("admin1", "a", "user", "u1", ip_address="192.168.1.1")
        events = self.store.get_events()
        self.assertEqual(events[0]["ip_address"], "192.168.1.1")

    def test_delete_events_for_target(self):
        self.store.log_event("admin1", "a", "user", "u1")
        self.store.log_event("admin1", "b", "user", "u1")
        self.store.log_event("admin1", "c", "user", "u2")
        deleted = self.store.delete_events_for_target("user", "u1")
        self.assertEqual(deleted, 2)
        self.assertEqual(self.store.count_events(), 1)

    def test_convenience_function(self):
        mod = self.mod
        # Reset singleton
        mod._admin_audit_store = None
        mod._admin_audit_store = self.store
        entry_id = mod.log_admin_event("admin1", "test", "user", "u1")
        self.assertGreater(entry_id, 0)

    def test_availability_flag(self):
        self.assertTrue(self.mod.ADMIN_AUDIT_AVAILABLE)


# ============================================================================
# Part 6: Per-user plugin configurations
# ============================================================================


class TestPluginUserConfig(unittest.TestCase):
    """Test PluginUserConfigStore."""

    def setUp(self):
        self.mod = _load_plugin_user_config()
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, "puc.db")
        self.store = self.mod.PluginUserConfigStore(db_path=self.db_path)

    def test_no_config_returns_none(self):
        result = self.store.get_config("user1", "plugin_a")
        self.assertIsNone(result)

    def test_set_and_get_config(self):
        self.store.set_config("user1", "plugin_a", enabled=True, preferences={"k": "v"})
        result = self.store.get_config("user1", "plugin_a")
        self.assertIsNotNone(result)
        self.assertTrue(result["enabled"])
        self.assertEqual(result["preferences"]["k"], "v")

    def test_disable_plugin(self):
        self.store.set_config("user1", "plugin_a", enabled=True)
        self.store.set_config("user1", "plugin_a", enabled=False)
        result = self.store.get_config("user1", "plugin_a")
        self.assertFalse(result["enabled"])

    def test_is_plugin_enabled_default(self):
        # No config = default enabled
        self.assertTrue(self.store.is_plugin_enabled("user1", "unknown_plugin"))

    def test_is_plugin_enabled_explicit(self):
        self.store.set_config("user1", "plugin_a", enabled=False)
        self.assertFalse(self.store.is_plugin_enabled("user1", "plugin_a"))

    def test_get_all_configs(self):
        self.store.set_config("user1", "plugin_a", enabled=True)
        self.store.set_config("user1", "plugin_b", enabled=False)
        self.store.set_config("user2", "plugin_a", enabled=True)
        configs = self.store.get_all_configs("user1")
        self.assertEqual(len(configs), 2)

    def test_delete_config(self):
        self.store.set_config("user1", "plugin_a", enabled=True)
        self.assertTrue(self.store.delete_config("user1", "plugin_a"))
        self.assertIsNone(self.store.get_config("user1", "plugin_a"))

    def test_delete_all_configs(self):
        self.store.set_config("user1", "plugin_a", enabled=True)
        self.store.set_config("user1", "plugin_b", enabled=True)
        count = self.store.delete_all_configs("user1")
        self.assertEqual(count, 2)
        self.assertEqual(len(self.store.get_all_configs("user1")), 0)

    def test_preferences_merge(self):
        self.store.set_config("user1", "p", preferences={"a": 1, "b": 2})
        self.store.set_config("user1", "p", preferences={"b": 99, "c": 3})
        result = self.store.get_config("user1", "p")
        self.assertEqual(result["preferences"]["a"], 1)
        self.assertEqual(result["preferences"]["b"], 99)
        self.assertEqual(result["preferences"]["c"], 3)

    def test_cross_user_isolation(self):
        """User1 config should not leak to user2."""
        self.store.set_config("user1", "plugin_a", enabled=False, preferences={"secret": "x"})
        self.store.set_config("user2", "plugin_a", enabled=True)
        u1 = self.store.get_config("user1", "plugin_a")
        u2 = self.store.get_config("user2", "plugin_a")
        self.assertFalse(u1["enabled"])
        self.assertTrue(u2["enabled"])
        self.assertNotIn("secret", u2.get("preferences", {}))

    def test_availability_flag(self):
        self.assertTrue(self.mod.PLUGIN_USER_CONFIG_AVAILABLE)


# ============================================================================
# Part 7: Per-user RAG collection namespacing
# ============================================================================


class TestRAGNamespacing(unittest.TestCase):
    """Test RAG collection namespacing helpers."""

    def setUp(self):
        self.mod = _load_user_data_manager()

    def test_user_collection_name(self):
        name = self.mod.user_collection_name("alice", "research")
        self.assertEqual(name, "user_alice_research")

    def test_is_user_collection_true(self):
        self.assertTrue(
            self.mod.is_user_collection("alice", "user_alice_research")
        )

    def test_is_user_collection_false(self):
        self.assertFalse(
            self.mod.is_user_collection("bob", "user_alice_research")
        )

    def test_get_user_collections(self):
        all_colls = [
            "user_alice_docs",
            "user_alice_papers",
            "user_bob_notes",
            "global_shared",
        ]
        result = self.mod.get_user_collections("alice", all_colls)
        self.assertEqual(len(result), 2)
        self.assertIn("user_alice_docs", result)
        self.assertIn("user_alice_papers", result)

    def test_strip_user_prefix(self):
        result = self.mod.strip_user_prefix("alice", "user_alice_research")
        self.assertEqual(result, "research")

    def test_strip_user_prefix_no_match(self):
        result = self.mod.strip_user_prefix("bob", "user_alice_research")
        self.assertEqual(result, "user_alice_research")


# ============================================================================
# Part 8: User data export
# ============================================================================


class TestUserDataExporter(unittest.TestCase):
    """Test UserDataExporter."""

    def setUp(self):
        self.mod = _load_user_data_manager()

    def test_exporter_class_exists(self):
        self.assertTrue(hasattr(self.mod, "UserDataExporter"))

    def test_export_returns_dict_structure(self):
        exporter = self.mod.UserDataExporter()
        # With no backends connected, should return empty collections
        data = exporter.export("test_user")
        self.assertIn("export_metadata", data)
        self.assertIn("conversations", data)
        self.assertIn("memories", data)
        self.assertIn("rag_collections", data)
        self.assertIn("plugin_configs", data)
        self.assertIn("settings", data)
        self.assertEqual(data["export_metadata"]["user_id"], "test_user")

    def test_export_metadata_has_timestamp(self):
        exporter = self.mod.UserDataExporter()
        data = exporter.export("user1")
        self.assertIn("exported_at", data["export_metadata"])
        self.assertIsInstance(data["export_metadata"]["exported_at"], float)

    def test_export_format_version(self):
        exporter = self.mod.UserDataExporter()
        data = exporter.export("user1")
        self.assertEqual(data["export_metadata"]["format_version"], "1.0")

    def test_singleton(self):
        self.mod._exporter = None
        e1 = self.mod.get_user_data_exporter()
        e2 = self.mod.get_user_data_exporter()
        self.assertIs(e1, e2)


# ============================================================================
# Part 9: User data deletion
# ============================================================================


class TestUserDataDeleter(unittest.TestCase):
    """Test UserDataDeleter."""

    def setUp(self):
        self.mod = _load_user_data_manager()

    def test_deleter_class_exists(self):
        self.assertTrue(hasattr(self.mod, "UserDataDeleter"))

    def test_delete_returns_summary(self):
        deleter = self.mod.UserDataDeleter()
        result = deleter.delete_all("test_user")
        self.assertIn("user_id", result)
        self.assertIn("deleted_at", result)
        self.assertIn("conversations", result)
        self.assertIn("memories", result)
        self.assertIn("rag_collections", result)
        self.assertIn("plugin_configs", result)
        self.assertIn("settings", result)
        self.assertIn("encryption_keys", result)

    def test_delete_user_id_in_result(self):
        deleter = self.mod.UserDataDeleter()
        result = deleter.delete_all("user123")
        self.assertEqual(result["user_id"], "user123")

    def test_singleton(self):
        self.mod._deleter = None
        d1 = self.mod.get_user_data_deleter()
        d2 = self.mod.get_user_data_deleter()
        self.assertIs(d1, d2)


# ============================================================================
# Part 10: API routes file structure
# ============================================================================


class TestRoutesUsersFile(unittest.TestCase):
    """Test routes_users.py exists and has correct structure."""

    def test_file_exists(self):
        path = os.path.join(_API, "routes_users.py")
        self.assertTrue(os.path.isfile(path))

    def test_ast_valid(self):
        path = os.path.join(_API, "routes_users.py")
        with open(path) as fh:
            tree = ast.parse(fh.read())
        # Check for expected function names
        func_names = [
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        self.assertIn("export_user_data", func_names)
        self.assertIn("delete_user_data", func_names)
        self.assertIn("get_admin_audit", func_names)
        self.assertIn("get_key_status", func_names)
        self.assertIn("derive_user_key", func_names)
        self.assertIn("wipe_key_cache", func_names)
        self.assertIn("get_user_plugins", func_names)
        self.assertIn("set_user_plugin", func_names)

    def test_router_registered_in_app(self):
        path = os.path.join(_API, "app.py")
        with open(path) as fh:
            content = fh.read()
        self.assertIn("routes_users", content)
        self.assertIn("users_router", content)

    def test_routes_users_has_schemas(self):
        path = os.path.join(_API, "routes_users.py")
        with open(path) as fh:
            tree = ast.parse(fh.read())
        class_names = [
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        ]
        self.assertIn("DeriveKeyRequest", class_names)
        self.assertIn("PluginConfigRequest", class_names)
        self.assertIn("DeleteDataResponse", class_names)


# ============================================================================
# Part 11: Cross-user data isolation
# ============================================================================


class TestCrossUserIsolation(unittest.TestCase):
    """Test that users cannot access each other's data."""

    def test_encryption_keys_isolated(self):
        """Different users get different encryption keys."""
        mod = _load_user_key_manager()
        tmp = tempfile.mkdtemp()
        mgr = mod.UserKeyManager(db_path=os.path.join(tmp, "auth.db"))
        mgr.initialize_user_key("alice", "alice_pass")
        mgr.initialize_user_key("bob", "bob_pass")

        key_alice = mgr.get_user_key_bytes("alice")
        key_bob = mgr.get_user_key_bytes("bob")
        self.assertNotEqual(key_alice, key_bob)

        # Alice's key cannot be retrieved as Bob
        mgr.wipe_user_key("alice")
        self.assertIsNone(mgr.get_user_key("alice"))
        self.assertIsNotNone(mgr.get_user_key("bob"))

    def test_plugin_configs_isolated(self):
        """Users have independent plugin configurations."""
        mod = _load_plugin_user_config()
        tmp = tempfile.mkdtemp()
        store = mod.PluginUserConfigStore(db_path=os.path.join(tmp, "puc.db"))

        store.set_config("alice", "search", enabled=True, preferences={"engine": "brave"})
        store.set_config("bob", "search", enabled=False, preferences={"engine": "ddg"})

        alice_cfg = store.get_config("alice", "search")
        bob_cfg = store.get_config("bob", "search")
        self.assertTrue(alice_cfg["enabled"])
        self.assertFalse(bob_cfg["enabled"])
        self.assertNotEqual(
            alice_cfg["preferences"]["engine"],
            bob_cfg["preferences"]["engine"],
        )

    def test_rag_collections_isolated(self):
        """RAG collections are namespaced per user."""
        mod = _load_user_data_manager()
        all_colls = [
            "user_alice_docs",
            "user_bob_notes",
            "shared_global",
        ]
        alice_colls = mod.get_user_collections("alice", all_colls)
        bob_colls = mod.get_user_collections("bob", all_colls)
        self.assertEqual(len(alice_colls), 1)
        self.assertEqual(len(bob_colls), 1)
        self.assertNotEqual(alice_colls, bob_colls)

    def test_admin_can_access_any_user(self):
        """Admin should be able to access any user's effective ID."""
        mod = _load_rbac()
        admin = {"sub": "admin1", "role": "admin"}
        # Admin targeting alice
        eff = mod.get_effective_user_id("alice", admin)
        self.assertEqual(eff, "alice")
        # Admin targeting bob
        eff = mod.get_effective_user_id("bob", admin)
        self.assertEqual(eff, "bob")

    def test_user_cannot_access_other(self):
        """Non-admin user cannot target another user."""
        mod = _load_rbac()
        alice = {"sub": "alice", "role": "user"}
        with self.assertRaises(Exception):
            mod.get_effective_user_id("bob", alice)


# ============================================================================
# Part 12: Version bump + module structure
# ============================================================================


class TestVersionAndModules(unittest.TestCase):
    """Test version bump and new module presence."""

    def test_version_3_1_2(self):
        spec = importlib.util.spec_from_file_location(
            "ver", os.path.join(_PKG, "__version__.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.__version__, "3.1.2")

    def test_user_key_manager_exists(self):
        self.assertTrue(os.path.isfile(os.path.join(_PKG, "user_key_manager.py")))

    def test_rbac_enforcement_exists(self):
        self.assertTrue(os.path.isfile(os.path.join(_PKG, "rbac_enforcement.py")))

    def test_admin_audit_exists(self):
        self.assertTrue(os.path.isfile(os.path.join(_PKG, "admin_audit.py")))

    def test_user_data_manager_exists(self):
        self.assertTrue(os.path.isfile(os.path.join(_PKG, "user_data_manager.py")))

    def test_plugin_user_config_exists(self):
        self.assertTrue(os.path.isfile(os.path.join(_PKG, "plugin_user_config.py")))

    def test_routes_users_exists(self):
        self.assertTrue(os.path.isfile(os.path.join(_API, "routes_users.py")))

    def test_deps_has_s142_entries(self):
        path = os.path.join(_API, "deps.py")
        with open(path) as fh:
            content = fh.read()
        self.assertIn("USER_KEY_MANAGER_AVAILABLE", content)
        self.assertIn("RBAC_ENFORCEMENT_AVAILABLE", content)
        self.assertIn("ADMIN_AUDIT_AVAILABLE", content)
        self.assertIn("USER_DATA_MANAGER_AVAILABLE", content)
        self.assertIn("PLUGIN_USER_CONFIG_AVAILABLE", content)

    def test_all_new_modules_ast_valid(self):
        modules = [
            "user_key_manager.py",
            "rbac_enforcement.py",
            "admin_audit.py",
            "user_data_manager.py",
            "plugin_user_config.py",
        ]
        for m in modules:
            path = os.path.join(_PKG, m)
            with open(path) as fh:
                ast.parse(fh.read())

    def test_app_py_ast_valid(self):
        path = os.path.join(_API, "app.py")
        with open(path) as fh:
            ast.parse(fh.read())

    def test_deps_py_ast_valid(self):
        path = os.path.join(_API, "deps.py")
        with open(path) as fh:
            ast.parse(fh.read())


if __name__ == "__main__":
    unittest.main()
