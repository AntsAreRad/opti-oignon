"""
Tests for S98 -- Multi-User & Authentication.

Validates:
- Part 1: AuthManager (user CRUD, password hashing, JWT, sessions)
- Part 2: Per-user isolation (migration, settings store, query helpers)
- Part 3: Shared projects & RBAC (share, permissions, audit log)
- Part 4: API routes (schemas, endpoints, auth guard)
- Part 5: Frontend (types, API client, auth store, pages, UserMenu, wiring)
- Part 6: Config (auth.yaml)
- Part 7: Integration wiring (deps.py, app.py, version bump)
- Zero regressions

Target: ~62 tests
"""

import importlib.util
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
BACKEND_DIR = os.path.join(PROJECT_ROOT, "opti_oignon")
API_DIR = os.path.join(BACKEND_DIR, "api")
CONFIG_DIR = os.path.join(BACKEND_DIR, "config")
FRONTEND_SRC = os.path.join(PROJECT_ROOT, "frontend", "src")
COMPONENTS_DIR = os.path.join(FRONTEND_SRC, "lib", "components")
ROUTES_DIR = os.path.join(FRONTEND_SRC, "routes")
API_TS_DIR = os.path.join(FRONTEND_SRC, "lib", "api")
STORES_DIR = os.path.join(FRONTEND_SRC, "lib", "stores")


def _load_module(name, path):
    """Load a Python module from file path without triggering __init__ imports."""
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
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _read(path):
    """Read file contents as string."""
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Load auth module with temp DB
# ---------------------------------------------------------------------------

_tmpdir = tempfile.mkdtemp()
_auth_cfg_path = os.path.join(_tmpdir, "auth.yaml")
_auth_db_path = os.path.join(_tmpdir, "auth.db")

with open(_auth_cfg_path, "w") as f:
    yaml.dump(
        {
            "jwt": {
                "secret_key": "test-secret-key-for-unit-tests-only",
                "access_token_expiry_minutes": 60,
                "refresh_token_expiry_days": 30,
                "algorithm": "HS256",
            },
            "password": {
                "min_length": 8,
                "bcrypt_rounds": 4,  # Fast for tests
                "require_uppercase": False,
                "require_digit": False,
                "require_special": False,
            },
            "users": {
                "allow_registration": True,
                "default_role": "user",
                "max_users": 0,
                "require_email": False,
            },
            "session": {
                "max_sessions": 3,
                "invalidate_on_password_change": True,
            },
            "single_user_mode": False,
            "db_path": _auth_db_path,
        },
        f,
    )

auth_mod = _load_module("opti_oignon.auth", os.path.join(BACKEND_DIR, "auth.py"))

# Load user_isolation module
iso_mod = _load_module(
    "opti_oignon.user_isolation",
    os.path.join(BACKEND_DIR, "user_isolation.py"),
)


# ===========================================================================
# Part 1: AuthManager -- User CRUD, passwords, JWT, sessions
# ===========================================================================


class TestAuthManagerUserCRUD(unittest.TestCase):
    """Test user creation, retrieval, update, deletion."""

    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "auth_crud.db")
        self.mgr = auth_mod.AuthManager(
            config_path=_auth_cfg_path, db_path=self.db
        )

    def test_create_user_success(self):
        u = self.mgr.create_user("alice", "password123", email="alice@test.com")
        self.assertIsNotNone(u)
        self.assertEqual(u.username, "alice")
        self.assertEqual(u.email, "alice@test.com")
        self.assertEqual(u.role, "user")
        self.assertTrue(len(u.user_id) > 10)

    def test_create_user_duplicate_username(self):
        self.mgr.create_user("bob", "password123")
        dup = self.mgr.create_user("bob", "anotherpassword")
        self.assertIsNone(dup)

    def test_create_user_short_password(self):
        u = self.mgr.create_user("charlie", "short")
        self.assertIsNone(u)

    def test_create_user_short_username(self):
        u = self.mgr.create_user("x", "password123")
        self.assertIsNone(u)

    def test_get_user_by_id(self):
        created = self.mgr.create_user("dave", "password123")
        fetched = self.mgr.get_user(created.user_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.username, "dave")

    def test_get_user_by_username(self):
        self.mgr.create_user("eve", "password123")
        fetched = self.mgr.get_user_by_username("eve")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.username, "eve")

    def test_get_nonexistent_user(self):
        self.assertIsNone(self.mgr.get_user("nonexistent-id"))
        self.assertIsNone(self.mgr.get_user_by_username("nonexistent"))

    def test_update_user(self):
        u = self.mgr.create_user("frank", "password123")
        updated = self.mgr.update_user(u.user_id, email="frank@new.com")
        self.assertIsNotNone(updated)
        self.assertEqual(updated.email, "frank@new.com")

    def test_update_user_metadata(self):
        u = self.mgr.create_user("grace", "password123")
        updated = self.mgr.update_user(u.user_id, metadata={"theme": "dark"})
        self.assertEqual(updated.metadata["theme"], "dark")

    def test_delete_user(self):
        u = self.mgr.create_user("hank", "password123")
        self.assertTrue(self.mgr.delete_user(u.user_id))
        self.assertIsNone(self.mgr.get_user(u.user_id))

    def test_delete_nonexistent(self):
        self.assertFalse(self.mgr.delete_user("fake-id"))

    def test_list_users(self):
        self.mgr.create_user("user1", "password123")
        self.mgr.create_user("user2", "password123")
        users = self.mgr.list_users()
        self.assertEqual(len(users), 2)

    def test_count_users(self):
        self.assertEqual(self.mgr.count_users(), 0)
        self.mgr.create_user("counter", "password123")
        self.assertEqual(self.mgr.count_users(), 1)

    def test_user_to_dict_excludes_hash(self):
        u = self.mgr.create_user("ivan", "password123")
        d = u.to_dict()
        self.assertNotIn("password_hash", d)
        d_with = u.to_dict(include_hash=True)
        self.assertIn("password_hash", d_with)


class TestPasswordHashing(unittest.TestCase):
    """Test password hashing and verification."""

    def test_hash_and_verify(self):
        pw = "mysecretpassword"
        hashed = auth_mod.hash_password(pw, rounds=4)
        self.assertTrue(auth_mod.verify_password(pw, hashed))

    def test_wrong_password(self):
        hashed = auth_mod.hash_password("correct", rounds=4)
        self.assertFalse(auth_mod.verify_password("wrong", hashed))

    def test_hash_is_different_each_time(self):
        h1 = auth_mod.hash_password("same", rounds=4)
        h2 = auth_mod.hash_password("same", rounds=4)
        self.assertNotEqual(h1, h2)  # Different salts


class TestJWT(unittest.TestCase):
    """Test JWT encode/decode."""

    def test_encode_decode(self):
        payload = {"sub": "user-1", "role": "admin", "exp": int(time.time()) + 3600}
        token = auth_mod.jwt_encode(payload, "secret")
        decoded = auth_mod.jwt_decode(token, "secret")
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded["sub"], "user-1")

    def test_invalid_signature(self):
        payload = {"sub": "user-1", "exp": int(time.time()) + 3600}
        token = auth_mod.jwt_encode(payload, "secret")
        decoded = auth_mod.jwt_decode(token, "wrong-secret")
        self.assertIsNone(decoded)

    def test_expired_token(self):
        payload = {"sub": "user-1", "exp": int(time.time()) - 10}
        token = auth_mod.jwt_encode(payload, "secret")
        decoded = auth_mod.jwt_decode(token, "secret")
        self.assertIsNone(decoded)

    def test_malformed_token(self):
        self.assertIsNone(auth_mod.jwt_decode("not.a.valid.token", "secret"))
        self.assertIsNone(auth_mod.jwt_decode("", "secret"))
        self.assertIsNone(auth_mod.jwt_decode("abc", "secret"))

    def test_no_expiry(self):
        payload = {"sub": "user-1"}
        token = auth_mod.jwt_encode(payload, "secret")
        decoded = auth_mod.jwt_decode(token, "secret")
        self.assertIsNotNone(decoded)


class TestAuthManagerSessions(unittest.TestCase):
    """Test authentication, token creation, refresh, logout."""

    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "auth_sess.db")
        self.mgr = auth_mod.AuthManager(
            config_path=_auth_cfg_path, db_path=self.db
        )
        self.user = self.mgr.create_user("sessuser", "password123")

    def test_authenticate_success(self):
        u = self.mgr.authenticate("sessuser", "password123")
        self.assertIsNotNone(u)
        self.assertEqual(u.username, "sessuser")

    def test_authenticate_wrong_password(self):
        u = self.mgr.authenticate("sessuser", "wrongpassword")
        self.assertIsNone(u)

    def test_authenticate_wrong_username(self):
        u = self.mgr.authenticate("nobody", "password123")
        self.assertIsNone(u)

    def test_create_tokens(self):
        tokens = self.mgr.create_tokens(self.user)
        self.assertTrue(len(tokens.access_token) > 20)
        self.assertTrue(len(tokens.refresh_token) > 20)
        self.assertEqual(tokens.token_type, "bearer")
        self.assertEqual(tokens.user_id, self.user.user_id)

    def test_validate_access_token(self):
        tokens = self.mgr.create_tokens(self.user)
        payload = self.mgr.validate_token(tokens.access_token)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["sub"], self.user.user_id)
        self.assertEqual(payload["type"], "access")

    def test_refresh_tokens(self):
        tokens = self.mgr.create_tokens(self.user)
        new_tokens = self.mgr.refresh_tokens(tokens.refresh_token)
        self.assertIsNotNone(new_tokens)
        self.assertNotEqual(new_tokens.refresh_token, tokens.refresh_token)

    def test_refresh_invalidates_old(self):
        tokens = self.mgr.create_tokens(self.user)
        self.mgr.refresh_tokens(tokens.refresh_token)
        # Old refresh token should now be invalid
        second = self.mgr.refresh_tokens(tokens.refresh_token)
        self.assertIsNone(second)

    def test_logout(self):
        tokens = self.mgr.create_tokens(self.user)
        self.assertTrue(self.mgr.logout(tokens.refresh_token))
        # Cannot refresh after logout
        self.assertIsNone(self.mgr.refresh_tokens(tokens.refresh_token))

    def test_logout_all(self):
        t1 = self.mgr.create_tokens(self.user)
        t2 = self.mgr.create_tokens(self.user)
        count = self.mgr.logout_all(self.user.user_id)
        self.assertGreaterEqual(count, 2)

    def test_change_password_invalidates_sessions(self):
        tokens = self.mgr.create_tokens(self.user)
        ok = self.mgr.change_password(self.user.user_id, "newpassword123")
        self.assertTrue(ok)
        # Old refresh token should be invalid
        self.assertIsNone(self.mgr.refresh_tokens(tokens.refresh_token))
        # Can authenticate with new password
        u = self.mgr.authenticate("sessuser", "newpassword123")
        self.assertIsNotNone(u)

    def test_max_sessions_enforced(self):
        # Config has max_sessions=3
        t1 = self.mgr.create_tokens(self.user)
        t2 = self.mgr.create_tokens(self.user)
        t3 = self.mgr.create_tokens(self.user)
        t4 = self.mgr.create_tokens(self.user)  # Should deactivate oldest
        # t1's refresh should now be invalid (oldest deactivated)
        self.assertIsNone(self.mgr.refresh_tokens(t1.refresh_token))
        # t4 should still work
        self.assertIsNotNone(self.mgr.refresh_tokens(t4.refresh_token))


# ===========================================================================
# Part 2: Per-user isolation
# ===========================================================================


class TestMigration(unittest.TestCase):
    """Test per-user isolation migrations."""

    def test_migrate_adds_user_id_column(self):
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE conversations (id TEXT PRIMARY KEY, title TEXT)")
        conn.execute("INSERT INTO conversations VALUES ('c1', 'Test')")
        conn.commit()
        conn.close()

        result = iso_mod.migrate_table(db_path, "conversations")
        self.assertTrue(result)

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM conversations WHERE id='c1'").fetchone()
        self.assertEqual(row["user_id"], "local")
        conn.close()

    def test_migrate_idempotent(self):
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test2.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY)")
        conn.commit()
        conn.close()

        self.assertTrue(iso_mod.migrate_table(db_path, "memories"))
        self.assertFalse(iso_mod.migrate_table(db_path, "memories"))

    def test_migrate_nonexistent_db(self):
        self.assertFalse(iso_mod.migrate_table("/tmp/nonexistent.db", "t"))

    def test_migrate_nonexistent_table(self):
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "empty.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE other (id TEXT)")
        conn.commit()
        conn.close()
        self.assertFalse(iso_mod.migrate_table(db_path, "missing_table"))

    def test_run_migrations(self):
        tmpdir = tempfile.mkdtemp()
        # Create one target DB
        db_path = os.path.join(tmpdir, "conversations.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE conversations (id TEXT PRIMARY KEY)")
        conn.commit()
        conn.close()

        results = iso_mod.run_migrations(tmpdir)
        self.assertIn("conversations.db.conversations", results)
        self.assertTrue(results["conversations.db.conversations"])


class TestUserSettingsStore(unittest.TestCase):
    """Test per-user settings storage."""

    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "settings.db")
        self.store = iso_mod.UserSettingsStore(db_path=self.db)

    def test_get_defaults(self):
        s = self.store.get_settings("user-1")
        self.assertEqual(s.user_id, "user-1")
        self.assertEqual(s.theme, "dark")
        self.assertEqual(s.default_model, "")

    def test_update_settings(self):
        s = self.store.update_settings("user-1", theme="light", default_model="qwen3:32b")
        self.assertEqual(s.theme, "light")
        self.assertEqual(s.default_model, "qwen3:32b")

    def test_update_preferences_merged(self):
        self.store.update_settings("user-1", preferences={"a": 1})
        s = self.store.update_settings("user-1", preferences={"b": 2})
        self.assertEqual(s.preferences["a"], 1)
        self.assertEqual(s.preferences["b"], 2)

    def test_delete_settings(self):
        self.store.get_settings("user-2")  # Create
        self.assertTrue(self.store.delete_settings("user-2"))
        self.assertFalse(self.store.delete_settings("nonexistent"))

    def test_separate_users(self):
        self.store.update_settings("a", theme="light")
        self.store.update_settings("b", theme="dark")
        self.assertEqual(self.store.get_settings("a").theme, "light")
        self.assertEqual(self.store.get_settings("b").theme, "dark")


class TestQueryHelpers(unittest.TestCase):
    """Test SQL query helper functions."""

    def test_user_filter_sql_single_user(self):
        sql, params = iso_mod.user_filter_sql("user-1", single_user_mode=True)
        self.assertEqual(sql, "")
        self.assertEqual(params, [])

    def test_user_filter_sql_multi_user(self):
        sql, params = iso_mod.user_filter_sql("user-1", single_user_mode=False)
        self.assertIn("user_id = ?", sql)
        self.assertEqual(params, ["user-1"])

    def test_user_filter_sql_none_user(self):
        sql, params = iso_mod.user_filter_sql(None, single_user_mode=False)
        self.assertEqual(sql, "")

    def test_effective_user_id_single_user(self):
        self.assertEqual(iso_mod.effective_user_id(None), "local")
        self.assertEqual(iso_mod.effective_user_id("abc", True), "local")

    def test_effective_user_id_multi_user(self):
        self.assertEqual(iso_mod.effective_user_id("abc", False), "abc")


# ===========================================================================
# Part 3: Shared projects & RBAC
# ===========================================================================


class TestRBAC(unittest.TestCase):
    """Test shared projects, role hierarchy, audit log."""

    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "auth_rbac.db")
        self.mgr = auth_mod.AuthManager(
            config_path=_auth_cfg_path, db_path=self.db
        )
        self.owner = self.mgr.create_user("owner", "password123", role="admin")
        self.editor = self.mgr.create_user("editor", "password123")
        self.viewer = self.mgr.create_user("viewer", "password123")

    def test_share_project(self):
        result = self.mgr.share_project(
            "proj-1", self.editor.user_id, "editor", self.owner.user_id
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["role"], "editor")
        self.assertIn("invite_token", result)

    def test_check_permission_hierarchy(self):
        self.mgr.share_project("proj-2", self.editor.user_id, "editor")
        # Editor can view
        self.assertTrue(self.mgr.check_permission("proj-2", self.editor.user_id, "viewer"))
        # Editor can edit
        self.assertTrue(self.mgr.check_permission("proj-2", self.editor.user_id, "editor"))
        # Editor cannot own
        self.assertFalse(self.mgr.check_permission("proj-2", self.editor.user_id, "owner"))

    def test_no_access(self):
        self.assertFalse(self.mgr.check_permission("proj-3", self.viewer.user_id, "viewer"))

    def test_list_project_members(self):
        self.mgr.share_project("proj-4", self.editor.user_id, "editor")
        self.mgr.share_project("proj-4", self.viewer.user_id, "viewer")
        members = self.mgr.list_project_members("proj-4")
        self.assertEqual(len(members), 2)

    def test_list_user_shared_projects(self):
        self.mgr.share_project("proj-5", self.editor.user_id, "editor")
        self.mgr.share_project("proj-6", self.editor.user_id, "viewer")
        projects = self.mgr.list_user_shared_projects(self.editor.user_id)
        self.assertEqual(len(projects), 2)

    def test_remove_project_access(self):
        self.mgr.share_project("proj-7", self.editor.user_id, "editor")
        self.assertTrue(self.mgr.remove_project_access("proj-7", self.editor.user_id))
        self.assertIsNone(self.mgr.get_project_role("proj-7", self.editor.user_id))

    def test_audit_log(self):
        self.mgr.share_project(
            "proj-8", self.editor.user_id, "editor", self.owner.user_id
        )
        logs = self.mgr.get_audit_log(user_id=self.owner.user_id)
        self.assertGreaterEqual(len(logs), 1)
        self.assertEqual(logs[0]["action"], "share_project")

    def test_single_user_mode_property(self):
        # Our test config has single_user_mode=False
        self.assertFalse(self.mgr.single_user_mode)

    def test_invalid_role_rejected(self):
        result = self.mgr.share_project("proj-9", self.editor.user_id, "superadmin")
        self.assertIsNone(result)


# ===========================================================================
# Part 4: API routes (file structure, schemas)
# ===========================================================================


class TestRoutesAuthFile(unittest.TestCase):
    """Test routes_auth.py file structure and schemas."""

    @classmethod
    def setUpClass(cls):
        cls.src = _read(os.path.join(API_DIR, "routes_auth.py"))

    def test_file_exists(self):
        self.assertTrue(os.path.isfile(os.path.join(API_DIR, "routes_auth.py")))

    def test_router_prefix(self):
        self.assertIn('prefix="/api/auth"', self.src)

    def test_register_endpoint(self):
        self.assertIn("/register", self.src)
        self.assertIn("RegisterRequest", self.src)

    def test_login_endpoint(self):
        self.assertIn("/login", self.src)
        self.assertIn("LoginRequest", self.src)

    def test_logout_endpoint(self):
        self.assertIn("/logout", self.src)

    def test_refresh_endpoint(self):
        self.assertIn("/refresh", self.src)

    def test_me_endpoints(self):
        self.assertIn('"/me"', self.src)
        self.assertIn("ProfileUpdateRequest", self.src)

    def test_password_change_endpoint(self):
        self.assertIn("/me/password", self.src)
        self.assertIn("PasswordChangeRequest", self.src)

    def test_settings_endpoints(self):
        self.assertIn("/settings", self.src)
        self.assertIn("SettingsUpdateRequest", self.src)

    def test_users_admin_endpoint(self):
        self.assertIn("/users", self.src)
        self.assertIn("_require_admin", self.src)

    def test_share_project_endpoint(self):
        self.assertIn("/projects/share", self.src)
        self.assertIn("ShareProjectRequest", self.src)

    def test_audit_endpoint(self):
        self.assertIn("/audit", self.src)

    def test_status_endpoint(self):
        self.assertIn("/status", self.src)

    def test_single_user_bypass(self):
        self.assertIn("single_user_mode", self.src)
        self.assertIn('"local"', self.src)

    def test_bearer_token_parsing(self):
        self.assertIn("bearer", self.src.lower())
        self.assertIn("Authorization", self.src)


# ===========================================================================
# Part 5: Frontend files
# ===========================================================================


class TestFrontendTypes(unittest.TestCase):
    """Test auth types in types.ts."""

    @classmethod
    def setUpClass(cls):
        cls.src = _read(os.path.join(FRONTEND_SRC, "lib", "types.ts"))

    def test_auth_user_interface(self):
        self.assertIn("export interface AuthUser", self.src)
        self.assertIn("user_id: string", self.src)
        self.assertIn("username: string", self.src)

    def test_auth_tokens_interface(self):
        self.assertIn("export interface AuthTokens", self.src)
        self.assertIn("access_token: string", self.src)
        self.assertIn("refresh_token: string", self.src)

    def test_auth_status_interface(self):
        self.assertIn("export interface AuthStatus", self.src)
        self.assertIn("single_user_mode: boolean", self.src)

    def test_user_settings_interface(self):
        self.assertIn("export interface UserSettings", self.src)
        self.assertIn("default_model: string", self.src)

    def test_all_request_interfaces(self):
        for iface in ["RegisterRequest", "LoginRequest", "ProfileUpdateRequest",
                       "PasswordChangeRequest", "SettingsUpdateRequest",
                       "ShareProjectRequest"]:
            self.assertIn(f"export interface {iface}", self.src, f"Missing {iface}")

    def test_project_member_interface(self):
        self.assertIn("export interface ProjectMember", self.src)

    def test_audit_log_interface(self):
        self.assertIn("export interface AuditLogEntry", self.src)


class TestFrontendAuthAPI(unittest.TestCase):
    """Test auth.ts API client."""

    @classmethod
    def setUpClass(cls):
        cls.src = _read(os.path.join(API_TS_DIR, "auth.ts"))

    def test_file_exists(self):
        self.assertTrue(os.path.isfile(os.path.join(API_TS_DIR, "auth.ts")))

    def test_imports_client(self):
        self.assertIn("from './client'", self.src)

    def test_exports_all_functions(self):
        for fn in ["getAuthStatus", "register", "login", "refreshToken",
                    "logout", "getMe", "updateMe", "changePassword",
                    "getUserSettings", "updateUserSettings", "listUsers",
                    "deleteUser", "shareProject", "listProjectMembers",
                    "removeProjectMember", "getAuditLog"]:
            self.assertIn(f"export async function {fn}", self.src, f"Missing {fn}")

    def test_base_path(self):
        self.assertIn("'/api/auth'", self.src)


class TestFrontendAuthStore(unittest.TestCase):
    """Test auth store."""

    @classmethod
    def setUpClass(cls):
        cls.src = _read(os.path.join(STORES_DIR, "auth.ts"))

    def test_file_exists(self):
        self.assertTrue(os.path.isfile(os.path.join(STORES_DIR, "auth.ts")))

    def test_exports_stores(self):
        for store in ["currentUser", "authStatus", "userSettings",
                       "authLoading", "isAuthenticated", "isSingleUserMode"]:
            self.assertIn(store, self.src, f"Missing store {store}")

    def test_exports_actions(self):
        for fn in ["initAuth", "doRegister", "doLogin", "doLogout",
                    "doChangePassword", "updateSettings", "needsLogin"]:
            self.assertIn(fn, self.src, f"Missing action {fn}")

    def test_token_persistence(self):
        self.assertIn("localStorage", self.src)
        self.assertIn("oo-access-token", self.src)
        self.assertIn("oo-refresh-token", self.src)

    def test_setAccessToken_import(self):
        self.assertIn("setAccessToken", self.src)

    def test_single_user_synthetic(self):
        # In single-user mode, sets a synthetic local user
        self.assertIn("'local'", self.src)


class TestFrontendClientAuth(unittest.TestCase):
    """Test auth header injection in client.ts."""

    @classmethod
    def setUpClass(cls):
        cls.src = _read(os.path.join(API_TS_DIR, "client.ts"))

    def test_set_access_token_exported(self):
        self.assertIn("export function setAccessToken", self.src)

    def test_get_access_token_exported(self):
        self.assertIn("export function getAccessToken", self.src)

    def test_auth_headers_function(self):
        self.assertIn("function authHeaders()", self.src)

    def test_auth_injected_in_methods(self):
        # authHeaders() should appear in every fetch method
        self.assertGreaterEqual(self.src.count("...authHeaders()"), 5)


class TestFrontendPages(unittest.TestCase):
    """Test login and register page files."""

    def test_login_page_exists(self):
        path = os.path.join(ROUTES_DIR, "login", "+page.svelte")
        self.assertTrue(os.path.isfile(path))

    def test_register_page_exists(self):
        path = os.path.join(ROUTES_DIR, "register", "+page.svelte")
        self.assertTrue(os.path.isfile(path))

    def test_login_page_structure(self):
        src = _read(os.path.join(ROUTES_DIR, "login", "+page.svelte"))
        self.assertIn("doLogin", src)
        self.assertIn("username", src)
        self.assertIn("password", src)
        self.assertIn("/register", src)  # Link to register

    def test_register_page_structure(self):
        src = _read(os.path.join(ROUTES_DIR, "register", "+page.svelte"))
        self.assertIn("doRegister", src)
        self.assertIn("confirmPassword", src)
        self.assertIn("/login", src)  # Link to login

    def test_no_hardcoded_hex_login(self):
        src = _read(os.path.join(ROUTES_DIR, "login", "+page.svelte"))
        style = re.search(r"<style[^>]*>(.*?)</style>", src, re.DOTALL)
        if style:
            hexes = re.findall(r"#[0-9a-fA-F]{3,8}", style.group(1))
            self.assertEqual(hexes, [], f"Hardcoded hex in login: {hexes}")

    def test_no_hardcoded_hex_register(self):
        src = _read(os.path.join(ROUTES_DIR, "register", "+page.svelte"))
        style = re.search(r"<style[^>]*>(.*?)</style>", src, re.DOTALL)
        if style:
            hexes = re.findall(r"#[0-9a-fA-F]{3,8}", style.group(1))
            self.assertEqual(hexes, [], f"Hardcoded hex in register: {hexes}")


class TestUserMenuComponent(unittest.TestCase):
    """Test UserMenu.svelte component."""

    @classmethod
    def setUpClass(cls):
        cls.path = os.path.join(COMPONENTS_DIR, "ui", "UserMenu.svelte")
        cls.src = _read(cls.path)

    def test_file_exists(self):
        self.assertTrue(os.path.isfile(self.path))

    def test_imports_auth_store(self):
        self.assertIn("from '$lib/stores/auth'", self.src)

    def test_logout_handler(self):
        self.assertIn("doLogout", self.src)

    def test_hidden_in_single_user(self):
        self.assertIn("isSingleUserMode", self.src)

    def test_no_hardcoded_hex(self):
        style = re.search(r"<style[^>]*>(.*?)</style>", self.src, re.DOTALL)
        if style:
            hexes = re.findall(r"#[0-9a-fA-F]{3,8}", style.group(1))
            self.assertEqual(hexes, [], f"Hardcoded hex in UserMenu: {hexes}")


# ===========================================================================
# Part 6: Config
# ===========================================================================


class TestAuthConfig(unittest.TestCase):
    """Test auth.yaml configuration file."""

    @classmethod
    def setUpClass(cls):
        cls.path = os.path.join(CONFIG_DIR, "auth.yaml")
        with open(cls.path) as f:
            cls.cfg = yaml.safe_load(f)

    def test_file_exists(self):
        self.assertTrue(os.path.isfile(self.path))

    def test_jwt_section(self):
        self.assertIn("jwt", self.cfg)
        jwt = self.cfg["jwt"]
        self.assertIn("access_token_expiry_minutes", jwt)
        self.assertIn("refresh_token_expiry_days", jwt)
        self.assertIn("algorithm", jwt)

    def test_password_section(self):
        self.assertIn("password", self.cfg)
        pw = self.cfg["password"]
        self.assertIn("min_length", pw)
        self.assertIn("bcrypt_rounds", pw)

    def test_users_section(self):
        self.assertIn("users", self.cfg)
        self.assertIn("allow_registration", self.cfg["users"])
        self.assertIn("default_role", self.cfg["users"])

    def test_session_section(self):
        self.assertIn("session", self.cfg)
        self.assertIn("max_sessions", self.cfg["session"])

    def test_single_user_mode(self):
        self.assertIn("single_user_mode", self.cfg)
        self.assertTrue(self.cfg["single_user_mode"])  # Default is True

    def test_db_path(self):
        self.assertIn("db_path", self.cfg)


# ===========================================================================
# Part 7: Integration wiring
# ===========================================================================


class TestIntegrationWiring(unittest.TestCase):
    """Test deps.py, app.py wiring and version bump."""

    @classmethod
    def setUpClass(cls):
        cls.deps_src = _read(os.path.join(API_DIR, "deps.py"))
        cls.app_src = _read(os.path.join(API_DIR, "app.py"))

    def test_deps_auth_available(self):
        self.assertIn("AUTH_AVAILABLE", self.deps_src)
        self.assertIn("auth_manager", self.deps_src)

    def test_deps_user_settings(self):
        self.assertIn("USER_SETTINGS_AVAILABLE", self.deps_src)
        self.assertIn("user_settings_store", self.deps_src)

    def test_app_imports_auth_router(self):
        self.assertIn("from .routes_auth import router as auth_router", self.app_src)

    def test_app_registers_auth_router(self):
        self.assertIn("app.include_router(auth_router)", self.app_src)

    def test_app_version_bumped(self):
        self.assertIn('version="1.10.3"', self.app_src)

    def test_health_version_bumped(self):
        self.assertIn('"version": "1.10.3"', self.app_src)

    def test_health_auth_module(self):
        self.assertIn('"auth": AUTH_AVAILABLE', self.app_src)
        self.assertIn('"user_settings": USER_SETTINGS_AVAILABLE', self.app_src)

    def test_pyproject_version(self):
        pyproject = _read(os.path.join(PROJECT_ROOT, "pyproject.toml"))
        self.assertIn('version = "1.10.3"', pyproject)

    def test_setup_version(self):
        setup = _read(os.path.join(PROJECT_ROOT, "setup.py"))
        self.assertIn('version="1.10.3"', setup)

    def test_pyproject_auth_dependency(self):
        pyproject = _read(os.path.join(PROJECT_ROOT, "pyproject.toml"))
        self.assertIn("bcrypt", pyproject)

    def test_layout_auth_guard(self):
        layout = _read(os.path.join(ROUTES_DIR, "+layout.svelte"))
        self.assertIn("initAuth", layout)
        self.assertIn("authLoading", layout)
        self.assertIn("PUBLIC_ROUTES", layout)

    def test_appshell_user_menu(self):
        appshell = _read(
            os.path.join(COMPONENTS_DIR, "layout", "AppShell.svelte")
        )
        self.assertIn("UserMenu", appshell)
        self.assertIn("<UserMenu", appshell)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main()
