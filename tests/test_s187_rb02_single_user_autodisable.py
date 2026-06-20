"""S187 auth fix RB-02 -- single_user_mode auto-disables once a second user exists.

Before the fix, AuthManager.single_user_mode returned the static config flag, so an
operator who provisioned multiple users but left the default ``single_user_mode:
true`` kept a full authentication bypass in Daily (the auth middleware and the RBAC
dependency skip authentication when single_user_mode is on and Bulbe is not active).

The fix makes the property effective and fail-safe: an explicit opt-out wins, it
auto-disables (and latches off for the process lifetime) once ``count_users() > 1``,
and an undeterminable user count is treated as multi-user (authentication required).

The module is loaded in isolation via spec_from_file_location with a stubbed
opti_oignon.db_utils whose safe_connect returns a real on-disk sqlite connection
(Daily/plaintext is sufficient here); the optional secure_bytes/bcrypt imports
degrade gracefully against the bare opti_oignon package.
"""

import importlib.util
import sqlite3
import sys
import types
from pathlib import Path

import yaml

# Bare package + a db_utils stub whose safe_connect is a plain sqlite3 connect.
sys.modules.setdefault("opti_oignon", types.ModuleType("opti_oignon"))

_db_utils_stub = types.ModuleType("opti_oignon.db_utils")


def _safe_connect(db_path, *, check_same_thread: bool = True, timeout: float = 5.0):
    return sqlite3.connect(
        str(db_path), check_same_thread=check_same_thread, timeout=timeout
    )


_db_utils_stub.safe_connect = _safe_connect
sys.modules["opti_oignon.db_utils"] = _db_utils_stub

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PATH = _REPO_ROOT / "opti_oignon" / "auth.py"


def _load():
    spec = importlib.util.spec_from_file_location("auth_rb02", _PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # register before exec (3.12+ dataclass ordering)
    spec.loader.exec_module(mod)
    return mod


auth = _load()


def _mgr(tmp_path, *, single_user_mode: bool = True):
    cfg_path = tmp_path / "auth.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "single_user_mode": single_user_mode,
                "jwt": {"secret_key": "x" * 64, "algorithm": "HS512"},
                "password": {"min_length": 8, "bcrypt_rounds": 4},
            }
        ),
        encoding="utf-8",
    )
    return auth.AuthManager(config_path=cfg_path, db_path=tmp_path / "auth.db")


def _add_user(mgr, name: str):
    user = mgr.create_user(name, "password123")
    assert user is not None, f"failed to create user {name!r}"
    return user


# ---------------------------------------------------------------------------
# Single-user install is unaffected (0 or 1 user keeps the bypass on)
# ---------------------------------------------------------------------------

def test_zero_users_keeps_single_user_mode(tmp_path):
    mgr = _mgr(tmp_path, single_user_mode=True)
    assert mgr.count_users() == 0
    assert mgr.single_user_mode is True


def test_one_user_keeps_single_user_mode(tmp_path):
    mgr = _mgr(tmp_path, single_user_mode=True)
    _add_user(mgr, "alice")
    assert mgr.count_users() == 1
    assert mgr.single_user_mode is True


# ---------------------------------------------------------------------------
# A second user auto-disables single-user mode
# ---------------------------------------------------------------------------

def test_second_user_disables_single_user_mode(tmp_path):
    mgr = _mgr(tmp_path, single_user_mode=True)
    _add_user(mgr, "alice")
    _add_user(mgr, "bob")
    assert mgr.count_users() == 2
    assert mgr.single_user_mode is False


# ---------------------------------------------------------------------------
# Once observed, the off state latches (dropping back to one user does not
# silently re-enable the bypass within the process)
# ---------------------------------------------------------------------------

def test_latch_stays_off_after_dropping_back_to_one_user(tmp_path):
    mgr = _mgr(tmp_path, single_user_mode=True)
    _add_user(mgr, "alice")
    bob = _add_user(mgr, "bob")
    # Reading the property at count == 2 latches the off state.
    assert mgr.single_user_mode is False
    assert mgr.delete_user(bob.user_id) is True
    assert mgr.count_users() == 1
    # Still off: the latch is one-way.
    assert mgr.single_user_mode is False


# ---------------------------------------------------------------------------
# Explicit opt-out always wins and short-circuits the user count
# ---------------------------------------------------------------------------

def test_explicit_opt_out_wins_regardless_of_count(tmp_path):
    mgr = _mgr(tmp_path, single_user_mode=False)
    assert mgr.single_user_mode is False
    _add_user(mgr, "alice")
    assert mgr.single_user_mode is False


# ---------------------------------------------------------------------------
# Fail safe: an undeterminable user count is treated as multi-user
# ---------------------------------------------------------------------------

def test_undeterminable_count_fails_safe(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path, single_user_mode=True)

    def _boom():
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(mgr, "count_users", _boom)
    assert mgr.single_user_mode is False


# ---------------------------------------------------------------------------
# Source guard: the property is derived + fail-safe, not the old static return
# ---------------------------------------------------------------------------

def test_property_source_is_derived_and_fail_safe():
    src = _PATH.read_text(encoding="utf-8")
    start = src.index("def single_user_mode(")
    end = src.index("def _row_to_user(", start)
    body = src[start:end]
    assert "self.count_users()" in body
    assert "_multi_user_latched" in body
    assert "except Exception:" in body
    # The old one-line static return must be gone.
    assert 'return self.config.get("single_user_mode", True)' not in body
