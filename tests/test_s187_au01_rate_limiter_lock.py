"""S187 auth fix AU-01 -- LoginRateLimiter is thread-safe under a single lock.

Before the fix, LoginRateLimiter mutated the per-IP / per-username maps and the
per-entry attempt lists -- including the read-modify-write in _clean_window
(``entry.attempts = [...]``) -- with no lock, despite a docstring claiming
thread-safety. Concurrent logins could lose attempt records or interleave
check/record, weakening brute-force protection.

The fix adds one threading.Lock guarding both maps and their attempt lists across
check_rate_limit, record_failure, record_success and get_status.

The concurrency test widens the read-modify-write window inside _clean_window
(snapshot + brief sleep + write-back) so that, without the lock, a concurrent
append landing during a clean would be dropped; under the lock no increment is
lost (barrier + counting, like the IB-02 test). A plain list.append alone is hard
to race under the GIL, hence the widened window.

The module is loaded in isolation via spec_from_file_location with a stubbed
opti_oignon.db_utils (only needed so auth.py imports; the limiter itself uses no
DB).
"""

import importlib.util
import sqlite3
import sys
import threading
import time
import types
from pathlib import Path

# Bare package + a db_utils stub so auth.py imports (the limiter needs no DB).
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
    spec = importlib.util.spec_from_file_location("auth_au01", _PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # register before exec (3.12+ dataclass ordering)
    spec.loader.exec_module(mod)
    return mod


auth = _load()


def _make_slow_clean(limiter):
    """A _clean_window that widens the read-modify-write window.

    Mirrors the real filter (cutoff = now - window) but snapshots first and
    sleeps, so without the lock a concurrent append would be overwritten.
    """

    def _slow_clean(entry, now):
        snapshot = list(entry.attempts)
        time.sleep(0.01)
        cutoff = now - limiter.window_seconds
        entry.attempts = [t for t in snapshot if t > cutoff]

    return _slow_clean


# ---------------------------------------------------------------------------
# Core property: concurrent record/check loses no increments
# ---------------------------------------------------------------------------

def test_concurrent_record_failure_loses_no_increments(monkeypatch):
    limiter = auth.LoginRateLimiter(
        config={
            "enabled": True,
            "login_max_attempts": 10_000_000,
            "account_lock_threshold": 10_000_000,
            "login_window_seconds": 10_000,
        }
    )
    # Widen the clean window so a missing lock would drop concurrent appends.
    monkeypatch.setattr(limiter, "_clean_window", _make_slow_clean(limiter))

    ip, username = "10.0.0.1", "victim"
    n = 24
    barrier = threading.Barrier(n)
    errors: list = []

    def worker():
        try:
            barrier.wait(timeout=5)
            limiter.record_failure(ip, username)   # append (locked)
            limiter.check_rate_limit(ip, username)  # clean = read-modify-write (locked)
        except Exception as exc:  # pragma: no cover - surfaced via assert
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert all(not t.is_alive() for t in threads), "rate limiter deadlocked"
    assert not errors, errors
    assert len(limiter._user_entries[username].attempts) == n
    assert len(limiter._ip_entries[ip].attempts) == n


# ---------------------------------------------------------------------------
# Behaviour preserved: lockout still triggers, record_success still resets
# ---------------------------------------------------------------------------

def test_lockout_still_triggers_after_max_attempts():
    limiter = auth.LoginRateLimiter(
        config={
            "enabled": True,
            "login_max_attempts": 3,
            "lockout_base_seconds": 60,
            "login_window_seconds": 300,
            "account_lock_threshold": 100,
        }
    )
    ip, username = "1.2.3.4", "u"
    for _ in range(3):
        limiter.record_failure(ip, username)
    allowed, retry = limiter.check_rate_limit(ip, username)
    assert allowed is False
    assert retry > 0


def test_record_success_resets_username_under_lock():
    limiter = auth.LoginRateLimiter(config={"enabled": True})
    limiter.record_failure("ip", "alice")
    assert "alice" in limiter._user_entries
    limiter.record_success("ip", "alice")
    assert "alice" not in limiter._user_entries


# ---------------------------------------------------------------------------
# The lock exists and the docstring no longer overclaims
# ---------------------------------------------------------------------------

def test_limiter_has_a_lock():
    limiter = auth.LoginRateLimiter(config={"enabled": True})
    assert hasattr(limiter, "_lock")
    assert hasattr(limiter._lock, "acquire") and hasattr(limiter._lock, "release")


def test_docstring_no_longer_claims_time_based_only():
    doc = auth.LoginRateLimiter.__doc__ or ""
    assert "lock" in doc.lower()
    assert "time-based expiry" not in doc


# ---------------------------------------------------------------------------
# Source guard: every mutating/reading method takes the lock
# ---------------------------------------------------------------------------

def test_methods_take_the_lock_source():
    src = _PATH.read_text(encoding="utf-8")
    for marker in (
        "def check_rate_limit(",
        "def record_failure(",
        "def record_success(",
        "def get_status(",
    ):
        start = src.index(marker)
        try:
            end = src.index("\n    def ", start + 1)
        except ValueError:
            end = len(src)  # last method in the class
        body = src[start:end]
        assert "with self._lock:" in body, marker
