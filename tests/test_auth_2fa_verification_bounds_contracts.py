#!/usr/bin/env python3
"""Two-factor verification contracts: bounded, replay-proof, closed.

Every verification path in the two-factor module is bounded and fails
closed: a consumed one-time code never validates twice, attempt counters
short-circuit before storage is touched and re-arm only when their window
elapses, an unverified or unknown enrollment is refused, an internal
storage failure yields a refusal rather than an exception, and the
hardened mode makes the second factor mandatory before any status lookup.
This suite pins that behavior:

  * VB1 -- a TOTP code is single-use inside its validity window; the next
    time-step still validates, so the protection is replay-scoped;
  * VB2 -- the TOTP attempt cap refuses the next attempt without touching
    storage and re-arms after the documented window;
  * VB3 -- an unverified enrollment and an unknown user are refused even
    with a correct code;
  * VB4 -- an internal storage failure during validation is a closed
    verdict, never a propagated exception;
  * VB5 -- a recovery code is consumed on first use and the use is
    audited; the remaining codes stay valid;
  * VB6 -- the recovery attempt cap refuses the next attempt without
    touching storage and re-arms after an hour;
  * VB7 -- the hardened mode makes the second factor mandatory without
    consulting stored status; the daily mode without enrollment does not.

Loads the two-factor module in isolation under a stand-in package; every
``opti_oignon.*`` entry plus the model client and TOTP library entries are
snapshotted and evicted first, and only deterministic stubs are seeded: a
recording ``safe_connect`` that routes each database path to a shared
in-memory store and journals every statement, an encryption module with
flippable manager/key state, a security-mode module, and a deterministic
TOTP library. A meta-path guard refuses any project submodule that was not
seeded, so the load behaves identically whether or not the project is
installed. The module clock is replaced by a controllable clock shared
with the TOTP stub. Local-only. Runs under pytest or the __main__ runner.
"""

import base64
import hashlib
import hmac
import importlib.util
import sqlite3
import sys
import types
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


class _IsolationGuard:
    """Refuse every project submodule the test did not seed.

    A stand-in package whose ``__path__`` is empty isolates the tree only
    while the parent path is the sole way to resolve a submodule. That
    assumption breaks wherever the project is installed in editable mode:
    such an install registers a finder that answers on the module NAME and
    ignores the parent path, so a real submodule resolves behind the test's
    back -- silently importing live code and reopening real databases. This
    guard sits ahead of every finder and refuses the names that were not
    seeded, so a load behaves identically whether the project is installed
    or not.
    """

    _PREFIX = "opti_oignon."

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(self._PREFIX):
            raise ModuleNotFoundError(
                f"not seeded in the isolation window: {fullname}",
                name=fullname,
            )
        return None


class _Clock:
    """Deterministic clock shared by the module under test and the stub."""

    def __init__(self, start=1_700_000_000.0):
        self.now = float(start)

    def time(self):
        return self.now

    def advance(self, seconds):
        self.now += float(seconds)


class _RecordingConnection:
    """Journal every statement against a shared in-memory store.

    The module opens and closes a connection per operation; ``close`` is
    absorbed so the shared store survives across those cycles and the
    recorded state stays observable.
    """

    def __init__(self, real, state):
        self._real = real
        self._state = state

    @property
    def row_factory(self):
        return self._real.row_factory

    @row_factory.setter
    def row_factory(self, value):
        self._real.row_factory = value

    def execute(self, sql, params=()):
        if self._state.fail_marker and self._state.fail_marker in sql:
            raise sqlite3.OperationalError("recorder-injected failure")
        self._state.statements.append((sql, tuple(params)))
        return self._real.execute(sql, params)

    def executescript(self, script):
        self._state.statements.append(("<script>", (script,)))
        return self._real.executescript(script)

    def commit(self):
        return self._real.commit()

    def close(self):
        self._state.closes += 1


class _AtRestRecorder:
    """Recording stand-in for ``safe_connect``.

    Routes each database path to one shared in-memory store, journals every
    (sql, params) pair, and can inject a storage failure on demand.
    """

    def __init__(self):
        self.statements = []
        self.paths = []
        self.closes = 0
        self.fail_marker = None
        self._stores = {}

    def connect(self, db_path, **kwargs):
        key = str(db_path)
        self.paths.append(key)
        real = self._stores.get(key)
        if real is None:
            real = sqlite3.connect(":memory:", check_same_thread=False)
            self._stores[key] = real
        return _RecordingConnection(real, self)

    def reset_log(self):
        self.statements.clear()

    def close_all(self):
        for real in self._stores.values():
            try:
                real.close()
            except Exception:
                pass
        self._stores.clear()


def _totp_code(secret, step):
    """Deterministic six-digit code for a (secret, time-step) pair."""
    digest = hashlib.sha256(f"{secret}:{step}".encode("utf-8")).hexdigest()
    return str(int(digest[:8], 16) % 1_000_000).zfill(6)


def _pyotp_stub(clock):
    """Deterministic TOTP library keyed to the shared clock."""
    module = types.ModuleType("pyotp")

    class _TOTP:
        def __init__(self, secret, *args, **kwargs):
            self.secret = secret
            self.interval = 30

        def at(self, for_time):
            return _totp_code(self.secret, int(for_time // self.interval))

        def verify(self, code, valid_window=0):
            current = int(clock.time() // self.interval)
            for offset in range(-valid_window, valid_window + 1):
                candidate = _totp_code(self.secret, current + offset)
                if hmac.compare_digest(candidate, str(code)):
                    return True
            return False

        def provisioning_uri(self, name="", issuer_name=""):
            return f"otpauth://totp/{issuer_name}:{name}"

    module.TOTP = _TOTP
    module.random_base32 = lambda: "TESTSEEDBASE32AB"
    return module


class _EncryptionState:
    """Flippable state behind the seeded encryption module."""

    def __init__(self, master_key=b"\x11" * 32, enabled=True, has_key=True):
        self.master_key = master_key
        self.enabled = enabled
        self.has_key = has_key


def _encryption_seed(state):
    module = types.ModuleType("opti_oignon.encryption")

    class _Manager:
        @property
        def enabled(self):
            return state.enabled

        @property
        def has_key(self):
            return state.has_key

        def encrypt(self, plaintext):
            payload = base64.b64encode(plaintext.encode("utf-8")).decode()
            return "ENC:" + payload

        def decrypt(self, ciphertext):
            if not ciphertext.startswith("ENC:"):
                raise ValueError("not a stub ciphertext")
            return base64.b64decode(ciphertext[4:]).decode("utf-8")

    module.EncryptionManager = _Manager
    module.get_encryption_key = lambda: state.master_key
    module.get_encryption_status = lambda: {
        "key_available": state.master_key is not None,
        "kdf": "test-kdf",
    }
    return module


class _ModeState:
    """Flippable state behind the seeded security-mode module."""

    def __init__(self, bulbe=False):
        self.bulbe = bulbe
        self.audit = []


def _security_mode_seed(state):
    module = types.ModuleType("opti_oignon.security_mode")
    module.is_bulbe = lambda: state.bulbe

    def _audit_log(event, **kwargs):
        state.audit.append((event, kwargs))

    module._audit_log = _audit_log
    return module


def _load(master_key=b"\x11" * 32, enabled=True, has_key=True, bulbe=False):
    """Load auth_2fa.py under a stand-in package with recording stubs."""
    keys = ["ollama", "pyotp"] + [
        k
        for k in list(sys.modules)
        if k == "opti_oignon" or k.startswith("opti_oignon.")
    ]
    saved = {k: sys.modules[k] for k in keys if k in sys.modules}
    for k in keys:
        sys.modules.pop(k, None)
    sys.modules["ollama"] = None  # no client import exists; drift fails loud

    clock = _Clock()
    sys.modules["pyotp"] = _pyotp_stub(clock)

    recorder = _AtRestRecorder()
    enc_state = _EncryptionState(
        master_key=master_key, enabled=enabled, has_key=has_key
    )
    mode_state = _ModeState(bulbe=bulbe)

    root = types.ModuleType("opti_oignon")
    root.__path__ = []
    sys.modules["opti_oignon"] = root
    db_utils = types.ModuleType("opti_oignon.db_utils")
    db_utils.safe_connect = recorder.connect
    seeds = {
        "db_utils": db_utils,
        "encryption": _encryption_seed(enc_state),
        "security_mode": _security_mode_seed(mode_state),
    }
    for name, module in seeds.items():
        sys.modules[f"opti_oignon.{name}"] = module
        setattr(root, name, module)

    guard = _IsolationGuard()
    sys.meta_path.insert(0, guard)

    def restore():
        try:
            sys.meta_path.remove(guard)
        except ValueError:
            pass
        for k in list(sys.modules):
            if k == "opti_oignon" or k.startswith("opti_oignon."):
                del sys.modules[k]
        sys.modules.pop("ollama", None)
        sys.modules.pop("pyotp", None)
        for k, v in saved.items():
            sys.modules[k] = v
        recorder.close_all()

    full = "opti_oignon.auth_2fa"
    spec = importlib.util.spec_from_file_location(full, _OO / "auth_2fa.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    root.auth_2fa = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        restore()
        raise

    mod.time = SimpleNamespace(time=clock.time)
    return SimpleNamespace(
        mod=mod,
        recorder=recorder,
        enc=enc_state,
        mode=mode_state,
        clock=clock,
        restore=restore,
    )


def _activate_totp(ctx, user):
    """Enroll and verify a TOTP seed; return the seed."""
    setup = ctx.mod.two_factor_manager.totp_setup(user)
    assert setup["success"], f"setup must succeed, got {setup}"
    seed = setup["secret"]
    step = int(ctx.clock.time() // 30)
    verified = ctx.mod.two_factor_manager.totp_verify(
        user, _totp_code(seed, step)
    )
    assert verified["success"], f"activation must succeed, got {verified}"
    return seed


# ---------------------------------------------------------------------------
# VB1 -- a TOTP code is single-use inside its validity window
# ---------------------------------------------------------------------------
def test_vb1_totp_code_is_single_use_within_its_window():
    ctx = _load()
    try:
        mgr = ctx.mod.two_factor_manager
        seed = _activate_totp(ctx, "uid-vb1")
        step = int(ctx.clock.time() // 30)
        code = _totp_code(seed, step)
        assert mgr.totp_validate("uid-vb1", code) is True, (
            "the first use of a live code must validate"
        )
        assert mgr.totp_validate("uid-vb1", code) is False, (
            "a captured live code must be refused on replay inside its "
            "validity window"
        )
        ctx.clock.advance(30)
        next_code = _totp_code(seed, step + 1)
        assert mgr.totp_validate("uid-vb1", next_code) is True, (
            "the next time-step must still validate: the protection is "
            "replay-scoped, not a lockout"
        )
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# VB2 -- the TOTP attempt cap short-circuits storage and re-arms
# ---------------------------------------------------------------------------
def test_vb2_totp_attempt_cap_refuses_without_touching_storage():
    ctx = _load()
    try:
        mgr = ctx.mod.two_factor_manager
        assert ctx.mod.MAX_TOTP_ATTEMPTS_PER_WINDOW == 5
        assert ctx.mod.TOTP_WINDOW_SECONDS == 300
        for _ in range(5):
            assert mgr.totp_validate("uid-vb2", "000000") is False
        ctx.recorder.reset_log()
        assert mgr.totp_validate("uid-vb2", "000000") is False, (
            "the attempt past the cap must be refused"
        )
        assert ctx.recorder.statements == [], (
            "the capped refusal must not touch storage at all, got "
            f"{ctx.recorder.statements[:2]}"
        )
        ctx.clock.advance(ctx.mod.TOTP_WINDOW_SECONDS + 1)
        ctx.recorder.reset_log()
        assert mgr.totp_validate("uid-vb2", "000000") is False
        assert ctx.recorder.statements, (
            "after the window elapses the counter must re-arm and the "
            "lookup must reach storage again"
        )
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# VB3 -- unverified enrollment and unknown user are refused
# ---------------------------------------------------------------------------
def test_vb3_unverified_or_unknown_enrollment_is_refused():
    ctx = _load()
    try:
        mgr = ctx.mod.two_factor_manager
        setup = mgr.totp_setup("uid-vb3")
        seed = setup["secret"]
        step = int(ctx.clock.time() // 30)
        code = _totp_code(seed, step)
        assert mgr.totp_validate("uid-vb3", code) is False, (
            "a correct code against an unverified enrollment must be "
            "refused"
        )
        assert mgr.totp_validate("uid-ghost", code) is False, (
            "an unknown user must be refused"
        )
        verified = mgr.totp_verify("uid-vb3", code)
        assert verified["success"]
        assert mgr.totp_validate("uid-vb3", code) is True, (
            "the same code must validate once the enrollment is verified, "
            "proving the earlier refusal was the verification flag"
        )
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# VB4 -- an internal storage failure is a closed verdict
# ---------------------------------------------------------------------------
def test_vb4_internal_storage_failure_is_a_closed_verdict():
    ctx = _load()
    try:
        mgr = ctx.mod.two_factor_manager
        seed = _activate_totp(ctx, "uid-vb4")
        ctx.clock.advance(30)
        step = int(ctx.clock.time() // 30)
        code = _totp_code(seed, step)
        ctx.recorder.fail_marker = "SELECT secret_encrypted"
        verdict = mgr.totp_validate("uid-vb4", code)
        assert verdict is False, (
            "a storage failure during validation must yield a refusal, "
            f"got {verdict!r}"
        )
        ctx.recorder.fail_marker = None
        assert mgr.totp_validate("uid-vb4", code) is True, (
            "the control run without the injected failure must validate, "
            "proving the closed verdict came from the failure"
        )
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# VB5 -- a recovery code is consumed on first use, and audited
# ---------------------------------------------------------------------------
def test_vb5_recovery_code_is_consumed_on_first_use():
    ctx = _load()
    try:
        mgr = ctx.mod.two_factor_manager
        codes = mgr.generate_recovery_codes("uid-vb5")
        assert mgr.validate_recovery_code("uid-vb5", codes[0]) is True
        consumed = [
            sql
            for sql, _ in ctx.recorder.statements
            if "UPDATE recovery_codes" in sql and "used = 1" in sql
        ]
        assert consumed, "the matched code must be marked consumed at rest"
        assert mgr.validate_recovery_code("uid-vb5", codes[0]) is False, (
            "a consumed one-time code must never validate again"
        )
        assert mgr.validate_recovery_code("uid-vb5", codes[1]) is True, (
            "the remaining codes must stay valid"
        )
        used_events = [e for e, _ in ctx.mode.audit if "recovery" in e]
        assert len(used_events) >= 2, (
            "each successful recovery use must be audited, got "
            f"{ctx.mode.audit}"
        )
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# VB6 -- the recovery attempt cap short-circuits storage and re-arms
# ---------------------------------------------------------------------------
def test_vb6_recovery_attempt_cap_refuses_then_rearms_after_an_hour():
    ctx = _load()
    try:
        mgr = ctx.mod.two_factor_manager
        assert ctx.mod.MAX_RECOVERY_ATTEMPTS_PER_HOUR == 3
        codes = mgr.generate_recovery_codes("uid-vb6")
        for _ in range(3):
            assert mgr.validate_recovery_code("uid-vb6", "f" * 16) is False
        ctx.recorder.reset_log()
        assert mgr.validate_recovery_code("uid-vb6", codes[0]) is False, (
            "past the cap even a correct code must be refused"
        )
        assert ctx.recorder.statements == [], (
            "the capped refusal must not touch storage at all"
        )
        ctx.clock.advance(3601)
        assert mgr.validate_recovery_code("uid-vb6", codes[0]) is True, (
            "after an hour the counter must re-arm and the correct code "
            "must validate"
        )
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# VB7 -- the hardened mode makes the second factor mandatory
# ---------------------------------------------------------------------------
def test_vb7_hardened_mode_makes_second_factor_mandatory():
    ctx = _load(bulbe=True)
    try:
        mgr = ctx.mod.two_factor_manager
        ctx.recorder.reset_log()
        assert mgr.is_2fa_required("uid-nobody") is True, (
            "the hardened mode must require the second factor for every "
            "user, enrolled or not"
        )
        assert ctx.recorder.statements == [], (
            "the mandatory verdict must come from the mode, not from a "
            "status lookup"
        )
    finally:
        ctx.restore()

    daily = _load(bulbe=False)
    try:
        assert daily.mod.two_factor_manager.is_2fa_required(
            "uid-nobody"
        ) is False, (
            "the daily mode without enrollment must not require the "
            "second factor"
        )
    finally:
        daily.restore()


if __name__ == "__main__":
    _failures = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except Exception as _e:  # noqa: BLE001
                _failures += 1
                print(f"FAIL {_name}: {_e!r}")
    print(f"\n{'OK' if _failures == 0 else str(_failures) + ' FAILED'}")
    sys.exit(1 if _failures else 0)
