#!/usr/bin/env python3
"""Two-factor secret at-rest contracts: nothing sensitive is stored bare.

The two-factor module keeps every credential artifact behind the encrypted
connection helper and never persists a plaintext secret: the TOTP seed is
written through the project encryption manager (or, degraded, behind an
explicit non-bare wrapper while the database layer carries the at-rest
protection), recovery codes and app passwords are persisted only as keyed
hashes derived from the master key on a dedicated domain, and a keyed hash
that cannot be verified is refused rather than guessed. This suite pins
that behavior:

  * SA1 -- the TOTP seed written at rest is the encryption manager output,
    never the plaintext seed, and the plaintext appears in no stored value;
  * SA2 -- with the manager unavailable the stored value is the explicit
    non-bare wrapper, never the bare seed, and the write flows through the
    seeded encrypted-connection seam;
  * SA3 -- recovery codes persist only as keyed hashes under the
    master-key scheme; the ten plaintext codes carry the documented
    length and never reach a statement or a stored value;
  * SA4 -- an app password persists only as its keyed hash; the plaintext
    is returned once and never stored;
  * SA5 -- every statement is parameterized: hostile-looking values appear
    only in bind parameters, never in SQL text;
  * SA6 -- a master-key hash with no master key available verifies closed.

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
    (sql, params) pair, and can inject a storage failure on demand. The
    journal is the at-rest observation surface of these contracts.
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

    def sql_texts(self):
        return [sql for sql, _ in self.statements]

    def bound_values(self):
        flat = []
        for _, params in self.statements:
            flat.extend(str(value) for value in params)
        return flat


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
    """Load auth_2fa.py under a stand-in package with recording stubs.

    Returns a context namespace carrying the module, the at-rest recorder,
    the encryption and mode states, the shared clock, and ``restore``.
    """
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


def _single_insert(recorder, table):
    """Return the (sql, params) of the single INSERT into ``table``."""
    hits = [
        (sql, params)
        for sql, params in recorder.statements
        if "INSERT" in sql and table in sql
    ]
    assert len(hits) == 1, f"expected one INSERT into {table}, got {len(hits)}"
    return hits[0]


# ---------------------------------------------------------------------------
# SA1 -- the TOTP seed at rest is manager output, never the plaintext
# ---------------------------------------------------------------------------
def test_sa1_totp_seed_is_encrypted_at_rest():
    ctx = _load()
    try:
        result = ctx.mod.two_factor_manager.totp_setup("uid-sa1")
        assert result["success"], f"setup must succeed, got {result}"
        seed = result["secret"]
        sql, params = _single_insert(ctx.recorder, "totp_config")
        stored = params[1]
        assert stored.startswith("ENC:"), (
            "with the manager enabled the stored seed must be manager "
            f"output, got {stored!r}"
        )
        assert stored != seed, "the stored value must not be the seed"
        assert seed not in sql, "the seed must never reach SQL text"
        assert seed not in ctx.recorder.bound_values(), (
            "the plaintext seed must appear in no stored value"
        )
        assert ctx.mod._decrypt_secret(stored) == seed, (
            "the stored value must round-trip through the manager"
        )
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# SA2 -- degraded storage stays non-bare and flows through the seam
# ---------------------------------------------------------------------------
def test_sa2_degraded_storage_is_wrapped_and_uses_the_seam():
    ctx = _load(enabled=False, has_key=False)
    try:
        result = ctx.mod.two_factor_manager.totp_setup("uid-sa2")
        assert result["success"], f"setup must succeed, got {result}"
        seed = result["secret"]
        sql, params = _single_insert(ctx.recorder, "totp_config")
        stored = params[1]
        assert stored.startswith("B64:"), (
            "with the manager unavailable the stored value must carry the "
            f"explicit non-bare wrapper, got {stored!r}"
        )
        assert stored != seed, "the bare seed must never be stored"
        assert base64.b64decode(stored[4:]).decode("utf-8") == seed, (
            "the wrapper must be the documented reversible degradation"
        )
        assert str(ctx.mod._2FA_DB_PATH) in ctx.recorder.paths, (
            "at-rest access must flow through the seeded encrypted-"
            "connection seam"
        )
        assert seed not in sql
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# SA3 -- recovery codes persist only as keyed hashes
# ---------------------------------------------------------------------------
def test_sa3_recovery_codes_persist_only_keyed_hashes():
    ctx = _load()
    try:
        mgr = ctx.mod.two_factor_manager
        codes = mgr.generate_recovery_codes("uid-sa3")
        assert len(codes) == ctx.mod.RECOVERY_CODE_COUNT == 10, (
            f"the documented code count must hold, got {len(codes)}"
        )
        hexdigits = set("0123456789abcdef")
        for code in codes:
            assert len(code) == ctx.mod.RECOVERY_CODE_LENGTH == 16, (
                f"the documented code length must hold, got {code!r}"
            )
            assert set(code) <= hexdigits, f"codes are hex, got {code!r}"
        inserts = [
            (sql, params)
            for sql, params in ctx.recorder.statements
            if "INSERT" in sql and "recovery_codes" in sql
        ]
        assert len(inserts) == 10, "one row per code must be written"
        hashes = [params[1] for _, params in inserts]
        assert all(h.startswith("v2:") for h in hashes), (
            "every stored hash must carry the master-key scheme prefix, "
            f"got {hashes[:2]}"
        )
        assert hashes[0] == ctx.mod._hash_code(codes[0]), (
            "the stored hash must be the keyed hash of the first code"
        )
        stored_values = ctx.recorder.bound_values()
        for code in codes:
            assert all(code not in sql for sql in ctx.recorder.sql_texts()), (
                "a plaintext code must never reach SQL text"
            )
            assert code not in stored_values, (
                "a plaintext code must never be a stored value"
            )
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# SA4 -- an app password persists only as its keyed hash
# ---------------------------------------------------------------------------
def test_sa4_app_password_persists_only_its_hash():
    ctx = _load()
    try:
        mgr = ctx.mod.two_factor_manager
        result = mgr.create_app_password("uid-sa4", "cli-tool")
        password = result["password"]
        sql, params = _single_insert(ctx.recorder, "app_passwords")
        stored = params[3]
        assert stored.startswith("v2:") and stored != password, (
            f"only the keyed hash may be stored, got {stored!r}"
        )
        assert password not in sql
        assert password not in ctx.recorder.bound_values(), (
            "the plaintext password must appear in no stored value"
        )
        assert mgr.validate_app_password("uid-sa4", password) is True, (
            "the returned plaintext must validate against the stored hash"
        )
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# SA5 -- every statement is parameterized; values never enter SQL text
# ---------------------------------------------------------------------------
def test_sa5_statements_are_parameterized_never_interpolated():
    ctx = _load()
    try:
        mgr = ctx.mod.two_factor_manager
        user = "uid-'; DROP TABLE recovery_codes;--"
        label = "na'me\" -- probe"
        mgr.totp_setup(user)
        mgr.generate_recovery_codes(user)
        mgr.create_app_password(user, label)
        mgr.get_status(user)
        mgr.validate_recovery_code(user, "f" * 16)
        assert any(
            user in params_flat
            for params_flat in (ctx.recorder.bound_values(),)
        ), "the probe value must have been bound at least once"
        for sql, _params in ctx.recorder.statements:
            assert user not in sql, (
                f"a caller value leaked into SQL text: {sql!r}"
            )
            assert label not in sql, (
                f"a caller value leaked into SQL text: {sql!r}"
            )
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# SA6 -- a master-key hash with no master key verifies closed
# ---------------------------------------------------------------------------
def test_sa6_keyed_hash_without_key_verifies_closed():
    ctx = _load(master_key=None)
    try:
        verdict = ctx.mod._verify_code("anything", "v2:" + "0" * 64)
        assert verdict == (False, False), (
            "a master-key hash that cannot be verified must be refused, "
            f"got {verdict}"
        )
    finally:
        ctx.restore()

    control = _load()
    try:
        stored = control.mod._hash_code("known-value")
        assert stored.startswith("v2:")
        assert control.mod._verify_code("known-value", stored) == (True, False), (
            "the keyed scheme must verify its own hash when the key exists"
        )
    finally:
        control.restore()


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
