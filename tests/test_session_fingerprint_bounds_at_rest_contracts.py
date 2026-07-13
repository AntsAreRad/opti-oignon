#!/usr/bin/env python3
"""Context-fingerprint contracts: bounded state, seamed storage, lean output.

The coding-context fingerprint summarizes an ongoing coding run into a
compact blob for prompt injection. Its only persistent surface -- the
checkpoint-preference store -- flows through the encrypted-connection
helper with parameterized statements; every in-memory dimension is hard
bounded; and the serialized blob is minimized: file paths are reduced to
basenames, raw failure text never leaves the tracker, and a dimension
weighted to zero is omitted entirely. This suite pins that behavior:

  * FP1 -- checkpoint preferences persist through the seeded encrypted-
    connection seam with parameterized statements, and round-trip;
  * FP2 -- context anchors are truncated to the documented length,
    deduplicated by content, and capped with oldest-first eviction;
  * FP3 -- the failure history honours its configured cap and the test
    health window holds its fixed size;
  * FP4 -- the serialized blob carries only file basenames and never the
    raw failure text, while the failure category still surfaces;
  * FP5 -- a dimension weighted to zero is omitted from the blob even
    when it holds data, and reappears with a positive weight;
  * FP6 -- a checkpoint action outside the decision vocabulary writes
    nothing; a vocabulary action writes exactly one row.

Loads the fingerprint module in isolation under a stand-in package; every
``opti_oignon.*`` entry plus the model client entry is snapshotted and
evicted first, and the only project seed is a recording ``safe_connect``
that routes each database path to a shared in-memory store and journals
every statement. A meta-path guard refuses any project submodule that was
not seeded, so the load behaves identically whether or not the project is
installed. Local-only. Runs under pytest or the __main__ runner.
"""

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


class _RecordingConnection:
    """Journal every statement against a shared in-memory store.

    The store opens and closes a connection per operation; ``close`` is
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

    Routes each database path to one shared in-memory store and journals
    every (sql, params) pair. The journal is the at-rest observation
    surface of these contracts.
    """

    def __init__(self):
        self.statements = []
        self.paths = []
        self.closes = 0
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

    def preference_inserts(self):
        return [
            (sql, params)
            for sql, params in self.statements
            if "INSERT INTO preferences" in sql
        ]


def _load():
    """Load session_fingerprint.py under a stand-in package."""
    keys = ["ollama"] + [
        k
        for k in list(sys.modules)
        if k == "opti_oignon" or k.startswith("opti_oignon.")
    ]
    saved = {k: sys.modules[k] for k in keys if k in sys.modules}
    for k in keys:
        sys.modules.pop(k, None)
    sys.modules["ollama"] = None  # no client import exists; drift fails loud

    recorder = _AtRestRecorder()

    root = types.ModuleType("opti_oignon")
    root.__path__ = []
    sys.modules["opti_oignon"] = root
    db_utils = types.ModuleType("opti_oignon.db_utils")
    db_utils.safe_connect = recorder.connect
    sys.modules["opti_oignon.db_utils"] = db_utils
    root.db_utils = db_utils

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
        for k, v in saved.items():
            sys.modules[k] = v
        recorder.close_all()

    full = "opti_oignon.session_fingerprint"
    spec = importlib.util.spec_from_file_location(
        full, _OO / "session_fingerprint.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    root.session_fingerprint = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        restore()
        raise

    return SimpleNamespace(mod=mod, recorder=recorder, restore=restore)


def _manager(ctx, db_path, **config_overrides):
    """Build a manager with an explicit store path and config overrides."""
    config = ctx.mod.FingerprintConfig(**config_overrides)
    store = ctx.mod.UserPreferencesStore(db_path=db_path)
    return ctx.mod.FingerprintManager(config=config, preferences_store=store)


# ---------------------------------------------------------------------------
# FP1 -- preferences persist through the seam, parameterized, round-trip
# ---------------------------------------------------------------------------
def test_fp1_preferences_persist_through_the_seam_parameterized():
    ctx = _load()
    try:
        store = ctx.mod.UserPreferencesStore(db_path="prefs-under-test.db")
        assert "prefs-under-test.db" in ctx.recorder.paths, (
            "the store must open its path through the seeded encrypted-"
            "connection seam"
        )
        ctx.recorder.reset_log()
        marker = "ctx-with-'quote--marker"
        store.record("approve", "plan", marker)
        inserts = ctx.recorder.preference_inserts()
        assert len(inserts) == 1, f"expected one insert, got {len(inserts)}"
        sql, params = inserts[0]
        assert "?" in sql, "the statement must use bind placeholders"
        assert marker not in sql, (
            f"a caller value leaked into SQL text: {sql!r}"
        )
        assert marker in [str(p) for p in params], (
            "the caller value must travel as a bound parameter"
        )
        ratios = store.get_ratios()
        assert ratios["approve"] == 1.0, (
            f"the recorded decision must round-trip, got {ratios}"
        )
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# FP2 -- anchors are truncated, deduplicated, and capped oldest-first
# ---------------------------------------------------------------------------
def test_fp2_anchors_are_truncated_deduplicated_and_capped():
    ctx = _load()
    try:
        mgr = _manager(ctx, "fp2.db", max_anchors=3)
        long_text = "invariant-" + "x" * 400
        mgr.add_anchor(long_text)
        anchors = mgr._anchors.anchors
        assert len(anchors) == 1 and len(anchors[0]) == 200, (
            f"anchor text must be truncated to 200, got {len(anchors[0])}"
        )
        mgr.add_anchor(long_text)
        assert len(mgr._anchors.anchors) == 1, (
            "the same content must not be stored twice"
        )
        for i in range(5):
            mgr.add_anchor(f"anchor-number-{i}")
        kept = mgr._anchors.anchors
        assert len(kept) == 3, (
            f"the configured cap must hold, got {len(kept)}"
        )
        assert kept == [f"anchor-number-{i}" for i in (2, 3, 4)], (
            f"eviction must drop the oldest entries first, got {kept}"
        )
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# FP3 -- failure history and test-health window are bounded
# ---------------------------------------------------------------------------
def test_fp3_failure_history_and_health_window_are_bounded():
    ctx = _load()
    try:
        mgr = _manager(ctx, "fp3.db", max_bug_history=4)
        for i in range(7):
            mgr.on_test(
                {"passed": False, "error": f"AssertionError case {i}"}
            )
        assert mgr._bugs.serialize()["total"] == 4, (
            "the configured failure-history cap must hold"
        )
        for _ in range(25):
            mgr.on_test({"passed": True})
        assert mgr._test_health.total_runs == 20, (
            "the health window must hold its fixed size"
        )
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# FP4 -- the blob carries basenames only and never raw failure text
# ---------------------------------------------------------------------------
def test_fp4_blob_carries_basenames_and_never_raw_failure_text():
    ctx = _load()
    try:
        mgr = _manager(ctx, "fp4.db")
        step = {
            "file_path": "/srv/private-tree/deep/dir/hot_module.py",
            "content": "def alpha_beta():\n    return 1\n",
            "completed": True,
        }
        for _ in range(3):
            mgr.on_step(dict(step))
        mgr.on_test(
            {
                "passed": False,
                "error": "SECRETTRACE-90210 AssertionError raised under "
                "/srv/private-tree/deep",
            }
        )
        blob = mgr.serialize_compact()
        full = repr(mgr.serialize())
        assert "hot_module.py" in blob, (
            f"the hot file must surface by basename, got {blob!r}"
        )
        assert "/srv/private-tree" not in blob, (
            "no full path may leave the tracker in the compact blob"
        )
        assert "/srv/private-tree" not in full, (
            "no full path may leave the tracker in the serialized dict"
        )
        assert "SECRETTRACE-90210" not in blob, (
            "raw failure text must never enter the compact blob"
        )
        assert "SECRETTRACE-90210" not in full, (
            "raw failure text must never enter the serialized dict"
        )
        assert "assertion" in blob, (
            "the failure category must still surface, proving the "
            f"dimension was active, got {blob!r}"
        )
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# FP5 -- a zero-weight dimension is omitted from the blob
# ---------------------------------------------------------------------------
def test_fp5_zero_weight_dimension_is_omitted():
    ctx = _load()
    try:
        muted = _manager(
            ctx, "fp5-muted.db", dimension_weights={"user_preferences": 0.0}
        )
        muted.on_checkpoint({"action": "approve", "phase": "plan"})
        assert "prefs" not in muted.serialize(), (
            "a zero-weight dimension must be omitted even when it holds "
            "data"
        )
        voiced = _manager(
            ctx, "fp5-voiced.db", dimension_weights={"user_preferences": 0.9}
        )
        voiced.on_checkpoint({"action": "approve", "phase": "plan"})
        serialized = voiced.serialize()
        assert "prefs" in serialized, (
            f"a positive-weight dimension must surface, got {serialized}"
        )
        assert serialized["prefs"]["approve"] == 1.0
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# FP6 -- only vocabulary checkpoint actions write anything
# ---------------------------------------------------------------------------
def test_fp6_out_of_vocabulary_checkpoint_actions_write_nothing():
    ctx = _load()
    try:
        mgr = _manager(ctx, "fp6.db")
        ctx.recorder.reset_log()
        mgr.on_checkpoint(
            {"action": "detonate", "phase": "apply", "context": "junk"}
        )
        assert ctx.recorder.preference_inserts() == [], (
            "an action outside the decision vocabulary must write nothing"
        )
        mgr.on_checkpoint({"action": "modify", "phase": "apply"})
        inserts = ctx.recorder.preference_inserts()
        assert len(inserts) == 1 and inserts[0][1][0] == "modify", (
            f"a vocabulary action must write exactly one row, got {inserts}"
        )
    finally:
        ctx.restore()


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
