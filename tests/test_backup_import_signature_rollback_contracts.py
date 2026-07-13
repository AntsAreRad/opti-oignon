#!/usr/bin/env python3
"""Backup import contracts: signature policy ladder and honest rollback.

Restoring a backup is the moment the platform trusts outside bytes, so the
gate walks a refusal ladder before anything applies: a signature that fails
verification is rejected unconditionally -- no override reaches it; an
unsigned document is refused whenever a local keypair exists, and a signed
one is refused whenever it cannot be verified, both yielding only to the
explicit override; a document with no signature on an install with no
keypair imports for backward compatibility; a verified signature imports.
Once applying, a failing importer rolls back every applied section AND the
failing one itself, a failed pre-import snapshot is a skip sentinel that is
never replayed as an empty replace (which would wipe the live section), and
a section born with an export error is skipped with a named failure while
its siblings apply -- an honest partial, never a silent one. This suite
pins that behavior:

  * BI1 -- a failed signature verification rejects the import even under
    the explicit override, before any snapshot or importer runs;
  * BI2 -- an unsigned document is refused when a local keypair exists,
    and nothing applies;
  * BI3 -- that refusal yields only to the explicit override, which
    lets the import proceed;
  * BI4 -- a signed document that cannot be verified is refused with an
    error naming the impossibility, and nothing applies;
  * BI5 -- no signature and no local keypair imports for backward
    compatibility;
  * BI6 -- a verified signature imports, and verification ran against the
    embedded key;
  * BI7 -- a failing importer rolls back the applied sections and the
    failing one itself, in reverse order, and the result says so;
  * BI8 -- a failed snapshot becomes a skip sentinel: the section is never
    rolled back onto an empty replace, so nothing is wiped;
  * BI9 -- a section carrying an export error is a named failure whose
    importer never runs, while a valid sibling applies without rollback.

Loads the backup manager module in isolation under a stand-in package;
every ``opti_oignon.*`` entry plus the model-client entry is snapshotted
and evicted first, and the seeds are deterministic recorders. A meta-path
guard refuses any project submodule that was not seeded, so the load
behaves identically whether or not the project is installed. Local-only.
Runs under pytest or the __main__ runner.
"""

import base64
import importlib.util
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
    back -- silently importing live code. This guard sits ahead of every
    finder and refuses the names that were not seeded, so a load behaves
    identically whether the project is installed or not.
    """

    _PREFIX = "opti_oignon."

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(self._PREFIX):
            raise ModuleNotFoundError(
                f"not seeded in the isolation window: {fullname}",
                name=fullname,
            )
        return None


class _PQCState:
    """Mutable state driving the post-quantum signing stand-in."""

    def __init__(
        self,
        available=True,
        keypair_exists=False,
        verify_result=None,
    ):
        self.available = available
        self.keypair_exists = keypair_exists
        self.verify_result = verify_result
        self.verify_calls = []


class _SemanticCache:
    """Semantic-cache stand-in recording configuration traffic."""

    def __init__(self, config):
        self._config = dict(config)
        self.get_calls = 0
        self.update_calls = []

    def get_config(self):
        self.get_calls += 1
        return dict(self._config)

    def update_config(self, updates):
        self.update_calls.append(dict(updates))


class _Humanizer:
    """Humanizer stand-in; optionally detonates on every update."""

    def __init__(self, config, raise_on_update=False):
        self._config = dict(config)
        self.raise_on_update = raise_on_update
        self.update_calls = []

    def get_config(self):
        return dict(self._config)

    def update_config(self, **updates):
        self.update_calls.append(dict(updates))
        if self.raise_on_update:
            raise RuntimeError("importer detonated")


class _PresetManager:
    """Preset manager stand-in recording create/update/delete traffic."""

    def __init__(self, existing=None):
        self._presets = dict(existing or {})
        self.deleted = []
        self.created = []
        self.updated = []

    def get_all(self):
        return dict(self._presets)

    def get(self, pid):
        return self._presets.get(pid)

    def delete(self, pid):
        self.deleted.append(pid)
        self._presets.pop(pid, None)

    def create_from_dict(self, pid, pdata):
        self.created.append((pid, pdata))
        self._presets[pid] = pdata

    def update_from_dict(self, pid, pdata):
        self.updated.append((pid, pdata))
        self._presets[pid] = pdata


def _load(pqc_state=None, humanizer_raises=False, presets_existing=None):
    """Load the backup manager under a stand-in package."""
    state = pqc_state or _PQCState()
    keys = ["ollama"] + [
        k
        for k in list(sys.modules)
        if k == "opti_oignon" or k.startswith("opti_oignon.")
    ]
    saved = {k: sys.modules[k] for k in keys if k in sys.modules}
    for k in keys:
        sys.modules.pop(k, None)
    sys.modules["ollama"] = None  # no client import exists; drift fails loud

    root = types.ModuleType("opti_oignon")
    root.__path__ = []
    sys.modules["opti_oignon"] = root

    version = types.ModuleType("opti_oignon.__version__")
    version.__version__ = "0.0.0-isolated"
    sys.modules["opti_oignon.__version__"] = version
    setattr(root, "__version__", version)

    pqc = types.ModuleType("opti_oignon.pqc_signatures")
    pqc.PQC_AVAILABLE = state.available
    pqc.is_pqc_enabled = lambda: False
    pqc.pqc_keypair_exists = lambda path=None: state.keypair_exists
    pqc.load_pqc_keypair = lambda path=None: (b"PUB", b"PRIV")
    pqc.sign_backup = lambda payload, private_key: b"SIGBYTES"

    def _verify_backup(payload, signature, public_key):
        state.verify_calls.append(
            (bytes(payload), bytes(signature), bytes(public_key))
        )
        return state.verify_result

    pqc.verify_backup = _verify_backup
    sys.modules["opti_oignon.pqc_signatures"] = pqc
    root.pqc_signatures = pqc

    semantic = _SemanticCache({"threshold": 1})
    semantic_mod = types.ModuleType("opti_oignon.semantic_cache")
    semantic_mod.semantic_cache = semantic
    sys.modules["opti_oignon.semantic_cache"] = semantic_mod
    root.semantic_cache = semantic_mod

    humanizer = _Humanizer({"tone": "warm"}, raise_on_update=humanizer_raises)
    humanizer_mod = types.ModuleType("opti_oignon.humanizer")
    humanizer_mod.humanizer_engine = humanizer
    sys.modules["opti_oignon.humanizer"] = humanizer_mod
    root.humanizer = humanizer_mod

    presets = _PresetManager(existing=presets_existing)
    presets_mod = types.ModuleType("opti_oignon.presets")
    presets_mod.preset_manager = presets
    sys.modules["opti_oignon.presets"] = presets_mod
    root.presets = presets_mod

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

    full = "opti_oignon.backup_manager"
    spec = importlib.util.spec_from_file_location(full, _OO / "backup_manager.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    root.backup_manager = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        restore()
        raise

    return SimpleNamespace(
        mod=mod,
        pqc=state,
        semantic=semantic,
        humanizer=humanizer,
        presets=presets,
        restore=restore,
    )


def _bare_backup(sections):
    return {"schema_version": "1.0", "metadata": {}, "sections": sections}


def _signed(backup):
    signed = dict(backup)
    signed["_pqc_signature"] = base64.urlsafe_b64encode(b"SIG").decode("ascii")
    signed["_pqc_public_key"] = base64.urlsafe_b64encode(b"PUB").decode("ascii")
    return signed


def test_failed_verification_rejects_even_with_override():
    """BI1 -- a failed verification is final; the override never reaches it."""
    ctx = _load(_PQCState(verify_result=False))
    try:
        mgr = ctx.mod.BackupManager()
        backup = _signed(_bare_backup({"semantic_cache": {"threshold": 2}}))
        result = mgr.import_backup(backup, "replace", allow_unsigned=True)
        assert result.success is False
        assert any("verification failed" in e for e in result.errors), (
            result.errors
        )
        assert ctx.semantic.update_calls == [], (
            "nothing may apply after a failed verification"
        )
        assert ctx.semantic.get_calls == 0, (
            "no snapshot may be taken after a failed verification"
        )
        assert len(ctx.pqc.verify_calls) == 1
    finally:
        ctx.restore()


def test_unsigned_refused_when_local_keypair_exists():
    """BI2 -- unsigned plus a local keypair is a refusal, nothing applies."""
    ctx = _load(_PQCState(keypair_exists=True))
    try:
        mgr = ctx.mod.BackupManager()
        backup = _bare_backup({"semantic_cache": {"threshold": 2}})
        result = mgr.import_backup(backup, "replace")
        assert result.success is False
        assert any("Unsigned backup refused" in e for e in result.errors), (
            result.errors
        )
        assert ctx.semantic.update_calls == [], (
            "nothing may apply on a refused unsigned document"
        )
    finally:
        ctx.restore()


def test_unsigned_refusal_yields_only_to_explicit_override():
    """BI3 -- the explicit override lets the unsigned import proceed."""
    ctx = _load(_PQCState(keypair_exists=True))
    try:
        mgr = ctx.mod.BackupManager()
        backup = _bare_backup({"semantic_cache": {"threshold": 2}})
        result = mgr.import_backup(backup, "replace", allow_unsigned=True)
        assert result.success is True, result.errors
        assert ctx.semantic.update_calls == [{"threshold": 2}], (
            ctx.semantic.update_calls
        )
    finally:
        ctx.restore()


def test_signed_but_unverifiable_is_refused():
    """BI4 -- a signature nobody can verify is a refusal, nothing applies."""
    ctx = _load(_PQCState(available=False))
    try:
        mgr = ctx.mod.BackupManager()
        backup = _signed(_bare_backup({"semantic_cache": {"threshold": 2}}))
        result = mgr.import_backup(backup, "replace")
        assert result.success is False
        assert any("cannot be verified" in e for e in result.errors), (
            result.errors
        )
        assert ctx.semantic.update_calls == [], (
            "nothing may apply on an unverifiable signature"
        )
    finally:
        ctx.restore()


def test_no_signature_no_keypair_imports_for_compatibility():
    """BI5 -- neither signature nor keypair: the import proceeds."""
    ctx = _load(_PQCState(keypair_exists=False))
    try:
        mgr = ctx.mod.BackupManager()
        backup = _bare_backup({"semantic_cache": {"threshold": 2}})
        result = mgr.import_backup(backup, "replace")
        assert result.success is True, result.errors
        assert ctx.semantic.update_calls == [{"threshold": 2}]
    finally:
        ctx.restore()


def test_verified_signature_imports_against_embedded_key():
    """BI6 -- a verified signature imports; the embedded key was checked."""
    ctx = _load(_PQCState(verify_result=True))
    try:
        mgr = ctx.mod.BackupManager()
        backup = _signed(_bare_backup({"semantic_cache": {"threshold": 2}}))
        result = mgr.import_backup(backup, "replace")
        assert result.success is True, result.errors
        assert ctx.semantic.update_calls == [{"threshold": 2}]
        assert len(ctx.pqc.verify_calls) == 1
        assert ctx.pqc.verify_calls[0][2] == b"PUB", (
            "verification runs against the key embedded in the document"
        )
    finally:
        ctx.restore()


def test_failing_importer_rolls_back_applied_and_failing_sections():
    """BI7 -- rollback covers the applied sections AND the failing one."""
    ctx = _load(humanizer_raises=True)
    try:
        mgr = ctx.mod.BackupManager()
        backup = _bare_backup(
            {
                "semantic_cache": {"threshold": 2},
                "humanizer": {"tone": "strict"},
            }
        )
        result = mgr.import_backup(backup, "replace", allow_unsigned=True)
        assert result.success is False
        assert result.rolled_back is True
        assert result.sections_imported == ["semantic_cache"]
        assert result.sections_failed == ["humanizer"]
        assert any("humanizer" in e for e in result.errors), result.errors
        assert ctx.semantic.update_calls == [
            {"threshold": 2},
            {"threshold": 1},
        ], (
            "the applied section must be rolled back to its snapshot, got "
            f"{ctx.semantic.update_calls}"
        )
        assert ctx.humanizer.update_calls == [
            {"tone": "strict"},
            {"tone": "warm"},
        ], (
            "the FAILING section itself must be rolled back too (it may have "
            f"partially applied), got {ctx.humanizer.update_calls}"
        )
    finally:
        ctx.restore()


def test_failed_snapshot_is_skip_sentinel_never_empty_replace():
    """BI8 -- a failed snapshot skips rollback; nothing is wiped."""
    ctx = _load(
        humanizer_raises=True,
        presets_existing={"keep": {"name": "keep-me"}},
    )
    try:
        mgr = ctx.mod.BackupManager()

        def _snapshot_detonates():
            raise RuntimeError("snapshot detonated")

        mgr._section_exporters["presets"] = _snapshot_detonates
        backup = _bare_backup(
            {
                "presets": {"p1": {"name": "incoming"}},
                "humanizer": {"tone": "strict"},
            }
        )
        result = mgr.import_backup(backup, "replace", allow_unsigned=True)
        assert result.success is False
        assert result.rolled_back is True
        assert ctx.presets.deleted == ["keep"], (
            "only the replace-apply may delete; a rollback onto an empty "
            "snapshot would wipe the section, got "
            f"{ctx.presets.deleted}"
        )
        assert ctx.presets.created == [("p1", {"name": "incoming"})]
    finally:
        ctx.restore()


def test_error_marked_section_is_named_failure_sibling_applies():
    """BI9 -- an export-error section never reaches its importer."""
    ctx = _load()
    try:
        mgr = ctx.mod.BackupManager()
        backup = _bare_backup(
            {
                "presets": {"_error": "exporter detonated upstream"},
                "semantic_cache": {"threshold": 2},
            }
        )
        result = mgr.import_backup(backup, "replace", allow_unsigned=True)
        assert result.success is False
        assert result.rolled_back is False, (
            "an honest partial is not a rollback"
        )
        assert result.sections_failed == ["presets"]
        assert result.sections_imported == ["semantic_cache"]
        assert any("export error" in e for e in result.errors), result.errors
        assert ctx.presets.created == [] and ctx.presets.deleted == [], (
            "the importer of an error-marked section must never run"
        )
        assert ctx.semantic.update_calls == [{"threshold": 2}], (
            "the valid sibling must apply"
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
