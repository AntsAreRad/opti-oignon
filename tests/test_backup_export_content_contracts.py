#!/usr/bin/env python3
"""Backup export contracts: honest shape, no private material, no host writes.

A configuration backup is a self-describing JSON document the user carries
away. Its honesty is load-bearing: the section list must be exactly what was
asked for, a failing exporter must leave an explicit error marker instead of
poisoning its siblings, the signing step must never leak the private half of
the keypair and must degrade to an unsigned document rather than fail the
export, the signed payload must be the canonical serialization so a verifier
can rebuild it byte for byte, and generating a backup must never write
anything to the host tree. This suite pins that behavior:

  * BX1 -- an unknown section name is refused with an error naming it,
    before any exporter runs;
  * BX2 -- the exported document carries the schema version, an honest
    metadata block, and exactly the requested sections;
  * BX3 -- a failing exporter yields an explicit error marker for its own
    section while sibling sections export intact;
  * BX4 -- the signed document embeds the public key only; the private key
    is handed to the signer and never appears anywhere in the document;
  * BX5 -- a signing failure degrades to an unsigned document and never
    raises out of the export;
  * BX6 -- the signed payload is the canonical sorted serialization of the
    document without the signature fields, rebuildable by a verifier;
  * BX7 -- a full export writes nothing on disk under the package tree.

Loads the backup manager module in isolation under a stand-in package;
every ``opti_oignon.*`` entry plus the model-client entry is snapshotted
and evicted first, and the seeds are deterministic recorders. A meta-path
guard refuses any project submodule that was not seeded, so the load
behaves identically whether or not the project is installed. Local-only.
Runs under pytest or the __main__ runner.
"""

import base64
import importlib.util
import json
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
        enabled=False,
        keypair_exists=False,
        keypair=(b"PUBKEYBYTES", b"PRIVKEYBYTES"),
        sign_raises=False,
        verify_result=None,
    ):
        self.available = available
        self.enabled = enabled
        self.keypair_exists = keypair_exists
        self.keypair = keypair
        self.sign_raises = sign_raises
        self.verify_result = verify_result
        self.sign_calls = []
        self.verify_calls = []


class _ConfigStore:
    """Config-holder stand-in with call recorders."""

    def __init__(self, config):
        self._config = dict(config)
        self.get_calls = 0
        self.update_calls = []

    def get_config(self):
        self.get_calls += 1
        return dict(self._config)

    def update_config(self, updates):
        self.update_calls.append(dict(updates))


def _load(pqc_state=None):
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
    pqc.is_pqc_enabled = lambda: state.enabled
    pqc.pqc_keypair_exists = lambda path=None: state.keypair_exists
    pqc.load_pqc_keypair = lambda path=None: state.keypair

    def _sign_backup(payload, private_key):
        state.sign_calls.append((bytes(payload), bytes(private_key)))
        if state.sign_raises:
            raise RuntimeError("signer detonated")
        return b"SIGBYTES"

    def _verify_backup(payload, signature, public_key):
        state.verify_calls.append(
            (bytes(payload), bytes(signature), bytes(public_key))
        )
        return state.verify_result

    pqc.sign_backup = _sign_backup
    pqc.verify_backup = _verify_backup
    sys.modules["opti_oignon.pqc_signatures"] = pqc
    root.pqc_signatures = pqc

    semantic = _ConfigStore({"threshold": 0.5})
    semantic_mod = types.ModuleType("opti_oignon.semantic_cache")
    semantic_mod.semantic_cache = semantic
    sys.modules["opti_oignon.semantic_cache"] = semantic_mod
    root.semantic_cache = semantic_mod

    humanizer = _ConfigStore({"tone": "warm"})
    humanizer_mod = types.ModuleType("opti_oignon.humanizer")
    humanizer_mod.humanizer_engine = humanizer
    sys.modules["opti_oignon.humanizer"] = humanizer_mod
    root.humanizer = humanizer_mod

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
        restore=restore,
    )


def test_unknown_section_refused_before_any_exporter():
    """BX1 -- an unknown name is refused by name, no exporter runs."""
    ctx = _load()
    try:
        mgr = ctx.mod.BackupManager()
        witnessed = []
        mgr._section_exporters = {
            name: (lambda n=name: witnessed.append(n) or {})
            for name in ctx.mod.BACKUP_SECTIONS
        }
        refused = None
        try:
            mgr.export_sections(["semantic_cache", "no_such_section"])
        except ValueError as exc:
            refused = exc
        assert refused is not None, (
            "an unknown section must refuse the export with a ValueError"
        )
        assert "no_such_section" in str(refused), str(refused)
        assert witnessed == [], (
            f"no exporter may run on a refused request, saw {witnessed}"
        )
    finally:
        ctx.restore()


def test_export_shape_is_honest_and_exact():
    """BX2 -- schema version, honest metadata, exactly the asked sections."""
    ctx = _load()
    try:
        mgr = ctx.mod.BackupManager()
        data = mgr.export_sections(["semantic_cache"])
        assert data["schema_version"] == ctx.mod.BACKUP_SCHEMA_VERSION
        meta = data["metadata"]
        assert meta["opti_oignon_version"] == "0.0.0-isolated"
        assert meta["sections_included"] == ["semantic_cache"], (
            "sections_included must name exactly the requested sections, "
            f"got {meta['sections_included']}"
        )
        assert set(data["sections"].keys()) == {"semantic_cache"}, (
            f"only the requested sections may appear, got {data['sections']}"
        )
        assert data["sections"]["semantic_cache"] == {"threshold": 0.5}
        assert "_pqc_signature" not in data, (
            "signing is off in this window; the document must be unsigned"
        )
    finally:
        ctx.restore()


def test_failing_exporter_leaves_honest_marker_siblings_intact():
    """BX3 -- an exporter failure marks its own section, siblings export."""
    ctx = _load()
    try:
        mgr = ctx.mod.BackupManager()

        def _boom():
            raise RuntimeError("exporter detonated")

        mgr._section_exporters["semantic_cache"] = _boom
        data = mgr.export_sections(["semantic_cache", "humanizer"])
        assert data["sections"]["semantic_cache"] == {
            "_error": "exporter detonated"
        }, data["sections"]["semantic_cache"]
        assert data["sections"]["humanizer"] == {"tone": "warm"}, (
            "a sibling section must export intact next to a failed one"
        )
    finally:
        ctx.restore()


def test_signed_document_embeds_public_key_never_private():
    """BX4 -- public key embedded, private key only handed to the signer."""
    ctx = _load(_PQCState(enabled=True, keypair_exists=True))
    try:
        mgr = ctx.mod.BackupManager()
        data = mgr.export_sections(["semantic_cache"])
        pub_b64 = base64.urlsafe_b64encode(b"PUBKEYBYTES").decode("ascii")
        priv_b64 = base64.urlsafe_b64encode(b"PRIVKEYBYTES").decode("ascii")
        sig_b64 = base64.urlsafe_b64encode(b"SIGBYTES").decode("ascii")
        assert data["_pqc_signature"] == sig_b64
        assert data["_pqc_public_key"] == pub_b64, (
            "the document must embed the PUBLIC key, "
            f"got {data['_pqc_public_key']!r}"
        )
        assert len(ctx.pqc.sign_calls) == 1
        assert ctx.pqc.sign_calls[0][1] == b"PRIVKEYBYTES", (
            "the signer receives the private key; the document never does"
        )
        serialized = json.dumps(data, ensure_ascii=False)
        assert priv_b64 not in serialized, (
            "the private key must never appear anywhere in the document"
        )
    finally:
        ctx.restore()


def test_signing_failure_degrades_to_unsigned_never_raises():
    """BX5 -- a signer failure yields an unsigned document, export survives."""
    ctx = _load(_PQCState(enabled=True, keypair_exists=True, sign_raises=True))
    try:
        mgr = ctx.mod.BackupManager()
        data = mgr.export_sections(["semantic_cache"])
        assert "_pqc_signature" not in data and "_pqc_public_key" not in data, (
            "a failed signing must leave the document unsigned, not half-signed"
        )
        assert data["sections"]["semantic_cache"] == {"threshold": 0.5}, (
            "the export itself must survive a signing failure"
        )
    finally:
        ctx.restore()


def test_signed_payload_is_canonical_rebuildable_serialization():
    """BX6 -- payload == sorted canonical JSON without the signature fields."""
    ctx = _load(_PQCState(enabled=True, keypair_exists=True))
    try:
        mgr = ctx.mod.BackupManager()
        data = mgr.export_sections(["semantic_cache"])
        assert len(ctx.pqc.sign_calls) == 1
        recorded = ctx.pqc.sign_calls[0][0]
        clean = {
            k: v
            for k, v in data.items()
            if k not in ("_pqc_signature", "_pqc_public_key")
        }
        rebuilt = json.dumps(
            clean, ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
        assert recorded == rebuilt, (
            "the signed payload must be the canonical sorted serialization a "
            "verifier can rebuild from the document"
        )
    finally:
        ctx.restore()


def test_full_export_writes_nothing_under_package_tree():
    """BX7 -- exporting everything leaves the package tree byte-stable."""
    ctx = _load()
    try:

        def _walk():
            snapshot = {}
            for path in _OO.rglob("*"):
                if path.is_file():
                    stat = path.stat()
                    snapshot[str(path)] = (stat.st_size, stat.st_mtime_ns)
            return snapshot

        before = _walk()
        mgr = ctx.mod.BackupManager()
        data = mgr.export_all()
        after = _walk()
        assert after == before, (
            "a full export must not create, grow, or touch any file under "
            "the package tree"
        )
        assert set(data["sections"].keys()) == set(ctx.mod.BACKUP_SECTIONS)
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
