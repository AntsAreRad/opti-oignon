"""
S194 F7d -- backup & restore fix lot tests.

Covers:
- BK-03: the allow_unsigned override is exposed through both import
  routes and schemas (source/schema assertions; the route import chain
  is container-heavy per the established protocol).
- BK-04: a section whose importer raises mid-apply is itself rolled
  back to its snapshot, along with previously applied sections.
- BK-05: a snapshot that fails at capture time is a None sentinel and
  rollback SKIPS it instead of replaying {} over live state.

backup_manager.py is loaded with the S185 loader idiom; the empty
opti_oignon stub is removed after load to avoid cross-suite pollution.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[1]


def _read(rel):
    return (_PROJECT / rel).read_text(encoding="utf-8")


def _load_backup_manager():
    created_stub = "opti_oignon" not in sys.modules
    if created_stub:
        sys.modules["opti_oignon"] = types.ModuleType("opti_oignon")
    spec = importlib.util.spec_from_file_location(
        "s194d_backup_manager", str(_PROJECT / "opti_oignon" / "backup_manager.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        if created_stub:
            entry = sys.modules.get("opti_oignon")
            if entry is not None and not getattr(entry, "__file__", None):
                del sys.modules["opti_oignon"]
    return mod


bm = _load_backup_manager()


def _backup_with(sections):
    return {
        "schema_version": "1.0",
        "metadata": {},
        "sections": sections,
    }


class _Recorder:
    """Records importer calls; raises on demand."""

    def __init__(self):
        self.calls = []

    def importer(self, name, fail=False, partial_state=None):
        def _imp(data, strategy):
            self.calls.append((name, data, strategy))
            if partial_state is not None:
                partial_state.append(name)
            if fail:
                raise RuntimeError(f"boom in {name}")
        return _imp


class TestBK04FailingSectionRolledBack(unittest.TestCase):
    """BK-04: the failing section is included in the rollback set."""

    def test_failing_section_restored_from_snapshot(self):
        mgr = bm.BackupManager()
        rec = _Recorder()

        # Two sections: 'presets' applies cleanly, 'theme' raises.
        mgr._section_exporters = {
            "presets": lambda: {"snap": "presets"},
            "theme": lambda: {"snap": "theme"},
        }
        mgr._section_importers = {
            "presets": rec.importer("presets"),
            "theme": rec.importer("theme", fail=True),
        }

        result = mgr.import_backup(
            _backup_with({"presets": {"p": 1}, "theme": {"t": 2}}),
            strategy="merge",
        )

        self.assertFalse(result.success)
        self.assertTrue(result.rolled_back)
        self.assertIn("theme", result.sections_failed)

        # Call sequence: apply presets, apply theme (raises), then
        # rollback in reverse: theme snapshot, presets snapshot.
        names = [c[0] for c in rec.calls]
        self.assertEqual(names, ["presets", "theme", "theme", "presets"])
        # Rollback calls replay the SNAPSHOTS with replace strategy.
        self.assertEqual(rec.calls[2][1], {"snap": "theme"})
        self.assertEqual(rec.calls[2][2], bm.STRATEGY_REPLACE)
        self.assertEqual(rec.calls[3][1], {"snap": "presets"})

    def test_clean_import_has_no_rollback(self):
        mgr = bm.BackupManager()
        rec = _Recorder()
        mgr._section_exporters = {"presets": lambda: {"snap": 1}}
        mgr._section_importers = {"presets": rec.importer("presets")}

        result = mgr.import_backup(
            _backup_with({"presets": {"p": 1}}), strategy="merge"
        )
        self.assertTrue(result.success)
        self.assertFalse(result.rolled_back)
        self.assertEqual([c[0] for c in rec.calls], ["presets"])


class TestBK05SnapshotSentinel(unittest.TestCase):
    """BK-05: failed snapshots are None and rollback skips them."""

    def _raise(self):
        raise RuntimeError("export down")

    def test_failed_snapshot_not_replayed(self):
        mgr = bm.BackupManager()
        rec = _Recorder()

        # 'presets' snapshot RAISES; 'theme' snapshot fine; theme import
        # fails so a rollback happens.
        mgr._section_exporters = {
            "presets": self._raise,
            "theme": lambda: {"snap": "theme"},
        }
        mgr._section_importers = {
            "presets": rec.importer("presets"),
            "theme": rec.importer("theme", fail=True),
        }

        result = mgr.import_backup(
            _backup_with({"presets": {"p": 1}, "theme": {"t": 2}}),
            strategy="replace",
        )

        self.assertTrue(result.rolled_back)
        names = [c[0] for c in rec.calls]
        # presets applied, theme raised, theme rolled back; presets
        # rollback SKIPPED (None snapshot) -- never replayed with {}.
        self.assertEqual(names, ["presets", "theme", "theme"])
        for _, data, _ in rec.calls:
            self.assertNotEqual(data, {})

    def test_missing_exporter_yields_none_sentinel(self):
        mgr = bm.BackupManager()
        rec = _Recorder()
        mgr._section_exporters = {}
        mgr._section_importers = {"theme": rec.importer("theme", fail=True)}

        result = mgr.import_backup(
            _backup_with({"theme": {"t": 1}}), strategy="merge"
        )
        self.assertTrue(result.rolled_back)
        # Only the failed apply call; no rollback replay possible.
        self.assertEqual([c[0] for c in rec.calls], ["theme"])


class TestBK03OverrideExposed(unittest.TestCase):
    """BK-03: allow_unsigned is reachable through the API surface."""

    def test_schema_carries_field(self):
        src = _read("opti_oignon/api/schemas.py")
        block = src.split("class BackupImportRequest")[1].split("class ")[0]
        self.assertIn("allow_unsigned: bool = False", block)

    def test_routes_thread_the_flag(self):
        src = _read("opti_oignon/api/routes_backup.py")
        self.assertEqual(src.count("allow_unsigned=request.allow_unsigned"), 1)
        self.assertEqual(src.count("allow_unsigned=req.allow_unsigned"), 1)
        enc_block = src.split("class EncryptedImportRequest")[1].split("@router")[0]
        self.assertIn("allow_unsigned: bool = Field(", enc_block)
        self.assertIn("default=False", enc_block)

    def test_manager_default_remains_false(self):
        import inspect
        sig = inspect.signature(bm.BackupManager.import_backup)
        self.assertIs(sig.parameters["allow_unsigned"].default, False)

    def test_invalid_signature_never_relaxed(self):
        # Re-assert the S185 invariant against the edited file: a present
        # but failing signature is rejected even with allow_unsigned.
        mgr = bm.BackupManager()
        mgr._section_exporters = {}
        mgr._section_importers = {}
        data = _backup_with({})
        data[bm._PQC_SIGNATURE_KEY] = "c2ln"
        data[bm._PQC_PUBLIC_KEY_KEY] = "cHVi"
        original = bm.BackupManager._verify_backup_pqc
        bm.BackupManager._verify_backup_pqc = lambda self, d: False
        try:
            result = mgr.import_backup(data, allow_unsigned=True)
        finally:
            bm.BackupManager._verify_backup_pqc = original
        self.assertFalse(result.success)
        self.assertTrue(any("verification failed" in e for e in result.errors))


if __name__ == "__main__":
    unittest.main()
