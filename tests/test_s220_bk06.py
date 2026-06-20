#!/usr/bin/env python3
"""S220 -- BK-06: backup inclusion completeness against ATREST_INVENTORY.md.

Covers:
- BACKUP_SECTIONS 14 -> 21: seven new sections (semantic_cache,
  benchmark_auto_trigger, humanizer, fine_tune, custom_pipelines,
  execution_pipelines, projects_settings), each with an exporter and an
  importer following the S121 handler idiom.
- Config-vs-data dispositions per ATREST_INVENTORY: config-only for
  semantic_cache / benchmark_auto_trigger / humanizer / projects_settings,
  config PLUS variants-registry data for fine_tune (A/B comparison
  results excluded), user-authored data for the two pipeline sections
  (builtins excluded).
- Forward-compatibility decision: schema version STAYS 1.0; the
  new-backup-on-old-install asymmetry is accepted and documented.
- ATREST_INVENTORY.md rolled (candidates resolved), ROADMAP_POST_AUDIT
  BK-06 line closed, pyproject deselects for the three superseded S121
  pins.

Supersessions (deselect-plus-reassert; originals never edited):
- tests/test_s121_backup_restore.py::TestBackupSections::
  test_14_sections_in_tuple -> reasserted here at 21
  (test_tuple_has_21_quoted_names_supersedes_s121_pin).
- tests/test_s121_backup_restore.py::TestBackupManagerLogic::
  test_backup_sections_count -> reasserted here at 21
  (test_tuple_length_21_supersedes_s121_pin).
- tests/test_s121_backup_restore.py::TestBackupManagerLogic::
  test_list_sections_returns_all -> reasserted here at 21
  (test_list_sections_returns_21_supersedes_s121_pin).
- tests/test_s219_ud04_rev2.py::TestDocs::test_inventory_bk06_candidates
  pinned the CANDIDATE wording of the ATREST BK-06 view; superseded by
  the S220 resolution and reasserted here against the resolved wording
  (test_resolved_dispositions_named_supersedes_s219_candidates_pin).
- tests/test_s219_ud04_rev2.py::TestDocs::test_roadmap_f7_rolled pinned
  'STAGED for S220' alongside 'ADVANCED at S219'; the STAGED half is
  superseded by the CLOSED roll, the surviving 'ADVANCED at S219' half
  is carried by test_bk06_closed_in_roadmap.

Red-before discipline: on the pristine S219 tree the new sections,
handlers, doc rolls, and deselects do not exist, so those tests fail by
construction; the invariants that held (the original 14 sections,
schema 1.0, unknown-section rejection, valid strategies) are predicted
green on pristine and must stay green after.
"""

import ast
import importlib
import re
import unittest
from pathlib import Path

# Sweep-order robustness: several sibling suites in the regression
# selection isolate modules via spec_from_file_location and pre-seed
# sys.modules["opti_oignon"] with a non-package stub, sometimes at
# import time of the test module itself and without restoring it.
# Submodule import only goes through the parent on a cache miss, so
# caching the real submodules under their canonical names at collection
# time makes this suite order-independent. When a stub parent is
# already installed, it is swapped for a minimal package shim (correct
# __path__; the real __init__ is NOT re-executed) just long enough to
# import the children, then restored untouched.
_PRELOAD_TARGETS = (
    "opti_oignon.backup_manager",
    "opti_oignon.benchmark_auto_trigger",
    "opti_oignon.fine_tune_tracker",
    "opti_oignon.humanizer",
    "opti_oignon.pipeline_manager",
    "opti_oignon.pipelines",
    "opti_oignon.semantic_cache",
)


def _preload_real_submodules() -> None:
    """Cache the real opti_oignon submodules this suite touches."""
    import sys
    import types

    missing = [n for n in _PRELOAD_TARGETS if n not in sys.modules]
    if not missing:
        return
    parent = sys.modules.get("opti_oignon")
    if parent is not None and not hasattr(parent, "__path__"):
        shim = types.ModuleType("opti_oignon")
        shim.__path__ = [str(Path(__file__).parent.parent / "opti_oignon")]
        sys.modules["opti_oignon"] = shim
        try:
            for name in missing:
                importlib.import_module(name)
        finally:
            sys.modules["opti_oignon"] = parent
    else:
        for name in missing:
            importlib.import_module(name)


_preload_real_submodules()

PROJECT_ROOT = Path(__file__).parent.parent
BACKUP_MANAGER = PROJECT_ROOT / "opti_oignon" / "backup_manager.py"
ATREST = PROJECT_ROOT / "ATREST_INVENTORY.md"
ROADMAP = PROJECT_ROOT / "ROADMAP_POST_AUDIT.md"
PYPROJECT = PROJECT_ROOT / "pyproject.toml"

NEW_SECTIONS = (
    "semantic_cache",
    "benchmark_auto_trigger",
    "humanizer",
    "fine_tune",
    "custom_pipelines",
    "execution_pipelines",
    "projects_settings",
)

ORIGINAL_14 = (
    "presets",
    "system_presets",
    "routing",
    "learned_routing",
    "plugins",
    "rag_metadata",
    "compression",
    "telemetry",
    "sandbox",
    "theme",
    "model_profiles",
    "cascading",
    "speculative",
    "benchmarks",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _manager():
    from opti_oignon.backup_manager import BackupManager

    return BackupManager()


# ---------------------------------------------------------------------------
# Section tuple
# ---------------------------------------------------------------------------


class TestSectionTupleS220(unittest.TestCase):
    """BACKUP_SECTIONS grew 14 -> 21."""

    def test_tuple_length_21_supersedes_s121_pin(self):
        # Supersedes test_s121_backup_restore.py::TestBackupManagerLogic::
        # test_backup_sections_count (was 14).
        from opti_oignon.backup_manager import BACKUP_SECTIONS

        self.assertEqual(len(BACKUP_SECTIONS), 21)

    def test_tuple_has_21_quoted_names_supersedes_s121_pin(self):
        # Supersedes test_s121_backup_restore.py::TestBackupSections::
        # test_14_sections_in_tuple (source-level regex count, was 14).
        content = _read(BACKUP_MANAGER)
        match = re.search(r"BACKUP_SECTIONS\s*=\s*\((.*?)\)", content, re.DOTALL)
        self.assertIsNotNone(match)
        section_names = re.findall(r'"(\w+)"', match.group(1))
        self.assertEqual(len(section_names), 21)

    def test_tuple_contains_semantic_cache(self):
        from opti_oignon.backup_manager import BACKUP_SECTIONS

        self.assertIn("semantic_cache", BACKUP_SECTIONS)

    def test_tuple_contains_benchmark_auto_trigger(self):
        from opti_oignon.backup_manager import BACKUP_SECTIONS

        self.assertIn("benchmark_auto_trigger", BACKUP_SECTIONS)

    def test_tuple_contains_humanizer(self):
        from opti_oignon.backup_manager import BACKUP_SECTIONS

        self.assertIn("humanizer", BACKUP_SECTIONS)

    def test_tuple_contains_fine_tune(self):
        from opti_oignon.backup_manager import BACKUP_SECTIONS

        self.assertIn("fine_tune", BACKUP_SECTIONS)

    def test_tuple_contains_custom_pipelines(self):
        from opti_oignon.backup_manager import BACKUP_SECTIONS

        self.assertIn("custom_pipelines", BACKUP_SECTIONS)

    def test_tuple_contains_execution_pipelines(self):
        from opti_oignon.backup_manager import BACKUP_SECTIONS

        self.assertIn("execution_pipelines", BACKUP_SECTIONS)

    def test_tuple_contains_projects_settings(self):
        from opti_oignon.backup_manager import BACKUP_SECTIONS

        self.assertIn("projects_settings", BACKUP_SECTIONS)

    def test_original_14_preserved(self):
        # Invariant: predicted green on pristine, must stay green after.
        from opti_oignon.backup_manager import BACKUP_SECTIONS

        for name in ORIGINAL_14:
            self.assertIn(name, BACKUP_SECTIONS)


# ---------------------------------------------------------------------------
# Handler registration
# ---------------------------------------------------------------------------


class TestHandlersRegistered(unittest.TestCase):
    """Every new section has an exporter and an importer registered."""

    @classmethod
    def setUpClass(cls):
        cls.mgr = _manager()

    def test_exporters_count_21(self):
        self.assertEqual(len(self.mgr._section_exporters), 21)

    def test_importers_count_21(self):
        self.assertEqual(len(self.mgr._section_importers), 21)

    def test_exporter_registered_semantic_cache(self):
        self.assertTrue(callable(self.mgr._section_exporters["semantic_cache"]))

    def test_exporter_registered_benchmark_auto_trigger(self):
        self.assertTrue(
            callable(self.mgr._section_exporters["benchmark_auto_trigger"])
        )

    def test_exporter_registered_humanizer(self):
        self.assertTrue(callable(self.mgr._section_exporters["humanizer"]))

    def test_exporter_registered_fine_tune(self):
        self.assertTrue(callable(self.mgr._section_exporters["fine_tune"]))

    def test_exporter_registered_custom_pipelines(self):
        self.assertTrue(callable(self.mgr._section_exporters["custom_pipelines"]))

    def test_exporter_registered_execution_pipelines(self):
        self.assertTrue(
            callable(self.mgr._section_exporters["execution_pipelines"])
        )

    def test_exporter_registered_projects_settings(self):
        self.assertTrue(callable(self.mgr._section_exporters["projects_settings"]))

    def test_importer_registered_semantic_cache(self):
        self.assertTrue(callable(self.mgr._section_importers["semantic_cache"]))

    def test_importer_registered_benchmark_auto_trigger(self):
        self.assertTrue(
            callable(self.mgr._section_importers["benchmark_auto_trigger"])
        )

    def test_importer_registered_humanizer(self):
        self.assertTrue(callable(self.mgr._section_importers["humanizer"]))

    def test_importer_registered_fine_tune(self):
        self.assertTrue(callable(self.mgr._section_importers["fine_tune"]))

    def test_importer_registered_custom_pipelines(self):
        self.assertTrue(callable(self.mgr._section_importers["custom_pipelines"]))

    def test_importer_registered_execution_pipelines(self):
        self.assertTrue(
            callable(self.mgr._section_importers["execution_pipelines"])
        )

    def test_importer_registered_projects_settings(self):
        self.assertTrue(callable(self.mgr._section_importers["projects_settings"]))


# ---------------------------------------------------------------------------
# Source structure
# ---------------------------------------------------------------------------


class TestSourceStructure(unittest.TestCase):
    """The handler definitions exist in source; the asymmetry is documented."""

    @classmethod
    def setUpClass(cls):
        cls.content = _read(BACKUP_MANAGER)

    def test_source_has_export_semantic_cache(self):
        self.assertIn("def _export_semantic_cache(", self.content)

    def test_source_has_export_benchmark_auto_trigger(self):
        self.assertIn("def _export_benchmark_auto_trigger(", self.content)

    def test_source_has_export_humanizer(self):
        self.assertIn("def _export_humanizer(", self.content)

    def test_source_has_export_fine_tune(self):
        self.assertIn("def _export_fine_tune(", self.content)

    def test_source_has_export_custom_pipelines(self):
        self.assertIn("def _export_custom_pipelines(", self.content)

    def test_source_has_export_execution_pipelines(self):
        self.assertIn("def _export_execution_pipelines(", self.content)

    def test_source_has_export_projects_settings(self):
        self.assertIn("def _export_projects_settings(", self.content)

    def test_source_has_import_semantic_cache(self):
        self.assertIn("def _import_semantic_cache(", self.content)

    def test_source_has_import_benchmark_auto_trigger(self):
        self.assertIn("def _import_benchmark_auto_trigger(", self.content)

    def test_source_has_import_humanizer(self):
        self.assertIn("def _import_humanizer(", self.content)

    def test_source_has_import_fine_tune(self):
        self.assertIn("def _import_fine_tune(", self.content)

    def test_source_has_import_custom_pipelines(self):
        self.assertIn("def _import_custom_pipelines(", self.content)

    def test_source_has_import_execution_pipelines(self):
        self.assertIn("def _import_execution_pipelines(", self.content)

    def test_source_has_import_projects_settings(self):
        self.assertIn("def _import_projects_settings(", self.content)

    def test_docstring_documents_forward_asymmetry(self):
        # D3: schema stays 1.0; the asymmetry is documented in the module
        # docstring (older install rejects newer backups explicitly).
        import opti_oignon.backup_manager as bm

        doc = bm.__doc__ or ""
        self.assertIn("older install", doc)
        self.assertIn("unknown section", doc)

    def test_pipeline_replace_never_touches_builtins(self):
        # Both pipeline importers clear via list_custom() only.
        for fn_name in ("_import_custom_pipelines", "_import_execution_pipelines"):
            body = self.content.split(f"def {fn_name}")[1].split("\n    def ")[0]
            self.assertIn("list_custom()", body)
            self.assertNotIn("list_builtin", body)


# ---------------------------------------------------------------------------
# list_sections
# ---------------------------------------------------------------------------


class TestListSections(unittest.TestCase):
    def setUp(self):
        self.mgr = _manager()

    def test_list_sections_returns_21_supersedes_s121_pin(self):
        # Supersedes test_s121_backup_restore.py::TestBackupManagerLogic::
        # test_list_sections_returns_all (was 14).
        sections = self.mgr.list_sections()
        self.assertEqual(len(sections), 21)
        names = {s["name"] for s in sections}
        for name in NEW_SECTIONS:
            self.assertIn(name, names)

    def test_new_sections_have_descriptions(self):
        sections = {s["name"]: s for s in self.mgr.list_sections()}
        for name in NEW_SECTIONS:
            self.assertTrue(sections[name]["description"].strip())


# ---------------------------------------------------------------------------
# Export shapes (config-vs-data dispositions)
# ---------------------------------------------------------------------------


class TestExportShapes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mgr = _manager()

    def _section(self, name):
        backup = self.mgr.export_sections([name])
        return backup["sections"][name]

    def test_semantic_cache_exports_config_keys(self):
        data = self._section("semantic_cache")
        self.assertIn("similarity_threshold", data)
        self.assertIn("ttl_seconds", data)

    def test_semantic_cache_never_exports_content(self):
        # Cached entries are regenerable content, excluded by design.
        data = self._section("semantic_cache")
        for forbidden in ("entries", "cache_entries", "responses"):
            self.assertNotIn(forbidden, data)

    def test_auto_trigger_exports_config_keys(self):
        data = self._section("benchmark_auto_trigger")
        self.assertIn("poll_interval_seconds", data)
        self.assertIn("cooldown_seconds", data)

    def test_humanizer_exports_config_keys(self):
        data = self._section("humanizer")
        self.assertIn("intensity", data)
        self.assertIn("mode", data)

    def test_fine_tune_export_two_halves(self):
        data = self._section("fine_tune")
        self.assertIn("config", data)
        self.assertIn("variants", data)
        self.assertIsInstance(data["variants"], dict)

    def test_fine_tune_never_exports_comparisons(self):
        # A/B comparison results are telemetry-class data, excluded.
        data = self._section("fine_tune")
        self.assertNotIn("comparisons", data)
        content = _read(BACKUP_MANAGER)
        body = content.split("def _export_fine_tune")[1].split("\n    def ")[0]
        self.assertNotIn("list_comparisons", body)

    def test_projects_settings_exports_yaml(self):
        data = self._section("projects_settings")
        self.assertIn("projects", data)

    def test_custom_pipelines_excludes_builtins(self):
        from opti_oignon.pipeline_manager import get_pipeline_manager

        data = self._section("custom_pipelines")
        builtin_ids = {p.id for p in get_pipeline_manager().list_builtin()}
        self.assertTrue(builtin_ids, "expected builtin pipelines to exist")
        self.assertFalse(builtin_ids & set(data.keys()))

    def test_execution_pipelines_excludes_builtins(self):
        from opti_oignon.pipelines import get_pipeline_store

        data = self._section("execution_pipelines")
        builtin_ids = {p.id for p in get_pipeline_store().list_builtin()}
        self.assertFalse(builtin_ids & set(data.keys()))

    def test_export_all_includes_all_21(self):
        backup = self.mgr.export_all()
        for name in ORIGINAL_14 + NEW_SECTIONS:
            self.assertIn(name, backup["sections"])


# ---------------------------------------------------------------------------
# Import semantics (temp-pathed stores; repo files never mutated)
# ---------------------------------------------------------------------------


class TestImportSemantics(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.mgr = _manager()
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_execution_pipelines_merge_and_replace(self):
        pl = importlib.import_module("opti_oignon.pipelines")

        original = pl._pipeline_store
        try:
            pl._pipeline_store = pl.PipelineStore(data_dir=self.tmp_path)
            store = pl._pipeline_store
            step = [{"step_type": "direct", "name": "s1"}]
            payload = {
                "imported_one": {"name": "Imported one", "steps": step},
            }
            self.mgr._import_execution_pipelines(payload, "merge")
            self.assertIn(
                "imported_one", {p.id for p in store.list_custom()}
            )
            # merge keeps the existing version
            self.mgr._import_execution_pipelines(
                {"imported_one": {"name": "Changed", "steps": step}}, "merge"
            )
            self.assertEqual(store.get("imported_one").name, "Imported one")
            # replace overwrites and clears customs absent from the backup
            self.mgr._import_execution_pipelines(
                {"imported_two": {"name": "Imported two", "steps": step}},
                "replace",
            )
            ids = {p.id for p in store.list_custom()}
            self.assertIn("imported_two", ids)
            self.assertNotIn("imported_one", ids)
        finally:
            pl._pipeline_store = original

    def test_execution_pipelines_replace_keeps_builtins(self):
        pl = importlib.import_module("opti_oignon.pipelines")

        original = pl._pipeline_store
        try:
            pl._pipeline_store = pl.PipelineStore(data_dir=self.tmp_path)
            store = pl._pipeline_store
            builtin_before = {p.id for p in store.list_builtin()}
            self.mgr._import_execution_pipelines({}, "replace")
            self.assertEqual(
                {p.id for p in store.list_builtin()}, builtin_before
            )
        finally:
            pl._pipeline_store = original

    def test_custom_pipelines_merge_and_replace(self):
        pm = importlib.import_module("opti_oignon.pipeline_manager")

        original = pm._pipeline_manager
        try:
            mgr_inst = pm.PipelineManager()
            mgr_inst._custom_file = self.tmp_path / "pipelines_custom.yaml"
            # start from a clean custom set in the temp store
            for p in list(mgr_inst.list_custom()):
                mgr_inst.delete(p.id)
            pm._pipeline_manager = mgr_inst
            builtin = mgr_inst.list_builtin()[0].to_dict()
            payload = {
                "imported_chain": {
                    "name": "Imported chain",
                    "description": "",
                    "pattern": builtin["pattern"],
                    "steps": builtin["steps"],
                }
            }
            self.mgr._import_custom_pipelines(payload, "merge")
            self.assertIn(
                "imported_chain", {p.id for p in mgr_inst.list_custom()}
            )
            self.mgr._import_custom_pipelines(
                {
                    "imported_other": {
                        "name": "Other",
                        "description": "",
                        "pattern": builtin["pattern"],
                        "steps": builtin["steps"],
                    }
                },
                "replace",
            )
            ids = {p.id for p in mgr_inst.list_custom()}
            self.assertIn("imported_other", ids)
            self.assertNotIn("imported_chain", ids)
        finally:
            pm._pipeline_manager = original

    def test_semantic_cache_merge_keeps_existing_keys(self):
        sc = importlib.import_module("opti_oignon.semantic_cache")

        original = sc.semantic_cache
        try:
            sc.semantic_cache = sc.SemanticCache(
                db_path=self.tmp_path / "semantic_cache.db",
                config_path=self.tmp_path / "cache.yaml",
            )
            before = sc.semantic_cache.get_config()["similarity_threshold"]
            self.mgr._import_semantic_cache(
                {"similarity_threshold": 0.55}, "merge"
            )
            self.assertEqual(
                sc.semantic_cache.get_config()["similarity_threshold"], before
            )
            self.mgr._import_semantic_cache(
                {"similarity_threshold": 0.55}, "replace"
            )
            self.assertEqual(
                sc.semantic_cache.get_config()["similarity_threshold"], 0.55
            )
        finally:
            sc.semantic_cache = original

    def test_fine_tune_variants_merge_and_upsert_replace(self):
        ft = importlib.import_module("opti_oignon.fine_tune_tracker")

        if ft.fine_tune_tracker is None:
            self.skipTest("fine-tune tracker unavailable")
        original = ft.fine_tune_tracker
        try:
            ft.fine_tune_tracker = ft.FineTuneTracker(
                db_path=self.tmp_path / "fine_tune_variants.db",
                config_path=self.tmp_path / "fine_tune.yaml",
            )
            tracker = ft.fine_tune_tracker
            payload = {
                "variants": {
                    "v1": {
                        "variant_id": "v1",
                        "name": "first",
                        "base_model": "base:1b",
                        "variant_model": "tuned:1b",
                    }
                }
            }
            # config key deliberately absent: the repo fine_tune.yaml must
            # never be touched by this test.
            self.mgr._import_fine_tune(payload, "merge")
            self.assertIsNotNone(tracker.get_variant("v1"))
            # merge skips the existing id
            payload["variants"]["v1"]["name"] = "renamed"
            self.mgr._import_fine_tune(payload, "merge")
            self.assertEqual(tracker.get_variant("v1").name, "first")
            # replace is an upsert: updates existing, never clears others
            self.mgr._import_fine_tune(payload, "replace")
            self.assertEqual(tracker.get_variant("v1").name, "renamed")
        finally:
            ft.fine_tune_tracker = original

    def test_importers_raise_when_subsystem_unavailable(self):
        sc = importlib.import_module("opti_oignon.semantic_cache")

        original = sc.semantic_cache
        try:
            sc.semantic_cache = None
            with self.assertRaises(RuntimeError):
                self.mgr._import_semantic_cache({"enabled": True}, "merge")
        finally:
            sc.semantic_cache = original


# ---------------------------------------------------------------------------
# Invariants that held (predicted green on pristine)
# ---------------------------------------------------------------------------


class TestInvariantsHeld(unittest.TestCase):
    def setUp(self):
        self.mgr = _manager()

    def test_schema_version_stays_1_0(self):
        # D3: the format is NOT versioned; sections grow additively.
        from opti_oignon.backup_manager import BACKUP_SCHEMA_VERSION

        self.assertEqual(BACKUP_SCHEMA_VERSION, "1.0")

    def test_valid_strategies_unchanged(self):
        from opti_oignon.backup_manager import VALID_STRATEGIES

        self.assertEqual(VALID_STRATEGIES, ("merge", "replace"))

    def test_validate_rejects_unknown_section(self):
        errors = self.mgr.validate_backup(
            {
                "schema_version": "1.0",
                "metadata": {},
                "sections": {"totally_unknown_section_xyz": {}},
            }
        )
        self.assertTrue(
            any("totally_unknown_section_xyz" in e for e in errors)
        )

    def test_validate_accepts_minimal_backup(self):
        errors = self.mgr.validate_backup(
            {"schema_version": "1.0", "metadata": {}, "sections": {}}
        )
        self.assertEqual(errors, [])

    def test_validate_accepts_new_sections(self):
        errors = self.mgr.validate_backup(
            {
                "schema_version": "1.0",
                "metadata": {},
                "sections": {name: {} for name in NEW_SECTIONS},
            }
        )
        self.assertEqual(errors, [])

    def test_export_metadata_shape_unchanged(self):
        backup = self.mgr.export_sections(["presets"])
        self.assertEqual(backup["schema_version"], "1.0")
        self.assertIn("opti_oignon_version", backup["metadata"])
        self.assertIn("sections_included", backup["metadata"])

    def test_preview_invalid_strategy_rejected(self):
        preview = self.mgr.preview_import(
            {"schema_version": "1.0", "metadata": {}, "sections": {}},
            strategy="sideways",
        )
        self.assertFalse(preview.valid)

    def test_unknown_section_export_still_raises(self):
        with self.assertRaises(ValueError):
            self.mgr.export_sections(["no_such_section"])


# ---------------------------------------------------------------------------
# Documents rolled
# ---------------------------------------------------------------------------


class TestAtrestRolled(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content = _read(ATREST)

    def test_headline_gap_closed(self):
        self.assertIn("the headline gap closed", self.content)

    def test_bk06_view_resolved(self):
        self.assertIn("resolved at S220", self.content)
        self.assertIn("14 -> 21", self.content)

    def test_empty_candidates_recorded(self):
        self.assertIn("no persisted config surface", self.content)
        self.assertIn("no persisted memory-settings surface", self.content)

    def test_no_live_bk06_candidate_rows_remain(self):
        matrix_rows = [
            line
            for line in self.content.splitlines()
            if line.startswith("|") and "bk06-candidate" in line
        ]
        self.assertEqual(matrix_rows, [])

    def test_per_user_stores_routed_to_ud04_export(self):
        self.assertIn("UD-04 per-user export", self.content)

    def test_resolved_dispositions_named_supersedes_s219_candidates_pin(self):
        # Supersedes test_s219_ud04_rev2.py::TestDocs::
        # test_inventory_bk06_candidates, which pinned the candidate
        # wording; the substance (every candidate store named with its
        # disposition) is reasserted against the resolved wording.
        for needle in (
            "semantic_cache",
            "benchmark_auto_trigger",
            "humanizer",
            "fine_tune",
            "custom_pipelines",
            "execution_pipelines",
            "projects_settings",
            "pipelines_custom.yaml",
            "cache.yaml",
            "no persisted config surface",
            "no persisted memory-settings surface",
        ):
            self.assertIn(needle, self.content)


class TestRoadmapRolled(unittest.TestCase):
    def test_bk06_closed_in_roadmap(self):
        # Supersedes the STAGED half of test_s219_ud04_rev2.py::TestDocs::
        # test_roadmap_f7_rolled and carries its surviving half
        # ('ADVANCED at S219' stays pinned).
        content = _read(ROADMAP)
        self.assertIn(
            "BK-06 :: backup inclusion completeness :: CLOSED at", content
        )
        self.assertNotIn("STAGED for S220", content)
        self.assertIn("ADVANCED at S219", content)


class TestPyprojectDeselects(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content = _read(PYPROJECT)

    def test_deselect_s121_tuple_pin(self):
        self.assertIn(
            "--deselect=tests/test_s121_backup_restore.py::"
            "TestBackupSections::test_14_sections_in_tuple",
            self.content,
        )

    def test_deselect_s121_count_pin(self):
        self.assertIn(
            "--deselect=tests/test_s121_backup_restore.py::"
            "TestBackupManagerLogic::test_backup_sections_count",
            self.content,
        )

    def test_deselect_s121_list_sections_pin(self):
        self.assertIn(
            "--deselect=tests/test_s121_backup_restore.py::"
            "TestBackupManagerLogic::test_list_sections_returns_all",
            self.content,
        )


# ---------------------------------------------------------------------------
# AST validity
# ---------------------------------------------------------------------------


class TestASTValid(unittest.TestCase):
    def test_backup_manager_ast_valid(self):
        ast.parse(_read(BACKUP_MANAGER))


if __name__ == "__main__":
    unittest.main()
