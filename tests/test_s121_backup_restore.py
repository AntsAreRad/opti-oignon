#!/usr/bin/env python3
"""
Tests for S121 -- Full Config Backup/Restore.

Test groups:
1. BackupManager: export_all, export_sections, list_sections, validate_backup
2. Import: merge strategy, replace strategy, rollback on error
3. Preview: diff computation, summary counts, validation errors
4. Round-trip: export -> import -> verify identical
5. API routes: routes_backup.py structure and endpoint definitions
6. API schemas: Pydantic models for backup endpoints
7. Frontend: BackupRestorePanel.svelte structure, imports, sections
8. Frontend: backup.ts API client functions
9. Frontend: types.ts backup interfaces
10. Settings page: backup tab integration
11. deps.py: BackupManager singleton registration
12. CSS variable compliance: no hardcoded hex in S121 components
13. HTML tag balance: all S121 Svelte components
14. Version bump: 2.3.0 across all version files
15. AST validation: all S121 Python files

Target: ~93 tests
"""

import ast
import json
import os
import re
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "opti_oignon"
API_DIR = BACKEND_DIR / "api"
FRONTEND = ROOT / "frontend" / "src"
SETTINGS_COMPONENTS = FRONTEND / "lib" / "components" / "settings"
API_CLIENT_DIR = FRONTEND / "lib" / "api"
TYPES_FILE = FRONTEND / "lib" / "types.ts"
VERSION_FILE = BACKEND_DIR / "__version__.py"

# Files under test
BACKUP_MANAGER = BACKEND_DIR / "backup_manager.py"
ROUTES_BACKUP = API_DIR / "routes_backup.py"
SCHEMAS = API_DIR / "schemas.py"
DEPS = API_DIR / "deps.py"
APP = API_DIR / "app.py"
BACKUP_PANEL = SETTINGS_COMPONENTS / "BackupRestorePanel.svelte"
BACKUP_API_TS = API_CLIENT_DIR / "backup.ts"
SETTINGS_PAGE = FRONTEND / "routes" / "settings" / "+page.svelte"


def _read(path: Path) -> str:
    """Read a file as UTF-8 text."""
    return path.read_text(encoding="utf-8")


def _find_hex_colors(content: str) -> list[str]:
    """Find hex color values that are NOT CSS variable fallbacks or HTML entities."""
    matches = re.findall(r"#[0-9a-fA-F]{3,8}\b", content)
    real_hex = []
    for m in matches:
        idx = content.find(m)
        if idx > 0 and content[idx - 1] == "&":
            continue
        preceding = content[max(0, idx - 60):idx]
        if re.search(r"var\(--oo-[a-zA-Z0-9-]+,\s*$", preceding):
            continue
        real_hex.append(m)
    return real_hex


def _check_html_balance(content: str) -> list[str]:
    """Check HTML tag balance in Svelte content. Returns list of issues."""
    # Remove script and style blocks
    c = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL)
    c = re.sub(r"<style[^>]*>.*?</style>", "", c, flags=re.DOTALL)
    # Remove Svelte logic blocks
    c = re.sub(r"\{[#/:@].*?\}", "", c)
    # Self-closing void tags
    void_tags = {"input", "img", "br", "hr", "meta", "link", "source",
                 "area", "col", "embed", "wbr", "path"}
    # Count opens (non-self-closing, non-void)
    open_counts: dict[str, int] = {}
    for m in re.finditer(r"<([a-zA-Z][a-zA-Z0-9-]*)\b[^>]*?(/?)>", c):
        tag = m.group(1).lower()
        self_close = m.group(2) == "/"
        if not self_close and tag not in void_tags:
            open_counts[tag] = open_counts.get(tag, 0) + 1
    close_counts: dict[str, int] = {}
    for m in re.finditer(r"</([a-zA-Z][a-zA-Z0-9-]*)\s*>", c):
        tag = m.group(1).lower()
        close_counts[tag] = close_counts.get(tag, 0) + 1
    issues = []
    all_tags = set(list(open_counts.keys()) + list(close_counts.keys()))
    for t in sorted(all_tags):
        o = open_counts.get(t, 0)
        cl = close_counts.get(t, 0)
        if o != cl:
            issues.append(f"{t}: {o} open, {cl} close")
    return issues


# =========================================================================
# 1. BackupManager module structure
# =========================================================================

class TestBackupManagerModule(unittest.TestCase):
    """Test backup_manager.py module structure."""

    def setUp(self):
        self.content = _read(BACKUP_MANAGER)
        self.tree = ast.parse(self.content)

    def test_file_exists(self):
        self.assertTrue(BACKUP_MANAGER.is_file())

    def test_has_class_backup_manager(self):
        classes = [n.name for n in ast.walk(self.tree) if isinstance(n, ast.ClassDef)]
        self.assertIn("BackupManager", classes)

    def test_has_class_backup_diff_item(self):
        classes = [n.name for n in ast.walk(self.tree) if isinstance(n, ast.ClassDef)]
        self.assertIn("BackupDiffItem", classes)

    def test_has_class_backup_preview(self):
        classes = [n.name for n in ast.walk(self.tree) if isinstance(n, ast.ClassDef)]
        self.assertIn("BackupPreview", classes)

    def test_has_class_import_result(self):
        classes = [n.name for n in ast.walk(self.tree) if isinstance(n, ast.ClassDef)]
        self.assertIn("ImportResult", classes)

    def test_has_export_all_method(self):
        self.assertIn("def export_all(", self.content)

    def test_has_export_sections_method(self):
        self.assertIn("def export_sections(", self.content)

    def test_has_import_backup_method(self):
        self.assertIn("def import_backup(", self.content)

    def test_has_preview_import_method(self):
        self.assertIn("def preview_import(", self.content)

    def test_has_validate_backup_method(self):
        self.assertIn("def validate_backup(", self.content)

    def test_has_list_sections_method(self):
        self.assertIn("def list_sections(", self.content)

    def test_has_rollback_method(self):
        self.assertIn("def _rollback(", self.content)

    def test_schema_version_constant(self):
        self.assertIn('BACKUP_SCHEMA_VERSION = "1.0"', self.content)

    def test_backup_sections_tuple(self):
        self.assertIn("BACKUP_SECTIONS", self.content)

    def test_strategy_constants(self):
        self.assertIn('STRATEGY_MERGE = "merge"', self.content)
        self.assertIn('STRATEGY_REPLACE = "replace"', self.content)

    def test_singleton_at_module_level(self):
        self.assertIn("backup_manager = BackupManager()", self.content)
        self.assertIn("BACKUP_AVAILABLE = True", self.content)


# =========================================================================
# 2. Section exporters/importers registered
# =========================================================================

class TestBackupSections(unittest.TestCase):
    """Test that all 14 sections have exporters and importers."""

    def setUp(self):
        self.content = _read(BACKUP_MANAGER)

    def test_14_sections_in_tuple(self):
        # Count quoted strings in BACKUP_SECTIONS tuple
        match = re.search(r"BACKUP_SECTIONS\s*=\s*\((.*?)\)", self.content, re.DOTALL)
        self.assertIsNotNone(match)
        section_names = re.findall(r'"(\w+)"', match.group(1))
        self.assertEqual(len(section_names), 14)

    def test_has_exporter_presets(self):
        self.assertIn("def _export_presets(", self.content)

    def test_has_exporter_system_presets(self):
        self.assertIn("def _export_system_presets(", self.content)

    def test_has_exporter_routing(self):
        self.assertIn("def _export_routing(", self.content)

    def test_has_exporter_learned_routing(self):
        self.assertIn("def _export_learned_routing(", self.content)

    def test_has_exporter_plugins(self):
        self.assertIn("def _export_plugins(", self.content)

    def test_has_exporter_rag_metadata(self):
        self.assertIn("def _export_rag_metadata(", self.content)

    def test_has_exporter_compression(self):
        self.assertIn("def _export_compression(", self.content)

    def test_has_exporter_telemetry(self):
        self.assertIn("def _export_telemetry(", self.content)

    def test_has_exporter_sandbox(self):
        self.assertIn("def _export_sandbox(", self.content)

    def test_has_exporter_theme(self):
        self.assertIn("def _export_theme(", self.content)

    def test_has_exporter_model_profiles(self):
        self.assertIn("def _export_model_profiles(", self.content)

    def test_has_exporter_cascading(self):
        self.assertIn("def _export_cascading(", self.content)

    def test_has_exporter_speculative(self):
        self.assertIn("def _export_speculative(", self.content)

    def test_has_exporter_benchmarks(self):
        self.assertIn("def _export_benchmarks(", self.content)

    def test_has_importer_presets(self):
        self.assertIn("def _import_presets(", self.content)

    def test_has_importer_system_presets(self):
        self.assertIn("def _import_system_presets(", self.content)

    def test_has_importer_routing(self):
        self.assertIn("def _import_routing(", self.content)

    def test_has_importer_learned_routing(self):
        self.assertIn("def _import_learned_routing(", self.content)

    def test_has_importer_plugins(self):
        self.assertIn("def _import_plugins(", self.content)

    def test_has_importer_rag_metadata(self):
        self.assertIn("def _import_rag_metadata(", self.content)

    def test_has_importer_compression(self):
        self.assertIn("def _import_compression(", self.content)

    def test_has_importer_telemetry(self):
        self.assertIn("def _import_telemetry(", self.content)

    def test_has_importer_sandbox(self):
        self.assertIn("def _import_sandbox(", self.content)

    def test_has_importer_theme(self):
        self.assertIn("def _import_theme(", self.content)

    def test_has_importer_model_profiles(self):
        self.assertIn("def _import_model_profiles(", self.content)

    def test_has_importer_cascading(self):
        self.assertIn("def _import_cascading(", self.content)

    def test_has_importer_speculative(self):
        self.assertIn("def _import_speculative(", self.content)

    def test_has_importer_benchmarks(self):
        self.assertIn("def _import_benchmarks(", self.content)


# =========================================================================
# 3. BackupManager logic (using importlib isolation)
# =========================================================================

class TestBackupManagerLogic(unittest.TestCase):
    """Test BackupManager export/import/preview/validate logic in isolation."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "backup_manager_iso", str(BACKUP_MANAGER)
        )
        mod = importlib.util.module_from_spec(spec)
        # Patch out subsystem imports - they will fail in test container
        import sys
        sys.modules["opti_oignon"] = type(sys)("opti_oignon")
        sys.modules["opti_oignon.__version__"] = type(sys)("opti_oignon.__version__")
        sys.modules["opti_oignon.__version__"].__version__ = "2.4.0"
        try:
            spec.loader.exec_module(mod)
        except Exception:
            mod = None
        cls.mod = mod

    def test_module_loads(self):
        self.assertIsNotNone(self.mod)

    def test_backup_sections_count(self):
        if not self.mod:
            self.skipTest("Module not loaded")
        self.assertEqual(len(self.mod.BACKUP_SECTIONS), 14)

    def test_valid_strategies(self):
        if not self.mod:
            self.skipTest("Module not loaded")
        self.assertEqual(self.mod.VALID_STRATEGIES, ("merge", "replace"))

    def test_validate_backup_missing_fields(self):
        if not self.mod:
            self.skipTest("Module not loaded")
        mgr = self.mod.BackupManager()
        errors = mgr.validate_backup({})
        self.assertTrue(len(errors) >= 2)  # missing schema_version and sections

    def test_validate_backup_invalid_type(self):
        if not self.mod:
            self.skipTest("Module not loaded")
        mgr = self.mod.BackupManager()
        errors = mgr.validate_backup("not a dict")
        self.assertIn("Backup must be a JSON object", errors)

    def test_validate_backup_unknown_section(self):
        if not self.mod:
            self.skipTest("Module not loaded")
        mgr = self.mod.BackupManager()
        data = {
            "schema_version": "1.0",
            "metadata": {},
            "sections": {"fake_section": {}},
        }
        errors = mgr.validate_backup(data)
        found = [e for e in errors if "fake_section" in e]
        self.assertTrue(len(found) > 0)

    def test_validate_backup_valid_structure(self):
        if not self.mod:
            self.skipTest("Module not loaded")
        mgr = self.mod.BackupManager()
        data = {
            "schema_version": "1.0",
            "metadata": {"opti_oignon_version": "2.4.0"},
            "sections": {"presets": {}},
        }
        errors = mgr.validate_backup(data)
        self.assertEqual(len(errors), 0)

    def test_export_sections_unknown_raises(self):
        if not self.mod:
            self.skipTest("Module not loaded")
        mgr = self.mod.BackupManager()
        with self.assertRaises(ValueError):
            mgr.export_sections(["nonexistent_section"])

    def test_export_all_returns_metadata(self):
        if not self.mod:
            self.skipTest("Module not loaded")
        mgr = self.mod.BackupManager()
        result = mgr.export_all()
        self.assertIn("schema_version", result)
        self.assertIn("metadata", result)
        self.assertIn("sections", result)
        self.assertEqual(result["schema_version"], "1.0")

    def test_export_metadata_fields(self):
        if not self.mod:
            self.skipTest("Module not loaded")
        mgr = self.mod.BackupManager()
        result = mgr.export_all()
        meta = result["metadata"]
        self.assertIn("opti_oignon_version", meta)
        self.assertIn("timestamp", meta)
        self.assertIn("timestamp_iso", meta)
        self.assertIn("platform", meta)
        self.assertIn("sections_included", meta)

    def test_export_metadata_platform(self):
        if not self.mod:
            self.skipTest("Module not loaded")
        mgr = self.mod.BackupManager()
        result = mgr.export_all()
        plat = result["metadata"]["platform"]
        self.assertIn("system", plat)
        self.assertIn("release", plat)
        self.assertIn("machine", plat)
        self.assertIn("python_version", plat)

    def test_export_partial_sections(self):
        if not self.mod:
            self.skipTest("Module not loaded")
        mgr = self.mod.BackupManager()
        result = mgr.export_sections(["presets", "routing"])
        self.assertEqual(
            set(result["metadata"]["sections_included"]),
            {"presets", "routing"},
        )

    def test_list_sections_returns_all(self):
        if not self.mod:
            self.skipTest("Module not loaded")
        mgr = self.mod.BackupManager()
        sections = mgr.list_sections()
        self.assertEqual(len(sections), 14)
        names = {s["name"] for s in sections}
        self.assertIn("presets", names)
        self.assertIn("routing", names)
        self.assertIn("sandbox", names)

    def test_list_sections_has_description(self):
        if not self.mod:
            self.skipTest("Module not loaded")
        mgr = self.mod.BackupManager()
        sections = mgr.list_sections()
        for s in sections:
            self.assertIn("description", s)
            self.assertIn("item_count", s)
            self.assertIn("available", s)

    def test_preview_invalid_strategy(self):
        if not self.mod:
            self.skipTest("Module not loaded")
        mgr = self.mod.BackupManager()
        data = {
            "schema_version": "1.0",
            "metadata": {},
            "sections": {"presets": {}},
        }
        preview = mgr.preview_import(data, strategy="bad_strategy")
        self.assertFalse(preview.valid)
        self.assertTrue(len(preview.errors) > 0)

    def test_preview_invalid_backup(self):
        if not self.mod:
            self.skipTest("Module not loaded")
        mgr = self.mod.BackupManager()
        preview = mgr.preview_import({"no_schema": True})
        self.assertFalse(preview.valid)

    def test_import_invalid_strategy(self):
        if not self.mod:
            self.skipTest("Module not loaded")
        mgr = self.mod.BackupManager()
        result = mgr.import_backup({}, strategy="bad")
        self.assertFalse(result.success)
        self.assertTrue(len(result.errors) > 0)

    def test_import_invalid_backup(self):
        if not self.mod:
            self.skipTest("Module not loaded")
        mgr = self.mod.BackupManager()
        result = mgr.import_backup({"bad": True})
        self.assertFalse(result.success)

    def test_diff_item_to_dict(self):
        if not self.mod:
            self.skipTest("Module not loaded")
        item = self.mod.BackupDiffItem(
            section="presets", key="my_preset", action="add",
            incoming_value="test",
        )
        d = item.to_dict()
        self.assertEqual(d["section"], "presets")
        self.assertEqual(d["action"], "add")

    def test_backup_preview_to_dict(self):
        if not self.mod:
            self.skipTest("Module not loaded")
        preview = self.mod.BackupPreview(valid=True, strategy="merge")
        d = preview.to_dict()
        self.assertTrue(d["valid"])
        self.assertEqual(d["strategy"], "merge")

    def test_import_result_to_dict(self):
        if not self.mod:
            self.skipTest("Module not loaded")
        result = self.mod.ImportResult(
            success=True,
            sections_imported=["presets"],
        )
        d = result.to_dict()
        self.assertTrue(d["success"])
        self.assertEqual(d["sections_imported"], ["presets"])

    def test_summarize_truncates_long_string(self):
        if not self.mod:
            self.skipTest("Module not loaded")
        result = self.mod._summarize("x" * 200)
        self.assertTrue(len(result) < 200)
        self.assertTrue(result.endswith("..."))

    def test_summarize_truncates_large_dict(self):
        if not self.mod:
            self.skipTest("Module not loaded")
        d = {f"key_{i}": i for i in range(20)}
        result = self.mod._summarize(d)
        self.assertTrue(len(result) <= 5)

    def test_summarize_truncates_large_list(self):
        if not self.mod:
            self.skipTest("Module not loaded")
        lst = list(range(20))
        result = self.mod._summarize(lst)
        self.assertTrue(len(result) <= 5)

    def test_summarize_passthrough_small(self):
        if not self.mod:
            self.skipTest("Module not loaded")
        self.assertEqual(self.mod._summarize(42), 42)
        self.assertEqual(self.mod._summarize("short"), "short")


# =========================================================================
# 4. API routes structure
# =========================================================================

class TestRoutesBackup(unittest.TestCase):
    """Test routes_backup.py structure."""

    def setUp(self):
        self.content = _read(ROUTES_BACKUP)
        self.tree = ast.parse(self.content)

    def test_file_exists(self):
        self.assertTrue(ROUTES_BACKUP.is_file())

    def test_ast_valid(self):
        # If setUp parsed without error, AST is valid
        self.assertIsNotNone(self.tree)

    def test_router_prefix(self):
        self.assertIn('prefix="/api/backup"', self.content)

    def test_tags_backup(self):
        self.assertIn('tags=["backup"]', self.content)

    def test_endpoint_sections(self):
        self.assertIn('"/sections"', self.content)
        self.assertIn("def list_backup_sections", self.content)

    def test_endpoint_export(self):
        self.assertIn('"/export"', self.content)
        self.assertIn("def export_backup", self.content)

    def test_endpoint_preview(self):
        self.assertIn('"/preview"', self.content)
        self.assertIn("def preview_import", self.content)

    def test_endpoint_import(self):
        self.assertIn('"/import"', self.content)
        self.assertIn("def import_backup", self.content)

    def test_imports_deps(self):
        self.assertIn("from .deps import BACKUP_AVAILABLE, backup_manager", self.content)

    def test_imports_schemas(self):
        self.assertIn("BackupImportRequest", self.content)
        self.assertIn("BackupPreviewRequest", self.content)
        self.assertIn("BackupSectionsResponse", self.content)

    def test_check_available_function(self):
        self.assertIn("def _check_available", self.content)

    def test_content_disposition_header(self):
        self.assertIn("Content-Disposition", self.content)
        self.assertIn(".oo-backup.json", self.content)


# =========================================================================
# 5. API schemas
# =========================================================================

class TestBackupSchemas(unittest.TestCase):
    """Test backup-related Pydantic models in schemas.py."""

    def setUp(self):
        self.content = _read(SCHEMAS)

    def test_backup_section_info(self):
        self.assertIn("class BackupSectionInfo(BaseModel):", self.content)

    def test_backup_sections_response(self):
        self.assertIn("class BackupSectionsResponse(BaseModel):", self.content)

    def test_backup_import_request(self):
        self.assertIn("class BackupImportRequest(BaseModel):", self.content)

    def test_backup_preview_request(self):
        self.assertIn("class BackupPreviewRequest(BaseModel):", self.content)

    def test_backup_diff_item_response(self):
        self.assertIn("class BackupDiffItemResponse(BaseModel):", self.content)

    def test_backup_preview_response(self):
        self.assertIn("class BackupPreviewResponse(BaseModel):", self.content)

    def test_backup_import_response(self):
        self.assertIn("class BackupImportResponse(BaseModel):", self.content)

    def test_import_request_has_strategy(self):
        # Find the class block and check strategy field
        idx = self.content.find("class BackupImportRequest")
        block = self.content[idx:idx + 200]
        self.assertIn("strategy", block)

    def test_import_response_has_rolled_back(self):
        idx = self.content.find("class BackupImportResponse")
        block = self.content[idx:idx + 300]
        self.assertIn("rolled_back", block)


# =========================================================================
# 6. deps.py registration
# =========================================================================

class TestDepsRegistration(unittest.TestCase):
    """Test BackupManager registration in deps.py."""

    def setUp(self):
        self.content = _read(DEPS)

    def test_imports_backup_manager(self):
        self.assertIn("from opti_oignon.backup_manager import", self.content)

    def test_backup_available_flag(self):
        self.assertIn("BACKUP_AVAILABLE", self.content)

    def test_backup_manager_singleton(self):
        self.assertIn("backup_manager", self.content)


# =========================================================================
# 7. app.py registration
# =========================================================================

class TestAppRegistration(unittest.TestCase):
    """Test backup router in app.py."""

    def setUp(self):
        self.content = _read(APP)

    def test_imports_backup_router(self):
        self.assertIn("from .routes_backup import router as backup_router", self.content)

    def test_includes_backup_router(self):
        self.assertIn("app.include_router(backup_router)", self.content)


# =========================================================================
# 8. Frontend: BackupRestorePanel.svelte
# =========================================================================

class TestBackupRestorePanel(unittest.TestCase):
    """Test BackupRestorePanel.svelte structure."""

    def setUp(self):
        self.content = _read(BACKUP_PANEL)

    def test_file_exists(self):
        self.assertTrue(BACKUP_PANEL.is_file())

    def test_imports_api_functions(self):
        self.assertIn("listBackupSections", self.content)
        self.assertIn("downloadBackup", self.content)
        self.assertIn("previewImport", self.content)
        self.assertIn("importBackup", self.content)

    def test_imports_types(self):
        self.assertIn("BackupSectionInfo", self.content)
        self.assertIn("BackupData", self.content)
        self.assertIn("BackupPreviewResponse", self.content)

    def test_export_section_exists(self):
        self.assertIn("Export Configuration", self.content)

    def test_import_section_exists(self):
        self.assertIn("Import Configuration", self.content)

    def test_strategy_toggle(self):
        self.assertIn("Import Strategy", self.content)
        self.assertIn("Merge", self.content)
        self.assertIn("Replace", self.content)

    def test_preview_button(self):
        self.assertIn("Preview Changes", self.content)

    def test_drop_zone(self):
        self.assertIn(".oo-backup.json", self.content)
        self.assertIn("dragover", self.content)
        self.assertIn("dragleave", self.content)

    def test_diff_table_headers(self):
        self.assertIn("Section", self.content)
        self.assertIn("Action", self.content)

    def test_summary_badges(self):
        self.assertIn("summary.add", self.content)
        self.assertIn("summary.update", self.content)
        self.assertIn("summary.skip", self.content)

    def test_import_result_display(self):
        self.assertIn("Import Successful", self.content)
        self.assertIn("Import Failed", self.content)
        self.assertIn("rolled_back", self.content)

    def test_select_all_toggle(self):
        self.assertIn("Select All", self.content)
        self.assertIn("Deselect All", self.content)

    def test_file_input(self):
        self.assertIn("backup-file-input", self.content)
        self.assertIn('accept=".json,.oo-backup.json"', self.content)


# =========================================================================
# 9. Frontend: backup.ts API client
# =========================================================================

class TestBackupApiClient(unittest.TestCase):
    """Test backup.ts API client."""

    def setUp(self):
        self.content = _read(BACKUP_API_TS)

    def test_file_exists(self):
        self.assertTrue(BACKUP_API_TS.is_file())

    def test_list_backup_sections(self):
        self.assertIn("export async function listBackupSections", self.content)

    def test_export_backup(self):
        self.assertIn("export async function exportBackup", self.content)

    def test_download_backup(self):
        self.assertIn("export async function downloadBackup", self.content)

    def test_preview_import(self):
        self.assertIn("export async function previewImport", self.content)

    def test_import_backup(self):
        self.assertIn("export async function importBackup", self.content)

    def test_api_paths(self):
        self.assertIn("/api/backup/sections", self.content)
        self.assertIn("/api/backup/export", self.content)
        self.assertIn("/api/backup/preview", self.content)
        self.assertIn("/api/backup/import", self.content)

    def test_download_creates_blob(self):
        self.assertIn("Blob", self.content)
        self.assertIn("createObjectURL", self.content)

    def test_download_filename(self):
        self.assertIn(".oo-backup.json", self.content)


# =========================================================================
# 10. Frontend: types.ts interfaces
# =========================================================================

class TestBackupTypes(unittest.TestCase):
    """Test backup types in types.ts."""

    def setUp(self):
        self.content = _read(TYPES_FILE)

    def test_backup_section_info(self):
        self.assertIn("export interface BackupSectionInfo", self.content)

    def test_backup_sections_response(self):
        self.assertIn("export interface BackupSectionsResponse", self.content)

    def test_backup_diff_item(self):
        self.assertIn("export interface BackupDiffItem", self.content)

    def test_backup_preview_response(self):
        self.assertIn("export interface BackupPreviewResponse", self.content)

    def test_backup_import_response(self):
        self.assertIn("export interface BackupImportResponse", self.content)

    def test_backup_metadata(self):
        self.assertIn("export interface BackupMetadata", self.content)

    def test_backup_data(self):
        self.assertIn("export interface BackupData", self.content)

    def test_backup_diff_item_action_type(self):
        # Check action uses union type
        idx = self.content.find("export interface BackupDiffItem")
        block = self.content[idx:idx + 300]
        self.assertIn("'add'", block)
        self.assertIn("'update'", block)
        self.assertIn("'skip'", block)


# =========================================================================
# 11. Settings page integration
# =========================================================================

class TestSettingsPageIntegration(unittest.TestCase):
    """Test backup tab in settings page."""

    def setUp(self):
        self.content = _read(SETTINGS_PAGE)

    def test_imports_backup_panel(self):
        self.assertIn("BackupRestorePanel", self.content)

    def test_tab_type_includes_backup(self):
        self.assertIn("'backup'", self.content)

    def test_tab_label(self):
        self.assertIn("Backup", self.content)

    def test_tab_content_section(self):
        self.assertIn("activeTab === 'backup'", self.content)
        self.assertIn("<BackupRestorePanel", self.content)


# =========================================================================
# 12. CSS variable compliance
# =========================================================================

class TestCSSCompliance(unittest.TestCase):
    """No hardcoded hex colors in S121 Svelte components."""

    def test_backup_panel_no_hex(self):
        content = _read(BACKUP_PANEL)
        # Remove script block for analysis
        no_script = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL)
        hex_found = _find_hex_colors(no_script)
        self.assertEqual(hex_found, [], f"Hardcoded hex colors found: {hex_found}")


# =========================================================================
# 13. HTML tag balance
# =========================================================================

class TestHTMLBalance(unittest.TestCase):
    """All S121 Svelte components have balanced HTML tags."""

    def test_backup_panel_balanced(self):
        content = _read(BACKUP_PANEL)
        issues = _check_html_balance(content)
        self.assertEqual(issues, [], f"Unbalanced tags: {issues}")


# =========================================================================
# 14. Version bump
# =========================================================================

class TestVersionBump(unittest.TestCase):
    """Verify version is 2.3.0."""

    def test_version_file(self):
        content = _read(VERSION_FILE)
        self.assertIn('"2.4.0"', content)

    def test_app_py_imports_version(self):
        content = _read(APP)
        self.assertIn("from opti_oignon.__version__ import __version__", content)


# =========================================================================
# 15. AST validation for all S121 Python files
# =========================================================================

class TestASTValidation(unittest.TestCase):
    """All S121 Python files pass AST parsing."""

    def test_backup_manager_ast(self):
        ast.parse(_read(BACKUP_MANAGER))

    def test_routes_backup_ast(self):
        ast.parse(_read(ROUTES_BACKUP))

    def test_schemas_ast(self):
        ast.parse(_read(SCHEMAS))

    def test_deps_ast(self):
        ast.parse(_read(DEPS))

    def test_app_ast(self):
        ast.parse(_read(APP))


if __name__ == "__main__":
    unittest.main()
