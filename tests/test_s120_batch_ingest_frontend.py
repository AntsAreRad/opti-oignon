#!/usr/bin/env python3
"""
Tests for S120 -- RAG Batch Ingestion Frontend (Upload UI + Document Manager).

Test groups:
1. BatchUpload.svelte: structure, drag-and-drop zone, file validation, collection selector
2. IngestProgress.svelte: polling, per-file status, progress bar, cancel button
3. DocumentManager.svelte: search, filters, pagination, bulk delete, table structure
4. FolderScan.svelte: path input, recursive toggle, IngestProgress reuse
5. KnowledgeBasePanel.svelte: S120 integration (new tabs, imports, component wiring)
6. API client (rag.ts): new batch functions, enhanced listDocuments
7. TypeScript types: new batch ingest interfaces
8. CSS variable compliance: no hardcoded hex in any S120 component
9. HTML tag balance: all Svelte components
10. Version bump: 2.3.0 across all version files

Target: ~70 tests
"""

import ast
import os
import re
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend" / "src"
RAG_COMPONENTS = FRONTEND / "lib" / "components" / "rag"
SETTINGS_COMPONENTS = FRONTEND / "lib" / "components" / "settings"
API_DIR = FRONTEND / "lib" / "api"
TYPES_FILE = FRONTEND / "lib" / "types.ts"
BACKEND_DIR = ROOT / "opti_oignon"
VERSION_FILE = BACKEND_DIR / "__version__.py"


def _read(path: Path) -> str:
    """Read a file as UTF-8 text."""
    return path.read_text(encoding="utf-8")


def _find_hex_colors(content: str) -> list[str]:
    """Find hex color values that are NOT CSS variable fallbacks or HTML entities."""
    # Find all # followed by hex chars
    matches = re.findall(r"#[0-9a-fA-F]{3,8}\b", content)
    # Filter out HTML entities (&#10005; etc.) and CSS var fallbacks
    real_hex = []
    for m in matches:
        # Check if preceded by "var(--oo-..., " (fallback) or "&#" (entity)
        # Simple heuristic: if it's 3-8 pure hex after # and NOT an entity pattern
        idx = content.find(m)
        if idx > 0 and content[idx - 1] == "&":
            continue  # HTML entity like &#10005;
        # Check if inside a var() fallback
        preceding = content[max(0, idx - 60):idx]
        if re.search(r"var\(--oo-[a-zA-Z0-9-]+,\s*$", preceding):
            continue  # CSS variable fallback — allowed
        real_hex.append(m)
    return real_hex


def _check_html_balance(content: str) -> dict:
    """Check HTML tag balance in a Svelte file (after </script>)."""
    parts = content.split("</script>")
    if len(parts) < 2:
        return {"balanced": False, "error": "No </script> found"}
    html = parts[-1]

    void_tags = {"input", "br", "hr", "img", "meta", "link", "area", "base",
                 "col", "embed", "source", "track", "wbr"}
    # Only check lowercase tags (HTML), skip PascalCase (Svelte components)
    open_tags = [t for t in re.findall(r"<([a-z][a-z0-9]*)\b[^/]*?(?<!/)\s*>", html)
                 if t not in void_tags]
    close_tags = re.findall(r"</([a-z][a-z0-9]*)\s*>", html)

    from collections import Counter
    opens = Counter(open_tags)
    closes = Counter(close_tags)
    mismatches = {}
    for tag in set(list(opens.keys()) + list(closes.keys())):
        if opens.get(tag, 0) != closes.get(tag, 0):
            mismatches[tag] = (opens.get(tag, 0), closes.get(tag, 0))

    return {"balanced": len(mismatches) == 0, "mismatches": mismatches,
            "open_count": sum(opens.values()), "close_count": sum(closes.values())}


# =========================================================================
# 1. BatchUpload.svelte
# =========================================================================

class TestBatchUploadComponent(unittest.TestCase):
    """Validate BatchUpload.svelte structure and content."""

    @classmethod
    def setUpClass(cls):
        cls.path = RAG_COMPONENTS / "BatchUpload.svelte"
        cls.content = _read(cls.path)

    def test_file_exists(self):
        self.assertTrue(self.path.exists(), "BatchUpload.svelte must exist")

    def test_has_script_section(self):
        self.assertIn("<script lang=\"ts\">", self.content)

    def test_imports_ingest_batch(self):
        self.assertIn("ingestBatch", self.content)

    def test_imports_list_collections(self):
        self.assertIn("listCollections", self.content)

    def test_imports_create_collection(self):
        self.assertIn("createCollection", self.content)

    def test_has_drag_drop_zone(self):
        self.assertIn("on:drop", self.content)
        self.assertIn("on:dragover", self.content)
        self.assertIn("on:dragleave", self.content)

    def test_has_file_input(self):
        self.assertIn('type="file"', self.content)
        self.assertIn("multiple", self.content)

    def test_has_collection_selector(self):
        self.assertIn("data-testid=\"batch-collection-select\"", self.content)

    def test_has_new_collection_input(self):
        self.assertIn("data-testid=\"batch-new-collection-input\"", self.content)

    def test_has_upload_button(self):
        self.assertIn("data-testid=\"batch-upload-btn\"", self.content)

    def test_has_file_queue(self):
        self.assertIn("data-testid=\"batch-file-queue\"", self.content)

    def test_has_remove_file_button(self):
        self.assertIn("data-testid=\"batch-remove-file-btn\"", self.content)

    def test_has_clear_queue_button(self):
        self.assertIn("data-testid=\"batch-clear-queue-btn\"", self.content)

    def test_supported_extensions_defined(self):
        self.assertIn("SUPPORTED_EXTENSIONS", self.content)
        self.assertIn(".pdf", self.content)
        self.assertIn(".docx", self.content)
        self.assertIn(".xlsx", self.content)

    def test_max_file_size_defined(self):
        self.assertIn("MAX_FILE_SIZE_MB", self.content)

    def test_dispatches_job_started_event(self):
        self.assertIn("jobStarted", self.content)
        self.assertIn("dispatch", self.content)

    def test_validate_file_function(self):
        self.assertIn("validateFile", self.content)

    def test_file_size_display(self):
        self.assertIn("formatSize", self.content)


# =========================================================================
# 2. IngestProgress.svelte
# =========================================================================

class TestIngestProgressComponent(unittest.TestCase):
    """Validate IngestProgress.svelte structure and content."""

    @classmethod
    def setUpClass(cls):
        cls.path = RAG_COMPONENTS / "IngestProgress.svelte"
        cls.content = _read(cls.path)

    def test_file_exists(self):
        self.assertTrue(self.path.exists())

    def test_imports_get_ingest_job(self):
        self.assertIn("getIngestJob", self.content)

    def test_imports_delete_ingest_job(self):
        self.assertIn("deleteIngestJob", self.content)

    def test_has_job_id_prop(self):
        self.assertIn("export let jobId", self.content)

    def test_has_initial_job_prop(self):
        self.assertIn("export let initialJob", self.content)

    def test_poll_interval_defined(self):
        self.assertIn("POLL_INTERVAL_MS", self.content)

    def test_terminal_statuses_defined(self):
        self.assertIn("TERMINAL_STATUSES", self.content)
        self.assertIn("completed", self.content)
        self.assertIn("failed", self.content)
        self.assertIn("cancelled", self.content)

    def test_has_progress_bar(self):
        self.assertIn("data-testid=\"ingest-progress-bar\"", self.content)

    def test_has_cancel_button(self):
        self.assertIn("data-testid=\"ingest-cancel-btn\"", self.content)

    def test_has_file_status_icons(self):
        self.assertIn("fileStatusIcon", self.content)

    def test_has_file_list(self):
        self.assertIn("data-testid=\"ingest-file-list\"", self.content)

    def test_has_status_display(self):
        self.assertIn("data-testid=\"ingest-progress-status\"", self.content)

    def test_dispatches_completed_event(self):
        self.assertIn("dispatch('completed'", self.content)

    def test_dispatches_cancelled_event(self):
        self.assertIn("dispatch('cancelled'", self.content)

    def test_has_on_destroy_cleanup(self):
        self.assertIn("onDestroy", self.content)
        self.assertIn("stopPolling", self.content)

    def test_per_file_status_colors(self):
        """Each file status should have a distinct color mapping."""
        self.assertIn("fileStatusColor", self.content)
        for status in ["queued", "processing", "done", "error", "skipped"]:
            self.assertIn(f"'{status}'", self.content)


# =========================================================================
# 3. DocumentManager.svelte
# =========================================================================

class TestDocumentManagerComponent(unittest.TestCase):
    """Validate DocumentManager.svelte structure and content."""

    @classmethod
    def setUpClass(cls):
        cls.path = RAG_COMPONENTS / "DocumentManager.svelte"
        cls.content = _read(cls.path)

    def test_file_exists(self):
        self.assertTrue(self.path.exists())

    def test_has_search_input(self):
        self.assertIn("data-testid=\"doc-search-input\"", self.content)

    def test_has_filetype_filter(self):
        self.assertIn("data-testid=\"doc-filetype-filter\"", self.content)

    def test_has_collection_filter(self):
        self.assertIn("data-testid=\"doc-collection-filter\"", self.content)

    def test_has_document_table(self):
        self.assertIn("data-testid=\"doc-table\"", self.content)

    def test_has_pagination(self):
        self.assertIn("data-testid=\"doc-pagination\"", self.content)

    def test_has_select_all_checkbox(self):
        self.assertIn("data-testid=\"doc-select-all\"", self.content)

    def test_has_row_checkboxes(self):
        self.assertIn("data-testid=\"doc-row-checkbox\"", self.content)

    def test_has_bulk_delete_button(self):
        self.assertIn("data-testid=\"doc-bulk-delete-btn\"", self.content)

    def test_has_per_row_delete(self):
        self.assertIn("data-testid=\"doc-row-delete-btn\"", self.content)

    def test_has_total_count_display(self):
        self.assertIn("data-testid=\"doc-total-count\"", self.content)

    def test_exports_refresh_function(self):
        self.assertIn("export function refresh()", self.content)

    def test_search_debounce(self):
        self.assertIn("searchDebounceTimer", self.content)
        self.assertIn("setTimeout", self.content)

    def test_page_size_defined(self):
        self.assertIn("PAGE_SIZE", self.content)

    def test_file_type_labels(self):
        self.assertIn("FILE_TYPE_LABELS", self.content)
        for ft in ["PDF", "Word", "Excel", "CSV", "Markdown"]:
            self.assertIn(ft, self.content)

    def test_uses_list_documents_with_search(self):
        """API call should pass search param."""
        self.assertIn("search:", self.content)
        self.assertIn("file_type:", self.content)

    def test_grid_template_columns(self):
        """Table should use CSS grid layout."""
        self.assertIn("grid-template-columns", self.content)


# =========================================================================
# 4. FolderScan.svelte
# =========================================================================

class TestFolderScanComponent(unittest.TestCase):
    """Validate FolderScan.svelte structure and content."""

    @classmethod
    def setUpClass(cls):
        cls.path = RAG_COMPONENTS / "FolderScan.svelte"
        cls.content = _read(cls.path)

    def test_file_exists(self):
        self.assertTrue(self.path.exists())

    def test_imports_ingest_folder(self):
        self.assertIn("ingestFolder", self.content)

    def test_imports_ingest_progress(self):
        self.assertIn("IngestProgress", self.content)

    def test_has_path_input(self):
        self.assertIn("data-testid=\"folder-path-input\"", self.content)

    def test_has_recursive_toggle(self):
        self.assertIn("data-testid=\"folder-recursive-toggle\"", self.content)

    def test_has_collection_selector(self):
        self.assertIn("data-testid=\"folder-collection-select\"", self.content)

    def test_has_scan_button(self):
        self.assertIn("data-testid=\"folder-scan-btn\"", self.content)

    def test_dispatches_job_started(self):
        self.assertIn("jobStarted", self.content)

    def test_reuses_ingest_progress(self):
        """Must render IngestProgress component for active job."""
        self.assertIn("<IngestProgress", self.content)

    def test_has_clear_button(self):
        self.assertIn("data-testid=\"folder-clear-job-btn\"", self.content)


# =========================================================================
# 5. KnowledgeBasePanel.svelte Integration
# =========================================================================

class TestKnowledgeBasePanelIntegration(unittest.TestCase):
    """Validate S120 integration into KnowledgeBasePanel."""

    @classmethod
    def setUpClass(cls):
        cls.path = SETTINGS_COMPONENTS / "KnowledgeBasePanel.svelte"
        cls.content = _read(cls.path)

    def test_imports_batch_upload(self):
        self.assertIn("import BatchUpload from", self.content)

    def test_imports_ingest_progress(self):
        self.assertIn("import IngestProgress from", self.content)

    def test_imports_document_manager(self):
        self.assertIn("import DocumentManager from", self.content)

    def test_imports_folder_scan(self):
        self.assertIn("import FolderScan from", self.content)

    def test_has_batch_upload_tab(self):
        self.assertIn("batch-upload", self.content)
        self.assertIn("Batch Upload", self.content)

    def test_has_documents_tab(self):
        self.assertIn("'documents'", self.content)

    def test_has_folder_scan_tab(self):
        self.assertIn("folder-scan", self.content)
        self.assertIn("Folder Scan", self.content)

    def test_renders_batch_upload_component(self):
        self.assertIn("<BatchUpload", self.content)

    def test_renders_document_manager_component(self):
        self.assertIn("<DocumentManager", self.content)

    def test_renders_folder_scan_component(self):
        self.assertIn("<FolderScan", self.content)

    def test_active_jobs_tracking(self):
        self.assertIn("activeJobs", self.content)

    def test_doc_manager_ref(self):
        self.assertIn("docManagerRef", self.content)
        self.assertIn("bind:this={docManagerRef}", self.content)

    def test_no_old_upload_functions(self):
        """Old S99 drag-and-drop functions should be removed."""
        self.assertNotIn("handleFileUpload", self.content)
        self.assertNotIn("handleDrop", self.content)
        self.assertNotIn("handleDragOver", self.content)

    def test_no_old_doc_list(self):
        """Old inline document list should be removed."""
        self.assertNotIn("handleDeleteDoc", self.content)
        self.assertNotIn("docsLoading", self.content)


# =========================================================================
# 6. API Client (rag.ts)
# =========================================================================

class TestRagApiClient(unittest.TestCase):
    """Validate S120 additions to rag.ts."""

    @classmethod
    def setUpClass(cls):
        cls.path = API_DIR / "rag.ts"
        cls.content = _read(cls.path)

    def test_ingest_batch_function(self):
        self.assertIn("export async function ingestBatch(", self.content)

    def test_ingest_folder_function(self):
        self.assertIn("export async function ingestFolder(", self.content)

    def test_list_ingest_jobs_function(self):
        self.assertIn("export async function listIngestJobs(", self.content)

    def test_get_ingest_job_function(self):
        self.assertIn("export async function getIngestJob(", self.content)

    def test_delete_ingest_job_function(self):
        self.assertIn("export async function deleteIngestJob(", self.content)

    def test_ingest_batch_uses_form_data(self):
        self.assertIn("FormData", self.content)
        self.assertIn("formData.append('files'", self.content)

    def test_ingest_batch_endpoint(self):
        self.assertIn("/api/rag/ingest/batch", self.content)

    def test_ingest_folder_endpoint(self):
        self.assertIn("/api/rag/ingest/folder", self.content)

    def test_list_jobs_endpoint(self):
        self.assertIn("/api/rag/ingest/jobs", self.content)

    def test_list_documents_has_search_param(self):
        self.assertIn("if (params?.search)", self.content)

    def test_list_documents_has_file_type_param(self):
        self.assertIn("if (params?.file_type)", self.content)

    def test_imports_new_types(self):
        self.assertIn("RAGIngestJob", self.content)
        self.assertIn("RAGIngestJobsListResponse", self.content)
        self.assertIn("RAGIngestJobDeleteResponse", self.content)
        self.assertIn("RAGFolderScanRequest", self.content)


# =========================================================================
# 7. TypeScript Types
# =========================================================================

class TestTypescriptTypes(unittest.TestCase):
    """Validate S120 type definitions in types.ts."""

    @classmethod
    def setUpClass(cls):
        cls.content = _read(TYPES_FILE)

    def test_rag_ingest_file_status_interface(self):
        self.assertIn("export interface RAGIngestFileStatus", self.content)

    def test_rag_ingest_job_interface(self):
        self.assertIn("export interface RAGIngestJob", self.content)

    def test_rag_ingest_jobs_list_response(self):
        self.assertIn("export interface RAGIngestJobsListResponse", self.content)

    def test_rag_ingest_job_delete_response(self):
        self.assertIn("export interface RAGIngestJobDeleteResponse", self.content)

    def test_rag_folder_scan_request(self):
        self.assertIn("export interface RAGFolderScanRequest", self.content)

    def test_ingest_file_status_fields(self):
        """RAGIngestFileStatus must have key fields."""
        # Extract interface body
        match = re.search(
            r"export interface RAGIngestFileStatus\s*\{([^}]+)\}",
            self.content, re.DOTALL
        )
        self.assertIsNotNone(match, "RAGIngestFileStatus interface not found")
        body = match.group(1)
        for field in ["file_id", "job_id", "filename", "status", "chunk_count", "error_message"]:
            self.assertIn(field, body, f"Missing field: {field}")

    def test_ingest_job_fields(self):
        """RAGIngestJob must have key fields."""
        match = re.search(
            r"export interface RAGIngestJob\s*\{([^}]+)\}",
            self.content, re.DOTALL
        )
        self.assertIsNotNone(match, "RAGIngestJob interface not found")
        body = match.group(1)
        for field in ["job_id", "status", "collection", "progress", "total_files",
                       "completed_files", "failed_files", "files"]:
            self.assertIn(field, body, f"Missing field: {field}")

    def test_ingest_job_status_type(self):
        """Status should be a string union type."""
        self.assertIn("'pending'", self.content)
        self.assertIn("'running'", self.content)
        self.assertIn("'completed'", self.content)


# =========================================================================
# 8. CSS Variable Compliance
# =========================================================================

class TestCSSVariableCompliance(unittest.TestCase):
    """Ensure all S120 components use --oo-* CSS variables exclusively."""

    S120_FILES = [
        RAG_COMPONENTS / "BatchUpload.svelte",
        RAG_COMPONENTS / "IngestProgress.svelte",
        RAG_COMPONENTS / "DocumentManager.svelte",
        RAG_COMPONENTS / "FolderScan.svelte",
    ]

    def test_no_hardcoded_hex_in_batch_upload(self):
        content = _read(self.S120_FILES[0])
        hexes = _find_hex_colors(content)
        self.assertEqual(hexes, [], f"Hardcoded hex in BatchUpload: {hexes}")

    def test_no_hardcoded_hex_in_ingest_progress(self):
        content = _read(self.S120_FILES[1])
        hexes = _find_hex_colors(content)
        self.assertEqual(hexes, [], f"Hardcoded hex in IngestProgress: {hexes}")

    def test_no_hardcoded_hex_in_document_manager(self):
        content = _read(self.S120_FILES[2])
        hexes = _find_hex_colors(content)
        self.assertEqual(hexes, [], f"Hardcoded hex in DocumentManager: {hexes}")

    def test_no_hardcoded_hex_in_folder_scan(self):
        content = _read(self.S120_FILES[3])
        hexes = _find_hex_colors(content)
        self.assertEqual(hexes, [], f"Hardcoded hex in FolderScan: {hexes}")

    def test_all_components_use_oo_vars(self):
        """Every S120 component should reference at least 5 --oo-* variables."""
        for path in self.S120_FILES:
            content = _read(path)
            count = len(re.findall(r"var\(--oo-", content))
            self.assertGreaterEqual(
                count, 5,
                f"{path.name} has only {count} --oo-* var references"
            )


# =========================================================================
# 9. HTML Tag Balance
# =========================================================================

class TestHTMLTagBalance(unittest.TestCase):
    """Verify balanced HTML tags in all S120 Svelte components."""

    S120_FILES = [
        RAG_COMPONENTS / "BatchUpload.svelte",
        RAG_COMPONENTS / "IngestProgress.svelte",
        RAG_COMPONENTS / "DocumentManager.svelte",
        RAG_COMPONENTS / "FolderScan.svelte",
        SETTINGS_COMPONENTS / "KnowledgeBasePanel.svelte",
    ]

    def test_batch_upload_balanced(self):
        result = _check_html_balance(_read(self.S120_FILES[0]))
        self.assertTrue(result["balanced"], f"BatchUpload: {result.get('mismatches')}")

    def test_ingest_progress_balanced(self):
        result = _check_html_balance(_read(self.S120_FILES[1]))
        self.assertTrue(result["balanced"], f"IngestProgress: {result.get('mismatches')}")

    def test_document_manager_balanced(self):
        result = _check_html_balance(_read(self.S120_FILES[2]))
        self.assertTrue(result["balanced"], f"DocumentManager: {result.get('mismatches')}")

    def test_folder_scan_balanced(self):
        result = _check_html_balance(_read(self.S120_FILES[3]))
        self.assertTrue(result["balanced"], f"FolderScan: {result.get('mismatches')}")

    def test_knowledge_base_panel_balanced(self):
        result = _check_html_balance(_read(self.S120_FILES[4]))
        self.assertTrue(result["balanced"], f"KnowledgeBasePanel: {result.get('mismatches')}")


# =========================================================================
# 10. Version Bump
# =========================================================================

class TestVersionBump(unittest.TestCase):
    """Verify version is 2.3.0 across all files."""

    def test_version_file(self):
        content = _read(VERSION_FILE)
        self.assertIn('"2.3.1"', content)

    def test_app_py_imports_version(self):
        """app.py should import from __version__.py (dynamic)."""
        path = BACKEND_DIR / "api" / "app.py"
        content = _read(path)
        self.assertIn("from opti_oignon.__version__ import __version__", content)

    def test_init_imports_version(self):
        """__init__.py should import from __version__.py (dynamic)."""
        content = _read(BACKEND_DIR / "__init__.py")
        self.assertIn("from .__version__ import __version__", content)


# =========================================================================
# 11. English Only
# =========================================================================

class TestEnglishOnly(unittest.TestCase):
    """No French text in S120 components."""

    S120_FILES = [
        RAG_COMPONENTS / "BatchUpload.svelte",
        RAG_COMPONENTS / "IngestProgress.svelte",
        RAG_COMPONENTS / "DocumentManager.svelte",
        RAG_COMPONENTS / "FolderScan.svelte",
    ]

    FRENCH_WORDS = ["fichier", "telechar", "ajouter", "supprim", "dossier",
                    "recherch", "telecharg", "annuler"]

    def test_no_french_in_components(self):
        for path in self.S120_FILES:
            content = _read(path).lower()
            for word in self.FRENCH_WORDS:
                self.assertNotIn(
                    word, content,
                    f"French word '{word}' found in {path.name}"
                )


# =========================================================================
# 12. Component Directory Structure
# =========================================================================

class TestComponentDirectoryStructure(unittest.TestCase):
    """Verify rag/ component directory exists with all expected files."""

    def test_rag_directory_exists(self):
        self.assertTrue(RAG_COMPONENTS.is_dir())

    def test_batch_upload_exists(self):
        self.assertTrue((RAG_COMPONENTS / "BatchUpload.svelte").exists())

    def test_ingest_progress_exists(self):
        self.assertTrue((RAG_COMPONENTS / "IngestProgress.svelte").exists())

    def test_document_manager_exists(self):
        self.assertTrue((RAG_COMPONENTS / "DocumentManager.svelte").exists())

    def test_folder_scan_exists(self):
        self.assertTrue((RAG_COMPONENTS / "FolderScan.svelte").exists())


if __name__ == "__main__":
    unittest.main()
