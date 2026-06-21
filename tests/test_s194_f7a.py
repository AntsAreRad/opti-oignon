"""
S194 F7a -- fine-tuning fix lot tests.

Covers:
- FT-02: export/preview paginate the conversation store instead of the
  broken limit=0 call (SQL LIMIT 0 returns zero rows).
- FT-03: A/B comparison inference is bounded by comparison_timeout.
- FT-04: yaml import guarded in fine_tune_export and fine_tune_tracker.
- FT-05: unencrypted-export warning present in FineTunePanel.svelte.
- FT-10: dead ExportPreviewParams schema removed from routes_fine_tune.

Modules are loaded via spec_from_file_location to avoid the package
import chain (executor -> ollama, absent in the container).
"""

import importlib.util
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
BACKEND_DIR = os.path.join(PROJECT_ROOT, "opti_oignon")
API_DIR = os.path.join(BACKEND_DIR, "api")
PANEL_PATH = os.path.join(
    PROJECT_ROOT, "frontend", "src", "lib", "components",
    "settings", "FineTunePanel.svelte",
)


def _load_module(name, path):
    """Load a module from file path without triggering __init__ imports."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


fte = _load_module(
    "s194_fte", os.path.join(BACKEND_DIR, "fine_tune_export.py")
)
ftt = _load_module(
    "s194_ftt", os.path.join(BACKEND_DIR, "fine_tune_tracker.py")
)


class _SqlLikeManager:
    """Conversation manager mirroring the real SQL LIMIT/OFFSET semantics.

    The production manager executes SELECT ... LIMIT ? OFFSET ?, so
    limit=0 returns zero rows. This stub reproduces that exactly.
    """

    def __init__(self, count):
        self._convs = [
            {"id": f"c{i}", "updated_at": "2026-06-01T00:00:00Z", "model": "m"}
            for i in range(count)
        ]
        self.calls = []

    def list_conversations(self, limit=50, offset=0):
        self.calls.append((limit, offset))
        if limit == 0:
            return []
        return self._convs[offset:offset + limit]

    def get_messages(self, cid):
        return [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]


class TestFT02ExportPagination(unittest.TestCase):
    """FT-02: export fetches all conversations via pagination."""

    def test_export_returns_all_conversations(self):
        mgr = _SqlLikeManager(3)
        exporter = fte.FineTuneExporter(conversation_manager=mgr)
        result = exporter.export(fmt="sharegpt")
        self.assertEqual(result.conversation_count, 3)
        self.assertGreater(result.message_count, 0)

    def test_preview_sees_all_conversations(self):
        mgr = _SqlLikeManager(3)
        exporter = fte.FineTuneExporter(conversation_manager=mgr)
        preview = exporter.preview(fmt="jsonl")
        self.assertEqual(preview["total_conversations"], 3)

    def test_no_limit_zero_call_issued(self):
        mgr = _SqlLikeManager(2)
        exporter = fte.FineTuneExporter(conversation_manager=mgr)
        exporter.export(fmt="jsonl")
        self.assertTrue(all(limit != 0 for limit, _ in mgr.calls))

    def test_pagination_crosses_chunk_boundary(self):
        mgr = _SqlLikeManager(502)
        exporter = fte.FineTuneExporter(conversation_manager=mgr)
        result = exporter.export(fmt="jsonl")
        self.assertEqual(result.conversation_count, 502)
        offsets = [offset for _, offset in mgr.calls]
        self.assertIn(500, offsets)


class TestFT03ComparisonTimeout(unittest.TestCase):
    """FT-03: _timed_inference enforces comparison_timeout."""

    def setUp(self):
        self.tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp_db.close()
        self.tmp_cfg = tempfile.NamedTemporaryFile(
            suffix=".yaml", delete=False, mode="w", encoding="utf-8"
        )
        self.tmp_cfg.write("tracking:\n  comparison_timeout: 0.5\n")
        self.tmp_cfg.close()
        self.tracker = ftt.FineTuneTracker(
            db_path=Path(self.tmp_db.name),
            config_path=Path(self.tmp_cfg.name),
        )

    def tearDown(self):
        os.unlink(self.tmp_db.name)
        os.unlink(self.tmp_cfg.name)

    def test_timeout_config_loaded(self):
        self.assertEqual(self.tracker.comparison_timeout, 0.5)

    def test_hung_inference_is_abandoned(self):
        def hung(model, prompt):
            time.sleep(5)
            return "late"

        start = time.perf_counter()
        response, latency_ms = self.tracker._timed_inference(hung, "m", "p")
        elapsed = time.perf_counter() - start
        self.assertIn("timeout", response)
        self.assertLess(elapsed, 3.0)
        self.assertGreaterEqual(latency_ms, 400.0)

    def test_fast_inference_unaffected(self):
        response, latency_ms = self.tracker._timed_inference(
            lambda m, p: "ok", "m", "p"
        )
        self.assertEqual(response, "ok")
        self.assertLess(latency_ms, 500.0)

    def test_run_comparison_completes_with_timeouts(self):
        v = self.tracker.register_variant(
            ftt.FineTuneVariant(
                name="t", base_model="base:1b", variant_model="ft:1b"
            )
        )
        comp = self.tracker.create_comparison(v.variant_id, ["q"])

        def hung(model, prompt):
            time.sleep(5)
            return "late"

        result = self.tracker.run_comparison(
            comp.comparison_id, inference_fn=hung
        )
        self.assertEqual(result.status, "completed")
        self.assertIn("timeout", result.prompts[0].base_response)


class TestFT04YamlGuard(unittest.TestCase):
    """FT-04: yaml imports are guarded in both fine-tune modules."""

    def test_export_module_guard(self):
        src = _read(os.path.join(BACKEND_DIR, "fine_tune_export.py"))
        self.assertIn("YAML_AVAILABLE = True", src)
        self.assertIn("except ImportError", src)
        self.assertTrue(hasattr(fte, "YAML_AVAILABLE"))

    def test_tracker_module_guard(self):
        src = _read(os.path.join(BACKEND_DIR, "fine_tune_tracker.py"))
        self.assertIn("YAML_AVAILABLE = True", src)
        self.assertIn("except ImportError", src)
        self.assertTrue(hasattr(ftt, "YAML_AVAILABLE"))

    def test_config_loaders_check_flag(self):
        for fname in ("fine_tune_export.py", "fine_tune_tracker.py"):
            src = _read(os.path.join(BACKEND_DIR, fname))
            self.assertIn("if YAML_AVAILABLE and self._config_path.exists()", src)


class TestFT05ExportWarning(unittest.TestCase):
    """FT-05: FineTunePanel surfaces the unencrypted-export warning."""

    def test_warning_present(self):
        src = _read(PANEL_PATH)
        self.assertIn("This export is unencrypted", src)
        self.assertIn("conversation content", src)

    def test_warning_uses_design_tokens(self):
        src = _read(PANEL_PATH)
        self.assertIn("--oo-warning-bg", src)
        self.assertIn("--oo-warning-bd", src)


class TestFT10DeadSchemaRemoved(unittest.TestCase):
    """FT-10: ExportPreviewParams no longer exists in routes_fine_tune."""

    def test_schema_removed(self):
        src = _read(os.path.join(API_DIR, "routes_fine_tune.py"))
        self.assertNotIn("ExportPreviewParams", src)

    def test_preview_endpoint_intact(self):
        src = _read(os.path.join(API_DIR, "routes_fine_tune.py"))
        self.assertIn('@router.get("/export/preview")', src)


if __name__ == "__main__":
    unittest.main()
