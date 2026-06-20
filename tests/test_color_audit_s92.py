"""
test_color_audit_s92.py -- Tests for the S92 color audit scanner.
"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Load the scanner module directly from file path
_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "audit_colors.py"
_spec = importlib.util.spec_from_file_location("audit_colors", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
audit_colors = importlib.util.module_from_spec(_spec)
sys.modules["audit_colors"] = audit_colors
_spec.loader.exec_module(audit_colors)


class TestHexDetection:
    """Hardcoded hex color detection."""

    def test_detects_6digit_hex(self, tmp_path: Path) -> None:
        svelte = tmp_path / "Test.svelte"
        svelte.write_text('<div style="color: #f87171;">red</div>')
        violations = audit_colors.scan_file(svelte)
        hex_v = [v for v in violations if v["type"] == "hardcoded_hex"]
        assert len(hex_v) >= 1
        assert hex_v[0]["match"] == "#f87171"

    def test_detects_3digit_hex(self, tmp_path: Path) -> None:
        svelte = tmp_path / "Test.svelte"
        svelte.write_text('<span style="color: #fff;">text</span>')
        violations = audit_colors.scan_file(svelte)
        hex_v = [v for v in violations if v["type"] == "hardcoded_hex"]
        assert len(hex_v) >= 1

    def test_ignores_css_variable_hex(self, tmp_path: Path) -> None:
        svelte = tmp_path / "Test.svelte"
        svelte.write_text('<div style="color: var(--oo-fg-primary);">ok</div>')
        violations = audit_colors.scan_file(svelte)
        hex_v = [v for v in violations if v["type"] == "hardcoded_hex"]
        assert len(hex_v) == 0

    def test_ignores_svelte_each(self, tmp_path: Path) -> None:
        svelte = tmp_path / "Test.svelte"
        svelte.write_text("{#each items as item}\n<div>{item}</div>\n{/each}")
        violations = audit_colors.scan_file(svelte)
        hex_v = [v for v in violations if v["type"] == "hardcoded_hex"]
        assert len(hex_v) == 0

    def test_ignores_html_entity(self, tmp_path: Path) -> None:
        svelte = tmp_path / "Test.svelte"
        svelte.write_text("<span>&#9733; star</span>")
        violations = audit_colors.scan_file(svelte)
        hex_v = [v for v in violations if v["type"] == "hardcoded_hex"]
        assert len(hex_v) == 0

    def test_detects_hex_in_script(self, tmp_path: Path) -> None:
        svelte = tmp_path / "Test.svelte"
        svelte.write_text("<script>\nconst c = '#B07D56';\n</script>")
        violations = audit_colors.scan_file(svelte)
        hex_v = [v for v in violations if v["type"] == "hardcoded_hex"]
        assert len(hex_v) >= 1


class TestRgbaDetection:
    """Inline rgba detection."""

    def test_detects_inline_rgba(self, tmp_path: Path) -> None:
        svelte = tmp_path / "Test.svelte"
        svelte.write_text('<div style="background-color: rgba(239, 68, 68, 0.12);">err</div>')
        violations = audit_colors.scan_file(svelte)
        rgba_v = [v for v in violations if v["type"] == "inline_rgba"]
        assert len(rgba_v) >= 1

    def test_allows_rgba_in_var_fallback(self, tmp_path: Path) -> None:
        svelte = tmp_path / "Test.svelte"
        svelte.write_text(
            '<div style="background: var(--oo-error-bg, rgba(239, 68, 68, 0.12));">ok</div>'
        )
        violations = audit_colors.scan_file(svelte)
        rgba_v = [v for v in violations if v["type"] == "inline_rgba"]
        assert len(rgba_v) == 0

    def test_detects_rgb_no_alpha(self, tmp_path: Path) -> None:
        svelte = tmp_path / "Test.svelte"
        svelte.write_text('<div style="color: rgb(255, 0, 0);">red</div>')
        violations = audit_colors.scan_file(svelte)
        rgba_v = [v for v in violations if v["type"] == "inline_rgba"]
        assert len(rgba_v) >= 1


class TestWhiteDetection:
    """Hardcoded 'white' keyword detection."""

    def test_detects_bg_white(self, tmp_path: Path) -> None:
        svelte = tmp_path / "Test.svelte"
        svelte.write_text('<div style="background-color: white;">toggle</div>')
        violations = audit_colors.scan_file(svelte)
        white_v = [v for v in violations if v["type"] == "inline_white"]
        assert len(white_v) >= 1

    def test_no_false_positive_for_var(self, tmp_path: Path) -> None:
        svelte = tmp_path / "Test.svelte"
        svelte.write_text('<div style="background-color: var(--oo-bg-elevated);">ok</div>')
        violations = audit_colors.scan_file(svelte)
        white_v = [v for v in violations if v["type"] == "inline_white"]
        assert len(white_v) == 0


class TestTailwindColorDetection:
    """Tailwind color utility class detection."""

    def test_detects_text_red(self, tmp_path: Path) -> None:
        svelte = tmp_path / "Test.svelte"
        svelte.write_text('<span class="text-red-400">Error</span>')
        violations = audit_colors.scan_file(svelte)
        tw_v = [v for v in violations if v["type"] == "tailwind_color"]
        assert len(tw_v) >= 1
        assert tw_v[0]["match"] == "text-red-400"

    def test_detects_bg_green(self, tmp_path: Path) -> None:
        svelte = tmp_path / "Test.svelte"
        svelte.write_text('<div class="bg-green-500">OK</div>')
        violations = audit_colors.scan_file(svelte)
        tw_v = [v for v in violations if v["type"] == "tailwind_color"]
        assert len(tw_v) >= 1

    def test_detects_border_amber(self, tmp_path: Path) -> None:
        svelte = tmp_path / "Test.svelte"
        svelte.write_text('<div class="border-amber-800/40">Warn</div>')
        violations = audit_colors.scan_file(svelte)
        tw_v = [v for v in violations if v["type"] == "tailwind_color"]
        assert len(tw_v) >= 1

    def test_ignores_surface_classes(self, tmp_path: Path) -> None:
        svelte = tmp_path / "Test.svelte"
        svelte.write_text('<div class="bg-surface-800 text-surface-200">OK</div>')
        violations = audit_colors.scan_file(svelte)
        tw_v = [v for v in violations if v["type"] == "tailwind_color"]
        assert len(tw_v) == 0

    def test_ignores_accent_classes(self, tmp_path: Path) -> None:
        svelte = tmp_path / "Test.svelte"
        svelte.write_text('<div class="text-accent-400 bg-accent-600">Accent</div>')
        violations = audit_colors.scan_file(svelte)
        tw_v = [v for v in violations if v["type"] == "tailwind_color"]
        assert len(tw_v) == 0

    def test_detects_text_white(self, tmp_path: Path) -> None:
        svelte = tmp_path / "Test.svelte"
        svelte.write_text('<span class="text-white">Label</span>')
        violations = audit_colors.scan_file(svelte)
        named_v = [v for v in violations if v["type"] == "tailwind_named_color"]
        assert len(named_v) >= 1

    def test_detects_multiple_classes(self, tmp_path: Path) -> None:
        svelte = tmp_path / "Test.svelte"
        svelte.write_text(
            '<div class="bg-red-900/30 text-red-300 border-red-800">Error block</div>'
        )
        violations = audit_colors.scan_file(svelte)
        tw_v = [v for v in violations if v["type"] == "tailwind_color"]
        assert len(tw_v) >= 3


class TestSuggestFix:
    """Fix suggestion generation."""

    def test_hex_suggestion(self) -> None:
        fix = audit_colors.suggest_fix("hardcoded_hex", "#ff0000")
        assert "var(--oo-" in fix

    def test_rgba_suggestion(self) -> None:
        fix = audit_colors.suggest_fix("inline_rgba", "rgba(255, 0, 0, 0.5)")
        assert "var(--oo-" in fix

    def test_tailwind_red(self) -> None:
        fix = audit_colors.suggest_fix("tailwind_color", "text-red-400")
        assert "--oo-error" in fix

    def test_tailwind_green(self) -> None:
        fix = audit_colors.suggest_fix("tailwind_color", "bg-green-500")
        assert "--oo-success" in fix

    def test_tailwind_amber(self) -> None:
        fix = audit_colors.suggest_fix("tailwind_color", "text-amber-400")
        assert "--oo-warning" in fix

    def test_tailwind_blue(self) -> None:
        fix = audit_colors.suggest_fix("tailwind_color", "bg-blue-500")
        assert "--oo-info" in fix


class TestDirectoryScan:
    """Directory-level scanning."""

    def test_scans_multiple_files(self, tmp_path: Path) -> None:
        (tmp_path / "A.svelte").write_text('<div class="text-red-400">a</div>')
        (tmp_path / "B.svelte").write_text('<div class="bg-green-500">b</div>')
        violations = audit_colors.scan_directory(tmp_path)
        assert len(violations) >= 2
        files = {v["file"] for v in violations}
        assert len(files) == 2

    def test_ignores_non_svelte(self, tmp_path: Path) -> None:
        (tmp_path / "style.css").write_text("color: #ff0000;")
        violations = audit_colors.scan_directory(tmp_path)
        assert len(violations) == 0


class TestSummary:
    """Summary generation."""

    def test_summary_structure(self) -> None:
        violations = [
            {"file": "a.svelte", "line": 1, "type": "hardcoded_hex", "match": "#fff", "context": "", "fix": ""},
            {"file": "a.svelte", "line": 2, "type": "tailwind_color", "match": "text-red-400", "context": "", "fix": ""},
            {"file": "b.svelte", "line": 5, "type": "tailwind_color", "match": "bg-green-500", "context": "", "fix": ""},
        ]
        summary = audit_colors.make_summary(violations)
        assert summary["total_violations"] == 3
        assert summary["files_with_violations"] == 2
        assert summary["by_type"]["tailwind_color"] == 2
        assert summary["by_type"]["hardcoded_hex"] == 1

    def test_empty_violations(self) -> None:
        summary = audit_colors.make_summary([])
        assert summary["total_violations"] == 0
        assert summary["files_with_violations"] == 0
