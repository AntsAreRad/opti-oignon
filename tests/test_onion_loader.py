#!/usr/bin/env python3
"""
Tests for OnionLoader.svelte — Opti-Oignon S76

Structural tests for the OnionLoader component: file presence,
SVG structure, CSS keyframes, props, accessibility, and integration
in StreamingIndicator, BenchmarkRunner, and CodingAgentPanel.
"""

import os
import re
import unittest

_COMPONENT_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "frontend", "src", "lib", "components",
)
_CHAT_DIR = os.path.join(_COMPONENT_DIR, "chat")
_PANELS_DIR = os.path.join(_COMPONENT_DIR, "panels")

_LOADER_PATH = os.path.join(_COMPONENT_DIR, "OnionLoader.svelte")
_STREAMING_PATH = os.path.join(_CHAT_DIR, "StreamingIndicator.svelte")
_BENCHMARK_PATH = os.path.join(_PANELS_DIR, "BenchmarkRunner.svelte")
_CODING_PATH = os.path.join(_PANELS_DIR, "CodingAgentPanel.svelte")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


class TestOnionLoaderFileExists(unittest.TestCase):
    """OnionLoader.svelte must exist at the expected path."""

    def test_file_exists(self):
        self.assertTrue(
            os.path.isfile(_LOADER_PATH),
            f"OnionLoader.svelte not found at {_LOADER_PATH}",
        )


class TestOnionLoaderProps(unittest.TestCase):
    """OnionLoader exports size, color, and label props with defaults."""

    def setUp(self):
        self.src = _read(_LOADER_PATH)

    def test_size_prop(self):
        self.assertIn("export let size", self.src)

    def test_size_default(self):
        self.assertRegex(self.src, r"export\s+let\s+size.*=\s*24")

    def test_color_prop(self):
        self.assertIn("export let color", self.src)

    def test_color_default(self):
        # S94: default changed from acc-500 to tobacco
        self.assertIn("var(--oo-tobacco)", self.src)

    def test_label_prop(self):
        self.assertIn("export let label", self.src)


class TestOnionLoaderSVG(unittest.TestCase):
    """OnionLoader must render the onion as line art SVG."""

    def setUp(self):
        self.src = _read(_LOADER_PATH)

    def test_contains_svg_element(self):
        self.assertIn("<svg", self.src)
        self.assertIn("</svg>", self.src)

    def test_viewbox_1024(self):
        self.assertIn('viewBox="0 0 1024 1024"', self.src)

    def test_line_art_stroke(self):
        # All paths should use fill="none" stroke=
        self.assertIn('fill="none"', self.src)
        self.assertIn("stroke=", self.src)

    def test_multiple_layers(self):
        # At least 3 path elements (outer, middle, inner, core)
        paths = re.findall(r"<path\s", self.src)
        self.assertGreaterEqual(len(paths), 3, "Expected at least 3 SVG paths for onion layers")

    def test_stroke_uses_css_variable(self):
        self.assertIn("var(--ol-color)", self.src)


class TestOnionLoaderCSS(unittest.TestCase):
    """OnionLoader uses CSS keyframes only, no JS animation loop."""

    def setUp(self):
        self.src = _read(_LOADER_PATH)

    def test_no_js_animation(self):
        # Should not have requestAnimationFrame or setInterval for animation
        self.assertNotIn("requestAnimationFrame", self.src)
        self.assertNotIn("setInterval", self.src)

    def test_has_rotate_keyframe(self):
        self.assertIn("@keyframes onion-rotate", self.src)

    def test_has_bump_keyframe(self):
        self.assertIn("@keyframes onion-bump", self.src)

    def test_no_road_scroll_keyframe(self):
        # S94: road track removed for cleaner look
        self.assertNotIn("@keyframes road-scroll", self.src)

    def test_rotateY_animation(self):
        self.assertIn("rotateY", self.src)

    def test_perspective(self):
        self.assertIn("perspective", self.src)

    def test_squash_on_landing(self):
        # Squash effect uses scaleX/scaleY or scale()
        self.assertIn("scale(", self.src)

    def test_translateY_bounce(self):
        self.assertIn("translateY(", self.src)


class TestOnionLoaderRoad(unittest.TestCase):
    """S94: Road track removed for cleaner look."""

    def setUp(self):
        self.src = _read(_LOADER_PATH)

    def test_road_track_element(self):
        self.assertNotIn("road-track", self.src)

    def test_road_line_element(self):
        self.assertNotIn("road-line", self.src)

    def test_road_gradient_removed(self):
        self.assertNotIn("repeating-linear-gradient", self.src)


class TestOnionLoaderAccessibility(unittest.TestCase):
    """OnionLoader must have proper accessibility attributes."""

    def setUp(self):
        self.src = _read(_LOADER_PATH)

    def test_role_status(self):
        self.assertIn('role="status"', self.src)

    def test_aria_label(self):
        self.assertIn('aria-label="Loading"', self.src)

    def test_svg_aria_hidden(self):
        self.assertIn('aria-hidden="true"', self.src)


class TestOnionLoaderCustomProperties(unittest.TestCase):
    """OnionLoader uses CSS custom properties for size and color."""

    def setUp(self):
        self.src = _read(_LOADER_PATH)

    def test_ol_size_variable(self):
        self.assertIn("--ol-size", self.src)

    def test_ol_color_variable(self):
        self.assertIn("--ol-color", self.src)

    def test_size_binding(self):
        self.assertIn("{size}px", self.src)

    def test_color_binding(self):
        self.assertIn("{color}", self.src)


class TestStreamingIndicatorIntegration(unittest.TestCase):
    """StreamingIndicator must import and use OnionLoader."""

    def setUp(self):
        self.src = _read(_STREAMING_PATH)

    def test_imports_onion_loader(self):
        self.assertIn("import OnionLoader", self.src)

    def test_uses_onion_loader(self):
        self.assertIn("<OnionLoader", self.src)

    def test_no_bouncing_dots(self):
        # Old dots spinner should be removed
        self.assertNotIn("animate-bounce", self.src)

    def test_no_model_display(self):
        # S94: model name removed from streaming indicator
        self.assertNotIn("export let model", self.src)


class TestBenchmarkRunnerIntegration(unittest.TestCase):
    """BenchmarkRunner must import and use OnionLoader during progress."""

    def setUp(self):
        self.src = _read(_BENCHMARK_PATH)

    def test_imports_onion_loader(self):
        self.assertIn("import OnionLoader", self.src)

    def test_uses_onion_loader(self):
        self.assertIn("<OnionLoader", self.src)


class TestCodingAgentPanelIntegration(unittest.TestCase):
    """CodingAgentPanel must import and use OnionLoader with label."""

    def setUp(self):
        self.src = _read(_CODING_PATH)

    def test_imports_onion_loader(self):
        self.assertIn("import OnionLoader", self.src)

    def test_uses_onion_loader(self):
        self.assertIn("<OnionLoader", self.src)

    def test_loader_has_label(self):
        # At least one OnionLoader usage includes a label prop
        self.assertRegex(self.src, r"<OnionLoader[^>]*label=")

    def test_loader_has_phase_label(self):
        self.assertIn('label={phase}', self.src)


class TestCodingAgentPanelHistory(unittest.TestCase):
    """CodingAgentPanel must have task history sidebar elements."""

    def setUp(self):
        self.src = _read(_CODING_PATH)

    def test_history_toggle_button(self):
        self.assertIn("Task History", self.src)

    def test_fetch_history_function(self):
        self.assertIn("fetchHistory", self.src)

    def test_resume_function(self):
        self.assertIn("resumeTask", self.src)

    def test_delete_function(self):
        self.assertIn("deleteHistory", self.src)

    def test_history_api_endpoint(self):
        self.assertIn("/api/coding/history", self.src)

    def test_resume_api_endpoint(self):
        self.assertIn("/api/coding/resume/", self.src)

    def test_history_item_class(self):
        self.assertIn("history-item", self.src)

    def test_resume_button(self):
        self.assertIn("btn-resume", self.src)


class TestAssetsSVG(unittest.TestCase):
    """The onion SVG asset must exist and have expected structure."""

    _ASSET_PATH = os.path.join(
        os.path.dirname(__file__), os.pardir, "assets", "opti-oignon.svg",
    )

    def test_asset_exists(self):
        self.assertTrue(os.path.isfile(self._ASSET_PATH))

    def test_asset_is_svg(self):
        src = _read(self._ASSET_PATH)
        self.assertIn("<svg", src)

    def test_asset_has_paths(self):
        src = _read(self._ASSET_PATH)
        paths = re.findall(r"<path\s", src)
        self.assertGreaterEqual(len(paths), 3)


if __name__ == "__main__":
    unittest.main()
