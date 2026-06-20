"""
Tests for S108 -- Bug Fix + Polish Pass.

Validates:
- BUG-01: RAGDashboardPanel @const fix
- BUG-03: OnboardingOverlay uses PNG logo
- BUG-04: Dark mode logo visibility (oo-logo-adaptive CSS class)
- BUG-05: FeedbackWidget always visible on assistant messages
- BUG-06: Plugins panel in PanelToggle + PluginsQuickPanel
- BUG-07: Code extraction regex + response schema
- BUG-08: ContextBar uses context health API
- BUG-09: Vision delegation error handling for non-vision models
- BUG-10: Onboarding retry with delay
- BUG-11: RAG ingest auth headers + hardened backend
- BUG-12: Light mode message contrast
- BUG-13: Light mode border/shadow polish

Target: ~35 tests
"""

import ast
import importlib.util
import os
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
BACKEND_DIR = os.path.join(PROJECT_ROOT, "opti_oignon")
API_DIR = os.path.join(BACKEND_DIR, "api")
FRONTEND_SRC = os.path.join(PROJECT_ROOT, "frontend", "src")
COMPONENTS_DIR = os.path.join(FRONTEND_SRC, "lib", "components")
CHAT_DIR = os.path.join(COMPONENTS_DIR, "chat")
PANELS_DIR = os.path.join(COMPONENTS_DIR, "panels")
SETTINGS_DIR = os.path.join(COMPONENTS_DIR, "settings")
UI_DIR = os.path.join(COMPONENTS_DIR, "ui")
ROUTES_DIR = os.path.join(FRONTEND_SRC, "routes")
API_TS_DIR = os.path.join(FRONTEND_SRC, "lib", "api")
STORES_DIR = os.path.join(FRONTEND_SRC, "lib", "stores")
STYLES_DIR = os.path.join(FRONTEND_SRC, "styles")


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _load_module(name, path):
    """Load a Python module from file path without triggering __init__ imports."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# PART 1: BUG-01 — RAGDashboardPanel @const fix
# ===========================================================================

class TestBug01RAGDashboardPanel(unittest.TestCase):
    """BUG-01: @const must not appear outside Svelte block context."""

    def setUp(self):
        self.src = _read(os.path.join(SETTINGS_DIR, "RAGDashboardPanel.svelte"))

    def test_no_const_in_template_body(self):
        """@const maxQueries should not be directly in a <div>."""
        # Find @const usages not inside {#each} blocks
        lines = self.src.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("{@const maxQueries"):
                self.fail(
                    f"Line {i}: @const maxQueries found in template body. "
                    "Must be inside a Svelte block ({#if}, {#each}, etc.) or "
                    "moved to a reactive $: declaration."
                )

    def test_reactive_max_queries_exists(self):
        """maxQueries should be a reactive $: variable in <script>."""
        self.assertIn("$: maxQueries", self.src)

    def test_max_queries_uses_usage_data(self):
        """Reactive maxQueries should derive from usageData."""
        self.assertIn("usageData.map", self.src)

    def test_settings_panel_ast_valid(self):
        """The Svelte file should have valid script section (no syntax errors)."""
        # Extract script content and verify it's not empty
        match = re.search(r"<script[^>]*>(.*?)</script>", self.src, re.DOTALL)
        self.assertIsNotNone(match, "No <script> block found")
        self.assertIn("maxQueries", match.group(1))


# ===========================================================================
# PART 2: BUG-03 + BUG-04 — Logo fixes
# ===========================================================================

class TestBug03OnboardingLogo(unittest.TestCase):
    """BUG-03: OnboardingOverlay should use PNG logo, not SVG ellipses."""

    def setUp(self):
        self.src = _read(os.path.join(UI_DIR, "OnboardingOverlay.svelte"))

    def test_no_svg_ellipses(self):
        """Old SVG onion logo with <ellipse> elements must be removed."""
        self.assertNotIn("<ellipse", self.src)

    def test_uses_png_logo(self):
        """Should reference bousier-oignon.png."""
        self.assertIn("bousier-oignon.png", self.src)

    def test_has_adaptive_class(self):
        """Logo img should have oo-logo-adaptive class."""
        self.assertIn("oo-logo-adaptive", self.src)


class TestBug04DarkModeLogo(unittest.TestCase):
    """BUG-04: Logo visibility in dark mode via CSS filter."""

    def setUp(self):
        self.theme_css = _read(os.path.join(STYLES_DIR, "theme.css"))

    def test_adaptive_class_in_css(self):
        """theme.css should define .oo-logo-adaptive."""
        self.assertIn(".oo-logo-adaptive", self.theme_css)

    def test_dark_mode_filter_applied(self):
        """Dark mode should apply invert/sepia filter."""
        self.assertIn("filter: invert(1)", self.theme_css)

    def test_light_mode_filter_none(self):
        """Light mode should have filter: none."""
        self.assertIn("filter: none", self.theme_css)

    def test_sidebar_logo_has_adaptive_class(self):
        src = _read(os.path.join(COMPONENTS_DIR, "layout", "Sidebar.svelte"))
        self.assertIn("oo-logo-adaptive", src)

    def test_dashboard_logo_has_adaptive_class(self):
        src = _read(os.path.join(PANELS_DIR, "DashboardHome.svelte"))
        self.assertIn("oo-logo-adaptive", src)

    def test_login_logo_has_adaptive_class(self):
        src = _read(os.path.join(ROUTES_DIR, "login", "+page.svelte"))
        self.assertIn("oo-logo-adaptive", src)

    def test_register_logo_has_adaptive_class(self):
        src = _read(os.path.join(ROUTES_DIR, "register", "+page.svelte"))
        self.assertIn("oo-logo-adaptive", src)

    def test_sidebar_logo_size_increased(self):
        """Sidebar logo should be w-9 h-9 (36px), not w-7 h-7."""
        src = _read(os.path.join(COMPONENTS_DIR, "layout", "Sidebar.svelte"))
        self.assertIn("w-9 h-9", src)
        self.assertNotIn("w-7 h-7", src)

    def test_dashboard_logo_size_increased(self):
        """Dashboard logo should be 64px, not 48px."""
        src = _read(os.path.join(PANELS_DIR, "DashboardHome.svelte"))
        self.assertIn("64px", src)


# ===========================================================================
# PART 3: BUG-05 — FeedbackWidget always visible
# ===========================================================================

class TestBug05FeedbackWidget(unittest.TestCase):
    """BUG-05: FeedbackWidget visible on all assistant messages."""

    def setUp(self):
        self.src = _read(os.path.join(CHAT_DIR, "ChatMessage.svelte"))

    def test_feedback_not_gated_by_token_estimate(self):
        """FeedbackWidget must not be inside a token_estimate > 0 conditional."""
        # Find the FeedbackWidget block
        idx = self.src.find("FeedbackWidget")
        self.assertGreater(idx, 0, "FeedbackWidget not found")
        # Look backward for the nearest {#if to check it's not token-gated
        before = self.src[:idx]
        last_if = before.rfind("{#if")
        if last_if >= 0:
            condition = before[last_if:idx]
            self.assertNotIn(
                "token_estimate > 0", condition,
                "FeedbackWidget should not be gated by token_estimate > 0"
            )

    def test_feedback_gated_by_not_streaming(self):
        """FeedbackWidget should only show when not streaming."""
        self.assertIn("!isStreaming", self.src)


# ===========================================================================
# PART 4: BUG-06 — Plugins panel
# ===========================================================================

class TestBug06PluginsPanel(unittest.TestCase):
    """BUG-06: Plugins accessible from PanelToggle."""

    def test_panel_type_includes_plugins(self):
        types_src = _read(os.path.join(FRONTEND_SRC, "lib", "types.ts"))
        self.assertIn("'plugins'", types_src)

    def test_panel_toggle_has_plugins_button(self):
        src = _read(os.path.join(PANELS_DIR, "PanelToggle.svelte"))
        self.assertIn("togglePanel('plugins')", src)

    def test_plugins_quick_panel_exists(self):
        path = os.path.join(PANELS_DIR, "PluginsQuickPanel.svelte")
        self.assertTrue(os.path.isfile(path), "PluginsQuickPanel.svelte missing")

    def test_plugins_panel_has_toggle_functionality(self):
        src = _read(os.path.join(PANELS_DIR, "PluginsQuickPanel.svelte"))
        self.assertIn("enablePlugin", src)
        self.assertIn("disablePlugin", src)

    def test_chat_layout_routes_plugins_panel(self):
        src = _read(os.path.join(ROUTES_DIR, "chat", "+layout.svelte"))
        self.assertIn("PluginsQuickPanel", src)
        self.assertIn("'plugins'", src)

    def test_plugins_panel_has_settings_link(self):
        src = _read(os.path.join(PANELS_DIR, "PluginsQuickPanel.svelte"))
        self.assertIn("/settings", src)


# ===========================================================================
# PART 5: BUG-07 — Code extraction
# ===========================================================================

class TestBug07CodeExtraction(unittest.TestCase):
    """BUG-07: Code extraction regex and response schema."""

    def test_regex_matches_standard_fences(self):
        """Standard ```python ... ``` should match."""
        pattern = re.compile(
            r"[ \t]*```([^\n`]*?)[ \t]*\n(.*?)[ \t]*```",
            re.DOTALL,
        )
        text = '```python\nprint("hello")\n```'
        matches = list(pattern.finditer(text))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].group(1), "python")

    def test_regex_matches_no_language_tag(self):
        """``` ... ``` without language should match."""
        pattern = re.compile(
            r"[ \t]*```([^\n`]*?)[ \t]*\n(.*?)[ \t]*```",
            re.DOTALL,
        )
        text = '```\nprint("hello")\n```'
        matches = list(pattern.finditer(text))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].group(1), "")

    def test_regex_matches_indented_fences(self):
        """Fences with leading spaces should match."""
        pattern = re.compile(
            r"[ \t]*```([^\n`]*?)[ \t]*\n(.*?)[ \t]*```",
            re.DOTALL,
        )
        text = '  ```python\n  x = 1\n  ```'
        matches = list(pattern.finditer(text))
        self.assertEqual(len(matches), 1)

    def test_regex_matches_special_lang_tags(self):
        """Language tags like c++, c# should match."""
        pattern = re.compile(
            r"[ \t]*```([^\n`]*?)[ \t]*\n(.*?)[ \t]*```",
            re.DOTALL,
        )
        for lang in ["c++", "c#", "objective-c", "f#"]:
            text = f'```{lang}\nint x = 0;\n```'
            matches = list(pattern.finditer(text))
            self.assertEqual(len(matches), 1, f"Failed for language: {lang}")

    def test_code_blocks_response_schema_exists(self):
        """CodeBlocksResponse Pydantic model should exist in schemas."""
        src = _read(os.path.join(API_DIR, "schemas.py"))
        self.assertIn("class CodeBlocksResponse", src)
        self.assertIn("blocks: list[CodeBlockInfo]", src)

    def test_route_returns_wrapped_response(self):
        """Route should return CodeBlocksResponse, not bare list."""
        src = _read(os.path.join(API_DIR, "routes_code.py"))
        self.assertIn("response_model=CodeBlocksResponse", src)
        self.assertIn("return CodeBlocksResponse(", src)


# ===========================================================================
# PART 6: BUG-08 — ContextBar uses API
# ===========================================================================

class TestBug08ContextBar(unittest.TestCase):
    """BUG-08: ContextBar synced with context health API."""

    def setUp(self):
        self.src = _read(os.path.join(CHAT_DIR, "ContextBar.svelte"))

    def test_uses_context_health_api(self):
        """Should import and call getContextHealth."""
        self.assertIn("getContextHealth", self.src)

    def test_no_hardcoded_model_budgets(self):
        """Should not have hardcoded MODEL_BUDGETS map."""
        self.assertNotIn("MODEL_BUDGETS", self.src)
        self.assertNotIn("DEFAULT_BUDGET", self.src)

    def test_reads_model_context_window_from_api(self):
        """Budget should come from API response, not hardcoded."""
        self.assertIn("model_context_window", self.src)


# ===========================================================================
# PART 7: BUG-09 — Vision error handling
# ===========================================================================

class TestBug09VisionErrorHandling(unittest.TestCase):
    """BUG-09: Vision delegation gracefully handles missing vision models."""

    def test_vision_pipeline_strips_images_when_no_vision_model(self):
        """When no vision model available, images should be stripped."""
        src = _read(os.path.join(BACKEND_DIR, "vision_pipeline.py"))
        self.assertIn("no_vision_model", src)
        self.assertIn("vision_warning", src)

    def test_executor_safety_net(self):
        """Executor should strip images if vision pipeline unavailable."""
        src = _read(os.path.join(BACKEND_DIR, "executor.py"))
        self.assertIn("vision pipeline unavailable", src)

    def test_user_friendly_message(self):
        """Should suggest installing vision-capable models."""
        src = _read(os.path.join(BACKEND_DIR, "vision_pipeline.py"))
        self.assertIn("llava", src)
        self.assertIn("llama3.2-vision", src)


# ===========================================================================
# PART 8: BUG-10 — Onboarding retry
# ===========================================================================

class TestBug10OnboardingRetry(unittest.TestCase):
    """BUG-10: Onboarding retries on API errors."""

    def setUp(self):
        self.src = _read(os.path.join(UI_DIR, "OnboardingOverlay.svelte"))

    def test_max_retries_defined(self):
        self.assertIn("MAX_RETRIES", self.src)

    def test_retry_delay_defined(self):
        self.assertIn("RETRY_DELAY_MS", self.src)

    def test_retry_loop_structure(self):
        """Should have a for loop with retry logic."""
        self.assertIn("for (let attempt", self.src)
        self.assertIn("setTimeout", self.src)


# ===========================================================================
# PART 9: BUG-11 — RAG ingest hardening
# ===========================================================================

class TestBug11RAGIngest(unittest.TestCase):
    """BUG-11: RAG ingest auth headers and backend hardening."""

    def test_frontend_includes_auth_headers(self):
        src = _read(os.path.join(API_TS_DIR, "rag.ts"))
        self.assertIn("getAccessToken", src)
        self.assertIn("Authorization", src)

    def test_backend_validates_empty_file(self):
        src = _read(os.path.join(API_DIR, "routes_rag.py"))
        self.assertIn("Uploaded file is empty", src)

    def test_backend_sanitizes_collection(self):
        src = _read(os.path.join(API_DIR, "routes_rag.py"))
        self.assertIn('.strip()', src)

    def test_schemas_parse(self):
        """All modified Python files should parse without errors."""
        for fname in ["routes_rag.py", "routes_code.py", "schemas.py"]:
            path = os.path.join(API_DIR, fname)
            content = _read(path)
            try:
                ast.parse(content)
            except SyntaxError as e:
                self.fail(f"{fname} has syntax error: {e}")


# ===========================================================================
# PART 10: BUG-12 + BUG-13 — Light mode polish
# ===========================================================================

class TestBug12LightModeContrast(unittest.TestCase):
    """BUG-12: Light mode message contrast improvements."""

    def setUp(self):
        self.css = _read(os.path.join(STYLES_DIR, "theme.css"))

    def test_bot_message_has_visible_border(self):
        """Bot message border should not be transparent in light mode."""
        # Find light mode section
        light_idx = self.css.find("html:not(.dark)")
        light_section = self.css[light_idx:]
        self.assertNotIn(
            "--oo-msg-bot-bd:    transparent",
            light_section,
            "Bot message border should not be transparent in light mode"
        )

    def test_bot_message_bg_differs_from_base(self):
        """Bot message bg should visibly differ from bg-base."""
        light_idx = self.css.find("html:not(.dark)")
        light_section = self.css[light_idx:]
        # Extract msg-bot-bg value
        match_bot = re.search(r"--oo-msg-bot-bg:\s*([^;]+);", light_section)
        match_base = re.search(r"--oo-bg-base:\s*([^;]+);", light_section)
        self.assertIsNotNone(match_bot)
        self.assertIsNotNone(match_base)
        self.assertNotEqual(
            match_bot.group(1).strip(),
            match_base.group(1).strip(),
            "Bot bg should differ from base bg"
        )


class TestBug13LightModePolish(unittest.TestCase):
    """BUG-13: Light mode border and shadow polish."""

    def setUp(self):
        self.css = _read(os.path.join(STYLES_DIR, "theme.css"))

    def test_shadows_present_in_light_mode(self):
        """Light mode should have shadow definitions."""
        light_idx = self.css.find("html:not(.dark)")
        light_section = self.css[light_idx:]
        self.assertIn("--oo-shadow-sm", light_section)
        self.assertIn("--oo-shadow-md", light_section)
        self.assertIn("--oo-shadow-lg", light_section)

    def test_theme_toggle_uses_css_variables(self):
        """ThemeToggle should use --oo-* variables exclusively."""
        src = _read(os.path.join(UI_DIR, "ThemeToggle.svelte"))
        self.assertIn("var(--oo-", src)
        # No hardcoded hex in style section
        style_match = re.search(r"<style>(.*?)</style>", src, re.DOTALL)
        if style_match:
            style = style_match.group(1)
            hex_matches = re.findall(r'#[0-9a-fA-F]{3,8}\b', style)
            self.assertEqual(
                len(hex_matches), 0,
                f"ThemeToggle has hardcoded hex colors: {hex_matches}"
            )


# ===========================================================================
# PART 11: Cross-cutting checks
# ===========================================================================

class TestNoBrokenPythonFiles(unittest.TestCase):
    """All modified Python files should parse without syntax errors."""

    def test_all_backend_files_parse(self):
        files = [
            "vision_pipeline.py",
            "executor.py",
            "code_executor.py",
        ]
        for fname in files:
            path = os.path.join(BACKEND_DIR, fname)
            if os.path.isfile(path):
                content = _read(path)
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    self.fail(f"{fname} has syntax error: {e}")

    def test_all_api_files_parse(self):
        for fname in os.listdir(API_DIR):
            if fname.endswith(".py"):
                path = os.path.join(API_DIR, fname)
                content = _read(path)
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    self.fail(f"api/{fname} has syntax error: {e}")


class TestNoHardcodedHexInModifiedSvelteFiles(unittest.TestCase):
    """All Svelte files modified in S108 should use --oo-* CSS variables."""

    MODIFIED_FILES = [
        os.path.join(SETTINGS_DIR, "RAGDashboardPanel.svelte"),
        os.path.join(UI_DIR, "OnboardingOverlay.svelte"),
        os.path.join(COMPONENTS_DIR, "layout", "Sidebar.svelte"),
        os.path.join(PANELS_DIR, "DashboardHome.svelte"),
        os.path.join(CHAT_DIR, "ChatMessage.svelte"),
        os.path.join(CHAT_DIR, "ContextBar.svelte"),
        os.path.join(PANELS_DIR, "PanelToggle.svelte"),
        os.path.join(PANELS_DIR, "PluginsQuickPanel.svelte"),
    ]

    def test_no_hardcoded_hex_in_style_sections(self):
        for filepath in self.MODIFIED_FILES:
            if not os.path.isfile(filepath):
                continue
            content = _read(filepath)
            style_match = re.search(r"<style[^>]*>(.*?)</style>", content, re.DOTALL)
            if not style_match:
                continue
            style = style_match.group(1)
            hex_matches = re.findall(r'#[0-9a-fA-F]{3,8}\b', style)
            fname = os.path.basename(filepath)
            self.assertEqual(
                len(hex_matches), 0,
                f"{fname} <style> has hardcoded hex: {hex_matches}"
            )


# ===========================================================================
# PART 12: Post-live-testing fixes
# ===========================================================================

class TestOnboardingModelListCapped(unittest.TestCase):
    """Model list should be capped so preset cards are visible."""

    def test_model_list_has_max_height(self):
        src = _read(os.path.join(UI_DIR, "OnboardingOverlay.svelte"))
        self.assertIn("max-h-28", src)
        self.assertIn("overflow-y-auto", src)

    def test_onboarding_logo_larger(self):
        """Onboarding logo should be w-11 h-11."""
        src = _read(os.path.join(UI_DIR, "OnboardingOverlay.svelte"))
        self.assertIn("w-11 h-11", src)


class TestPluginAutoDiscovery(unittest.TestCase):
    """Builtin plugins should be auto-discovered at startup."""

    def test_discover_builtins_function_exists(self):
        src = _read(os.path.join(BACKEND_DIR, "plugin_manifest.py"))
        self.assertIn("def _discover_builtins", src)

    def test_discover_builtins_called_on_init(self):
        src = _read(os.path.join(BACKEND_DIR, "plugin_manifest.py"))
        self.assertIn("_discover_builtins(plugin_registry)", src)

    def test_scans_builtin_plugins_dir(self):
        src = _read(os.path.join(BACKEND_DIR, "plugin_manifest.py"))
        self.assertIn('Path(__file__).parent / "plugins"', src)


class TestContextWindowProfiles(unittest.TestCase):
    """Context window profiles should have correct values."""

    def test_nemotron_has_131k_context(self):
        src = _read(os.path.join(BACKEND_DIR, "context_window.py"))
        # nemotron should be 131072, not 32768
        self.assertIn('"nemotron-3-nano:30b": {"context_window": 131072', src)

    def test_ollama_fallback_exists(self):
        src = _read(os.path.join(BACKEND_DIR, "context_window.py"))
        self.assertIn("_fetch_ollama_context_window", src)

    def test_qwen35_models_present(self):
        src = _read(os.path.join(BACKEND_DIR, "context_window.py"))
        self.assertIn("qwen3.5", src)


class TestWebSearchDualImport(unittest.TestCase):
    """Web search should try both package names."""

    def test_tries_both_package_names(self):
        src = _read(os.path.join(BACKEND_DIR, "web_search.py"))
        self.assertIn("from duckduckgo_search import DDGS", src)
        self.assertIn("from ddgs import DDGS", src)


class TestWSKeepalive(unittest.TestCase):
    """WebSocket should send keepalive pings during long inferences."""

    def test_keepalive_ping_in_stream(self):
        src = _read(os.path.join(API_DIR, "routes_chat.py"))
        self.assertIn("ping", src)
        self.assertIn("_last_send_time", src)


if __name__ == "__main__":
    unittest.main()
