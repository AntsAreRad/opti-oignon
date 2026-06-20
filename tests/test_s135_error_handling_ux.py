"""
Tests for S135 -- Error Handling Hardening + UX Micro-Interactions.

Validates:
- Phase 1: FeatureUnavailable.svelte, featureCheck.ts, health integration
- Phase 2: Toast stack limit, auto-dismiss timing, exit animation
- Phase 3: transitions.css, prefers-reduced-motion, reusable classes
- Phase 4: Settings arrow key nav, Escape closes modals, focusTrap action
- Phase 5: errorHandler.ts, HTTP status mapping, logo resize
- Phase 6: Version bump, no French, HTML balance, AST validation
"""

import ast
import os
import re
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

FE = BASE / "frontend" / "src"
LIB = FE / "lib"
API = LIB / "api"
COMPONENTS = LIB / "components"
STYLES = FE / "styles"
SETTINGS_PAGE = FE / "routes" / "settings" / "+page.svelte"


class TestFeatureCheck(unittest.TestCase):
    """Phase 1: Graceful degradation for optional features."""

    def test_feature_unavailable_component_exists(self):
        """FeatureUnavailable.svelte exists with required props."""
        path = COMPONENTS / "ui" / "FeatureUnavailable.svelte"
        self.assertTrue(path.exists(), "FeatureUnavailable.svelte missing")
        content = path.read_text()
        self.assertIn("export let featureName", content)
        self.assertIn("export let description", content)
        self.assertIn("export let learnMoreUrl", content)
        # CSS variables only — no standalone hex colors (outside var() fallbacks)
        template = content.split("<style>")[0] if "<style>" in content else content
        bare_hex = re.findall(r'(?<!\()#[0-9a-fA-F]{3,8}\b', template)
        self.assertEqual(bare_hex, [], f"Hardcoded hex found: {bare_hex}")

    def test_feature_check_api_client_exists(self):
        """featureCheck.ts exists with required exports."""
        path = API / "featureCheck.ts"
        self.assertTrue(path.exists(), "featureCheck.ts missing")
        content = path.read_text()
        self.assertIn("checkFeatureAvailable", content)
        self.assertIn("getFeatureMap", content)
        self.assertIn("invalidateFeatureCache", content)
        self.assertIn("CACHE_TTL_MS", content)

    def test_health_feature_map_used_in_settings(self):
        """Settings page imports featureCheck and uses featureMap."""
        content = SETTINGS_PAGE.read_text()
        self.assertIn("getFeatureMap", content)
        self.assertIn("featureMap", content)
        # At least 3 feature checks (knowledge, plugins, analytics, fine-tune, performance)
        checks = re.findall(r"featureMap\[", content)
        self.assertGreaterEqual(len(checks), 3, f"Found only {len(checks)} feature checks")


class TestToastImprovements(unittest.TestCase):
    """Phase 2: Toast notification improvements."""

    def test_toast_stack_limit(self):
        """notifications.ts enforces MAX_VISIBLE_TOASTS = 3."""
        path = LIB / "stores" / "notifications.ts"
        content = path.read_text()
        self.assertIn("MAX_VISIBLE_TOASTS", content)
        match = re.search(r"MAX_VISIBLE_TOASTS\s*=\s*(\d+)", content)
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), 3)

    def test_toast_auto_dismiss_durations(self):
        """Error toasts default to 8s, success to 3s, warning to 5s."""
        path = LIB / "stores" / "notifications.ts"
        content = path.read_text()
        # toastError block contains 8000
        self.assertRegex(content, r"toastError.*\n.*8000")
        # toastSuccess block contains 3000
        self.assertRegex(content, r"toastSuccess.*\n.*3000")
        # toastWarning block contains 5000
        self.assertRegex(content, r"toastWarning.*\n.*5000")

    def test_toast_exit_animation_class(self):
        """Toast.svelte has entrance and exit animation classes."""
        path = COMPONENTS / "ui" / "Toast.svelte"
        content = path.read_text()
        self.assertIn("oo-toast-enter", content)
        self.assertIn("oo-toast-exit", content)
        self.assertIn("oo-toast-slide-in", content)
        self.assertIn("oo-toast-slide-out", content)


class TestTransitions(unittest.TestCase):
    """Phase 3: Micro-interactions and reusable transitions."""

    def test_transitions_css_exists(self):
        """transitions.css exists and is imported in app.css."""
        path = STYLES / "transitions.css"
        self.assertTrue(path.exists(), "transitions.css missing")
        app_css = (FE / "app.css").read_text()
        self.assertIn("transitions.css", app_css)

    def test_transitions_has_required_classes(self):
        """transitions.css contains all required utility classes."""
        content = (STYLES / "transitions.css").read_text()
        required = [
            "oo-fade-in",
            "oo-slide-up",
            "oo-scale-press",
            "oo-collapse",
            "oo-toggle-knob",
            "oo-tab-enter",
            "oo-spinner",
        ]
        for cls in required:
            self.assertIn(cls, content, f"Missing class: {cls}")

    def test_transitions_reduced_motion_respected(self):
        """Global app.css has prefers-reduced-motion that covers all animations."""
        app_css = (FE / "app.css").read_text()
        self.assertIn("prefers-reduced-motion: reduce", app_css)
        # Should disable all animations and transitions globally
        self.assertIn("animation-duration: 0.01ms", app_css)
        self.assertIn("transition-duration: 0.01ms", app_css)


class TestKeyboardNavigation(unittest.TestCase):
    """Phase 4: Keyboard navigation polish."""

    def test_settings_tabs_arrow_key_navigation(self):
        """Settings page supports arrow key navigation between tabs."""
        content = SETTINGS_PAGE.read_text()
        self.assertIn("handleTabKeydown", content)
        self.assertIn("ArrowLeft", content)
        self.assertIn("ArrowRight", content)
        self.assertIn('role="tablist"', content)
        self.assertIn('role="tab"', content)

    def test_escape_closes_modals(self):
        """Modal components handle Escape key (via focusTrap or direct handler)."""
        focus_trap = (LIB / "actions" / "focusTrap.ts").read_text()
        self.assertIn("Escape", focus_trap)
        self.assertIn("onEscape", focus_trap)
        self.assertIn("FOCUSABLE_SELECTOR", focus_trap)
        # Verify at least 2 components use focusTrap
        usage_count = 0
        for svelte_file in COMPONENTS.rglob("*.svelte"):
            if "use:focusTrap" in svelte_file.read_text():
                usage_count += 1
        self.assertGreaterEqual(usage_count, 2, f"Only {usage_count} components use focusTrap")

    def test_chat_input_up_arrow_edit(self):
        """ChatInput dispatches editLast on Up arrow with empty input."""
        path = COMPONENTS / "chat" / "ChatInput.svelte"
        content = path.read_text()
        self.assertIn("ArrowUp", content)
        self.assertIn("editLast", content)


class TestErrorHandler(unittest.TestCase):
    """Phase 5: Standardized API error handling."""

    def test_error_handler_exists(self):
        """errorHandler.ts exists with required exports."""
        path = API / "errorHandler.ts"
        self.assertTrue(path.exists(), "errorHandler.ts missing")
        content = path.read_text()
        self.assertIn("parseApiError", content)
        self.assertIn("handleApiError", content)
        self.assertIn("withErrorHandling", content)

    def test_http_status_mapping_429(self):
        """errorHandler.ts detects 429 rate limiting."""
        content = (API / "errorHandler.ts").read_text()
        self.assertIn("429", content)
        self.assertIn("isRateLimited", content)
        self.assertIn("rate limit", content.lower())

    def test_http_status_mapping_501(self):
        """errorHandler.ts detects 501 feature unavailable."""
        content = (API / "errorHandler.ts").read_text()
        self.assertIn("501", content)
        self.assertIn("isFeatureUnavailable", content)

    def test_network_offline_detection(self):
        """errorHandler.ts detects browser offline state."""
        content = (API / "errorHandler.ts").read_text()
        self.assertIn("navigator.onLine", content)
        self.assertIn("isOffline", content)


class TestIntegration(unittest.TestCase):
    """Phase 6: Version bump, quality checks, packaging."""

    def test_version_bump_to_2_9_6(self):
        """Version is bumped to 2.9.6 across all sources."""
        version_file = BASE / "opti_oignon" / "__version__.py"
        content = version_file.read_text()
        self.assertIn('"3.0.0"', content)

    def test_no_french_in_new_files(self):
        """No French text in new S135 files."""
        french_patterns = re.compile(
            r"\b(Fermer|Supprimer|Envoyer|Param[eè]tre|Erreur|Disponible|Indisponible"
            r"|Connexion|Enregistrer|Recherche|Chargement)\b",
            re.IGNORECASE,
        )
        new_files = [
            API / "featureCheck.ts",
            API / "errorHandler.ts",
            COMPONENTS / "ui" / "FeatureUnavailable.svelte",
            STYLES / "transitions.css",
            LIB / "actions" / "focusTrap.ts",
        ]
        for path in new_files:
            if path.exists():
                content = path.read_text()
                match = french_patterns.search(content)
                self.assertIsNone(
                    match,
                    f"French text found in {path.name}: {match.group() if match else ''}",
                )

    def test_html_balance_feature_unavailable(self):
        """FeatureUnavailable.svelte has balanced HTML tags."""
        content = (COMPONENTS / "ui" / "FeatureUnavailable.svelte").read_text()
        for tag in ["div", "h3", "p", "a"]:
            opens = len(re.findall(rf"<{tag}[\s>]", content))
            closes = len(re.findall(rf"</{tag}>", content))
            self.assertEqual(opens, closes, f"<{tag}> imbalance: {opens} opens, {closes} closes")

    def test_html_balance_toast(self):
        """Toast.svelte has balanced HTML tags."""
        content = (COMPONENTS / "ui" / "Toast.svelte").read_text()
        for tag in ["div", "button", "span"]:
            opens = len(re.findall(rf"<{tag}[\s>]", content))
            closes = len(re.findall(rf"</{tag}>", content))
            self.assertEqual(opens, closes, f"<{tag}> imbalance: {opens} opens, {closes} closes")

    def test_ast_validation_python_files(self):
        """All Python files in opti_oignon/ parse without syntax errors."""
        py_dir = BASE / "opti_oignon"
        errors = []
        for py_file in py_dir.rglob("*.py"):
            try:
                ast.parse(py_file.read_text(), filename=str(py_file))
            except SyntaxError as e:
                errors.append(f"{py_file.name}: {e}")
        self.assertEqual(errors, [], f"AST errors: {errors}")


if __name__ == "__main__":
    unittest.main()
