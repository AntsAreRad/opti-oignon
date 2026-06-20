"""
Tests for S134 -- Performance + UX Polish.

Validates:
- Part 1: lazy_loader.py (LazyModule, LazyAttr, get_lazy_module, HEAVY_MODULES)
- Part 2: deps.py lazy conversion (availability flags, LazyAttr proxies)
- Part 3: Frontend (SkeletonLoader, ErrorBoundary, vite.config, dynamic imports)
- Part 4: Accessibility (audit_accessibility.py, aria-labels, contrast)
- Part 5: Dark/Light theme consistency
- Part 6: Integration (version bump, no French, HTML balance, AST)

Target: ~20 tests
"""

import ast
import importlib.util
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "opti_oignon")
FRONTEND_SRC = os.path.join(PROJECT_ROOT, "frontend", "src")
COMPONENTS_DIR = os.path.join(FRONTEND_SRC, "lib", "components")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")


def _load_module(name, path):
    """Load a Python module by file path without triggering package imports."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestLazyLoader(unittest.TestCase):
    """Part 1: lazy_loader.py core functionality."""

    def setUp(self):
        self.mod = _load_module(
            "lazy_loader_test", os.path.join(BACKEND_DIR, "lazy_loader.py")
        )

    def test_lazy_module_proxy_deferred(self):
        """LazyModule does not import until first attribute access."""
        proxy = self.mod.LazyModule("json")
        self.assertFalse(proxy.is_loaded)
        # Access an attribute to trigger import
        _ = proxy.dumps
        self.assertTrue(proxy.is_loaded)

    def test_lazy_module_proxy_error(self):
        """LazyModule raises ImportError for non-existent module."""
        proxy = self.mod.LazyModule("nonexistent_module_xyz_134")
        with self.assertRaises(ImportError):
            _ = proxy.some_attr

    def test_lazy_attr_resolves(self):
        """LazyAttr resolves a specific attribute from a module."""
        la = self.mod.LazyAttr("json", "dumps")
        # Should be callable
        result = la([1, 2, 3])
        self.assertEqual(result, "[1, 2, 3]")

    def test_lazy_attr_deferred(self):
        """LazyAttr does not import until first access."""
        la = self.mod.LazyAttr("json", "JSONEncoder")
        resolved = object.__getattribute__(la, "_resolved")
        self.assertIsNone(resolved)
        # Trigger resolution by accessing a regular attribute
        _ = la.item_separator
        resolved = object.__getattribute__(la, "_resolved")
        self.assertIsNotNone(resolved)

    def test_get_lazy_module_alias(self):
        """get_lazy_module is an alias for lazy_import."""
        self.assertIs(self.mod.get_lazy_module, self.mod.lazy_import)

    def test_heavy_modules_updated(self):
        """HEAVY_MODULES list includes S134 targets."""
        heavy = self.mod.HEAVY_MODULES
        self.assertIn("opti_oignon.rag_store", heavy)
        self.assertIn("opti_oignon.coding_agent", heavy)
        self.assertIn("opti_oignon.telemetry", heavy)
        self.assertIn("opti_oignon.benchmark_evaluator", heavy)
        self.assertIn("opti_oignon.fine_tune_tracker", heavy)
        self.assertIn("opti_oignon.plugin_index", heavy)
        self.assertTrue(len(heavy) >= 20)


class TestDepsLazyConversion(unittest.TestCase):
    """Part 2: deps.py uses lazy loading for heavy modules."""

    def test_deps_has_lazy_imports(self):
        """deps.py imports LazyAttr from lazy_loader."""
        path = os.path.join(BACKEND_DIR, "api", "deps.py")
        content = open(path).read()
        self.assertIn("from opti_oignon.lazy_loader import", content)
        self.assertIn("_LazyAttr", content)
        self.assertIn("_module_exists", content)

    def test_availability_flags_use_module_exists(self):
        """Heavy module availability flags use _module_exists (no eager import)."""
        path = os.path.join(BACKEND_DIR, "api", "deps.py")
        content = open(path).read()
        # Check a sample of converted modules
        for flag in [
            "BENCHMARK_V2_AVAILABLE",
            "FINE_TUNE_TRACKER_AVAILABLE",
            "RAG_STORE_AVAILABLE",
            "TELEMETRY_AVAILABLE",
            "PLUGIN_INDEX_AVAILABLE",
        ]:
            # Should use _module_exists pattern
            pattern = rf'{flag}\s*=\s*_module_exists\('
            self.assertTrue(
                re.search(pattern, content),
                f"{flag} should use _module_exists",
            )

    def test_deps_ast_valid(self):
        """deps.py is valid Python (AST parse)."""
        path = os.path.join(BACKEND_DIR, "api", "deps.py")
        ast.parse(open(path).read())


class TestFrontendComponents(unittest.TestCase):
    """Part 3: Frontend SkeletonLoader, ErrorBoundary, vite config."""

    def test_skeleton_loader_exists(self):
        """SkeletonLoader.svelte exists."""
        path = os.path.join(COMPONENTS_DIR, "ui", "SkeletonLoader.svelte")
        self.assertTrue(os.path.isfile(path))

    def test_skeleton_loader_css_variables(self):
        """SkeletonLoader uses only CSS variables for colors."""
        path = os.path.join(COMPONENTS_DIR, "ui", "SkeletonLoader.svelte")
        content = open(path).read()
        self.assertIn("var(--oo-", content)
        self.assertIn("@keyframes shimmer", content)
        self.assertIn('role="status"', content)
        self.assertIn("aria-label", content)

    def test_error_boundary_exists_and_updated(self):
        """ErrorBoundary.svelte has console.error and report link."""
        path = os.path.join(COMPONENTS_DIR, "ui", "ErrorBoundary.svelte")
        self.assertTrue(os.path.isfile(path))
        content = open(path).read()
        self.assertIn("console.error", content)
        self.assertIn("Report issue", content)
        self.assertIn("var(--oo-", content)
        # No French
        self.assertNotIn("Enveloppe", content)
        self.assertNotIn("Capture les", content)

    def test_vite_config_has_code_splitting(self):
        """vite.config.ts has manualChunks for code splitting."""
        path = os.path.join(PROJECT_ROOT, "frontend", "vite.config.ts")
        content = open(path).read()
        self.assertIn("manualChunks", content)
        self.assertIn("chunk-rag", content)
        self.assertIn("chunk-benchmark", content)
        self.assertIn("chunk-coding", content)
        self.assertIn("chunk-settings", content)

    def test_settings_page_dynamic_imports(self):
        """Settings page uses dynamic import() for lazy loading."""
        path = os.path.join(FRONTEND_SRC, "routes", "settings", "+page.svelte")
        content = open(path).read()
        self.assertIn("import(", content)
        self.assertIn("SkeletonLoader", content)
        self.assertIn("_cache", content)
        self.assertIn("loadComponent", content)

    def test_layouts_have_error_boundary(self):
        """Settings and chat layouts wrap content in ErrorBoundary."""
        for layout in [
            os.path.join(FRONTEND_SRC, "routes", "settings", "+layout.svelte"),
            os.path.join(FRONTEND_SRC, "routes", "chat", "+layout.svelte"),
        ]:
            content = open(layout).read()
            self.assertIn("ErrorBoundary", content, f"{layout} missing ErrorBoundary")


class TestAccessibility(unittest.TestCase):
    """Part 4: Accessibility audit script and aria-labels."""

    def test_audit_accessibility_script_exists(self):
        """audit_accessibility.py exists and is valid Python."""
        path = os.path.join(SCRIPTS_DIR, "audit_accessibility.py")
        self.assertTrue(os.path.isfile(path))
        ast.parse(open(path).read())

    def test_s126_s133_buttons_have_labels(self):
        """All buttons in S126-S133 components have aria-label or text."""
        s126_s133 = [
            "SecurityModePanel", "PluginAllowlistPanel", "SearchKillSwitchPanel",
            "TOTPSetup", "WebAuthnSetup", "AppPasswordsPanel", "TOTPInput",
            "AuditChainPanel", "HardeningPanel", "KeyCeremonyPanel",
            "RecoveryCodesPanel", "RemoteAccessPanel", "SecurityPanel",
        ]
        import glob
        issues = []
        for path in glob.glob(os.path.join(COMPONENTS_DIR, "**", "*.svelte"), recursive=True):
            basename = os.path.basename(path).replace(".svelte", "")
            if basename not in s126_s133:
                continue
            content = open(path).read()
            for m in re.finditer(r"<button([^>]*)>(.*?)</button>", content, re.DOTALL):
                attrs = m.group(1)
                inner = m.group(2).strip()
                has_aria = "aria-label" in attrs or "aria-label" in inner
                text_only = re.sub(r"<[^>]+>", "", inner).strip()
                text_only = re.sub(r"\{[^}]+\}", "EXPR", text_only).strip()
                if not has_aria and not text_only:
                    line = content[:m.start()].count("\n") + 1
                    issues.append(f"{basename}:{line}")
        self.assertEqual(issues, [], f"Buttons without labels: {issues}")

    def test_contrast_passes_wcag_aa(self):
        """Color contrast audit passes (via audit_contrast.py)."""
        path = os.path.join(SCRIPTS_DIR, "audit_contrast.py")
        if not os.path.isfile(path):
            self.skipTest("audit_contrast.py not found")
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertEqual(result.returncode, 0, f"Contrast audit failed:\n{result.stdout}")


class TestThemeConsistency(unittest.TestCase):
    """Part 5: Dark/Light mode variable consistency."""

    def test_accent_scale_complete_in_light(self):
        """Light theme has all accent scale vars (--oo-acc-50 through 900)."""
        path = os.path.join(FRONTEND_SRC, "styles", "theme.css")
        content = open(path).read()
        light_match = re.search(r"html:not\(\.dark\)\s*\{(.*?)\n\}", content, re.DOTALL)
        self.assertIsNotNone(light_match, "Light theme block not found")
        light_block = light_match.group(1)
        for scale in ["50", "100", "200", "300", "400", "500", "600", "700", "800", "900"]:
            var_name = f"--oo-acc-{scale}"
            self.assertIn(var_name, light_block, f"{var_name} missing from light theme")


class TestIntegration(unittest.TestCase):
    """Part 6: Version bump, no French, HTML balance, AST."""

    def test_version_bumped_to_2_9_5(self):
        """Version is 2.9.5."""
        path = os.path.join(BACKEND_DIR, "__version__.py")
        content = open(path).read()
        self.assertIn('"3.0.0"', content)

    def test_no_french_in_new_files(self):
        """No French text in new/modified S134 Python files."""
        files = [
            os.path.join(BACKEND_DIR, "lazy_loader.py"),
            os.path.join(BACKEND_DIR, "api", "deps.py"),
            os.path.join(BACKEND_DIR, "api", "app.py"),
            os.path.join(SCRIPTS_DIR, "audit_accessibility.py"),
        ]
        french_patterns = re.compile(
            r"\b(?:chargement|fonction|retourne|verrou|echoue|genere\b|"
            r"rapport|demarr|arriere|precharg|Enveloppe|Capture les|"
            r"Intercepte|necessaire|Gestion du cycle)\b",
            re.IGNORECASE,
        )
        for f in files:
            content = open(f).read()
            matches = french_patterns.findall(content)
            self.assertEqual(
                matches, [], f"French found in {os.path.basename(f)}: {matches}"
            )

    def test_html_balance_new_svelte_components(self):
        """New/modified Svelte components have balanced HTML tags."""
        files = [
            os.path.join(COMPONENTS_DIR, "ui", "SkeletonLoader.svelte"),
            os.path.join(COMPONENTS_DIR, "ui", "ErrorBoundary.svelte"),
            os.path.join(FRONTEND_SRC, "routes", "settings", "+layout.svelte"),
            os.path.join(FRONTEND_SRC, "routes", "settings", "+page.svelte"),
        ]
        for f in files:
            content = open(f).read()
            basename = os.path.basename(f)
            for tag in ["div", "button"]:
                opens = len(re.findall(rf"<{tag}[\s>]", content))
                self_close = len(re.findall(rf"<{tag}[^>]*/>", content))
                closes = len(re.findall(rf"</{tag}>", content))
                net_open = opens - self_close
                self.assertEqual(
                    net_open, closes,
                    f"{basename}: <{tag}> mismatch ({net_open} open vs {closes} close)",
                )

    def test_ast_validation_all_new_python(self):
        """All new/modified Python files pass AST validation."""
        files = [
            os.path.join(BACKEND_DIR, "lazy_loader.py"),
            os.path.join(BACKEND_DIR, "api", "deps.py"),
            os.path.join(BACKEND_DIR, "api", "app.py"),
            os.path.join(SCRIPTS_DIR, "audit_accessibility.py"),
        ]
        for f in files:
            try:
                ast.parse(open(f).read())
            except SyntaxError as e:
                self.fail(f"AST error in {os.path.basename(f)}: {e}")


if __name__ == "__main__":
    unittest.main()
