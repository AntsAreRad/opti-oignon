#!/usr/bin/env python3
"""
Tests for S149 — Frontend E2E Tests (Playwright)
==================================================

Covers:
- Part 1:  Playwright config file exists and is valid TypeScript
- Part 2:  E2E directory structure correct
- Part 3:  run_e2e.sh script exists and is executable
- Part 4:  Mock data file present and contains expected exports
- Part 5:  Mock routes file present and contains expected functions
- Part 6:  Mock index barrel export present
- Part 7:  Fixtures file present with test extensions
- Part 8:  E2E spec files present (7 scenarios + visual)
- Part 9:  E2E spec files are valid (no syntax issues)
- Part 10: package.json has Playwright dependency and scripts
- Part 11: Visual regression spec has screenshot assertions
- Part 12: Mobile spec uses mobile viewport
- Part 13: Settings spec covers all 12 tabs
- Part 14: Screenshots directory exists with README
- Part 15: Version bump check (3.2.0)
"""

import importlib.util
import json
import os
import re
import stat
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

_PROJECT = Path(__file__).resolve().parent.parent
_FRONTEND = _PROJECT / "frontend"
_E2E = _PROJECT / "tests" / "e2e"
_MOCKS = _E2E / "mocks"
_SCRIPTS = _PROJECT / "scripts"


def _load_version():
    """Load __version__ via importlib to avoid import chain."""
    spec = importlib.util.spec_from_file_location(
        "opti_oignon.__version__",
        _PROJECT / "opti_oignon" / "__version__.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.__version__


# ---------------------------------------------------------------------------
# Part 1: Playwright config
# ---------------------------------------------------------------------------

class TestPart01PlaywrightConfig:
    """Playwright config file exists and is valid."""

    def test_config_exists(self):
        path = _FRONTEND / "playwright.config.ts"
        assert path.exists(), f"Missing {path}"

    def test_config_has_define_config(self):
        content = (_FRONTEND / "playwright.config.ts").read_text()
        assert "defineConfig" in content

    def test_config_has_test_dir(self):
        content = (_FRONTEND / "playwright.config.ts").read_text()
        assert "testDir" in content
        assert "tests/e2e" in content

    def test_config_has_web_server(self):
        content = (_FRONTEND / "playwright.config.ts").read_text()
        assert "webServer" in content
        assert "npm run dev" in content

    def test_config_has_base_url(self):
        content = (_FRONTEND / "playwright.config.ts").read_text()
        assert "baseURL" in content
        assert "5173" in content

    def test_config_has_screenshot_on_failure(self):
        content = (_FRONTEND / "playwright.config.ts").read_text()
        assert "screenshot" in content
        assert "only-on-failure" in content

    def test_config_has_html_reporter(self):
        content = (_FRONTEND / "playwright.config.ts").read_text()
        assert "html" in content
        assert "list" in content

    def test_config_has_chromium_project(self):
        content = (_FRONTEND / "playwright.config.ts").read_text()
        assert "chromium" in content

    def test_config_has_mobile_project(self):
        content = (_FRONTEND / "playwright.config.ts").read_text()
        assert "mobile-chrome" in content


# ---------------------------------------------------------------------------
# Part 2: E2E directory structure
# ---------------------------------------------------------------------------

class TestPart02DirectoryStructure:
    """E2E directory structure is correct."""

    def test_e2e_dir_exists(self):
        assert _E2E.is_dir()

    def test_mocks_dir_exists(self):
        assert _MOCKS.is_dir()

    def test_screenshots_dir_exists(self):
        assert (_E2E / "screenshots").is_dir()

    def test_gitignore_exists(self):
        assert (_E2E / ".gitignore").exists()

    def test_readme_exists(self):
        assert (_E2E / "README.md").exists()


# ---------------------------------------------------------------------------
# Part 3: run_e2e.sh
# ---------------------------------------------------------------------------

class TestPart03RunE2EScript:
    """run_e2e.sh script exists and is executable."""

    def test_script_exists(self):
        assert (_SCRIPTS / "run_e2e.sh").exists()

    def test_script_is_executable(self):
        path = _SCRIPTS / "run_e2e.sh"
        mode = path.stat().st_mode
        assert mode & stat.S_IXUSR, "run_e2e.sh should be user-executable"

    def test_script_has_shebang(self):
        content = (_SCRIPTS / "run_e2e.sh").read_text()
        assert content.startswith("#!/usr/bin/env bash")

    def test_script_references_playwright(self):
        content = (_SCRIPTS / "run_e2e.sh").read_text()
        assert "playwright" in content

    def test_script_has_set_euo_pipefail(self):
        content = (_SCRIPTS / "run_e2e.sh").read_text()
        assert "set -euo pipefail" in content


# ---------------------------------------------------------------------------
# Part 4: Mock data file
# ---------------------------------------------------------------------------

class TestPart04MockData:
    """Mock data file is present and contains expected exports."""

    def test_data_file_exists(self):
        assert (_MOCKS / "data.ts").exists()

    @pytest.mark.parametrize("export_name", [
        "MOCK_USER",
        "MOCK_TOKENS",
        "MOCK_AUTH_STATUS",
        "MOCK_AUTH_STATUS_SINGLE",
        "MOCK_USER_SETTINGS",
        "MOCK_CONVERSATION",
        "MOCK_CONVERSATIONS_LIST",
        "MOCK_MESSAGES",
        "MOCK_OLLAMA_MODELS",
        "MOCK_RAG_COLLECTIONS",
        "MOCK_RAG_INGEST",
        "MOCK_RAG_QUERY",
        "MOCK_PLUGINS_LIST",
        "MOCK_SECURITY_MODE",
        "MOCK_KILL_SWITCH",
        "MOCK_HEALTH",
    ])
    def test_data_has_export(self, export_name):
        content = (_MOCKS / "data.ts").read_text()
        assert f"export const {export_name}" in content


# ---------------------------------------------------------------------------
# Part 5: Mock routes file
# ---------------------------------------------------------------------------

class TestPart05MockRoutes:
    """Mock routes file is present and contains expected functions."""

    def test_routes_file_exists(self):
        assert (_MOCKS / "routes.ts").exists()

    @pytest.mark.parametrize("func_name", [
        "setupAuthMocks",
        "setupConversationMocks",
        "setupChatMocks",
        "setupModelMocks",
        "setupRAGMocks",
        "setupPluginMocks",
        "setupSecurityMocks",
        "setupFallbackMock",
        "setupAllMocks",
    ])
    def test_routes_has_function(self, func_name):
        content = (_MOCKS / "routes.ts").read_text()
        assert f"export async function {func_name}" in content

    def test_routes_imports_data(self):
        content = (_MOCKS / "routes.ts").read_text()
        assert "from './data'" in content


# ---------------------------------------------------------------------------
# Part 6: Mock index barrel
# ---------------------------------------------------------------------------

class TestPart06MockIndex:
    """Mock index barrel export is present."""

    def test_index_exists(self):
        assert (_MOCKS / "index.ts").exists()

    def test_index_exports_data(self):
        content = (_MOCKS / "index.ts").read_text()
        assert "'./data'" in content

    def test_index_exports_routes(self):
        content = (_MOCKS / "index.ts").read_text()
        assert "'./routes'" in content


# ---------------------------------------------------------------------------
# Part 7: Fixtures file
# ---------------------------------------------------------------------------

class TestPart07Fixtures:
    """Fixtures file is present with test extensions."""

    def test_fixtures_exists(self):
        assert (_E2E / "fixtures.ts").exists()

    def test_fixtures_has_authed_page(self):
        content = (_E2E / "fixtures.ts").read_text()
        assert "authedPage" in content

    def test_fixtures_has_single_user_page(self):
        content = (_E2E / "fixtures.ts").read_text()
        assert "singleUserPage" in content

    def test_fixtures_imports_setup_all_mocks(self):
        content = (_E2E / "fixtures.ts").read_text()
        assert "setupAllMocks" in content


# ---------------------------------------------------------------------------
# Part 8: E2E spec files present
# ---------------------------------------------------------------------------

EXPECTED_SPECS = [
    "auth.spec.ts",
    "chat.spec.ts",
    "settings.spec.ts",
    "rag.spec.ts",
    "plugins.spec.ts",
    "security.spec.ts",
    "mobile.spec.ts",
    "visual.spec.ts",
]


class TestPart08SpecFilesPresent:
    """All E2E spec files are present."""

    @pytest.mark.parametrize("spec", EXPECTED_SPECS)
    def test_spec_exists(self, spec):
        path = _E2E / spec
        assert path.exists(), f"Missing spec: {path}"

    def test_total_spec_count(self):
        specs = list(_E2E.glob("*.spec.ts"))
        assert len(specs) >= 8


# ---------------------------------------------------------------------------
# Part 9: E2E spec files are valid
# ---------------------------------------------------------------------------

class TestPart09SpecFilesValid:
    """E2E spec files have valid structure (imports, describe, test)."""

    @pytest.mark.parametrize("spec", EXPECTED_SPECS)
    def test_spec_has_imports(self, spec):
        content = (_E2E / spec).read_text()
        assert "import" in content
        assert "from" in content

    @pytest.mark.parametrize("spec", EXPECTED_SPECS)
    def test_spec_has_test_describe(self, spec):
        content = (_E2E / spec).read_text()
        assert "test.describe" in content

    @pytest.mark.parametrize("spec", EXPECTED_SPECS)
    def test_spec_uses_mocks(self, spec):
        content = (_E2E / spec).read_text()
        assert "setupAllMocks" in content


# ---------------------------------------------------------------------------
# Part 10: package.json
# ---------------------------------------------------------------------------

class TestPart10PackageJson:
    """package.json has Playwright dependency and scripts."""

    def test_playwright_dependency(self):
        pkg = json.loads((_FRONTEND / "package.json").read_text())
        dev_deps = pkg.get("devDependencies", {})
        assert "@playwright/test" in dev_deps

    def test_e2e_script(self):
        pkg = json.loads((_FRONTEND / "package.json").read_text())
        scripts = pkg.get("scripts", {})
        assert "test:e2e" in scripts
        assert "playwright" in scripts["test:e2e"]

    def test_e2e_ui_script(self):
        pkg = json.loads((_FRONTEND / "package.json").read_text())
        scripts = pkg.get("scripts", {})
        assert "test:e2e:ui" in scripts

    def test_e2e_headed_script(self):
        pkg = json.loads((_FRONTEND / "package.json").read_text())
        scripts = pkg.get("scripts", {})
        assert "test:e2e:headed" in scripts


# ---------------------------------------------------------------------------
# Part 11: Visual regression spec
# ---------------------------------------------------------------------------

class TestPart11VisualSpec:
    """Visual regression spec has screenshot assertions."""

    def test_has_to_have_screenshot(self):
        content = (_E2E / "visual.spec.ts").read_text()
        assert "toHaveScreenshot" in content

    def test_desktop_screenshots(self):
        content = (_E2E / "visual.spec.ts").read_text()
        for name in [
            "desktop-chat-empty.png",
            "desktop-chat-messages.png",
            "desktop-login.png",
            "desktop-settings-quick.png",
            "desktop-settings-security.png",
        ]:
            assert name in content, f"Missing screenshot: {name}"

    def test_mobile_screenshots(self):
        content = (_E2E / "visual.spec.ts").read_text()
        for name in [
            "mobile-chat-empty.png",
            "mobile-login.png",
            "mobile-settings.png",
        ]:
            assert name in content, f"Missing screenshot: {name}"

    def test_screenshot_count(self):
        content = (_E2E / "visual.spec.ts").read_text()
        count = content.count("toHaveScreenshot")
        assert count >= 15


# ---------------------------------------------------------------------------
# Part 12: Mobile spec
# ---------------------------------------------------------------------------

class TestPart12MobileSpec:
    """Mobile spec uses mobile viewport."""

    def test_mobile_viewport(self):
        content = (_E2E / "mobile.spec.ts").read_text()
        assert "393" in content  # Pixel 5 width
        assert "851" in content  # Pixel 5 height

    def test_mobile_is_mobile_flag(self):
        content = (_E2E / "mobile.spec.ts").read_text()
        assert "isMobile: true" in content

    def test_mobile_has_touch(self):
        content = (_E2E / "mobile.spec.ts").read_text()
        assert "hasTouch: true" in content

    def test_sidebar_toggle_test(self):
        content = (_E2E / "mobile.spec.ts").read_text()
        assert "Toggle sidebar" in content

    def test_touch_target_test(self):
        content = (_E2E / "mobile.spec.ts").read_text()
        assert "44" in content  # 44px touch target


# ---------------------------------------------------------------------------
# Part 13: Settings spec
# ---------------------------------------------------------------------------

SETTINGS_TABS = [
    "Quick", "Presets", "Models", "Prompt", "Analytics",
    "Observe", "Fine-Tune", "Knowledge", "Plugins",
    "Backup", "Security", "Advanced",
]


class TestPart13SettingsSpec:
    """Settings spec covers all 12 tabs."""

    @pytest.mark.parametrize("tab_label", SETTINGS_TABS)
    def test_tab_in_spec(self, tab_label):
        content = (_E2E / "settings.spec.ts").read_text()
        assert tab_label in content, f"Missing tab: {tab_label}"

    def test_sequential_navigation_test(self):
        content = (_E2E / "settings.spec.ts").read_text()
        assert "sequentially" in content.lower() or "pageerror" in content


# ---------------------------------------------------------------------------
# Part 14: Screenshots directory
# ---------------------------------------------------------------------------

class TestPart14ScreenshotsDir:
    """Screenshots directory exists with README."""

    def test_dir_exists(self):
        assert (_E2E / "screenshots").is_dir()

    def test_readme_exists(self):
        assert (_E2E / "screenshots" / "README.md").exists()

    def test_readme_has_content(self):
        content = (_E2E / "screenshots" / "README.md").read_text()
        assert len(content) > 50


# ---------------------------------------------------------------------------
# Part 15: Version bump
# ---------------------------------------------------------------------------

class TestPart15VersionBump:
    """Version is bumped to 3.2.0."""

    def test_version_is_rc5(self):
        assert _load_version() == "3.2.0"

    def test_version_file_content(self):
        content = (_PROJECT / "opti_oignon" / "__version__.py").read_text()
        assert '"3.2.0"' in content
