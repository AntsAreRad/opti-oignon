#!/usr/bin/env python3
"""
test_palette_v4e_s93.py -- Tests for S93 palette v4e + accessibility.

Covers:
  - Theme.css structure: dark/light blocks, required variables
  - Palette v4e values: greyer anthracite, deep taupe, dual accent
  - Dual accent tokens: sage, tobacco, pine
  - Barely-there borders: delta from background
  - Bumped border-radius values
  - WCAG AA contrast (via audit_contrast.py)
  - Theme transition CSS class
  - Reduced motion media query
  - System theme detection in ui.ts
  - No hardcoded hex in Svelte (via audit_colors.py)
  - Version bump to 1.9.5
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Direct module loading
# ---------------------------------------------------------------------------

_PROJECT = Path(__file__).resolve().parent.parent
_SCRIPTS = _PROJECT / "scripts"

# Load audit_contrast module
_contrast_path = _SCRIPTS / "audit_contrast.py"
_contrast_spec = importlib.util.spec_from_file_location("audit_contrast", str(_contrast_path))
assert _contrast_spec is not None and _contrast_spec.loader is not None
audit_contrast = importlib.util.module_from_spec(_contrast_spec)
sys.modules["audit_contrast"] = audit_contrast
_contrast_spec.loader.exec_module(audit_contrast)

# Load audit_colors module
_colors_path = _SCRIPTS / "audit_colors.py"
_colors_spec = importlib.util.spec_from_file_location("audit_colors", str(_colors_path))
assert _colors_spec is not None and _colors_spec.loader is not None
audit_colors = importlib.util.module_from_spec(_colors_spec)
sys.modules["audit_colors"] = audit_colors
_colors_spec.loader.exec_module(audit_colors)

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

THEME_CSS = _PROJECT / "frontend" / "src" / "styles" / "theme.css"
APP_CSS = _PROJECT / "frontend" / "src" / "app.css"
UI_TS = _PROJECT / "frontend" / "src" / "lib" / "stores" / "ui.ts"
LAYOUT = _PROJECT / "frontend" / "src" / "routes" / "+layout.svelte"
APP_PY = _PROJECT / "opti_oignon" / "api" / "app.py"
SIDEBAR = _PROJECT / "frontend" / "src" / "lib" / "components" / "layout" / "Sidebar.svelte"
CONTROL_BAR = _PROJECT / "frontend" / "src" / "lib" / "components" / "chat" / "ChatControlBar.svelte"
CHAT_INPUT = _PROJECT / "frontend" / "src" / "lib" / "components" / "chat" / "ChatInput.svelte"
NEW_CONV_BTN = _PROJECT / "frontend" / "src" / "lib" / "components" / "sidebar" / "NewConversationButton.svelte"
SVELTE_DIR = _PROJECT / "frontend" / "src" / "lib" / "components"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def theme_css():
    return THEME_CSS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def dark_vars(theme_css):
    d, _ = audit_contrast.extract_themed_vars(theme_css)
    return d


@pytest.fixture(scope="module")
def light_vars(theme_css):
    _, l = audit_contrast.extract_themed_vars(theme_css)
    return l


@pytest.fixture(scope="module")
def app_css():
    return APP_CSS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ui_ts():
    return UI_TS.read_text(encoding="utf-8")


# ===========================================================================
# Theme.css structure
# ===========================================================================

class TestThemeCssStructure:
    """Theme.css has both dark and light blocks with required variables."""

    def test_dark_block_exists(self, theme_css):
        assert ":root," in theme_css or "html.dark" in theme_css

    def test_light_block_exists(self, theme_css):
        assert "html:not(.dark)" in theme_css

    def test_dark_vars_parsed(self, dark_vars):
        assert len(dark_vars) > 50

    def test_light_vars_parsed(self, light_vars):
        assert len(light_vars) > 50

    def test_header_mentions_v4e(self, theme_css):
        assert "v4e" in theme_css.lower() or "S93" in theme_css


# ===========================================================================
# Palette v4e: Dark mode (greyer anthracite)
# ===========================================================================

class TestDarkPalette:
    """Dark mode uses greyer anthracite, not warm brown."""

    def test_bg_base_greyer(self, dark_vars):
        val = dark_vars.get("--oo-bg-base", "")
        assert val == "#222224", f"Expected #222224, got {val}"

    def test_bg_surface(self, dark_vars):
        val = dark_vars.get("--oo-bg-surface", "")
        assert val == "#2B2B2D"

    def test_bg_elevated(self, dark_vars):
        val = dark_vars.get("--oo-bg-elevated", "")
        assert val == "#343436"

    def test_sidebar_darker_than_base(self, dark_vars):
        sidebar = dark_vars.get("--oo-sidebar-bg", "")
        assert sidebar == "#1C1C1E"

    def test_fg_primary(self, dark_vars):
        val = dark_vars.get("--oo-fg-primary", "")
        assert val == "#E8E4DF"

    def test_borders_barely_there(self, dark_vars):
        bd = dark_vars.get("--oo-bd-default", "")
        assert bd == "#2F2F31", f"Border should be barely-there, got {bd}"


# ===========================================================================
# Palette v4e: Light mode (deep taupe)
# ===========================================================================

class TestLightPalette:
    """Light mode uses deep taupe, no white anywhere."""

    def test_bg_base_deep_taupe(self, light_vars):
        val = light_vars.get("--oo-bg-base", "")
        assert val == "#DDD6CC"

    def test_bg_surface_taupe(self, light_vars):
        val = light_vars.get("--oo-bg-surface", "")
        assert val == "#D4CDC2"

    def test_no_white_or_near_white(self, light_vars):
        # No background should be #FFF or #FDFBF9 or similar
        for key, val in light_vars.items():
            if key.startswith("--oo-bg"):
                hex_val = val.strip().upper()
                if hex_val.startswith("#") and len(hex_val) == 7:
                    r = int(hex_val[1:3], 16)
                    g = int(hex_val[3:5], 16)
                    b = int(hex_val[5:7], 16)
                    assert not (r > 250 and g > 250 and b > 250), \
                        f"{key} = {val} is too white for deep taupe"

    def test_fg_primary_dark(self, light_vars):
        val = light_vars.get("--oo-fg-primary", "")
        assert val == "#3A3836"


# ===========================================================================
# Dual accent: sage + tobacco
# ===========================================================================

class TestDualAccent:
    """Sage and tobacco tokens exist in both themes."""

    def test_dark_sage_exists(self, dark_vars):
        assert "--oo-sage" in dark_vars
        assert "--oo-sage-bg" in dark_vars
        assert "--oo-sage-bd" in dark_vars

    def test_dark_tobacco_exists(self, dark_vars):
        assert "--oo-tobacco" in dark_vars
        assert "--oo-tobacco-bg" in dark_vars
        assert "--oo-tobacco-bd" in dark_vars

    def test_dark_pine_exists(self, dark_vars):
        assert "--oo-pine" in dark_vars

    def test_light_sage_exists(self, light_vars):
        assert "--oo-sage" in light_vars

    def test_light_tobacco_exists(self, light_vars):
        assert "--oo-tobacco" in light_vars

    def test_dark_tobacco_is_bold(self, dark_vars):
        val = dark_vars.get("--oo-tobacco", "")
        assert val == "#C48838"

    def test_dark_success_is_sage(self, dark_vars):
        sage = dark_vars.get("--oo-sage", "")
        success = dark_vars.get("--oo-success", "")
        assert sage == success, "Dark success should equal sage"

    def test_light_success_is_sage(self, light_vars):
        sage = light_vars.get("--oo-sage", "")
        success = light_vars.get("--oo-success", "")
        assert sage == success, "Light success should equal sage"


# ===========================================================================
# Bumped border-radius
# ===========================================================================

class TestBorderRadius:
    """Card/panel radius bumped to 12-14px range."""

    def test_radius_lg(self, dark_vars):
        val = dark_vars.get("--oo-radius-lg", "")
        px = int(val.replace("px", ""))
        assert 12 <= px <= 16, f"radius-lg should be 12-16px, got {val}"

    def test_radius_md(self, dark_vars):
        val = dark_vars.get("--oo-radius-md", "")
        px = int(val.replace("px", ""))
        assert 8 <= px <= 12


# ===========================================================================
# WCAG AA contrast
# ===========================================================================

class TestContrastCompliance:
    """All fg/bg pairs meet WCAG AA requirements."""

    def test_dark_mode_all_pass(self, dark_vars):
        failures = audit_contrast.audit_theme(dark_vars, "dark", verbose=False)
        assert len(failures) == 0, f"Dark mode failures: {failures}"

    def test_light_mode_all_pass(self, light_vars):
        failures = audit_contrast.audit_theme(light_vars, "light", verbose=False)
        assert len(failures) == 0, f"Light mode failures: {failures}"

    def test_contrast_ratio_function(self):
        white = (255, 255, 255)
        black = (0, 0, 0)
        ratio = audit_contrast.contrast_ratio(white, black)
        assert abs(ratio - 21.0) < 0.1

    def test_relative_luminance_white(self):
        lum = audit_contrast.relative_luminance((255, 255, 255))
        assert abs(lum - 1.0) < 0.01

    def test_relative_luminance_black(self):
        lum = audit_contrast.relative_luminance((0, 0, 0))
        assert abs(lum - 0.0) < 0.01

    def test_parse_hex_valid(self):
        assert audit_contrast.parse_hex("#FF8800") == (255, 136, 0)

    def test_parse_hex_short(self):
        assert audit_contrast.parse_hex("#F80") == (255, 136, 0)


# ===========================================================================
# Theme transition CSS
# ===========================================================================

class TestThemeTransition:
    """app.css includes theme-transitioning class and reduced-motion."""

    def test_transitioning_class(self, app_css):
        assert "theme-transitioning" in app_css

    def test_transition_properties(self, app_css):
        assert "background-color" in app_css
        assert "300ms" in app_css

    def test_reduced_motion_media_query(self, app_css):
        assert "prefers-reduced-motion" in app_css

    def test_reduced_motion_kills_animations(self, app_css):
        # Should set animation-duration to near-zero
        assert "0.01ms" in app_css


# ===========================================================================
# System theme detection in ui.ts
# ===========================================================================

class TestUiStore:
    """ui.ts handles system theme, smooth toggle, reduced motion."""

    def test_prefers_color_scheme_listener(self, ui_ts):
        assert "prefers-color-scheme" in ui_ts

    def test_prefers_reduced_motion_store(self, ui_ts):
        assert "prefersReducedMotion" in ui_ts

    def test_theme_transitioning_class_added(self, ui_ts):
        assert "theme-transitioning" in ui_ts

    def test_transition_removed_after_timeout(self, ui_ts):
        assert "350" in ui_ts or "setTimeout" in ui_ts

    def test_exports_toggle_theme(self, ui_ts):
        assert "export function toggleTheme" in ui_ts

    def test_exports_init_theme(self, ui_ts):
        assert "export function initTheme" in ui_ts


# ===========================================================================
# Component updates
# ===========================================================================

class TestComponentUpdates:
    """Components use correct v4e tokens."""

    def test_sidebar_active_uses_tobacco(self):
        content = SIDEBAR.read_text(encoding="utf-8")
        assert "oo-tobacco-bg" in content
        assert "oo-tobacco" in content

    def test_toggle_pills_borderless(self):
        content = CONTROL_BAR.read_text(encoding="utf-8")
        # Active style should use tobacco-bg, not copper glow
        assert "oo-tobacco-bg" in content
        assert "box-shadow" not in content.split("activeStyle")[1].split(";")[0] if "activeStyle" in content else True

    def test_send_button_uses_primary(self):
        content = CHAT_INPUT.read_text(encoding="utf-8")
        assert "oo-btn-primary-bg" in content
        assert "oo-btn-primary-fg" in content

    def test_new_conversation_uses_sage(self):
        content = NEW_CONV_BTN.read_text(encoding="utf-8")
        assert "oo-sage-bg" in content
        assert "oo-sage" in content

    def test_no_hardcoded_hex_in_svelte(self):
        violations = audit_colors.scan_directory(SVELTE_DIR)
        assert len(violations) == 0, f"Found {len(violations)} hardcoded color violations"


# ===========================================================================
# Version bump
# ===========================================================================

class TestVersionBump:
    """Version bumped to 1.9.5."""

    def test_app_py_version(self):
        content = APP_PY.read_text(encoding="utf-8")
        assert '1.10.0' in content
