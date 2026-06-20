"""
tests/test_s153_shortcuts_a11y.py -- S153 keyboard shortcuts + accessibility tests.

Verifies:
- ShortcutBinding dataclass: combo_string, display_string, serialization
- Default shortcuts map completeness
- ShortcutRegistry: register, lookup, conflict detection, browser conflicts
- Custom binding validation and application
- Reset single/all to defaults
- Export custom diff
- Parse combo string utility
- API schemas existence
- API endpoint registration in routes_settings
- Frontend file existence (ShortcutSettings, shortcuts API client)
- Accessibility: skip-to-content, aria-live, landmarks in AppShell
- Focus indicator CSS rules
- Version bump check
"""

import importlib.util
import os
import re
import sys
import types

# -- Isolation stubs (standard pattern) --
for mod_name in [
    "opti_oignon",
    "opti_oignon.db_utils",
    "opti_oignon.config",
    "opti_oignon.auth",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KS_PATH = os.path.join(PROJECT_ROOT, "opti_oignon", "keyboard_shortcuts.py")
SCHEMAS_PATH = os.path.join(PROJECT_ROOT, "opti_oignon", "api", "schemas.py")
ROUTES_PATH = os.path.join(PROJECT_ROOT, "opti_oignon", "api", "routes_settings.py")
VERSION_PATH = os.path.join(PROJECT_ROOT, "opti_oignon", "__version__.py")
APPSHELL_PATH = os.path.join(
    PROJECT_ROOT, "frontend", "src", "lib", "components", "layout", "AppShell.svelte"
)
LAYOUT_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "routes", "+layout.svelte")
APP_CSS_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "app.css")
SHORTCUTS_SVELTE_PATH = os.path.join(
    PROJECT_ROOT, "frontend", "src", "lib", "components", "ui", "KeyboardShortcuts.svelte"
)
SETTINGS_SVELTE_PATH = os.path.join(
    PROJECT_ROOT, "frontend", "src", "lib", "components", "settings", "ShortcutSettings.svelte"
)
API_CLIENT_PATH = os.path.join(
    PROJECT_ROOT, "frontend", "src", "lib", "api", "shortcuts.ts"
)


def _load_module(name: str, path: str):
    """Load a Python module by file path (isolation pattern)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ks():
    """Load keyboard_shortcuts module."""
    return _load_module("keyboard_shortcuts", KS_PATH)


# ---- ShortcutBinding dataclass ----

class TestShortcutBinding:
    """Tests for the ShortcutBinding dataclass."""

    def test_combo_string_simple(self, ks):
        b = ks.ShortcutBinding(action="test", key="n", ctrl=True)
        assert b.combo_string() == "ctrl+n"

    def test_combo_string_multi_modifiers(self, ks):
        b = ks.ShortcutBinding(action="test", key="t", ctrl=True, shift=True)
        assert b.combo_string() == "ctrl+shift+t"

    def test_combo_string_no_modifiers(self, ks):
        b = ks.ShortcutBinding(action="test", key="?")
        assert b.combo_string() == "?"

    def test_combo_string_alt(self, ks):
        b = ks.ShortcutBinding(action="test", key="x", alt=True)
        assert b.combo_string() == "alt+x"

    def test_combo_string_meta(self, ks):
        b = ks.ShortcutBinding(action="test", key="k", meta=True)
        assert b.combo_string() == "meta+k"

    def test_display_string_ctrl_n(self, ks):
        b = ks.ShortcutBinding(action="test", key="n", ctrl=True)
        assert b.display_string() == "Ctrl + N"

    def test_display_string_ctrl_shift_t(self, ks):
        b = ks.ShortcutBinding(
            action="test", key="t", ctrl=True, shift=True
        )
        assert b.display_string() == "Ctrl + Shift + T"

    def test_display_string_escape(self, ks):
        b = ks.ShortcutBinding(action="test", key="escape")
        assert b.display_string() == "Esc"

    def test_display_string_enter(self, ks):
        b = ks.ShortcutBinding(action="test", key="enter", ctrl=True)
        assert b.display_string() == "Ctrl + Enter"

    def test_to_dict(self, ks):
        b = ks.ShortcutBinding(
            action="test", key="n", ctrl=True, description="Test"
        )
        d = b.to_dict()
        assert d["action"] == "test"
        assert d["key"] == "n"
        assert d["ctrl"] is True
        assert d["shift"] is False
        assert d["description"] == "Test"

    def test_from_dict(self, ks):
        data = {"action": "foo", "key": "x", "ctrl": True, "shift": False}
        b = ks.ShortcutBinding.from_dict(data)
        assert b.action == "foo"
        assert b.key == "x"
        assert b.ctrl is True

    def test_roundtrip_serialization(self, ks):
        original = ks.ShortcutBinding(
            action="roundtrip", key="r", ctrl=True, shift=True, alt=True,
            description="Roundtrip test", category="test"
        )
        restored = ks.ShortcutBinding.from_dict(original.to_dict())
        assert restored.combo_string() == original.combo_string()
        assert restored.description == original.description
        assert restored.category == original.category


# ---- Default shortcuts ----

class TestDefaultShortcuts:
    """Tests for default shortcut definitions."""

    EXPECTED_ACTIONS = [
        "new_chat", "send_message", "toggle_sidebar",
        "search_conversations", "open_settings", "toggle_theme",
        "export_conversation", "show_shortcuts", "close_dialog",
    ]

    def test_all_expected_actions_present(self, ks):
        for action in self.EXPECTED_ACTIONS:
            assert action in ks.DEFAULT_SHORTCUTS_MAP, f"Missing: {action}"

    def test_default_count(self, ks):
        assert len(ks.DEFAULT_SHORTCUTS) == 9

    def test_default_map_matches_list(self, ks):
        assert len(ks.DEFAULT_SHORTCUTS_MAP) == len(ks.DEFAULT_SHORTCUTS)

    def test_new_chat_is_ctrl_n(self, ks):
        s = ks.DEFAULT_SHORTCUTS_MAP["new_chat"]
        assert s.combo_string() == "ctrl+n"

    def test_toggle_theme_is_ctrl_shift_t(self, ks):
        s = ks.DEFAULT_SHORTCUTS_MAP["toggle_theme"]
        assert s.combo_string() == "ctrl+shift+t"

    def test_toggle_sidebar_is_ctrl_b(self, ks):
        s = ks.DEFAULT_SHORTCUTS_MAP["toggle_sidebar"]
        assert s.combo_string() == "ctrl+b"

    def test_get_default_shortcuts_returns_list(self, ks):
        result = ks.get_default_shortcuts()
        assert isinstance(result, list)
        assert len(result) == 9
        assert all(isinstance(d, dict) for d in result)

    def test_get_default_shortcuts_map_returns_dict(self, ks):
        result = ks.get_default_shortcuts_map()
        assert isinstance(result, dict)
        assert "new_chat" in result


# ---- ShortcutRegistry ----

class TestShortcutRegistry:
    """Tests for the ShortcutRegistry class."""

    def test_initial_load_defaults(self, ks):
        reg = ks.ShortcutRegistry()
        assert len(reg.get_all()) == 9

    def test_get_existing_action(self, ks):
        reg = ks.ShortcutRegistry()
        b = reg.get("new_chat")
        assert b is not None
        assert b.combo_string() == "ctrl+n"

    def test_get_nonexistent_action(self, ks):
        reg = ks.ShortcutRegistry()
        assert reg.get("nonexistent") is None

    def test_lookup_by_combo(self, ks):
        reg = ks.ShortcutRegistry()
        b = reg.lookup_by_combo("ctrl+n")
        assert b is not None
        assert b.action == "new_chat"

    def test_lookup_by_combo_not_found(self, ks):
        reg = ks.ShortcutRegistry()
        assert reg.lookup_by_combo("ctrl+alt+z") is None

    def test_register_new_action(self, ks):
        reg = ks.ShortcutRegistry()
        b = reg.register("custom_action", "x", ctrl=True, description="Custom")
        assert b.action == "custom_action"
        assert reg.get("custom_action") is not None

    def test_unregister_action(self, ks):
        reg = ks.ShortcutRegistry()
        assert reg.unregister("new_chat") is True
        assert reg.get("new_chat") is None

    def test_unregister_nonexistent(self, ks):
        reg = ks.ShortcutRegistry()
        assert reg.unregister("nonexistent") is False

    def test_get_all_serialized(self, ks):
        reg = ks.ShortcutRegistry()
        result = reg.get_all_serialized()
        assert isinstance(result, dict)
        assert "new_chat" in result
        assert isinstance(result["new_chat"], dict)
        assert "key" in result["new_chat"]


# ---- Conflict detection ----

class TestConflictDetection:
    """Tests for shortcut conflict detection."""

    def test_no_conflicts_in_defaults(self, ks):
        reg = ks.ShortcutRegistry()
        conflicts = reg.detect_conflicts()
        assert len(conflicts) == 0

    def test_detect_internal_conflict(self, ks):
        reg = ks.ShortcutRegistry()
        # Register a duplicate combo
        reg.register("duplicate", "n", ctrl=True, description="Dup")
        conflicts = reg.detect_conflicts()
        assert len(conflicts) >= 1
        combo_actions = conflicts[0]["actions"]
        assert "new_chat" in combo_actions
        assert "duplicate" in combo_actions

    def test_detect_conflicts_exclude_action(self, ks):
        reg = ks.ShortcutRegistry()
        reg.register("duplicate", "n", ctrl=True)
        # Exclude "duplicate" from detection
        conflicts = reg.detect_conflicts(exclude_action="duplicate")
        assert len(conflicts) == 0

    def test_browser_conflicts_exist(self, ks):
        reg = ks.ShortcutRegistry()
        # ctrl+n conflicts with browser new window
        warnings = reg.check_browser_conflicts()
        combos = [w["combo"] for w in warnings]
        assert "ctrl+n" in combos

    def test_browser_conflict_keys_nonempty(self, ks):
        assert len(ks.BROWSER_CONFLICT_KEYS) > 10


# ---- Custom bindings ----

class TestCustomBindings:
    """Tests for custom binding application and validation."""

    def test_apply_custom_bindings(self, ks):
        reg = ks.ShortcutRegistry()
        warnings = reg.apply_custom_bindings({
            "new_chat": {"key": "m", "ctrl": True}
        })
        assert len(warnings) == 0
        b = reg.get("new_chat")
        assert b.combo_string() == "ctrl+m"

    def test_apply_unknown_action_warns(self, ks):
        reg = ks.ShortcutRegistry()
        warnings = reg.apply_custom_bindings({
            "nonexistent_action": {"key": "z"}
        })
        assert len(warnings) == 1
        assert "Unknown action" in warnings[0]

    def test_apply_invalid_binding_warns(self, ks):
        reg = ks.ShortcutRegistry()
        warnings = reg.apply_custom_bindings({
            "new_chat": {"key": ""}
        })
        assert len(warnings) == 1
        assert "Invalid binding" in warnings[0]

    def test_apply_preserves_description(self, ks):
        reg = ks.ShortcutRegistry()
        reg.apply_custom_bindings({
            "new_chat": {"key": "m", "ctrl": True}
        })
        b = reg.get("new_chat")
        assert b.description == "New conversation"

    def test_export_custom_diff_empty_on_defaults(self, ks):
        reg = ks.ShortcutRegistry()
        diff = reg.export_custom_diff()
        assert len(diff) == 0

    def test_export_custom_diff_after_change(self, ks):
        reg = ks.ShortcutRegistry()
        reg.apply_custom_bindings({
            "new_chat": {"key": "m", "ctrl": True}
        })
        diff = reg.export_custom_diff()
        assert "new_chat" in diff
        assert diff["new_chat"]["key"] == "m"

    def test_reset_single_action(self, ks):
        reg = ks.ShortcutRegistry()
        reg.apply_custom_bindings({
            "new_chat": {"key": "m", "ctrl": True}
        })
        assert reg.reset_action("new_chat") is True
        b = reg.get("new_chat")
        assert b.combo_string() == "ctrl+n"

    def test_reset_unknown_action(self, ks):
        reg = ks.ShortcutRegistry()
        assert reg.reset_action("nonexistent") is False

    def test_reset_all(self, ks):
        reg = ks.ShortcutRegistry()
        reg.apply_custom_bindings({
            "new_chat": {"key": "m", "ctrl": True},
            "toggle_theme": {"key": "d", "ctrl": True},
        })
        reg.reset_all()
        assert reg.get("new_chat").combo_string() == "ctrl+n"
        assert reg.get("toggle_theme").combo_string() == "ctrl+shift+t"


# ---- Validation utilities ----

class TestValidation:
    """Tests for validation functions."""

    def test_validate_key_single_char(self, ks):
        assert ks.validate_key("a") is True
        assert ks.validate_key("Z") is True

    def test_validate_key_special(self, ks):
        assert ks.validate_key("enter") is True
        assert ks.validate_key("escape") is True
        assert ks.validate_key("f1") is True

    def test_validate_key_punctuation(self, ks):
        assert ks.validate_key(",") is True
        assert ks.validate_key("?") is True

    def test_validate_key_empty(self, ks):
        assert ks.validate_key("") is False

    def test_validate_key_invalid(self, ks):
        assert ks.validate_key("notakey") is False

    def test_validate_binding_valid(self, ks):
        result = ks.validate_binding({"key": "n", "ctrl": True})
        assert result is not None
        assert result["key"] == "n"
        assert result["ctrl"] is True

    def test_validate_binding_missing_key(self, ks):
        assert ks.validate_binding({"ctrl": True}) is None

    def test_validate_binding_empty_key(self, ks):
        assert ks.validate_binding({"key": ""}) is None

    def test_validate_binding_not_dict(self, ks):
        assert ks.validate_binding("invalid") is None

    def test_validate_custom_bindings_valid(self, ks):
        ok, errors = ks.validate_custom_bindings({
            "new_chat": {"key": "m", "ctrl": True}
        })
        assert ok is True
        assert len(errors) == 0

    def test_validate_custom_bindings_unknown_action(self, ks):
        ok, errors = ks.validate_custom_bindings({
            "fake_action": {"key": "x"}
        })
        assert ok is False
        assert any("Unknown action" in e for e in errors)

    def test_validate_custom_bindings_not_dict(self, ks):
        ok, errors = ks.validate_custom_bindings("invalid")
        assert ok is False

    def test_parse_combo_string_valid(self, ks):
        result = ks.parse_combo_string("ctrl+shift+t")
        assert result is not None
        assert result["key"] == "t"
        assert result["ctrl"] is True
        assert result["shift"] is True

    def test_parse_combo_string_single_key(self, ks):
        result = ks.parse_combo_string("?")
        assert result is not None
        assert result["key"] == "?"
        assert result["ctrl"] is False

    def test_parse_combo_string_invalid(self, ks):
        assert ks.parse_combo_string("") is None
        assert ks.parse_combo_string("invalid+mod+x") is None

    def test_check_combo_browser_conflict(self, ks):
        assert ks.check_combo_browser_conflict("ctrl+t") is not None
        assert ks.check_combo_browser_conflict("ctrl+alt+z") is None


# ---- API schemas and routes ----

class TestAPIIntegration:
    """Tests for API schema and route existence."""

    def test_schemas_file_exists(self):
        assert os.path.isfile(SCHEMAS_PATH)

    def test_keyboard_shortcuts_response_schema(self):
        content = open(SCHEMAS_PATH).read()
        assert "class KeyboardShortcutsResponse" in content

    def test_keyboard_shortcuts_update_request_schema(self):
        content = open(SCHEMAS_PATH).read()
        assert "class KeyboardShortcutsUpdateRequest" in content

    def test_keyboard_shortcuts_update_response_schema(self):
        content = open(SCHEMAS_PATH).read()
        assert "class KeyboardShortcutsUpdateResponse" in content

    def test_shortcut_binding_schema(self):
        content = open(SCHEMAS_PATH).read()
        assert "class ShortcutBindingSchema" in content

    def test_routes_get_keyboard_shortcuts(self):
        content = open(ROUTES_PATH).read()
        assert '"/keyboard_shortcuts"' in content
        assert "def get_keyboard_shortcuts" in content

    def test_routes_put_keyboard_shortcuts(self):
        content = open(ROUTES_PATH).read()
        assert "def update_keyboard_shortcuts" in content

    def test_routes_before_catch_all(self):
        content = open(ROUTES_PATH).read()
        get_pos = content.find("def get_keyboard_shortcuts")
        catchall_pos = content.find('def get_setting(key: str)')
        assert get_pos < catchall_pos, "Shortcuts route must be before catch-all"


# ---- Frontend file existence ----

class TestFrontendFiles:
    """Tests for frontend file existence and content."""

    def test_shortcuts_api_client_exists(self):
        assert os.path.isfile(API_CLIENT_PATH)

    def test_shortcuts_api_has_get(self):
        content = open(API_CLIENT_PATH).read()
        assert "getKeyboardShortcuts" in content

    def test_shortcuts_api_has_update(self):
        content = open(API_CLIENT_PATH).read()
        assert "updateKeyboardShortcuts" in content

    def test_shortcuts_api_has_reset(self):
        content = open(API_CLIENT_PATH).read()
        assert "resetAllShortcuts" in content

    def test_shortcut_settings_svelte_exists(self):
        assert os.path.isfile(SETTINGS_SVELTE_PATH)

    def test_shortcut_settings_has_rebind(self):
        content = open(SETTINGS_SVELTE_PATH).read()
        assert "startRebind" in content

    def test_shortcut_settings_has_reset_all(self):
        content = open(SETTINGS_SVELTE_PATH).read()
        assert "resetAll" in content

    def test_keyboard_shortcuts_svelte_exists(self):
        assert os.path.isfile(SHORTCUTS_SVELTE_PATH)

    def test_keyboard_shortcuts_has_toggle_sidebar(self):
        content = open(SHORTCUTS_SVELTE_PATH).read()
        assert "toggle_sidebar" in content

    def test_keyboard_shortcuts_has_custom_event_listener(self):
        content = open(SHORTCUTS_SVELTE_PATH).read()
        assert "opti-shortcuts-updated" in content

    def test_keyboard_shortcuts_has_load_custom(self):
        content = open(SHORTCUTS_SVELTE_PATH).read()
        assert "loadCustomBindings" in content

    def test_layout_has_toggle_sidebar_prop(self):
        content = open(LAYOUT_PATH).read()
        assert "onToggleSidebar" in content
        assert "toggleSidebar" in content


# ---- Accessibility ----

class TestAccessibility:
    """Tests for WCAG accessibility improvements."""

    def test_skip_to_content_in_appshell(self):
        content = open(APPSHELL_PATH).read()
        assert 'skip-to-content' in content
        assert '#main-content' in content

    def test_main_has_id(self):
        content = open(APPSHELL_PATH).read()
        assert 'id="main-content"' in content

    def test_aria_live_region_exists(self):
        content = open(APPSHELL_PATH).read()
        assert 'aria-live="polite"' in content
        assert 'oo-route-announcer' in content

    def test_sidebar_uses_nav_landmark(self):
        content = open(APPSHELL_PATH).read()
        assert '<nav' in content
        assert 'aria-label="Sidebar navigation"' in content

    def test_panel_uses_aside_landmark(self):
        content = open(APPSHELL_PATH).read()
        assert '<aside' in content
        assert 'aria-label="Side panel"' in content

    def test_header_exists(self):
        content = open(APPSHELL_PATH).read()
        assert '<header' in content

    def test_route_announcer_in_layout(self):
        content = open(LAYOUT_PATH).read()
        assert 'oo-route-announcer' in content

    def test_skip_to_content_css_exists(self):
        content = open(APP_CSS_PATH).read()
        assert '.skip-to-content' in content

    def test_focus_visible_css_exists(self):
        content = open(APP_CSS_PATH).read()
        assert ':focus-visible' in content

    def test_focus_visible_uses_oo_variable(self):
        content = open(APP_CSS_PATH).read()
        # Ensure focus ring uses --oo- variable not hardcoded color
        idx = content.find(':focus-visible')
        block = content[idx:idx + 200]
        assert '--oo-acc' in block


# ---- Version ----

class TestVersion:
    """Version bump check."""

    def test_version_is_3_2_2(self):
        content = open(VERSION_PATH).read()
        assert '"3.2.2"' in content
