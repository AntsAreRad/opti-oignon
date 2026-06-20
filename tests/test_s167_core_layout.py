#!/usr/bin/env python3
"""
test_s167_core_layout.py -- S167 core layout refonte.

Covers (spec 12.2):
  - Application shell: Header + StatusFooter extracted, Sidebar split into
    App Nav + SectionContextList; AppShell landmarks and mobile swipe
    machinery preserved.
  - Theme switcher: header palette + density dropdown with WAI-ARIA menu
    semantics, keyboard nav, wired to the preferences store; the 5 theme
    files raised to :root[data-oo-theme] specificity.
  - Header status merge: BackendStatus folds in the Ollama network status
    (former NetworkIndicator).
  - Section context per route; chat sub-components on ds primitives.
  - Accessibility: global skip link to #main-content, live regions on
    streaming output and on toasts.
  - Toast cutover: single ds/Toast mount in the root layout.

It also re-asserts, against the new structure, the prior-session
assertions superseded by this refonte (and deselected in pyproject).

These checks read source files as text; no ollama or node_modules needed.
"""

import re
from pathlib import Path

import pytest

_PROJECT = Path(__file__).resolve().parent.parent
_FE = _PROJECT / "frontend" / "src"
_STYLES = _FE / "styles"
_DS = _FE / "lib" / "ds"
_COMPONENTS = _FE / "lib" / "components"
_LAYOUT = _COMPONENTS / "layout"
_SIDEBAR = _COMPONENTS / "sidebar"
_CHAT = _COMPONENTS / "chat"
_UI = _COMPONENTS / "ui"
_ROUTES = _FE / "routes"
_STORES = _FE / "lib" / "stores"
_APP_CSS = _FE / "app.css"

# French diacritics used to assert English-only code/UI.
_FRENCH = re.compile(r"[àâäéèêëîïôùûüçÀÂÄÉÈÊËÎÏÔÙÛÜÇ]")
# Raw 6-digit hex, excluding the permitted var(--oo-*, #fallback) form.
_HEX = re.compile(r"#[0-9a-fA-F]{6}\b")
_VAR_FALLBACK = re.compile(r"var\(\s*--oo-[a-z0-9-]+\s*,\s*#[0-9a-fA-F]{3,6}\s*\)")


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _order_ok(text: str, needles: list[str]) -> bool:
    idx = [text.find(n) for n in needles]
    assert all(i >= 0 for i in idx), f"missing one of {needles}: {idx}"
    return idx == sorted(idx)


def _no_raw_hex(text: str) -> list[str]:
    stripped = _VAR_FALLBACK.sub("", text)
    out = []
    for i, line in enumerate(stripped.splitlines(), 1):
        s = line.strip()
        if s.startswith("<!--") or s.startswith("//"):
            continue
        m = _HEX.search(s)
        if m:
            out.append(f"line {i}: {m.group()}")
    return out


def _no_french(text: str) -> list[str]:
    return [
        f"line {i}: {line.strip()[:60]}"
        for i, line in enumerate(text.splitlines(), 1)
        if _FRENCH.search(line)
    ]


# ---------------------------------------------------------------------------
# 1. Application shell
# ---------------------------------------------------------------------------
class TestApplicationShell:
    def test_header_component_exists(self):
        assert (_LAYOUT / "Header.svelte").is_file()

    def test_status_footer_component_exists(self):
        assert (_LAYOUT / "StatusFooter.svelte").is_file()

    def test_appshell_renders_header(self):
        src = _read(_LAYOUT / "AppShell.svelte")
        assert "import Header" in src and "<Header" in src

    def test_appshell_renders_status_footer(self):
        src = _read(_LAYOUT / "AppShell.svelte")
        assert "import StatusFooter" in src and "<StatusFooter" in src

    def test_appshell_keeps_landmarks(self):
        src = _read(_LAYOUT / "AppShell.svelte")
        assert "<header" in src
        assert "<nav" in src and 'aria-label="Sidebar navigation"' in src
        assert "<aside" in src and 'aria-label="Side panel"' in src
        assert 'id="main-content"' in src

    def test_appshell_keeps_skip_and_announcer(self):
        src = _read(_LAYOUT / "AppShell.svelte")
        assert "skip-to-content" in src and "#main-content" in src
        assert 'aria-live="polite"' in src and "oo-route-announcer" in src

    def test_appshell_keeps_mobile_swipe(self):
        src = _read(_LAYOUT / "AppShell.svelte")
        for marker in (
            "h-viewport",
            "handleSidebarTouchStart",
            "on:touchstart",
            "SWIPE_THRESHOLD",
            "touch-target",
            "safe-area-pad",
        ):
            assert marker in src, f"AppShell lost mobile marker: {marker}"

    def test_appshell_no_french(self):
        assert _no_french(_read(_LAYOUT / "AppShell.svelte")) == []


# ---------------------------------------------------------------------------
# 2. Header status cluster (re-asserts superseded s106/s107 placement)
# ---------------------------------------------------------------------------
class TestHeaderCluster:
    def test_header_imports_cluster(self):
        src = _read(_LAYOUT / "Header.svelte")
        for comp in ("BackendStatus", "ThemeSwitcher", "NotificationCenter", "UserMenu"):
            assert f"import {comp}" in src, f"Header missing import {comp}"

    def test_header_renders_cluster(self):
        src = _read(_LAYOUT / "Header.svelte")
        for comp in ("BackendStatus", "ThemeSwitcher", "NotificationCenter", "UserMenu"):
            assert f"<{comp}" in src, f"Header does not render {comp}"

    def test_header_cluster_order(self):
        src = _read(_LAYOUT / "Header.svelte")
        assert _order_ok(
            src, ["<BackendStatus", "<ThemeSwitcher", "<NotificationCenter", "<UserMenu"]
        )

    def test_header_uses_switcher_not_legacy_toggle(self):
        src = _read(_LAYOUT / "Header.svelte")
        assert "ThemeSwitcher" in src
        assert "<ThemeToggle" not in src

    def test_header_includes_backend_status(self):
        # Re-assertion of the superseded s106 test_appshell_includes_backend_status.
        src = _read(_LAYOUT / "Header.svelte")
        assert "BackendStatus" in src and "import BackendStatus" in src


# ---------------------------------------------------------------------------
# 3. Theme switcher accessibility + wiring
# ---------------------------------------------------------------------------
class TestThemeSwitcher:
    @pytest.fixture(scope="class")
    def src(self):
        return _read(_UI / "ThemeSwitcher.svelte")

    def test_exists(self):
        assert (_UI / "ThemeSwitcher.svelte").is_file()

    def test_trigger_aria(self, src):
        assert 'aria-haspopup="true"' in src
        assert "aria-expanded={open}" in src

    def test_menu_role(self, src):
        assert 'role="menu"' in src

    def test_menuitemradio(self, src):
        assert 'role="menuitemradio"' in src
        assert "aria-checked=" in src

    def test_keyboard_nav(self, src):
        for key in ("ArrowDown", "ArrowUp", "Home", "End", "Escape", "Enter"):
            assert key in src, f"ThemeSwitcher missing key handler: {key}"

    def test_roving_tabindex(self, src):
        assert "activeIndex" in src and "tabindex={activeIndex" in src

    def test_wired_to_preferences(self, src):
        for sym in ("setPalette", "setDensity", "PALETTES", "DENSITIES"):
            assert sym in src, f"ThemeSwitcher missing preferences symbol: {sym}"

    def test_no_raw_hex(self, src):
        assert _no_raw_hex(src) == []

    def test_density_options(self, src):
        assert "Density" in src


# ---------------------------------------------------------------------------
# 4. Preferences store + theme-file cascade fix
# ---------------------------------------------------------------------------
class TestPreferencesStore:
    @pytest.fixture(scope="class")
    def src(self):
        return _read(_STORES / "preferences.ts")

    def test_exists(self):
        assert (_STORES / "preferences.ts").is_file()

    def test_exports(self, src):
        for sym in (
            "palette",
            "density",
            "statusFooterVisible",
            "setPalette",
            "setDensity",
            "PALETTES",
            "DENSITIES",
            "PALETTE_SWATCH",
        ):
            assert f"export const {sym}" in src or f"export function {sym}" in src, sym

    def test_applies_data_oo_theme(self, src):
        assert "data-oo-theme" in src and "setAttribute" in src

    def test_applies_density_class(self, src):
        assert "oo-density-" in src

    def test_syncs_dark_mode(self, src):
        assert "darkMode.set" in src

    def test_persists_localstorage(self, src):
        assert "localStorage.setItem" in src and "oo-palette" in src and "oo-density" in src

    def test_five_palettes_three_densities(self, src):
        assert src.count("'anthracite'") >= 1
        # The PALETTES array enumerates the five named palettes.
        for p in ("anthracite", "parchment", "slate", "linen", "high-contrast"):
            assert f"'{p}'" in src, f"palette missing: {p}"
        for d in ("compact", "comfortable", "spacious"):
            assert f"'{d}'" in src, f"density missing: {d}"

    def test_theme_files_specificity_fix(self):
        for name in ("anthracite", "parchment", "slate", "linen", "high-contrast"):
            css = _read(_STYLES / f"theme-{name}.css")
            assert f':root[data-oo-theme="{name}"]' in css, (
                f"theme-{name}.css not raised to :root[data-oo-theme] specificity"
            )


# ---------------------------------------------------------------------------
# 5. Status footer
# ---------------------------------------------------------------------------
class TestStatusFooter:
    @pytest.fixture(scope="class")
    def src(self):
        return _read(_LAYOUT / "StatusFooter.svelte")

    def test_fields(self, src):
        for label in ("Mode", "Model", "Ctx", "Tokens"):
            assert f">{label}<" in src or f"{label}</span>" in src, f"missing field: {label}"

    def test_gated_by_preference_and_route(self, src):
        assert "statusFooterVisible" in src and "onChat" in src

    def test_uses_context_health(self, src):
        assert "getContextHealth" in src

    def test_aria_label(self, src):
        assert 'aria-label="Session status"' in src

    def test_no_raw_hex(self, src):
        assert _no_raw_hex(src) == []


# ---------------------------------------------------------------------------
# 6. Header status merge (NetworkIndicator folded into BackendStatus)
# ---------------------------------------------------------------------------
class TestBackendStatusMerge:
    @pytest.fixture(scope="class")
    def src(self):
        return _read(_UI / "BackendStatus.svelte")

    def test_fetches_network_status(self, src):
        assert "fetch('/api/network/status')" in src

    def test_shows_ollama(self, src):
        assert "Ollama" in src

    def test_keeps_status_vars(self, src):
        for v in ("--oo-success", "--oo-warning", "--oo-error"):
            assert v in src

    def test_no_six_digit_hex(self, src):
        # s106 parity: BackendStatus must not carry raw 6-digit hex.
        assert _no_raw_hex(src) == []

    def test_control_bar_no_longer_imports_indicator(self):
        # Re-assertion of the superseded routes_network import check (the merge).
        src = _read(_CHAT / "ChatControlBar.svelte")
        assert "NetworkIndicator" not in src


# ---------------------------------------------------------------------------
# 7. Sidebar split (App Nav + Section Context)
# ---------------------------------------------------------------------------
class TestSidebarSplit:
    @pytest.fixture(scope="class")
    def src(self):
        return _read(_LAYOUT / "Sidebar.svelte")

    def test_imports_section_context_list(self, src):
        assert "import SectionContextList" in src and "<SectionContextList" in src

    def test_keeps_s132_markers(self, src):
        for m in ("min-height: 44px", "touch-scroll", "safe-area-bottom", "touch-target", "S132"):
            assert m in src, f"Sidebar lost s132 marker: {m}"

    def test_app_nav_links(self, src):
        for href in ("/chat", "/projects", "/settings", "/benchmark", "/health"):
            assert f'href="{href}"' in src or f"href={{'{href}'}}" in src or href in src, href

    def test_footer_toggle_palette_aware(self, src):
        # Re-assertion of the superseded s107 footer-toggle check: the footer
        # toggle now drives the palette system rather than the binary store.
        assert "setPalette" in src and "touch-target" in src

    def test_section_context_list_exists(self):
        assert (_SIDEBAR / "SectionContextList.svelte").is_file()

    def test_no_french(self, src):
        assert _no_french(src) == []


# ---------------------------------------------------------------------------
# 8. Section context per route
# ---------------------------------------------------------------------------
class TestSectionContextList:
    @pytest.fixture(scope="class")
    def src(self):
        return _read(_SIDEBAR / "SectionContextList.svelte")

    def test_route_driven(self, src):
        assert "$page" in src
        for route in ("/projects", "/settings", "/benchmark", "/health"):
            assert route in src, f"SectionContextList missing route key: {route}"

    def test_chat_variant(self, src):
        assert "NewConversationButton" in src and "ConversationList" in src

    def test_passes_on_new_conversation(self, src):
        # Re-assertion of the superseded s87 test_on_new_conversation_passed.
        assert "onNewConversation={onCreate}" in src

    def test_settings_variant(self, src):
        assert "Settings sections" in src and "/settings?tab=" in src

    def test_no_french(self, src):
        assert _no_french(src) == []


# ---------------------------------------------------------------------------
# 9. Sidebar components (primitive adoption, re-assertions)
# ---------------------------------------------------------------------------
class TestSidebarComponents:
    def test_new_conversation_uses_button(self):
        # Re-assertion of the superseded s93 test_new_conversation_uses_sage:
        # the button now uses the ds Button primitive.
        src = _read(_SIDEBAR / "NewConversationButton.svelte")
        assert "import Button" in src and "<Button" in src

    def test_conversation_list_date_grouping(self):
        src = _read(_SIDEBAR / "ConversationList.svelte")
        for label in ("Today", "Yesterday", "Previous 7 days", "Older"):
            assert label in src, f"ConversationList missing date group: {label}"

    def test_conversation_list_empty_state_icon(self):
        # Re-assertion of the superseded s87 test_chat_icon_in_guidance: the
        # guidance bubble now uses the Icon primitive.
        src = _read(_SIDEBAR / "ConversationList.svelte")
        assert 'Icon name="message-square"' in src

    def test_conversation_item_tooltip_icon(self):
        src = _read(_SIDEBAR / "ConversationItem.svelte")
        assert "import Tooltip" in src and "import Icon" in src

    def test_security_badge_tooltip(self):
        src = _read(_SIDEBAR / "SecurityBadge.svelte")
        assert "import Tooltip" in src

    def test_new_conversation_english(self):
        assert _no_french(_read(_SIDEBAR / "NewConversationButton.svelte")) == []


# ---------------------------------------------------------------------------
# 10. Chat sub-components on ds primitives
# ---------------------------------------------------------------------------
class TestChatPrimitives:
    def test_chat_message_icon_and_live_region(self):
        src = _read(_CHAT / "ChatMessage.svelte")
        assert "import Icon" in src
        assert "aria-live" in src

    def test_chat_message_keeps_s132(self):
        src = _read(_CHAT / "ChatMessage.svelte")
        assert "px-2.5" in src and "sm:px-4" in src and "overflow-x: auto" in src

    def test_chat_input_icon_and_s132(self):
        src = _read(_CHAT / "ChatInput.svelte")
        assert "import Icon" in src
        assert 'enterkeyhint="send"' in src
        assert "font-size: 16px" in src
        assert "width: 44px" in src and "touch-target" in src

    def test_model_selector_icon_and_ram_hint(self):
        src = _read(_CHAT / "ModelSelector.svelte")
        assert "import Icon" in src
        # E1 RAM hint surfaces the on-disk footprint.
        assert "model.size" in src and "effectiveSize" in src

    def test_preset_selector_icon(self):
        src = _read(_CHAT / "PresetSelector.svelte")
        assert "import Icon" in src

    def test_feedback_widget_icon_button(self):
        src = _read(_CHAT / "FeedbackWidget.svelte")
        assert "import Icon" in src and "import Button" in src
        assert 'name="thumbs-up"' in src

    def test_project_context_badge_tooltip(self):
        src = _read(_CHAT / "ProjectContextBadge.svelte")
        assert "import Tooltip" in src

    def test_sandbox_badge_icon_no_hex(self):
        src = _read(_CHAT / "SandboxIsolationBadge.svelte")
        assert "import Icon" in src
        assert "#d97706" not in src and "#dc2626" not in src

    def test_message_skeleton_english(self):
        assert _no_french(_read(_CHAT / "MessageSkeleton.svelte")) == []


# ---------------------------------------------------------------------------
# 11. Layout accessibility + toast cutover
# ---------------------------------------------------------------------------
class TestLayoutA11y:
    @pytest.fixture(scope="class")
    def root(self):
        return _read(_ROUTES / "+layout.svelte")

    def test_skip_link_present(self, root):
        assert 'class="oo-skip-link"' in root and 'href="#main-content"' in root

    def test_skip_link_css_exists(self):
        assert ".oo-skip-link" in _read(_APP_CSS)

    def test_root_mounts_ds_toast(self, root):
        assert "$lib/ds/Toast.svelte" in root and "<Toast" in root

    def test_single_toast_mount(self, root):
        assert root.count("<Toast") == 1

    def test_per_route_toast_removed(self):
        # Re-assertion of the superseded s107 test_chat_layout_has_toast:
        # toasts are now mounted once at the root, not per route.
        for route in ("chat", "settings", "health", "projects"):
            src = _read(_ROUTES / route / "+layout.svelte")
            assert "ui/Toast" not in src, f"{route} layout still mounts ui/Toast"
            assert "<Toast" not in src, f"{route} layout still mounts a Toast"

    def test_ds_toast_live_region(self):
        src = _read(_DS / "Toast.svelte")
        assert "aria-live" in src

    def test_route_announcer_in_layout(self, root):
        assert "oo-route-announcer" in root


# ---------------------------------------------------------------------------
# 12. Sanity: balance + no new raw hex on new files
# ---------------------------------------------------------------------------
class TestSanity:
    NEW_SVELTE = [
        _LAYOUT / "Header.svelte",
        _LAYOUT / "StatusFooter.svelte",
        _UI / "ThemeSwitcher.svelte",
        _SIDEBAR / "SectionContextList.svelte",
    ]

    @pytest.mark.parametrize("path", NEW_SVELTE, ids=lambda p: p.name)
    def test_brace_balanced(self, path):
        s = _read(path)
        assert s.count("{") == s.count("}"), f"unbalanced braces in {path.name}"

    @pytest.mark.parametrize("path", NEW_SVELTE, ids=lambda p: p.name)
    def test_no_raw_hex(self, path):
        assert _no_raw_hex(_read(path)) == [], f"raw hex in {path.name}"

    def test_if_blocks_balanced_theme_switcher(self):
        s = _read(_UI / "ThemeSwitcher.svelte")
        assert s.count("{#if") + s.count("{#each") == s.count("{/if") + s.count("{/each")
