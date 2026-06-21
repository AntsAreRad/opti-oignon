#!/usr/bin/env python3
"""
Tests for S107 -- UX Improvements.

Covers:
- D1: Keyboard shortcuts (Ctrl+K, Ctrl+Enter, Ctrl+Shift+T, Escape, ?)
- D2: Dark/light toggle in navbar (ThemeToggle.svelte)
- D3: Conversation export from chat (already existed, verify integration)
- D4: Notification center (persistent history, bell badge, dropdown)
- D5: Dashboard home screen (models, health, recent chats, quick actions)
- Logo replacement: bousier-oignon in sidebar, login, register, favicon
- French comment cleanup in modified files
- CSS variable compliance (no hardcoded hex colors)
"""

import ast
import os
import re
import sys
from pathlib import Path
from typing import List, Set

import pytest

# =========================================================================
# PATHS
# =========================================================================

ROOT = Path(__file__).resolve().parent.parent
OO_DIR = ROOT / "opti_oignon"
FRONTEND_DIR = ROOT / "frontend"
SRC_DIR = FRONTEND_DIR / "src"
COMPONENTS_DIR = SRC_DIR / "lib" / "components"
STORES_DIR = SRC_DIR / "lib" / "stores"
ROUTES_DIR = SRC_DIR / "routes"
STATIC_DIR = FRONTEND_DIR / "static"
ASSETS_DIR = ROOT / "assets"


# =========================================================================
# HELPERS
# =========================================================================

def _read(filepath: Path) -> str:
    """Read a file as text."""
    return filepath.read_text(encoding="utf-8")


def _svelte_files(directory: Path) -> list[Path]:
    """Recursively find all .svelte files in a directory."""
    return list(directory.rglob("*.svelte"))


def _ts_files(directory: Path) -> list[Path]:
    """Recursively find all .ts files in a directory."""
    return list(directory.rglob("*.ts"))


# =========================================================================
# D1: KEYBOARD SHORTCUTS
# =========================================================================

class TestKeyboardShortcuts:
    """Tests for the extended keyboard shortcuts system."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ks_path = COMPONENTS_DIR / "ui" / "KeyboardShortcuts.svelte"
        assert self.ks_path.exists(), "KeyboardShortcuts.svelte must exist"
        self.content = _read(self.ks_path)

    def test_file_exists(self):
        """KeyboardShortcuts.svelte exists."""
        assert self.ks_path.exists()

    def test_ctrl_shift_t_theme_shortcut(self):
        """Ctrl+Shift+T theme toggle shortcut is defined."""
        assert "key: 't'" in self.content or "key: \"t\"" in self.content
        assert "Toggle dark" in self.content or "theme" in self.content.lower()

    def test_ctrl_enter_send_shortcut(self):
        """Ctrl+Enter send message shortcut is defined."""
        assert "key: 'Enter'" in self.content or "key: \"Enter\"" in self.content
        assert "Send message" in self.content

    def test_ctrl_k_search_shortcut(self):
        """Ctrl+K search shortcut is defined."""
        assert "key: 'k'" in self.content or "key: \"k\"" in self.content
        assert "search" in self.content.lower()

    def test_escape_shortcut(self):
        """Escape close shortcut is defined."""
        assert "key: 'Escape'" in self.content or "key: \"Escape\"" in self.content
        assert "Close" in self.content

    def test_question_mark_help(self):
        """? shortcut for showing help is defined."""
        assert "key: '?'" in self.content or "key: \"?\"" in self.content
        assert "keyboard shortcuts" in self.content.lower()

    def test_on_toggle_theme_prop(self):
        """onToggleTheme callback prop exists."""
        assert "onToggleTheme" in self.content

    def test_opti_send_message_event(self):
        """Dispatches opti-send-message custom event for Ctrl+Enter."""
        assert "opti-send-message" in self.content

    def test_all_comments_english(self):
        """All comments in KeyboardShortcuts are in English (no French)."""
        french_patterns = [
            r"Gestionnaire", r"raccourcis", r"clavier", r"champ de saisie",
            r"uniquement hors", r"fermer", r"Noms de touches", r"affichage",
        ]
        for pat in french_patterns:
            assert not re.search(pat, self.content, re.IGNORECASE), \
                f"French pattern found: {pat}"

    def test_help_dialog_uses_oo_variables(self):
        """Help dialog uses --oo-* CSS variables, not Tailwind surface classes in <style>."""
        # Extract <style> block
        style_match = re.search(r"<style>(.*?)</style>", self.content, re.DOTALL)
        if style_match:
            style = style_match.group(1)
            assert "--oo-" in style, "Help dialog <style> should use --oo-* CSS variables"

    def test_shortcut_count(self):
        """At least 8 shortcuts defined (including new ones)."""
        shortcut_matches = re.findall(r"\{\s*key:", self.content)
        assert len(shortcut_matches) >= 8, \
            f"Expected >=8 shortcuts, found {len(shortcut_matches)}"


class TestKeyboardShortcutsIntegration:
    """Test that keyboard shortcuts are properly wired in root layout."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.layout_path = ROUTES_DIR / "+layout.svelte"
        assert self.layout_path.exists()
        self.content = _read(self.layout_path)

    def test_toggle_theme_imported(self):
        """toggleTheme is imported from ui store."""
        assert "toggleTheme" in self.content
        assert "from '$lib/stores/ui'" in self.content

    def test_toggle_theme_passed(self):
        """onToggleTheme prop is passed to KeyboardShortcuts."""
        assert "onToggleTheme" in self.content

    def test_all_callbacks_wired(self):
        """All shortcut callbacks are wired to KeyboardShortcuts."""
        for prop in ["onNewConversation", "onExportConversation",
                      "onGoToSettings", "onToggleSearch", "onToggleTheme"]:
            assert prop in self.content, f"Missing prop: {prop}"


class TestChatInputCtrlEnter:
    """Test Ctrl+Enter integration in ChatInput."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.path = COMPONENTS_DIR / "chat" / "ChatInput.svelte"
        assert self.path.exists()
        self.content = _read(self.path)

    def test_ctrl_enter_in_keydown(self):
        """ChatInput handles Ctrl+Enter in its own keydown handler."""
        assert "event.ctrlKey || event.metaKey" in self.content
        assert "event.key === 'Enter'" in self.content or "Enter" in self.content

    def test_global_send_event_listener(self):
        """ChatInput listens for opti-send-message global event."""
        assert "opti-send-message" in self.content

    def test_send_tooltip_hint(self):
        """Send button tooltip includes Ctrl+Enter hint."""
        assert "Ctrl+Enter" in self.content

    def test_english_comments(self):
        """ChatInput header comment is in English."""
        first_lines = self.content[:500]
        assert "Zone de saisie" not in first_lines
        assert "Text input" in first_lines or "ChatInput" in first_lines

    def test_search_placeholder_hint(self):
        """Search input placeholder shows Ctrl+K hint."""
        conv_list = COMPONENTS_DIR / "sidebar" / "ConversationList.svelte"
        if conv_list.exists():
            content = _read(conv_list)
            assert "Ctrl+K" in content, "Search placeholder should show Ctrl+K hint"


# =========================================================================
# D2: DARK/LIGHT TOGGLE IN NAVBAR
# =========================================================================

class TestThemeToggle:
    """Tests for the ThemeToggle component."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.path = COMPONENTS_DIR / "ui" / "ThemeToggle.svelte"
        assert self.path.exists(), "ThemeToggle.svelte must exist"
        self.content = _read(self.path)

    def test_file_exists(self):
        """ThemeToggle.svelte exists."""
        assert self.path.exists()

    def test_imports_toggle_theme(self):
        """Imports toggleTheme from ui store."""
        assert "toggleTheme" in self.content
        assert "from '$lib/stores/ui'" in self.content

    def test_imports_dark_mode(self):
        """Imports darkMode store for conditional rendering."""
        assert "darkMode" in self.content

    def test_sun_icon_present(self):
        """Sun icon SVG is present (for dark mode -> click to go light)."""
        # Sun icon typically has a circle and rays
        assert "circle" in self.content.lower() or "r=\"5\"" in self.content

    def test_moon_icon_present(self):
        """Moon icon SVG is present (for light mode -> click to go dark)."""
        assert "12.79" in self.content or "moon" in self.content.lower()

    def test_shortcut_hint_in_tooltip(self):
        """Tooltip shows Ctrl+Shift+T shortcut hint."""
        assert "Ctrl+Shift+T" in self.content

    def test_uses_oo_css_variables(self):
        """Uses --oo-* CSS variables exclusively."""
        style_match = re.search(r"<style>(.*?)</style>", self.content, re.DOTALL)
        if style_match:
            style = style_match.group(1)
            assert "--oo-" in style
            # No hardcoded hex colors
            hex_matches = re.findall(r"#[0-9a-fA-F]{3,6}\b", style)
            assert len(hex_matches) == 0, f"Hardcoded hex found in style: {hex_matches}"

    def test_accessible_aria_label(self):
        """Has aria-label for accessibility."""
        assert "aria-label" in self.content


class TestThemeToggleInAppShell:
    """Test ThemeToggle is placed in the AppShell header."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.path = COMPONENTS_DIR / "layout" / "AppShell.svelte"
        assert self.path.exists()
        self.content = _read(self.path)

    def test_theme_toggle_imported(self):
        """ThemeToggle is imported in AppShell."""
        assert "import ThemeToggle" in self.content

    def test_theme_toggle_rendered(self):
        """ThemeToggle is rendered in the header."""
        assert "<ThemeToggle" in self.content

    def test_order_backend_theme_notif_user(self):
        """Header order: BackendStatus, ThemeToggle, NotificationCenter, UserMenu."""
        idx_backend = self.content.index("<BackendStatus")
        idx_theme = self.content.index("<ThemeToggle")
        idx_user = self.content.index("<UserMenu")
        assert idx_backend < idx_theme < idx_user


# =========================================================================
# D3: CONVERSATION EXPORT
# =========================================================================

class TestExportDialog:
    """Tests for ExportDialog (pre-existing, verify English + integration)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.path = COMPONENTS_DIR / "chat" / "ExportDialog.svelte"
        assert self.path.exists()
        self.content = _read(self.path)

    def test_export_dialog_exists(self):
        """ExportDialog.svelte exists."""
        assert self.path.exists()

    def test_formats_available(self):
        """Markdown, JSON, HTML formats are available."""
        assert "markdown" in self.content.lower()
        assert "json" in self.content.lower()
        assert "html" in self.content.lower()

    def test_english_header_comment(self):
        """Header comment is in English."""
        first_500 = self.content[:500]
        assert "Dialogue modal" not in first_500, "French found in header comment"

    def test_export_button_in_chat_header(self):
        """Export button exists in the chat layout header."""
        layout = _read(ROUTES_DIR / "chat" / "+layout.svelte")
        assert "Export conversation" in layout

    def test_ctrl_shift_e_hint(self):
        """Export button has Ctrl+Shift+E shortcut hint."""
        layout = _read(ROUTES_DIR / "chat" / "+layout.svelte")
        assert "Ctrl+Shift+E" in layout


# =========================================================================
# D4: NOTIFICATION CENTER
# =========================================================================

class TestNotificationStore:
    """Tests for the extended notification store."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.path = STORES_DIR / "notifications.ts"
        assert self.path.exists()
        self.content = _read(self.path)

    def test_notification_item_type(self):
        """NotificationItem interface is defined."""
        assert "NotificationItem" in self.content

    def test_notification_history_store(self):
        """notificationHistory writable store is defined."""
        assert "notificationHistory" in self.content

    def test_unread_count_derived(self):
        """unreadCount derived store is defined."""
        assert "unreadCount" in self.content

    def test_mark_notification_read(self):
        """markNotificationRead function exists."""
        assert "markNotificationRead" in self.content

    def test_mark_all_read(self):
        """markAllRead function exists."""
        assert "markAllRead" in self.content

    def test_clear_history(self):
        """clearNotificationHistory function exists."""
        assert "clearNotificationHistory" in self.content

    def test_add_toast_populates_history(self):
        """addToast function also adds to notificationHistory."""
        # Check that addToast body references notificationHistory.update
        add_toast_area = self.content[self.content.index("function addToast"):]
        assert "notificationHistory.update" in add_toast_area

    def test_max_history_limit(self):
        """History is capped at MAX_HISTORY items."""
        assert "MAX_HISTORY" in self.content
        assert "slice(0, MAX_HISTORY)" in self.content

    def test_timestamp_field(self):
        """NotificationItem has a timestamp field."""
        assert "timestamp" in self.content
        assert "Date.now()" in self.content

    def test_read_field(self):
        """NotificationItem has a read boolean field."""
        assert "read: boolean" in self.content or "read:" in self.content

    def test_english_comments(self):
        """All comments are in English."""
        french_patterns = [
            r"Pile de notifications",
            r"Gere une pile",
            r"Affiche une notification",
            r"Supprime un toast",
            r"Raccourcis pour",
        ]
        for pat in french_patterns:
            assert not re.search(pat, self.content), f"French pattern found: {pat}"


class TestNotificationCenter:
    """Tests for the NotificationCenter component."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.path = COMPONENTS_DIR / "ui" / "NotificationCenter.svelte"
        assert self.path.exists(), "NotificationCenter.svelte must exist"
        self.content = _read(self.path)

    def test_file_exists(self):
        """NotificationCenter.svelte exists."""
        assert self.path.exists()

    def test_bell_icon(self):
        """Bell icon SVG is present."""
        # Bell icon path: M18 8A6 6 0 006 8c0 7-3 9-3 9h18...
        assert "M18 8" in self.content or "bell" in self.content.lower()

    def test_unread_badge(self):
        """Unread count badge is displayed."""
        assert "unreadCount" in self.content
        assert "notif-badge" in self.content or "badge" in self.content.lower()

    def test_notification_history_import(self):
        """Imports notificationHistory from store."""
        assert "notificationHistory" in self.content

    def test_mark_all_read_on_open(self):
        """Marks all notifications as read when panel opens."""
        assert "markAllRead" in self.content

    def test_clear_action(self):
        """Clear all button exists."""
        assert "clearNotificationHistory" in self.content or "Clear" in self.content

    def test_empty_state(self):
        """Shows empty state message when no notifications."""
        assert "No notifications" in self.content

    def test_timestamp_display(self):
        """Shows relative timestamps for notifications."""
        assert "formatTime" in self.content or "ago" in self.content

    def test_type_icons(self):
        """Different icons for notification types (success, error, warning, info)."""
        assert "success" in self.content
        assert "error" in self.content
        assert "warning" in self.content

    def test_uses_oo_css_variables(self):
        """Uses --oo-* CSS variables exclusively."""
        style_match = re.search(r"<style>(.*?)</style>", self.content, re.DOTALL)
        if style_match:
            style = style_match.group(1)
            assert "--oo-" in style
            hex_matches = re.findall(r":\s*#[0-9a-fA-F]{3,6}\b", style)
            assert len(hex_matches) == 0, f"Hardcoded hex found: {hex_matches}"

    def test_dropdown_panel(self):
        """Dropdown panel with notification list exists."""
        assert "notif-panel" in self.content or "panel" in self.content.lower()

    def test_click_outside_closes(self):
        """Panel closes on click outside."""
        assert "handleClickOutside" in self.content or "clickOutside" in self.content


class TestNotificationCenterInAppShell:
    """Test NotificationCenter is placed in AppShell header."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.content = _read(COMPONENTS_DIR / "layout" / "AppShell.svelte")

    def test_imported(self):
        """NotificationCenter is imported."""
        assert "import NotificationCenter" in self.content

    def test_rendered(self):
        """NotificationCenter is rendered."""
        assert "<NotificationCenter" in self.content


# =========================================================================
# D5: DASHBOARD HOME SCREEN
# =========================================================================

class TestDashboardHome:
    """Tests for the DashboardHome component."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.path = COMPONENTS_DIR / "panels" / "DashboardHome.svelte"
        assert self.path.exists(), "DashboardHome.svelte must exist"
        self.content = _read(self.path)

    def test_file_exists(self):
        """DashboardHome.svelte exists."""
        assert self.path.exists()

    def test_health_status_display(self):
        """Shows backend health status."""
        assert "backendStatus" in self.content
        assert "Connected" in self.content
        assert "Disconnected" in self.content

    def test_version_display(self):
        """Shows backend version."""
        assert "backendVersion" in self.content

    def test_modules_count(self):
        """Shows active modules count."""
        assert "backendModules" in self.content
        assert "moduleActive" in self.content or "active" in self.content.lower()

    def test_models_list(self):
        """Fetches and displays available models."""
        assert "/api/models" in self.content
        assert "models" in self.content

    def test_recent_conversations(self):
        """Fetches and displays recent conversations."""
        assert "/api/conversations" in self.content
        assert "recentChats" in self.content or "recent" in self.content.lower()

    def test_new_chat_action(self):
        """New conversation quick action button exists."""
        assert "New conversation" in self.content
        assert "handleNewChat" in self.content or "createNewConversation" in self.content

    def test_benchmark_link(self):
        """Benchmark quick action link exists."""
        assert "/benchmark" in self.content
        assert "benchmark" in self.content.lower()

    def test_settings_link(self):
        """Settings quick action link exists."""
        assert "/settings" in self.content

    def test_keyboard_hint(self):
        """Shows keyboard shortcut hint."""
        assert "keyboard shortcuts" in self.content.lower() or "?" in self.content

    def test_bousier_logo(self):
        """Uses bousier-oignon logo."""
        assert "bousier-oignon.png" in self.content

    def test_uses_oo_css_variables(self):
        """Uses --oo-* CSS variables exclusively."""
        style_match = re.search(r"<style>(.*?)</style>", self.content, re.DOTALL)
        if style_match:
            style = style_match.group(1)
            assert "--oo-" in style
            hex_matches = re.findall(r":\s*#[0-9a-fA-F]{3,6}\b", style)
            assert len(hex_matches) == 0, f"Hardcoded hex found: {hex_matches}"

    def test_responsive_grid(self):
        """Grid layout has responsive breakpoint."""
        assert "grid-template-columns" in self.content
        assert "@media" in self.content

    def test_loading_state(self):
        """Shows loading state while fetching data."""
        assert "loading" in self.content
        assert "Loading" in self.content

    def test_empty_models_state(self):
        """Shows hint when no models found."""
        assert "No models" in self.content or "Ollama" in self.content

    def test_empty_chats_state(self):
        """Shows hint when no conversations exist."""
        assert "No conversations" in self.content or "Start a new one" in self.content


class TestDashboardIntegration:
    """Test DashboardHome is wired in the chat route."""

    def test_chat_page_uses_dashboard(self):
        """chat/+page.svelte imports and renders DashboardHome."""
        path = ROUTES_DIR / "chat" / "+page.svelte"
        content = _read(path)
        assert "DashboardHome" in content
        assert "<DashboardHome" in content


# =========================================================================
# LOGO REPLACEMENT
# =========================================================================

class TestLogoReplacement:
    """Tests for bousier-oignon logo integration."""

    def test_bousier_png_in_assets(self):
        """bousier-oignon.png exists in assets/."""
        assert (ASSETS_DIR / "bousier-oignon.png").exists()

    def test_bousier_png_in_static(self):
        """bousier-oignon.png exists in frontend/static/."""
        assert (STATIC_DIR / "bousier-oignon.png").exists()

    def test_favicon_ico_exists(self):
        """favicon.ico exists in frontend/static/."""
        assert (STATIC_DIR / "favicon.ico").exists()

    def test_favicon_32_exists(self):
        """favicon-32.png exists in frontend/static/."""
        assert (STATIC_DIR / "favicon-32.png").exists()

    def test_icon_192_exists(self):
        """icon-192.png exists in frontend/static/."""
        assert (STATIC_DIR / "icon-192.png").exists()

    def test_app_html_favicon(self):
        """app.html references favicon.ico."""
        content = _read(SRC_DIR / "app.html")
        assert "favicon.ico" in content

    def test_app_html_no_emoji_favicon(self):
        """app.html no longer uses inline emoji favicon."""
        content = _read(SRC_DIR / "app.html")
        assert "data:image/svg+xml" not in content

    def test_sidebar_uses_bousier(self):
        """Sidebar header uses bousier-oignon.png image."""
        content = _read(COMPONENTS_DIR / "layout" / "Sidebar.svelte")
        assert "bousier-oignon.png" in content

    def test_login_uses_bousier(self):
        """Login page uses bousier-oignon.png instead of SVG ellipses."""
        content = _read(ROUTES_DIR / "login" / "+page.svelte")
        assert "bousier-oignon.png" in content
        # No more SVG ellipses for the logo
        assert 'rx="18" ry="22"' not in content

    def test_register_uses_bousier(self):
        """Register page uses bousier-oignon.png instead of SVG ellipses."""
        content = _read(ROUTES_DIR / "register" / "+page.svelte")
        assert "bousier-oignon.png" in content
        assert 'rx="18" ry="22"' not in content


# =========================================================================
# FRENCH COMMENT CLEANUP
# =========================================================================

class TestFrenchCommentCleanup:
    """Verify French comments were removed from modified files."""

    MODIFIED_FILES = [
        COMPONENTS_DIR / "ui" / "KeyboardShortcuts.svelte",
        COMPONENTS_DIR / "ui" / "Toast.svelte",
        COMPONENTS_DIR / "chat" / "ChatInput.svelte",
        COMPONENTS_DIR / "chat" / "ExportDialog.svelte",
        COMPONENTS_DIR / "layout" / "Sidebar.svelte",
        STORES_DIR / "notifications.ts",
        ROUTES_DIR / "chat" / "+page.svelte",
        ROUTES_DIR / "chat" / "+layout.svelte",
    ]

    # French words that should NOT appear in comments
    FRENCH_PATTERNS = [
        r"\bGestionnaire\b",
        r"\braccourcis\b",
        r"\bclavier\b",
        r"\bConteneur\b",
        r"\bGere\b",
        r"\bafficher\b",
        r"\bmasquer\b",
        r"\bPile de\b",
        r"\bZone de saisie\b",
        r"\bBarre laterale\b",
        r"\bDialogue modal\b",
        r"\bapercu\b",
        r"\btelecharger\b",
        r"\bpresse-papiers\b",
    ]

    @pytest.mark.parametrize("filepath", MODIFIED_FILES)
    def test_no_french_in_comments(self, filepath: Path):
        """No French words in comments of modified files."""
        if not filepath.exists():
            pytest.skip(f"File not found: {filepath}")
        content = _read(filepath)
        for pat in self.FRENCH_PATTERNS:
            match = re.search(pat, content, re.IGNORECASE)
            assert match is None, \
                f"French pattern '{pat}' found in {filepath.name}: ...{content[max(0, match.start()-20):match.end()+20]}..."


# =========================================================================
# CSS VARIABLE COMPLIANCE
# =========================================================================

class TestCSSVariableCompliance:
    """Verify new S107 components use --oo-* CSS variables only."""

    NEW_COMPONENTS = [
        COMPONENTS_DIR / "ui" / "ThemeToggle.svelte",
        COMPONENTS_DIR / "ui" / "NotificationCenter.svelte",
        COMPONENTS_DIR / "panels" / "DashboardHome.svelte",
        COMPONENTS_DIR / "ui" / "KeyboardShortcuts.svelte",
    ]

    @pytest.mark.parametrize("filepath", NEW_COMPONENTS)
    def test_no_hardcoded_hex_in_style(self, filepath: Path):
        """No hardcoded hex colors in <style> blocks."""
        if not filepath.exists():
            pytest.skip(f"File not found: {filepath}")
        content = _read(filepath)
        style_match = re.search(r"<style>(.*?)</style>", content, re.DOTALL)
        if not style_match:
            return  # No style block, OK
        style = style_match.group(1)
        # Find hex colors like #fff, #1a1a1a but not in comments
        hex_matches = re.findall(r":\s*#[0-9a-fA-F]{3,8}\b", style)
        assert len(hex_matches) == 0, \
            f"Hardcoded hex in {filepath.name}: {hex_matches}"

    @pytest.mark.parametrize("filepath", NEW_COMPONENTS)
    def test_uses_oo_variables(self, filepath: Path):
        """Style blocks reference --oo-* CSS variables."""
        if not filepath.exists():
            pytest.skip(f"File not found: {filepath}")
        content = _read(filepath)
        style_match = re.search(r"<style>(.*?)</style>", content, re.DOTALL)
        if not style_match:
            return
        style = style_match.group(1)
        assert "--oo-" in style, \
            f"{filepath.name} <style> should use --oo-* variables"


# =========================================================================
# STRUCTURAL INTEGRITY
# =========================================================================

class TestStructuralIntegrity:
    """Verify files haven't been accidentally broken."""

    def test_app_shell_has_all_imports(self):
        """AppShell imports BackendStatus, ThemeToggle, NotificationCenter, UserMenu."""
        content = _read(COMPONENTS_DIR / "layout" / "AppShell.svelte")
        for component in ["BackendStatus", "ThemeToggle", "NotificationCenter", "UserMenu"]:
            assert f"import {component}" in content, f"Missing import: {component}"

    def test_root_layout_has_keyboard_shortcuts(self):
        """Root layout still mounts KeyboardShortcuts."""
        content = _read(ROUTES_DIR / "+layout.svelte")
        assert "<KeyboardShortcuts" in content

    def test_chat_layout_has_toast(self):
        """Chat layout still renders Toast component."""
        content = _read(ROUTES_DIR / "chat" / "+layout.svelte")
        assert "<Toast" in content

    def test_chat_layout_has_export_dialog(self):
        """Chat layout still renders ExportDialog."""
        content = _read(ROUTES_DIR / "chat" / "+layout.svelte")
        assert "<ExportDialog" in content
        assert "ExportDialog" in content

    def test_sidebar_still_has_nav_links(self):
        """Sidebar still has navigation links."""
        content = _read(COMPONENTS_DIR / "layout" / "Sidebar.svelte")
        for link in ["/chat", "/projects", "/settings", "/benchmark", "/health"]:
            assert link in content, f"Missing nav link: {link}"

    def test_sidebar_still_has_theme_toggle_in_footer(self):
        """Sidebar footer still has the theme toggle (legacy location preserved)."""
        content = _read(COMPONENTS_DIR / "layout" / "Sidebar.svelte")
        assert "toggleTheme" in content

    def test_toast_still_uses_correct_imports(self):
        """Toast.svelte still imports from notifications store correctly."""
        content = _read(COMPONENTS_DIR / "ui" / "Toast.svelte")
        assert "toasts" in content
        assert "removeToast" in content
        assert "from '$lib/stores/notifications'" in content

    def test_notifications_store_backward_compatible(self):
        """All pre-existing exports still present in notifications store."""
        content = _read(STORES_DIR / "notifications.ts")
        for export_name in ["toasts", "addToast", "removeToast",
                            "toastSuccess", "toastError", "toastWarning", "toastInfo",
                            "ToastItem", "ToastType"]:
            assert export_name in content, f"Missing export: {export_name}"
