"""
Tests for S87 -- UX Fixes & Web Search Polish.

Validates:
- Version bump to 1.8.9 (app.py, pyproject.toml, setup.py, health endpoint)
- ChatControlBar: responsive labels, unified copper glow, ddgs check, model grouping
- ConversationList: empty state guidance card, onNewConversation prop, delete redirect
- ConversationItem: S83 palette (copper highlight, CSS variable styles)
- ProxySettingsPanel: proxy status badge, ddgs install warning
- client.ts: actionable error messages, network vs API error distinction
- Settings Quick tab: version display header
- Sidebar: dynamic version from health endpoint
- No regressions on existing test conventions
"""

import ast
import os
import re
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
BACKEND_DIR = os.path.join(PROJECT_ROOT, 'opti_oignon')
API_DIR = os.path.join(BACKEND_DIR, 'api')
FRONTEND_SRC = os.path.join(PROJECT_ROOT, 'frontend', 'src')


def _read(path):
    """Read file content safely."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# ===================================================================
# 1. Version bump
# ===================================================================

class TestVersionBump(unittest.TestCase):
    """Verify version is 1.8.9 across all files."""

    def test_app_py_fastapi_version(self):
        content = _read(os.path.join(API_DIR, 'app.py'))
        self.assertIn('version="1.8.9"', content)

    def test_app_py_health_version(self):
        content = _read(os.path.join(API_DIR, 'app.py'))
        self.assertIn('"version": "1.8.9"', content)

    def test_pyproject_toml_version(self):
        content = _read(os.path.join(PROJECT_ROOT, 'pyproject.toml'))
        self.assertIn('version = "1.8.9"', content)

    def test_setup_py_version(self):
        content = _read(os.path.join(PROJECT_ROOT, 'setup.py'))
        self.assertIn('version="1.8.9"', content)

    def test_no_old_version_in_app(self):
        content = _read(os.path.join(API_DIR, 'app.py'))
        self.assertNotIn('"1.8.8"', content)


# ===================================================================
# 2. ChatControlBar
# ===================================================================

class TestChatControlBarResponsiveLabels(unittest.TestCase):
    """Verify responsive label pattern in ChatControlBar."""

    @classmethod
    def setUpClass(cls):
        cls.content = _read(os.path.join(
            FRONTEND_SRC, 'lib', 'components', 'chat', 'ChatControlBar.svelte'))

    def test_think_label_responsive(self):
        self.assertIn('hidden sm:inline">Think</span>', self.content)

    def test_search_label_responsive(self):
        self.assertIn('hidden sm:inline">Search</span>', self.content)

    def test_cache_label_responsive(self):
        self.assertIn('hidden sm:inline">Cache</span>', self.content)

    def test_cascade_label_responsive(self):
        self.assertIn('hidden sm:inline">Cascade</span>', self.content)

    def test_opti_label_responsive(self):
        self.assertIn('hidden sm:inline">Opti</span>', self.content)

    def test_human_label_responsive(self):
        self.assertIn('hidden sm:inline">Human</span>', self.content)

    def test_all_labels_use_hidden_sm(self):
        """Every toggle label uses the same responsive pattern."""
        labels = ['Think', 'Search', 'Cache', 'Cascade', 'Opti', 'Human']
        for label in labels:
            pattern = f'hidden sm:inline">{label}</span>'
            self.assertIn(pattern, self.content,
                          f"Label '{label}' missing responsive class")


class TestChatControlBarUnifiedStyle(unittest.TestCase):
    """Verify all toggles use unified copper glow style."""

    @classmethod
    def setUpClass(cls):
        cls.content = _read(os.path.join(
            FRONTEND_SRC, 'lib', 'components', 'chat', 'ChatControlBar.svelte'))

    def test_active_style_uses_copper(self):
        self.assertIn('var(--oo-msg-user-bd)', self.content)

    def test_active_style_uses_copper_border(self):
        self.assertIn('var(--oo-input-focus)', self.content)

    def test_active_style_uses_copper_glow(self):
        self.assertIn('var(--oo-acc-400)', self.content)

    def test_no_amber_in_toggle_styles(self):
        """No old amber rgba(245, 158, 11) in toggle button styles."""
        self.assertNotIn('rgba(245, 158, 11', self.content)

    def test_unified_active_style_variable(self):
        """All toggles reference the same activeStyle constant."""
        self.assertIn("const activeStyle = '", self.content)
        # Count occurrences of activeStyle usage in template
        count = self.content.count('activeStyle')
        # Should appear at least 8 times: definition + 6 toggles + inactive
        self.assertGreaterEqual(count, 7)

    def test_unified_inactive_style_variable(self):
        self.assertIn("const inactiveStyle = '", self.content)

    def test_disabled_style_for_search(self):
        self.assertIn("const disabledStyle = '", self.content)
        self.assertIn('cursor: not-allowed', self.content)


class TestChatControlBarDdgsCheck(unittest.TestCase):
    """Verify ddgs availability check and Search toggle disable logic."""

    @classmethod
    def setUpClass(cls):
        cls.content = _read(os.path.join(
            FRONTEND_SRC, 'lib', 'components', 'chat', 'ChatControlBar.svelte'))

    def test_ddgs_available_state(self):
        self.assertIn('let ddgsAvailable = true', self.content)

    def test_fetches_search_config(self):
        self.assertIn("fetch('/api/search/config')", self.content)

    def test_reads_ddgs_available_from_response(self):
        self.assertIn('data.ddgs_available', self.content)

    def test_search_button_disabled_when_unavailable(self):
        self.assertIn('disabled={!ddgsAvailable}', self.content)

    def test_search_tooltip_install_hint(self):
        self.assertIn('pip install duckduckgo-search', self.content)

    def test_toggle_search_guards_on_ddgs(self):
        self.assertIn('if (!ddgsAvailable) return', self.content)


class TestChatControlBarModelGrouping(unittest.TestCase):
    """Verify model family grouping and param badges in dropdown."""

    @classmethod
    def setUpClass(cls):
        cls.content = _read(os.path.join(
            FRONTEND_SRC, 'lib', 'components', 'chat', 'ChatControlBar.svelte'))

    def test_optgroup_element(self):
        self.assertIn('<optgroup label={group.family}>', self.content)

    def test_param_badge_display(self):
        self.assertIn('paramBadge', self.content)

    def test_parse_family_function(self):
        self.assertIn('function parseFamily(name: string)', self.content)

    def test_parse_param_badge_function(self):
        self.assertIn('function parseParamBadge(', self.content)

    def test_known_families_mapped(self):
        for family in ['Qwen', 'Llama', 'DeepSeek', 'Gemma', 'Mistral', 'Phi']:
            self.assertIn(f"'{family}'", self.content,
                          f"Family '{family}' not in family map")

    def test_model_groups_reactive(self):
        self.assertIn('let modelGroups:', self.content)

    def test_option_shows_badge(self):
        self.assertIn('model.paramBadge', self.content)


# ===================================================================
# 3. ConversationList
# ===================================================================

class TestConversationListEmptyState(unittest.TestCase):
    """Verify empty state guidance card."""

    @classmethod
    def setUpClass(cls):
        cls.content = _read(os.path.join(
            FRONTEND_SRC, 'lib', 'components', 'sidebar', 'ConversationList.svelte'))

    def test_guidance_card_text(self):
        self.assertIn('Start your first conversation', self.content)

    def test_guidance_card_description(self):
        self.assertIn('Ask a question, write code, or explore your local models', self.content)

    def test_new_conversation_button(self):
        self.assertIn('New conversation', self.content)

    def test_on_new_conversation_prop(self):
        self.assertIn('onNewConversation', self.content)

    def test_new_conversation_click(self):
        self.assertIn('on:click={onNewConversation}', self.content)

    def test_search_no_match_still_shows(self):
        self.assertIn('No matching conversations', self.content)

    def test_chat_icon_in_guidance(self):
        """Guidance card should include a chat bubble icon."""
        # SVG path for chat bubble
        self.assertIn('M21 15a2 2 0 01-2 2H7l-4 4V5', self.content)


class TestConversationListDeleteRedirect(unittest.TestCase):
    """Verify redirect to /chat after deleting conversations."""

    @classmethod
    def setUpClass(cls):
        cls.content = _read(os.path.join(
            FRONTEND_SRC, 'lib', 'components', 'sidebar', 'ConversationList.svelte'))

    def test_redirect_after_delete_active(self):
        self.assertIn("window.location.href = '/chat'", self.content)

    def test_checks_conversations_length(self):
        self.assertIn('$conversations.length === 0', self.content)

    def test_checks_was_active(self):
        self.assertIn('wasActive', self.content)


# ===================================================================
# 4. ConversationItem
# ===================================================================

class TestConversationItemPalette(unittest.TestCase):
    """Verify ConversationItem uses S83 copper palette."""

    @classmethod
    def setUpClass(cls):
        cls.content = _read(os.path.join(
            FRONTEND_SRC, 'lib', 'components', 'sidebar', 'ConversationItem.svelte'))

    def test_active_highlight_uses_copper(self):
        self.assertIn('var(--oo-msg-user-bg)', self.content)

    def test_no_amber_highlight(self):
        self.assertNotIn('rgba(245, 158, 11', self.content)

    def test_rename_input_uses_css_vars(self):
        self.assertIn('var(--oo-input-bg)', self.content)
        self.assertIn('var(--oo-acc-500)', self.content)

    def test_no_hardcoded_surface_classes_in_input(self):
        self.assertNotIn('bg-surface-800', self.content)
        self.assertNotIn('border-surface-600', self.content)

    def test_action_buttons_use_css_vars(self):
        self.assertIn('var(--oo-fg-muted)', self.content)

    def test_delete_confirm_uses_css_vars(self):
        self.assertIn('var(--oo-status-error)', self.content)


# ===================================================================
# 5. ProxySettingsPanel
# ===================================================================

class TestProxySettingsPanelBadge(unittest.TestCase):
    """Verify proxy status badge in ProxySettingsPanel."""

    @classmethod
    def setUpClass(cls):
        cls.content = _read(os.path.join(
            FRONTEND_SRC, 'lib', 'components', 'panels', 'ProxySettingsPanel.svelte'))

    def test_badge_no_proxy_label(self):
        self.assertIn('No proxy', self.content)

    def test_badge_connected_label(self):
        self.assertIn('Connected', self.content)

    def test_badge_disconnected_label(self):
        self.assertIn('Disconnected', self.content)

    def test_badge_not_checked_label(self):
        self.assertIn('Not checked', self.content)

    def test_badge_has_led_indicator(self):
        """Badge includes a colored dot."""
        self.assertIn('width: 6px; height: 6px; border-radius: 50%', self.content)

    def test_badge_green_for_connected(self):
        self.assertIn('var(--oo-status-success)', self.content)

    def test_badge_red_for_disconnected(self):
        self.assertIn('var(--oo-status-error)', self.content)


class TestProxySettingsDdgsWarning(unittest.TestCase):
    """Verify ddgs install warning in ProxySettingsPanel."""

    @classmethod
    def setUpClass(cls):
        cls.content = _read(os.path.join(
            FRONTEND_SRC, 'lib', 'components', 'panels', 'ProxySettingsPanel.svelte'))

    def test_ddgs_available_reactive(self):
        self.assertIn('ddgsAvailable', self.content)

    def test_warning_title(self):
        self.assertIn('Web search unavailable', self.content)

    def test_warning_install_command(self):
        self.assertIn('pip install duckduckgo-search', self.content)

    def test_warning_package_name(self):
        self.assertIn('duckduckgo-search', self.content)

    def test_warning_explains_proxy_still_works(self):
        self.assertIn('still configure proxy settings', self.content)

    def test_warning_only_when_unavailable(self):
        self.assertIn('!ddgsAvailable', self.content)

    def test_warning_amber_style(self):
        """Warning uses amber/warning colors."""
        self.assertIn('var(--oo-warning-bg)', self.content)


# ===================================================================
# 6. client.ts — actionable error messages
# ===================================================================

class TestClientActionableErrors(unittest.TestCase):
    """Verify client.ts has actionable error messages."""

    @classmethod
    def setUpClass(cls):
        cls.content = _read(os.path.join(
            FRONTEND_SRC, 'lib', 'api', 'client.ts'))

    def test_is_network_error_field(self):
        self.assertIn('isNetworkError: boolean', self.content)

    def test_actionable_message_function(self):
        self.assertIn('function actionableMessage(', self.content)

    def test_network_error_message_function(self):
        self.assertIn('function networkErrorMessage(', self.content)

    def test_handle_network_error_function(self):
        self.assertIn('function handleNetworkError(', self.content)

    def test_400_message(self):
        self.assertIn('Check your input and try again', self.content)

    def test_404_message(self):
        self.assertIn('not found', self.content)

    def test_500_message(self):
        self.assertIn('backend encountered an internal error', self.content)

    def test_502_message(self):
        self.assertIn('backend may be restarting', self.content)

    def test_503_message(self):
        self.assertIn('temporarily overloaded or down', self.content)

    def test_429_message(self):
        self.assertIn('Too many requests', self.content)

    def test_network_failed_to_fetch(self):
        self.assertIn('failed to fetch', self.content)

    def test_network_suggests_launch_sh(self):
        self.assertIn('launch.sh', self.content)

    def test_network_asks_is_backend_running(self):
        self.assertIn('Is the server running', self.content)

    def test_timeout_message(self):
        self.assertIn('timed out', self.content)

    def test_abort_error_handling(self):
        self.assertIn('AbortError', self.content)

    def test_api_error_backward_compatible(self):
        """Constructor default for isNetworkError is false."""
        self.assertIn('isNetworkError: boolean = false', self.content)

    def test_handle_response_receives_path(self):
        """handleResponse now takes path for contextual messages."""
        self.assertIn('handleResponse<T>(response, path)', self.content)

    def test_all_methods_use_handle_network_error(self):
        """All HTTP methods use handleNetworkError."""
        for method in ['apiGet', 'apiPost', 'apiPut', 'apiPatch', 'apiDelete']:
            self.assertIn(method, self.content)
        self.assertEqual(self.content.count('handleNetworkError(err, path)'), 5)


# ===================================================================
# 7. Settings Quick tab — version display
# ===================================================================

class TestSettingsVersionDisplay(unittest.TestCase):
    """Verify version display in Settings Quick tab."""

    @classmethod
    def setUpClass(cls):
        cls.content = _read(os.path.join(
            FRONTEND_SRC, 'routes', 'settings', '+page.svelte'))

    def test_app_version_state(self):
        self.assertIn("let appVersion = ''", self.content)

    def test_fetches_health_endpoint(self):
        self.assertIn("fetch('/api/health')", self.content)

    def test_reads_version_from_response(self):
        self.assertIn('data.version', self.content)

    def test_version_badge_in_quick_tab(self):
        self.assertIn('v{appVersion}', self.content)

    def test_version_badge_copper_style(self):
        """Version badge uses copper palette."""
        # Check the specific region
        self.assertIn('var(--oo-msg-user-bg)', self.content)

    def test_version_conditional_display(self):
        self.assertIn('{#if appVersion}', self.content)

    def test_quick_settings_header(self):
        self.assertIn('Quick Settings', self.content)


# ===================================================================
# 8. Sidebar — dynamic version
# ===================================================================

class TestSidebarDynamicVersion(unittest.TestCase):
    """Verify sidebar fetches version dynamically."""

    @classmethod
    def setUpClass(cls):
        cls.content = _read(os.path.join(
            FRONTEND_SRC, 'lib', 'components', 'layout', 'Sidebar.svelte'))

    def test_no_hardcoded_version(self):
        self.assertNotIn('v1.5.6', self.content)
        self.assertNotIn('v1.8.8', self.content)

    def test_app_version_state(self):
        self.assertIn("let appVersion = ''", self.content)

    def test_fetches_health_endpoint(self):
        self.assertIn("fetch('/api/health')", self.content)

    def test_version_displayed_conditionally(self):
        self.assertIn('{#if appVersion}', self.content)

    def test_version_in_header(self):
        self.assertIn('v{appVersion}', self.content)

    def test_imports_on_mount(self):
        self.assertIn("import { onMount } from 'svelte'", self.content)

    def test_on_new_conversation_passed(self):
        self.assertIn('onNewConversation={onCreate}', self.content)


# ===================================================================
# 9. Cross-cutting checks
# ===================================================================

class TestNoFrenchInNewCode(unittest.TestCase):
    """Verify new/modified files have English comments and UI text."""

    def _check_no_french(self, filepath, content):
        """Check that common French words are not in code comments or strings."""
        french_markers = [
            'Selecteur de ', 'Recherche', 'Liste des',
            'Barre de controle', 'Permet de', 'reagit aux',
        ]
        for marker in french_markers:
            self.assertNotIn(marker, content,
                             f"French text '{marker}' found in {filepath}")

    def test_chatcontrolbar_english(self):
        content = _read(os.path.join(
            FRONTEND_SRC, 'lib', 'components', 'chat', 'ChatControlBar.svelte'))
        self._check_no_french('ChatControlBar.svelte', content)

    def test_conversationlist_english(self):
        content = _read(os.path.join(
            FRONTEND_SRC, 'lib', 'components', 'sidebar', 'ConversationList.svelte'))
        self._check_no_french('ConversationList.svelte', content)


class TestNoEmojisInCode(unittest.TestCase):
    """Verify no emojis in modified files."""

    def _check_no_emoji(self, filepath):
        content = _read(filepath)
        emoji_pattern = re.compile(
            r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
            r'\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF'
            r'\U00002702-\U000027B0\U0001F900-\U0001F9FF]'
        )
        match = emoji_pattern.search(content)
        self.assertIsNone(match, f"Emoji found in {filepath}")

    def test_chatcontrolbar_no_emoji(self):
        self._check_no_emoji(os.path.join(
            FRONTEND_SRC, 'lib', 'components', 'chat', 'ChatControlBar.svelte'))

    def test_client_ts_no_emoji(self):
        self._check_no_emoji(os.path.join(
            FRONTEND_SRC, 'lib', 'api', 'client.ts'))

    def test_conversationlist_no_emoji(self):
        self._check_no_emoji(os.path.join(
            FRONTEND_SRC, 'lib', 'components', 'sidebar', 'ConversationList.svelte'))


class TestConsistentCopperPalette(unittest.TestCase):
    """Verify copper palette consistency across S87 changes."""

    def test_chatcontrolbar_no_amber_toggles(self):
        content = _read(os.path.join(
            FRONTEND_SRC, 'lib', 'components', 'chat', 'ChatControlBar.svelte'))
        # No amber (245, 158, 11) in toggle styles
        self.assertNotIn('rgba(245, 158, 11', content)

    def test_conversationitem_no_amber(self):
        content = _read(os.path.join(
            FRONTEND_SRC, 'lib', 'components', 'sidebar', 'ConversationItem.svelte'))
        self.assertNotIn('rgba(245, 158, 11', content)


class TestAppPyParsesClean(unittest.TestCase):
    """Verify app.py is valid Python after edits."""

    def test_app_py_parses(self):
        content = _read(os.path.join(API_DIR, 'app.py'))
        try:
            ast.parse(content)
        except SyntaxError as e:
            self.fail(f"app.py has syntax error: {e}")

    def test_setup_py_parses(self):
        content = _read(os.path.join(PROJECT_ROOT, 'setup.py'))
        try:
            ast.parse(content)
        except SyntaxError as e:
            self.fail(f"setup.py has syntax error: {e}")


class TestAllTooltipsPresent(unittest.TestCase):
    """Verify each toggle has a descriptive tooltip."""

    @classmethod
    def setUpClass(cls):
        cls.content = _read(os.path.join(
            FRONTEND_SRC, 'lib', 'components', 'chat', 'ChatControlBar.svelte'))

    def test_think_tooltip(self):
        self.assertIn('chain-of-thought', self.content)

    def test_search_tooltip(self):
        self.assertIn('DuckDuckGo', self.content)

    def test_cache_tooltip(self):
        self.assertIn('semantic cache', self.content.lower())

    def test_cascade_tooltip(self):
        self.assertIn('multi-tier model routing', self.content)

    def test_opti_tooltip(self):
        self.assertIn('optimize prompts', self.content)

    def test_humanize_tooltip(self):
        self.assertIn('make output more natural', self.content)


if __name__ == '__main__':
    unittest.main()
