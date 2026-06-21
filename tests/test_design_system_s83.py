"""
Tests for S83 — Design System Refresh: Warm Neutral Palette.

Validates:
- theme.css contains all required CSS custom properties (dark + light)
- No hardcoded hex colors remain in Svelte components
- No legacy var(--color-*) or var(--oo-acc2-*) references
- No var(--oo-*, #fallback) patterns outside theme.css
- Theme toggle store has localStorage persistence
- app.html has inline FOUC prevention script
- tailwind.config.js uses new warm neutral palette
- All comments in modified files are in English
"""

import glob
import os
import re
import unittest

FRONTEND_SRC = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'src')
THEME_CSS = os.path.join(FRONTEND_SRC, 'styles', 'theme.css')
APP_CSS = os.path.join(FRONTEND_SRC, 'app.css')
APP_HTML = os.path.join(FRONTEND_SRC, 'app.html')
UI_STORE = os.path.join(FRONTEND_SRC, 'lib', 'stores', 'ui.ts')
TAILWIND_CFG = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'tailwind.config.js')
LAYOUT_SVELTE = os.path.join(FRONTEND_SRC, 'routes', '+layout.svelte')
APP_PY = os.path.join(os.path.dirname(__file__), '..', 'opti_oignon', 'api', 'app.py')


def _read(path):
    with open(os.path.abspath(path), encoding='utf-8') as f:
        return f.read()


def _svelte_files():
    pattern = os.path.join(os.path.abspath(FRONTEND_SRC), '**', '*.svelte')
    return glob.glob(pattern, recursive=True)


class TestThemeCSSProperties(unittest.TestCase):
    """Verify theme.css contains all required CSS custom properties."""

    @classmethod
    def setUpClass(cls):
        cls.css = _read(THEME_CSS)

    # -- Dark mode background tokens --
    def test_dark_bg_base(self):
        self.assertIn('--oo-bg-base:', self.css)

    def test_dark_bg_surface(self):
        self.assertIn('--oo-bg-surface:', self.css)

    def test_dark_bg_elevated(self):
        self.assertIn('--oo-bg-elevated:', self.css)

    def test_dark_bg_overlay(self):
        self.assertIn('--oo-bg-overlay:', self.css)

    def test_dark_bg_subtle(self):
        self.assertIn('--oo-bg-subtle:', self.css)

    # -- Text tokens --
    def test_fg_primary(self):
        self.assertIn('--oo-fg-primary:', self.css)

    def test_fg_secondary(self):
        self.assertIn('--oo-fg-secondary:', self.css)

    def test_fg_tertiary(self):
        self.assertIn('--oo-fg-tertiary:', self.css)

    def test_fg_muted(self):
        self.assertIn('--oo-fg-muted:', self.css)

    # -- Border tokens --
    def test_bd_default(self):
        self.assertIn('--oo-bd-default:', self.css)

    def test_bd_subtle(self):
        self.assertIn('--oo-bd-subtle:', self.css)

    def test_bd_strong(self):
        self.assertIn('--oo-bd-strong:', self.css)

    # -- Accent scale --
    def test_acc_400(self):
        self.assertIn('--oo-acc-400:', self.css)

    def test_acc_500(self):
        self.assertIn('--oo-acc-500:', self.css)

    def test_acc_600(self):
        self.assertIn('--oo-acc-600:', self.css)

    # -- Semantic tokens --
    def test_success(self):
        self.assertIn('--oo-success:', self.css)

    def test_error(self):
        self.assertIn('--oo-error:', self.css)

    def test_warning(self):
        self.assertIn('--oo-warning:', self.css)

    def test_info(self):
        self.assertIn('--oo-info:', self.css)

    def test_success_bg(self):
        self.assertIn('--oo-success-bg:', self.css)

    def test_error_bg(self):
        self.assertIn('--oo-error-bg:', self.css)

    def test_warning_bg(self):
        self.assertIn('--oo-warning-bg:', self.css)

    # -- Component-specific tokens --
    def test_sidebar_bg(self):
        self.assertIn('--oo-sidebar-bg:', self.css)

    def test_panel_bg(self):
        self.assertIn('--oo-panel-bg:', self.css)

    def test_input_bg(self):
        self.assertIn('--oo-input-bg:', self.css)

    def test_input_focus(self):
        self.assertIn('--oo-input-focus:', self.css)

    # -- Message tokens --
    def test_msg_user_bg(self):
        self.assertIn('--oo-msg-user-bg:', self.css)

    def test_msg_bot_bg(self):
        self.assertIn('--oo-msg-bot-bg:', self.css)

    # -- Button tokens --
    def test_btn_primary_bg(self):
        self.assertIn('--oo-btn-primary-bg:', self.css)

    def test_btn_secondary_bg(self):
        self.assertIn('--oo-btn-secondary-bg:', self.css)

    # -- Pipeline badges --
    def test_pipe_direct(self):
        self.assertIn('--oo-pipe-direct:', self.css)

    def test_pipe_tools(self):
        self.assertIn('--oo-pipe-tools:', self.css)

    def test_pipe_think(self):
        self.assertIn('--oo-pipe-think:', self.css)

    def test_pipe_search(self):
        self.assertIn('--oo-pipe-search:', self.css)

    def test_pipe_code(self):
        self.assertIn('--oo-pipe-code:', self.css)

    def test_pipe_reason(self):
        self.assertIn('--oo-pipe-reason:', self.css)

    def test_pipe_consensus(self):
        self.assertIn('--oo-pipe-consensus:', self.css)

    def test_pipe_correct(self):
        self.assertIn('--oo-pipe-correct:', self.css)

    # -- Scrollbar --
    def test_scrollbar_thumb(self):
        self.assertIn('--oo-scrollbar-thumb:', self.css)

    # -- Radius --
    def test_radius_md(self):
        self.assertIn('--oo-radius-md:', self.css)

    def test_radius_lg(self):
        self.assertIn('--oo-radius-lg:', self.css)


class TestThemePaletteValues(unittest.TestCase):
    """Verify the new warm neutral palette is applied."""

    @classmethod
    def setUpClass(cls):
        cls.css = _read(THEME_CSS)

    def test_dark_bg_is_warm_anthracite(self):
        self.assertIn('#1E1C1A', self.css)

    def test_light_bg_is_warm_stone(self):
        self.assertIn('#F5F0EB', self.css)

    def test_accent_is_copper(self):
        self.assertIn('#B07D56', self.css)

    def test_success_is_sage(self):
        self.assertIn('#7FA882', self.css)

    def test_error_is_dusty_rose(self):
        self.assertIn('#CF7070', self.css)

    def test_no_teal_accent(self):
        self.assertNotIn('--oo-acc2-', self.css)

    def test_no_neon_green(self):
        self.assertNotIn('#4ade80', self.css)
        self.assertNotIn('#22c55e', self.css)

    def test_no_neon_amber(self):
        self.assertNotIn('#f59e0b', self.css)
        self.assertNotIn('#d97706', self.css)

    def test_light_mode_section_exists(self):
        self.assertIn('html:not(.dark)', self.css)

    def test_dark_mode_section_exists(self):
        self.assertIn('html.dark', self.css)


class TestNoHardcodedHexInSvelte(unittest.TestCase):
    """Audit: no hardcoded hex colors in Svelte component styles/templates."""

    # Patterns that are NOT color hex codes (Svelte template syntax)
    EXCLUDE_PATTERNS = [
        r'{#each', r'{#if', r'&#\d', r'msg\.id', r'role}-',
        r'timestamp', r'`\$', r'\.id\b',
    ]

    # The only allowed hex is the OnionLoader default prop
    ALLOWED = ["color: string = '#B07D56'"]

    def test_no_hardcoded_hex_in_svelte(self):
        hex_re = re.compile(r'#[0-9a-fA-F]{6}\b')
        violations = []
        for fpath in _svelte_files():
            content = _read(fpath)
            for i, line in enumerate(content.splitlines(), 1):
                if not hex_re.search(line):
                    continue
                if 'var(--oo-' in line:
                    continue
                if any(allowed in line for allowed in self.ALLOWED):
                    continue
                if any(re.search(pat, line) for pat in self.EXCLUDE_PATTERNS):
                    continue
                fname = os.path.basename(fpath)
                violations.append(f'{fname}:{i}: {line.strip()[:100]}')
        self.assertEqual(violations, [],
                         f'Found {len(violations)} hardcoded hex:\n' +
                         '\n'.join(violations[:20]))


class TestNoLegacyColorVariables(unittest.TestCase):
    """No legacy var(--color-*) or var(--oo-acc2-*) references in Svelte."""

    def test_no_legacy_color_vars(self):
        violations = []
        for fpath in _svelte_files():
            content = _read(fpath)
            for i, line in enumerate(content.splitlines(), 1):
                if 'var(--color-' in line:
                    fname = os.path.basename(fpath)
                    violations.append(f'{fname}:{i}: {line.strip()[:100]}')
        self.assertEqual(violations, [],
                         'Found legacy var(--color-*):\n' +
                         '\n'.join(violations[:20]))

    def test_no_acc2_references(self):
        violations = []
        for fpath in _svelte_files():
            content = _read(fpath)
            if '--oo-acc2-' in content:
                violations.append(os.path.basename(fpath))
        self.assertEqual(violations, [],
                         f'Found --oo-acc2-* in: {violations}')


class TestNoFallbackPatterns(unittest.TestCase):
    """No var(--oo-*, #hex_fallback) outside theme.css."""

    def test_no_oo_vars_with_hex_fallbacks(self):
        pattern = re.compile(r'var\(--oo-[^)]*#[0-9a-fA-F]')
        violations = []
        for fpath in _svelte_files():
            content = _read(fpath)
            for i, line in enumerate(content.splitlines(), 1):
                if pattern.search(line):
                    fname = os.path.basename(fpath)
                    violations.append(f'{fname}:{i}: {line.strip()[:100]}')
        # Also check app.css
        content = _read(APP_CSS)
        for i, line in enumerate(content.splitlines(), 1):
            if pattern.search(line):
                violations.append(f'app.css:{i}: {line.strip()[:100]}')
        self.assertEqual(violations, [],
                         'Found var(--oo-*, #fallback):\n' +
                         '\n'.join(violations[:20]))


class TestNoRgbOldPalette(unittest.TestCase):
    """No rgb()/rgba() with old palette values in Svelte."""

    OLD_RGB_PATTERNS = [
        r'rgb\(34,\s*197,\s*94',
        r'rgb\(239,\s*68,\s*68',
        r'rgb\(245,\s*158,\s*11',
        r'rgb\(96,\s*165,\s*250',
        r'rgb\(248,\s*113,\s*113',
        r'rgb\(252,\s*165,\s*165',
        r'rgb\(134,\s*239,\s*172',
        r'rgb\(252,\s*211,\s*77',
        r'rgb\(74,\s*222,\s*128',
        r'rgb\(147,\s*197,\s*253',
    ]

    def test_no_old_palette_rgb(self):
        violations = []
        for fpath in _svelte_files():
            content = _read(fpath)
            for pat in self.OLD_RGB_PATTERNS:
                if re.search(pat, content):
                    violations.append(f'{os.path.basename(fpath)}: {pat}')
        self.assertEqual(violations, [],
                         'Old palette rgb() found:\n' + '\n'.join(violations[:20]))


class TestThemeTogglePersistence(unittest.TestCase):
    """Theme toggle uses localStorage with key oo-theme."""

    @classmethod
    def setUpClass(cls):
        cls.ui_ts = _read(UI_STORE)
        cls.app_html = _read(APP_HTML)
        cls.layout = _read(LAYOUT_SVELTE)

    def test_ui_store_reads_localStorage(self):
        self.assertIn("localStorage.getItem('oo-theme')", self.ui_ts)

    def test_ui_store_writes_localStorage(self):
        self.assertIn("localStorage.setItem('oo-theme'", self.ui_ts)

    def test_ui_store_exports_initTheme(self):
        self.assertIn('export function initTheme', self.ui_ts)

    def test_ui_store_exports_toggleTheme(self):
        self.assertIn('export function toggleTheme', self.ui_ts)

    def test_ui_store_prefers_color_scheme_fallback(self):
        self.assertIn('prefers-color-scheme', self.ui_ts)

    def test_app_html_no_hardcoded_dark_class(self):
        self.assertNotIn('class="dark"', self.app_html)

    def test_app_html_has_fouc_prevention_script(self):
        self.assertIn("localStorage.getItem('oo-theme')", self.app_html)
        self.assertIn("classList.add('dark')", self.app_html)

    def test_layout_calls_initTheme(self):
        self.assertIn('initTheme()', self.layout)

    def test_layout_no_old_localStorage_key(self):
        self.assertNotIn('opti-oignon-theme', self.layout)


class TestTailwindConfig(unittest.TestCase):
    """Verify tailwind.config.js uses the new warm neutral palette."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = _read(TAILWIND_CFG)

    def test_surface_950_warm(self):
        self.assertIn('#1A1816', self.cfg)

    def test_surface_50_warm(self):
        self.assertIn('#F5F0EB', self.cfg)

    def test_accent_500_copper(self):
        self.assertIn('#B07D56', self.cfg)

    def test_no_old_amber_accent(self):
        self.assertNotIn('#f59e0b', self.cfg)
        self.assertNotIn('#d97706', self.cfg)

    def test_no_old_blue_grey_surface(self):
        self.assertNotIn('#2a2a31', self.cfg)
        self.assertNotIn('#212127', self.cfg)


class TestVersionBump(unittest.TestCase):
    """Verify version bumped to 1.8.9."""

    def test_app_py_version(self):
        content = _read(APP_PY)
        self.assertIn('1.8.9', content)
        self.assertNotIn('1.8.4', content)


class TestAppCSSNoOldPalette(unittest.TestCase):
    """Verify app.css uses the new palette in light mode overrides."""

    @classmethod
    def setUpClass(cls):
        cls.css = _read(APP_CSS)

    def test_no_old_text_colors(self):
        self.assertNotIn('#1c1917', self.css)
        self.assertNotIn('#44403c', self.css)
        self.assertNotIn('#78716c', self.css)

    def test_no_old_amber_accent(self):
        self.assertNotIn('#d97706', self.css)
        self.assertNotIn('#b45309', self.css)

    def test_uses_new_warm_charcoal(self):
        self.assertIn('var(--oo-fg-primary)', self.css)

    def test_uses_new_copper_accent(self):
        self.assertIn('var(--oo-acc-400)', self.css)

    def test_comments_in_english(self):
        self.assertNotIn('Permet', self.css)
        self.assertNotIn('depuis', self.css)
        self.assertNotIn('generique', self.css)
        self.assertNotIn('Specifiques', self.css)


class TestEnglishComments(unittest.TestCase):
    """Verify modified key files have no French comments."""

    def test_theme_css_english(self):
        css = _read(THEME_CSS)
        self.assertNotIn('Fond principal', css)
        self.assertNotIn('Texte', css)
        self.assertNotIn('Bordures', css)
        self.assertNotIn('Boutons', css)

    def test_ui_store_english(self):
        ts = _read(UI_STORE)
        self.assertNotIn('sombre', ts)
        self.assertNotIn('Sauvegarder', ts)
        self.assertNotIn('initialise', ts)


if __name__ == '__main__':
    unittest.main()
