"""
S170 -- Polish / Accessibility / Themes test suite (closes Bloc 1).

Validates the S170 goals (spec 12.5, Goal 1-4):

1. Accessibility sweep: tokenized focus ring, skip link + #main-content +
   aria-live regions, icon-button labels (EventTimeline close), A8 inputs on
   BranchExplorer.
2. Empty + error states: shared EmptyState / InlineError primitives wired into
   the canonical list/panel surfaces, errorHandler.ts gains a `retriable` flag
   that drives the toast Retry affordance.
3. Theme switcher + micro-polish + shortcut help: five-theme token parity,
   tokenized transitions, ThemeSwitcher menu animation, KeyboardShortcuts label
   refresh (spec 8.8).
4. Carry-over chat + benchmark refactors onto ds primitives: SearchResults,
   ReasoningDisplay, ToolCallDisplay, CodingAgentInline, CodingAgentProgress
   wrapped in Card (a new padding="none" option); LiveMetricsOverlay anchored to
   the chat surface instead of floating over the viewport; BenchmarkRunner /
   BenchmarkHistory action buttons on the Button primitive (OnionLoader kept).

These are file-content assertions (the repo convention for frontend checks).
The TestThemeTransitionSupersede class deliberately re-asserts the *intent* of
test_palette_v4e_s93::TestThemeTransition::test_transition_properties
(deselected because the literal "300ms" in app.css was tokenized to
var(--oo-motion-slow)) against the new token-sourced rule.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FE = ROOT / "frontend" / "src"
CHAT = FE / "lib" / "components" / "chat"
DS = FE / "lib" / "ds"
STYLES = FE / "styles"
PANELS = FE / "lib" / "components" / "panels"
BENCH = PANELS / "benchmark"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


THEMES = [
    "theme-anthracite.css",
    "theme-parchment.css",
    "theme-slate.css",
    "theme-linen.css",
    "theme-high-contrast.css",
]

# A small, robust subset of the --oo-* token surface every theme must define.
CORE_TOKENS = [
    "--oo-bg-base",
    "--oo-acc-500",
    "--oo-bd-default",
    "--oo-fg-muted",
    "--oo-success",
]


# ===========================================================================
# Goal 1 -- Accessibility sweep
# ===========================================================================


class TestFocusAndLandmarks:
    """Tokenized focus ring + skip link + landmark + live regions."""

    def test_focus_visible_uses_ring_tokens(self):
        src = _read(FE / "app.css")
        assert ":focus-visible" in src
        assert "--oo-focus-ring-width" in src
        assert "--oo-focus-ring-offset" in src

    def test_skip_link_token_color(self):
        # The skip link foreground is tokenized (no raw 'white').
        src = _read(FE / "app.css")
        assert ".skip-to-content" in src
        assert "--oo-fg-on-accent" in src

    def test_appshell_has_skip_link(self):
        assert "skip-to-content" in _read(FE / "lib" / "components" / "layout" / "AppShell.svelte")

    def test_appshell_has_main_content_landmark(self):
        assert 'id="main-content"' in _read(FE / "lib" / "components" / "layout" / "AppShell.svelte")

    def test_appshell_has_live_region(self):
        assert "aria-live" in _read(FE / "lib" / "components" / "layout" / "AppShell.svelte")

    def test_event_timeline_close_button_labelled(self):
        src = _read(PANELS / "EventTimeline.svelte")
        assert 'aria-label="Close event detail"' in src

    def test_branch_explorer_inputs_labelled(self):
        src = _read(CHAT / "BranchExplorer.svelte")
        assert 'aria-label="Branch name"' in src
        assert 'aria-label="Rename branch"' in src


# ===========================================================================
# Goal 2 -- Empty + error states
# ===========================================================================


class TestStatePrimitives:
    """EmptyState / InlineError primitives exist and are exported."""

    def test_empty_state_file(self):
        assert (DS / "EmptyState.svelte").is_file()

    def test_inline_error_file(self):
        assert (DS / "InlineError.svelte").is_file()

    def test_inline_error_is_alert(self):
        assert 'role="alert"' in _read(DS / "InlineError.svelte")

    def test_state_primitives_exported(self):
        idx = _read(DS / "index.ts")
        assert "EmptyState" in idx
        assert "InlineError" in idx


class TestStateAdoption:
    """Canonical list/panel surfaces use the shared state primitives."""

    def test_project_list_states(self):
        src = _read(PANELS / "ProjectList.svelte")
        assert "EmptyState" in src
        assert "InlineError" in src

    def test_benchmark_leaderboard_states(self):
        src = _read(BENCH / "BenchmarkLeaderboard.svelte")
        assert "EmptyState" in src
        assert "InlineError" in src

    def test_benchmark_history_section_empty(self):
        assert "EmptyState" in _read(BENCH / "BenchmarkHistorySection.svelte")

    def test_benchmark_trends_empty(self):
        assert "EmptyState" in _read(BENCH / "BenchmarkTrends.svelte")


class TestErrorHandlerRetriable:
    """errorHandler.ts surfaces a retriable flag for the toast Retry path."""

    def test_retriable_flag_present(self):
        assert "retriable" in _read(FE / "lib" / "api" / "errorHandler.ts")


# ===========================================================================
# Goal 3 -- Theme parity, transitions, switcher, shortcut help
# ===========================================================================


class TestThemeParity:
    """All five palettes define the same token surface + selector."""

    def test_all_theme_files_present(self):
        for name in THEMES:
            assert (STYLES / name).is_file(), f"missing {name}"

    def test_theme_selector_present(self):
        for name in THEMES:
            assert "data-oo-theme" in _read(STYLES / name), f"{name} lacks data-oo-theme"

    def test_theme_core_tokens_present(self):
        for name in THEMES:
            src = _read(STYLES / name)
            for tok in CORE_TOKENS:
                assert tok in src, f"{name} missing {tok}"

    def test_theme_token_count_parity(self):
        defs = re.compile(r"^\s*--oo-[a-z0-9-]+\s*:", re.MULTILINE)
        counts = {name: len(defs.findall(_read(STYLES / name))) for name in THEMES}
        assert len(set(counts.values())) == 1, f"token-count drift: {counts}"
        assert min(counts.values()) >= 30, counts


class TestTransitionsTokenized:
    """Reusable transition timings come from motion tokens."""

    def test_transitions_use_motion_tokens(self):
        src = _read(STYLES / "transitions.css")
        assert "var(--oo-motion-normal)" in src
        assert "var(--oo-ease-default)" in src

    def test_transitions_no_stray_literal_easing(self):
        # The ad-hoc literal timings (200ms/100ms/150ms/250ms) were tokenized.
        src = _read(STYLES / "transitions.css")
        for lit in ("200ms", "150ms", "250ms"):
            assert lit not in src, f"stray literal {lit} in transitions.css"


class TestThemeSwitcherPolish:
    """ThemeSwitcher trigger/menu use tokens + a reduced-motion-safe animation."""

    def test_switcher_uses_motion_tokens(self):
        assert "--oo-motion" in _read(FE / "lib" / "components" / "ui" / "ThemeSwitcher.svelte")

    def test_switcher_menu_animation(self):
        assert "oo-ts-menu-in" in _read(FE / "lib" / "components" / "ui" / "ThemeSwitcher.svelte")

    def test_switcher_respects_reduced_motion(self):
        assert "prefers-reduced-motion" in _read(FE / "lib" / "components" / "ui" / "ThemeSwitcher.svelte")


class TestKeyboardShortcutsHelp:
    """Shortcut help overlay reflects the spec 8.8 label refresh."""

    def test_context_search_relabelled(self):
        assert "Focus context-list search" in _read(FE / "lib" / "components" / "ui" / "KeyboardShortcuts.svelte")


class TestAxeSweep:
    """The axe-core e2e sweep covers the theme x density contrast matrix."""

    def test_axe_spec_present(self):
        assert (ROOT / "tests" / "e2e" / "a11y.spec.ts").is_file()

    def test_axe_uses_playwright_integration(self):
        assert "@axe-core/playwright" in _read(ROOT / "tests" / "e2e" / "a11y.spec.ts")

    def test_axe_checks_contrast_and_wcag(self):
        src = _read(ROOT / "tests" / "e2e" / "a11y.spec.ts")
        assert "color-contrast" in src
        assert "wcag2aa" in src

    def test_axe_covers_five_palettes(self):
        src = _read(ROOT / "tests" / "e2e" / "a11y.spec.ts")
        for pal in ("anthracite", "parchment", "slate", "linen", "high-contrast"):
            assert pal in src, f"axe sweep omits {pal}"

    def test_axe_dependency_declared(self):
        assert "@axe-core/playwright" in _read(FE.parent / "package.json")


# ===========================================================================
# Goal 4 -- Carry-over chat refactors onto primitives
# ===========================================================================


class TestCardPrimitiveNone:
    """Card primitive gains an edge-to-edge padding option."""

    def test_card_padding_none_option(self):
        src = _read(DS / "Card.svelte")
        assert "'none'" in src
        assert "data-padding='none'" in src

    def test_card_retains_variant_and_padding_props(self):
        src = _read(DS / "Card.svelte")
        assert "variant" in src
        assert "padding" in src


class TestChatCardAdoption:
    """Heavier chat sub-components are wrapped in the Card primitive."""

    CARD_COMPONENTS = [
        "SearchResults",
        "ReasoningDisplay",
        "ToolCallDisplay",
        "CodingAgentInline",
        "CodingAgentProgress",
    ]

    def test_components_import_card(self):
        for name in self.CARD_COMPONENTS:
            src = _read(CHAT / f"{name}.svelte")
            assert "import { Card }" in src, f"{name} does not import Card"

    def test_components_use_card(self):
        for name in self.CARD_COMPONENTS:
            src = _read(CHAT / f"{name}.svelte")
            assert "<Card" in src, f"{name} does not use <Card>"

    def test_tool_call_display_preserves_plugin_badge(self):
        src = _read(CHAT / "ToolCallDisplay.svelte")
        assert "PluginPermissionBadge" in src
        assert "pluginPerms" in src

    def test_coding_agent_inline_preserves_contract(self):
        src = _read(CHAT / "CodingAgentInline.svelte")
        for sym in ("SandboxIsolationBadge", "planSteps", "Tests passed", "Tests failed", "hasVision"):
            assert sym in src, f"CodingAgentInline lost {sym}"

    def test_coding_agent_progress_preserves_contract(self):
        src = _read(CHAT / "CodingAgentProgress.svelte")
        for sym in ("planSteps", "implementedFiles", "animate-pulse"):
            assert sym in src, f"CodingAgentProgress lost {sym}"

    def test_search_results_french_stripped(self):
        src = _read(CHAT / "SearchResults.svelte")
        assert "recherche" not in src
        assert "Resultats" not in src

    def test_reasoning_display_french_stripped(self):
        src = _read(CHAT / "ReasoningDisplay.svelte")
        assert "etapes" not in src
        assert "Indicateur" not in src

    def test_context_panel_french_stripped(self):
        src = _read(CHAT / "ContextPanel.svelte")
        for word in ("rafraich", "donnees", "panneau", "modele change"):
            assert word.lower() not in src.lower(), f"ContextPanel still mentions '{word}'"


class TestLiveMetricsAnchored:
    """LiveMetricsOverlay is anchored to the chat surface, not the viewport."""

    def test_overlay_is_absolute(self):
        src = _read(CHAT / "LiveMetricsOverlay.svelte")
        assert "position: absolute" in src

    def test_overlay_not_fixed(self):
        src = _read(CHAT / "LiveMetricsOverlay.svelte")
        assert "position: fixed" not in src

    def test_overlay_uses_z_token(self):
        src = _read(CHAT / "LiveMetricsOverlay.svelte")
        assert "var(--oo-z-overlay)" in src

    def test_main_content_is_positioning_context(self):
        src = _read(FE / "lib" / "components" / "layout" / "AppShell.svelte")
        # #main-content carries `relative` so the anchored overlay docks to it.
        assert "overflow-hidden relative" in src


class TestBenchmarkPrimitiveAdoption:
    """Benchmark action buttons adopt the Button primitive (OnionLoader kept)."""

    def test_runner_imports_button(self):
        assert "import { Button }" in _read(PANELS / "BenchmarkRunner.svelte")

    def test_runner_uses_button(self):
        assert "<Button" in _read(PANELS / "BenchmarkRunner.svelte")

    def test_runner_keeps_onion_loader(self):
        src = _read(PANELS / "BenchmarkRunner.svelte")
        assert "import OnionLoader" in src
        assert "<OnionLoader" in src

    def test_history_imports_button(self):
        assert "import { Button }" in _read(PANELS / "BenchmarkHistory.svelte")

    def test_history_uses_button(self):
        assert "<Button" in _read(PANELS / "BenchmarkHistory.svelte")


# ===========================================================================
# Supersede -- re-assert the intent of a deselected pinned-literal test
# ===========================================================================


class TestThemeTransitionSupersede:
    """Re-assert test_palette_v4e_s93::TestThemeTransition::test_transition_properties.

    The original pinned the literal "300ms" in app.css. S170 tokenizes the
    cross-fade durations to var(--oo-motion-slow); the effective 300ms is now
    sourced from tokens.css, so the behavior is unchanged. The literal-string
    assertion is therefore superseded (deselected) and re-expressed here.
    """

    def test_theme_transitioning_still_animates_background(self):
        assert "background-color" in _read(FE / "app.css")

    def test_theme_transitioning_uses_motion_slow_token(self):
        src = _read(FE / "app.css")
        assert "theme-transitioning" in src
        assert "var(--oo-motion-slow)" in src

    def test_app_css_no_longer_pins_literal_duration(self):
        assert "300ms" not in _read(FE / "app.css")

    def test_motion_slow_token_preserves_300ms(self):
        # Effective duration is unchanged: the token resolves to 300ms.
        assert "--oo-motion-slow: 300ms" in _read(STYLES / "tokens.css")


# ===========================================================================
# Supersede -- version bump 3.3.0 -> 3.4.0-rc (closes Bloc 1)
# ===========================================================================


class TestVersionBumpSupersede:
    """Re-assert the version surface after the 3.4.0 release.

    The S164 release suite pinned 3.3.0 in __version__.py / pyproject.toml /
    CHANGELOG ordering and required a bare X.Y.Z (no pre-release suffix). S170
    closed Bloc 1 with the release candidate 3.4.0-rc; S171 graduates the RC to
    the final 3.4.0 (RC -> final). The following S164 tests stay deselected and
    are re-expressed here against the final version:
      - TestVersion::test_version_string
      - TestVersion::test_version_file_contains_330
      - TestStaleVersions::test_version_py_is_330
      - TestStaleVersions::test_changelog_first_entry_is_v330
      - TestPyprojectToml::test_version_matches
    (test_version_pep440_compliant is no longer deselected: the bare 3.4.0
    final satisfies its ^X.Y.Z$ check; the deselect was dropped in S171.)
    The historical v3.3.0 and v3.4.0-rc CHANGELOG entries are retained.
    """

    @staticmethod
    def _version_from_file() -> str:
        content = _read(ROOT / "opti_oignon" / "__version__.py")
        m = re.search(r'__version__\s*=\s*"([^"]+)"', content)
        assert m, "no __version__ assignment found"
        return m.group(1)

    def test_version_string_is_final(self):
        assert self._version_from_file() == "3.4.0"

    def test_version_file_contains_final(self):
        assert '"3.4.0"' in _read(ROOT / "opti_oignon" / "__version__.py")

    def test_version_is_final_bare_form(self):
        # RC graduated to final: the version is a bare X.Y.Z (no -rc suffix).
        assert re.match(r"^\d+\.\d+\.\d+$", self._version_from_file())

    def test_pyproject_version_matches(self):
        assert 'version = "3.4.0"' in _read(ROOT / "pyproject.toml")

    def test_pyproject_consistent_with_version_file(self):
        v = self._version_from_file()
        assert f'version = "{v}"' in _read(ROOT / "pyproject.toml")

    def test_changelog_has_rc_entry(self):
        assert "## v3.4.0-rc -- 2026-06-01 (S170)" in _read(ROOT / "CHANGELOG.md")

    def test_changelog_first_entry_is_rc(self):
        entries = re.findall(r"## v(\d+\.\d+\.\d+)", _read(ROOT / "CHANGELOG.md"))
        assert entries and entries[0] == "3.4.0"

    def test_changelog_retains_v330_entry(self):
        assert "## v3.3.0" in _read(ROOT / "CHANGELOG.md")
