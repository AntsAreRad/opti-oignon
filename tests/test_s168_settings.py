"""
S168 -- Settings Consolidation test suite.

Validates the consolidated /settings hub (spec 5.5 / 9.2 / 12.3): the nine
sections, search, deep-linking, migration completeness, the now-reachable
formerly-orphan/buried panels, the Appearance build-out (palette / density /
typography / motion + reintegrated ThemeCustomizer & ShortcutSettings), the
Speculative merge, and login/register theme parity on ds primitives.

These tests are file-content assertions (the repo convention for frontend
checks). Several deliberately re-assert the *intent* of assertions that S168
deselected from earlier suites (s84 system presets, s109 auth toggle, s87
version badge, routes_speculative panel) against the new structure.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FE = ROOT / "frontend" / "src"

SETTINGS_PAGE = FE / "routes" / "settings" / "+page.svelte"
SETTINGS_GROUP = FE / "lib" / "components" / "settings" / "SettingsGroup.svelte"
APPEARANCE = FE / "lib" / "components" / "settings" / "sections" / "AppearanceSection.svelte"
CONV_DEFAULTS = FE / "lib" / "components" / "settings" / "sections" / "ConversationDefaults.svelte"
ACCOUNT_AUTH = FE / "lib" / "components" / "settings" / "sections" / "AccountAuthMode.svelte"
SPECULATIVE = FE / "lib" / "components" / "settings" / "SpeculativeSettings.svelte"
SIDEBAR = FE / "lib" / "components" / "sidebar" / "SectionContextList.svelte"
PREFERENCES = FE / "lib" / "stores" / "preferences.ts"
TOKENS = FE / "styles" / "tokens.css"
DENSITY = FE / "styles" / "density.css"
APP_CSS = FE / "app.css"
LOGIN = FE / "routes" / "login" / "+page.svelte"
REGISTER = FE / "routes" / "register" / "+page.svelte"

NEW_SVELTE = [SETTINGS_GROUP, APPEARANCE, CONV_DEFAULTS, ACCOUNT_AUTH, SPECULATIVE, SETTINGS_PAGE, LOGIN, REGISTER]

SECTION_IDS = [
    "appearance", "account", "conversation", "models",
    "knowledge", "plugins", "performance", "network", "data",
]

LEGACY_TABS = [
    "quick", "presets", "prompt", "models", "analytics", "performance",
    "fine-tune", "knowledge", "plugins", "backup", "security", "advanced",
]

# Accented Latin letters that flag French text leaking into code/UI.
_FRENCH = re.compile(r"[\u00e0\u00e2\u00e4\u00e9\u00e8\u00ea\u00eb\u00ee\u00ef"
                     r"\u00f4\u00f9\u00fb\u00fc\u00e7\u00c0\u00c9\u00c8\u00ca\u00c7]")
_HEX = re.compile(r"#[0-9a-fA-F]{6}\b")
_VAR_FALLBACK = re.compile(r"var\(\s*--oo-[a-z0-9-]+\s*,\s*#[0-9a-fA-F]{3,6}\s*\)")

# Matches a section header (id immediately followed by a label) in both the
# multi-line page registry and the single-line sidebar list. Group ids use
# `title:` not `label:`, so this isolates the 9 section ids. Reads ground
# truth from file content, so it does not depend on any module-level literal.
SECTION_HDR = re.compile(r"id:\s*'([a-z]+)'\s*,\s*\n?\s*label:\s*'")


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# =====================================================================
# 1. File presence
# =====================================================================

class TestFilePresence:
    def test_settings_page_exists(self):
        assert SETTINGS_PAGE.is_file()

    def test_settings_group_exists(self):
        assert SETTINGS_GROUP.is_file()

    def test_appearance_section_exists(self):
        assert APPEARANCE.is_file()

    def test_conversation_defaults_exists(self):
        assert CONV_DEFAULTS.is_file()

    def test_account_auth_mode_exists(self):
        assert ACCOUNT_AUTH.is_file()

    def test_speculative_settings_exists(self):
        assert SPECULATIVE.is_file()


# =====================================================================
# 2. The nine sections
# =====================================================================

class TestNineSections:
    def test_all_section_ids_present(self):
        # Derive the section ids from the page itself (uncorruptible by other
        # tests) and require exactly the nine unique headers.
        ids = SECTION_HDR.findall(_read(SETTINGS_PAGE))
        assert len(ids) == 9, f"expected 9 section headers, found {ids}"
        assert len(set(ids)) == 9, f"duplicate section ids: {ids}"

    def test_section_labels_present(self):
        c = _read(SETTINGS_PAGE)
        for label in ("Appearance", "Account & Security", "Conversation & Chat",
                      "Models & Inference", "Knowledge (RAG)", "Plugins & Extensions",
                      "Performance & Telemetry", "Network & Privacy", "Backup & Data"):
            assert label in c, f"section label missing: {label}"

    def test_exactly_nine_section_entries(self):
        ids = SECTION_HDR.findall(_read(SETTINGS_PAGE))
        assert len(ids) == 9

    def test_tablist_present(self):
        c = _read(SETTINGS_PAGE)
        assert 'role="tablist"' in c
        assert 'role="tab"' in c

    def test_section_nav_iterates_sections(self):
        c = _read(SETTINGS_PAGE)
        assert "each SECTIONS as s" in c


# =====================================================================
# 3. Deep-linking
# =====================================================================

class TestDeepLinking:
    def test_reads_section_param(self):
        c = _read(SETTINGS_PAGE)
        assert "searchParams.get('section')" in c

    def test_reads_tab_param_for_compat(self):
        c = _read(SETTINGS_PAGE)
        assert "searchParams.get('tab')" in c

    def test_legacy_map_present(self):
        c = _read(SETTINGS_PAGE)
        assert "LEGACY_TAB_TO_SECTION" in c

    def test_legacy_map_covers_all_old_tabs(self):
        c = _read(SETTINGS_PAGE)
        # The map block resolves each legacy tab id.
        for tab in LEGACY_TABS:
            assert f"'{tab}'" in c, f"legacy tab not mapped: {tab}"

    def test_backup_quoted_key_present(self):
        # s121 backwards-compat: 'backup' string survives as a legacy key.
        c = _read(SETTINGS_PAGE)
        assert "'backup'" in c

    def test_legacy_redirect_uses_replace_state(self):
        c = _read(SETTINGS_PAGE)
        assert "replaceState: true" in c

    def test_query_param_handled(self):
        c = _read(SETTINGS_PAGE)
        assert "urlQuery" in c
        assert "searchParams.get('q')" in c

    def test_group_deep_link(self):
        c = _read(SETTINGS_PAGE)
        assert "deepGroup" in c
        assert "scrollToGroup" in c

    def test_group_anchor_prefix(self):
        # SettingsGroup renders id="oo-set-<id>"; the page scrolls to it.
        assert "oo-set-" in _read(SETTINGS_PAGE)
        assert "oo-set-" in _read(SETTINGS_GROUP)

    def test_resolve_section_function(self):
        c = _read(SETTINGS_PAGE)
        assert "function resolveSection" in c


# =====================================================================
# 4. Search
# =====================================================================

class TestSearch:
    def test_search_index_built(self):
        c = _read(SETTINGS_PAGE)
        assert "searchIndex" in c
        assert "haystack" in c

    def test_results_derived_from_input(self):
        c = _read(SETTINGS_PAGE)
        assert "results" in c
        assert "searchInput" in c

    def test_token_and_matching(self):
        # Every whitespace-separated token must be contained in the haystack.
        c = _read(SETTINGS_PAGE)
        assert ".every(" in c and "includes(" in c

    def test_inline_index_present(self):
        c = _read(SETTINGS_PAGE)
        assert "INLINE_INDEX" in c

    def test_enter_opens_top_result(self):
        c = _read(SETTINGS_PAGE)
        assert "handleSearchKeydown" in c
        assert "'Enter'" in c
        assert "openResult(results[0])" in c

    def test_escape_clears_search(self):
        c = _read(SETTINGS_PAGE)
        assert "'Escape'" in c
        assert "clearSearch" in c

    def test_clear_button_present(self):
        c = _read(SETTINGS_PAGE)
        assert "Clear search" in c

    def test_query_sync_debounced(self):
        c = _read(SETTINGS_PAGE)
        assert "setTimeout" in c and "onSearchInput" in c

    def test_search_finds_llamacpp(self):
        # A search for "llama.cpp" must hit the merged speculative group.
        assert "'llama.cpp'" in _read(SETTINGS_PAGE)

    def test_search_finds_dark_mode(self):
        assert "'dark mode'" in _read(SETTINGS_PAGE)

    def test_search_finds_chunk_size(self):
        assert "'chunk size'" in _read(SETTINGS_PAGE)

    def test_search_finds_two_factor(self):
        c = _read(SETTINGS_PAGE)
        assert "'2fa'" in c


# =====================================================================
# 5. Migration completeness -- every old tab's content has a home
# =====================================================================

# Panels that must be referenced somewhere in the new structure.
PANELS_IN_PAGE = [
    "PresetManager", "PromptConfigPanel", "CompressionSettings", "ContextOptimizerPanel",
    "HumanizerPanel", "ModelHealthWidget", "ModelProfilePanel", "ModelAssignment",
    "LearnedRouterPanel", "CascadingPanel", "VisionModelSelector", "KnowledgeBasePanel",
    "RAGDashboardPanel", "PluginsPanel", "PluginMarketplace", "PluginAllowlistPanel",
    "CacheStatsPanel", "ObservabilityPanel", "TelemetryDashboard", "TelemetryHistoryPanel",
    "ProfilerDashboard", "PerformanceTunerPanel", "PerformanceDashboard", "AnalyticsDashboard",
    "ProxySettingsPanel", "RemoteAccessPanel", "SearchKillSwitchPanel", "SecurityModePanel",
    "TOTPSetup", "WebAuthnSetup", "RecoveryCodesPanel", "AppPasswordsPanel",
    "HardeningPanel", "KeyCeremonyPanel", "AuditChainPanel", "BackupRestorePanel",
    "FineTunePanel",
]


class TestMigrationCompleteness:
    def test_all_panels_referenced_in_page(self):
        c = _read(SETTINGS_PAGE)
        missing = [p for p in PANELS_IN_PAGE if p not in c]
        assert not missing, f"panels not referenced in settings page: {missing}"

    def test_lazy_loading_preserved(self):
        # S134 dynamic-import pattern retained.
        c = _read(SETTINGS_PAGE)
        assert "import(" in c
        assert "_cache" in c
        assert "loadComponent" in c
        assert "SkeletonLoader" in c

    def test_feature_map_gating_preserved(self):
        # S135 featureMap gating retained (>= 3 references).
        c = _read(SETTINGS_PAGE)
        assert "getFeatureMap" in c
        assert len(re.findall(r"featureMap\[", c)) >= 3

    def test_keyboard_nav_preserved(self):
        c = _read(SETTINGS_PAGE)
        assert "handleTabKeydown" in c
        assert "ArrowLeft" in c
        assert "ArrowRight" in c

    def test_observability_observe_synonym(self):
        # s114 intent: observability reachable and "Observe" discoverable.
        c = _read(SETTINGS_PAGE)
        assert "ObservabilityPanel" in c
        assert "Observe" in c

    def test_humanizer_left_advanced(self):
        # Humanizer now lives in Conversation (out of the old "advanced").
        c = _read(SETTINGS_PAGE)
        assert "output-humanizer" in c
        assert "HumanizerPanel" in c

    def test_cache_and_proxy_have_homes(self):
        c = _read(SETTINGS_PAGE)
        assert "CacheStatsPanel" in c  # Performance
        assert "ProxySettingsPanel" in c  # Network


# =====================================================================
# 6. Previously unreachable settings now reachable (spec 5.8)
# =====================================================================

class TestNowReachable:
    def test_plugin_allowlist_top_level_group(self):
        c = _read(SETTINGS_PAGE)
        assert "plugin-allowlist" in c
        assert "PluginAllowlistPanel" in c

    def test_audit_chain_top_level_group(self):
        c = _read(SETTINGS_PAGE)
        assert "audit-chain" in c
        assert "AuditChainPanel" in c

    def test_telemetry_history_top_level_group(self):
        c = _read(SETTINGS_PAGE)
        assert "telemetry-history" in c
        assert "TelemetryHistoryPanel" in c

    def test_rag_dashboard_top_level_group(self):
        c = _read(SETTINGS_PAGE)
        assert "rag-dashboard" in c
        assert "RAGDashboardPanel" in c

    def test_theme_customizer_reintegrated(self):
        c = _read(APPEARANCE)
        assert "ThemeCustomizer" in c
        assert "import ThemeCustomizer" in c

    def test_shortcut_settings_reintegrated(self):
        c = _read(APPEARANCE)
        assert "ShortcutSettings" in c
        assert "import ShortcutSettings" in c

    def test_orphans_opened_in_drawer(self):
        c = _read(APPEARANCE)
        assert 'variant="drawer-right"' in c


# =====================================================================
# 7. Speculative merge (re-asserts routes_speculative intent)
# =====================================================================

class TestSpeculativeMerge:
    def test_merge_host_imports_both_panels(self):
        c = _read(SPECULATIVE)
        assert "import SpeculativePanel" in c
        assert "import SpeculativeDecodingPanel" in c

    def test_page_uses_merge_host(self):
        c = _read(SETTINGS_PAGE)
        assert "SpeculativeSettings" in c

    def test_single_speculative_group(self):
        c = _read(SETTINGS_PAGE)
        assert "id: 'speculative'" in c

    def test_merge_has_two_tabs(self):
        c = _read(SPECULATIVE)
        assert "generation" in c
        assert "decoding" in c

    def test_both_panels_still_rendered(self):
        c = _read(SPECULATIVE)
        assert "<SpeculativePanel" in c
        assert "<SpeculativeDecodingPanel" in c


# =====================================================================
# 8. Appearance build-out
# =====================================================================

class TestAppearanceSection:
    def test_palette_controls(self):
        c = _read(APPEARANCE)
        assert "setPalette" in c
        assert "PALETTES" in c

    def test_density_controls(self):
        c = _read(APPEARANCE)
        assert "setDensity" in c
        assert "DENSITIES" in c

    def test_typography_controls(self):
        c = _read(APPEARANCE)
        assert "setTypeScale" in c
        assert "TYPE_SCALES" in c

    def test_motion_controls(self):
        c = _read(APPEARANCE)
        assert "setMotionPref" in c
        assert "MOTION_PREFS" in c

    def test_per_group_reset(self):
        c = _read(APPEARANCE)
        assert "resetTheme" in c
        assert "resetDensity" in c
        assert "resetTypeScale" in c
        assert "resetMotion" in c
        assert "onReset=" in c

    def test_immediate_apply_toasts(self):
        c = _read(APPEARANCE)
        assert "toastSuccess" in c

    def test_radiogroup_semantics(self):
        c = _read(APPEARANCE)
        assert 'role="radiogroup"' in c
        assert 'role="radio"' in c


# =====================================================================
# 9. Preferences store extension
# =====================================================================

class TestPreferencesStore:
    def test_type_scale_store(self):
        c = _read(PREFERENCES)
        assert "export const typeScale" in c

    def test_motion_store(self):
        c = _read(PREFERENCES)
        assert "export const motionPref" in c

    def test_type_scales_four(self):
        c = _read(PREFERENCES)
        m = re.search(r"TYPE_SCALES:\s*TypeScale\[\]\s*=\s*\[([^\]]*)\]", c)
        assert m, "TYPE_SCALES not found"
        assert m.group(1).count("'") == 8  # 4 quoted values

    def test_motion_prefs_three(self):
        c = _read(PREFERENCES)
        m = re.search(r"MOTION_PREFS:\s*MotionPref\[\]\s*=\s*\[([^\]]*)\]", c)
        assert m, "MOTION_PREFS not found"
        assert m.group(1).count("'") == 6  # 3 quoted values

    def test_setters_exported(self):
        c = _read(PREFERENCES)
        assert "export function setTypeScale" in c
        assert "export function setMotionPref" in c

    def test_apply_type_scale_sets_var(self):
        c = _read(PREFERENCES)
        assert "--oo-type-scale" in c

    def test_apply_motion_classes(self):
        c = _read(PREFERENCES)
        assert "oo-reduce-motion" in c
        assert "oo-motion-full" in c

    def test_init_applies_new_prefs(self):
        c = _read(PREFERENCES)
        block = c[c.index("export function initPreferences"):]
        assert "applyTypeScale" in block
        assert "applyMotion" in block

    def test_motion_syncs_reduced_motion_store(self):
        c = _read(PREFERENCES)
        assert "prefersReducedMotion.set" in c


# =====================================================================
# 10. Typography tokens (calc multiplier)
# =====================================================================

class TestTypographyTokens:
    def test_type_scale_declared_in_tokens(self):
        c = _read(TOKENS)
        assert re.search(r"--oo-type-scale:\s*1\s*;", c)

    def test_tokens_text_use_multiplier(self):
        c = _read(TOKENS)
        assert "calc(" in c
        assert "var(--oo-type-scale" in c
        # every text token multiplies
        for t in ("--oo-text-xs", "--oo-text-base", "--oo-text-2xl"):
            assert re.search(rf"{t}:\s*calc\([0-9]+px \* var\(--oo-type-scale", c)

    def test_density_text_use_multiplier(self):
        c = _read(DENSITY)
        assert c.count("var(--oo-type-scale") >= 24  # 3 densities x 8 tokens

    def test_no_unresolved_type_scale(self):
        # The token is declared, so var(--oo-type-scale) resolves in CSS.
        assert "--oo-type-scale:" in _read(TOKENS)


# =====================================================================
# 11. Motion CSS (app.css)
# =====================================================================

class TestMotionCss:
    def test_forced_reduce_rule(self):
        c = _read(APP_CSS)
        assert "html.oo-reduce-motion" in c

    def test_full_motion_opt_out(self):
        c = _read(APP_CSS)
        assert "html:not(.oo-motion-full)" in c

    def test_reduced_durations_present(self):
        c = _read(APP_CSS)
        assert "animation-duration: 0.01ms" in c
        assert "transition-duration: 0.01ms" in c


# =====================================================================
# 12. ConversationDefaults (re-asserts s84 system-preset intent)
# =====================================================================

class TestConversationDefaults:
    def test_imports_system_presets_api(self):
        c = _read(CONV_DEFAULTS)
        assert "systemPresets" in c
        assert "listSystemPresets" in c

    def test_apply_system_preset_handler(self):
        c = _read(CONV_DEFAULTS)
        assert "handleApplySystemPreset" in c
        assert "applySystemPreset" in c

    def test_recommended_badge(self):
        c = _read(CONV_DEFAULTS)
        assert "Recommended" in c

    def test_human_readable_descriptions(self):
        c = _read(CONV_DEFAULTS)
        assert "preset.description" in c

    def test_onboarding_reset(self):
        c = _read(CONV_DEFAULTS)
        assert "resetOnboarding" in c
        assert "handleResetOnboarding" in c

    def test_defaults_controls(self):
        c = _read(CONV_DEFAULTS)
        assert "Default model" in c
        assert "Default temperature" in c
        assert "Code execution" in c
        assert "Memory injection" in c

    def test_defaults_reset(self):
        c = _read(CONV_DEFAULTS)
        assert "resetDefaults" in c

    def test_immediate_apply(self):
        c = _read(CONV_DEFAULTS)
        assert "saveSetting" in c
        assert "toastSuccess" in c


# =====================================================================
# 13. AccountAuthMode (re-asserts s109 auth-toggle intent)
# =====================================================================

class TestAccountAuthMode:
    def test_authentication_label(self):
        c = _read(ACCOUNT_AUTH)
        assert "Authentication" in c

    def test_toggle_auth_mode(self):
        c = _read(ACCOUNT_AUTH)
        assert "toggleAuthMode" in c
        assert "/api/auth/mode" in c

    def test_fetches_auth_status(self):
        c = _read(ACCOUNT_AUTH)
        assert "/api/auth/status" in c
        assert "singleUserMode" in c


# =====================================================================
# 14. Version badge (re-asserts s87 intent)
# =====================================================================

class TestVersionBadge:
    def test_app_version_state(self):
        c = _read(SETTINGS_PAGE)
        assert "appVersion" in c

    def test_fetches_health(self):
        c = _read(SETTINGS_PAGE)
        assert "fetch('/api/health')" in c
        assert "data.version" in c

    def test_version_badge_rendered(self):
        c = _read(SETTINGS_PAGE)
        assert "v{appVersion}" in c
        assert "{#if appVersion}" in c


# =====================================================================
# 15. Auth pages on primitives + theme parity (spec 12.3)
# =====================================================================

class TestAuthPagesPrimitives:
    def test_login_uses_primitives(self):
        c = _read(LOGIN)
        assert "$lib/ds/Card.svelte" in c
        assert "$lib/ds/Input.svelte" in c
        assert "$lib/ds/Button.svelte" in c

    def test_register_uses_primitives(self):
        c = _read(REGISTER)
        assert "$lib/ds/Card.svelte" in c
        assert "$lib/ds/Input.svelte" in c
        assert "$lib/ds/Button.svelte" in c

    def test_login_no_legacy_tokens(self):
        c = _read(LOGIN)
        for legacy in ("--oo-btn-primary", "--oo-accent-primary", "--oo-input-bd", "--oo-input-focus"):
            assert legacy not in c, f"legacy token still in login: {legacy}"

    def test_register_no_legacy_tokens(self):
        c = _read(REGISTER)
        for legacy in ("--oo-btn-primary", "--oo-accent-primary", "--oo-input-bd", "--oo-input-focus"):
            assert legacy not in c, f"legacy token still in register: {legacy}"

    def test_palette_token_background(self):
        # Palette parity: page background reads a palette token.
        assert "var(--oo-bg-base)" in _read(LOGIN)
        assert "var(--oo-bg-base)" in _read(REGISTER)

    def test_login_logic_preserved(self):
        c = _read(LOGIN)
        assert "doLogin" in c
        assert '/register' in c

    def test_register_logic_preserved(self):
        c = _read(REGISTER)
        assert "doRegister" in c
        assert "registrationDisabled" in c
        assert '/login' in c


# =====================================================================
# 16. Sidebar updated to nine sections
# =====================================================================

class TestSidebarSections:
    def test_sidebar_nine_sections(self):
        # Sidebar section ids derived from the file, and must match the page.
        side_ids = SECTION_HDR.findall(_read(SIDEBAR))
        page_ids = SECTION_HDR.findall(_read(SETTINGS_PAGE))
        assert len(side_ids) == 9, f"sidebar expected 9 sections, found {side_ids}"
        assert set(side_ids) == set(page_ids), \
            f"sidebar {sorted(set(side_ids))} != page {sorted(set(page_ids))}"

    def test_sidebar_legacy_map(self):
        c = _read(SIDEBAR)
        assert "LEGACY_TAB_TO_SECTION" in c

    def test_sidebar_settings_links(self):
        c = _read(SIDEBAR)
        assert "/settings?tab=" in c


# =====================================================================
# 17. Code hygiene -- no hardcoded hex, English only, balanced tags
# =====================================================================

class TestNoHardcodedHex:
    def test_new_svelte_no_raw_hex(self):
        for p in NEW_SVELTE:
            t = _VAR_FALLBACK.sub("", _read(p))
            # ignore comment lines
            hits = [m.group() for line in t.splitlines()
                    for m in [_HEX.search(line)]
                    if m and not line.strip().startswith(("<!--", "//", "*"))]
            assert not hits, f"raw hex in {p.name}: {hits[:5]}"


class TestEnglishOnly:
    def test_new_svelte_no_french(self):
        for p in NEW_SVELTE + [PREFERENCES]:
            t = _read(p)
            lines = [i + 1 for i, line in enumerate(t.splitlines()) if _FRENCH.search(line)]
            assert not lines, f"French detected in {p.name} at lines {lines[:5]}"


class TestHtmlBalance:
    def test_new_svelte_balanced_tags(self):
        for p in NEW_SVELTE:
            t = _read(p)
            for tag in ("div", "button", "section"):
                opens = len(re.findall(rf"<{tag}[\s>]", t))
                selfc = len(re.findall(rf"<{tag}[^>]*/>", t))
                closes = len(re.findall(rf"</{tag}>", t))
                assert opens - selfc == closes, \
                    f"{p.name}: <{tag}> {opens - selfc} open vs {closes} close"
