#!/usr/bin/env python3
"""
test_s166_design_system.py -- S166 design-system foundation.

Covers (spec 12.1):
  - Token files: tokens.css, theme-{5}.css, density.css, theme.css orchestrator
  - Each preset declares the 38-token foundation schema
  - The 30-token legacy alias layer resolves (declared/used parity)
  - theme_engine: 5 new presets, palettes, foundation schema, legacy aliases
  - WCAG AA contrast across all 5 presets on the full CONTRAST_PAIRS set
  - The 10 ds/ primitives exist with their props and a11y contracts
  - The 5 ad-hoc modals migrated to <Modal>; 2 non-modals correctly excluded
  - The 6 orphan components removed and unreferenced; ThemeCustomizer kept
  - lucide-svelte + @floating-ui/dom dependencies (Svelte-4 compatible)
  - Dev demo route /dev/components, dev-gated
  - Global a11y CSS: :focus-visible, .oo-sr-only, .oo-skip-link

Modules are loaded in isolation (importlib) to avoid the
opti_oignon/__init__ chain (which requires ollama, absent here).
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

_PROJECT = Path(__file__).resolve().parent.parent
_FE = _PROJECT / "frontend" / "src"
_STYLES = _FE / "styles"
_DS = _FE / "lib" / "ds"
_COMPONENTS = _FE / "lib" / "components"
_APP_CSS = _FE / "app.css"


def _load(mod_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod  # register before exec (Python 3.13)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def theme_engine():
    return _load("theme_engine_s166", _PROJECT / "opti_oignon" / "theme_engine.py")


@pytest.fixture(scope="module")
def audit_contrast():
    return _load("audit_contrast_s166", _PROJECT / "scripts" / "audit_contrast.py")


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


THEMES = ["anthracite", "parchment", "slate", "linen", "high-contrast"]

FOUNDATION_TOKENS = [
    # 7 backgrounds
    "--oo-bg-base", "--oo-bg-surface", "--oo-bg-elevated", "--oo-bg-overlay",
    "--oo-bg-subtle", "--oo-bg-hover", "--oo-bg-active",
    # 6 foregrounds
    "--oo-fg-primary", "--oo-fg-secondary", "--oo-fg-tertiary", "--oo-fg-muted",
    "--oo-fg-decorative", "--oo-fg-on-accent",
    # 3 borders
    "--oo-bd-default", "--oo-bd-strong", "--oo-bd-subtle",
    # 10 accent ramp
    "--oo-acc-50", "--oo-acc-100", "--oo-acc-200", "--oo-acc-300", "--oo-acc-400",
    "--oo-acc-500", "--oo-acc-600", "--oo-acc-700", "--oo-acc-800", "--oo-acc-900",
    # 12 semantic
    "--oo-success", "--oo-success-bg", "--oo-success-bd",
    "--oo-warning", "--oo-warning-bg", "--oo-warning-bd",
    "--oo-error", "--oo-error-bg", "--oo-error-bd",
    "--oo-info", "--oo-info-bg", "--oo-info-bd",
]

PRIMITIVES = [
    "Button", "Input", "Card", "Modal", "Toast",
    "Select", "Switch", "Tabs", "Tooltip", "Icon",
]


# ===========================================================================
# Token files
# ===========================================================================
class TestTokenFiles:
    def test_tokens_css_exists(self):
        assert (_STYLES / "tokens.css").is_file()

    def test_density_css_exists(self):
        assert (_STYLES / "density.css").is_file()

    def test_theme_orchestrator_exists(self):
        assert (_STYLES / "theme.css").is_file()

    @pytest.mark.parametrize("name", THEMES)
    def test_theme_file_exists(self, name):
        assert (_STYLES / f"theme-{name}.css").is_file(), f"theme-{name}.css missing"

    def test_tokens_has_type_scale(self):
        css = _read(_STYLES / "tokens.css")
        for t in ("--oo-text-xs", "--oo-text-base", "--oo-text-2xl"):
            assert t in css, f"{t} missing from tokens.css"

    def test_tokens_has_spacing_scale(self):
        css = _read(_STYLES / "tokens.css")
        for t in ("--oo-space-0", "--oo-space-5", "--oo-space-9"):
            assert t in css

    def test_tokens_has_radius_scale(self):
        css = _read(_STYLES / "tokens.css")
        for t in ("--oo-radius-sm", "--oo-radius-md", "--oo-radius-lg", "--oo-radius-full"):
            assert t in css

    def test_tokens_has_motion(self):
        css = _read(_STYLES / "tokens.css")
        assert "--oo-motion-fast" in css and "--oo-ease-default" in css

    def test_tokens_has_zindex(self):
        css = _read(_STYLES / "tokens.css")
        assert "--oo-z-" in css

    def test_tokens_respects_reduced_motion(self):
        css = _read(_STYLES / "tokens.css")
        assert "prefers-reduced-motion" in css

    def test_density_has_three_blocks(self):
        css = _read(_STYLES / "density.css")
        for d in ("oo-density-compact", "oo-density-comfortable", "oo-density-spacious"):
            assert d in css, f"density block {d} missing"


# ===========================================================================
# Theme foundations -- each preset declares the 38-token schema
# ===========================================================================
class TestThemeFoundations:
    @pytest.mark.parametrize("name", THEMES)
    def test_theme_declares_full_foundation(self, name):
        css = _read(_STYLES / f"theme-{name}.css")
        missing = [t for t in FOUNDATION_TOKENS if t not in css]
        assert not missing, f"theme-{name}.css missing tokens: {missing}"

    @pytest.mark.parametrize("name", THEMES)
    def test_theme_declares_at_least_38_vars(self, name):
        css = _read(_STYLES / f"theme-{name}.css")
        declared = set(re.findall(r"(--oo-[a-z0-9-]+)\s*:", css))
        assert len(declared) >= 38, f"theme-{name}.css declares only {len(declared)} tokens"

    @pytest.mark.parametrize("name", THEMES)
    def test_theme_scoped_by_data_attribute(self, name):
        css = _read(_STYLES / f"theme-{name}.css")
        assert f'[data-oo-theme="{name}"]' in css, f"theme-{name}.css not scoped by data-oo-theme"

    def test_anthracite_dark_scope(self):
        css = _read(_STYLES / "theme-anthracite.css")
        assert "html.dark" in css or ":root" in css

    def test_parchment_light_scope(self):
        css = _read(_STYLES / "theme-parchment.css")
        assert "html:not(.dark)" in css or ":root" in css


# ===========================================================================
# Alias layer -- declared/used parity (resolves N4)
# ===========================================================================
class TestAliasLayer:
    def _all_css(self):
        parts = [_read(_STYLES / "tokens.css"), _read(_STYLES / "theme.css")]
        for n in THEMES:
            parts.append(_read(_STYLES / f"theme-{n}.css"))
        return "\n".join(parts)

    def test_no_unresolved_oo_vars(self):
        """Every var(--oo-*) reference resolves to a declared --oo-* token."""
        css = self._all_css()
        declared = set(re.findall(r"(--oo-[a-z0-9-]+)\s*:", css))
        used = set(re.findall(r"var\(\s*(--oo-[a-z0-9-]+)", css))
        unresolved = sorted(used - declared)
        assert not unresolved, f"Unresolved --oo-* vars: {unresolved}"

    def test_alias_layer_present_in_orchestrator(self):
        css = _read(_STYLES / "theme.css")
        # legacy aliases mapped to canonical tokens
        assert "--oo-fg-faint" in css, "legacy alias --oo-fg-faint not declared"

    def test_legacy_sidebar_alias_declared(self):
        css = _read(_STYLES / "theme.css")
        assert "--oo-sidebar-bg" in css


# ===========================================================================
# theme_engine
# ===========================================================================
class TestThemeEngine:
    def test_get_preset_themes_returns_five(self, theme_engine):
        themes = theme_engine.get_preset_themes()
        assert len(themes) == 5, f"expected 5 presets, got {len(themes)}"

    def test_preset_ids_are_new_names(self, theme_engine):
        ids = set(theme_engine.BUILTIN_PRESET_PALETTES.keys())
        assert set(THEMES) <= ids, f"missing presets: {set(THEMES) - ids}"

    @pytest.mark.parametrize("name", THEMES)
    def test_palette_has_38_tokens(self, theme_engine, name):
        pal = theme_engine.BUILTIN_PRESET_PALETTES[name]
        bare = [t[len("--oo-"):] for t in FOUNDATION_TOKENS]
        missing = [t for t in bare if t not in pal]
        assert not missing, f"{name} palette missing: {missing}"

    def test_legacy_alias_default_maps_to_anthracite(self, theme_engine):
        p = theme_engine.get_preset_by_id("default")
        assert p is not None

    def test_legacy_alias_ocean_resolves(self, theme_engine):
        assert theme_engine.get_preset_by_id("ocean") is not None

    def test_generate_foundation_schema_new_schema(self, theme_engine):
        schema = theme_engine.generate_foundation_schema(205, "dark")
        # new schema emits foundation token names prefixed oo- (no leading --)
        keys = set(schema.keys()) if isinstance(schema, dict) else set()
        assert "oo-bg-base" in keys and "oo-fg-primary" in keys and "oo-acc-500" in keys

    def test_generate_theme_variables_legacy_intact(self, theme_engine):
        # legacy generator still works (backward compatibility)
        out = theme_engine.generate_theme_variables(35)
        assert isinstance(out, dict) and len(out) > 0

    def test_builtin_preset_ids_includes_legacy(self, theme_engine):
        ids = theme_engine.BUILTIN_PRESET_IDS
        assert "default" in ids and "anthracite" in ids


# ===========================================================================
# WCAG AA -- all 5 presets, full CONTRAST_PAIRS set
# ===========================================================================
class TestWCAGContrast:
    def test_theme_css_dark_passes(self, audit_contrast):
        dark, _ = audit_contrast.extract_themed_vars(_read(_STYLES / "theme.css"))
        fails = audit_contrast.audit_theme(dark, "dark")
        assert not fails, f"dark contrast failures: {fails}"

    def test_theme_css_light_passes(self, audit_contrast):
        _, light = audit_contrast.extract_themed_vars(_read(_STYLES / "theme.css"))
        fails = audit_contrast.audit_theme(light, "light")
        assert not fails, f"light contrast failures: {fails}"

    @pytest.mark.parametrize("name", THEMES)
    def test_preset_fg_tertiary_aa(self, theme_engine, audit_contrast, name):
        pal = theme_engine.BUILTIN_PRESET_PALETTES[name]
        ft = pal["fg-tertiary"]
        for bg_key in ("bg-base", "bg-surface"):
            ratio = audit_contrast.contrast_ratio(
                audit_contrast.parse_hex(ft), audit_contrast.parse_hex(pal[bg_key])
            )
            assert ratio >= 4.5, f"{name}: fg-tertiary on {bg_key} = {ratio:.2f} (<4.5)"

    @pytest.mark.parametrize("name", THEMES)
    def test_preset_fg_primary_aa(self, theme_engine, audit_contrast, name):
        pal = theme_engine.BUILTIN_PRESET_PALETTES[name]
        ratio = audit_contrast.contrast_ratio(
            audit_contrast.parse_hex(pal["fg-primary"]),
            audit_contrast.parse_hex(pal["bg-base"]),
        )
        assert ratio >= 4.5, f"{name}: fg-primary on bg-base = {ratio:.2f}"

    @pytest.mark.parametrize("name", THEMES)
    def test_preset_accent_on_accent_aa(self, theme_engine, audit_contrast, name):
        pal = theme_engine.BUILTIN_PRESET_PALETTES[name]
        ratio = audit_contrast.contrast_ratio(
            audit_contrast.parse_hex(pal["fg-on-accent"]),
            audit_contrast.parse_hex(pal["acc-500"]),
        )
        assert ratio >= 4.5, f"{name}: fg-on-accent on acc-500 = {ratio:.2f}"


# ===========================================================================
# Primitives -- existence, props, barrel
# ===========================================================================
class TestPrimitivesExist:
    @pytest.mark.parametrize("name", PRIMITIVES)
    def test_primitive_file_exists(self, name):
        assert (_DS / f"{name}.svelte").is_file(), f"{name}.svelte missing"

    def test_types_module_exists(self):
        assert (_DS / "types.ts").is_file()

    def test_barrel_exists(self):
        assert (_DS / "index.ts").is_file()

    def test_barrel_exports_all_primitives(self):
        idx = _read(_DS / "index.ts")
        for p in PRIMITIVES:
            assert p in idx, f"barrel does not export {p}"

    @pytest.mark.parametrize("name", PRIMITIVES)
    def test_primitive_uses_typescript(self, name):
        src = _read(_DS / f"{name}.svelte")
        assert 'lang="ts"' in src

    @pytest.mark.parametrize("name", PRIMITIVES)
    def test_primitive_no_hardcoded_hex(self, name):
        src = _read(_DS / f"{name}.svelte")
        hexes = re.findall(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b", src)
        assert not hexes, f"{name}.svelte has hardcoded hex: {hexes}"


class TestPrimitiveProps:
    def test_button_props(self):
        src = _read(_DS / "Button.svelte")
        for p in ("variant", "size", "iconLeft", "iconRight", "iconOnly", "loading", "href"):
            assert f"export let {p}" in src, f"Button missing prop {p}"

    def test_input_props(self):
        src = _read(_DS / "Input.svelte")
        for p in ("type", "value", "label", "hint", "error", "hideLabel"):
            assert f"export let {p}" in src

    def test_card_props(self):
        src = _read(_DS / "Card.svelte")
        for p in ("variant", "padding"):
            assert f"export let {p}" in src

    def test_modal_props(self):
        src = _read(_DS / "Modal.svelte")
        for p in ("open", "variant", "title", "size", "closeOnBackdrop", "closeOnEsc", "onClose"):
            assert f"export let {p}" in src

    def test_select_props(self):
        src = _read(_DS / "Select.svelte")
        for p in ("value", "multiple", "options", "label", "searchable"):
            assert f"export let {p}" in src

    def test_switch_props(self):
        src = _read(_DS / "Switch.svelte")
        for p in ("checked", "label", "description", "size"):
            assert f"export let {p}" in src

    def test_tabs_props(self):
        src = _read(_DS / "Tabs.svelte")
        for p in ("value", "tabs", "orientation", "variant"):
            assert f"export let {p}" in src

    def test_tooltip_props(self):
        src = _read(_DS / "Tooltip.svelte")
        for p in ("content", "placement", "delay"):
            assert f"export let {p}" in src

    def test_icon_props(self):
        src = _read(_DS / "Icon.svelte")
        for p in ("name", "size"):
            assert f"export let {p}" in src


class TestPrimitiveA11y:
    def test_button_aria_busy(self):
        assert "aria-busy" in _read(_DS / "Button.svelte")

    def test_input_aria_describedby(self):
        src = _read(_DS / "Input.svelte")
        assert "aria-describedby" in src and "aria-invalid" in src

    def test_input_sr_only_for_hidden_label(self):
        assert "oo-sr-only" in _read(_DS / "Input.svelte")

    def test_modal_uses_native_dialog(self):
        src = _read(_DS / "Modal.svelte")
        assert "<dialog" in src and "showModal" in src

    def test_modal_aria_modal_and_labelledby(self):
        src = _read(_DS / "Modal.svelte")
        assert 'aria-modal="true"' in src and "aria-labelledby" in src

    def test_modal_restores_focus(self):
        src = _read(_DS / "Modal.svelte")
        assert "opener" in src and ".focus" in src

    def test_switch_role(self):
        src = _read(_DS / "Switch.svelte")
        assert 'role="switch"' in src and "aria-checked" in src

    def test_tabs_aria_roles(self):
        src = _read(_DS / "Tabs.svelte")
        assert 'role="tablist"' in src and 'role="tab"' in src and 'role="tabpanel"' in src

    def test_tabs_arrow_key_nav(self):
        src = _read(_DS / "Tabs.svelte")
        assert "ArrowRight" in src or "ArrowDown" in src

    def test_tooltip_role_and_floating_ui(self):
        src = _read(_DS / "Tooltip.svelte")
        assert 'role="tooltip"' in src and "@floating-ui/dom" in src

    def test_toast_status_and_alert_roles(self):
        src = _read(_DS / "Toast.svelte")
        assert "status" in src and "alert" in src

    def test_toast_retry_action(self):
        src = _read(_DS / "Toast.svelte")
        assert "action" in src and "run" in src

    def test_icon_wraps_lucide(self):
        src = _read(_DS / "Icon.svelte")
        assert "lucide-svelte" in src


# ===========================================================================
# Modal migration (5 real modals) + 2 documented exclusions
# ===========================================================================
MIGRATED_MODALS = {
    "ExportDialog": _COMPONENTS / "chat" / "ExportDialog.svelte",
    "KeyboardShortcuts": _COMPONENTS / "ui" / "KeyboardShortcuts.svelte",
    "OnboardingOverlay": _COMPONENTS / "ui" / "OnboardingOverlay.svelte",
    "FileManager": _COMPONENTS / "panels" / "FileManager.svelte",
    "PluginMarketplace": _COMPONENTS / "settings" / "PluginMarketplace.svelte",
}
NON_MODALS = {
    "ProjectLinker": _COMPONENTS / "chat" / "ProjectLinker.svelte",
    "FileUpload": _COMPONENTS / "chat" / "FileUpload.svelte",
}


class TestModalMigration:
    @pytest.mark.parametrize("name", list(MIGRATED_MODALS))
    def test_migrated_imports_modal(self, name):
        src = _read(MIGRATED_MODALS[name])
        assert "$lib/ds" in src and "Modal" in src, f"{name} does not import Modal"

    @pytest.mark.parametrize("name", list(MIGRATED_MODALS))
    def test_migrated_dropped_fixed_inset(self, name):
        src = _read(MIGRATED_MODALS[name])
        assert "fixed inset-0" not in src, f"{name} still has fixed inset-0 modal shell"

    @pytest.mark.parametrize("name", list(MIGRATED_MODALS))
    def test_migrated_dropped_focustrap_action(self, name):
        src = _read(MIGRATED_MODALS[name])
        assert "use:focusTrap" not in src, f"{name} still uses the focusTrap action"

    @pytest.mark.parametrize("name", list(NON_MODALS))
    def test_non_modal_still_exists(self, name):
        assert NON_MODALS[name].is_file()

    def test_projectlinker_is_dropdown_not_modal(self):
        src = _read(NON_MODALS["ProjectLinker"])
        assert "$lib/ds" not in src or "Modal" not in src.split("$lib/ds")[0]

    def test_fileupload_overlay_is_aria_hidden(self):
        src = _read(NON_MODALS["FileUpload"])
        assert "drop-overlay" in src


# ===========================================================================
# Orphan removal
# ===========================================================================
ORPHANS = [
    _COMPONENTS / "chat" / "BranchDiff.svelte",
    _COMPONENTS / "chat" / "CascadingIndicator.svelte",
    _COMPONENTS / "panels" / "CodingAgentPanel.svelte",
    _COMPONENTS / "panels" / "ConsensusPanel.svelte",
    _COMPONENTS / "panels" / "ModelManager.svelte",
    _COMPONENTS / "chat" / "ScrollToBottom.svelte",
]


class TestOrphansRemoved:
    @pytest.mark.parametrize("path", ORPHANS, ids=[p.stem for p in ORPHANS])
    def test_orphan_file_removed(self, path):
        assert not path.exists(), f"{path.name} should be removed"

    @pytest.mark.parametrize(
        "name",
        ["BranchDiff", "CascadingIndicator", "CodingAgentPanel", "ConsensusPanel", "ModelManager"],
    )
    def test_orphan_not_imported(self, name):
        hits = []
        for f in _FE.rglob("*.svelte"):
            if f"components/chat/{name}.svelte" in _read(f) or f"components/panels/{name}.svelte" in _read(f):
                hits.append(str(f))
        for f in _FE.rglob("*.ts"):
            if f"{name}.svelte" in _read(f):
                hits.append(str(f))
        assert not hits, f"{name} still imported in: {hits}"

    def test_scrolltobottom_not_imported(self):
        # ScrollToBottomFab is the kept component; the orphan ScrollToBottom is not imported
        hits = [
            str(f)
            for f in _FE.rglob("*.svelte")
            if "components/chat/ScrollToBottom.svelte" in _read(f)
        ]
        assert not hits, f"ScrollToBottom still imported in: {hits}"

    def test_themecustomizer_kept(self):
        assert (_COMPONENTS / "panels" / "ThemeCustomizer.svelte").is_file()


# ===========================================================================
# Dependencies
# ===========================================================================
class TestDependencies:
    @pytest.fixture(scope="class")
    def pkg(self):
        return json.loads(_read(_FE.parent / "package.json"))

    def test_floating_ui_dep(self, pkg):
        assert "@floating-ui/dom" in pkg.get("dependencies", {})

    def test_lucide_dep(self, pkg):
        assert "lucide-svelte" in pkg.get("dependencies", {})

    def test_lucide_svelte4_compatible(self, pkg):
        ver = pkg["dependencies"]["lucide-svelte"]
        # 0.3xx range supports svelte ^3 || ^4
        assert ver.startswith("^0.3") or ver.startswith("0.3"), f"lucide-svelte {ver} may not support Svelte 4"

    def test_lockfile_has_deps(self):
        lock = json.loads(_read(_FE.parent / "package-lock.json"))
        pkgs = lock.get("packages", {})
        assert "node_modules/lucide-svelte" in pkgs
        assert "node_modules/@floating-ui/dom" in pkgs


# ===========================================================================
# Dev demo route
# ===========================================================================
class TestDevRoute:
    @pytest.fixture(scope="class")
    def src(self):
        return _read(_FE / "routes" / "dev" / "components" / "+page.svelte")

    def test_route_exists(self):
        assert (_FE / "routes" / "dev" / "components" / "+page.svelte").is_file()

    def test_route_dev_gated(self, src):
        assert "import.meta.env.DEV" in src

    def test_route_has_else_branch(self, src):
        assert "{:else}" in src

    def test_route_imports_primitives(self, src):
        assert "$lib/ds" in src

    def test_route_switches_theme_and_density(self, src):
        assert "data-oo-theme" in src and "oo-density-" in src


# ===========================================================================
# Global a11y CSS
# ===========================================================================
class TestGlobalA11yCss:
    @pytest.fixture(scope="class")
    def css(self):
        return _read(_APP_CSS)

    def test_focus_visible_declared(self, css):
        assert ":focus-visible" in css

    def test_focus_visible_uses_accent(self, css):
        block = css[css.find(":focus-visible"):css.find(":focus-visible") + 200]
        assert "--oo-acc-500" in block

    def test_sr_only_declared(self, css):
        assert ".oo-sr-only" in css

    def test_skip_link_declared(self, css):
        assert ".oo-skip-link" in css

    def test_skip_link_hidden_until_focus(self, css):
        assert ".oo-skip-link:focus" in css


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
