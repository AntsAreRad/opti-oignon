"""S197 F10a -- design-system primitives + theme engine.

Per-fix tests for the F10a lot:
- DS-01: var(--oo-*, #hex) fallbacks stripped across the Svelte tree.
- DS-02: ThemeCustomizer carries no raw hex; literals live in a TS module;
  the contrast badge reads the live --oo-bg-surface of the current palette.
- DS-03: deselect-plus-reassert for the superseded s83 palette values and
  the s152 version pin (current anchors and 3.6.0 asserted here).
Plus structural locks for the five S169 prop contracts verified this
session. All checks are source-level (no node toolchain in the container).
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_SRC = ROOT / "frontend" / "src"
DS = FRONTEND_SRC / "lib" / "ds"
PANELS = FRONTEND_SRC / "lib" / "components" / "panels"


def _svelte_files():
    return sorted(FRONTEND_SRC.rglob("*.svelte"))


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# -- DS-01: no var(--oo-*, #hex) fallback anywhere (svelte + app.css) --

FALLBACK_PAT = re.compile(r"var\(--oo-[^)]*#[0-9a-fA-F]")


def test_ds01_no_oo_var_hex_fallbacks():
    violations = []
    targets = list(_svelte_files()) + [FRONTEND_SRC / "app.css"]
    for f in targets:
        for i, line in enumerate(_read(f).splitlines(), 1):
            if FALLBACK_PAT.search(line):
                violations.append(f"{f.name}:{i}: {line.strip()[:90]}")
    assert violations == [], "var(--oo-*, #hex) fallbacks present:\n" + "\n".join(
        violations[:20]
    )


def test_ds01_stale_fallback_hexes_gone():
    # Spot anchors: the pre-S166 night-blue and Tailwind-red fallback values
    # that motivated the finding must not reappear anywhere in Svelte.
    stale = ("#1a1a2e", "#dc2626", "#c85050", "#d97706")
    hits = []
    for f in _svelte_files():
        content = _read(f)
        for h in stale:
            if h in content:
                hits.append(f"{f.name}: {h}")
    assert hits == [], f"stale fallback hexes resurfaced: {hits}"


# -- DS-02: ThemeCustomizer hex-free; constants module; live contrast bg --

CUSTOMIZER = PANELS / "ThemeCustomizer.svelte"
CONSTANTS = PANELS / "themeCustomizerConstants.ts"

HEX6 = re.compile(r"#[0-9a-fA-F]{6}\b")
# Mirrors the s83 invariant's allowance: the OnionLoader default prop.
RAW_HEX_ALLOWED = ("color: string = '#B07D56'",)
RAW_HEX_EXCLUDE = (r"\{#each", r"\{#if", r"&#\d", r"msg\.id", r"role\}-",
                   r"timestamp", r"`\$", r"\.id\b")


def test_ds02_no_raw_hex_in_any_svelte():
    violations = []
    for f in _svelte_files():
        for i, line in enumerate(_read(f).splitlines(), 1):
            if not HEX6.search(line):
                continue
            if "var(--oo-" in line:
                continue
            if any(a in line for a in RAW_HEX_ALLOWED):
                continue
            if any(re.search(p, line) for p in RAW_HEX_EXCLUDE):
                continue
            violations.append(f"{f.name}:{i}: {line.strip()[:90]}")
    assert violations == [], "raw hex in Svelte:\n" + "\n".join(violations[:20])


def test_ds02_constants_module_holds_the_literals():
    assert CONSTANTS.exists(), "themeCustomizerConstants.ts missing"
    src = _read(CONSTANTS)
    for name in ("SAMPLE_ACCENT_HEX", "SAMPLE_SECONDARY_HEX", "FALLBACK_BG_SURFACE"):
        assert f"export const {name}" in src, f"{name} not exported"


def test_ds02_customizer_uses_constants_and_live_surface():
    src = _read(CUSTOMIZER)
    assert "from './themeCustomizerConstants'" in src
    assert "placeholder={SAMPLE_ACCENT_HEX}" in src
    assert "placeholder={SAMPLE_SECONDARY_HEX}" in src
    # The contrast badge and the wheel cutout both read the live token.
    assert "function liveBgSurface()" in src
    assert src.count("liveBgSurface()") >= 3  # def + 2 call sites
    assert "getPropertyValue('--oo-bg-surface')" in src
    # The luminance parser needs #rrggbb; the helper guards for it.
    assert "/^#[0-9a-fA-F]{6}$/" in src
    # The stale hardcoded contrast pair is gone.
    assert "$darkMode ? '#222224'" not in src


# -- DS-03: re-assertions superseding the deselected stales --

THEME_CSS = FRONTEND_SRC / "lib" / "styles" / "theme.css"


def _theme_css() -> str:
    # theme.css location is stable, but resolve defensively.
    if THEME_CSS.exists():
        return _read(THEME_CSS)
    cands = list(FRONTEND_SRC.rglob("theme.css"))
    assert cands, "theme.css not found"
    return _read(cands[0])


def test_ds03_current_palette_anchors():
    css = _theme_css()
    # S166 foundations (supersede the s83 palette-value assertions).
    assert "#1F1F22" in css, "anthracite base missing"
    assert "#E5DECE" in css, "parchment base missing"
    for preset in ("anthracite", "parchment", "slate", "linen", "high-contrast"):
        assert f"theme-{preset}.css" in css, f"{preset} preset import missing"


def test_ds03_version_file_is_360():
    # Supersedes test_s152_theme_engine::test_version_is_3_2_1 (deselected).
    content = (ROOT / "opti_oignon" / "__version__.py").read_text(encoding="utf-8")
    assert '"3.6.0"' in content or "'3.6.0'" in content


# -- S169 prop-contract locks (verified in F10a; locked against drift) --


def test_contract_modal_onclose_callback_prop():
    src = _read(DS / "Modal.svelte")
    assert "export let onClose: () => void;" in src


def test_contract_card_forwards_no_arbitrary_attributes():
    src = _read(DS / "Card.svelte")
    assert "$$restProps" not in src and "{...$$props}" not in src
    assert "export { className as class }" in src


def test_contract_tabs_value_is_plain_string():
    src = _read(DS / "Tabs.svelte")
    assert "export let value: string;" in src


def test_contract_icon_resolves_kebab_to_pascal():
    src = _read(DS / "Icon.svelte")
    assert "function toPascal" in src
    assert "toPascal(name)" in src


def test_contract_select_consumers_keep_typeof_guards():
    consumers = {
        FRONTEND_SRC / "lib" / "components" / "panels" / "ProjectList.svelte":
            "typeof e.detail === 'string'",
        FRONTEND_SRC / "lib" / "components" / "panels" / "MemoriesPanel.svelte":
            "typeof value === 'string'",
        FRONTEND_SRC / "lib" / "components" / "health" / "CacheManager.svelte":
            "typeof e.detail === 'string'",
    }
    for f, guard in consumers.items():
        assert guard in _read(f), f"typeof guard missing in {f.name}"
