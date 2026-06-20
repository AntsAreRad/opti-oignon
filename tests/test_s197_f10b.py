"""S197 F10b -- shells / panels / dashboards + keyboard shortcuts + cartography.

Per-fix tests for the F10b lot:
- KS-01: preventDefault only on handled shortcut paths (unhandled Escape keeps
  its default action, which fires the native <dialog> cancel ds Modals use).
- KS-02: '?' matched shift-agnostically, before the modifier gate.
- KS-03: case-insensitive key comparison (custom overrides arrive lowercased).
- SHL-01: the right panel <aside> establishes a positioning context on
  desktop so the absolute resize handle anchors to the panel, not the window.
- SHL-02: PanelType pruned of its two dead members.
Plus the two-way cartography lock, the frontend/backend shortcut-defaults
parity lock, and the deselect-plus-reassert supersessions owned by F10b
(frontend_performance apiClient symbol; s153 version pin). All checks are
source-level. Runtime confirmation of KS-01 (Esc closes every Modal) and
SHL-01 (handle sits on the panel edge) is flagged for the live shakedown.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_SRC = ROOT / "frontend" / "src"
KS = FRONTEND_SRC / "lib" / "components" / "ui" / "KeyboardShortcuts.svelte"
APPSHELL = FRONTEND_SRC / "lib" / "components" / "layout" / "AppShell.svelte"
TYPES = FRONTEND_SRC / "lib" / "types.ts"
SPEC = ROOT / "FRONTEND_REDESIGN_SPEC.md"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# -- KS-01: conditional preventDefault --


def test_ks01_unhandled_escape_keeps_default():
    src = _read(KS)
    # close_dialog only preventDefaults under showHelp.
    assert re.search(
        r"if \(s\.action === 'close_dialog'\) \{\s*if \(showHelp\) \{\s*"
        r"e\.preventDefault\(\);\s*closeHelp\(\);",
        src,
    ), "close_dialog must gate preventDefault on showHelp"
    # The handler path preventDefaults only when a handler is bound.
    assert re.search(
        r"if \(handler\) \{\s*e\.preventDefault\(\);\s*handler\(\);", src
    ), "handler path must preventDefault only when bound"
    # The old unconditional shape (preventDefault before the action switch)
    # is gone: every preventDefault now sits inside a handled branch.
    body = src[src.index("// Find matching shortcut"):]
    first_pd = body.index("e.preventDefault()")
    first_branch = body.index("if (s.action === 'show_shortcuts')")
    assert first_pd > first_branch, "preventDefault precedes the action switch"


# -- KS-02: '?' shift-agnostic, ahead of the modifier gate --


def test_ks02_question_mark_matches_with_shift():
    src = _read(KS)
    early = re.search(
        r"if \(s\.key === '\?'\) \{\s*return e\.key === '\?' && "
        r"!e\.ctrlKey && !e\.metaKey && !e\.altKey;",
        src,
    )
    assert early, "'?' must match on e.key alone (shift-agnostic)"
    gate = src.index("if (ctrl !== (e.ctrlKey || e.metaKey))")
    assert early.start() < gate, "'?' special case must precede the gate"


# -- KS-03: case-insensitive comparison --


def test_ks03_case_insensitive_key_match():
    src = _read(KS)
    assert "e.key.toLowerCase() === s.key.toLowerCase()" in src
    # The case-sensitive special cases are superseded.
    assert "if (s.key === 'Enter') return e.key === 'Enter';" not in src
    assert "if (s.key === 'Escape') return e.key === 'Escape';" not in src


# -- SHL-01: panel aside positioning context on desktop --


def test_shl01_aside_relative_on_desktop():
    src = _read(APPSHELL)
    assert ": 'relative'}" in src, "desktop aside branch must be 'relative'"
    # The handle is still the absolutely positioned child it anchors.
    assert 'aria-label="Resize panel"' in src
    # Mobile branch keeps 'fixed' (relative must NOT coexist with it:
    # Tailwind emits .relative after .fixed, so it would win on mobile).
    assert "'fixed inset-y-0 right-0 z-50" in src
    mobile = re.search(r"\? '([^']*)'\s*\n\s*: 'relative'\}", src)
    assert mobile and "relative" not in mobile.group(1)


# -- SHL-02: PanelType pruned of dead members --


def test_shl02_paneltype_pruned():
    src = _read(TYPES)
    line = next(l for l in src.splitlines() if "export type PanelType" in l)
    assert "'consensus'" not in line and "'analytics'" not in line
    for member in ("'none'", "'artifacts'", "'code'", "'memory'",
                   "'pipelines'", "'context'", "'exec-pipelines'", "'plugins'"):
        assert member in line, f"{member} missing from PanelType"


def test_shl02_host_and_toggles_cover_every_openable_panel():
    # Every non-'none' PanelType member has a toggle opener and a host branch.
    types_line = next(
        l for l in _read(TYPES).splitlines() if "export type PanelType" in l
    )
    members = set(re.findall(r"'([a-z-]+)'", types_line)) - {"none"}
    toggles = _read(
        FRONTEND_SRC / "lib" / "components" / "panels" / "PanelToggle.svelte"
    )
    host = _read(FRONTEND_SRC / "routes" / "chat" / "+layout.svelte")
    for m in sorted(members):
        assert f"togglePanel('{m}')" in toggles, f"no toggle for panel '{m}'"
        assert f"$activePanel === '{m}'" in host, f"no host branch for '{m}'"


# -- Cartography lock (two-way) --

RETIRED = {
    "BranchDiff.svelte", "ScrollToBottom.svelte", "CodingAgentPanel.svelte",
    "CascadingIndicator.svelte", "ConsensusPanel.svelte", "ModelManager.svelte",
}


def test_cartography_every_component_registered():
    spec = _read(SPEC)
    missing = sorted(
        p.name
        for p in (FRONTEND_SRC / "lib").rglob("*.svelte")
        if p.name not in spec
    )
    assert missing == [], f"components on disk not in the spec: {missing}"


def test_cartography_no_live_ghosts():
    spec = _read(SPEC)
    on_disk = {p.name for p in FRONTEND_SRC.rglob("*.svelte")}
    mentioned = set(re.findall(r"(?<![+\w])([A-Z][A-Za-z0-9_]*\.svelte)", spec))
    ghosts = sorted(m for m in mentioned if m not in on_disk)
    unexplained = [g for g in ghosts if g not in RETIRED]
    assert unexplained == [], f"spec references non-RETIRE ghosts: {unexplained}"
    # And no live code imports a retired component.
    for f in list(FRONTEND_SRC.rglob("*.svelte")) + list(FRONTEND_SRC.rglob("*.ts")):
        content = _read(f)
        for g in RETIRED:
            stem = g.removesuffix(".svelte")
            assert f"/{stem}.svelte" not in content, f"{f.name} imports retired {g}"


# -- Frontend/backend shortcut-defaults parity --


def test_shortcut_defaults_parity():
    fe = _read(KS)
    fe_actions = set(re.findall(r"action: '([a-z_]+)'", fe))
    be = (ROOT / "opti_oignon" / "keyboard_shortcuts.py").read_text(
        encoding="utf-8"
    )
    be_actions = set(re.findall(r'action="([a-z_]+)"', be))
    assert fe_actions == be_actions, (
        f"frontend/backend default actions diverge: "
        f"fe-only={fe_actions - be_actions}, be-only={be_actions - fe_actions}"
    )


# -- Supersessions owned by F10b (deselect-plus-reassert) --


def test_performance_api_goes_through_shared_client():
    # Supersedes test_frontend_performance::test_uses_api_client (deselected):
    # the module was refactored from an 'apiClient' object onto the shared
    # apiGet/apiPost wrappers; the INTENT (shared client, no raw fetch) holds.
    src = _read(FRONTEND_SRC / "lib" / "api" / "performance.ts")
    assert "from './client'" in src
    assert "apiGet" in src or "apiPost" in src
    assert "fetch(" not in src


def test_version_file_is_360_f10b():
    # Supersedes test_s153_shortcuts_a11y::test_version_is_3_2_2 (deselected).
    content = (ROOT / "opti_oignon" / "__version__.py").read_text(
        encoding="utf-8"
    )
    assert '"3.6.0"' in content or "'3.6.0'" in content


# -- FRD-02: the three management panels are wired into Settings --


def test_frd02_panels_wired_into_settings():
    src = _read(FRONTEND_SRC / "routes" / "settings" / "+page.svelte")
    # Loader map entries
    for name, path in (
        ("MemoriesPanel", "$lib/components/panels/MemoriesPanel.svelte"),
        ("SkillsPanel", "$lib/components/panels/SkillsPanel.svelte"),
        ("SyncPanel", "$lib/components/panels/SyncPanel.svelte"),
    ):
        assert f"{name}: () => import('{path}')" in src, f"loader missing: {name}"
    # Registry rows, in their arbitrated sections
    assert re.search(
        r"id: 'memories'.*panel: 'MemoriesPanel'", src
    ), "memories registry row missing"
    assert re.search(r"id: 'skills'.*panel: 'SkillsPanel'", src)
    assert re.search(r"id: 'device-sync'.*panel: 'SyncPanel'", src)
    # Section placement: each row sits inside its section's groups block.
    conv = src.index("label: 'Conversation & Chat'")
    models = src.index("label: 'Models & Inference'")
    assert conv < src.index("panel: 'MemoriesPanel'") < models
    plugins = src.index("label: 'Plugins & Extensions'")
    perf = src.index("label: 'Performance & Telemetry'")
    assert plugins < src.index("panel: 'SkillsPanel'") < perf
    network = src.index("label: 'Network & Privacy'")
    data = src.index("label: 'Backup & Data'")
    assert network < src.index("panel: 'SyncPanel'") < data


def test_frd02_agentpanel_residual_documented():
    spec = _read(SPEC)
    assert "FRD-02 residual S197" in spec
    assert "mount pending" in spec


# -- FRD-01: spec dispositions no longer lie about the six dead components --


def test_frd01_spec_dispositions_corrected():
    spec = _read(SPEC)
    assert spec.count("removal recorded FRD-01 (S197)") == 6
    # The three never-performed REINTEGRATEs are marked as such.
    assert "Not performed: login kept its inline logic" in spec
    assert "reimplemented the surface instead of reintegrating" in spec


def test_frd01_ds_toast_is_the_mounted_one():
    # Supersedes s107::test_toast_still_uses_correct_imports (deselected):
    # the layout mounts ds/Toast, which wires to the notifications store
    # through the current dismissToast API.
    layout = _read(FRONTEND_SRC / "routes" / "+layout.svelte")
    assert "import Toast from '$lib/ds/Toast.svelte';" in layout
    ds_toast = _read(FRONTEND_SRC / "lib" / "ds" / "Toast.svelte")
    assert "from '$lib/stores/notifications'" in ds_toast
    assert "toasts" in ds_toast and "dismissToast" in ds_toast
