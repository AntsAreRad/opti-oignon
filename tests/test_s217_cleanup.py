"""S217 cleanup batch per-fix suite (FRD-01 + DS-04 + DS-05).

Covers, in order:
- FRD-01: the six dead superseded component files are deleted and nothing in
  live code references their paths.
- Replacement intent: the surfaces that superseded the six (ds/Toast,
  ThemeSwitcher, BackendStatus, ToolCallApprovalDrawer, the settings 2FA
  setup pair) exist, are mounted where relevant, and carry the behaviour the
  deselected content tests used to pin on the dead files.
- Cartography lock re-asserted (supersedes, deselected in pyproject:
  tests/test_s197_f10b.py::test_cartography_no_live_ghosts): the original
  two-way ghost rule with the five S217 deletions added to the RETIRED set.
  ui/Toast.svelte never enters the ghost set because ghosts are keyed by
  file NAME and ds/Toast.svelte keeps "Toast.svelte" on disk.
- Spec dispositions counter-pin (supersedes, deselected:
  tests/test_s197_f10b.py::test_frd01_spec_dispositions_corrected): the six
  FRONTEND_REDESIGN_SPEC.md rows now record the S217 deletion; the honesty
  phrasing about the never-performed REINTEGRATEs is preserved.
- DS-04: zero nested var(--oo-X, var(--oo-Y)) fallbacks remain anywhere in
  frontend/src (83 removed at S217: the 52 audit-family occurrences plus 31
  newer ones); every --oo- token referenced by the touched files is declared
  in the style layer; spot anchors prove the option-B canonical rewrites.
- DS-05: ds/Icon warns in dev on an unresolved lucide name; the render
  contract (no element on unresolved) is unchanged.
- Structural integrity (tag/block balance) on the 20 touched Svelte files,
  the 14 S217 pyproject deselect pins, and AST validity of this file.

Red-before (pristine proof): the FRD-01 absence tests, the DS-04
zero-nested and spot-anchor "old form absent" clauses, the spec counter-pin,
the DS-05 warn, and the pyproject pins are red by construction on the
incoming tree. The cartography re-assertion, the replacement-intent tests
and the integrity checks hold there too (the replacements predate S217);
that split is expected and mirrors the S216 precedent.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_SRC = ROOT / "frontend" / "src"
STYLES = FRONTEND_SRC / "styles"
SPEC = ROOT / "FRONTEND_REDESIGN_SPEC.md"
PYPROJECT = ROOT / "pyproject.toml"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _live_sources():
    return list(FRONTEND_SRC.rglob("*.svelte")) + list(FRONTEND_SRC.rglob("*.ts"))


DELETED = [
    "lib/components/ui/Toast.svelte",
    "lib/components/ui/ThemeToggle.svelte",
    "lib/components/chat/NetworkIndicator.svelte",
    "lib/components/chat/ToolCallApproval.svelte",
    "lib/components/auth/TOTPInput.svelte",
    "lib/components/auth/WebAuthnChallenge.svelte",
]

# Path keys, not bare names: "ToolCallApproval.svelte" alone would match the
# live ToolCallApprovalDrawer import path's prefix in no case, but the bare
# name "NetworkIndicator" appears in an honest supersession comment inside
# BackendStatus; the path form is what an import would need.
DELETED_PATH_KEYS = [
    "components/ui/Toast.svelte",
    "components/ui/ThemeToggle.svelte",
    "components/chat/NetworkIndicator.svelte",
    "components/chat/ToolCallApproval.svelte",
    "components/auth/TOTPInput.svelte",
    "components/auth/WebAuthnChallenge.svelte",
]


class TestFRD01Absence:
    @pytest.mark.parametrize("rel", DELETED)
    def test_deleted(self, rel):
        assert not (FRONTEND_SRC / rel).exists(), f"{rel} should be deleted"

    def test_auth_components_dir_empty_or_gone(self):
        d = FRONTEND_SRC / "lib" / "components" / "auth"
        assert (not d.exists()) or list(d.iterdir()) == []

    @pytest.mark.parametrize("key", DELETED_PATH_KEYS)
    def test_no_live_path_references(self, key):
        offenders = [
            str(f) for f in _live_sources() if key in _read(f)
        ]
        assert offenders == [], f"live code references deleted {key}: {offenders}"


class TestReplacementsCarryIntent:
    def test_ds_toast_mounted_in_root_layout(self):
        layout = _read(FRONTEND_SRC / "routes" / "+layout.svelte")
        assert "import Toast from '$lib/ds/Toast.svelte';" in layout
        assert "<Toast" in layout

    def test_theme_switcher_carries_the_toggle(self):
        # Supersedes the deselected ux_improvements_s107::TestThemeToggle
        # content tests: the binary toggle lives inside ThemeSwitcher.
        src = _read(FRONTEND_SRC / "lib" / "components" / "ui" / "ThemeSwitcher.svelte")
        assert "function toggle()" in src
        assert "aria-label" in src
        assert "var(--oo-" in src

    def test_theme_switcher_no_raw_hex(self):
        # Supersedes the deselected
        # bugfix_s108::test_theme_toggle_uses_css_variables on the live
        # surface: tokens only, no raw hex outside the var fallback form.
        src = _read(FRONTEND_SRC / "lib" / "components" / "ui" / "ThemeSwitcher.svelte")
        masked = re.sub(r"var\(--oo-[a-z0-9-]+\s*,\s*#[0-9a-fA-F]{3,8}\)", "var(M)", src)
        assert re.findall(r"#[0-9a-fA-F]{3,8}\b", masked) == []

    def test_backend_status_carries_network(self):
        # Supersedes the deselected
        # routes_network::test_network_indicator_exists: the merge target
        # owns the network concern.
        src = _read(FRONTEND_SRC / "lib" / "components" / "ui" / "BackendStatus.svelte")
        assert "merged from NetworkIndicator" in src
        assert "/api/network/status" in src

    def test_drawer_is_the_approval_surface(self):
        # Supersedes the deselected s128 TestToolCallApprovalComponent
        # content tests: approve/deny and the risk display live in the
        # Drawer, the surface that reimplemented the card.
        src = _read(
            FRONTEND_SRC / "lib" / "components" / "chat" / "ToolCallApprovalDrawer.svelte"
        )
        assert "approveToolCall" in src
        assert "denyToolCall" in src
        assert "risk_level" in src

    def test_drawer_mounted_in_chat_layout(self):
        layout = _read(FRONTEND_SRC / "routes" / "chat" / "+layout.svelte")
        assert "ToolCallApprovalDrawer.svelte" in layout
        assert "<ToolCallApprovalDrawer />" in layout

    def test_inline_card_retired_drawer_lives(self):
        # Counter-pin for the deselected
        # s169::test_inline_approval_card_preserved: the preserved-card
        # assertion is superseded by the card's deletion; the Drawer is the
        # one approval surface.
        assert not (
            FRONTEND_SRC / "lib" / "components" / "chat" / "ToolCallApproval.svelte"
        ).exists()
        layout = _read(FRONTEND_SRC / "routes" / "chat" / "+layout.svelte")
        assert "<ToolCallApprovalDrawer />" in layout

    def test_live_2fa_surfaces_present(self):
        # Supersedes the deselected s127 content tests on the never-imported
        # login widgets: the live 2FA surfaces are the settings setup pair;
        # the auth core (untouched this session) carries the flow.
        assert (FRONTEND_SRC / "lib" / "components" / "settings" / "TOTPSetup.svelte").exists()
        assert (
            FRONTEND_SRC / "lib" / "components" / "settings" / "WebAuthnSetup.svelte"
        ).exists()
        settings = _read(FRONTEND_SRC / "routes" / "settings" / "+page.svelte")
        assert "TOTPSetup" in settings and "WebAuthnSetup" in settings
        assert (ROOT / "opti_oignon" / "auth_2fa.py").exists()


# The original S197 set, verbatim from tests/test_s197_f10b.py.
RETIRED_S197 = {
    "BranchDiff.svelte", "ScrollToBottom.svelte", "CodingAgentPanel.svelte",
    "CascadingIndicator.svelte", "ConsensusPanel.svelte", "ModelManager.svelte",
}
# The five S217 deletions that become spec-mentioned ghosts. ui/Toast.svelte
# is absent here by design: ghosts are keyed by file name and ds/Toast.svelte
# keeps "Toast.svelte" on disk.
RETIRED_S217 = {
    "ThemeToggle.svelte", "NetworkIndicator.svelte", "ToolCallApproval.svelte",
    "TOTPInput.svelte", "WebAuthnChallenge.svelte",
}
RETIRED = RETIRED_S197 | RETIRED_S217


class TestCartographyReasserted:
    def test_no_unexplained_ghosts(self):
        # Supersedes (deselected) f10b::test_cartography_no_live_ghosts with
        # the same rule and the S217 additions.
        spec = _read(SPEC)
        on_disk = {p.name for p in FRONTEND_SRC.rglob("*.svelte")}
        mentioned = set(re.findall(r"(?<![+\w])([A-Z][A-Za-z0-9_]*\.svelte)", spec))
        ghosts = sorted(m for m in mentioned if m not in on_disk)
        unexplained = [g for g in ghosts if g not in RETIRED]
        assert unexplained == [], f"spec references non-RETIRE ghosts: {unexplained}"

    def test_s217_ghosts_are_real_ghosts(self):
        # The five must actually be off disk (guards the set against rot).
        on_disk = {p.name for p in FRONTEND_SRC.rglob("*.svelte")}
        still_here = sorted(RETIRED_S217 & on_disk)
        assert still_here == [], f"S217-retired names still on disk: {still_here}"

    def test_no_live_imports_of_retired(self):
        # The original clause, over the grown set, path-keyed by "/Name.svelte".
        offenders = []
        for f in _live_sources():
            content = _read(f)
            for g in sorted(RETIRED):
                stem = g.removesuffix(".svelte")
                if f"/{stem}.svelte" in content:
                    offenders.append(f"{f.name} -> {g}")
        assert offenders == [], f"live code imports retired components: {offenders}"


NEW_ANNOTATION = "deleted at S217 (FRD-01 landed; absence locked by tests/test_s217_cleanup.py)"
OLD_ANNOTATION = "removal recorded FRD-01 (S197)"


class TestSpecDispositions:
    def test_six_rows_record_the_s217_deletion(self):
        # Supersedes (deselected) f10b::test_frd01_spec_dispositions_corrected.
        spec = _read(SPEC)
        assert spec.count(NEW_ANNOTATION) == 6
        assert spec.count(OLD_ANNOTATION) == 0

    def test_honesty_phrases_preserved(self):
        spec = _read(SPEC)
        assert "Not performed: login kept its inline logic" in spec
        assert "reimplemented the surface instead of reintegrating" in spec


NESTED = re.compile(r"var\(\s*--oo-[a-z0-9-]+\s*,\s*var\(\s*--oo-[a-z0-9-]+\s*\)\s*\)")

DS04_FILES = [
    "lib/components/chat/BranchExplorer.svelte",
    "lib/components/chat/ChatControlBar.svelte",
    "lib/components/chat/ChatInput.svelte",
    "lib/components/chat/ChatMessage.svelte",
    "lib/components/chat/CodingAgentInline.svelte",
    "lib/components/chat/CodingAgentProgress.svelte",
    "lib/components/panels/BenchmarkRunner.svelte",
    "lib/components/panels/EventTimeline.svelte",
    "lib/components/panels/ObservabilityPanel.svelte",
    "lib/components/panels/ProfilerDashboard.svelte",
    "lib/components/panels/TelemetryDashboard.svelte",
    "lib/components/panels/TelemetryHistoryPanel.svelte",
    "lib/components/settings/ContextOptimizerPanel.svelte",
    "lib/components/settings/PluginAllowlistPanel.svelte",
    "lib/components/settings/SearchKillSwitchPanel.svelte",
    "lib/components/settings/SecurityModePanel.svelte",
    "lib/ds/Input.svelte",
    "lib/ds/Select.svelte",
    "routes/chat/[id]/+page.svelte",
]

TOUCHED_SVELTE = DS04_FILES + ["lib/ds/Icon.svelte"]


def _declared_tokens() -> set:
    decl = re.compile(r"--oo-([a-z0-9-]+)\s*:")
    out = set()
    for css in list(STYLES.glob("*.css")) + [FRONTEND_SRC / "app.css"]:
        out |= set(decl.findall(_read(css)))
    return out


class TestDS04NestedFallbacks:
    def test_zero_nested_var_fallbacks_tree_wide(self):
        offenders = []
        for f in list(FRONTEND_SRC.rglob("*.svelte")) + list(
            FRONTEND_SRC.rglob("*.css")
        ) + list(FRONTEND_SRC.rglob("*.ts")):
            for m in NESTED.finditer(_read(f)):
                offenders.append(f"{f.name}: {m.group(0)[:70]}")
        assert offenders == [], "nested var() fallbacks remain:\n" + "\n".join(offenders[:20])

    def test_all_tokens_used_by_touched_files_are_declared(self):
        declared = _declared_tokens()
        used = set()
        for rel in TOUCHED_SVELTE:
            used |= set(re.findall(r"var\(--oo-([a-z0-9-]+)", _read(FRONTEND_SRC / rel)))
        undeclared = sorted(used - declared)
        assert undeclared == [], f"touched files use undeclared tokens: {undeclared}"

    @pytest.mark.parametrize(
        "token", ["bd-subtle", "fg-on-accent", "acc-400", "error", "bd-strong"]
    )
    def test_canonical_targets_declared(self, token):
        assert token in _declared_tokens()

    def test_spot_anchor_profiler_bd_subtle(self):
        src = _read(FRONTEND_SRC / "lib" / "components" / "panels" / "ProfilerDashboard.svelte")
        assert src.count("var(--oo-bd-subtle)") >= 8
        assert "var(--oo-border-subtle, var(" not in src

    def test_spot_anchor_inputs_focus(self):
        for rel in ("lib/ds/Input.svelte", "lib/ds/Select.svelte"):
            src = _read(FRONTEND_SRC / rel)
            assert "var(--oo-input-focus)" in src
            assert "var(--oo-input-focus, var(" not in src

    def test_spot_anchor_history_fg_on_accent(self):
        src = _read(
            FRONTEND_SRC / "lib" / "components" / "panels" / "TelemetryHistoryPanel.svelte"
        )
        assert src.count("var(--oo-fg-on-accent)") >= 2
        assert "var(--oo-text-on-accent, var(" not in src


class TestDS05IconWarning:
    def test_dev_warning_present(self):
        src = _read(FRONTEND_SRC / "lib" / "ds" / "Icon.svelte")
        assert "import.meta.env.DEV" in src
        assert "console.warn" in src
        assert "[ds/Icon] unresolved icon name" in src

    def test_render_contract_unchanged(self):
        src = _read(FRONTEND_SRC / "lib" / "ds" / "Icon.svelte")
        assert "{#if Cmp}" in src
        assert "{:else}" not in src  # still renders nothing on unresolved


class TestSvelteIntegrity:
    @pytest.mark.parametrize("rel", TOUCHED_SVELTE)
    def test_blocks_balanced(self, rel):
        src = _read(FRONTEND_SRC / rel)
        for opener, closer in (
            ("{#if", "{/if}"),
            ("{#each", "{/each}"),
            ("{#await", "{/await}"),
            ("{#key", "{/key}"),
        ):
            assert src.count(opener) == src.count(closer), f"{rel}: {opener} unbalanced"
        for tag in ("script", "style"):
            assert len(re.findall(rf"<{tag}[ >]", src)) == src.count(
                f"</{tag}>"
            ), f"{rel}: <{tag}> unbalanced"


S217_DESELECTS = [
    "--deselect=tests/test_ux_improvements_s107.py::TestThemeToggle",
    "--deselect=tests/test_bugfix_s108.py::TestBug13LightModePolish::test_theme_toggle_uses_css_variables",
    "--deselect=tests/test_routes_network.py::TestFrontendFiles::test_network_indicator_exists",
    "--deselect=tests/test_s127_2fa_login_flow.py::TestFrontendComponents::test_component_exists[component_path1]",
    "--deselect=tests/test_s127_2fa_login_flow.py::TestFrontendComponents::test_component_exists[component_path3]",
    "--deselect=tests/test_s127_2fa_login_flow.py::TestFrontendComponents::test_totp_input_auto_submits",
    "--deselect=tests/test_s127_2fa_login_flow.py::TestFrontendComponents::test_totp_input_has_recovery_option",
    "--deselect=tests/test_s127_2fa_login_flow.py::TestFrontendComponents::test_webauthn_challenge_has_fallback",
    "--deselect=tests/test_s128_killswitch_ui_tool_approval.py::TestToolCallApprovalComponent::test_component_valid",
    "--deselect=tests/test_s128_killswitch_ui_tool_approval.py::TestToolCallApprovalComponent::test_countdown_timer",
    "--deselect=tests/test_s128_killswitch_ui_tool_approval.py::TestToolCallApprovalComponent::test_risk_level_display",
    "--deselect=tests/test_s169_specialized_views.py::TestChatSurface::test_inline_approval_card_preserved",
    "--deselect=tests/test_s197_f10b.py::test_cartography_no_live_ghosts",
    "--deselect=tests/test_s197_f10b.py::test_frd01_spec_dispositions_corrected",
]


class TestSupersessionPins:
    @pytest.mark.parametrize("flag", S217_DESELECTS, ids=range(len(S217_DESELECTS)))
    def test_pyproject_carries_the_deselect(self, flag):
        # Delimiter-anchored: "::TestThemeToggle" must not be satisfied by the
        # pre-existing "::TestThemeToggleInAppShell" deselects via substring.
        assert re.search(re.escape(flag) + r'[\s"]', _read(PYPROJECT))


class TestASTValidity:
    def test_this_suite_parses(self):
        ast.parse(Path(__file__).read_text(encoding="utf-8"))
