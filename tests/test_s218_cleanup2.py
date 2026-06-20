"""S218 cleanup lot 2 -- DS-01 drift fix locks + FRD-02 re-anchor pins.

Per-fix suite for the S218 insertable lot (the second cleanup lot, arbitrated
at the S218 gate after the gate check found no shakedown findings register):

- DS-01: the 67 var(--oo-*, #hex) fallback sites removed across the six
  Sandbox* panel components (9 alias-named primaries rewritten to their
  canonical theme.css targets: border-subtle -> bd-subtle, border-default ->
  bd-default, accent -> acc-500; 58 canonical primaries stripped of the dead
  fallback), restoring the S197 DS-01 lock tree-wide. Plus the 2 sibling
  sites found at the S218 gate in lib/api/keyCeremony.ts (a .ts file, outside
  the svelte+app.css scan scope of the S197 f10a / s83 locks): the fg-error /
  fg-warning alias fallbacks rewritten to var(--oo-error) / var(--oo-warning).
  Every removed fallback was dead by declaredness; rendering is identical.
- FRD-02 doc drift: the AgentPanel row in FRONTEND_REDESIGN_SPEC.md carries
  the truthful historical note (the FRD-02 residual S197, left with mount
  pending then, closed by the S210 mount), which names
  test_s197_f10b::test_frd02_agentpanel_residual_documented green again.
- Roadmap statuses rolled in ROADMAP_POST_AUDIT.md: DS-01 drift and the
  FRD-02 doc drift LANDED at S218; DOC-01 corrected to ABSORBED at S214
  (the eight pages joined the nav there; verified at the S218 gate).

The three named-green flips (f10a ds01, the s83 twin, the f10b frd02 pin)
live in adjacency suites outside the sweep selection; this file re-asserts
their invariants inside the sweep, and extends the DS-01 rule to the lib .ts
surface the original locks never scanned. All checks are source-level (no
node toolchain in the container).
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_SRC = ROOT / "frontend" / "src"
PANELS = FRONTEND_SRC / "lib" / "components" / "panels"
LIB = FRONTEND_SRC / "lib"
THEME_CSS = FRONTEND_SRC / "styles" / "theme.css"
APP_CSS = FRONTEND_SRC / "app.css"
KEY_CEREMONY = LIB / "api" / "keyCeremony.ts"
SPEC = ROOT / "FRONTEND_REDESIGN_SPEC.md"
ROADMAP = ROOT / "ROADMAP_POST_AUDIT.md"

# The f10a / s83 rule, verbatim.
FALLBACK_PAT = re.compile(r"var\(--oo-[^)]*#[0-9a-fA-F]")
HEX_PAT = re.compile(r"#[0-9a-fA-F]{3,8}\b")
NESTED_PAT = re.compile(r"var\(\s*--oo-[a-z0-9-]+\s*,\s*var\(")
# The four pre-S166 stale fallback anchors f10a hunts on the svelte side.
STALE_HEXES = ("#1a1a2e", "#dc2626", "#c85050", "#d97706")

SIX_PANELS = [
    "SandboxDiffReview.svelte",
    "SandboxSettingsStrip.svelte",
    "SandboxWorkspaceList.svelte",
    "SandboxUploadZone.svelte",
    "SandboxHostExplorer.svelte",
    "SandboxPanel.svelte",
]
SEVEN_PANELS = SIX_PANELS + ["SandboxFileManager.svelte"]


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _svelte_files():
    return sorted(FRONTEND_SRC.rglob("*.svelte"))


def _lib_ts_files():
    return sorted(LIB.rglob("*.ts"))


# -- DS-01: the svelte-side lock restored (the rule the flips assert) --


class TestDs01SvelteLockRestored:
    def test_no_fallback_svelte_tree(self):
        violations = []
        for f in list(_svelte_files()) + [APP_CSS]:
            for i, line in enumerate(_read(f).splitlines(), 1):
                if FALLBACK_PAT.search(line):
                    violations.append(f"{f.name}:{i}: {line.strip()[:90]}")
        assert violations == [], "var(--oo-*, #hex) fallbacks present:\n" + "\n".join(
            violations[:20]
        )

    @pytest.mark.parametrize("name", SEVEN_PANELS)
    def test_no_fallback_per_panel(self, name):
        hits = [
            f"{i}: {line.strip()[:90]}"
            for i, line in enumerate(_read(PANELS / name).splitlines(), 1)
            if FALLBACK_PAT.search(line)
        ]
        assert hits == [], f"{name} carries fallback forms:\n" + "\n".join(hits[:10])

    @pytest.mark.parametrize("name", SIX_PANELS)
    def test_six_panels_hex_free(self, name):
        # Pre-fix, every hex in the six panels was a fallback hex (audited at
        # the S218 read phase: total_hex == fallback_hex per file), so the fix
        # leaves them with zero hex literals of any kind.
        hits = HEX_PAT.findall(_read(PANELS / name))
        assert hits == [], f"{name} carries hex literals: {hits[:10]}"

    def test_stale_anchor_hexes_absent_svelte(self):
        hits = []
        for f in _svelte_files():
            content = _read(f)
            for h in STALE_HEXES:
                if h in content:
                    hits.append(f"{f.name}: {h}")
        assert hits == [], f"stale fallback hexes in svelte: {hits}"


# -- DS-01 extension: the lib .ts surface the original locks never scanned --


class TestDs01TsExtension:
    def test_no_fallback_lib_ts_tree(self):
        violations = []
        for f in _lib_ts_files():
            for i, line in enumerate(_read(f).splitlines(), 1):
                if FALLBACK_PAT.search(line):
                    violations.append(f"{f.name}:{i}: {line.strip()[:90]}")
        assert violations == [], "var(--oo-*, #hex) fallbacks in lib ts:\n" + "\n".join(
            violations[:20]
        )

    def test_keyceremony_no_fallback_form(self):
        assert not FALLBACK_PAT.search(_read(KEY_CEREMONY))

    def test_keyceremony_canonical_pair_present(self):
        content = _read(KEY_CEREMONY)
        assert "var(--oo-error)" in content
        assert "var(--oo-warning)" in content

    def test_keyceremony_alias_names_gone(self):
        content = _read(KEY_CEREMONY)
        assert "--oo-fg-error" not in content
        assert "--oo-fg-warning" not in content

    def test_stale_anchor_hexes_absent_lib_ts(self):
        hits = []
        for f in _lib_ts_files():
            content = _read(f)
            for h in STALE_HEXES:
                if h in content:
                    hits.append(f"{f.name}: {h}")
        assert hits == [], f"stale fallback hexes in lib ts: {hits}"


# -- The 9 alias rewrites: alias names gone, canonical targets present --


class TestAliasRewritesLanded:
    @pytest.mark.parametrize("alias", ["border-subtle", "border-default", "accent"])
    def test_panels_alias_free(self, alias):
        pat = re.compile(r"var\(--oo-" + re.escape(alias) + r"[,)]")
        offenders = [name for name in SIX_PANELS if pat.search(_read(PANELS / name))]
        assert offenders == [], f"alias --oo-{alias} still referenced in: {offenders}"

    @pytest.mark.parametrize(
        ("name", "token"),
        [
            ("SandboxDiffReview.svelte", "bd-default"),
            ("SandboxDiffReview.svelte", "bd-subtle"),
            ("SandboxSettingsStrip.svelte", "bd-subtle"),
            ("SandboxUploadZone.svelte", "acc-500"),
            ("SandboxHostExplorer.svelte", "acc-500"),
            ("SandboxHostExplorer.svelte", "bd-default"),
            ("SandboxHostExplorer.svelte", "bd-subtle"),
            ("SandboxWorkspaceList.svelte", "bd-subtle"),
            ("SandboxWorkspaceList.svelte", "acc-400"),
        ],
    )
    def test_rewritten_canonicals_present(self, name, token):
        assert f"var(--oo-{token})" in _read(PANELS / name)


# -- Token hygiene over the touched surface --


class TestTokenHygiene:
    def _surface(self):
        return [PANELS / n for n in SEVEN_PANELS] + [KEY_CEREMONY]

    def test_all_referenced_tokens_declared(self):
        declared = set(
            re.findall(r"(--oo-[a-z0-9-]+)\s*:", _read(THEME_CSS) + _read(APP_CSS))
        )
        missing = []
        for f in self._surface():
            for tok in sorted(set(re.findall(r"var\((--oo-[a-z0-9-]+)", _read(f)))):
                if tok not in declared:
                    missing.append(f"{f.name}: {tok}")
        assert missing == [], f"undeclared tokens referenced: {missing}"

    def test_no_nested_forms(self):
        offenders = [f.name for f in self._surface() if NESTED_PAT.search(_read(f))]
        assert offenders == [], f"nested var() fallbacks present: {offenders}"


# -- FRD-02: the truthful historical note, pinned and co-located --


class TestFrd02ReAnchor:
    def test_residual_string_present(self):
        assert "FRD-02 residual S197" in _read(SPEC)

    def test_mount_pending_string_present(self):
        assert "mount pending" in _read(SPEC)

    def test_history_colocated_in_agentpanel_row(self):
        rows = [
            line
            for line in _read(SPEC).splitlines()
            if "`AgentPanel.svelte` | NEW" in line
        ]
        assert len(rows) == 1, f"expected one AgentPanel NEW row, got {len(rows)}"
        row = rows[0]
        for needle in (
            "FRD-02 residual S197",
            "mount pending",
            "S210",
            "FRD-03 closed",
        ):
            assert needle in row, f"AgentPanel row lacks {needle!r}"


# -- Roadmap statuses rolled --


def _bullet(marker: str) -> str:
    bullets = [b for b in re.split(r"\n(?=- )", _read(ROADMAP)) if b.startswith(f"- {marker}")]
    assert len(bullets) == 1, f"expected one {marker!r} bullet, got {len(bullets)}"
    return bullets[0]


class TestRoadmapStatuses:
    def test_ds01_drift_landed_s218(self):
        assert "LANDED at S218" in _bullet("DS-01 drift")

    def test_frd02_doc_drift_landed_s218(self):
        assert "LANDED at S218" in _bullet("FRD-02 doc drift")

    def test_doc01_absorbed_s214(self):
        assert "ABSORBED at S214" in _bullet("DOC-01")


# -- Svelte integrity of the six touched panels --


class TestSvelteIntegrity:
    @pytest.mark.parametrize("name", SIX_PANELS)
    def test_tag_balance(self, name):
        content = _read(PANELS / name)
        assert content.count("<script") == content.count("</script")
        assert content.count("<style") == content.count("</style")

    @pytest.mark.parametrize("name", SIX_PANELS)
    def test_block_balance(self, name):
        content = _read(PANELS / name)
        for opener, closer in (
            (r"\{#if ", "{/if}"),
            (r"\{#each ", "{/each}"),
            (r"\{#await ", "{/await}"),
            (r"\{#key ", "{/key}"),
        ):
            assert len(re.findall(opener, content)) == content.count(closer)


# -- The suite parses --


class TestAstValidity:
    def test_suite_parses(self):
        ast.parse(Path(__file__).read_text(encoding="utf-8"))
