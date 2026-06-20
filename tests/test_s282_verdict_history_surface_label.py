#!/usr/bin/env python3
"""S282 -- the verdict-history surface-label surfacing (the shared sub-component).

The container-provable half of a labelled, host-assured walk: the S280 fan-out
fed all three verifier surfaces (the S277 ``AnswerVerifier`` on
``/verify-answer``, the S269 ``ClaimVerifier`` on ``/claims``, the S275
``CitationVerifier`` on ``/verify-citations``) into one shared, session-local
verdict history, but the shared sub-component ``VerdictHistory.svelte`` rendered
each entry without its surface label, so the commingled list could not tell an
answer verdict from a claim or a citation verdict (the S280 walk recorded the
list as "distinguished by nothing visible today", the surface field reserved for
exactly this surfacing). This lot surfaces it: the shared sub-component gains a
per-entry surface badge that renders ``entry.surface``, so a verdict run on one
surface is distinguishable on the others. The edit is additive and lands in the
one shared sub-component, so the affordance reaches all three surfaces at once
and the supersession surface is exactly that one component.

A clean-twin add-only lot -- the edit flips no canonical pin (the S279 and S280
sub-component pins are presence and conformance, never the absence of the
badge), the failing-name register stays byte-identical (the fifteenth twin of
the S266 register), the addopts ledger holds at 248, and the version holds at
3.13.0.

Seven families, the S279 / S280 idiom:

1. the surface-label affordance -- ``VerdictHistory.svelte``: the rendered
   ``entry.surface`` label, the dedicated badge class, the badge style block
   (RED on the pristine S281 tree), plus the preserved conformance (no raw hex,
   block-balanced, pure ASCII, design-green) and the preserved surface-agnostic
   shape (no hardcoded surface literal; design-green);
2. the preserved S279 surface -- the store subscription, the ds primitives, the
   Clear control, the per-entry render, the empty state (design-green);
3. the preserved S280 fan-out -- all three verifier surfaces still render and
   import the shared sub-component (design-green);
4. the host-assured runbook -- ``VERDICT_HISTORY_SURFACE_LABEL_E2E_S282.md``:
   existence, status, the container-vs-host split, the required sections, the
   companions, the auth-core edit-free note, pure ASCII (content RED on
   pristine, the ascii pin design-green because an absent file reads empty);
5. the held version -- 3.13.0 in both anchors (design-green);
6. the clean-twin posture -- no cartography row added (the s217 annotation
   counts intact at 6 and 0), the addopts ledger held at 248, and the shared
   store left unedited (design-green);
7. suite structure -- the suite parses, is pure ASCII, and avoids the canonical
   selection literal (built only in split form) so the selection grep raw count
   is unchanged.

All read helpers return an empty string on absence, so a missing artifact yields
a bare assertion failure (never a collection or exception failure): the
red-before is a clean ``failures`` set with zero errors and zero skips.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_SRC = ROOT / "frontend" / "src"
PANELS = FRONTEND_SRC / "lib" / "components" / "panels"
STORE = FRONTEND_SRC / "lib" / "stores" / "verdictHistory.ts"
SUBCOMPONENT = PANELS / "VerdictHistory.svelte"
ANSWER_VERIFIER = PANELS / "AnswerVerifier.svelte"
CLAIM_VERIFIER = PANELS / "ClaimVerifier.svelte"
CITATION_VERIFIER = PANELS / "CitationVerifier.svelte"
SPEC = ROOT / "FRONTEND_REDESIGN_SPEC.md"
RUNBOOK = ROOT / "VERDICT_HISTORY_SURFACE_LABEL_E2E_S282.md"
PYPROJECT = ROOT / "pyproject.toml"
VERSION = ROOT / "opti_oignon" / "__version__.py"
OWN = Path(__file__).resolve()

# Mirror of the test_s217 cartography annotation counters (the cartography
# lock): a clean-twin lot adds no component row, so these stay 6 and 0.
NEW_ANNOTATION = "deleted at S217 (FRD-01 landed; absence locked by tests/test_s217_cleanup.py)"
OLD_ANNOTATION = "removal recorded FRD-01 (S197)"

# The expected addopts ledger size (one long line in pyproject.toml).
ADDOPTS_DESELECTS = 248


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return ""


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _is_ascii(text: str) -> bool:
    return all(ord(ch) < 128 for ch in text)


def _strip_token_fallbacks(text: str) -> str:
    return re.sub(r"var\(--oo-[^)]*\)", "", text)


def _has_raw_hex(text: str) -> bool:
    return re.search(r"#[0-9a-fA-F]{3,8}\b", _strip_token_fallbacks(text)) is not None


def _block_balanced(text: str) -> bool:
    for blk in ("if", "each", "await", "key"):
        if text.count("{#" + blk) != text.count("{/" + blk + "}"):
            return False
    if text.count("<script") != text.count("</script>"):
        return False
    if text.count("<style") != text.count("</style>"):
        return False
    return True


# ---------------------------------------------------------------------------
# Family 1 -- the surface-label affordance (content RED, conformance design-green)
# ---------------------------------------------------------------------------


class TestSurfaceLabelAffordance:
    def test_subcomponent_exists(self):
        assert _read(SUBCOMPONENT)

    def test_subcomponent_renders_surface_label(self):
        # The shared sub-component now renders the entry's surface, so the
        # commingled list distinguishes which verifier produced each verdict.
        assert "entry.surface" in _read(SUBCOMPONENT)

    def test_subcomponent_has_surface_badge_class(self):
        # The surface label rides a dedicated class, so it reads as a badge and
        # not as another inline verdict word.
        assert "vh-surface" in _read(SUBCOMPONENT)

    def test_subcomponent_surface_badge_styled(self):
        # The badge carries its own style block (a scoped selector), so the
        # label is a visually distinct affordance rather than bare text.
        assert ".vh-surface" in _read(SUBCOMPONENT)

    def test_subcomponent_no_raw_hex(self):
        assert not _has_raw_hex(_read(SUBCOMPONENT))

    def test_subcomponent_block_balanced(self):
        assert _block_balanced(_read(SUBCOMPONENT))

    def test_subcomponent_ascii(self):
        assert _is_ascii(_read(SUBCOMPONENT))

    def test_subcomponent_stays_surface_agnostic(self):
        # The badge renders the dynamic surface field, never a hardcoded surface
        # literal, so the sub-component stays shared across all three surfaces
        # (the S280 surface-agnostic invariant survives the badge).
        s = _read(SUBCOMPONENT)
        assert "entry.verdict" in s
        assert "'answer'" not in s


# ---------------------------------------------------------------------------
# Family 2 -- the preserved S279 surface (design-green)
# ---------------------------------------------------------------------------


class TestS279SurfacePreserved:
    def test_store_subscription_preserved(self):
        s = _read(SUBCOMPONENT)
        assert "from '$lib/stores/verdictHistory'" in s
        assert "verdictHistory" in s

    def test_ds_primitives_preserved(self):
        s = _read(SUBCOMPONENT)
        assert "from '$lib/ds'" in s
        assert "Card" in s
        assert "EmptyState" in s

    def test_clear_control_preserved(self):
        s = _read(SUBCOMPONENT)
        assert "clearVerdicts" in s
        assert "Clear" in s

    def test_each_render_preserved(self):
        assert "{#each" in _read(SUBCOMPONENT)

    def test_empty_state_preserved(self):
        assert "EmptyState" in _read(SUBCOMPONENT)


# ---------------------------------------------------------------------------
# Family 3 -- the preserved S280 fan-out (design-green)
# ---------------------------------------------------------------------------


class TestS280FanoutPreserved:
    def test_all_surfaces_render_history(self):
        for surface in (ANSWER_VERIFIER, CLAIM_VERIFIER, CITATION_VERIFIER):
            assert "<VerdictHistory" in _read(surface), surface.name

    def test_all_surfaces_import_subcomponent(self):
        for surface in (ANSWER_VERIFIER, CLAIM_VERIFIER, CITATION_VERIFIER):
            assert "./VerdictHistory.svelte" in _read(surface), surface.name


# ---------------------------------------------------------------------------
# Family 4 -- the host-assured runbook (content RED on pristine save ascii)
# ---------------------------------------------------------------------------


class TestRunbook:
    def test_exists_and_titled(self):
        assert _read(RUNBOOK).startswith("# VERDICT_HISTORY_SURFACE_LABEL_E2E_S282")

    def test_status_and_discipline(self):
        text = _flat(_read(RUNBOOK))
        assert "written at S282" in text
        assert "host-assured" in text
        assert "never simulated in the container" in text
        assert "produces findings, not fixes" in text

    def test_container_vs_host_split(self):
        text = _flat(_read(RUNBOOK))
        assert "container-provable" in text
        assert "host-assured" in text

    def test_required_sections(self):
        text = _flat(_read(RUNBOOK))
        assert "surface" in text
        assert "session-local" in text
        assert "Clear" in text
        assert "no mode gate" in text
        assert "findings register" in text

    def test_companions_named(self):
        text = _flat(_read(RUNBOOK))
        assert "VERDICT_HISTORY_E2E_S279.md" in text
        assert "VERDICT_HISTORY_FANOUT_E2E_S280.md" in text
        assert "VerdictHistory.svelte" in text
        assert "verdictHistory.ts" in text

    def test_auth_core_editfree(self):
        text = _flat(_read(RUNBOOK))
        assert "auth core" in text
        assert "edit-free" in text

    def test_runbook_ascii(self):
        assert _is_ascii(_read(RUNBOOK))


# ---------------------------------------------------------------------------
# Family 5 -- the held version (design-green)
# ---------------------------------------------------------------------------


class TestVersionHeld:
    def test_pyproject_version_held(self):
        assert 'version = "3.13.0"' in _read(PYPROJECT)

    def test_version_module_held(self):
        assert '__version__ = "3.13.0"' in _read(VERSION)


# ---------------------------------------------------------------------------
# Family 6 -- the clean-twin posture (design-green, preserved by the edit)
# ---------------------------------------------------------------------------


class TestCleanTwinPosture:
    def test_no_new_cartography_row(self):
        spec = _read(SPEC)
        assert spec.count(NEW_ANNOTATION) == 6
        assert spec.count(OLD_ANNOTATION) == 0

    def test_addopts_held(self):
        assert _read(PYPROJECT).count("--deselect") == ADDOPTS_DESELECTS

    def test_store_contract_unedited(self):
        # The surfacing edits only the sub-component; the shared store keeps its
        # S279 contract and its surface-bearing entry shape, unedited.
        s = _read(STORE)
        assert "export const verdictHistory" in s
        assert "export function recordVerdict" in s
        assert "export function clearVerdicts" in s
        assert "surface" in s


# ---------------------------------------------------------------------------
# Family 7 -- suite structure (design-green self-checks)
# ---------------------------------------------------------------------------


class TestSuiteStructure:
    def test_suite_parses(self):
        ast.parse(_read(OWN))

    def test_suite_ascii(self):
        assert _is_ascii(_read(OWN))

    def test_suite_avoids_canonical_literal(self):
        forbidden = "sandbox" + "_" + "manager"
        assert forbidden not in _read(OWN)
