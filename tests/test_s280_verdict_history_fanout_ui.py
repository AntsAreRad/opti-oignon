#!/usr/bin/env python3
"""S280 -- the verdict-history fan-out (the claim and citation surfaces).

The container-provable half of a labelled, host-assured walk: the S279
verdict-history affordance, fanned out from the single S277 ``AnswerVerifier``
surface to the two remaining verifier surfaces -- the S269 ``ClaimVerifier`` on
``/claims`` and the S275 ``CitationVerifier`` on ``/verify-citations``. The
S279 store (``verdictHistory.ts``) and sub-component (``VerdictHistory.svelte``)
are reused unchanged: the store was deliberately shared-shaped and
surface-labelled (the ``surface`` field), so each component records its own
verdict under its own surface label ('claim', 'citation') and renders the same
shared session history. The per-component edit is exactly the S279 posture
proven safe on ``AnswerVerifier``: two imports (the store, the sub-component),
the emit immediately after the server-returned result, and the rendered history
at the foot.

This is a supersession-bearing lot, not a twin. Editing the two other verifiers
flips the S279 proven-ZERO pin
``test_s279 ...::test_other_verifiers_untouched_by_affordance``; per the
deselect-plus-reassert discipline that node is deselected and the new truth (all
three verifiers carry the affordance) is re-asserted here. Adding that deselect
bumps the addopts ledger 242 -> 248, which in turn supersedes the five current
242-count pins (test_s266 / test_s270 / test_s276 / test_s278 / test_s279); each
is deselected and the new count re-asserted here. The version holds at 3.13.0 (a
frontend affordance does not bump).

Seven families, the S277 / S278 / S279 idiom:

1. the ClaimVerifier fan-out -- the store and sub-component imports, the emit on
   verify, the rendered history, the 'claim' surface label (content RED on the
   pristine S279 tree), plus the preserved presence and per-file conformance
   (design-green and preserved by the additive edit);
2. the CitationVerifier fan-out -- the same wiring with the 'citation' surface
   label (content RED on pristine), plus the preserved aggregate / per-pair
   surface and conformance (design-green);
3. the reused store and sub-component -- ``verdictHistory.ts`` and
   ``VerdictHistory.svelte`` unchanged, the store contract intact, the
   sub-component surface-agnostic (design-green, reused as-is);
4. the host-assured runbook -- ``VERDICT_HISTORY_FANOUT_E2E_S280.md``: existence,
   status, the container-vs-host split, the required sections, the companions,
   the auth-core edit-free note, pure ASCII (content RED on pristine, the ascii
   pin design-green);
5. the held version -- 3.13.0 in both anchors (design-green);
6. the supersession posture -- the addopts ledger re-asserted at 248, the six
   superseded nodes recorded as deselect entries, all three verifiers carrying
   the affordance (RED on pristine), plus the cartography lock held and the
   unedited AnswerVerifier still wired and clean (design-green);
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
RUNBOOK = ROOT / "VERDICT_HISTORY_FANOUT_E2E_S280.md"
PYPROJECT = ROOT / "pyproject.toml"
VERSION = ROOT / "opti_oignon" / "__version__.py"
OWN = Path(__file__).resolve()

# Mirror of the test_s217 cartography annotation counters (the cartography
# lock): a fan-out lot adds no component row, so these stay 6 and 0.
NEW_ANNOTATION = "deleted at S217 (FRD-01 landed; absence locked by tests/test_s217_cleanup.py)"
OLD_ANNOTATION = "removal recorded FRD-01 (S197)"

# The addopts ledger after the six S280 deselects (242 + 6). One long line.
ADDOPTS_DESELECTS = 248

# The six nodes superseded by this fan-out, recorded as deselect entries in
# pyproject.toml (one the affordance flip on the other verifiers, five the
# 242-count pins the resulting ledger bump makes stale).
SUPERSEDED_NODES = (
    "tests/test_s279_verdict_history_ui.py::TestProvenZeroPosture::test_other_verifiers_untouched_by_affordance",
    "tests/test_s266_release.py::TestAddoptsLineage::test_count_grew_by_exactly_fourteen",
    "tests/test_s270_claims_nav.py::TestProvenZeroPosture::test_addopts_held",
    "tests/test_s276_verify_citations_nav.py::TestProvenZeroPosture::test_addopts_held",
    "tests/test_s278_verify_answer_nav.py::TestProvenZeroPosture::test_addopts_held",
    "tests/test_s279_verdict_history_ui.py::TestProvenZeroPosture::test_addopts_held",
)


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
# Family 1 -- the ClaimVerifier fan-out (content RED on the pristine S279 tree)
# ---------------------------------------------------------------------------


class TestClaimVerifierFanout:
    def test_cv_imports_store(self):
        s = _read(CLAIM_VERIFIER)
        assert "from '$lib/stores/verdictHistory'" in s
        assert "recordVerdict" in s

    def test_cv_imports_subcomponent(self):
        assert "./VerdictHistory.svelte" in _read(CLAIM_VERIFIER)

    def test_cv_emits_on_verify(self):
        assert "recordVerdict(" in _read(CLAIM_VERIFIER)

    def test_cv_renders_history(self):
        assert "<VerdictHistory" in _read(CLAIM_VERIFIER)

    def test_cv_surface_label(self):
        assert "surface: 'claim'" in _read(CLAIM_VERIFIER)

    def test_cv_preserved_presence(self):
        # The S269 surface behaviour is preserved by the additive edit.
        s = _read(CLAIM_VERIFIER)
        assert "await verifyClaim(" in s
        assert "disabled" in s
        for verdict in ("supported", "unsupported", "uncertain"):
            assert verdict in s, verdict

    def test_cv_no_raw_hex(self):
        assert not _has_raw_hex(_read(CLAIM_VERIFIER))

    def test_cv_block_balanced(self):
        assert _block_balanced(_read(CLAIM_VERIFIER))

    def test_cv_ascii(self):
        assert _is_ascii(_read(CLAIM_VERIFIER))


# ---------------------------------------------------------------------------
# Family 2 -- the CitationVerifier fan-out (content RED on pristine)
# ---------------------------------------------------------------------------


class TestCitationVerifierFanout:
    def test_xv_imports_store(self):
        s = _read(CITATION_VERIFIER)
        assert "from '$lib/stores/verdictHistory'" in s
        assert "recordVerdict" in s

    def test_xv_imports_subcomponent(self):
        assert "./VerdictHistory.svelte" in _read(CITATION_VERIFIER)

    def test_xv_emits_on_verify(self):
        assert "recordVerdict(" in _read(CITATION_VERIFIER)

    def test_xv_renders_history(self):
        assert "<VerdictHistory" in _read(CITATION_VERIFIER)

    def test_xv_surface_label(self):
        assert "surface: 'citation'" in _read(CITATION_VERIFIER)

    def test_xv_preserved_presence(self):
        # The S275 aggregate / per-pair surface is preserved by the edit.
        s = _read(CITATION_VERIFIER)
        assert "await verifyCitations(" in s
        assert "disabled" in s
        assert "results" in s
        assert "pairs" in s
        assert "{#each" in s

    def test_xv_no_raw_hex(self):
        assert not _has_raw_hex(_read(CITATION_VERIFIER))

    def test_xv_block_balanced(self):
        assert _block_balanced(_read(CITATION_VERIFIER))

    def test_xv_ascii(self):
        assert _is_ascii(_read(CITATION_VERIFIER))


# ---------------------------------------------------------------------------
# Family 3 -- the reused store and sub-component (design-green, reused as-is)
# ---------------------------------------------------------------------------


class TestReusedStoreAndSubcomponent:
    def test_store_contract_intact(self):
        s = _read(STORE)
        assert "export const verdictHistory" in s
        assert "export function recordVerdict" in s
        assert "export function clearVerdicts" in s

    def test_subcomponent_surface_agnostic(self):
        # The shared sub-component renders the entry verdict and summary, never
        # a single surface literal, so the fan-out reuses it unedited.
        s = _read(SUBCOMPONENT)
        assert "verdictHistory" in s
        assert "entry.verdict" in s
        assert "'answer'" not in s


# ---------------------------------------------------------------------------
# Family 4 -- the host-assured runbook (content RED on pristine save ascii)
# ---------------------------------------------------------------------------


class TestRunbook:
    def test_exists_and_titled(self):
        assert _read(RUNBOOK).startswith("# VERDICT_HISTORY_FANOUT_E2E_S280")

    def test_status_and_discipline(self):
        text = _flat(_read(RUNBOOK))
        assert "written at S280" in text
        assert "host-assured" in text
        assert "never simulated in the container" in text
        assert "produces findings, not fixes" in text

    def test_container_vs_host_split(self):
        text = _flat(_read(RUNBOOK))
        assert "container-provable" in text
        assert "host-assured" in text

    def test_required_sections(self):
        text = _flat(_read(RUNBOOK))
        assert "/claims" in text
        assert "/verify-citations" in text
        assert "session-local" in text
        assert "Clear" in text
        assert "no mode gate" in text
        assert "findings register" in text
        assert "shared" in text

    def test_companions_named(self):
        text = _flat(_read(RUNBOOK))
        assert "CLAIM_VERIFICATION_UI_E2E_S269.md" in text
        assert "CITATION_VERIFICATION_UI_E2E_S275.md" in text
        assert "VERDICT_HISTORY_E2E_S279.md" in text
        assert "verdictHistory.ts" in text
        assert "VerdictHistory.svelte" in text
        assert "ClaimVerifier.svelte" in text
        assert "CitationVerifier.svelte" in text

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
# Family 6 -- the supersession posture (the new truth RED, the locks green)
# ---------------------------------------------------------------------------


class TestSupersessionPosture:
    def test_addopts_count_bumped(self):
        # The fan-out adds exactly six deselects (one affordance flip, five
        # 242-count pins made stale by the resulting bump): 242 -> 248.
        assert _read(PYPROJECT).count("--deselect") == ADDOPTS_DESELECTS

    def test_superseded_nodes_deselected(self):
        src = _read(PYPROJECT)
        for node in SUPERSEDED_NODES:
            assert node in src, node

    def test_all_three_verifiers_carry_affordance(self):
        # The fan-out re-asserts the new truth that supersedes the S279
        # other-verifiers-untouched pin: every verifier surface now records and
        # renders the verdict history.
        for surface in (ANSWER_VERIFIER, CLAIM_VERIFIER, CITATION_VERIFIER):
            s = _read(surface)
            assert "recordVerdict" in s, surface.name
            assert "VerdictHistory" in s, surface.name

    def test_cartography_lock_held(self):
        spec = _read(SPEC)
        assert spec.count(NEW_ANNOTATION) == 6
        assert spec.count(OLD_ANNOTATION) == 0

    def test_answer_verifier_still_wired(self):
        # The S277 surface is not edited this lot; its S279 wiring is intact.
        s = _read(ANSWER_VERIFIER)
        assert "recordVerdict" in s
        assert "<VerdictHistory" in s

    def test_answer_verifier_clean(self):
        s = _read(ANSWER_VERIFIER)
        assert not _has_raw_hex(s)
        assert _block_balanced(s)


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
