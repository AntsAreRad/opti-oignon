#!/usr/bin/env python3
"""S279 -- the verdict-history affordance (the answer-verification surface).

The container-provable half of a labelled, host-assured walk: a session-local
history of recent verdicts, wired into exactly one verifier surface (the S277
``AnswerVerifier`` on ``/verify-answer``, freshly discoverable since the S278
nav). A new standalone store (``verdictHistory.ts``) plus a new sub-component
(``VerdictHistory.svelte``) carry the affordance; the edit to ``AnswerVerifier``
is limited to importing the store and the sub-component, emitting each result,
and rendering the history. The two other verifiers (``ClaimVerifier``,
``CitationVerifier``) are untouched, so the supersession surface is exactly one
component. A proven-ZERO add-only lot -- the edit flips no canonical pin, the
failing-name register stays byte-identical (the thirteenth twin of the S266
register), and the version holds at 3.13.0.

Seven families, the S277 / S278 idiom:

1. the verdict store -- ``verdictHistory.ts``: a writable history, the record
   and clear functions, the entry shape, the session cap (RED on the pristine
   S278 tree, GREEN on the delivered tree);
2. the sub-component -- ``VerdictHistory.svelte``: the store subscription, the
   ds primitives, the Clear control, the per-entry render, the empty state, plus
   no raw hex / block-balanced / pure ASCII (content RED on pristine, the
   conformance pins design-green because an absent file reads empty);
3. the AnswerVerifier edit -- the store and sub-component imports, the emit on
   verify, the rendered history (RED on pristine), plus the preserved ascii;
4. the host-assured runbook -- ``VERDICT_HISTORY_E2E_S279.md``: existence,
   status, the container-vs-host split, the required sections, the companions,
   the auth-core edit-free note, pure ASCII (content RED on pristine, the ascii
   pin design-green);
5. the held version -- 3.13.0 in both anchors (design-green);
6. the proven-ZERO posture -- no cartography row added (the s217 annotation
   counts intact), the addopts ledger held at 242, the edited AnswerVerifier
   still no raw hex and block-balanced, and the two other verifiers untouched by
   the affordance (per-file conformance, design-green and preserved);
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
RUNBOOK = ROOT / "VERDICT_HISTORY_E2E_S279.md"
PYPROJECT = ROOT / "pyproject.toml"
VERSION = ROOT / "opti_oignon" / "__version__.py"
OWN = Path(__file__).resolve()

# Mirror of the test_s217 cartography annotation counters (the cartography
# lock): a proven-ZERO lot adds no component row, so these stay 6 and 0.
NEW_ANNOTATION = "deleted at S217 (FRD-01 landed; absence locked by tests/test_s217_cleanup.py)"
OLD_ANNOTATION = "removal recorded FRD-01 (S197)"

# The expected addopts ledger size (one long line in pyproject.toml).
ADDOPTS_DESELECTS = 242


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
# Family 1 -- the verdict store (RED on the pristine S278 tree)
# ---------------------------------------------------------------------------


class TestVerdictStore:
    def test_store_exists(self):
        assert _read(STORE)

    def test_store_uses_writable(self):
        s = _read(STORE)
        assert "writable" in s
        assert "from 'svelte/store'" in s

    def test_store_exports_history(self):
        assert "export const verdictHistory" in _read(STORE)

    def test_store_exports_record_and_clear(self):
        s = _read(STORE)
        assert "export function recordVerdict" in s
        assert "export function clearVerdicts" in s

    def test_store_entry_shape(self):
        s = _read(STORE)
        assert "export interface VerdictEntry" in s
        for field in ("id", "surface", "verdict", "ok", "summary", "at"):
            assert field in s, field

    def test_store_caps_history(self):
        s = _read(STORE)
        assert "MAX_VERDICT_HISTORY" in s
        assert ".slice(" in s

    def test_store_ascii(self):
        assert _is_ascii(_read(STORE))


# ---------------------------------------------------------------------------
# Family 2 -- the sub-component (content RED, conformance design-green)
# ---------------------------------------------------------------------------


class TestVerdictHistoryComponent:
    def test_subcomponent_exists(self):
        assert _read(SUBCOMPONENT)

    def test_subcomponent_subscribes_store(self):
        s = _read(SUBCOMPONENT)
        assert "from '$lib/stores/verdictHistory'" in s
        assert "verdictHistory" in s

    def test_subcomponent_imports_ds(self):
        s = _read(SUBCOMPONENT)
        assert "from '$lib/ds'" in s
        assert "Card" in s
        assert "EmptyState" in s

    def test_subcomponent_clear_control(self):
        s = _read(SUBCOMPONENT)
        assert "clearVerdicts" in s
        assert "Clear" in s

    def test_subcomponent_renders_entries(self):
        assert "{#each" in _read(SUBCOMPONENT)

    def test_subcomponent_empty_state(self):
        assert "EmptyState" in _read(SUBCOMPONENT)

    def test_subcomponent_no_raw_hex(self):
        assert not _has_raw_hex(_read(SUBCOMPONENT))

    def test_subcomponent_block_balanced(self):
        assert _block_balanced(_read(SUBCOMPONENT))

    def test_subcomponent_ascii(self):
        assert _is_ascii(_read(SUBCOMPONENT))


# ---------------------------------------------------------------------------
# Family 3 -- the AnswerVerifier edit (content RED, ascii preserved)
# ---------------------------------------------------------------------------


class TestAnswerVerifierWiring:
    def test_av_imports_store(self):
        s = _read(ANSWER_VERIFIER)
        assert "from '$lib/stores/verdictHistory'" in s
        assert "recordVerdict" in s

    def test_av_imports_subcomponent(self):
        assert "./VerdictHistory.svelte" in _read(ANSWER_VERIFIER)

    def test_av_emits_on_verify(self):
        assert "recordVerdict(" in _read(ANSWER_VERIFIER)

    def test_av_renders_history(self):
        assert "<VerdictHistory" in _read(ANSWER_VERIFIER)

    def test_av_ascii(self):
        assert _is_ascii(_read(ANSWER_VERIFIER))


# ---------------------------------------------------------------------------
# Family 4 -- the host-assured runbook (content RED on pristine save ascii)
# ---------------------------------------------------------------------------


class TestRunbook:
    def test_exists_and_titled(self):
        assert _read(RUNBOOK).startswith("# VERDICT_HISTORY_E2E_S279")

    def test_status_and_discipline(self):
        text = _flat(_read(RUNBOOK))
        assert "written at S279" in text
        assert "host-assured" in text
        assert "never simulated in the container" in text
        assert "produces findings, not fixes" in text

    def test_container_vs_host_split(self):
        text = _flat(_read(RUNBOOK))
        assert "container-provable" in text
        assert "host-assured" in text

    def test_required_sections(self):
        text = _flat(_read(RUNBOOK))
        assert "/verify-answer" in text
        assert "session-local" in text
        assert "Clear" in text
        assert "no mode gate" in text
        assert "findings register" in text

    def test_companions_named(self):
        text = _flat(_read(RUNBOOK))
        assert "ANSWER_VERIFICATION_UI_E2E_S277.md" in text
        assert "VERIFY_ANSWER_NAV_E2E_S278.md" in text
        assert "verdictHistory.ts" in text
        assert "VerdictHistory.svelte" in text
        assert "AnswerVerifier.svelte" in text

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
# Family 6 -- the proven-ZERO posture (design-green, preserved by the edit)
# ---------------------------------------------------------------------------


class TestProvenZeroPosture:
    def test_no_new_cartography_row(self):
        spec = _read(SPEC)
        assert spec.count(NEW_ANNOTATION) == 6
        assert spec.count(OLD_ANNOTATION) == 0

    def test_addopts_held(self):
        assert _read(PYPROJECT).count("--deselect") == ADDOPTS_DESELECTS

    def test_answer_verifier_no_raw_hex(self):
        assert not _has_raw_hex(_read(ANSWER_VERIFIER))

    def test_answer_verifier_block_balanced(self):
        assert _block_balanced(_read(ANSWER_VERIFIER))

    def test_other_verifiers_untouched_by_affordance(self):
        # The affordance is wired into exactly one verifier; the other two carry
        # no reference to the store or the sub-component, so the supersession
        # surface stays a single component.
        for other in (CLAIM_VERIFIER, CITATION_VERIFIER):
            s = _read(other)
            assert "verdictHistory" not in s, other.name
            assert "VerdictHistory" not in s, other.name


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
