"""S270 -- the /claims discoverability surface (Candidate A1).

The container-provable half of a labelled, host-assured nav walk: a single
sidebar nav entry that makes the S269 ``/claims`` page discoverable, on the
S248 ``/notes`` nav idiom. A proven-ZERO add-only lot -- it edits one existing
source (the sidebar component) without flipping any canonical pin, and adds a
test and a host-assured runbook; the failing-name register stays byte-identical
(the fourth twin of the S266 register) and the version holds at 3.13.0.

Seven families, the S253 / S269 idiom:

1. the nav entry -- the ``/claims`` object (href, label, icon) inside the
   ``navLinks`` array, placed after the ``/notes`` entry (RED on the pristine
   S269 tree, GREEN on the delivered tree);
2. the page seam -- the S269 ``/claims`` page still renders ``ClaimVerifier``
   (a design-green premise, true on pristine);
3. the nav idiom -- the sidebar declares ``navLinks``, carries the ``/notes``
   entry, and uses the ``Icon`` primitive (design-green premises);
4. the host-assured runbook -- existence, status, the container-vs-host split,
   the required sections, the auth-core edit-free note, pure ASCII (RED on
   pristine save the ascii pin, which is design-green because an absent file
   reads empty);
5. the held invariants -- the version held at 3.13.0 in both anchors
   (design-green);
6. the proven-ZERO posture -- no cartography row added (the s217 annotation
   counts intact), the addopts ledger held at 242, and the edited sidebar still
   carries no raw hex and stays block-balanced (per-file conformance,
   design-green and preserved);
7. suite structure -- the suite parses, is pure ASCII, and avoids the canonical
   selection literal (built only in split form) so the selection grep raw count
   is unchanged.

All read helpers return an empty string on absence, so a missing artifact
yields a bare assertion failure (never a collection or exception failure): the
red-before is a clean ``failures`` set with zero errors and zero skips.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_SRC = ROOT / "frontend" / "src"
SIDEBAR = FRONTEND_SRC / "lib" / "components" / "layout" / "Sidebar.svelte"
CLAIMS_PAGE = FRONTEND_SRC / "routes" / "claims" / "+page.svelte"
SPEC = ROOT / "FRONTEND_REDESIGN_SPEC.md"
RUNBOOK = ROOT / "CLAIMS_NAV_E2E_S270.md"
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


def _navlinks_region(src: str) -> str:
    """The substring spanning the navLinks array literal body (= [ ... ];).

    The type annotation ends in ``}[]`` before the ``= [`` literal, so the
    region is anchored on ``= [`` to skip it and closed at the ``];``. Returns
    the empty string when either anchor is absent so callers fall through to a
    bare assertion rather than an index error.
    """
    start = src.find("navLinks")
    if start < 0:
        return ""
    open_br = src.find("= [", start)
    if open_br < 0:
        return ""
    end = src.find("];", open_br)
    if end < 0:
        return ""
    return src[open_br:end]


# ---------------------------------------------------------------------------
# Family 1 -- the /claims nav entry (RED on the pristine S269 tree)
# ---------------------------------------------------------------------------


class TestNavEntry:
    def test_claims_href_present(self):
        assert "'/claims'" in _read(SIDEBAR)

    def test_claims_label_present(self):
        assert "label: 'Claims'" in _read(SIDEBAR)

    def test_claims_entry_well_formed(self):
        src = _read(SIDEBAR)
        pattern = r"\{\s*href:\s*'/claims',\s*label:\s*'Claims',\s*icon:\s*'[a-z0-9-]+'\s*\}"
        assert re.search(pattern, src) is not None

    def test_claims_entry_within_navlinks(self):
        region = _navlinks_region(_read(SIDEBAR))
        assert "'/claims'" in region

    def test_claims_placed_after_notes(self):
        src = _read(SIDEBAR)
        assert ("'/claims'" in src) and (src.find("'/claims'") > src.find("'/notes'"))


# ---------------------------------------------------------------------------
# Family 2 -- the page seam (design-green: S269 already ships the page)
# ---------------------------------------------------------------------------


class TestPageSeam:
    def test_claims_page_renders_verifier(self):
        assert "ClaimVerifier" in _read(CLAIMS_PAGE)


# ---------------------------------------------------------------------------
# Family 3 -- the nav idiom (design-green premises on the pristine tree)
# ---------------------------------------------------------------------------


class TestNavIdiom:
    def test_sidebar_declares_navlinks(self):
        assert "navLinks" in _read(SIDEBAR)

    def test_sidebar_notes_entry_present(self):
        src = _read(SIDEBAR)
        assert ("'/notes'" in src) and ("label: 'Notes'" in src)

    def test_sidebar_imports_icon(self):
        src = _read(SIDEBAR)
        assert ("$lib/ds/Icon.svelte" in src) and ("link.icon" in src)


# ---------------------------------------------------------------------------
# Family 4 -- the host-assured runbook (RED on pristine save the ascii pin)
# ---------------------------------------------------------------------------


class TestRunbook:
    def test_exists_and_titled(self):
        assert _read(RUNBOOK).startswith("# CLAIMS_NAV_E2E_S270")

    def test_status_and_discipline(self):
        text = _flat(_read(RUNBOOK))
        assert "written at S270" in text
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
        assert "sidebar" in text
        assert "no mode gate" in text
        assert "findings register" in text

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

    def test_sidebar_no_raw_hex(self):
        assert not _has_raw_hex(_read(SIDEBAR))

    def test_sidebar_block_balanced(self):
        assert _block_balanced(_read(SIDEBAR))


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
