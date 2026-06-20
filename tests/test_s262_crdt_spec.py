#!/usr/bin/env python3
"""S262 doc-pin suite: NOTES_CRDT_SPEC.md + the N.8 roadmap roll.

A spec-only bloc, N.8 taken spec-first: the document that decides the
collaboration model for the Notes body CRDT before any code. NO production
code is edited at S262. This suite pins three families so a later session
cannot drift silently:

1. The document pins -- NOTES_CRDT_SPEC.md exists at the repo top level and
   carries the arbitrated decisions, one pinned sentence per decision: the
   carrier (one Yjs document per note body, client-held; backend and Veilid
   relay opaque updates and never interpret them; the whole-blob row stays
   the rendered-state checkpoint), the at-rest update log (a new safe_connect
   SQLCipher append-only store, per-user isolation, parameterized SQL only,
   frozenset-allowlisted dynamic identifiers), the transport (a new record
   kind on the S256 seam, honoured never bypassed; the S258 phone-class serve
   gate as a floor over a live is_mobile_allowed lookup; signatures preserved
   end to end, the serve path never re-signs), client-side merge with
   PATCH-leg compaction and tombstone pruning, the S260 confirmed posture
   extended to collaborative edits, the fail-secure tail, and the explicit
   first-bloc non-goals plus the spec-only sentence.
2. The roadmap roll -- the N.8 entry in NOTES_FEATURE_ROADMAP.md is rolled
   "LANDED at S262" naming the spec document, WITHOUT disturbing the dated
   rolls the file already carries (S243 under N.1; S256 / S257 / S258 / S260
   and the N9-D1 / N9-D2 / N9-D3 decisions under N.9).
3. The seams the spec constrains -- source-level pins on the premises (the
   five exact notes routes, the agent tools module at zero mobile_allowed
   occurrences per N9-D3, the filter-at-serve and never-re-signs markers in
   the sync engine, the safe_connect / _ORDERABLE_COLUMNS house rules in the
   notes store), so a later edit that removes a premise turns this suite red
   instead of letting the spec rot.

Red-before discipline: on the pristine S261 tree (no NOTES_CRDT_SPEC.md,
roadmap not rolled) every family-1 pin and the two family-2 roll pins FAIL --
the read helper returns an empty string so absence is an assertion failure,
never a collection error -- while every family-2 guard and family-3 seam pin
passes by design (they pin pre-existing invariants the spec relies on).
Expected red-before split: 26 red / 8 design-green over 34. Document pins
read through the whitespace-flattening helper (the S221/S222/S233 lesson) so
line reflow that does not change wording cannot break them; source pins stay
raw. Nothing here imports the package, so no ollama chain is touched.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO / "NOTES_CRDT_SPEC.md"
ROADMAP_PATH = REPO / "NOTES_FEATURE_ROADMAP.md"
PKG = REPO / "opti_oignon"
ROUTES_NOTES = PKG / "api" / "routes_notes.py"
TOOLS_SRC = PKG / "agent" / "tools.py"
SYNC_ENGINE = PKG / "veilid" / "sync_engine.py"
NOTES_STORE = PKG / "notes" / "notes_store.py"


def _read(path: Path) -> str:
    """Raw text; empty string when absent so absence FAILS, never errors."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _flat(text: str) -> str:
    """Collapse all whitespace runs to single spaces (reflow-immune pins)."""
    return re.sub(r"\s+", " ", text)


def _spec() -> str:
    return _flat(_read(SPEC_PATH))


def _roadmap() -> str:
    return _flat(_read(ROADMAP_PATH))


# ---------------------------------------------------------------------------
# Family 1a -- NOTES_CRDT_SPEC.md exists, titled, ASCII
# ---------------------------------------------------------------------------


class TestSpecExists:
    def test_file_exists_and_titled(self):
        text = _read(SPEC_PATH)
        assert text.startswith("# NOTES_CRDT_SPEC"), (
            "NOTES_CRDT_SPEC.md missing or not titled at the repo top level"
        )

    def test_pure_ascii(self):
        text = _read(SPEC_PATH)
        assert text != "", "NOTES_CRDT_SPEC.md missing"
        assert all(ord(c) < 128 for c in text), "spec must be pure ASCII"


# ---------------------------------------------------------------------------
# Family 1b -- the section headings, one pin each
# ---------------------------------------------------------------------------


class TestSpecHeadings:
    def test_heading_status(self):
        assert "## 0. Status and scope" in _spec()

    def test_heading_carrier(self):
        assert "## 1. Carrier" in _spec()

    def test_heading_at_rest(self):
        assert "## 2. At rest" in _spec()

    def test_heading_transport(self):
        assert "## 3. Transport" in _spec()

    def test_heading_merge_compaction(self):
        assert "## 4. Merge and compaction" in _spec()

    def test_heading_failure_posture(self):
        assert "## 5. Failure posture" in _spec()

    def test_heading_non_goals(self):
        assert "## 6. Non-goals (first implementation bloc)" in _spec()

    def test_heading_test_surface(self):
        assert "## 7. Test surface" in _spec()


# ---------------------------------------------------------------------------
# Family 1c -- the decided sentences, one pin each (flattened)
# ---------------------------------------------------------------------------


class TestCarrierDecided:
    def test_carrier_sentence(self):
        assert (
            "The CRDT carrier is one Yjs document per note body, held by the"
            " Svelte client; the backend and Veilid store and relay Yjs binary"
            " updates as opaque blobs and never interpret them." in _spec()
        )

    def test_checkpoint_sentence(self):
        assert (
            "The existing whole-blob note row remains the rendered-state"
            " checkpoint, so the five notes routes and the S257 journaling"
            " glue stay unchanged by this model." in _spec()
        )


class TestAtRestDecided:
    def test_store_sentence(self):
        assert (
            "The Yjs update log lives at rest in a new SQLCipher store opened"
            " through safe_connect: an append-only note_update table, isolated"
            " per user via effective_user_id." in _spec()
        )

    def test_house_rules_sentence(self):
        assert (
            "All SQL in the update store is parameterized; no SQL f-strings;"
            " dynamic identifiers only via str.format over frozenset"
            " allowlists." in _spec()
        )


class TestTransportDecided:
    def test_seam_sentence(self):
        assert (
            "Updates ride the existing sync envelope as a new record kind on"
            " the S256 seam, which is honoured, never bypassed." in _spec()
        )

    def test_serve_gate_floor_sentence(self):
        assert (
            "The S258 device-class serve gate is a floor: toward a phone-class"
            " peer, a note update is served only when a live is_mobile_allowed"
            " lookup affirms the parent note's flag, fail-secure." in _spec()
        )

    def test_signature_sentence(self):
        assert (
            "Author signatures are preserved end to end; the serve path never"
            " re-signs an update." in _spec()
        )


class TestMergeCompactionDecided:
    def test_merge_sentence(self):
        assert (
            "Merge is client-side Yjs; the backend never merges and never"
            " interprets update content." in _spec()
        )

    def test_compaction_sentence(self):
        assert (
            "Compaction is the client writing the merged whole-blob state"
            " through the existing PATCH leg; the update log up to that"
            " checkpoint becomes prunable." in _spec()
        )

    def test_tombstone_sentence(self):
        assert (
            "Updates of a tombstoned note stop being served and become"
            " prunable." in _spec()
        )


class TestFailurePostureDecided:
    def test_confirmed_posture_sentence(self):
        assert (
            "The S260 confirmed posture extends to collaborative edits:"
            " nothing renders that the backend has not recorded." in _spec()
        )

    def test_fail_secure_sentence(self):
        assert (
            "Anything indeterminable fails secure: an update that cannot be"
            " attributed, gated, or persisted is refused." in _spec()
        )


class TestNonGoalsDecided:
    def test_non_goals_sentence(self):
        assert (
            "Non-goals for the first implementation bloc: server-side merge"
            " and yrs; OR-Set tags rework; live cursors or presence;"
            " cross-user collaboration; a mobile editor; bulk migration of"
            " existing note bodies." in _spec()
        )

    def test_spec_only_sentence(self):
        assert (
            "This bloc is spec-only: no implementation and no schema migration"
            " ships with it." in _spec()
        )


# ---------------------------------------------------------------------------
# Family 2 -- the roadmap roll (red) and the prior rolls intact (design-green)
# ---------------------------------------------------------------------------


class TestRoadmapRoll:
    def test_n8_rolled_landed_s262(self):
        assert "LANDED at S262" in _roadmap()

    def test_roll_names_the_spec_document(self):
        assert "NOTES_CRDT_SPEC.md" in _roadmap()


class TestRoadmapPriorRollsIntact:
    def test_n1_roll_intact(self):
        assert "LANDED at S243:" in _roadmap()

    def test_n9_rolls_intact(self):
        text = _roadmap()
        assert "LANDED at S256:" in text
        assert "LANDED at S257:" in text
        assert "LANDED at S258:" in text
        assert "LANDED at S260:" in text

    def test_n9_decisions_intact(self):
        text = _roadmap()
        assert "N9-D1" in text
        assert "N9-D2" in text
        assert "N9-D3" in text


# ---------------------------------------------------------------------------
# Family 3 -- the seams the spec constrains, raw source pins (design-green)
# ---------------------------------------------------------------------------


class TestReassertNotesRoutes:
    def test_exactly_five_route_decorators(self):
        src = _read(ROUTES_NOTES)
        assert src.count("@notes_router.") == 5

    def test_the_five_exact_routes(self):
        src = _read(ROUTES_NOTES)
        assert '@notes_router.get(""' in src
        assert '@notes_router.post(""' in src
        assert '@notes_router.get("/{note_id}"' in src
        assert '@notes_router.patch("/{note_id}"' in src
        assert '@notes_router.delete("/{note_id}"' in src


class TestReassertToolsFlagZero:
    def test_tools_zero_mobile_allowed(self):
        src = _read(TOOLS_SRC)
        assert src != "", "agent/tools.py missing"
        assert src.count("mobile_allowed") == 0, (
            "N9-D3: the gated tool surface must never touch the flag"
        )


class TestReassertSyncSurfaces:
    def test_sync_engine_serve_gate_markers(self):
        src = _read(SYNC_ENGINE)
        assert "filter-at-serve" in src
        assert "never re-signs" in src

    def test_notes_store_house_rules_markers(self):
        src = _read(NOTES_STORE)
        assert "safe_connect" in src
        assert "_ORDERABLE_COLUMNS" in src
