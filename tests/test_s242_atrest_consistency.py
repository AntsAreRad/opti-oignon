"""S242 -- RS-01 at-rest consistency (PEER-01 + the deferred-ledger CHF-04 slice).

This is a real source lot, not a host-assured runbook. It closes a standing
security-routed finding: the two remaining device-local veilid stores that still
opened their database with a plain ``sqlite3.connect`` -- the peer registry
(``peers.py``, PEER-01) and the deferred-ledger quarantine (``deferred_ledger.py``,
the CHF-04 plaintext slice the quarantine routed to) -- now join ``safe_connect``,
the same SQLCipher-at-rest pattern the veilid change feed adopted at S136. The
public routing keys are Kerckhoffs material either way; RS1-D1 records that the
open-registry rationale does NOT stand for the registry's privacy-relevant
metadata (the human labels / device names, the device topology, the watermarks,
and the SYN-02 local device identity), so the registry is encrypted at rest on a
host with SQLCipher. In-container SQLCipher is absent, so ``safe_connect`` degrades
to plaintext with a once-emitted warning -- the documented db_encryption posture,
identical to every other ``safe_connect`` store -- which is why the behavioural
family below stays green before and after (the in-container plaintext path is
unchanged) while the source-migration family flips red-before to green.

Six families:

 1. The source migration -- ``peers.py`` and ``deferred_ledger.py`` import
    ``safe_connect as _safe_connect`` with the S136 ImportError fallback, the
    ``_conn`` factory opens through ``_safe_connect``, and the fallback warning
    names the PLAINTEXT degradation so it stays auditable.
 2. Guards the migration must NOT break (green before and after) -- the S136
    change-feed precedent is intact, ``db_utils.safe_connect`` exists, and the
    ``checkpoint_before_apply = True`` sentinel survives in both edited modules.
 3. Behavioural (green before and after) -- a ``PeerStore`` and a
    ``DeferredLedger`` open against a temp directory, create their table, and
    report an empty count under WAL, proving the migrated ``_conn`` factory works
    in-container (degrading to plaintext) without breaking the store.
 4. The doc rolls and the decision -- ATREST_INVENTORY.md records the PEER-01 /
    CHF-04 closure and RS1-D1, ROADMAP_POST_AUDIT.md rolls the security-routed
    at-rest bullet, and AUDIT_FUNCTIONAL_FINDINGS.md marks the closure.
 5. AST validity of the two edited modules and this suite.
 6. The supersession reassert -- the deselect-plus-reassert counterpart of
    test_s233_cas7_spec.py::TestSeamPeers::test_registry_is_plain_sqlite (whose
    intent RS1-D1 supersedes): the registry's PRIMARY connection is now
    ``_safe_connect`` and the only surviving ``sqlite3.connect`` is the documented
    ImportError fallback lambda, so "the registry is plain sqlite" no longer holds
    by design even though the literal fallback string survives.

Red-before discipline on the pristine S241 tree: families 1, 4, and 6 FAIL (the
modules still open with ``conn = sqlite3.connect(`` and the docs are not rolled),
while families 2, 3, and 5 PASS by design (they pin pre-existing invariants and
the in-container plaintext behaviour the migration preserves). Document pins read
through a whitespace-flattening helper so reflow cannot break them; source pins
stay raw; the behavioural family imports lazily inside the tests so collection
touches no package import chain.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "opti_oignon"

PEERS_PATH = PKG / "veilid" / "peers.py"
DEFERRED_PATH = PKG / "veilid" / "deferred_ledger.py"
CHANGE_FEED_PATH = PKG / "veilid" / "change_feed.py"
DB_UTILS_PATH = PKG / "db_utils.py"

ATREST_PATH = REPO / "ATREST_INVENTORY.md"
ROADMAP_PATH = REPO / "ROADMAP_POST_AUDIT.md"
AUDIT_PATH = REPO / "AUDIT_FUNCTIONAL_FINDINGS.md"


def _read(path: Path) -> str:
    """Raw text; empty string when absent so absence FAILS, never errors."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _flat(text: str) -> str:
    """Collapse all whitespace runs to single spaces (reflow-immune pins)."""
    return re.sub(r"\s+", " ", text)


# ---------------------------------------------------------------------------
# Family 1 -- the source migration to safe_connect
# ---------------------------------------------------------------------------


class TestPeersMigration:
    def test_imports_safe_connect_with_fallback(self):
        src = _read(PEERS_PATH)
        assert "from opti_oignon.db_utils import safe_connect as _safe_connect" in src
        assert "_safe_connect = lambda p, **kw: sqlite3.connect(str(p), **kw)" in src

    def test_conn_factory_opens_through_safe_connect(self):
        src = _read(PEERS_PATH)
        assert "conn = _safe_connect(" in src

    def test_fallback_warning_names_plaintext(self):
        src = _read(PEERS_PATH)
        assert "veilid peer registry falling back to PLAINTEXT" in src


class TestDeferredMigration:
    def test_imports_safe_connect_with_fallback(self):
        src = _read(DEFERRED_PATH)
        assert "from opti_oignon.db_utils import safe_connect as _safe_connect" in src
        assert "_safe_connect = lambda p, **kw: sqlite3.connect(str(p), **kw)" in src

    def test_conn_factory_opens_through_safe_connect(self):
        src = _read(DEFERRED_PATH)
        assert "conn = _safe_connect(" in src

    def test_fallback_warning_names_plaintext(self):
        src = _read(DEFERRED_PATH)
        assert "veilid deferred ledger falling back to PLAINTEXT" in src


# ---------------------------------------------------------------------------
# Family 2 -- guards the migration must not break (green before and after)
# ---------------------------------------------------------------------------


class TestPrecedentGuards:
    def test_change_feed_s136_precedent_intact(self):
        src = _read(CHANGE_FEED_PATH)
        assert "S136 audit fix" in src
        assert "from opti_oignon.db_utils import safe_connect as _safe_connect" in src
        assert "conn = _safe_connect(" in src

    def test_safe_connect_defined_in_db_utils(self):
        assert "def safe_connect(" in _read(DB_UTILS_PATH)

    def test_checkpoint_sentinel_survives(self):
        assert "checkpoint_before_apply = True" in _read(PEERS_PATH)
        assert "checkpoint_before_apply = True" in _read(DEFERRED_PATH)


# ---------------------------------------------------------------------------
# Family 3 -- behavioural (green before and after; the migrated _conn works)
# ---------------------------------------------------------------------------


class TestStoresStillOpen:
    def test_peer_store_opens_and_counts(self, tmp_path):
        from opti_oignon.veilid.peers import PeerStore

        store = PeerStore(root=tmp_path)
        try:
            assert store.count() == 0
            assert store.journal_mode() == "wal"
            assert store.db_path.name == "veilid_peers.db"
        finally:
            store.close()

    def test_deferred_ledger_opens_and_counts(self, tmp_path):
        from opti_oignon.veilid.deferred_ledger import DeferredLedger

        ledger = DeferredLedger(root=tmp_path)
        try:
            assert ledger.count() == 0
            assert ledger.journal_mode() == "wal"
        finally:
            ledger.close()


# ---------------------------------------------------------------------------
# Family 4 -- the doc rolls and the RS1-D1 decision
# ---------------------------------------------------------------------------


class TestAtrestInventoryRolled:
    def test_closure_note_present(self):
        text = _flat(_read(ATREST_PATH))
        assert "PEER-01 / CHF-04 closure (S242" in text
        assert "RS1-D1" in text
        assert "joined safe_connect" in text
        assert "SYN-02 local device identity" in text

    def test_broader_chf04_class_kept_as_residual(self):
        text = _flat(_read(ATREST_PATH))
        assert "stays an RS-01-family residual for a later slice" in text


class TestRoadmapRolled:
    def test_security_routed_at_rest_bullet_rolled(self):
        text = _flat(_read(ROADMAP_PATH))
        assert "PEER-01 (plaintext peer registry -- LANDED at S242" in text
        assert "the deferred-ledger quarantine slice LANDED at S242" in text

    def test_residual_recorded(self):
        text = _flat(_read(ROADMAP_PATH))
        assert "stays an RS-01-family residual for a later slice" in text


class TestAuditDisposition:
    def test_peer01_closure_marked(self):
        text = _flat(_read(AUDIT_PATH))
        assert "LANDED at S242 (RS1-D1)" in text


# ---------------------------------------------------------------------------
# Family 5 -- AST validity
# ---------------------------------------------------------------------------


class TestASTValid:
    def test_edited_modules_parse(self):
        for path in (PEERS_PATH, DEFERRED_PATH):
            src = _read(path)
            assert src != "", str(path)
            ast.parse(src, filename=str(path))

    def test_this_suite_parses(self):
        src = _read(Path(__file__))
        assert src != ""
        ast.parse(src, filename=__file__)


# ---------------------------------------------------------------------------
# Family 6 -- the supersession reassert (deselect-plus-reassert counterpart)
# ---------------------------------------------------------------------------


class TestRegistryNoLongerPlainByDesign:
    """RS1-D1 supersedes test_s233 TestSeamPeers::test_registry_is_plain_sqlite.

    That test pinned the plaintext design ("the registry is plain sqlite"); it is
    deselected in pyproject addopts and its superseding truth is re-asserted here:
    the registry's primary connection is now safe_connect, and the only surviving
    sqlite3.connect is the documented ImportError fallback lambda.
    """

    def test_primary_path_is_safe_connect(self):
        src = _read(PEERS_PATH)
        assert "conn = _safe_connect(" in src
        assert "conn = sqlite3.connect(" not in src

    def test_only_surviving_sqlite3_connect_is_the_fallback(self):
        src = _read(PEERS_PATH)
        # exactly one residual sqlite3.connect( and it is the fallback lambda
        assert src.count("sqlite3.connect(") == 1
        assert "_safe_connect = lambda p, **kw: sqlite3.connect(str(p), **kw)" in src


# ---------------------------------------------------------------------------
# Family 7 -- the addopts lineage reassert (count grew by two at S242)
# ---------------------------------------------------------------------------


class TestAddoptsLineageS242:
    """206 deselects held from the S236 close through S241 (the mobile cycle's
    runbook/spec sessions added none). S242 adds exactly two and removes none
    (nothing is ever removed from the lineage): the RS1-D1 supersession of
    test_s233_cas7_spec.py::TestSeamPeers::test_registry_is_plain_sqlite (its
    plaintext-registry intent is overturned; the literal sqlite3.connect string
    survives only in the ImportError fallback, so it PASSES when re-exposed under
    -o addopts="") and the supersession of the now-stale absolute-count anchor
    test_s236_release.py::TestAddoptsLineage::test_count_grew_by_exactly_eight
    (== 206), which FAILS when re-exposed and joins the frozen set as the single
    documented net-new at S242.
    """

    def test_count_grew_by_exactly_two_to_208(self):
        src = _read(REPO / "pyproject.toml")
        assert src.count("--deselect=") == 208

    def test_carries_the_two_s242_supersessions(self):
        src = _read(REPO / "pyproject.toml")
        assert (
            "--deselect=tests/test_s233_cas7_spec.py::TestSeamPeers::test_registry_is_plain_sqlite"
            in src
        )
        assert (
            "--deselect=tests/test_s236_release.py::TestAddoptsLineage::test_count_grew_by_exactly_eight"
            in src
        )

    def test_prior_lineage_anchors_untouched(self):
        src = _read(REPO / "pyproject.toml")
        # nothing removed: the s232 anchor stays deselected as before
        assert (
            "--deselect=tests/test_s232_release.py::TestAddoptsLineage::test_count_grew_by_exactly_six"
            in src
        )
