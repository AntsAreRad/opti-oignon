"""S241 -- the Mobile app cycle's sync client bloc: a doc-pin suite for the
host-assured sync client runbook (MOBILE_SYNC_CLIENT_S241.md).

This is the host-assured-runbook analogue of the S238/S239/S240 implementation
suites. S241 produces no source feature module: the sync cycle already built and
released the whole desktop record surface the tier 1 phone client rides (records
encoding + integrity, the reconciler, the change feed and watermark, the protocol
envelope and its Daily-only wire boundary, the per-peer registry, the sync engine,
the per-record deferred ledger and its human approval gate), and PAIR-02 and VL-01
are built, so the only container-provable artifact this step adds is the directed
host runbook for the native sync client plus an additive roadmap roll. The native
Android sync client and the live sync round are host-assured and are never
simulated in this container.

Three families, the S238/S240 idiom:

 1. The runbook -- existence, status (host-assured, findings-not-fixes,
    never-simulated-in-container), structure, companions, the pull round and the
    monotonic watermark, VL-01 record authenticity from the phone (the hard
    constant closed), the PAIR-02 pending gate (PeerNotConfirmed / 409) and the
    re-pair demotion, the deferred ledger and the human approval gate (the gated
    wire apply vs the ungated local apply, refused-never-ledgered, unpair
    cascade), the Daily-only wire and the physical Bulbe refusal, the ephemeral
    display and FLAG_SECURE, the no-new-desktop-seam honesty (the sync cycle
    released; the mobile-allowed flag is a Notes-bloc item), the version held at
    3.11.0 and the auth core edit-free, and the two decision ids SY1-D1 / SY1-D2
    with the upstream stack not reopened.
 2. The roadmap roll -- the Mobile app cycle entry rolled additively to "the sync
    client bloc opened at S241" with the sync client runbook named, WITHOUT
    disturbing the S237/S238/S239/S240 roll phrases, the s222-pinned ordering
    prefix, the s233-pinned mobile-entry header, or the AGT/governor/cas7
    historical pins.
 3. The seams the phone is the client of -- source-level pins on the premises
    (the record encoding and integrity, the reconciler's LWW recipe and conflict
    log, the change feed delta, the protocol's gated wire apply and ungated local
    apply, the per-record deferred ledger and its unpair cascade, the per-peer
    registry's PAIR-02 confirm and monotonic watermark, the sync engine's
    PeerNotConfirmed gate and structured round result, the VL-01 signer and its
    hard constant, the PAIR-02 confirmation material, the Bulbe binding-layer
    guard, the deferred/run/pending routes, the emergency stop), so a later edit
    that removes a premise turns this suite red instead of letting the runbook
    rot.

Red-before discipline: on the pristine S240 tree (no MOBILE_SYNC_CLIENT_S241.md,
roadmap not rolled) every family-1 pin and the family-2 roll pins FAIL -- the read
helpers return empty strings so absence is a failure, never a collection error --
while the family-2 guards and every family-3 seam pin pass by design (they pin
pre-existing invariants this step relies on). Document pins read through a
whitespace-flattening helper (the S221/S222/S233/S237/S238 lesson) so line reflow
that does not change wording cannot break them; source pins stay raw. Seams are
read as text and AST-parsed; nothing here imports the package, so no ollama chain
is touched.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNBOOK_PATH = REPO / "MOBILE_SYNC_CLIENT_S241.md"
ROADMAP_PATH = REPO / "ROADMAP_POST_AUDIT.md"
PKG = REPO / "opti_oignon"

SEAM_SOURCES = {
    "records": PKG / "veilid" / "records.py",
    "reconcile": PKG / "veilid" / "reconcile.py",
    "change_feed": PKG / "veilid" / "change_feed.py",
    "protocol": PKG / "veilid" / "protocol.py",
    "deferred_ledger": PKG / "veilid" / "deferred_ledger.py",
    "peers": PKG / "veilid" / "peers.py",
    "sync_engine": PKG / "veilid" / "sync_engine.py",
    "signing": PKG / "veilid" / "signing.py",
    "pairing": PKG / "veilid" / "pairing.py",
    "guard": PKG / "veilid" / "guard.py",
    "routes_sync": PKG / "api" / "routes_sync.py",
    "emergency_stop": PKG / "emergency_stop.py",
}


def _read(path: Path) -> str:
    """Raw text; empty string when absent so absence FAILS, never errors."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _flat(text: str) -> str:
    """Collapse all whitespace runs to single spaces (reflow-immune pins)."""
    return re.sub(r"\s+", " ", text)


def _runbook() -> str:
    return _flat(_read(RUNBOOK_PATH))


def _roadmap() -> str:
    return _flat(_read(ROADMAP_PATH))


# ---------------------------------------------------------------------------
# Family 1 -- MOBILE_SYNC_CLIENT_S241.md
# ---------------------------------------------------------------------------


class TestRunbookExists:
    def test_file_exists(self):
        assert RUNBOOK_PATH.exists(), "MOBILE_SYNC_CLIENT_S241.md missing"

    def test_nonempty_and_titled(self):
        text = _read(RUNBOOK_PATH)
        assert text.startswith("# MOBILE_SYNC_CLIENT_S241")
        assert "the record surface under PAIR-02" in text
        assert "runbook + findings register" in text
        assert len(text) > 8000

    def test_status_and_discipline(self):
        text = _runbook()
        assert "written at S241" in text
        assert "the Mobile app cycle's sync client bloc" in text
        assert "host-assured" in text
        assert "produces findings, not fixes" in text
        assert "never simulated in the container" in text

    def test_pure_ascii_no_decoration(self):
        raw = _read(RUNBOOK_PATH)
        assert raw != ""
        assert all(ord(c) < 128 for c in raw)
        assert "====" not in raw


class TestRunbookStructure:
    REQUIRED = (
        "Preflight",
        "pull round and the watermark",
        "VL-01 record authenticity",
        "PAIR-02 pending gate",
        "deferred ledger and the human approval",
        "Daily-only wire",
        "FLAG_SECURE",
        "decision at the bloc's close",
        "Findings register",
        "Routing",
    )

    def test_all_required_sections_present(self):
        text = _runbook()
        for needle in self.REQUIRED:
            assert needle in text, needle

    def test_companion_documents_named(self):
        text = _runbook()
        for needle in (
            "ANDROID_APP_SPEC.md",
            "MOBILE_THREAT_MODEL.md",
            "VEILID_SPEC.md",
            "ROADMAP_SYNC_CYCLE.md",
            "MOBILE_CHAT_CLIENT_S240.md",
            "HOST_SHAKEDOWN_S236.md",
        ):
            assert needle in text, needle


class TestPullRoundAndWatermark:
    def test_watermark_monotonic(self):
        text = _runbook()
        assert "max(current, incoming)" in text
        assert "never regresses" in text

    def test_structured_summary(self):
        text = _runbook()
        for needle in ("applied", "deferred", "conflicts", "rejected"):
            assert needle in text, needle
        assert "the watermark before and after" in text

    def test_idempotent_round(self):
        assert "idempotent" in _runbook()


class TestRecordAuthenticityVL01:
    def test_mldsa65_signing(self):
        text = _runbook()
        assert "ML-DSA-65" in text
        assert "signing keypair" in text

    def test_canonical_bytes_bound(self):
        text = _runbook()
        assert "canonical_record_bytes" in text
        assert "ORIGIN device" in text

    def test_hard_constant_closed(self):
        text = _runbook()
        assert "ACCEPT_UNSIGNED_FROM_UNKEYED_ORIGINS" in text
        assert "hard constant" in text
        assert "False" in text

    def test_forged_record_refused(self):
        text = _runbook()
        assert "forged" in text
        assert "refused" in text


class TestPair02PendingGate:
    def test_pending_gate_409(self):
        text = _runbook()
        assert "PeerNotConfirmed" in text
        assert "409" in text

    def test_repair_demotion(self):
        text = _runbook()
        assert "changed signing key on re-pair demotes" in text
        assert "pending" in text


class TestDeferredLedgerGate:
    def test_same_human_gate(self):
        text = _runbook()
        assert "manage_skills" in text
        assert "manage_memory" in text

    def test_full_envelope_dedups(self):
        text = _runbook()
        assert "wire envelope" in text
        assert "dedups" in text

    def test_refused_not_ledgered(self):
        assert "do NOT enter the ledger" in _runbook()

    def test_unpair_cascades(self):
        assert "Unpairing a peer cascades" in _runbook()

    def test_gated_wire_vs_ungated_local(self):
        text = _runbook()
        assert "apply_record_batch" in text
        assert "apply_local_batch" in text


class TestDailyOnlyWireAndBulbe:
    def test_binding_layer_gate(self):
        text = _runbook()
        assert "binding-layer gate" in text
        assert "Daily-only" in text

    def test_bulbe_physical_refusal(self):
        text = _runbook()
        assert "the node does not bind in Bulbe" in text
        assert "no bypass parameter" in text


class TestEphemeralAndScreen:
    def test_in_memory_evicted(self):
        text = _runbook()
        assert "decrypted only in memory" in text
        assert "evicted on background" in text

    def test_provenance_only(self):
        assert "provenance only" in _runbook()

    def test_flag_secure_and_reauth(self):
        text = _runbook()
        assert "FLAG_SECURE" in text
        assert "foreground re-auth" in text


class TestNoNewSeamAndNotes:
    def test_new_client_of_built_surface(self):
        text = _runbook()
        assert "a new client of an already-built desktop surface" in text
        assert "the sync cycle" in text

    def test_mobile_allowed_flag_is_a_notes_bloc_item(self):
        text = _runbook()
        assert "the mobile-allowed per-item flag is a Notes-bloc item" in text
        assert "N.8" in text
        assert "N.9" in text


class TestVersionAndAuthCore:
    def test_version_held(self):
        assert "held at 3.11.0" in _runbook()

    def test_auth_core_and_estop_edit_free(self):
        text = _runbook()
        assert "the auth core (auth.py, auth_2fa.py)" in text
        assert "emergency_stop.py" in text
        assert "edit-free" in text


class TestDecisionsRecord:
    def test_decision_ids_present(self):
        text = _runbook()
        assert "SY1-D1" in text
        assert "SY1-D2" in text

    def test_upstream_not_reopened(self):
        text = _runbook()
        for needle in ("M0-D4", "M1-D1", "P1-D1", "P1-D2", "C1-D1", "C1-D2"):
            assert needle in text, needle


# ---------------------------------------------------------------------------
# Family 2 -- the roadmap roll (additive)
# ---------------------------------------------------------------------------


class TestRoadmapRolled:
    def test_mobile_entry_rolled_to_sync_client(self):
        text = _roadmap()
        assert "the sync client bloc opened at S241" in text
        assert "MOBILE_SYNC_CLIENT_S241.md" in text
        assert "the record surface under PAIR-02" in text

    def test_sequencing_line_references_the_sync_client(self):
        assert "sync client at S241" in _roadmap()


# ---------------------------------------------------------------------------
# Family 2 guards -- pins the roll must NOT break (green before and after)
# ---------------------------------------------------------------------------


class TestRoadmapGuards:
    def test_s237_roll_phrases_preserved(self):
        text = _roadmap()
        assert "spec WRITTEN at S237" in text
        assert "the Bloc 0 design contract" in text
        assert "Bloc 0 scoped" in text
        assert "spec at S237" in text

    def test_s238_roll_phrases_preserved(self):
        text = _roadmap()
        assert "implementation opened at S238" in text
        assert "the veilid-core JNI integration spike" in text
        assert "MOBILE_JNI_SPIKE_S238.md" in text
        assert "JNI spike at S238" in text

    def test_s239_roll_phrases_preserved(self):
        text = _roadmap()
        assert "the pairing UX bloc opened at S239" in text
        assert "MOBILE_PAIRING_UX_S239.md" in text
        assert "pairing UX at S239" in text

    def test_s240_roll_phrases_preserved(self):
        text = _roadmap()
        assert "the chat client bloc opened at S240" in text
        assert "MOBILE_CHAT_CLIENT_S240.md" in text
        assert "chat client at S240" in text

    def test_ordering_prefix_preserved(self):
        raw = _read(ROADMAP_PATH)
        governor = raw.find("3. Resource Governor cycle")
        agt = raw.find("4. Agent Performance cycle (AGT)")
        cas7 = raw.find("5. cas 7 -- remote inference delegation")
        assert 0 < governor < agt < cas7

    def test_mobile_entry_six_header_preserved(self):
        raw = _read(ROADMAP_PATH)
        assert "6. Mobile app cycle (Android first)" in raw

    def test_cas7_historical_pin_untouched(self):
        text = _roadmap()
        assert "LANDED and RELEASED at S236 (v3.11.0)" in text
        assert "spec WRITTEN at S233" in text

    def test_agt_historical_pin_untouched(self):
        text = _roadmap()
        assert "LANDED and RELEASED at S232 (v3.10.0)" in text
        assert "spec WRITTEN at S222" in text

    def test_governor_historical_pin_untouched(self):
        text = _roadmap()
        assert "LANDED and RELEASED at S227 (v3.9.0)" in text
        assert "spec WRITTEN at S221" in text

    def test_prior_sequencing_references_untouched(self):
        text = _roadmap()
        assert "spec at S222, AGT_SPEC.md" in text
        assert "spec at S233, REMOTE_INFERENCE_SPEC.md" in text


# ---------------------------------------------------------------------------
# Family 3 -- the seams the phone is the client of (green on pristine by design)
# ---------------------------------------------------------------------------


class TestSeamRecords:
    def test_record_encoding_and_integrity(self):
        src = _read(SEAM_SOURCES["records"])
        assert "def canonical_record_bytes(" in src
        assert "def verify_record_hash(" in src
        assert "def decode_record(" in src
        assert "class RecordKind" in src


class TestSeamReconcile:
    def test_lww_recipe_and_conflict_log(self):
        src = _read(SEAM_SOURCES["reconcile"])
        assert "def choose_winner(" in src
        assert "class ConflictEntry" in src
        assert "class MergeResult" in src


class TestSeamChangeFeed:
    def test_change_feed_delta(self):
        src = _read(SEAM_SOURCES["change_feed"])
        assert "class ChangeFeed" in src
        assert "class Delta" in src
        assert 'DB_FILENAME = "veilid_change_feed.db"' in src


class TestSeamProtocol:
    def test_delta_request_and_respond(self):
        src = _read(SEAM_SOURCES["protocol"])
        assert "def build_delta_request(" in src
        assert "def respond_to_request(" in src
        assert 'MSG_DELTA_REQUEST = "delta_request"' in src
        assert 'MSG_RECORD_BATCH = "record_batch"' in src

    def test_gated_wire_vs_ungated_local_apply(self):
        src = _read(SEAM_SOURCES["protocol"])
        assert "def apply_record_batch(" in src
        assert "def apply_local_batch(" in src


class TestSeamDeferredLedger:
    def test_ledger_offer_list_and_unpair_cascade(self):
        src = _read(SEAM_SOURCES["deferred_ledger"])
        assert "class DeferredLedger" in src
        assert "def offer(" in src
        assert "def list_entries(" in src
        assert "def remove_for_peer(" in src
        assert 'TABLE_NAME = "veilid_deferred_records"' in src


class TestSeamPeers:
    def test_registry_confirm_and_monotonic_watermark(self):
        src = _read(SEAM_SOURCES["peers"])
        assert "class PeerStore" in src
        assert "class PeerRecord" in src
        assert "def confirm_peer(" in src
        assert "def advance_watermark(" in src
        assert "def get_watermark(" in src


class TestSeamSyncEngine:
    def test_engine_pending_gate_and_round_result(self):
        src = _read(SEAM_SOURCES["sync_engine"])
        assert "class SyncEngine" in src
        assert "class PeerNotConfirmed(Exception)" in src
        assert "class RoundResult" in src


class TestSeamSigning:
    def test_vl01_signer_and_verify_present(self):
        src = _read(SEAM_SOURCES["signing"])
        assert "class PqcRecordSigner" in src
        assert "def verify_record_signature(" in src

    def test_grace_window_closed_hard_constant(self):
        src = _read(SEAM_SOURCES["signing"])
        assert "ACCEPT_UNSIGNED_FROM_UNKEYED_ORIGINS = False" in src
        assert "ML-DSA-65" in src


class TestSeamPairing:
    def test_pair02_confirmation_material_present(self):
        src = _read(SEAM_SOURCES["pairing"])
        assert "def pairing_canonical_material(" in src
        assert 'CONFIRM_CODE_SALT = b"oo-pairing-confirm-v1"' in src


class TestSeamGuard:
    def test_binding_layer_gate_present(self):
        src = _read(SEAM_SOURCES["guard"])
        assert "class VeilidDisabledInBulbe" in src
        assert "def assert_sync_allowed(" in src

    def test_no_bypass_parameter(self):
        assert "there is no parameter to bypass it" in _read(SEAM_SOURCES["guard"])


class TestSeamRoutesSync:
    def test_deferred_and_run_and_pending_routes(self):
        src = _read(SEAM_SOURCES["routes_sync"])
        assert '"/deferred"' in src
        assert '"/deferred/approve"' in src
        assert '"/deferred/refuse"' in src
        assert '"/peers/{peer_id}/run"' in src
        assert '"/pairing/pending"' in src


class TestSeamEmergencyStop:
    def test_estop_surface_present(self):
        src = _read(SEAM_SOURCES["emergency_stop"])
        for needle in ("def is_stopped()", "def guard_http()", "def refusal_payload()"):
            assert needle in src, needle


class TestASTValid:
    def test_seam_python_sources_parse(self):
        for name, path in SEAM_SOURCES.items():
            src = _read(path)
            assert src != "", name
            ast.parse(src, filename=str(path))

    def test_this_suite_parses(self):
        src = _read(Path(__file__))
        assert src != ""
        ast.parse(src, filename=__file__)
