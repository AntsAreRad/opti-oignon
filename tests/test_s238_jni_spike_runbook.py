"""S238 -- the Mobile app cycle's first implementation step: a doc-pin suite for
the veilid-core JNI integration spike runbook (MOBILE_JNI_SPIKE_S238.md).

This is the host-assured-runbook analogue of the S233/S237 spec-session suites.
S238 produces no source feature module: cas 7 already built and released the
whole desktop side the tier 1 phone client rides (the served handler, Option A
streaming, the per-device grant/revoke, PAIR-02, VL-01, the Bulbe gate, the
emergency stop), so the only container-provable artifact this step adds is the
directed host runbook for the native veilid-core JNI integration spike plus an
additive roadmap roll. The native Android/Rust work is host-assured and is never
simulated in this container.

Three families, the S237 idiom:

 1. The runbook -- existence, status (host-assured, findings-not-fixes,
    never-simulated-in-container), structure, companions, the binding-shape
    question (direct JNI vs a thin Rust JNI shim crate over veilid-core, decided
    at the spike's close as a decision id with the stack not reopened), the
    at-rest custody question, FLAG_SECURE / screen hardening, the tier 1 floor
    and the no-new-desktop-seam honesty (cas 7 complete; the mobile-allowed flag
    is a Notes-bloc item), the version held at 3.11.0 and the auth core edit-free.
 2. The roadmap roll -- the Mobile app cycle entry rolled additively to
    "implementation opened at S238" with the JNI spike runbook named, WITHOUT
    disturbing the S237 roll phrases, the s222-pinned ordering prefix, the
    s233-pinned mobile-entry header, or the AGT/governor/cas7 historical pins.
 3. The seams the phone is the client of -- source-level pins on the premises
    (the cas 7 served handler and Option A streaming, the per-device remote-chat
    grant and revoke, the PAIR-02 confirmation material, the VL-01 signer, the
    Bulbe binding-layer guard, the emergency stop), so a later edit that removes
    a premise turns this suite red instead of letting the runbook rot.

Red-before discipline: on the pristine S237 tree (no MOBILE_JNI_SPIKE_S238.md,
roadmap not rolled) every family-1 pin and the family-2 roll pins FAIL -- the
read helpers return empty strings so absence is a failure, never a collection
error -- while the family-2 guards and every family-3 seam pin pass by design
(they pin pre-existing invariants this step relies on). Document pins read
through a whitespace-flattening helper (the S221/S222/S233/S237 lesson) so line
reflow that does not change wording cannot break them; source pins stay raw.
Seams are read as text and AST-parsed; nothing here imports the package, so no
ollama chain is touched.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNBOOK_PATH = REPO / "MOBILE_JNI_SPIKE_S238.md"
ROADMAP_PATH = REPO / "ROADMAP_POST_AUDIT.md"
PKG = REPO / "opti_oignon"

SEAM_SOURCES = {
    "remote_inference": PKG / "veilid" / "remote_inference.py",
    "remote_streaming": PKG / "veilid" / "remote_streaming.py",
    "routes_sync": PKG / "api" / "routes_sync.py",
    "pairing": PKG / "veilid" / "pairing.py",
    "signing": PKG / "veilid" / "signing.py",
    "guard": PKG / "veilid" / "guard.py",
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
# Family 1 -- MOBILE_JNI_SPIKE_S238.md
# ---------------------------------------------------------------------------


class TestRunbookExists:
    def test_file_exists(self):
        assert RUNBOOK_PATH.exists(), "MOBILE_JNI_SPIKE_S238.md missing"

    def test_nonempty_and_titled(self):
        text = _read(RUNBOOK_PATH)
        assert text.startswith("# MOBILE_JNI_SPIKE_S238")
        assert "veilid-core JNI integration spike" in text
        assert "runbook + findings register" in text
        assert len(text) > 6000

    def test_status_and_discipline(self):
        text = _runbook()
        assert "written at S238" in text
        assert "the Mobile app cycle's first implementation step" in text
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
        "the binding shape",
        "at-rest custody",
        "FLAG_SECURE",
        "the decision at the spike's close",
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
            "REMOTE_INFERENCE_SPEC.md",
            "VEILID_SPEC.md",
            "HOST_SHAKEDOWN_S236.md",
        ):
            assert needle in text, needle


class TestBindingShapeQuestion:
    def test_two_candidate_shapes_framed(self):
        text = _runbook()
        assert "bind veilid-core directly over JNI" in text
        assert "a thin Rust JNI shim crate" in text

    def test_decided_at_close_as_decision_id(self):
        text = _runbook()
        assert "decided at the spike's close" in text
        assert "recorded as a decision id" in text
        assert "M1-D1" in text

    def test_stack_not_reopened(self):
        text = _runbook()
        assert "the stack is not reopened" in text
        assert "M0-D4 holds" in text


class TestAtRestCustody:
    def test_double_layer_mirrored(self):
        text = _runbook()
        assert "AES-256-GCM" in text
        assert "Argon2id" in text
        assert "StrongBox" in text
        assert "a user secret" in text

    def test_keystore_compromise_alone_opens_nothing(self):
        assert "a keystore compromise alone opens nothing" in _runbook()

    def test_no_google_backup_posture(self):
        text = _runbook()
        assert "allowBackup false" in text
        assert "extraction rules" in text

    def test_custody_shape_recorded(self):
        assert "M1-D2" in _runbook()


class TestScreenHardening:
    def test_flag_secure_and_lifecycle(self):
        text = _runbook()
        assert "FLAG_SECURE" in text
        assert "foreground re-auth" in text
        assert "evicted on background" in text


class TestTierOneFloorAndNoNewSeam:
    def test_tier_one_is_the_floor(self):
        text = _runbook()
        assert "the tier 1 bounded surface" in text
        assert "the floor" in text

    def test_phone_is_a_new_client_of_a_built_surface(self):
        text = _runbook()
        assert "a new client of an already-built desktop surface" in text
        assert "cas 7 is complete and released at S236" in text

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
        assert "M1-D1" in text
        assert "M1-D2" in text


# ---------------------------------------------------------------------------
# Family 2 -- the roadmap roll (additive)
# ---------------------------------------------------------------------------


class TestRoadmapRolled:
    def test_mobile_entry_rolled_to_implementation_opened(self):
        text = _roadmap()
        assert "implementation opened at S238" in text
        assert "the veilid-core JNI integration spike" in text
        assert "MOBILE_JNI_SPIKE_S238.md" in text

    def test_sequencing_line_references_the_spike(self):
        assert "JNI spike at S238" in _roadmap()


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


class TestSeamRemoteInference:
    def test_served_handler_present(self):
        src = _read(SEAM_SOURCES["remote_inference"])
        assert "def serve_remote_inference(" in src
        assert "def serve_remote_inference_continuation(" in src

    def test_bounded_surface_gate_present(self):
        src = _read(SEAM_SOURCES["remote_inference"])
        assert "def _enforce_bounded_surface(" in src
        assert "the tier 1 bounded surface" in src


class TestSeamRemoteStreaming:
    def test_option_a_pull_session_present(self):
        src = _read(SEAM_SOURCES["remote_streaming"])
        assert "def open_session(" in src
        assert "def pull(" in src
        assert "def kill_sessions_for_device(" in src

    def test_option_a_named(self):
        assert "Option A" in _read(SEAM_SOURCES["remote_streaming"])


class TestSeamGrants:
    def test_per_device_grant_and_revoke_present(self):
        src = _read(SEAM_SOURCES["routes_sync"])
        assert "def remote_chat_grant_payload(" in src
        assert "def revoke_remote_chat_payload(" in src

    def test_remote_chat_route_present(self):
        src = _read(SEAM_SOURCES["routes_sync"])
        assert '"/peers/{peer_id}/remote-chat"' in src


class TestSeamPairing:
    def test_pair02_confirmation_material_present(self):
        src = _read(SEAM_SOURCES["pairing"])
        assert "def pairing_canonical_material(" in src
        assert 'CONFIRM_CODE_SALT = b"oo-pairing-confirm-v1"' in src


class TestSeamSigning:
    def test_vl01_signer_and_verify_present(self):
        src = _read(SEAM_SOURCES["signing"])
        assert "class PqcRecordSigner" in src
        assert "def verify_record_signature(" in src

    def test_grace_window_closed_hard_constant(self):
        src = _read(SEAM_SOURCES["signing"])
        assert "ACCEPT_UNSIGNED_FROM_UNKEYED_ORIGINS = False" in src
        assert "ML-DSA-65" in src


class TestSeamGuard:
    def test_binding_layer_gate_present(self):
        src = _read(SEAM_SOURCES["guard"])
        assert "class VeilidDisabledInBulbe" in src
        assert "def assert_sync_allowed(" in src

    def test_no_bypass_parameter(self):
        assert "there is no parameter to bypass it" in _read(SEAM_SOURCES["guard"])


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
