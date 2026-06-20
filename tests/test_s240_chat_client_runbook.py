"""S240 -- the Mobile app cycle's chat client bloc: a doc-pin suite for the cas 7
tier 1 chat client runbook (MOBILE_CHAT_CLIENT_S240.md).

This is the host-assured-runbook analogue of the S238 JNI-spike suite and the
S239 pairing-UX suite. S240 produces no source feature module: cas 7 already
built and released the whole desktop side the tier 1 chat client rides (the
served handler, Option A streaming, the per-device grant/revoke, the channel
rate-limit and telemetry, PAIR-02, VL-01, the Bulbe gate, the emergency stop),
so the only container-provable artifact this step adds is the directed host
runbook for the native chat client plus an additive roadmap roll. The native
Android chat client and the live cas 7 round from the phone are host-assured and
are never simulated in this container.

The chat client is the next implementation step after the pairing UX (S239): the
cas 7 route opens to an already-paired device, so the chat client cannot start
before a device can pair. It is the first surface where the phone actually
borrows inference from the desktop over the live route, and it is the cas 7 tier
1 bounded surface (inference + RAG read-only), which is the floor.

Three families, the S238/S239 idiom:

 1. The runbook -- existence, status (host-assured, findings-not-fixes,
    never-simulated-in-container), structure, companions, the bounded-surface
    floor and the out-of-surface refusal, the Option A pull consumption and the
    per-chunk latency, the grant gate and the RAG read-only sub-grant, the
    mid-stream revoke buffer-kill and the unpair detach, the rate-limit breach
    and the physical Bulbe refusal, the ephemeral display and FLAG_SECURE on the
    chat screen, the two decisions (C1-D1 the chat-client streaming-consumption
    shape, C1-D2 the ephemeral-display posture) decided at the bloc's close with
    the stack / spike / pairing not reopened, the tier 1 floor and the
    no-new-desktop-seam honesty (cas 7 complete; the mobile-allowed flag a
    Notes-bloc item), the version held at 3.11.0 and the auth core edit-free.
 2. The roadmap roll -- the Mobile app cycle entry rolled additively to "the chat
    client bloc opened at S240" with the cas 7 tier 1 client runbook named,
    WITHOUT disturbing the S239/S238/S237 roll phrases, the s222-pinned ordering
    prefix, the s233-pinned mobile-entry header, or the AGT/governor/cas7
    historical pins.
 3. The seams the chat client is the client of -- source-level pins on the
    premises (the cas 7 served handler and Option A streaming, the per-device
    remote-chat grant and revoke, the channel rate-limit and telemetry, the
    PAIR-02 confirmation material, the VL-01 signer, the Bulbe binding-layer
    guard, the emergency stop), so a later edit that removes a premise turns this
    suite red instead of letting the runbook rot.

Red-before discipline: on the pristine S239 tree (no MOBILE_CHAT_CLIENT_S240.md,
roadmap not rolled) every family-1 pin and the family-2 roll pins FAIL -- the
read helpers return empty strings so absence is a failure, never a collection
error -- while the family-2 guards and every family-3 seam pin pass by design
(they pin pre-existing invariants this step relies on). Document pins read
through a whitespace-flattening helper (the S221/S222/S233/S237/S238 lesson) so
line reflow that does not change wording cannot break them; source pins stay raw
and are AST-parsed. Nothing here imports the package, so no ollama chain is
touched.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNBOOK_PATH = REPO / "MOBILE_CHAT_CLIENT_S240.md"
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
# Family 1 -- MOBILE_CHAT_CLIENT_S240.md
# ---------------------------------------------------------------------------


class TestRunbookExists:
    def test_file_exists(self):
        assert RUNBOOK_PATH.exists(), "MOBILE_CHAT_CLIENT_S240.md missing"

    def test_nonempty_and_titled(self):
        text = _read(RUNBOOK_PATH)
        assert text.startswith("# MOBILE_CHAT_CLIENT_S240")
        assert "cas 7 tier 1 chat client" in text
        assert "runbook + findings register" in text
        assert len(text) > 6000

    def test_status_and_discipline(self):
        text = _runbook()
        assert "written at S240" in text
        assert "the Mobile app cycle's chat client bloc" in text
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
        "the remote chat round",
        "the bounded surface",
        "the grant gate",
        "revocation",
        "the Bulbe refusal",
        "ephemeral display",
        "the decision at the bloc's close",
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
            "MOBILE_JNI_SPIKE_S238.md",
            "MOBILE_PAIRING_UX_S239.md",
        ):
            assert needle in text, needle


class TestBoundedSurfaceFloor:
    def test_tier_one_is_the_floor(self):
        text = _runbook()
        assert "the tier 1 bounded surface" in text
        assert "inference + RAG read-only" in text
        assert "the floor" in text

    def test_out_of_surface_field_refused(self):
        text = _runbook()
        assert "out-of-surface field" in text
        assert "refused, not silently dropped" in text


class TestStreamingConsumption:
    def test_option_a_pull_consumption(self):
        text = _runbook()
        assert "Option A" in text
        assert "app_call" in text
        assert "continuation" in text
        assert "cursor" in text
        assert "done marker" in text

    def test_per_chunk_latency_recorded(self):
        assert "per-chunk round-trip" in _runbook()


class TestGrantGate:
    def test_grant_gate_off_and_on(self):
        text = _runbook()
        assert "the grant gate" in text
        assert "structured refusal" in text
        assert "audit" in text

    def test_rag_read_only_subgrant(self):
        text = _runbook()
        assert "RAG read-only sub-grant" in text
        assert "rag_not_granted" in text


class TestRevocation:
    def test_mid_stream_revoke_buffer_kill(self):
        text = _runbook()
        assert "mid-stream" in text
        assert "killed" in text
        assert "the buffer is gone" in text

    def test_unpair_detach(self):
        text = _runbook()
        assert "the unpair detach" in text
        assert "the same way" in text


class TestRateLimitAndBulbe:
    def test_rate_limit_breach(self):
        text = _runbook()
        assert "rate limit" in text
        assert "structured refusal" in text
        assert "alert" in text

    def test_bulbe_refusal_physical(self):
        text = _runbook()
        assert "the node does not bind in Bulbe" in text
        assert "Daily-only" in text
        assert "cannot arrive" in text


class TestEphemeralDisplay:
    def test_ephemeral_not_persisted(self):
        text = _runbook()
        assert "display ephemerally" in text
        assert "not persisted by default" in text
        assert "thin client" in text

    def test_flag_secure_chat_screen_and_eviction(self):
        text = _runbook()
        assert "FLAG_SECURE" in text
        assert "the chat screen" in text
        assert "evicted on background" in text


class TestDecisionsRecord:
    def test_decided_at_close_with_ids(self):
        text = _runbook()
        assert "C1-D1" in text
        assert "C1-D2" in text
        assert "decided at the bloc's close" in text
        assert "recorded as a decision id" in text

    def test_stack_spike_pairing_not_reopened(self):
        text = _runbook()
        assert "the stack is not reopened" in text
        assert "M0-D4 holds" in text
        assert "the spike's M1-D1 is not reopened" in text
        assert "P1-D1" in text
        assert "P1-D2" in text


class TestNoNewSeamAndVersion:
    def test_phone_is_new_client_of_built_surface(self):
        text = _runbook()
        assert "a new client of an already-built desktop surface" in text
        assert "cas 7 is complete and released at S236" in text

    def test_mobile_allowed_flag_is_a_notes_bloc_item(self):
        text = _runbook()
        assert "the mobile-allowed per-item flag is a Notes-bloc item" in text
        assert "N.8" in text
        assert "N.9" in text

    def test_version_held(self):
        assert "held at 3.11.0" in _runbook()

    def test_auth_core_and_estop_edit_free(self):
        text = _runbook()
        assert "the auth core (auth.py, auth_2fa.py)" in text
        assert "emergency_stop.py" in text
        assert "edit-free" in text


# ---------------------------------------------------------------------------
# Family 2 -- the roadmap roll (additive)
# ---------------------------------------------------------------------------


class TestRoadmapRolled:
    def test_mobile_entry_rolled_to_chat_client(self):
        text = _roadmap()
        assert "the chat client bloc opened at S240" in text
        assert "the cas 7 tier 1 client" in text
        assert "MOBILE_CHAT_CLIENT_S240.md" in text

    def test_sequencing_line_references_chat_client(self):
        assert "chat client at S240" in _roadmap()


# ---------------------------------------------------------------------------
# Family 2 guards -- pins the roll must NOT break (green before and after)
# ---------------------------------------------------------------------------


class TestRoadmapGuards:
    def test_s239_roll_phrases_preserved(self):
        text = _roadmap()
        assert "the pairing UX bloc opened at S239" in text
        assert "MOBILE_PAIRING_UX_S239.md" in text
        assert "pairing UX at S239" in text

    def test_s238_roll_phrases_preserved(self):
        text = _roadmap()
        assert "implementation opened at S238" in text
        assert "the veilid-core JNI integration spike" in text
        assert "MOBILE_JNI_SPIKE_S238.md" in text
        assert "JNI spike at S238" in text

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
# Family 3 -- the seams the chat client is the client of (green on pristine)
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


class TestSeamRateLimitTelemetry:
    def test_per_device_rate_gate_present(self):
        src = _read(SEAM_SOURCES["remote_streaming"])
        assert "def check_rate(" in src
        assert "RATE_LIMIT_REQUESTS" in src
        assert "the channel telemetry" in src

    def test_telemetry_payload_and_route_present(self):
        routes = _read(SEAM_SOURCES["routes_sync"])
        assert "def remote_chat_telemetry_payload(" in routes
        assert '"/remote-chat/telemetry"' in routes


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

    def test_confirmation_code_present(self):
        assert "def confirmation_code(" in _read(SEAM_SOURCES["pairing"])


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
