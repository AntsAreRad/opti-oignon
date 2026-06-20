#!/usr/bin/env python3
"""S233 doc-pin suite: REMOTE_INFERENCE_SPEC.md + the roadmap roll + the seams.

A read-only spec session: cas 7 (remote inference delegation over Veilid)
gets its design contract at S233 and NO production code is edited. This
suite pins three families so a later session cannot drift silently:

1. The spec pins -- REMOTE_INFERENCE_SPEC.md exists and carries the
   arbitrated content: the structure, the tier-1 bounded trust boundary,
   the remote request lifecycle, the streaming-over-app_call framing (the
   central design point: chunked app_calls vs an app_message sequence,
   the lot decides), the authenticity inheritance (VL-01), the
   admission-no-bypass invariant, the containment invariants, the PEER-01
   precondition decision, the shared-with-mobile posture pointer, the
   delivery order with the container/host split, and the decisions
   record.
2. The roadmap roll -- the cas 7 entry is rolled to "spec WRITTEN at
   S233" and the sequencing line references the spec, WITHOUT disturbing
   the s222-pinned ordering prefix or the AGT/governor historical pins.
3. The seams the spec builds on -- source-level pins on the premises
   (the Bulbe binding-layer guard, the app_call transport and its
   responder, the per-record signer's encrypted-at-rest key custody, the
   peer store's public-key column and its plaintext-registry PEER-01
   note, the executor entry, the governor admission gate, the emergency
   stop), so a later edit that removes a premise turns this suite red
   instead of letting the spec rot.

Red-before discipline: on the pristine S232 tree (no REMOTE_INFERENCE_SPEC.md,
roadmap not rolled) every family-1 pin and the family-2 roll pins FAIL --
the read helpers return empty strings so absence is a failure, never a
collection error -- while the family-2 guards and every family-3 seam pin
pass by design (they pin pre-existing invariants the spec relies on).
Document pins read through a whitespace-flattening helper (the S221/S222
lesson) so line reflow that does not change wording cannot break them;
source pins stay raw. Seams are read as text and AST-parsed; nothing here
imports the package, so no ollama chain is touched.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO / "REMOTE_INFERENCE_SPEC.md"
ROADMAP_PATH = REPO / "ROADMAP_POST_AUDIT.md"
PKG = REPO / "opti_oignon"

SEAM_SOURCES = {
    "guard": PKG / "veilid" / "guard.py",
    "transport": PKG / "veilid" / "transport.py",
    "signing": PKG / "veilid" / "signing.py",
    "peers": PKG / "veilid" / "peers.py",
    "executor": PKG / "executor.py",
    "resource_governor": PKG / "resource_governor.py",
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


def _spec() -> str:
    return _flat(_read(SPEC_PATH))


def _roadmap() -> str:
    return _flat(_read(ROADMAP_PATH))


# ---------------------------------------------------------------------------
# Family 1 -- the spec pins
# ---------------------------------------------------------------------------


class TestSpecExists:
    def test_spec_file_exists(self):
        assert SPEC_PATH.exists(), "REMOTE_INFERENCE_SPEC.md missing"

    def test_spec_nonempty_and_titled(self):
        text = _read(SPEC_PATH)
        assert text.startswith("# REMOTE_INFERENCE_SPEC")
        assert len(text) > 15000

    def test_spec_status_decided(self):
        assert "Status: DECIDED" in _spec()

    def test_spec_is_a_s233_read_only_session_artifact(self):
        text = _spec()
        assert "written at S233" in text
        assert "read-only spec session" in text
        assert "nothing here is implemented at S233" in text

    def test_spec_pure_ascii_no_decoration(self):
        raw = _read(SPEC_PATH)
        assert raw != ""
        assert all(ord(c) < 128 for c in raw)
        assert "====" not in raw


class TestSpecStructure:
    REQUIRED = (
        "## 1. Executive Summary",
        "## 2. What exists today (do not rebuild)",
        "## 3. The trust boundary (tier 1 bounded surface)",
        "## 4. The remote request lifecycle",
        "## 5. The streaming design over app_call",
        "## 6. Authenticity inheritance (VL-01)",
        "## 7. Containment invariants",
        "## 8. The PEER-01 precondition decision",
        "## 9. Shared-with-mobile posture pointer",
        "## 10. Mode posture summary",
        "## 11. API surface",
        "## 12. Delivery order, container/host split, supersession forecast",
        "## 13. Decisions record",
        "## 14. Out of scope, risks, open questions",
        "## 15. Tests",
    )

    def test_all_required_sections_present(self):
        raw = _read(SPEC_PATH)
        missing = [h for h in self.REQUIRED if h not in raw]
        assert not missing, f"missing sections: {missing}"

    def test_companion_documents_named(self):
        text = _spec()
        for name in (
            "ROADMAP_POST_AUDIT.md",
            "ROADMAP_SYNC_CYCLE.md",
            "VEILID_SPEC.md",
            "RESOURCE_GOVERNOR_SPEC.md",
            "ATREST_INVENTORY.md",
            "SHAKEDOWN_S198_HANDOFF.md",
        ):
            assert name in text, name

    def test_spec_discipline_restated(self):
        text = _spec()
        assert "spec first, blocs after" in text
        assert "never simulated" in text
        assert "host-assured" in text


class TestTrustBoundary:
    def test_tier_one_bounded_surface_named(self):
        text = _spec()
        assert "tier 1 bounded surface" in text
        assert "inference + RAG read-only" in text

    def test_forbidden_surface_enumerated(self):
        text = _spec()
        assert "no state-mutation, sandbox, filesystem, shell, or config" in text

    def test_per_device_scoping_and_revocation(self):
        text = _spec()
        assert "per-device capability scoping" in text
        assert "instantly revocable" in text
        assert "RA-01" in text
        assert "the emergency stop" in text

    def test_default_tier_one_no_automatic_elevation(self):
        text = _spec()
        assert "the default is tier 1" in text
        assert "no automatic elevation" in text


class TestLifecycle:
    def test_lifecycle_arc(self):
        text = _spec()
        assert (
            "phone -> private route -> desktop executor/admission -> response"
            in text
        )

    def test_request_is_route_served_not_a_new_wire_route(self):
        text = _spec()
        assert "serve_app_call" in text
        assert "no new public HTTP route on the wire" in text


class TestStreamingDesignPoint:
    def test_central_design_point_named(self):
        text = _spec()
        assert "the central design point" in text
        assert "app_call" in text

    def test_both_options_framed(self):
        text = _spec()
        assert "chunked app_calls" in text
        assert "an app_message sequence" in text

    def test_the_lot_decides(self):
        text = _spec()
        assert "the implementation lot decides" in text
        assert "the spec frames both" in text

    def test_single_reply_today(self):
        text = _spec()
        assert "request-then-single-reply" in text


class TestAdmissionNoBypass:
    def test_same_funnel_as_local(self):
        text = _spec()
        assert "the same admission funnel as a local request" in text
        assert "no new bypass" in text

    def test_executor_entry_and_admit_named(self):
        text = _spec()
        assert "executor.execute" in text
        assert "admit()" in text
        assert "AdmissionDecision" in text

    def test_handler_never_calls_backend_directly(self):
        text = _spec()
        assert "never calls the backend directly" in text


class TestAuthenticityInheritance:
    def test_vl01_inherited_not_invented(self):
        text = _spec()
        assert "VL-01" in text
        assert "inherits" in text
        assert "invents no authenticity" in text

    def test_route_authenticated_peer_and_provenance(self):
        text = _spec()
        assert "route-authenticated peer" in text
        assert "provenance binding" in text

    def test_signature_facts(self):
        text = _spec()
        assert "ML-DSA-65" in text
        assert "ACCEPT_UNSIGNED" in text


class TestContainmentInvariants:
    def test_bulbe_means_nothing_remotely(self):
        text = _spec()
        assert "Bulbe means nothing remotely" in text
        assert "Daily-only" in text

    def test_binding_layer_refusal(self):
        text = _spec()
        assert "the guard refuses to bind" in text
        assert "fail-secure" in text

    def test_rate_limit_and_alert(self):
        text = _spec()
        assert "rate limiting" in text
        assert "refusal/alert" in text

    def test_everything_audit_chained(self):
        text = _spec()
        assert "audit-chained" in text or "hash-chain audit log" in text

    def test_stalled_peer_times_out(self):
        text = _spec()
        assert "VeilidTimeout" in text


class TestPeer01Decision:
    def test_peer01_is_not_a_precondition(self):
        text = _spec()
        assert "PEER-01" in text
        assert "not a precondition" in text

    def test_integrity_not_confidentiality_reasoning(self):
        text = _spec()
        assert "integrity, not confidentiality" in text

    def test_registry_holds_public_material_only(self):
        text = _spec()
        assert "only public material" in text
        assert "signing public key" in text

    def test_secret_already_encrypted_independently(self):
        text = _spec()
        assert "private signing key is already AES-256-GCM-encrypted" in text

    def test_routed_to_rs01_inherited_as_is(self):
        text = _spec()
        assert "RS-01" in text
        assert "inherits the registry as-is" in text


class TestSharedWithMobile:
    def test_tiered_model_lives_in_the_mobile_entry(self):
        text = _spec()
        assert "the tiered-trust model lives in the mobile entry" in text
        assert "tier 2" in text

    def test_this_spec_names_only_the_boundary(self):
        text = _spec()
        assert "this spec names the boundary" in text


class TestModePosture:
    def test_daily_only(self):
        assert "Daily-only" in _spec()

    def test_auth_core_untouched(self):
        assert "auth.py, auth_2fa.py) is untouched" in _spec()


class TestApiSurface:
    def test_syn06_auth_parity_inherited(self):
        text = _spec()
        assert "SYN-06" in text
        assert "auth parity" in text


class TestDelivery:
    def test_bloc0_then_lots_then_release(self):
        text = _spec()
        assert "Bloc 0 (this spec)" in text
        assert "the live blocs" in text
        assert "the release session" in text

    def test_container_host_split_named(self):
        raw = _read(SPEC_PATH)
        assert "Container-provable:" in raw
        assert "Host-assured" in raw

    def test_supersession_forecast_zero(self):
        text = _spec()
        assert "Supersession forecast: zero" in text

    def test_house_protocol_restated(self):
        text = _spec()
        assert "red-before" in text
        assert "deselect-plus-reassert" in text


class TestDecisionsRecord:
    def test_decisions_present(self):
        text = _spec()
        for marker in (
            "D1 -- the trust boundary",
            "D2 -- the streaming design",
            "D3 -- admission integration",
            "D4 -- the PEER-01 precondition",
            "D5 -- the lot cut",
        ):
            assert marker in text, marker

    def test_insertables_recorded(self):
        text = _spec()
        assert "GOV-W1" in text
        assert "FBK-01" in text


# ---------------------------------------------------------------------------
# Family 2 -- the roadmap roll (the new phrases: red before, green after)
# ---------------------------------------------------------------------------


class TestRoadmapRolled:
    def test_cas7_entry_rolled_to_spec_written(self):
        text = _roadmap()
        assert "spec WRITTEN at S233" in text
        assert "REMOTE_INFERENCE_SPEC.md is the design contract" in text

    def test_sequencing_line_references_the_spec(self):
        text = _roadmap()
        assert "spec at S233, REMOTE_INFERENCE_SPEC.md" in text


# ---------------------------------------------------------------------------
# Family 2 guards -- pins the roll must NOT break (green before and after)
# ---------------------------------------------------------------------------


class TestRoadmapGuards:
    def test_cas7_ordering_prefix_preserved(self):
        raw = _read(ROADMAP_PATH)
        governor = raw.find("3. Resource Governor cycle")
        agt = raw.find("4. Agent Performance cycle (AGT)")
        cas7 = raw.find("5. cas 7 -- remote inference delegation")
        assert 0 < governor < agt < cas7

    def test_mobile_entry_six_preserved(self):
        raw = _read(ROADMAP_PATH)
        assert "6. Mobile app cycle (Android first)" in raw

    def test_agt_historical_pin_untouched(self):
        text = _roadmap()
        assert "LANDED and RELEASED at S232 (v3.10.0)" in text
        assert "AGT_SPEC.md is the design contract" in text
        assert "spec WRITTEN at S222" in text

    def test_governor_historical_pin_untouched(self):
        text = _roadmap()
        assert "LANDED and RELEASED at S227 (v3.9.0)" in text
        assert "spec WRITTEN at S221" in text

    def test_prior_sequencing_reference_untouched(self):
        text = _roadmap()
        assert "spec at S222, AGT_SPEC.md" in text


# ---------------------------------------------------------------------------
# Family 3 -- the seams the spec builds on (green on pristine by design)
# ---------------------------------------------------------------------------


class TestSeamGuard:
    def test_binding_layer_gate_present(self):
        src = _read(SEAM_SOURCES["guard"])
        assert "def assert_sync_allowed" in src
        assert "def current_mode" in src

    def test_fail_secure_to_bulbe(self):
        src = _read(SEAM_SOURCES["guard"])
        assert "fail-secure" in src
        assert "class VeilidDisabledInBulbe" in src

    def test_no_bypass_parameter(self):
        src = _read(SEAM_SOURCES["guard"])
        assert "there is no parameter to bypass it" in src


class TestSeamTransport:
    def test_responder_present(self):
        src = _read(SEAM_SOURCES["transport"])
        assert "def serve_app_call" in src

    def test_app_call_messenger_present(self):
        src = _read(SEAM_SOURCES["transport"])
        assert "class ClientRouteMessenger" in src
        assert "app_call" in src

    def test_defensive_answer_decode(self):
        src = _read(SEAM_SOURCES["transport"])
        assert "def decode_answer" in src


class TestSeamSigning:
    def test_signer_classes_present(self):
        src = _read(SEAM_SOURCES["signing"])
        assert "class RecordSigner" in src
        assert "class PqcRecordSigner" in src

    def test_private_key_encrypted_at_rest(self):
        src = _read(SEAM_SOURCES["signing"])
        assert "AES-256-GCM" in src
        assert "SecureBytes" in src

    def test_verify_seam_present(self):
        src = _read(SEAM_SOURCES["signing"])
        assert "def verify_record_signature" in src


class TestSeamPeers:
    def test_signing_pub_column_public_only(self):
        src = _read(SEAM_SOURCES["peers"])
        assert "signing_pub TEXT" in src
        assert "class PeerStore" in src

    def test_peer01_at_rest_routed_to_rs01(self):
        src = _read(SEAM_SOURCES["peers"])
        assert "PEER-01" in src
        assert "the RS-01 lot's" in src

    def test_registry_is_plain_sqlite(self):
        src = _read(SEAM_SOURCES["peers"])
        assert "sqlite3.connect" in src


class TestSeamExecutorGovernorEstop:
    def test_executor_entry_present(self):
        src = _read(SEAM_SOURCES["executor"])
        assert "def execute(" in src

    def test_admission_gate_present(self):
        src = _read(SEAM_SOURCES["resource_governor"])
        assert "class AdmissionDecision" in src
        assert "def admit(" in src

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
        ast.parse(_read(Path(__file__)), filename=__file__)
