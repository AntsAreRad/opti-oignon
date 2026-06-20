#!/usr/bin/env python3
"""S237 doc-pin suite: MOBILE_THREAT_MODEL.md + ANDROID_APP_SPEC.md + the roll.

A read-only spec session, the Mobile app cycle's Bloc 0. The phone is the
user-facing client of cas 7 and a paired device under the sync cycle's
PAIR-02; Bloc 0 produces, IN ORDER, the threat model + security posture and
THEN the platform/stack decision -- security drives the stack choice. NO
production code is edited at S237. This suite pins three families so a later
session cannot drift silently:

1. The document pins -- MOBILE_THREAT_MODEL.md and ANDROID_APP_SPEC.md exist
   and carry the arbitrated content: the lower-trust device model, the four
   threats, the thin-client / containment / at-rest-double-layer / zero-Google
   / honest-residual principles, the tiered trust model (tier 1 default, tier 2
   desktop ceremony), attestation-as-warning, the surfaces the phone rides, and
   -- on the android side -- the posture constraints, the candidate evaluation,
   the security-first-and-performance-first stack decision (native Kotlin +
   JNI), the bounded integration spike, and the trust boundary inherited from
   cas 7 (consistent with REMOTE_INFERENCE_SPEC section 9, no drift).
2. The roadmap roll -- the Mobile app cycle entry is rolled to "spec WRITTEN at
   S237" with the Bloc 0 design contract named, WITHOUT disturbing the
   s222-pinned ordering prefix, the s233-pinned mobile-entry header, or the
   AGT/governor/cas7 historical pins.
3. The seams the spec builds on -- source-level pins on the premises (the cas 7
   served handler and Option A streaming, the per-device remote-chat grant and
   revoke, the PAIR-02 confirmation material, the VL-01 signer, the Bulbe
   binding-layer guard, the emergency stop), so a later edit that removes a
   premise turns this suite red instead of letting the spec rot.

Red-before discipline: on the pristine S236 tree (no MOBILE_THREAT_MODEL.md, no
ANDROID_APP_SPEC.md, roadmap not rolled) every family-1 pin and the family-2
roll pins FAIL -- the read helpers return empty strings so absence is a
failure, never a collection error -- while the family-2 guards and every
family-3 seam pin pass by design (they pin pre-existing invariants the spec
relies on). Document pins read through a whitespace-flattening helper (the
S221/S222/S233 lesson) so line reflow that does not change wording cannot break
them; source pins stay raw. Seams are read as text and AST-parsed; nothing here
imports the package, so no ollama chain is touched.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
THREAT_PATH = REPO / "MOBILE_THREAT_MODEL.md"
ANDROID_PATH = REPO / "ANDROID_APP_SPEC.md"
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


def _threat() -> str:
    return _flat(_read(THREAT_PATH))


def _android() -> str:
    return _flat(_read(ANDROID_PATH))


def _roadmap() -> str:
    return _flat(_read(ROADMAP_PATH))


# ---------------------------------------------------------------------------
# Family 1a -- MOBILE_THREAT_MODEL.md
# ---------------------------------------------------------------------------


class TestThreatModelExists:
    def test_file_exists(self):
        assert THREAT_PATH.exists(), "MOBILE_THREAT_MODEL.md missing"

    def test_nonempty_and_titled(self):
        text = _read(THREAT_PATH)
        assert text.startswith("# MOBILE_THREAT_MODEL")
        assert len(text) > 8000

    def test_status_decided(self):
        assert "Status: DECIDED" in _threat()

    def test_read_only_session_artifact(self):
        text = _threat()
        assert "written at S237" in text
        assert "read-only spec session" in text
        assert "nothing here is implemented at S237" in text

    def test_pure_ascii_no_decoration(self):
        raw = _read(THREAT_PATH)
        assert raw != ""
        assert all(ord(c) < 128 for c in raw)
        assert "====" not in raw


class TestThreatModelStructure:
    REQUIRED = (
        "## 1. Executive summary",
        "## 2. Threat model",
        "## 3. Security principles",
        "## 4. Tiered trust and capability elevation",
        "## 5. The surfaces the phone rides",
        "## 6. Companion documents",
        "## 7. Decisions record",
        "## 8. Container-provable versus host-assured",
        "## 9. Tests",
        "## 10. Out of scope, risks, open questions",
    )

    def test_all_required_sections_present(self):
        raw = _read(THREAT_PATH)
        missing = [h for h in self.REQUIRED if h not in raw]
        assert not missing, f"missing sections: {missing}"

    def test_companion_documents_named(self):
        text = _threat()
        for name in (
            "ROADMAP_POST_AUDIT.md",
            "REMOTE_INFERENCE_SPEC.md",
            "VEILID_SPEC.md",
            "NOTES_FEATURE_ROADMAP.md",
            "SHAKEDOWN_S198_HANDOFF.md",
            "ANDROID_APP_SPEC.md",
        ):
            assert name in text, name

    def test_discipline_restated(self):
        text = _threat()
        assert "spec first, the implementation blocs after" in text
        assert "host-assured" in text
        assert "never simulated" in text


class TestLowerTrustModel:
    def test_lower_trust_device_on_untrusted_os(self):
        text = _threat()
        assert "lower-trust device" in text
        assert "a not-trusted OS" in text

    def test_contains_the_blast_radius(self):
        assert "contains the blast radius" in _threat()


class TestThreats:
    def test_passive_os_and_google(self):
        text = _threat()
        assert "telemetry and cloud backup" in text
        assert "(passive)" in text

    def test_active_local_malware(self):
        text = _threat()
        assert "on-device malware" in text
        assert "(active local)" in text

    def test_at_rest_theft_or_seizure(self):
        text = _threat()
        assert "theft" in text
        assert "seizure" in text
        assert "(at-rest)" in text

    def test_pivot_to_the_desktop(self):
        assert "a compromised phone pivoting to the desktop" in _threat()


class TestSecurityPrinciples:
    def test_thin_client_not_replica(self):
        text = _threat()
        assert "Thin client, not replica" in text
        assert "the minimum plaintext" in text
        assert "display ephemerally" in text
        assert "not persisted by default" in text
        assert "mobile-allowed" in text

    def test_containment_is_the_top_property(self):
        text = _threat()
        assert "Containment is the top property" in text
        assert "a compromised phone must never pivot to the desktop" in text
        assert "per-device capability scoping" in text
        assert "instantly revocable" in text
        assert "RA-01" in text
        assert "the emergency stop" in text
        assert "Rate limiting" in text

    def test_at_rest_double_layer(self):
        text = _threat()
        assert "allowBackup false" in text
        assert "AES-256-GCM + Argon2id" in text
        assert "StrongBox" in text
        assert "FLAG_SECURE" in text
        assert "no plaintext temp files" in text

    def test_zero_google_by_construction(self):
        text = _threat()
        assert "Zero Google by construction" in text
        assert "outside the Play Store" in text
        assert "no Play Services" in text
        assert "reproducible builds" in text

    def test_honest_residual_risk(self):
        text = _threat()
        assert "OS-level surveillance is outside an app's control" in text
        assert "de-Googled OS" in text
        assert "the design minimizes, contains, and revokes" in text


class TestTieredTrust:
    def test_tier_one_default(self):
        text = _threat()
        assert "Tier 1 (default" in text
        assert "inference + RAG read-only" in text
        assert "No automatic elevation, ever" in text

    def test_tier_two_desktop_ceremony(self):
        text = _threat()
        assert "Tier 2 (elevated" in text
        assert "a deliberate elevation ceremony on the desktop" in text
        assert "visual code + password + 2FA" in text
        assert "revocable and time-boxed" in text

    def test_granular_least_privilege(self):
        text = _threat()
        assert "least-privilege" in text
        assert "never an all-access switch" in text
        assert "post-sandbox file validation" in text
        assert "a remote approval surface" in text

    def test_attestation_is_a_warning_not_a_gate(self):
        text = _threat()
        assert "Hardware attestation is a WARNING, not a condition" in text
        assert "it does not gate it" in text

    def test_invariants_under_elevation(self):
        text = _threat()
        assert "Bulbe means nothing, elevated or not" in text
        assert "Daily-only" in text
        assert "audit-chained" in text


class TestSurfacesRidden:
    def test_cas7_tier_one_surface(self):
        text = _threat()
        assert "the cas 7 tier 1 bounded surface" in text

    def test_pair02_and_notes_contract(self):
        text = _threat()
        assert "PAIR-02 mutual confirmation" in text
        assert "the Notes bloc's N.9" in text
        assert "phone-app sync contract" in text

    def test_container_host_split_stated(self):
        text = _threat()
        assert "container-provable" in text
        assert "host-assured" in text


class TestThreatModelDecisions:
    def test_trust_model_decisions_present(self):
        text = _threat()
        for marker in (
            "M0-D1",
            "M0-D2",
            "M0-D3",
        ):
            assert marker in text, marker


# ---------------------------------------------------------------------------
# Family 1b -- ANDROID_APP_SPEC.md
# ---------------------------------------------------------------------------


class TestAndroidSpecExists:
    def test_file_exists(self):
        assert ANDROID_PATH.exists(), "ANDROID_APP_SPEC.md missing"

    def test_nonempty_and_titled(self):
        text = _read(ANDROID_PATH)
        assert text.startswith("# ANDROID_APP_SPEC")
        assert len(text) > 8000

    def test_status_decided(self):
        assert "Status: DECIDED" in _android()

    def test_read_only_session_artifact(self):
        text = _android()
        assert "written at S237" in text
        assert "read-only spec session" in text
        assert "nothing here is implemented at S237" in text

    def test_pure_ascii_no_decoration(self):
        raw = _read(ANDROID_PATH)
        assert raw != ""
        assert all(ord(c) < 128 for c in raw)
        assert "====" not in raw


class TestAndroidSpecStructure:
    REQUIRED = (
        "## 1. Executive summary",
        "## 2. The posture constraints the stack must satisfy",
        "## 3. Candidate evaluation",
        "## 4. The decision",
        "## 5. The bounded integration spike",
        "## 6. The trust boundary inherited from cas 7",
        "## 7. The layers and the later blocs",
        "## 8. Companion documents",
        "## 9. Decisions record",
        "## 10. Delivery order, container/host split, supersession forecast",
        "## 11. Tests",
        "## 12. Out of scope, risks, open questions",
    )

    def test_all_required_sections_present(self):
        raw = _read(ANDROID_PATH)
        missing = [h for h in self.REQUIRED if h not in raw]
        assert not missing, f"missing sections: {missing}"

    def test_companion_documents_named(self):
        text = _android()
        for name in (
            "MOBILE_THREAT_MODEL.md",
            "ROADMAP_POST_AUDIT.md",
            "REMOTE_INFERENCE_SPEC.md",
            "VEILID_SPEC.md",
        ):
            assert name in text, name

    def test_security_drives_the_stack(self):
        assert "security drives the stack choice, not the reverse" in _android()


class TestPostureConstraints:
    def test_constraints_enumerated(self):
        text = _android()
        assert "reproducible builds" in text
        assert "no Play Services" in text
        assert "StrongBox" in text
        assert "FLAG_SECURE" in text
        assert "allowBackup false" in text
        assert "thin client" in text


class TestCandidateEvaluation:
    def test_three_candidates_named(self):
        text = _android()
        assert "Flutter + veilid-flutter" in text
        assert "Capacitor" in text
        assert "native Kotlin + JNI" in text

    def test_axis_is_security_and_performance(self):
        assert "security and performance" in _android()


class TestStackDecision:
    def test_chosen_stack_is_native_kotlin_jni(self):
        text = _android()
        assert "the chosen stack is native Kotlin + JNI" in text

    def test_security_first_and_performance_first(self):
        text = _android()
        assert "the most secure and the most performant" in text
        assert "smallest attack surface" in text

    def test_capacitor_rejected_on_webview_surface(self):
        text = _android()
        assert "Capacitor is rejected" in text
        assert "the WebView" in text
        assert "attack surface" in text

    def test_honest_residual_veilid_jni_binding(self):
        text = _android()
        assert "the Veilid JNI binding is the central" in text
        assert "a thin Rust JNI shim" in text
        assert "veilid-core" in text
        assert "the same Rust core the desktop uses" in text


class TestIntegrationSpike:
    def test_bounded_spike_on_the_jni_shim(self):
        text = _android()
        assert "a bounded integration spike" in text
        assert "the veilid-core JNI" in text
        assert "decided at the spike's close" in text


class TestTrustBoundaryInherited:
    def test_tier_one_is_the_floor(self):
        assert "the cas 7 tier 1 bounded surface is the floor" in _android()

    def test_tiered_model_lives_in_the_threat_model_no_drift(self):
        text = _android()
        assert "the tiered-trust model lives in MOBILE_THREAT_MODEL" in text
        assert "consistent with REMOTE_INFERENCE_SPEC" in text


class TestLayersAndBlocs:
    def test_three_layers_named(self):
        text = _android()
        assert "record sync" in text
        assert "remote chat" in text
        assert "the app itself" in text

    def test_later_blocs_named(self):
        text = _android()
        assert "pairing UX" in text
        assert "the chat client" in text
        assert "the sync client" in text

    def test_tier_two_lands_late_after_dependencies(self):
        text = _android()
        assert "tier 2" in text
        assert "late" in text
        assert "after the Sandbox Workspace and agent" in text


class TestAndroidSpecDecisions:
    def test_stack_decision_recorded(self):
        assert "M0-D4" in _android()


class TestAndroidSpecDelivery:
    def test_bloc0_then_the_implementation_blocs(self):
        text = _android()
        assert "Bloc 0" in text
        assert "the implementation blocs" in text

    def test_container_host_split_named(self):
        raw = _read(ANDROID_PATH)
        assert "Container-provable:" in raw
        assert "Host-assured" in raw

    def test_supersession_forecast_zero(self):
        assert "Supersession forecast: zero" in _android()

    def test_house_protocol_restated(self):
        text = _android()
        assert "red-before" in text
        assert "deselect-plus-reassert" in text


class TestAndroidSpecModePosture:
    def test_version_held(self):
        assert "the version is held at 3.11.0" in _android()

    def test_auth_core_edit_free(self):
        text = _android()
        assert "auth.py, auth_2fa.py" in text
        assert "edit-free" in text


# ---------------------------------------------------------------------------
# Family 2 -- the roadmap roll (new phrases: red before, green after)
# ---------------------------------------------------------------------------


class TestRoadmapRolled:
    def test_mobile_entry_rolled_to_spec_written(self):
        text = _roadmap()
        assert "spec WRITTEN at S237" in text
        assert "the Bloc 0 design contract" in text
        assert "MOBILE_THREAT_MODEL.md" in text

    def test_bloc0_scoped_marked(self):
        assert "Bloc 0 scoped" in _roadmap()

    def test_sequencing_line_references_the_spec(self):
        assert "spec at S237" in _roadmap()


# ---------------------------------------------------------------------------
# Family 2 guards -- pins the roll must NOT break (green before and after)
# ---------------------------------------------------------------------------


class TestRoadmapGuards:
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
# Family 3 -- the seams the spec builds on (green on pristine by design)
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
        ast.parse(_read(Path(__file__)), filename=__file__)
