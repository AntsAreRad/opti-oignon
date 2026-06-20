#!/usr/bin/env python3
"""S221 doc-pin suite: RESOURCE_GOVERNOR_SPEC.md + the roadmap roll + the seams.

A read-only spec session: no production code is edited at S221. This suite
pins three things so later sessions cannot drift silently:

1. The spec document itself -- its structure, the U1 input-contract respec,
   the arbitrated decisions (D1..D4), the per-bloc container/host split, and
   the R-04 no-behaviour-change clause.
2. The ROADMAP_POST_AUDIT roll -- the pre-U1 wording is gone, the spec is
   referenced from both the cycle entry and the sequencing list.
3. The verified seams the spec builds on -- source-level pins on
   emergency_stop, model_warmup, inference_backend, smart_router,
   speculative_decoding, and pipelines, so a later edit that removes a
   premise turns this suite red instead of letting the spec rot.

Red-before discipline: on the pristine S220 tree (no spec file, roadmap not
rolled) every test in the doc classes is red; the seam and AST classes are
green there by design (they pin pre-existing invariants the spec relies on).
Files are read lazily through _read() so a missing document yields failing
assertions, never collection errors.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPEC_PATH = REPO / "RESOURCE_GOVERNOR_SPEC.md"
ROADMAP_PATH = REPO / "ROADMAP_POST_AUDIT.md"
PKG = REPO / "opti_oignon"

SEAM_MODULES = {
    "emergency_stop": PKG / "emergency_stop.py",
    "model_warmup": PKG / "model_warmup.py",
    "inference_backend": PKG / "inference_backend.py",
    "smart_router": PKG / "smart_router.py",
    "speculative_decoding": PKG / "speculative_decoding.py",
    "pipelines": PKG / "pipelines.py",
}


def _read(path: Path) -> str:
    """Return the file text, or empty when absent (red, not an error)."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _flat(text: str) -> str:
    """Collapse all whitespace so phrase pins survive line reflow."""
    return " ".join(text.split())


def _spec() -> str:
    return _flat(_read(SPEC_PATH))


def _roadmap() -> str:
    return _flat(_read(ROADMAP_PATH))


# ---------------------------------------------------------------------------
# The spec document: existence and structure
# ---------------------------------------------------------------------------

class TestSpecExists:
    def test_spec_file_exists(self):
        assert SPEC_PATH.is_file(), "RESOURCE_GOVERNOR_SPEC.md missing"

    def test_spec_nonempty_and_titled(self):
        text = _spec()
        assert len(text) > 10000
        assert text.startswith("# RESOURCE_GOVERNOR_SPEC")

    def test_spec_status_decided(self):
        assert "Status: DECIDED" in _spec()

    def test_spec_is_a_s221_read_only_session_artifact(self):
        text = _spec()
        assert "Written at S221" in text
        assert "read-only spec session" in text


class TestSpecStructure:
    REQUIRED_SECTIONS = [
        "## 1. The input-contract respec (U1)",
        "## 2. What exists today",
        "## 3. Measurement layer: sources, ranking, fallback chain (Bloc 0)",
        "## 4. R-01 -- the admission gate (Bloc 1)",
        "## 5. R-02 -- runtime backpressure (Bloc 2)",
        "## 6. R-03 -- limit management (Bloc 3)",
        "## 7. Mode posture, security notes",
        "## 8. Caller map and the direct-caller residual",
        "## 9. API and frontend surface",
        "## 10. Configuration (config/resource_governor.yaml)",
        "## 11. Delivery order, per-bloc tests, supersession forecast",
        "## 12. Out of scope, risks, open questions",
        "## 13. Decisions record (S221 gate)",
        "## Appendix A -- host-bound probes (reference only, U1)",
    ]

    def test_all_required_sections_present(self):
        text = _spec()
        missing = [s for s in self.REQUIRED_SECTIONS if s not in text]
        assert not missing, f"missing sections: {missing}"

    def test_companion_documents_named(self):
        text = _spec()
        for companion in (
            "ROADMAP_POST_AUDIT.md",
            "AUDIT_FUNCTIONAL_FINDINGS.md",
            "SHAKEDOWN_S198_HANDOFF.md",
            "SANDBOX_WORKSPACE_SPEC.md",
            "SESSION_TRACKING_S65_S220.md",
        ):
            assert companion in text, f"companion not named: {companion}"

    def test_sandbox_spec_discipline_restated(self):
        text = _spec()
        assert "container-deliverable" in text
        assert "never simulated" in text
        assert "host-assured" in text.lower()


class TestInputContract:
    def test_runtime_self_measurement_is_the_contract(self):
        text = _spec()
        assert "runtime self-measurement" in text
        assert "never a pre-recorded register" in text

    def test_u1_named_and_register_declared_dead(self):
        text = _spec()
        assert "U1" in text
        assert "the shakedown register will never arrive" in text

    def test_no_hard_host_only_dependency(self):
        assert "NEVER a hard dependency on a host-only probe" in _spec()

    def test_measure_and_adapt_present(self):
        text = _spec()
        assert "measure-and-adapt" in text
        assert "a failed load lowers the learned" in text

    def test_nvidia_smi_class_is_reference_only(self):
        text = _spec()
        assert "nvidia-smi" in text
        assert "reference-only" in text


class TestMeasurementLayer:
    def test_source_ranking_s1_to_s4(self):
        text = _spec()
        assert "S1 -- the Ollama /api/ps view" in text
        assert "S2 -- the backend registry's own state" in text
        assert "S3 -- static estimation" in text
        assert "S4 -- capacity and host memory" in text

    def test_ps_view_reused_through_model_warmup(self):
        text = _spec()
        assert "get_loaded_models()" in text
        assert "CC-01" in text

    def test_static_table_reused_by_import_not_moved(self):
        text = _spec()
        assert "_VRAM_PER_BILLION_PARAMS" in text
        assert "reused BY IMPORT" in text
        assert "not moved and not duplicated" in text

    def test_s171_preflight_not_moved(self):
        text = _spec()
        assert "/proc/meminfo" in text
        assert "NOT moved out of smart_router" in text

    def test_snapshot_cached_with_ttl_and_eager_invalidation(self):
        text = _spec()
        assert "MUST NOT run synchronously" in text
        assert "invalidated eagerly" in text

    def test_no_audit_chain_on_the_admit_hot_path(self):
        assert "No audit-chain append happens on the per-request admit path" in _spec()

    def test_fail_open_arbitration_recorded(self):
        text = _spec()
        assert "fail-open with a logged warning" in text
        assert "lockout failure mode" in text

    def test_unknown_model_never_too_large(self):
        assert "never treated as too large" in _spec()

    def test_adapt_store_atrest_disposition_stated(self):
        text = _spec()
        assert "data/resource_governor.db" in text
        assert "regenerable" in text
        assert "excluded from backup" in text


class TestAdmissionGate:
    def test_dedicated_module_named(self):
        assert "opti_oignon/resource_governor.py" in _spec()

    def test_dual_seam_enforcement(self):
        text = _spec()
        assert "The mechanical seam" in text
        assert "The semantic seam" in text

    def test_backend_signatures_do_not_change(self):
        assert "Signatures of generate/stream DO NOT change" in _spec()

    def test_module_absent_means_unguarded_s216_posture(self):
        text = _spec()
        assert "proceed unguarded" in text
        assert "S216" in text

    def test_ctx_ladder_and_floor(self):
        text = _spec()
        assert "ctx ladder" in text or "config ladder" in text
        assert "per-caller floor" in text

    def test_first_num_ctx_ever_sent(self):
        assert 'options["num_ctx"]' in _spec()

    def test_num_gpu_conservative_partial_offload_deferred(self):
        text = _spec()
        assert "conservative in Bloc 1" in text
        assert "partial offload" in text

    def test_admission_ticket_shape(self):
        text = _spec()
        assert "AdmissionDecision" in text
        assert "ticket" in text

    def test_chat_downsize_then_refuse_never_silent_queue(self):
        text = _spec()
        assert "downsize-then-refuse" in text
        assert "Never queue silently on the interactive" in text

    def test_pipeline_abort_keeps_err_prefix(self):
        assert '"[ERR] Pipeline aborted:"' in _spec()

    def test_benchmark_never_silently_downsized(self):
        text = _spec()
        assert "NEVER silently downsized" in text
        assert "poisons its numbers" in text

    def test_agt_consumes_admitted_num_ctx_for_truncation_caps(self):
        assert "truncation caps" in _spec()


class TestR04Absorption:
    def test_stopped_flag_honoured_first(self):
        text = _spec()
        assert "honours the stopped flag FIRST" in text
        assert "is_stopped() checked before any fit math" in text

    def test_no_estop_behaviour_change(self):
        assert "No estop behaviour changes" in _spec()

    def test_existing_seams_only(self):
        text = _spec()
        assert "refusal_payload()" in text
        assert "guard_http()" in text

    def test_route_perimeter_and_s216_check_stay(self):
        text = _spec()
        assert "stay exactly as they are" in text
        assert "stays exactly as it is" in text


class TestBackpressure:
    def test_keep_alive_override_with_restore(self):
        text = _spec()
        assert "MUST restore it when pressure clears" in text
        assert "keepalive thread is never stopped by the governor" in text

    def test_targeted_eviction_audit_chained(self):
        text = _spec()
        assert "keep_alive=0" in text
        assert "Evictions are audit-chained" in text

    def test_queue_bounded_and_opt_in(self):
        text = _spec()
        assert "opt-in per caller" in text
        assert "Bounded in depth and wait" in text

    def test_queue_never_bypasses_estop(self):
        assert "never bypasses the estop check" in _spec()


class TestLimits:
    def test_ollama_env_limits_named(self):
        text = _spec()
        for var in ("OLLAMA_MAX_LOADED_MODELS", "OLLAMA_NUM_PARALLEL", "OLLAMA_MAX_QUEUE"):
            assert var in text, var

    def test_external_ollama_advisory_only_s145_precedent(self):
        text = _spec()
        assert "Advisory-only in all modes, never blocking startup" in text
        assert "S145" in text

    def test_rlimit_process_wide_caveat_and_off_by_default(self):
        text = _spec()
        assert "setrlimit is process-wide" in text
        assert "off-by-default" in text or "off by default" in text

    def test_cgroup_helper_host_bound_reference_only(self):
        text = _spec()
        assert "HOST-BOUND, reference only" in text
        assert "never executed by the application" in text


class TestModePosture:
    def test_governor_is_mode_free(self):
        text = _spec()
        assert "mode-free" in text
        assert "identically in Daily and Bulbe" in text

    def test_auth_core_untouched(self):
        assert "auth core (auth.py, auth_2fa.py) is untouched" in _spec()

    def test_kerckhoffs_note(self):
        assert "Kerckhoffs" in _spec()


class TestCallerMap:
    def test_funnels_listed(self):
        text = _spec()
        for funnel in (
            "executor.execute",
            "execute_simple",
            "execute_cascade",
            "execute_speculative",
            "PipelineRunner per step",
            "benchmark v2 runner",
        ):
            assert funnel in text, funnel

    def test_residual_named_and_queued_not_forgotten(self):
        text = _spec()
        assert "Named residual" in text
        assert "not silently forgotten" in text


class TestDelivery:
    def test_five_blocs_in_delivery_order(self):
        text = _spec()
        for marker in (
            "Bloc 0 -- measurement layer",
            "Bloc 1 -- R-01 + R-04",
            "Bloc 2 -- R-02",
            "Bloc 3 -- R-03",
            "Bloc 4 -- surfaces and close",
        ):
            assert marker in text, marker

    def test_every_bloc_names_container_and_host_split(self):
        text = _spec()
        assert text.count("Container-provable:") >= 5
        assert text.count("Host-assured (named):") >= 4

    def test_supersession_forecast_names_the_held_families(self):
        text = _spec()
        for family in ("s105", "s188", "s189", "test_s215_estop", "test_s171"):
            assert family in text, family

    def test_house_protocol_restated(self):
        text = _spec()
        assert "red-before pristine-proven" in text
        assert "deselect-plus-reassert" in text
        assert "ZERO failure-set delta" in text


class TestDecisionsRecord:
    def test_four_decisions_recorded(self):
        text = _spec()
        for marker in ("- D1 home:", "- D2 sources:", "- D3 defaults:", "- D4 blocs:"):
            assert marker in text, marker

    def test_d1_declined_alternative_recorded(self):
        assert "inference_backend-only extension declined" in _spec()


# ---------------------------------------------------------------------------
# The roadmap roll
# ---------------------------------------------------------------------------

class TestRoadmapRolled:
    def test_cycle_entry_rolled_to_spec_written(self):
        text = _roadmap()
        assert "spec WRITTEN at S221" in text
        assert "RESOURCE_GOVERNOR_SPEC.md is the design contract" in text

    def test_respec_recorded_in_the_entry(self):
        text = _roadmap()
        assert "respecified at S220/S221 onto runtime self-measurement" in text

    def test_r04_landed_wording(self):
        assert "landed at S215" in _roadmap()

    def test_old_spec_to_be_written_wording_gone(self):
        assert "spec to be written when scoped" not in _roadmap()

    def test_old_shakedown_inputs_wording_gone(self):
        assert "The shakedown's GPU/VRAM measurements are the spec's" not in _roadmap()

    def test_old_sequencing_wording_gone(self):
        assert "fed by the GPU/VRAM measurements" not in _roadmap()

    def test_sequencing_line_references_the_spec(self):
        assert "spec at S221, RESOURCE_GOVERNOR_SPEC.md" in _roadmap()


# ---------------------------------------------------------------------------
# The seams the spec builds on (source pins; green on pristine by design)
# ---------------------------------------------------------------------------

class TestSeamEmergencyStop:
    def test_flag_surface_present(self):
        src = _read(SEAM_MODULES["emergency_stop"])
        for needle in ("def is_stopped()", "def guard_http()", "def refusal_payload()"):
            assert needle in src, needle

    def test_flag_set_first_fail_secure(self):
        src = _read(SEAM_MODULES["emergency_stop"])
        assert "_stopped = True  # FIRST" in src

    def test_drain_unloads_every_registered_backend(self):
        src = _read(SEAM_MODULES["emergency_stop"])
        assert "registry.backends()" in src
        assert "unload_all" in src


class TestSeamModelWarmup:
    def test_ps_view_and_vram_summary(self):
        src = _read(SEAM_MODULES["model_warmup"])
        assert "def get_loaded_models(self)" in src
        assert "def get_vram_summary(self)" in src

    def test_dual_form_ps_handling(self):
        src = _read(SEAM_MODULES["model_warmup"])
        assert 'getattr(ps_response, "models"' in src

    def test_keep_alive_settable_property(self):
        src = _read(SEAM_MODULES["model_warmup"])
        assert "@keep_alive.setter" in src
        assert 'DEFAULT_KEEP_ALIVE = "30m"' in src

    def test_size_vram_field(self):
        assert "size_vram" in _read(SEAM_MODULES["model_warmup"])


class TestSeamInferenceBackend:
    def test_registry_snapshot_and_backends(self):
        src = _read(SEAM_MODULES["inference_backend"])
        assert "class BackendRegistry" in src
        assert "def backends(self)" in src

    def test_both_backends_carry_unload_all(self):
        src = _read(SEAM_MODULES["inference_backend"])
        assert src.count("def unload_all(self)") >= 2

    def test_ollama_eviction_idiom(self):
        assert "keep_alive=0" in _read(SEAM_MODULES["inference_backend"])

    def test_llama_cpp_fixed_ctx_at_constructor(self):
        src = _read(SEAM_MODULES["inference_backend"])
        assert "n_ctx: int = 4096" in src
        assert "n_gpu_layers" in src


class TestSeamSmartRouter:
    def test_ram_preflight_helpers(self):
        src = _read(SEAM_MODULES["smart_router"])
        assert "def _get_available_ram_mb()" in src
        assert "def _estimate_model_ram_mb(" in src

    def test_fail_open_documented(self):
        src = _read(SEAM_MODULES["smart_router"])
        assert "fail-open" in src

    def test_meminfo_source(self):
        assert "MemAvailable" in _read(SEAM_MODULES["smart_router"])


class TestSeamSpeculativeDecoding:
    def test_vram_table_and_calculator(self):
        src = _read(SEAM_MODULES["speculative_decoding"])
        assert "_VRAM_PER_BILLION_PARAMS" in src
        assert "class VRAMBudgetCalculator" in src

    def test_fit_check_present(self):
        src = _read(SEAM_MODULES["speculative_decoding"])
        assert "def check_fit(" in src
        assert "def estimate_model_vram(" in src


class TestSeamPipelines:
    def test_lazy_estop_resolver(self):
        src = _read(SEAM_MODULES["pipelines"])
        assert "def _resolve_emergency_stop()" in src

    def test_s216_inter_step_check(self):
        src = _read(SEAM_MODULES["pipelines"])
        assert "_estop.is_stopped()" in src
        assert "emergency stop engaged" in src


class TestASTValid:
    def test_all_seam_modules_parse(self):
        for name, path in SEAM_MODULES.items():
            src = _read(path)
            assert src, f"{name} unreadable"
            ast.parse(src)
