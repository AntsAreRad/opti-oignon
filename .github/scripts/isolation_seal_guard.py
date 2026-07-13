#!/usr/bin/env python3
"""Isolation-seal guard: every test window must go through the shared one.

A contract suite that loads a module from its file has to manufacture a package
window, and hand-rolling that window is how an estate ends up green for the
wrong reason. Two independent routes let a real project module resolve behind a
test's back -- the module cache, which Python reads before it consults any
finder, and a finder that answers on the module NAME and ignores the parent
package's path, which is what an editable install registers. A window that
closes one route and not the other silently blocks nothing, and the suite then
draws its conclusion from an absence it never created.

The shared window in ``tests/_isolation.py`` closes both routes and PROVES on
every call that it did. This guard exists so no suite can quietly stop using
it: a file that builds a window of its own is a violation unless it is named in
the ledger below.

RATCHET. The ledger is the debt that predates the shared window. It may only
SHRINK: a suite comes off the list when it migrates, and nothing may ever be
added. A new suite therefore cannot be born with a void window, and the
existing debt is enumerated rather than merely suspected.

The pure helper ``find_violations`` is import-safe and unit-tested; ``main``
scans the test tree and exits non-zero on any violation.
Usage: ``isolation_seal_guard.py [TESTS_DIR]`` (default: tests/).
"""

import sys
from pathlib import Path

# A file builds an isolation window when it loads a module from a path or puts
# a finder ahead of the import machinery.
_WINDOW_MARKS = ("spec_from_file_location", "sys.meta_path")

# A file uses the shared window when it imports it.
_SHARED_MARK = "from _isolation import"

# Debt that predates the shared window. MAY ONLY SHRINK -- see RATCHET above.
LEDGER = frozenset({
    "test_adaptive_routing_bounds_contracts.py",
    "test_agent_run_model_capability_gate_contracts.py",
    "test_agent_run_teacher_wiring_contracts.py",
    "test_agentic_classifier_contracts.py",
    "test_agentic_summary_alignment_contracts.py",
    "test_agentic_thinking_guard_contracts.py",
    "test_agents_exec_surface_contracts.py",
    "test_auth_2fa_secret_at_rest_contracts.py",
    "test_auth_2fa_verification_bounds_contracts.py",
    "test_auto_capture.py",
    "test_auto_tuner_sweep_bounds_contracts.py",
    "test_backend_routing.py",
    "test_backend_routing_cache.py",
    "test_backup_encrypted_format_contracts.py",
    "test_backup_export_content_contracts.py",
    "test_backup_import_signature_rollback_contracts.py",
    "test_benchmark_evaluator_scoring_bounds_contracts.py",
    "test_benchmark_recommendation_selection_contracts.py",
    "test_benchmark_trigger_optin_bounds_contracts.py",
    "test_blob_framed.py",
    "test_blob_transfer.py",
    "test_capability_manifest_contracts.py",
    "test_capability_wiring_contracts.py",
    "test_chat_eval_harness_contracts.py",
    "test_chat_exec_failure_hint_contracts.py",
    "test_chat_fallback_decision_contracts.py",
    "test_chat_fr_hint_curve_contracts.py",
    "test_chat_retry_model_contracts.py",
    "test_coding_agent_apply_boundary_contracts.py",
    "test_context_manifest_pinning_contracts.py",
    "test_context_optimizer_preservation_contracts.py",
    "test_conversation_apply.py",
    "test_direct_answer_reuse_contracts.py",
    "test_fine_tune_export_escaping_contracts.py",
    "test_fine_tune_tracker_persistence_contracts.py",
    "test_humanizer_preservation_contracts.py",
    "test_learned_router_integrity_contracts.py",
    "test_luks_detector_contracts.py",
    "test_luks_no_false_positive_contracts.py",
    "test_memory_agent_untrusted_wrap_contracts.py",
    "test_memory_block.py",
    "test_memory_canonical_apply.py",
    "test_memory_canonical_sql_hygiene_contracts.py",
    "test_memory_curation_conservatism_contracts.py",
    "test_memory_dedup_coordination_contracts.py",
    "test_memory_dual_layer_invariant_contracts.py",
    "test_memory_extraction_bounds_contracts.py",
    "test_memory_migration.py",
    "test_memory_routes.py",
    "test_model_download_digest_pin_contracts.py",
    "test_model_download_ssrf_defense_contracts.py",
    "test_model_load_provenance_gate_contracts.py",
    "test_model_provenance_contracts.py",
    "test_note_update_store_contracts.py",
    "test_notes_apply.py",
    "test_notes_mobile_optin_contracts.py",
    "test_notes_send_half_contracts.py",
    "test_pipeline_persistence.py",
    "test_plugin_allowlist_contracts.py",
    "test_plugin_discovery_paths_contracts.py",
    "test_plugin_tool_network_capability_contracts.py",
    "test_plugin_worker_context_contracts.py",
    "test_plugin_worker_isolation_contracts.py",
    "test_prompt_budget_bounds_contracts.py",
    "test_public_clean_guard_contracts.py",
    "test_quick_sandbox_binding_contracts.py",
    "test_quick_sandbox_effective_id_contracts.py",
    "test_redteam_egress_guard_contracts.py",
    "test_redteam_feedback_apply_gate_contracts.py",
    "test_redteam_feedback_sanitization_shape_contracts.py",
    "test_redteam_report_permissions_contracts.py",
    "test_redteam_runner_no_egress_bypass_contracts.py",
    "test_redteam_scoring_classification_bounds_contracts.py",
    "test_redteam_scoring_faithfulness_contracts.py",
    "test_remote_inference_continuation_contracts.py",
    "test_remote_inference_contracts.py",
    "test_remote_inference_reauth_contracts.py",
    "test_remote_streaming_channel_contracts.py",
    "test_resource_governor_contracts.py",
    "test_resource_governor_gate_contracts.py",
    "test_resource_governor_queue_contracts.py",
    "test_response_cache_contracts.py",
    "test_response_cache_invalidation_contracts.py",
    "test_response_cache_warm_contracts.py",
    "test_robust_toolcalling.py",
    "test_router_requirement_propagation_contracts.py",
    "test_router_tool_calling_enforcement_contracts.py",
    "test_sandbox_egress_gate_contracts.py",
    "test_sandbox_rest_confinement_contracts.py",
    "test_sandbox_tools_confinement_contracts.py",
    "test_security_auth_failclosed_contracts.py",
    "test_self_correction_loop_contracts.py",
    "test_semantic_cache_contracts.py",
    "test_semantic_cache_fallback_contracts.py",
    "test_semantic_cache_management_contracts.py",
    "test_semantic_cache_no_model_lookup_contracts.py",
    "test_semantic_cache_semantic_tier_contracts.py",
    "test_session_fingerprint_bounds_at_rest_contracts.py",
    "test_skill_apply.py",
    "test_skill_consultation_untrusted_wrap_contracts.py",
    "test_skill_lifecycle_versioning_contracts.py",
    "test_skill_registry_path_confinement_contracts.py",
    "test_skill_teacher_publish_gate_contracts.py",
    "test_skill_write_gate_failsecure_contracts.py",
    "test_speculative_argv_materialisation_contracts.py",
    "test_stage_manifest_routing_contracts.py",
    "test_startup_boot_guard_contracts.py",
    "test_startup_checklist_contracts.py",
    "test_startup_swap_no_false_positive_contracts.py",
    "test_syn01_blob_vault_contracts.py",
    "test_syn01_receive_contracts.py",
    "test_sync_deferred_remote_channel_contracts.py",
    "test_sync_pairing_trust_contracts.py",
    "test_sync_run_wire_gate_contracts.py",
    "test_sync_service.py",
    "test_telemetry_profiler_consumer_contracts.py",
    "test_tool_loop_attribution_contracts.py",
    "test_tool_transcript_contracts.py",
    "test_tool_transcript_gate_contracts.py",
    "test_vault_manifest.py",
    "test_vector_store_health.py",
})


def builds_a_window(text):
    """True when the file manufactures a package window of its own."""
    return any(mark in text for mark in _WINDOW_MARKS)


def uses_shared_window(text):
    """True when the file goes through the shared isolation window."""
    return _SHARED_MARK in text


def find_violations(files):
    """Return the files that hand-roll a window and are not owed by the ledger.

    ``files`` is an iterable of (name, text) pairs. Pure and import-safe so it
    can be unit-tested without touching a filesystem.
    """
    violations = []
    for name, text in files:
        if not builds_a_window(text):
            continue
        if uses_shared_window(text):
            continue
        if name in LEDGER:
            continue
        violations.append(name)
    return sorted(violations)


def find_stale_ledger_entries(files):
    """Ledger names that no longer hand-roll a window: the ratchet has slipped.

    An entry that has migrated must come OFF the ledger, or the debt count
    stops meaning anything and a later regression could hide behind it.
    """
    seen = {name: text for name, text in files}
    stale = []
    for name in LEDGER:
        text = seen.get(name)
        if text is None:
            stale.append(name)
        elif uses_shared_window(text) or not builds_a_window(text):
            stale.append(name)
    return sorted(stale)


def main(argv):
    root = Path(argv[1]) if len(argv) > 1 else Path("tests")
    files = [
        (p.name, p.read_text(encoding="utf-8", errors="ignore"))
        for p in sorted(root.glob("test_*.py"))
    ]
    violations = find_violations(files)
    stale = find_stale_ledger_entries(files)

    if violations:
        print("Isolation-seal violations -- these suites build a window of")
        print("their own instead of the shared one in tests/_isolation.py:")
        for name in violations:
            print(f"  {name}")
        print()
        print("A hand-rolled window does not close both routes into the")
        print("package, and a suite that reasons about an absence it never")
        print("created proves nothing. Use the shared window.")
    if stale:
        print("Stale ledger entries -- these have migrated and must be removed")
        print("from LEDGER so the debt count stays honest:")
        for name in stale:
            print(f"  {name}")

    if violations or stale:
        return 1

    shared = sum(1 for _, text in files if uses_shared_window(text))
    print(f"Isolation seal OK: {shared} suites on the shared window, "
          f"{len(LEDGER)} owed. The ledger may only shrink.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
