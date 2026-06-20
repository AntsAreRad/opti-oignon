#!/usr/bin/env python3
"""S222 doc-pin suite: AGT_SPEC.md + the roadmap roll + the seams.

Three families:

1. The spec pins -- AGT_SPEC.md exists and carries the arbitrated content:
   the opencode pin and MIT attribution, the ODYSSEUS drift note, the
   borrow-vs-reject verdicts, the three lots with their numbers, the
   governor one-way dependency, the harness honesty contract, Route A's
   binding conditions, the named supersession forecast, and D1-D4.
2. The ROADMAP_POST_AUDIT roll -- the AGT cycle entry is recorded between
   the governor and cas 7, the mobile entry renumbered, the sequencing
   line references the spec.
3. The seams the spec builds on -- source-level pins on the premises
   (allowlist frozensets, the seven schemas, the dispatch keys, the loop
   caps, the sandbox output caps and ro-binds, the benchmark runner
   idiom, the governor spec's AGT clauses), so a later edit that removes
   a premise turns this suite red instead of letting the spec rot.

Red-before discipline: on the pristine S221 tree (no AGT_SPEC.md, roadmap
not rolled) every family-1 and family-2 pin FAILS (the read helpers return
empty strings rather than raising, so absence is a failure, never an
error); every family-3 seam pin passes by design. Document pins read
through a whitespace-flattening helper (the S221 lesson) so line reflow
that does not change wording cannot break them; source pins stay raw.

Loaded file-by-file via ``spec_from_file_location`` with ``sys.modules``
package pre-seeding (the established isolation idiom; no ollama chain).
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO / "AGT_SPEC.md"
ROADMAP_PATH = REPO / "ROADMAP_POST_AUDIT.md"
GOVERNOR_SPEC_PATH = REPO / "RESOURCE_GOVERNOR_SPEC.md"

SEAM_SOURCES = {
    "allowlists": REPO / "opti_oignon" / "agent" / "allowlists.py",
    "tools": REPO / "opti_oignon" / "agent" / "tools.py",
    "dispatch": REPO / "opti_oignon" / "agent" / "dispatch.py",
    "loop": REPO / "opti_oignon" / "agent" / "loop.py",
    "untrusted": REPO / "opti_oignon" / "agent" / "untrusted_context.py",
    "tool_parsing": REPO / "opti_oignon" / "agent" / "tool_parsing.py",
    "sandbox_manager": REPO / "opti_oignon" / "sandbox_manager.py",
    "sandbox_tools": REPO / "opti_oignon" / "sandbox_tools.py",
    "benchmark_runner": REPO / "opti_oignon" / "benchmark_runner.py",
    "routes_benchmark_v2": REPO / "opti_oignon" / "api" / "routes_benchmark_v2.py",
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


def _governor() -> str:
    return _flat(_read(GOVERNOR_SPEC_PATH))


def _seed_packages() -> None:
    if "opti_oignon" not in sys.modules:
        pkg = types.ModuleType("opti_oignon")
        pkg.__path__ = [str(REPO / "opti_oignon")]
        sys.modules["opti_oignon"] = pkg
    if "opti_oignon.agent" not in sys.modules:
        apkg = types.ModuleType("opti_oignon.agent")
        apkg.__path__ = [str(REPO / "opti_oignon" / "agent")]
        sys.modules["opti_oignon.agent"] = apkg


def _load(name: str, path: Path):
    _seed_packages()
    full = f"opti_oignon.agent.{name}" if "agent" in str(path) else f"opti_oignon.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Family 1 -- the spec pins
# ---------------------------------------------------------------------------


class TestSpecExists:
    def test_spec_file_exists(self):
        assert SPEC_PATH.exists()

    def test_spec_nonempty_and_titled(self):
        text = _read(SPEC_PATH)
        assert text.startswith("# AGT_SPEC")
        assert len(text) > 20000

    def test_spec_is_a_s222_read_only_session_artifact(self):
        text = _spec()
        assert "written at S222" in text
        assert "read-only against production code" in text
        assert "nothing here is implemented at S222" in text

    def test_spec_pure_ascii_no_decoration(self):
        raw = _read(SPEC_PATH)
        assert raw != ""
        assert all(ord(c) < 128 for c in raw)
        assert "====" not in raw


class TestSpecStructure:
    REQUIRED = (
        "## 1. Executive Summary",
        "## 2. Reference Read: opencode at the Pin",
        "## 3. Existing Agent Surface",
        "## 4. The Borrow-vs-Reject Map",
        "## 5. Lot 1 -- Tools and Feedback",
        "## 6. Lot 2 -- Loop Hardening",
        "## 7. Lot 3 -- The Micro-task Eval Harness",
        "## 8. Route A",
        "## 9. Mode posture summary",
        "## 10. API surface",
        "## 11. Delivery order, container/host split, supersession forecast",
        "## 12. Decisions record",
        "## 13. Out of scope, risks, open questions",
        "## 14. Tests",
    )

    def test_all_required_sections_present(self):
        raw = _read(SPEC_PATH)
        for header in self.REQUIRED:
            assert header in raw, header

    def test_companion_documents_named(self):
        text = _spec()
        for name in (
            "ODYSSEUS_SPEC.md",
            "RESOURCE_GOVERNOR_SPEC.md",
            "SANDBOX_WORKSPACE_SPEC.md",
            "AUDIT_FUNCTIONAL_FINDINGS.md",
            "ROADMAP_POST_AUDIT.md",
        ):
            assert name in text, name

    def test_spec_discipline_restated(self):
        text = _spec()
        assert "spec first, blocs after" in text
        assert "never simulated" in text


class TestReferencePin:
    def test_repository_and_commit_pinned(self):
        text = _spec()
        assert "sst/opencode" in text
        assert "4519a1da329c1a4fc384054e7203ba7d06928205" in text
        assert "2026-06-06" in text

    def test_version_and_license_pinned(self):
        text = _spec()
        assert "1.16.2" in text
        assert "MIT" in text

    def test_attribution_travels_with_lots(self):
        text = _spec()
        assert "travels with any implementation lot" in text
        assert "No opencode code is copied" in text


class TestOdysseusDrift:
    def test_drift_supersedes_sections_22_23(self):
        text = _spec()
        assert "Odysseus 1.0" in text
        assert "no longer describes the live Odysseus agent" in text
        assert "no longer exists in the live agent" in text

    def test_historical_analysis_stands(self):
        text = _spec()
        assert "remains the S172 historical analysis" in text

    def test_three_format_parser_named_as_asset(self):
        text = _spec()
        assert "has no upstream counterpart to borrow from" in text
        assert "Nothing in this cycle weakens it" in text


class TestBorrowMap:
    def test_adapt_verdicts(self):
        text = _spec()
        for needle in (
            "| grep tool | adapt |",
            "| glob tool | adapt |",
            "| list (ls) | adapt |",
            "| diagnostics-after-write | adapt |",
            "| task (subagent) | adapt |",
            "| session compaction | adapt |",
        ):
            assert needle in text, needle

    def test_borrow_verdicts(self):
        text = _spec()
        for needle in (
            "| todowrite | borrow |",
            "| doom-loop detection | borrow |",
            "| output truncation + spill | borrow |",
            "| max-steps reminder | borrow |",
        ):
            assert needle in text, needle

    def test_reject_verdicts(self):
        text = _spec()
        for needle in (
            "| permission model | reject |",
            "| apply_patch | reject |",
            "| lsp tool | reject |",
            "| question tool | reject |",
            "| plan mode | reject |",
            "| webfetch / websearch split | reject |",
            "| native function calling only | reject |",
        ):
            assert needle in text, needle

    def test_already_held_verdicts(self):
        text = _spec()
        assert "| skill tool | already held |" in text
        assert "| invalid-tool feedback | already held |" in text

    def test_edit_recovery_subset_with_rejections_named(self):
        text = _spec()
        assert "| edit recovery chain | adapt, subset |" in text
        for rejected in (
            "BlockAnchor",
            "ContextAware",
            "MultiOccurrence",
            "TrimmedBoundary",
            "EscapeNormalized",
        ):
            assert rejected in text, rejected
        assert "REJECTED" in text


class TestLot1Tools:
    def test_sandbox_tool_names_grows_to_seven(self):
        text = _spec()
        assert (
            "SANDBOX_TOOL_NAMES grows to {bash, view, create_file, "
            "str_replace, grep, glob, ls}" in text
        )

    def test_schemas_grow_seven_to_twelve(self):
        text = _spec()
        assert "ALL_SCHEMAS grows from seven to twelve" in text

    def test_common_rules(self):
        text = _spec()
        assert "Active session required" in text
        assert "validate_sandbox_path" in text
        assert "null-byte sniff" in text

    def test_grep_contract(self):
        text = _spec()
        assert "max_results (default 100, hard cap 500)" in text
        assert "re.error returned as a structured" in text

    def test_glob_contract(self):
        text = _spec()
        assert "max_results (default 200, hard cap 1000)" in text
        assert "sorted by mtime" in text

    def test_ls_naming_decision(self):
        text = _spec()
        assert "named ls to avoid colliding with the legacy list_files" in text


class TestDiagnostics:
    def test_inside_the_sandbox_clause(self):
        text = _spec()
        assert "INSIDE the disposable sandbox" in text
        assert "never the host" in text

    def test_linter_ladder(self):
        text = _spec()
        assert "ruff check --quiet, else pyflakes, else python3 -m py_compile" in text

    def test_findings_only_byte_identity(self):
        text = _spec()
        assert "ONLY when findings exist" in text
        assert "A clean write returns byte-identical output to today" in text

    def test_bwrap_only_and_silent_skip(self):
        text = _spec()
        assert "only when the session's isolation backend is bwrap" in text
        assert "silent skip at logger.debug" in text

    def test_audit_chained(self):
        text = _spec()
        assert "the audit chain sees diagnostics runs like any other execution" in text

    def test_no_lsp_no_daemons(self):
        text = _spec()
        assert "no LSP servers, no daemons" in text


class TestTodoTask:
    def test_todo_no_persistence_no_atrest(self):
        text = _spec()
        assert "No persistence: nothing at rest, no ATREST row" in text

    def test_todo_both_modes_and_bulbe_handler_consequence(self):
        text = _spec()
        assert "present in BOTH modes" in text
        assert "Bulbe gains its first tool handler" in text

    def test_task_depth_one_no_recursion(self):
        text = _spec()
        assert "depth 1 (the child tool set excludes task -- no recursion)" in text

    def test_task_child_cap_formula(self):
        text = _spec()
        assert (
            "child cap = min(requested, TASK_CHILD_CAP = 6, "
            "parent_rounds_remaining - 1)" in text
        )

    def test_task_debit_rule(self):
        text = _spec()
        assert "DEBITED from the parent's remaining budget" in text

    def test_task_child_toolset_sandbox_only(self):
        text = _spec()
        assert "no network, no state mutation, no nested task" in text
        assert "shares the parent's SandboxToolSession" in text

    def test_verifier_stays(self):
        text = _spec()
        assert "_run_verifier remains the end-of-run check" in text


class TestLot2Hardening:
    def test_static_caps_numbers(self):
        text = _spec()
        assert "AGENT_OBS_MAX_BYTES = 16384" in text
        assert "AGENT_OBS_MAX_LINES = 256" in text
        assert "AGENT_ROUND_OBS_BUDGET = 49152" in text

    def test_spill_inside_workspace(self):
        text = _spec()
        assert ".agent/spill/" in text
        assert "The model can view / grep the spill" in text

    def test_copy_out_exclusion_cross_cycle_named(self):
        text = _spec()
        assert "the copy-out diff EXCLUDES it by a manifest rule" in text
        assert "Cross-cycle touch, named once" in text

    def test_prune_numbers_and_protection(self):
        text = _spec()
        assert "PRUNE_TRIGGER_CHARS (static default 98304" in text
        assert "PRUNE_TARGET_CHARS (default 65536)" in text
        assert "PRUNE_PROTECT_ROUNDS = 3" in text
        assert "the system prompt, the original task message" in text
        assert "skill_message blocks" in text

    def test_summarize_stage_flag_off(self):
        text = _spec()
        assert "OFF by default" in text
        assert "summarize_compaction: false" in text

    def test_doom_threshold_and_branches(self):
        text = _spec()
        assert "DOOM_LOOP_THRESHOLD = 3" in text
        assert "canonical JSON of arguments" in text
        assert "deny aborts the run" in text
        assert 'reason "doom_loop"' in text

    def test_recovery_three_replacers_one_candidate(self):
        text = _spec()
        assert "LineTrimmed" in text
        assert "WhitespaceNormalized" in text
        assert "IndentationFlexible" in text
        assert "ONLY when it yields exactly one candidate region" in text

    def test_recovery_miss_hint_and_strategy_visibility(self):
        text = _spec()
        assert "closest lines by difflib similarity" in text
        assert "the output names the strategy" in text

    def test_max_steps_reminder(self):
        text = _spec()
        assert "rounds remaining falls to 2" in text


class TestGovernorContract:
    def test_dependency_is_one_way(self):
        text = _spec()
        assert (
            "The dependency is one-way: this lot consumes the governor's "
            "output and provides nothing the governor needs" in text
        )

    def test_statics_are_floors(self):
        text = _spec()
        assert "the floor values thereafter" in text
        assert "never below the static floors" in text

    def test_ticket_formula(self):
        text = _spec()
        assert "budget_chars = admitted_num_ctx * 4 * obs_fraction" in text
        assert "default 0.35" in text

    def test_echoes_governor_42_clause(self):
        text = _spec()
        assert (
            "The same admitted value is the one the AGT loop-hardening "
            "lot's truncation caps consume" in text
        )

    def test_absent_ticket_statics_hold(self):
        text = _spec()
        assert "the static values hold" in text
        assert "pin both branches" in text


class TestHarness:
    def test_micro_suite_twelve_tasks(self):
        text = _spec()
        assert "micro.yaml is the v1 suite, 12 tasks" in text

    def test_auto_scored_by_tests_passing(self):
        text = _spec()
        assert "Scoring is auto, by tests passing" in text
        assert "no judge model, no rubric" in text

    def test_fresh_session_per_task(self):
        text = _spec()
        assert "a FRESH SandboxToolSession (disposable per" in text

    def test_admission_refuse_or_skip_never_silent_downsize(self):
        text = _spec()
        assert 'failure_class "not_admitted" and SKIPS' in text
        assert "never a silent downsize" in text
        assert "silently altered num_ctx poisons the numbers" in text

    def test_evict_between_models(self):
        text = _spec()
        assert "Evict-between-models is the runner's default" in text

    def test_honest_degradation_without_governor(self):
        text = _spec()
        assert "governor_present false" in text
        assert 'admitted "absent"' in text
        assert "visible, never masked" in text

    def test_dedicated_store_separate_from_benchmark(self):
        text = _spec()
        assert "data/agent_eval_results.db" in text
        assert "Kept SEPARATE from benchmark_results.db" in text

    def test_atrest_disposition_declared(self):
        text = _spec()
        assert "scope single-user, wipe pending-scoping" in text
        assert "backup excluded" in text

    def test_one_command_host_entry(self):
        text = _spec()
        assert "scripts/run_agent_eval.sh" in text
        assert "ONE command on the host" in text

    def test_routes_contract(self):
        text = _spec()
        assert "/api/agent-eval/run (409 when busy" in text

    def test_opencode_baseline_reference_only(self):
        text = _spec()
        assert 'engine "opencode-baseline"' in text
        assert "not a CI job, not a container path, never simulated" in text


class TestRouteA:
    def test_fallback_only_on_measured_gap(self):
        text = _spec()
        assert "explicitly-arbitrated fallback SPIKE" in text
        assert "ONLY if the AGT lots leave a measured gap" in text

    def test_binding_conditions(self):
        text = _spec()
        assert "Pinned opencode version" in text
        assert "LSP auto-download DISABLED and plugins DISABLED" in text
        assert "Egress allowlisted to the local Ollama endpoint ONLY" in text

    def test_posture_change_has_own_arbitration(self):
        text = _spec()
        assert "deliberate S73/S74 posture change" in text
        assert "its OWN arbitration" in text

    def test_spike_not_lot(self):
        text = _spec()
        assert "A spike, not a lot" in text


class TestModePosture:
    def test_bulbe_stays_a_derivation(self):
        text = _spec()
        assert "BULBE_ALLOWLIST stays a derivation" in text

    def test_per_call_approval_covers_new_surface(self):
        text = _spec()
        assert "including the new three, every child-task call" in text

    def test_fail_secure_defaults_hold(self):
        text = _spec()
        assert "unknown mode is Bulbe; no session means" in text

    def test_auth_core_untouched(self):
        text = _spec()
        assert "auth.py, auth_2fa.py) is untouched by every lot" in text


class TestDelivery:
    def test_per_lot_container_host_split_named(self):
        raw = _read(SPEC_PATH)
        assert raw.count("Container-provable:") >= 3
        assert raw.count("Host-assured, named:") >= 3

    def test_superseding_tests_named(self):
        text = _spec()
        for needle in (
            "test_seven_schemas",
            "test_manage_skills_is_third_non_sandbox",
            "test_daily_includes_manage_skills",
            "test_bulbe_has_no_handlers",
        ):
            assert needle in text, needle
        assert "deselect-plus-reassert" in text
        assert "originals never edited" in text

    def test_holding_pins_named(self):
        text = _spec()
        for needle in (
            "test_sandbox_four_match_allowlist",
            "test_bulbe_exposes_ sandbox_only",
            "test_daily_exposes_all_six",
            "test_sandbox_four_unchanged",
        ):
            assert needle in text, needle

    def test_lot2_zero_targeted(self):
        text = _spec()
        assert "Supersession forecast: ZERO targeted" in text

    def test_lot3_all_new(self):
        text = _spec()
        assert "Supersession forecast: zero (all-new surface)" in text


class TestDecisionsRecord:
    def test_d1_to_d4_present(self):
        text = _spec()
        for needle in (
            "D1 -- the borrow-vs-reject cut",
            "D2 -- the lot cut and order",
            "D3 -- harness scoring and register",
            "D4 -- sequencing",
        ):
            assert needle in text, needle

    def test_atl01_not_absorbed_atl02_not_owned(self):
        text = _spec()
        assert "ATL-01 is NOT absorbed" in text
        assert "ATL-02 is owned by the Sandbox Workspace cycle" in text

    def test_d4_records_recommendation_and_close_arbitration(self):
        text = _spec()
        assert "governor Bloc 0 (measurement) opens at S223" in text
        assert "The final order is the S222 close arbitration" in text


# ---------------------------------------------------------------------------
# Family 2 -- the roadmap roll
# ---------------------------------------------------------------------------


class TestRoadmapAGT:
    def test_cycle_entry_rolled_to_spec_written(self):
        text = _roadmap()
        assert "Agent Performance cycle (AGT) -- spec WRITTEN at S222" in text
        assert "AGT_SPEC.md is the design contract" in text

    def test_entry_carries_the_pin_and_route_a(self):
        text = _roadmap()
        assert "sst/opencode 4519a1da, v1.16.2, MIT" in text
        assert "explicitly-arbitrated fallback spike" in text

    def test_entry_ordering_governor_then_agt_then_cas7(self):
        raw = _read(ROADMAP_PATH)
        governor = raw.find("3. Resource Governor cycle")
        agt = raw.find("4. Agent Performance cycle (AGT)")
        cas7 = raw.find("5. cas 7 -- remote inference delegation")
        assert 0 < governor < agt < cas7

    def test_mobile_renumbered_to_six(self):
        raw = _read(ROADMAP_PATH)
        assert "6. Mobile app cycle (Android first)" in raw
        assert "5. Mobile app cycle (Android first)" not in raw

    def test_sequencing_line_references_the_spec(self):
        text = _roadmap()
        assert "spec at S222, AGT_SPEC.md" in text

    def test_ownership_clauses_carried(self):
        text = _roadmap()
        assert "ATL-02 stays Sandbox-Workspace-owned" in text
        assert "ATL-01 stays on the standing list" in text


# ---------------------------------------------------------------------------
# Family 3 -- the seams the spec builds on (green on pristine by design)
# ---------------------------------------------------------------------------


class TestSeamAllowlists:
    def test_frozensets_exact(self):
        al = _load("allowlists", SEAM_SOURCES["allowlists"])
        assert al.SANDBOX_TOOL_NAMES == frozenset(
            {"bash", "view", "create_file", "str_replace"}
        )
        assert al.NETWORK_TOOLS == frozenset({"web_search"})
        assert al.STATE_MUTATION_TOOLS == frozenset(
            {"manage_memory", "manage_skills"}
        )

    def test_bulbe_is_a_derivation(self):
        al = _load("allowlists", SEAM_SOURCES["allowlists"])
        assert al.BULBE_ALLOWLIST == frozenset(
            al.DAILY_ALLOWLIST - al.NETWORK_TOOLS - al.STATE_MUTATION_TOOLS
        )
        assert al.BULBE_ALLOWLIST < al.DAILY_ALLOWLIST

    def test_unknown_mode_fails_secure(self):
        src = _read(SEAM_SOURCES["allowlists"])
        assert "fails secure to Bulbe" in src or "fail secure" in src


class TestSeamToolsSchemas:
    def test_all_schemas_is_seven_today(self):
        _load("allowlists", SEAM_SOURCES["allowlists"])
        t = _load("tools", SEAM_SOURCES["tools"])
        assert len(t.ALL_SCHEMAS) == 7

    def test_sandboxed_schemas_match_allowlist(self):
        al = _load("allowlists", SEAM_SOURCES["allowlists"])
        t = _load("tools", SEAM_SOURCES["tools"])
        sandboxed = {s.name for s in t.ALL_SCHEMAS if s.sandboxed}
        assert sandboxed == set(al.SANDBOX_TOOL_NAMES)

    def test_handler_names_are_the_three(self):
        t = _load("tools", SEAM_SOURCES["tools"])
        assert t.HANDLER_TOOL_NAMES == frozenset(
            {"web_search", "manage_memory", "manage_skills"}
        )


class TestSeamDispatch:
    def test_sandbox_dispatch_keys_are_the_four(self):
        src = _read(SEAM_SOURCES["dispatch"])
        block = src.split("_SANDBOX_DISPATCH", 1)[1][:800]
        for key in ('"bash"', '"view"', '"create_file"', '"str_replace"'):
            assert key in block, key

    def test_sandbox_ready_gate_present(self):
        src = _read(SEAM_SOURCES["dispatch"])
        assert "def sandbox_ready" in src


class TestSeamLoop:
    def test_round_caps_pinned(self):
        src = _read(SEAM_SOURCES["loop"])
        assert "MAX_AGENT_ROUNDS = 20" in src
        assert "_VERIFIER_MAX_ROUNDS = 2" in src

    def test_verifier_present(self):
        src = _read(SEAM_SOURCES["loop"])
        assert "def _run_verifier" in src

    def test_observations_ride_untrusted_wrapping(self):
        src = _read(SEAM_SOURCES["loop"])
        assert "tool_output_message" in src

    def test_agent_event_surface(self):
        src = _read(SEAM_SOURCES["loop"])
        assert "class AgentEvent" in src


class TestSeamUntrusted:
    def test_wrappers_exported(self):
        src = _read(SEAM_SOURCES["untrusted"])
        for needle in (
            "def wrap(",
            "def tool_output_message(",
            "def skill_message(",
        ):
            assert needle in src, needle


class TestSeamToolParsing:
    def test_three_formats_supported(self):
        src = _read(SEAM_SOURCES["tool_parsing"])
        assert 'SUPPORTED_FORMATS = ("fenced", "bracketed", "xml")' in src


class TestSeamSandbox:
    def test_output_cap_default(self):
        src = _read(SEAM_SOURCES["sandbox_manager"])
        assert "max_output_bytes: int = 65536" in src

    def test_truncation_flags_on_command_result(self):
        src = _read(SEAM_SOURCES["sandbox_manager"])
        assert "truncated_stdout" in src
        assert "truncated_stderr" in src

    def test_ro_binds_expose_host_binaries(self):
        src = _read(SEAM_SOURCES["sandbox_manager"])
        assert "/usr, /bin, /lib" in src

    def test_path_confinement_present(self):
        src = _read(SEAM_SOURCES["sandbox_manager"])
        assert "def validate_sandbox_path" in src

    def test_session_tool_methods_present(self):
        src = _read(SEAM_SOURCES["sandbox_tools"])
        assert "class SandboxToolSession" in src
        for needle in (
            "def bash(",
            "def view(",
            "def create_file(",
            "def str_replace(",
        ):
            assert needle in src, needle


class TestSeamBenchmarkIdiom:
    def test_runner_lifecycle_seams(self):
        src = _read(SEAM_SOURCES["benchmark_runner"])
        assert "def is_busy" in src
        assert "class ResultsStore" in src
        assert "def start_run" in src

    def test_route_409_one_run_at_a_time(self):
        src = _read(SEAM_SOURCES["routes_benchmark_v2"])
        assert "status_code=409" in src


class TestSeamGovernorSpecClauses:
    def test_admitted_ctx_feeds_agt_caps(self):
        text = _governor()
        assert (
            "The same admitted value is the one the AGT loop-hardening "
            "lot's truncation caps consume" in text
        )

    def test_harness_admission_semantics(self):
        text = _governor()
        assert "no silent downsize" in text
        assert "evict-between-runs" in text

    def test_harness_named_as_funnel(self):
        text = _governor()
        assert "the AGT harness at S222+" in text


class TestASTValid:
    def test_seam_python_sources_parse(self):
        for name, path in SEAM_SOURCES.items():
            src = _read(path)
            assert src != "", name
            ast.parse(src, filename=str(path))

    def test_this_suite_parses(self):
        ast.parse(_read(Path(__file__)), filename=__file__)
