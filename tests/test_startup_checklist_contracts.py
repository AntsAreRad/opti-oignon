#!/usr/bin/env python3
"""Contracts for the startup security checklist aggregator.

The aggregator combines the individual runtime guards into one cached,
API-served report and decides whether startup must be refused. These
contracts pin the aggregation semantics that the boot guard and the
endpoint both rely on:

  * Contract 1 -- blocked aggregation: one critical failure among six
    checks sets blocked, names the check in block_reason, and clears
    all_passed; six passing checks set all_passed and never block.
  * Contract 2 -- cache: a second call returns the cached object;
    ``force=True`` re-runs; ``clear_cache`` empties the cache.
  * Contract 3 -- Ollama bind severity mapping: a blocked bind result
    maps to a critical failure, an exposed-but-unblocked result to a
    warning, a clean or undeterminable result passes.
  * Contract 4 -- LUKS mapping: encrypted passes with no deduction; an
    unencrypted result is an advisory warning carrying the detector's
    tips; a crashing detector degrades to a warning, never an exception.
  * Contract 5 -- serialization: the aggregated report serializes to
    JSON with coherent pass/fail counters.

Local-only (the public distribution ships no tests). Runs under pytest
or the __main__ runner. The checklist module is loaded on the shared
isolation window; the guard modules it imports lazily are seeded as
stand-ins so every clause is deterministic.
"""

import json
import sys
import traceback
import types
from types import SimpleNamespace

from _isolation import isolate, source  # noqa: E402

_CHECK_NAMES = (
    "_check_code_signing_scripts",
    "_check_ollama_bind",
    "_check_luks",
    "_check_security_mode",
    "_check_encrypted_swap",
    "_check_governor_ollama_limits",
    "_check_pqc_primitive",
    "_check_backend_provenance_coverage",
)


def _load_checklist_module(seed=None):
    """Load startup_checks.py on the shared isolation window.

    ``seed`` maps dotted module names to stand-in modules so the
    checklist's lazy imports resolve to them; nothing else in the
    package resolves.
    """
    loaded, restore = isolate(
        targets={"opti_oignon.startup_checks": source("startup_checks.py")},
        seeded=seed or {},
    )
    return loaded["opti_oignon.startup_checks"], restore


def _stub_module(name, **attrs):
    stub = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(stub, key, value)
    return stub


def _install_stub_checks(mod, failing=None, real=()):
    saved = {n: getattr(mod, n) for n in _CHECK_NAMES}
    for n in _CHECK_NAMES:
        if n in real:
            # Leave the real implementation in place (exercise it inside
            # run_startup_checks while every other check is stubbed).
            continue
        if n == failing:
            setattr(
                mod, n,
                lambda _n=n: mod.CheckItem(
                    name=_n, passed=False, severity="critical",
                    detail="stub critical failure", score_impact=-15,
                ),
            )
        else:
            setattr(
                mod, n,
                lambda _n=n: mod.CheckItem(
                    name=_n, passed=True, severity="info", detail="stub pass",
                ),
            )
    mod.clear_cache()

    def restore():
        for n, fn in saved.items():
            setattr(mod, n, fn)
        mod.clear_cache()

    return restore


# ---------------------------------------------------------------------------
# Contract 1 -- blocked aggregation and the all-pass verdict
# ---------------------------------------------------------------------------
def test_c1_critical_failure_blocks_and_names_the_check():
    mod, restore = _load_checklist_module()
    try:
        undo = _install_stub_checks(mod, failing="_check_ollama_bind")
        try:
            result = mod.run_startup_checks(force=True)
            assert result.blocked, "one critical failure must set blocked"
            assert "_check_ollama_bind" in result.block_reason, (
                f"block_reason must name the failing check, got: "
                f"{result.block_reason!r}"
            )
            assert not result.all_passed
        finally:
            undo()

        undo = _install_stub_checks(mod, failing=None)
        try:
            result = mod.run_startup_checks(force=True)
            assert result.all_passed, "six passing checks must set all_passed"
            assert not result.blocked, "an all-pass run must never block"
            assert result.block_reason == ""
        finally:
            undo()
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 2 -- cache semantics
# ---------------------------------------------------------------------------
def test_c2_cache_returns_same_object_until_forced_or_cleared():
    mod, restore = _load_checklist_module()
    try:
        undo = _install_stub_checks(mod, failing=None)
        try:
            first = mod.run_startup_checks()
            second = mod.run_startup_checks()
            assert second is first, "a second call must return the cached object"
            assert mod.get_cached_result() is first

            forced = mod.run_startup_checks(force=True)
            assert forced is not first, "force=True must re-run the checks"

            mod.clear_cache()
            assert mod.get_cached_result() is None, (
                "clear_cache must empty the cache"
            )
        finally:
            undo()
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 3 -- Ollama bind severity mapping (both outcomes and the unknowns)
# ---------------------------------------------------------------------------
def test_c3_ollama_bind_severity_mapping():
    outcomes = {}

    def _run(bind_result):
        stub = _stub_module(
            "opti_oignon.network_bind_guard",
            check_ollama_bind=lambda: bind_result,
        )
        mod, restore = _load_checklist_module(
            seed={"opti_oignon.network_bind_guard": stub},
        )
        try:
            return mod._check_ollama_bind()
        finally:
            restore()

    outcomes["blocked"] = _run(SimpleNamespace(
        checked=True, exposed=True, blocked=True,
        detail="exposed on a wildcard address",
    ))
    outcomes["exposed"] = _run(SimpleNamespace(
        checked=True, exposed=True, blocked=False,
        detail="exposed on a wildcard address",
    ))
    outcomes["clean"] = _run(SimpleNamespace(
        checked=True, exposed=False, blocked=False, detail="loopback only",
    ))
    outcomes["unknown"] = _run(SimpleNamespace(
        checked=False, exposed=False, blocked=False, detail="undeterminable",
    ))

    item = outcomes["blocked"]
    assert not item.passed and item.severity == "critical", (
        f"a blocked bind must map to a critical failure, got {item.severity}"
    )
    assert item.score_impact < 0

    item = outcomes["exposed"]
    assert not item.passed and item.severity == "warning", (
        f"exposed-but-unblocked must map to a warning, got {item.severity}"
    )

    assert outcomes["clean"].passed and outcomes["clean"].severity == "info"
    assert outcomes["unknown"].passed, (
        "an undeterminable bind must not fail the checklist"
    )


# ---------------------------------------------------------------------------
# Contract 4 -- LUKS mapping is advisory and fail-open
# ---------------------------------------------------------------------------
def test_c4_luks_mapping_is_advisory_and_fail_open():
    def _run(luks_result=None, raising=False):
        if raising:
            def _detector():
                raise OSError("detector crashed")
        else:
            def _detector():
                return luks_result
        stub = _stub_module(
            "opti_oignon.luks_detector", check_luks_encryption=_detector,
        )
        mod, restore = _load_checklist_module(
            seed={"opti_oignon.luks_detector": stub},
        )
        try:
            return mod._check_luks()
        finally:
            restore()

    item = _run(SimpleNamespace(
        checked=True, encrypted=True, detail="root is encrypted", tips=[],
    ))
    assert item.passed and item.severity == "info" and item.score_impact == 0

    item = _run(SimpleNamespace(
        checked=True, encrypted=False, detail="root is not encrypted",
        tips=["enable full-disk encryption"],
    ))
    assert not item.passed and item.severity == "warning", (
        "an unencrypted disk is an advisory warning, never critical"
    )
    assert item.tips, "the detector's tips must be carried to the report"

    item = _run(raising=True)
    assert item.severity == "warning" and not item.passed, (
        "a crashing detector must degrade to a warning, not an exception"
    )


# ---------------------------------------------------------------------------
# Contract 5 -- serialization and coherent counters
# ---------------------------------------------------------------------------
def test_c5_report_serializes_with_coherent_counters():
    mod, restore = _load_checklist_module()
    try:
        undo = _install_stub_checks(mod, failing="_check_luks")
        try:
            result = mod.run_startup_checks(force=True)
            payload = result.to_dict()
            json.dumps(payload)
            assert payload["check_count"] == len(_CHECK_NAMES)
            assert payload["failed_count"] == 1
            assert payload["passed_count"] == len(_CHECK_NAMES) - 1
            assert payload["passed_count"] + payload["failed_count"] == (
                payload["check_count"]
            )
        finally:
            undo()
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 6-8 -- active-backend provenance coverage advisory
# ---------------------------------------------------------------------------
def _coverage_seed(backend, mode):
    """Seed the two modules the coverage check imports, with controlled
    return values, so every branch is deterministic under isolation."""
    return {
        "opti_oignon.security_mode": _stub_module(
            "opti_oignon.security_mode",
            _default_backend=lambda: backend,
            get_current_mode=lambda: mode,
        ),
        "opti_oignon.model_provenance": _stub_module(
            "opti_oignon.model_provenance",
            backend_enforces_provenance=lambda b: b == "llama_cpp",
        ),
    }


def test_c6_gated_backend_reports_coverage_as_info():
    mod, restore = _load_checklist_module(
        seed=_coverage_seed("llama_cpp", "daily")
    )
    try:
        item = mod._check_backend_provenance_coverage()
        assert item.passed is True
        assert item.severity == "info"
        assert "llama_cpp" in item.detail
    finally:
        restore()


def test_c7_ungated_backend_warns_and_never_blocks():
    mod, restore = _load_checklist_module(
        seed=_coverage_seed("ollama", "daily")
    )
    try:
        item = mod._check_backend_provenance_coverage()
        assert item.passed is False
        assert item.severity == "warning"     # advisory, NOT critical
        assert item.severity != "critical"
        assert item.score_impact < 0
        assert item.tips
        assert "ollama" in item.detail
        # End-to-end: an advisory warning must never set the blocked flag.
        undo = _install_stub_checks(
            mod, real=("_check_backend_provenance_coverage",)
        )
        try:
            result = mod.run_startup_checks(force=True)
            assert result.blocked is False
            assert "backend_provenance_coverage" in [
                c.name for c in result.checks
            ]
        finally:
            undo()
    finally:
        restore()


def test_c8_bulbe_sharpens_message_but_stays_advisory():
    mod, restore = _load_checklist_module(
        seed=_coverage_seed("ollama", "bulbe")
    )
    try:
        item = mod._check_backend_provenance_coverage()
        assert item.severity == "warning"     # never critical, even in bulbe
        assert "bulbe" in item.detail.lower()
        undo = _install_stub_checks(
            mod, real=("_check_backend_provenance_coverage",)
        )
        try:
            assert mod.run_startup_checks(force=True).blocked is False
        finally:
            undo()
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner (pytest picks up the test_ functions; direct execution works too)
# ---------------------------------------------------------------------------
def _main(argv: list[str]) -> int:
    names = sorted(n for n in globals() if n.startswith("test_"))
    selected = [
        n for n in names if not argv or any(fragment in n for fragment in argv)
    ]
    failures = 0
    for name in selected:
        try:
            globals()[name]()
        except Exception as exc:
            failures += 1
            print(f"FAIL {name}: {exc.__class__.__name__}: {exc}")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
    print(f"{len(selected) - failures}/{len(selected)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
