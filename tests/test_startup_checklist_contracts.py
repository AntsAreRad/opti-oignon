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
or the __main__ runner. The checklist module is loaded in isolation
under a stub package; the guard modules it imports lazily are seeded
as stubs in sys.modules so every clause is deterministic.
"""

import importlib.util
import json
import sys
import traceback
import types
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"

_CHECK_NAMES = (
    "_check_code_signing_scripts",
    "_check_ollama_bind",
    "_check_luks",
    "_check_security_mode",
    "_check_encrypted_swap",
    "_check_governor_ollama_limits",
    "_check_pqc_primitive",
)


def _load_checklist_module(seed=None):
    """Load startup_checks.py in isolation under a stub package.

    ``seed`` maps dotted module names to stub modules pre-registered in
    sys.modules so the checklist's lazy imports resolve to them.
    """
    seed = seed or {}
    keys = ("opti_oignon", "opti_oignon.startup_checks", *seed.keys())
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg
    for name, stub in seed.items():
        sys.modules[name] = stub
        setattr(pkg, name.rsplit(".", 1)[-1], stub)

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.startup_checks", _OO / "startup_checks.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.startup_checks"] = mod
    spec.loader.exec_module(mod)
    pkg.startup_checks = mod

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return mod, restore


def _stub_module(name, **attrs):
    stub = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(stub, key, value)
    return stub


def _install_stub_checks(mod, failing=None):
    saved = {n: getattr(mod, n) for n in _CHECK_NAMES}
    for n in _CHECK_NAMES:
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
