#!/usr/bin/env python3
"""Contracts for boot-time enforcement of the startup security checklist.

The checklist module computes a blocked verdict (a critical guard
failure, e.g. Ollama exposed on a wildcard address in Bulbe mode), but
nothing consumed that verdict at boot: the checks only ran lazily when
the API endpoint was first queried, so a launch that should have been
refused proceeded and armed every service. These contracts pin the
enforcement seam:

  * Contract 1 -- a blocked verdict raises: ``enforce_boot_checks``
    raises ``StartupBlockedError`` carrying the block reason.
  * Contract 2 -- a passing verdict returns the result and never raises.
  * Contract 3 -- check machinery failure is fail-open: an exception
    inside the checklist run is swallowed and None is returned, so an
    unavailable check can never break the boot on its own.
  * Contract 4 -- the application lifespan refuses startup on a blocked
    verdict: entering the real ASGI lifespan propagates the error.
  * Contract 5 -- a refused boot arms nothing: on a blocked verdict the
    plugin loading and sync arming steps are never reached.
  * Contract 6 -- a passing boot populates the checklist cache, so the
    API endpoint serves the boot-time result without re-running.

Local-only (the public distribution ships no tests). Runs under pytest
or the __main__ runner. Contracts 1-3 load the checklist module in
isolation under a stub package; contracts 4-6 drive the real
application lifespan with the individual checks replaced by cheap
deterministic stubs and the heavy startup actions replaced by spies.
"""

import asyncio
import importlib.util
import sys
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"

_CHECK_NAMES = (
    "_check_code_signing_scripts",
    "_check_ollama_bind",
    "_check_luks",
    "_check_security_mode",
    "_check_encrypted_swap",
    "_check_governor_ollama_limits",
)


def _load_checklist_module():
    """Load startup_checks.py in isolation under a stub package."""
    keys = ("opti_oignon", "opti_oignon.startup_checks")
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

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


def _purge_stub_package():
    """Drop a leftover stub opti_oignon package so the real one imports."""
    pkg = sys.modules.get("opti_oignon")
    if pkg is not None and not getattr(pkg, "__path__", None):
        for key in [k for k in sys.modules if k.split(".")[0] == "opti_oignon"]:
            sys.modules.pop(key, None)


def _passing_item(mod, name):
    return mod.CheckItem(
        name=name, passed=True, severity="info", detail="stub pass",
    )


def _critical_item(mod, name):
    return mod.CheckItem(
        name=name, passed=False, severity="critical",
        detail="stub critical failure", score_impact=-15,
    )


def _install_stub_checks(mod, failing=None):
    """Replace the six individual checks with deterministic stubs.

    Returns a restore callable. ``failing`` names one check to fail
    critically; every other check passes.
    """
    saved = {n: getattr(mod, n) for n in _CHECK_NAMES}
    for n in _CHECK_NAMES:
        if n == failing:
            setattr(mod, n, lambda _n=n: _critical_item(mod, _n))
        else:
            setattr(mod, n, lambda _n=n: _passing_item(mod, _n))
    mod.clear_cache()

    def restore():
        for n, fn in saved.items():
            setattr(mod, n, fn)
        mod.clear_cache()

    return restore


# ---------------------------------------------------------------------------
# Contract 1 -- a blocked verdict raises with the block reason
# ---------------------------------------------------------------------------
def test_c1_blocked_verdict_raises_with_reason():
    mod, restore = _load_checklist_module()
    try:
        undo = _install_stub_checks(mod, failing="_check_ollama_bind")
        try:
            try:
                mod.enforce_boot_checks()
            except mod.StartupBlockedError as exc:
                assert "_check_ollama_bind" in str(exc) or "ollama" in str(exc).lower() or "Critical" in str(exc), (
                    f"the exception must carry the block reason, got: {exc}"
                )
            else:
                raise AssertionError(
                    "enforce_boot_checks returned instead of raising on a "
                    "blocked verdict"
                )
        finally:
            undo()
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 2 -- a passing verdict returns the result, no raise
# ---------------------------------------------------------------------------
def test_c2_passing_verdict_returns_result():
    mod, restore = _load_checklist_module()
    try:
        undo = _install_stub_checks(mod, failing=None)
        try:
            result = mod.enforce_boot_checks()
            assert result is not None, "a passing run must return the result"
            assert result.all_passed, "stubbed run should pass every check"
            assert not result.blocked, "a passing run must not be blocked"
        finally:
            undo()
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 3 -- machinery failure is fail-open (None, no raise)
# ---------------------------------------------------------------------------
def test_c3_machinery_failure_is_fail_open():
    mod, restore = _load_checklist_module()
    try:
        saved = mod.run_startup_checks

        def _boom(*, force=False):
            raise RuntimeError("machinery down")

        mod.run_startup_checks = _boom
        try:
            result = mod.enforce_boot_checks()
            assert result is None, (
                "machinery failure must return None instead of raising"
            )
        finally:
            mod.run_startup_checks = saved
    finally:
        restore()


# ---------------------------------------------------------------------------
# Real-application harness for contracts 4-6
# ---------------------------------------------------------------------------
def _real_app_harness():
    """Import the real app and neutralize the heavy startup actions.

    Returns (app module namespace dict, restore callable). The plugin
    loader and sync arming are replaced by recording spies; the memory
    migration is replaced by a no-op. The six checklist checks are left
    to the caller.
    """
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    _purge_stub_package()
    import opti_oignon.api.app as app_mod
    import opti_oignon.memory.migration as mig
    import opti_oignon.plugin_loader as pl
    import opti_oignon.startup_checks as sc
    import opti_oignon.veilid.sync_service as svc

    calls = {"plugins": 0, "arm": 0}
    saved = {
        "arm": svc.arm_if_enabled,
        "reset": svc.reset_sync_service,
        "mig": mig.run_boot_migration,
    }

    def _arm_spy():
        calls["arm"] += 1

    svc.arm_if_enabled = _arm_spy
    svc.reset_sync_service = lambda: None
    mig.run_boot_migration = lambda: None

    saved_pl = None
    if pl.plugin_loader is not None:
        saved_pl = (
            pl.plugin_loader.load_all_enabled,
            pl.plugin_loader.shutdown_all,
        )

        def _plugins_spy():
            calls["plugins"] += 1
            return []

        pl.plugin_loader.load_all_enabled = _plugins_spy
        pl.plugin_loader.shutdown_all = lambda: None

    def restore():
        svc.arm_if_enabled = saved["arm"]
        svc.reset_sync_service = saved["reset"]
        mig.run_boot_migration = saved["mig"]
        if saved_pl is not None:
            pl.plugin_loader.load_all_enabled = saved_pl[0]
            pl.plugin_loader.shutdown_all = saved_pl[1]
        sc.clear_cache()

    return {"app": app_mod.app, "sc": sc, "calls": calls}, restore


def _install_real_stub_checks(sc, failing=None):
    saved = {n: getattr(sc, n) for n in _CHECK_NAMES}
    for n in _CHECK_NAMES:
        if n == failing:
            setattr(sc, n, lambda _n=n: _critical_item(sc, _n))
        else:
            setattr(sc, n, lambda _n=n: _passing_item(sc, _n))
    sc.clear_cache()

    def restore():
        for n, fn in saved.items():
            setattr(sc, n, fn)
        sc.clear_cache()

    return restore


async def _enter_and_exit_lifespan(app):
    async with app.router.lifespan_context(app):
        return True


# ---------------------------------------------------------------------------
# Contract 4 -- the real lifespan refuses startup on a blocked verdict
# ---------------------------------------------------------------------------
def test_c4_lifespan_refuses_startup_when_blocked():
    harness, restore = _real_app_harness()
    try:
        undo = _install_real_stub_checks(
            harness["sc"], failing="_check_ollama_bind",
        )
        try:
            try:
                asyncio.run(_enter_and_exit_lifespan(harness["app"]))
            except harness["sc"].StartupBlockedError:
                pass
            else:
                raise AssertionError(
                    "the lifespan completed startup despite a blocked "
                    "checklist verdict"
                )
        finally:
            undo()
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 5 -- a refused boot arms nothing
# ---------------------------------------------------------------------------
def test_c5_refused_boot_arms_no_services():
    harness, restore = _real_app_harness()
    try:
        undo = _install_real_stub_checks(
            harness["sc"], failing="_check_ollama_bind",
        )
        try:
            try:
                asyncio.run(_enter_and_exit_lifespan(harness["app"]))
            except Exception:
                pass
            assert harness["calls"]["plugins"] == 0, (
                "plugin loading ran despite a refused boot"
            )
            assert harness["calls"]["arm"] == 0, (
                "sync arming ran despite a refused boot"
            )
        finally:
            undo()
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 6 -- a passing boot populates the checklist cache
# ---------------------------------------------------------------------------
def test_c6_passing_boot_populates_cache():
    harness, restore = _real_app_harness()
    try:
        undo = _install_real_stub_checks(harness["sc"], failing=None)
        try:
            ok = asyncio.run(_enter_and_exit_lifespan(harness["app"]))
            assert ok, "the lifespan should enter and exit cleanly"
            cached = harness["sc"].get_cached_result()
            assert cached is not None, (
                "boot must populate the checklist cache for the API endpoint"
            )
            assert cached.all_passed and not cached.blocked, (
                "the cached boot result should reflect the passing run"
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
