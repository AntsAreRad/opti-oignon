#!/usr/bin/env python3
"""Contracts: Bulbe never boots past a startup checklist that could not run.

The startup checklist refuses the boot when Ollama is bound to a wildcard address in Bulbe mode
(``_check_ollama_bind``), and nothing after the boot refuses over that bind. Bulbe booted without the check having run
in three ways: the check reported itself passed when it raised; ``enforce_boot_checks`` returned None when the
checklist itself raised; and the application lifespan skipped the checklist when its module did not import. Bulbe
then served with Ollama perhaps exposed to the network, which is what the check exists to refuse. The mode is read
fail-secure, as the bind guard reads it: a mode that cannot be read, or is not exactly a recognized one, is Bulbe.

  * UB1 -- the bind check that fails before it can look (its guard raising, or not importing) is not checked, never
    passed: in Bulbe, and in a mode that cannot be read or is not recognized, a critical failure that blocks the
    checklist and names the error; in Daily a warning that blocks nothing. The report says it was not checked.
  * UB2 -- a checklist that raises refuses the boot in Bulbe, and in a mode that cannot be read or is not
    recognized: ``StartupBlockedError`` naming the error. In Daily it returns None and the boot goes on, as c3 of
    tests/test_startup_boot_guard_contracts.py held for every mode.
  * UB3 -- the real lifespan, its checklist module not importing: in Bulbe, and in a mode that cannot be read or is
    not recognized, the boot is refused, naming the error, before plugins or sync are armed; in Daily it boots.
  * UB4 -- a tree none of whose modules import (b4 of tests/test_pqc_refusal_is_fired_contracts.py): the
    post-quantum check still warns and never refuses; the checklist blocks nothing in Daily, and in Bulbe, or a mode
    that cannot be read, it is blocked by the bind check alone, never by the post-quantum one.
  * UB5 -- each check runs on its own: one whose exception escapes (a swap name that is not UTF-8, an exception whose
    message cannot be made) is reported not checked, named by its error, and the checks after it still run; the
    boot goes on in Bulbe past every check but the bind check, which blocks it there and is a warning in Daily. The
    checks this contract lists are the ones the runner calls.
  * UB6 -- the boot runs the checklist afresh: a result an earlier run left in Daily is never the verdict of a boot
    in Bulbe.
  * UB7 -- a check that returns no CheckItem is reported not checked, as one that raises: the bind check so is
    critical in Bulbe and a warning in Daily, any other a warning, and the checks after it still run.
  * UB8 -- the bind guard reads ``ss`` whatever bytes a process name holds: an exposed Ollama is seen exposed (and
    blocks Bulbe as exposed, never as not checked), one on the loopback is seen there.

Local-only (the public distribution ships no tests). UB1, UB2 and UB4-UB7 load the checklist on the shared isolation
window, the mode and the bind guard seeded per case; UB8 loads the real bind guard there too, ``ss`` played by a
``printf`` child; UB3 drives the real application lifespan with the
startup boot guard suite's harness (heavy startup actions replaced by spies, the run journal pointed at an empty
directory).
"""

import ast
import asyncio
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_startup_boot_guard_contracts as bg_t  # noqa: E402
from _isolation import isolate, source  # noqa: E402

# Seconds per junit case, the parametrized ids included: the ladder reads this literal alone.
BUDGET_S = {
    "test_ub1_a_bind_check_that_cannot_look_is_not_checked_and_blocks_bulbe[bulbe-guard_raises]": 2.0,
    "test_ub1_a_bind_check_that_cannot_look_is_not_checked_and_blocks_bulbe[bulbe-guard_absent]": 2.0,
    "test_ub1_a_bind_check_that_cannot_look_is_not_checked_and_blocks_bulbe[daily-guard_raises]": 2.0,
    "test_ub1_a_bind_check_that_cannot_look_is_not_checked_and_blocks_bulbe[daily-guard_absent]": 2.0,
    "test_ub1_a_bind_check_that_cannot_look_is_not_checked_and_blocks_bulbe[unreadable-guard_raises]": 2.0,
    "test_ub1_a_bind_check_that_cannot_look_is_not_checked_and_blocks_bulbe[unreadable-guard_absent]": 2.0,
    "test_ub1_a_bind_check_that_cannot_look_is_not_checked_and_blocks_bulbe[raising-guard_raises]": 2.0,
    "test_ub1_a_bind_check_that_cannot_look_is_not_checked_and_blocks_bulbe[raising-guard_absent]": 2.0,
    "test_ub1_a_bind_check_that_cannot_look_is_not_checked_and_blocks_bulbe[unrecognized-guard_raises]": 2.0,
    "test_ub1_a_bind_check_that_cannot_look_is_not_checked_and_blocks_bulbe[unrecognized-guard_absent]": 2.0,
    "test_ub2_a_checklist_that_raises_refuses_the_boot_in_bulbe_and_lets_daily_boot[bulbe]": 2.0,
    "test_ub2_a_checklist_that_raises_refuses_the_boot_in_bulbe_and_lets_daily_boot[daily]": 2.0,
    "test_ub2_a_checklist_that_raises_refuses_the_boot_in_bulbe_and_lets_daily_boot[unreadable]": 2.0,
    "test_ub2_a_checklist_that_raises_refuses_the_boot_in_bulbe_and_lets_daily_boot[raising]": 2.0,
    "test_ub2_a_checklist_that_raises_refuses_the_boot_in_bulbe_and_lets_daily_boot[unrecognized]": 2.0,
    "test_ub3_the_lifespan_without_its_checklist_refuses_bulbe_and_boots_daily[bulbe]": 10.0,
    "test_ub3_the_lifespan_without_its_checklist_refuses_bulbe_and_boots_daily[daily]": 10.0,
    "test_ub3_the_lifespan_without_its_checklist_refuses_bulbe_and_boots_daily[raising]": 10.0,
    "test_ub3_the_lifespan_without_its_checklist_refuses_bulbe_and_boots_daily[unrecognized]": 10.0,
    "test_ub4_a_tree_that_does_not_import_is_blocked_by_the_bind_check_alone[bulbe]": 2.0,
    "test_ub4_a_tree_that_does_not_import_is_blocked_by_the_bind_check_alone[daily]": 2.0,
    "test_ub4_a_tree_that_does_not_import_is_blocked_by_the_bind_check_alone[unreadable]": 2.0,
    "test_ub5_a_check_that_raises_is_reported_not_checked_and_the_others_still_run[code_signing_scripts-bulbe]": 2.0,
    "test_ub5_a_check_that_raises_is_reported_not_checked_and_the_others_still_run[ollama_bind-bulbe]": 2.0,
    "test_ub5_a_check_that_raises_is_reported_not_checked_and_the_others_still_run[luks-bulbe]": 2.0,
    "test_ub5_a_check_that_raises_is_reported_not_checked_and_the_others_still_run[security_mode-bulbe]": 2.0,
    "test_ub5_a_check_that_raises_is_reported_not_checked_and_the_others_still_run[encrypted_swap-bulbe]": 2.0,
    "test_ub5_a_check_that_raises_is_reported_not_checked_and_the_others_still_run[governor_ollama_limits-bulbe]": 2.0,
    "test_ub5_a_check_that_raises_is_reported_not_checked_and_the_others_still_run[pqc_primitive-bulbe]": 2.0,
    "test_ub5_a_check_that_raises_is_reported_not_checked_and_the_others_still_run"
    "[backend_provenance_coverage-bulbe]": 2.0,
    "test_ub5_a_check_that_raises_is_reported_not_checked_and_the_others_still_run[process_secrets-bulbe]": 2.0,
    "test_ub5_a_check_that_raises_is_reported_not_checked_and_the_others_still_run[ollama_bind-daily]": 2.0,
    "test_ub6_the_boot_runs_the_checklist_afresh_never_an_earlier_runs_result": 2.0,
    "test_ub7_a_check_that_returns_no_item_is_reported_not_checked[bulbe]": 2.0,
    "test_ub7_a_check_that_returns_no_item_is_reported_not_checked[daily]": 2.0,
    "test_ub8_the_bind_guard_reads_ss_whatever_bytes_a_process_name_holds[exposed]": 3.0,
    "test_ub8_the_bind_guard_reads_ss_whatever_bytes_a_process_name_holds[loopback]": 3.0,
}

_MODES = ("bulbe", "daily", "unreadable", "raising", "unrecognized")
# The modes read as Bulbe: Bulbe itself, a mode that cannot be read (its module absent, or raising), and one that is
# not exactly a recognized value.
_READ_AS_BULBE = ("bulbe", "unreadable", "raising", "unrecognized")
_SAID = {"bulbe": "bulbe", "daily": "daily", "unrecognized": "daily "}


def _mode_stand_in(mode):
    """A security_mode module whose ``get_current_mode`` says ``mode``'s value, or raises for "raising"."""
    module = types.ModuleType("opti_oignon.security_mode")
    if mode == "raising":
        def get_current_mode():
            raise RuntimeError("the mode lock does not read")
    else:
        said = _SAID[mode]

        def get_current_mode():
            return said
    module.get_current_mode = get_current_mode
    return module


def _raising_guard():
    """A bind guard whose check raises before it can look."""
    module = types.ModuleType("opti_oignon.network_bind_guard")

    def check_ollama_bind(*args, **kwargs):
        raise OSError("the sockets cannot be listed")

    module.check_ollama_bind = check_ollama_bind
    return module


def _checklist(mode, guard=None):
    """startup_checks.py on the isolation window, the mode seeded unless it is "unreadable", the bind guard seeded
    raising for "guard_raises" and left out (the window refuses its name) otherwise."""
    seeded = {}
    if mode != "unreadable":
        seeded["opti_oignon.security_mode"] = _mode_stand_in(mode)
    if guard == "guard_raises":
        seeded["opti_oignon.network_bind_guard"] = _raising_guard()
    loaded, restore = isolate(
        targets={"opti_oignon.startup_checks": source("startup_checks.py")},
        seeded=seeded,
    )
    return loaded["opti_oignon.startup_checks"], restore


def _others_pass(mod):
    """Every check but the bind check replaced by a pass, the roster derived from the module as the boot guard suite
    derives it, so no other check answers on the host."""
    for name in bg_t._roster(mod):
        if name != "_check_ollama_bind":
            setattr(mod, name, lambda _n=name: bg_t._passing_item(mod, _n))
    mod.clear_cache()


@pytest.mark.parametrize("guard", ("guard_raises", "guard_absent"))
@pytest.mark.parametrize("mode", _MODES)
def test_ub1_a_bind_check_that_cannot_look_is_not_checked_and_blocks_bulbe(mode, guard):
    mod, restore = _checklist(mode, guard)
    try:
        error = "OSError" if guard == "guard_raises" else "ModuleNotFoundError"
        item = mod._check_ollama_bind()
        assert item.passed is False, (
            f"the bind check failed before it could look ({error}) and reports itself passed: the check that "
            f"refuses the boot over an exposed Ollama then refuses nothing"
        )
        assert "not checked" in item.detail and error in item.detail, (
            f"the detail must say the bind address was not checked and name the error: {item.detail!r}"
        )
        if mode in _READ_AS_BULBE:
            assert item.severity == "critical" and "BLOCKED" in item.detail, (
                f"mode {mode!r} reads as Bulbe, and the unchecked bind is rated {item.severity!r}: {item.detail!r}"
            )
        else:
            assert item.severity == "warning", (
                f"Daily is told the check did not run and is not refused: rated {item.severity!r}"
            )

        _others_pass(mod)
        result = mod.run_startup_checks(force=True)
        assert result.all_passed is False
        assert result.blocked is (mode in _READ_AS_BULBE), (
            f"mode {mode!r}: the checklist's verdict is blocked={result.blocked} ({result.block_reason!r})"
        )
        if mode in _READ_AS_BULBE:
            assert "ollama_bind" in result.block_reason, result.block_reason
        report = {c["name"]: c for c in result.to_dict()["checks"]}["ollama_bind"]
        assert report["passed"] is False and "not checked" in report["detail"], (
            f"the status report must say the bind check did not run: {report!r}"
        )
    finally:
        restore()


@pytest.mark.parametrize("mode", _MODES)
def test_ub2_a_checklist_that_raises_refuses_the_boot_in_bulbe_and_lets_daily_boot(mode):
    mod, restore = _checklist(mode)
    try:
        def _boom(*, force=False):
            raise RuntimeError("machinery down")

        mod.run_startup_checks = _boom
        if mode in _READ_AS_BULBE:
            with pytest.raises(mod.StartupBlockedError) as raised:
                mod.enforce_boot_checks()
            said = str(raised.value)
            assert "RuntimeError" in said and "machinery down" in said and "Bulbe" in said, (
                f"the refusal must name the error and say why it refuses: {said!r}"
            )
            assert isinstance(raised.value.__cause__, RuntimeError)
        else:
            assert mod.enforce_boot_checks() is None, (
                "Daily: a checklist that cannot run is logged and the boot goes on"
            )
    finally:
        restore()


def _patched_mode(security_mode, mode):
    """The real security_mode's ``get_current_mode`` made to say ``mode``; returns its undo."""
    saved = security_mode.get_current_mode
    security_mode.get_current_mode = _mode_stand_in(mode).get_current_mode

    def undo():
        security_mode.get_current_mode = saved

    return undo


@pytest.mark.parametrize("mode", ("bulbe", "daily", "raising", "unrecognized"))
def test_ub3_the_lifespan_without_its_checklist_refuses_bulbe_and_boots_daily(mode):
    harness, restore = bg_t._real_app_harness()
    try:
        import opti_oignon.security_mode as security_mode

        undo_mode = _patched_mode(security_mode, mode)
        checklist = sys.modules["opti_oignon.startup_checks"]
        # The module's name holds None: its import raises, as a checklist that does not import.
        sys.modules["opti_oignon.startup_checks"] = None
        try:
            if mode in _READ_AS_BULBE:
                with pytest.raises(RuntimeError) as raised:
                    asyncio.run(bg_t._enter_and_exit_lifespan(harness["app"]))
                said = str(raised.value)
                assert "do not import" in said and "Bulbe" in said and "ModuleNotFoundError" in said, (
                    f"the refusal must say the checklist did not import and name the error: {said!r}"
                )
                assert harness["calls"]["plugins"] == 0 and harness["calls"]["arm"] == 0, (
                    f"a boot refused for want of its checklist armed services: {harness['calls']}"
                )
            else:
                assert asyncio.run(bg_t._enter_and_exit_lifespan(harness["app"])) is True
                assert harness["calls"]["arm"] == 1, (
                    f"Daily boots on past a checklist that does not import: {harness['calls']}"
                )
        finally:
            sys.modules["opti_oignon.startup_checks"] = checklist
            undo_mode()
    finally:
        restore()


@pytest.mark.parametrize("mode", ("bulbe", "daily", "unreadable"))
def test_ub4_a_tree_that_does_not_import_is_blocked_by_the_bind_check_alone(mode):
    # Nothing seeded but the mode: the window refuses every other name, so no check can import what it inspects.
    mod, restore = _checklist(mode)
    try:
        item = mod._check_pqc_primitive()
        assert item.passed is False and item.severity == "warning", (
            f"the post-quantum check could not import the module it inspects and is rated {item.severity!r} "
            f"(passed={item.passed}): a check that cannot look at the signing primitive is a warning, never a "
            f"refusal and never a pass"
        )
        result = mod.run_startup_checks(force=True)
        failed_critical = [c.name for c in result.checks if not c.passed and c.severity == "critical"]
        if mode in _READ_AS_BULBE:
            assert result.blocked is True and failed_critical == ["ollama_bind"], (
                f"mode {mode!r}: a tree that does not import is blocked by the unchecked bind alone, got "
                f"{failed_critical} ({result.block_reason!r})"
            )
        else:
            assert result.blocked is False and failed_critical == [], (
                f"Daily: a tree that does not import blocks nothing, got {failed_critical}"
            )
    finally:
        restore()


# The checks the runner calls, each with the name it is reported under; ub5 holds this list to the runner's source.
_RUNNER_CHECKS = (
    ("_check_code_signing_scripts", "code_signing_scripts"),
    ("_check_ollama_bind", "ollama_bind"),
    ("_check_luks", "luks_encryption"),
    ("_check_security_mode", "security_mode"),
    ("_check_encrypted_swap", "encrypted_swap"),
    ("_check_governor_ollama_limits", "governor_ollama_limits"),
    ("_check_pqc_primitive", "pqc_primitive"),
    ("_check_backend_provenance_coverage", "backend_provenance_coverage"),
    ("_check_process_secrets", "process_secrets"),
)


def _called_by_runner():
    """The check functions ``run_startup_checks`` calls, read from its source as the stub coverage suite reads it."""
    tree = ast.parse(source("startup_checks.py").read_text(encoding="utf-8"))
    runner = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run_startup_checks")
    return {
        call.func.id for call in ast.walk(runner)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id.startswith("_check_")
    }


class _Unprintable(ValueError):
    """An exception whose message cannot be made."""

    def __str__(self):
        raise RuntimeError("no message")


def _raising(error):
    def check():
        raise error
    return check


_UB5_CASES = [
    pytest.param(function, name, "bulbe", id=f"{function[len('_check_'):]}-bulbe") for function, name in _RUNNER_CHECKS
] + [pytest.param("_check_ollama_bind", "ollama_bind", "daily", id="ollama_bind-daily")]


@pytest.mark.parametrize(("function", "name", "mode"), _UB5_CASES)
def test_ub5_a_check_that_raises_is_reported_not_checked_and_the_others_still_run(function, name, mode):
    assert {f for f, _ in _RUNNER_CHECKS} == _called_by_runner(), (
        "the checks this contract lists are not the ones the runner calls"
    )
    mod, restore = _checklist(mode)
    try:
        for other, _ in _RUNNER_CHECKS:
            setattr(mod, other, lambda _n=other: bg_t._passing_item(mod, _n))
        if function == "_check_ollama_bind":
            error, error_name = _Unprintable("unused"), "_Unprintable"
        else:
            # A swap file whose name is not UTF-8, as /proc/swaps prints it.
            error = UnicodeDecodeError("utf-8", b"/swap\xe9", 5, 6, "invalid continuation byte")
            error_name = "UnicodeDecodeError"
        setattr(mod, function, _raising(error))
        mod.clear_cache()
        blocks = name == "ollama_bind" and mode == "bulbe"
        if blocks:
            with pytest.raises(mod.StartupBlockedError) as raised:
                mod.enforce_boot_checks()
            assert "ollama_bind" in str(raised.value), str(raised.value)
            result = mod.get_cached_result()
        else:
            result = mod.enforce_boot_checks()
            assert result is not None and result is mod.get_cached_result(), (
                f"a {name} check that raises stopped the checklist: the boot has no result to serve"
            )
        assert len(result.checks) == len(_RUNNER_CHECKS), (
            f"the checks after {name} did not run: {[c.name for c in result.checks]}"
        )
        item = {c.name: c for c in result.checks}[name]
        assert item.passed is False and "not checked" in item.detail and error_name in item.detail, (
            f"the {name} check that raised must be reported not checked, named by its error: {item.detail!r}"
        )
        assert item.severity == ("critical" if blocks else "warning"), (
            f"mode {mode!r}: the {name} check that raised is rated {item.severity!r}"
        )
        assert result.blocked is blocks, (
            f"mode {mode!r}: a {name} check that raised gives blocked={result.blocked} ({result.block_reason!r})"
        )
        assert all(c.passed for c in result.checks if c.name != name)
    finally:
        restore()


def test_ub6_the_boot_runs_the_checklist_afresh_never_an_earlier_runs_result():
    mode = _mode_stand_in("daily")
    loaded, restore = isolate(
        targets={"opti_oignon.startup_checks": source("startup_checks.py")},
        seeded={"opti_oignon.security_mode": mode, "opti_oignon.network_bind_guard": _raising_guard()},
    )
    mod = loaded["opti_oignon.startup_checks"]
    try:
        _others_pass(mod)
        earlier = mod.run_startup_checks()
        assert earlier.blocked is False, "Daily: the unchecked bind is a warning"
        mode.get_current_mode = _mode_stand_in("bulbe").get_current_mode
        with pytest.raises(mod.StartupBlockedError):
            mod.enforce_boot_checks()
        assert mod.get_cached_result() is not earlier and mod.get_cached_result().blocked, (
            "the boot in Bulbe served the result an earlier run left in Daily"
        )
    finally:
        restore()


@pytest.mark.parametrize("mode", ("bulbe", "daily"))
def test_ub7_a_check_that_returns_no_item_is_reported_not_checked(mode):
    mod, restore = _checklist(mode)
    try:
        for other, _ in _RUNNER_CHECKS:
            setattr(mod, other, lambda _n=other: bg_t._passing_item(mod, _n))
        mod._check_ollama_bind = lambda: {"passed": True}
        mod._check_luks = lambda: None
        mod.clear_cache()
        if mode == "bulbe":
            with pytest.raises(mod.StartupBlockedError) as raised:
                mod.enforce_boot_checks()
            assert "ollama_bind" in str(raised.value), str(raised.value)
            result = mod.get_cached_result()
        else:
            result = mod.enforce_boot_checks()
            assert result is not None and result.blocked is False
        assert len(result.checks) == len(_RUNNER_CHECKS), [c.name for c in result.checks]
        items = {c.name: c for c in result.checks}
        for name, returned in (("ollama_bind", "dict"), ("luks_encryption", "NoneType")):
            item = items[name]
            assert item.passed is False and "not checked" in item.detail and returned in item.detail, (
                f"a {name} check that returned {returned} must be reported not checked, saying what it returned: "
                f"{item.detail!r}"
            )
        assert items["ollama_bind"].severity == ("critical" if mode == "bulbe" else "warning")
        assert items["luks_encryption"].severity == "warning"
    finally:
        restore()


# One line of ``ss -tlnp`` whose process name holds the byte 0xe9 (octal 351, as printf reads it), which is not UTF-8.
_SS_LINES = {
    "exposed": r'LISTEN 0 4096 0.0.0.0:11434 0.0.0.0:* users:(("\351llama",pid=7,fd=3))\n',
    "loopback": r'LISTEN 0 4096 127.0.0.1:11434 0.0.0.0:* users:(("\351llama",pid=7,fd=3))\n',
}


@pytest.mark.parametrize("listening", ("exposed", "loopback"))
def test_ub8_the_bind_guard_reads_ss_whatever_bytes_a_process_name_holds(listening, monkeypatch):
    secrets = types.ModuleType("opti_oignon.process_secrets")
    secrets.child_env = lambda extra=None: {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    loaded, restore = isolate(
        targets={
            "opti_oignon.network_bind_guard": source("network_bind_guard.py"),
            "opti_oignon.startup_checks": source("startup_checks.py"),
        },
        seeded={"opti_oignon.security_mode": _mode_stand_in("bulbe"), "opti_oignon.process_secrets": secrets},
    )
    guard, checks = loaded["opti_oignon.network_bind_guard"], loaded["opti_oignon.startup_checks"]
    try:
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        # Nothing on the IPv4 table: the guard falls back to ss, played by a printf child whose bytes are decoded
        # as the guard asks.
        guard._check_ollama_proc_net_tcp = lambda port: None
        real_run = subprocess.run

        def ss_played_by_printf(argv, **kwargs):
            assert argv == ["ss", "-tlnp"], argv
            return real_run(["printf", _SS_LINES[listening]], **kwargs)

        monkeypatch.setattr(guard.subprocess, "run", ss_played_by_printf)
        result = guard.check_ollama_bind()
        assert result.checked and result.method == "ss_command", (
            f"a process name that is not UTF-8 kept the guard from reading ss: {result.to_dict()}"
        )
        item = checks._check_ollama_bind()
        assert "not checked" not in item.detail, item.detail
        if listening == "exposed":
            assert result.exposed and result.blocked and result.bind_address == "0.0.0.0", result.to_dict()
            assert item.passed is False and item.severity == "critical" and "BLOCKED" in item.detail, item.detail
        else:
            assert not result.exposed and result.bind_address == "127.0.0.1", result.to_dict()
            assert item.passed is True, item.detail
    finally:
        restore()
