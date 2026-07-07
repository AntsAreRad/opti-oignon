#!/usr/bin/env python3
"""Contracts for the red team local-endpoint (loopback) egress guard.

The red team stack drives a local Ollama instance and analyses the
responses; the project's stated property is that it never reaches off
the local host. The default endpoint is loopback, but a default is not
a guarantee: a configuration file (or a directly constructed target)
could point the stack at a remote host. These contracts pin the guard
that makes the loopback property enforced rather than merely default,
at both the configuration boundary and every network entry point, and
pin that the sandbox target performs static analysis without ever
executing an attack payload.

  * Contract 1 -- the configuration boundary rejects a non-loopback
    Ollama URL outright.
  * Contract 2 -- loopback forms (IPv4 loopback range, localhost, IPv6
    ::1) are accepted, so the guard does not break valid setups.
  * Contract 3 -- every network entry point (chat target, generator,
    multilingual strategy) refuses a non-loopback URL at construction
    or entry, before any socket is opened.
  * Contract 4 -- the sandbox target flags an escape payload by static
    analysis and never executes it (a shell-escape payload leaves no
    side effect on the host).

Local-only (the public distribution ships no tests). Runs under pytest
or the __main__ runner. Modules are loaded in isolation under a stub
package so the full application import chain is not triggered.
"""

import importlib.util
import sys
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"

_MOD_NAMES = (
    "opti_oignon.redteam.config",
    "opti_oignon.redteam.targets",
    "opti_oignon.redteam.strategies",
    "opti_oignon.redteam.generator",
)


def _load_modules():
    """Load the four red team modules in isolation under a stub package.

    Returns (config, targets, strategies, generator, saved) where saved
    lets the caller restore sys.modules afterwards.
    """
    saved = {
        name: sys.modules.get(name)
        for name in ("opti_oignon", "opti_oignon.redteam") + _MOD_NAMES
    }

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg
    rt = types.ModuleType("opti_oignon.redteam")
    rt.__path__ = []
    sys.modules["opti_oignon.redteam"] = rt
    pkg.redteam = rt

    def _load(mod_name, filename):
        spec = importlib.util.spec_from_file_location(
            mod_name, _OO / "redteam" / filename,
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    # config is loaded first: the call sites import the guard from it.
    config = _load("opti_oignon.redteam.config", "config.py")
    targets = _load("opti_oignon.redteam.targets", "targets.py")
    strategies = _load("opti_oignon.redteam.strategies", "strategies.py")
    generator = _load("opti_oignon.redteam.generator", "generator.py")
    return config, targets, strategies, generator, saved


def _restore(saved):
    for name in _MOD_NAMES:
        sys.modules.pop(name, None)
    for name, value in saved.items():
        if value is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = value


def _raises_value_error(thunk):
    try:
        thunk()
    except ValueError:
        return True
    return False


def test_c1_config_rejects_non_loopback_url():
    config, _t, _s, _g, saved = _load_modules()
    try:
        assert _raises_value_error(
            lambda: config.RedTeamConfig(ollama_url="http://evil.example:11434")
        ), "config must reject a non-loopback ollama_url"
        # A bare remote host with no scheme must also be refused.
        assert _raises_value_error(
            lambda: config.RedTeamConfig(ollama_url="http://8.8.8.8:11434")
        ), "config must reject a routable remote address"
    finally:
        _restore(saved)


def test_c2_loopback_forms_accepted():
    config, _t, _s, _g, saved = _load_modules()
    try:
        for url in (
            "http://127.0.0.1:11434",
            "http://localhost:11434",
            "http://127.0.0.5:11434",
            "http://[::1]:11434",
        ):
            # Must construct without raising.
            config.RedTeamConfig(ollama_url=url)
    finally:
        _restore(saved)


def test_c3_call_sites_refuse_non_loopback():
    config, targets, strategies, generator, saved = _load_modules()
    try:
        bad = "http://evil.example:11434"
        assert _raises_value_error(
            lambda: targets.ChatTarget(ollama_url=bad)
        ), "ChatTarget must refuse a non-loopback url at construction"
        assert _raises_value_error(
            lambda: generator.AttackGenerator(ollama_url=bad)
        ), "AttackGenerator must refuse a non-loopback url at construction"
        assert _raises_value_error(
            lambda: strategies.strategy_multilingual("payload", ollama_url=bad)
        ), "multilingual strategy must refuse a non-loopback url at entry"
    finally:
        _restore(saved)


def test_c4_sandbox_target_flags_without_executing():
    config, targets, _s, _g, saved = _load_modules()
    marker = Path("/tmp/oo_redteam_exec_marker")
    try:
        if marker.exists():
            marker.unlink()
        payload = f"import os\nos.system('touch {marker}')"
        result = targets.SandboxTarget().run(payload)
        assert not marker.exists(), (
            "sandbox target must NOT execute attack payloads"
        )
        assert result.blocked, "an escape payload must be flagged"
        assert "os.system(" in result.metadata.get("dangerous_patterns", []), (
            "static analysis must record the dangerous pattern"
        )
    finally:
        if marker.exists():
            marker.unlink()
        _restore(saved)


_TESTS = [
    test_c1_config_rejects_non_loopback_url,
    test_c2_loopback_forms_accepted,
    test_c3_call_sites_refuse_non_loopback,
    test_c4_sandbox_target_flags_without_executing,
]


def _main():
    passed = 0
    for test in _TESTS:
        try:
            test()
        except Exception:
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
        else:
            print(f"PASS {test.__name__}")
            passed += 1
    print(f"{passed}/{len(_TESTS)} passed")
    return 0 if passed == len(_TESTS) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
