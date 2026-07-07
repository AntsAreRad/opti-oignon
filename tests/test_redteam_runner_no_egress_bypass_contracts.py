#!/usr/bin/env python3
"""Contracts for the red team runner: no egress bypass at orchestration.

The loopback egress guard is enforced by each network entry point, and a
separate suite pins those entry points in isolation. This suite pins the
orchestrator on top of them: the runner must not trust a config that
merely passed validation at construction time. The realistic threat is a
config whose Ollama URL is repointed after construction (which skips the
config's own post-init check); the runner must still refuse, because it
re-derives the URL and hands it to entry points that re-check it, and it
must not swallow that refusal.

  * Contract 1 -- run_single on a repointed (non-loopback) config raises,
    before any socket, because the generator entry point re-checks and
    the runner does not swallow it.
  * Contract 2 -- run_campaign on a repointed config raises for the same
    reason.
  * Contract 3 -- a loopback config lets the generator and the chat
    target construct, so the guard does not break a valid local setup.
  * Contract 4 -- the chat target hop re-checks too: with a repointed
    config, resolving the chat target raises. Enforcement is not limited
    to the generator hop.

Local-only (the public distribution ships no tests). Runs under pytest
or the __main__ runner. The runner and its dependencies are loaded in
isolation under a stub package. No test opens a network connection: each
refusal happens before any socket, and the accepted case only constructs
objects (construction is network free).
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
    "opti_oignon.redteam.runner",
)

_REMOTE = "http://evil.example:11434"


def _load_modules():
    """Load config, targets, strategies, generator and runner in isolation."""
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

    # config first: the runner's lazy imports resolve against these.
    config = _load("opti_oignon.redteam.config", "config.py")
    _load("opti_oignon.redteam.targets", "targets.py")
    _load("opti_oignon.redteam.strategies", "strategies.py")
    _load("opti_oignon.redteam.generator", "generator.py")
    runner = _load("opti_oignon.redteam.runner", "runner.py")
    return config, runner, saved


def _restore(saved):
    for name in _MOD_NAMES:
        sys.modules.pop(name, None)
    for name, value in saved.items():
        if value is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = value


def _repointed_runner(config, runner):
    """A runner over a loopback config repointed to a remote URL.

    Building a non-loopback config directly is impossible (the config
    rejects it at construction), so a valid config is built and then its
    URL is repointed -- exactly the post-construction bypass the runner
    must still defend against.
    """
    cfg = config.RedTeamConfig()
    cfg.ollama_url = _REMOTE
    return runner.RedTeamRunner(config=cfg)


def _raises_value_error(thunk):
    try:
        thunk()
    except ValueError:
        return True
    return False


def test_c1_run_single_refuses_repointed_config():
    config, runner, saved = _load_modules()
    try:
        r = _repointed_runner(config, runner)
        assert _raises_value_error(
            lambda: r.run_single("prompt_injection", "raw", "chat")
        ), "run_single must raise on a repointed non-loopback config"
    finally:
        _restore(saved)


def test_c2_run_campaign_refuses_repointed_config():
    config, runner, saved = _load_modules()
    try:
        r = _repointed_runner(config, runner)
        assert _raises_value_error(
            lambda: r.run_campaign()
        ), "run_campaign must raise on a repointed non-loopback config"
    finally:
        _restore(saved)


def test_c3_loopback_config_constructs_components():
    config, runner, saved = _load_modules()
    try:
        r = runner.RedTeamRunner(config=config.RedTeamConfig())
        gen = r._ensure_generator()
        tgt = r._ensure_target("chat")
        assert gen is not None, "generator must construct under a loopback config"
        assert tgt is not None, "chat target must construct under a loopback config"
    finally:
        _restore(saved)


def test_c4_chat_target_hop_rechecks_repointed_config():
    config, runner, saved = _load_modules()
    try:
        r = _repointed_runner(config, runner)
        assert _raises_value_error(
            lambda: r._ensure_target("chat")
        ), "resolving the chat target must re-check and refuse a remote URL"
    finally:
        _restore(saved)


_TESTS = [
    test_c1_run_single_refuses_repointed_config,
    test_c2_run_campaign_refuses_repointed_config,
    test_c3_loopback_config_constructs_components,
    test_c4_chat_target_hop_rechecks_repointed_config,
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
