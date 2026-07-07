#!/usr/bin/env python3
"""Contracts for fail-secure mode normalization in the network bind guard.

Bulbe mode forces a loopback bind at the socket level. The guard reads the
current mode and must fall to the restrictive interpretation whenever that
mode is not positively a recognized value: an unreadable mode already
fails secure to Bulbe, but a malformed or unrecognized mode string (a
hand-edited security.yaml carrying "Bulbe", a trailing space, or an empty
value) must not be treated as the permissive Daily path. These contracts
pin that:

  * Contract 1 -- normalization: any value that is not exactly a
    recognized mode resolves to "bulbe".
  * Contract 2 -- bind forcing: under an unrecognized mode the safe bind
    address is loopback even when remote access is enabled in config, so
    the malformed mode cannot open a non-local bind.
  * Contract 3 -- Ollama block propagation: under an unrecognized
    (Bulbe-intended) mode an exposed Ollama is marked blocked, so the
    startup checklist still refuses the boot.
  * Contract 4 -- Daily is preserved (no over-correction): the recognized
    Daily mode with remote access and TLS present still allows remote and
    still binds to the requested host.
  * Contract 5 -- Bulbe is preserved: the recognized Bulbe mode still
    forces loopback.

Local-only (the public distribution ships no tests). Runs under pytest or
the __main__ runner. The guard module is loaded in isolation under a stub
package; the mode it imports lazily is seeded per clause.
"""

import importlib.util
import os
import sys
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


def _load_guard_module(mode="daily", audit_events=None):
    """Load network_bind_guard.py in isolation with a driven mode stub."""
    keys = (
        "opti_oignon",
        "opti_oignon.network_bind_guard",
        "opti_oignon.security_mode",
        "opti_oignon.signed_audit_log",
    )
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    mode_stub = types.ModuleType("opti_oignon.security_mode")
    mode_stub.get_current_mode = lambda: mode
    sys.modules["opti_oignon.security_mode"] = mode_stub
    pkg.security_mode = mode_stub

    audit_stub = types.ModuleType("opti_oignon.signed_audit_log")

    def _chain_log(**kwargs):
        if audit_events is not None:
            audit_events.append(kwargs)

    audit_stub.chain_log = _chain_log
    sys.modules["opti_oignon.signed_audit_log"] = audit_stub
    pkg.signed_audit_log = audit_stub

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.network_bind_guard", _OO / "network_bind_guard.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.network_bind_guard"] = mod
    spec.loader.exec_module(mod)
    pkg.network_bind_guard = mod

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return mod, restore


class _EnvGuard:
    """Set or unset OLLAMA_HOST for the duration of a clause."""

    def __init__(self, value):
        self.value = value
        self.saved = None

    def __enter__(self):
        self.saved = os.environ.get("OLLAMA_HOST")
        if self.value is None:
            os.environ.pop("OLLAMA_HOST", None)
        else:
            os.environ["OLLAMA_HOST"] = self.value
        return self

    def __exit__(self, *exc):
        if self.saved is None:
            os.environ.pop("OLLAMA_HOST", None)
        else:
            os.environ["OLLAMA_HOST"] = self.saved
        return False


_UNRECOGNIZED_MODES = ("Bulbe", "bulbe ", "", "unknown", "BULBE")


# ---------------------------------------------------------------------------
# Contract 1 -- unrecognized modes normalize to bulbe
# ---------------------------------------------------------------------------
def test_c1_unrecognized_mode_normalizes_to_bulbe():
    for bad in _UNRECOGNIZED_MODES:
        mod, restore = _load_guard_module(mode=bad)
        try:
            got = mod._get_current_mode()
            assert got == "bulbe", (
                f"an unrecognized mode {bad!r} must resolve to 'bulbe' "
                f"(fail-secure), got {got!r}"
            )
        finally:
            restore()


# ---------------------------------------------------------------------------
# Contract 2 -- unrecognized mode forces a loopback bind
# ---------------------------------------------------------------------------
def test_c2_unrecognized_mode_forces_loopback_bind():
    mod, restore = _load_guard_module(mode="Bulbe")
    try:
        # Even if remote access were enabled in config, an unrecognized
        # mode must not open a non-local bind.
        mod._is_remote_enabled_in_config = lambda: True
        got = mod.get_safe_bind_address("0.0.0.0")
        assert got == "127.0.0.1", (
            f"an unrecognized mode must force loopback, got {got!r}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 3 -- exposed Ollama under an unrecognized mode is blocked
# ---------------------------------------------------------------------------
def test_c3_unrecognized_mode_blocks_exposed_ollama():
    audit_events = []
    mod, restore = _load_guard_module(mode="Bulbe", audit_events=audit_events)
    try:
        mod._check_ollama_proc_net_tcp = lambda port: None
        mod._check_ollama_ss = lambda port: None
        with _EnvGuard("0.0.0.0"):
            result = mod.check_ollama_bind()
        assert result.exposed, "a wildcard OLLAMA_HOST must report exposed"
        assert result.blocked, (
            "an unrecognized (Bulbe-intended) mode must block an exposed "
            "Ollama so the boot is refused"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 4 -- recognized Daily is preserved (no over-correction)
# ---------------------------------------------------------------------------
def test_c4_recognized_daily_still_allows_remote():
    mod, restore = _load_guard_module(mode="daily")
    try:
        mod._is_remote_enabled_in_config = lambda: True
        mod._tls_files_exist = lambda: True
        assert mod.is_remote_access_allowed(), (
            "recognized Daily with remote + TLS must still allow remote"
        )
        got = mod.get_safe_bind_address("0.0.0.0")
        assert got == "0.0.0.0", (
            f"recognized Daily with remote enabled must bind the requested "
            f"host, got {got!r}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 5 -- recognized Bulbe still forces loopback
# ---------------------------------------------------------------------------
def test_c5_recognized_bulbe_still_forces_loopback():
    mod, restore = _load_guard_module(mode="bulbe")
    try:
        mod._is_remote_enabled_in_config = lambda: True
        got = mod.get_safe_bind_address("0.0.0.0")
        assert got == "127.0.0.1", (
            f"recognized Bulbe must force loopback, got {got!r}"
        )
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
