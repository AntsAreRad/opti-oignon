#!/usr/bin/env python3
"""Contracts for the Ollama bind guard.

The guard detects whether Ollama listens on a wildcard address and, in
Bulbe mode, marks the result blocked so the startup checklist refuses
the boot. Detection walks three methods in order: the OLLAMA_HOST
environment variable, /proc/net/tcp parsing, and an ``ss`` subprocess
fallback. These contracts pin each stage:

  * Contract 1 -- host extraction: schemes, ports, paths and bracketed
    IPv6 are stripped correctly from OLLAMA_HOST values.
  * Contract 2 -- environment method, both outcomes: a wildcard value
    reports exposed, a loopback value reports clean, and the method is
    attributed to the environment variable.
  * Contract 3 -- mode split on exposure, both outcomes: Bulbe marks
    the result blocked (with an explicit blocked detail and a critical
    audit event); Daily stays a warning, never blocked.
  * Contract 4 -- /proc/net/tcp decoding: little-endian hex addresses
    decode to dotted quads, only LISTEN sockets count, and a wildcard
    listener is recognized.
  * Contract 5 -- ss fallback: a ``*:port`` listener normalizes to the
    wildcard address; a loopback listener reports clean.
  * Contract 6 -- undeterminable: with no environment variable and both
    probes silent, the result is unchecked with method "none" and does
    not claim exposure.

Local-only (the public distribution ships no tests). Runs under pytest
or the __main__ runner. The guard module is loaded in isolation under a
stub package; the security mode and audit chain it imports lazily are
seeded as stubs so both mode outcomes are driven deterministically.
"""

import importlib.util
import os
import sys
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"

_OLLAMA_PORT_HEX = f"{11434:04X}"  # 2CAA


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


def _silence_probes(mod):
    """Make methods 2 and 3 return nothing, isolating the env method."""
    mod._check_ollama_proc_net_tcp = lambda port: None
    mod._check_ollama_ss = lambda port: None


# ---------------------------------------------------------------------------
# Contract 1 -- host extraction from OLLAMA_HOST values
# ---------------------------------------------------------------------------
def test_c1_extract_host_strips_scheme_port_and_brackets():
    mod, restore = _load_guard_module()
    try:
        cases = {
            "0.0.0.0": "0.0.0.0",
            "0.0.0.0:11434": "0.0.0.0",
            "http://0.0.0.0:11434": "0.0.0.0",
            "https://127.0.0.1:11434/api": "127.0.0.1",
            "127.0.0.1:11434": "127.0.0.1",
            "[::]:8080": "::",
            "localhost": "localhost",
        }
        for raw, expected in cases.items():
            got = mod._extract_host(raw)
            assert got == expected, (
                f"_extract_host({raw!r}) -> {got!r}, expected {expected!r}"
            )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 2 -- environment method, both outcomes
# ---------------------------------------------------------------------------
def test_c2_env_method_reports_exposed_and_clean():
    mod, restore = _load_guard_module(mode="daily")
    try:
        _silence_probes(mod)
        with _EnvGuard("0.0.0.0:11434"):
            result = mod.check_ollama_bind()
        assert result.checked and result.method == "env_OLLAMA_HOST"
        assert result.exposed, "a wildcard OLLAMA_HOST must report exposed"
        assert result.bind_address == "0.0.0.0"

        with _EnvGuard("127.0.0.1:11434"):
            result = mod.check_ollama_bind()
        assert result.checked and result.method == "env_OLLAMA_HOST"
        assert not result.exposed, "a loopback OLLAMA_HOST must report clean"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 3 -- mode split on exposure: Bulbe blocks, Daily warns
# ---------------------------------------------------------------------------
def test_c3_bulbe_blocks_and_daily_warns_on_exposure():
    audit_events = []
    mod, restore = _load_guard_module(mode="bulbe", audit_events=audit_events)
    try:
        _silence_probes(mod)
        with _EnvGuard("0.0.0.0"):
            result = mod.check_ollama_bind()
        assert result.exposed and result.blocked, (
            "Bulbe mode must mark an exposed Ollama as blocked"
        )
        assert "BLOCKED" in result.detail, (
            f"the blocked detail must be explicit, got: {result.detail!r}"
        )
        assert any(
            e.get("event_type") == "ollama_bind_exposed" for e in audit_events
        ), "the Bulbe exposure must land a critical audit event"
    finally:
        restore()

    mod, restore = _load_guard_module(mode="daily")
    try:
        _silence_probes(mod)
        with _EnvGuard("0.0.0.0"):
            result = mod.check_ollama_bind()
        assert result.exposed and not result.blocked, (
            "Daily mode must warn on exposure without blocking"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 4 -- /proc/net/tcp decoding (little-endian, LISTEN only)
# ---------------------------------------------------------------------------
def test_c4_proc_net_tcp_decodes_listeners():
    mod, restore = _load_guard_module()
    try:
        import builtins

        real_open = builtins.open

        def _serve(content):
            def _fake_open(path, *args, **kwargs):
                if str(path) == "/proc/net/tcp":
                    import io
                    return io.StringIO(content)
                return real_open(path, *args, **kwargs)
            return _fake_open

        header = (
            "  sl  local_address rem_address   st tx_queue rx_queue\n"
        )
        wildcard = (
            header
            + f"   0: 00000000:{_OLLAMA_PORT_HEX} 00000000:0000 0A "
            + "00000000:00000000 00:00000000 00000000\n"
        )
        loopback = (
            header
            + f"   0: 0100007F:{_OLLAMA_PORT_HEX} 00000000:0000 0A "
            + "00000000:00000000 00:00000000 00000000\n"
        )
        established_only = (
            header
            + f"   0: 00000000:{_OLLAMA_PORT_HEX} 00000000:0000 01 "
            + "00000000:00000000 00:00000000 00000000\n"
        )

        builtins.open = _serve(wildcard)
        try:
            got = mod._check_ollama_proc_net_tcp(11434)
        finally:
            builtins.open = real_open
        assert got == "0.0.0.0", (
            f"a wildcard hex listener must decode to 0.0.0.0, got {got!r}"
        )

        builtins.open = _serve(loopback)
        try:
            got = mod._check_ollama_proc_net_tcp(11434)
        finally:
            builtins.open = real_open
        assert got == "127.0.0.1", (
            f"the loopback hex must decode little-endian, got {got!r}"
        )

        builtins.open = _serve(established_only)
        try:
            got = mod._check_ollama_proc_net_tcp(11434)
        finally:
            builtins.open = real_open
        assert got is None, (
            f"non-LISTEN sockets must not count, got {got!r}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 5 -- ss fallback normalizes the wildcard star
# ---------------------------------------------------------------------------
def test_c5_ss_fallback_normalizes_wildcard():
    def _ss_stub(stdout):
        class _Completed:
            returncode = 0

        completed = _Completed()
        completed.stdout = stdout

        def _run(*args, **kwargs):
            return completed
        return _run

    mod, restore = _load_guard_module(mode="daily")
    try:
        _star = (
            "State  Recv-Q Send-Q Local Address:Port Peer Address:Port\n"
            "LISTEN 0      4096   *:11434            *:*\n"
        )
        mod._check_ollama_proc_net_tcp = lambda port: None
        mod.subprocess.run = _ss_stub(_star)
        with _EnvGuard(None):
            result = mod.check_ollama_bind()
        assert result.method == "ss_command"
        assert result.bind_address == "0.0.0.0", (
            f"a *:port listener must normalize to 0.0.0.0, got "
            f"{result.bind_address!r}"
        )
        assert result.exposed
    finally:
        restore()

    mod, restore = _load_guard_module(mode="daily")
    try:
        _loop = (
            "State  Recv-Q Send-Q Local Address:Port Peer Address:Port\n"
            "LISTEN 0      4096   127.0.0.1:11434    0.0.0.0:*\n"
        )
        mod._check_ollama_proc_net_tcp = lambda port: None
        mod.subprocess.run = _ss_stub(_loop)
        with _EnvGuard(None):
            result = mod.check_ollama_bind()
        assert result.method == "ss_command"
        assert not result.exposed, "a loopback ss listener must report clean"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 6 -- undeterminable stays unchecked and unexposed
# ---------------------------------------------------------------------------
def test_c6_undeterminable_is_unchecked_not_exposed():
    mod, restore = _load_guard_module()
    try:
        _silence_probes(mod)
        with _EnvGuard(None):
            result = mod.check_ollama_bind()
        assert not result.checked, "no signal must leave the result unchecked"
        assert result.method == "none"
        assert not result.exposed and not result.blocked, (
            "an undeterminable bind must not claim exposure"
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
