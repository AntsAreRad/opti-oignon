#!/usr/bin/env python3
"""S213 -- Sandbox Workspace cycle, Bloc 4: the optional Daily-only network.

Per-fix suite for SANDBOX_WORKSPACE_SPEC section 8: the per-workspace
network flag (default off, user action only, never a config default, never
model-triggerable), the binding-layer gate in sandbox_egress.py (strictly
daily-only -- an unset, unknown, or undeterminable mode is treated as
Bulbe, fail-secure), the provision phase as the shipped egress mechanism
(a server-built, hash-pinned ``--require-hashes --only-binary=:all:``
install into a workspace venv; arbitrary task code never touches the
network -- every task argv keeps the unconditional ``--unshare-net``, the
provision argv differs exactly by its omission plus the name-resolution
file binds), the audit rows (network_on/network_off, the _refused
discipline, provision_run, the hash-chain rows), the route ladders for
``POST /{id}/network`` and ``POST /{id}/provision`` with destroy still the
LAST registered route, and the spec / cartography / FRONTEND_REDESIGN /
yaml / types / client registrations. The BULBE REFUSAL SUITE comes FIRST.

Harness: the s210/s211/s212 ``_load_fresh`` shape -- other suites in the
sweep pre-load the real package chain, so this file ALWAYS execs its own
module copies and re-pins them per test.
"""

import importlib.util
import os
import re
import sys
import types

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_OO = os.path.join(_ROOT, "opti_oignon")
_API = os.path.join(_OO, "api")
_AGENT = os.path.join(_OO, "agent")
_FRONT = os.path.join(_ROOT, "frontend", "src", "lib")


def _ensure_pkg(name: str, path: str) -> None:
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = [path]
        sys.modules[name] = mod


_ensure_pkg("opti_oignon", _OO)
_ensure_pkg("opti_oignon.api", _API)
_ensure_pkg("opti_oignon.agent", _AGENT)


def _load_fresh(relpath: str, register: str, bind: dict | None = None):
    """ALWAYS exec this file's own copy; never reuse a pre-loaded module.

    The s210-documented sweep-order class: other suites import the whole
    real opti_oignon chain, pre-loading the canonical names; reusing those
    would split exception/class identity. Temporarily register ``bind``
    plus the module's own name, exec the fresh copy, restore every touched
    sys.modules entry afterwards.
    """
    bind = dict(bind or {})
    touched = list(bind.keys()) + [register]
    saved = {name: sys.modules.get(name) for name in touched}
    try:
        for name, mod in bind.items():
            sys.modules[name] = mod
        path = os.path.join(_ROOT, relpath)
        spec = importlib.util.spec_from_file_location(register, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[register] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        for name, prior in saved.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


_sm = _load_fresh(
    os.path.join("opti_oignon", "sandbox_manager.py"),
    register="opti_oignon.sandbox_manager",
)
_st = _load_fresh(
    os.path.join("opti_oignon", "sandbox_tools.py"),
    register="opti_oignon.sandbox_tools",
    bind={"opti_oignon.sandbox_manager": _sm},
)
_ws = _load_fresh(
    os.path.join("opti_oignon", "sandbox_workspace.py"),
    register="opti_oignon.sandbox_workspace",
    bind={
        "opti_oignon.sandbox_manager": _sm,
        "opti_oignon.sandbox_tools": _st,
    },
)
_eg = _load_fresh(
    os.path.join("opti_oignon", "sandbox_egress.py"),
    register="opti_oignon.sandbox_egress",
)

SandboxConfig = _sm.SandboxConfig

GOOD_HASH = "a" * 64
GOOD_LINE = f"requests==2.31.0 --hash=sha256:{GOOD_HASH}"


@pytest.fixture(autouse=True)
def _bind_module_copies(monkeypatch):
    """Bind THIS file's module copies for the duration of each test."""
    pairs = {
        "opti_oignon.sandbox_manager": _sm,
        "opti_oignon.sandbox_tools": _st,
        "opti_oignon.sandbox_workspace": _ws,
        "opti_oignon.sandbox_egress": _eg,
    }
    for name, mod in pairs.items():
        monkeypatch.setitem(sys.modules, name, mod)
        parent = sys.modules.get("opti_oignon")
        if parent is not None:
            monkeypatch.setattr(
                parent, name.rsplit(".", 1)[1], mod, raising=False
            )
    yield


@pytest.fixture(autouse=True)
def _fresh_stores():
    # The S213 surface gate: every test in this suite targets a Bloc 4
    # mechanic, so the suite must be RED against a pre-S213 tree (the
    # pristine proof). Touching the S213 exception base here makes that
    # explicit -- the attribute does not exist before this session.
    _ = _eg.SandboxNetworkDisabledInBulbe
    _ws.reset_workspace_bindings()
    _ws.reset_workspace_manifests()
    yield
    _ws.reset_workspace_bindings()
    _ws.reset_workspace_manifests()


def _make_manager(tmp_path, **cfg_kw):
    defaults = dict(
        workspace_base=str(tmp_path / "sbx"),
        audit_db_path="audit.db",
        isolation_backend="tempdir",
        require_degraded_confirmation=False,
        strict_mode=False,
        idle_ttl_seconds=0,
    )
    defaults.update(cfg_kw)
    return _sm.SandboxManager(config=SandboxConfig(**defaults))


@pytest.fixture()
def manager(tmp_path):
    return _make_manager(tmp_path)


def _gate(monkeypatch, allowed: bool):
    """Force the manager-side gate answer (the binding layer under test)."""
    monkeypatch.setattr(
        _sm.SandboxManager,
        "_network_gate_allows",
        staticmethod(lambda: allowed),
    )


def _approval_actions(manager, sid):
    return [row["action"] for row in manager.audit.get_approval_log(sid)]


def _chain_recorder(monkeypatch):
    """Install a recording stand-in for the hash-chain logger."""
    calls: list[dict] = []
    mod = types.ModuleType("opti_oignon.signed_audit_log")

    def chain_log(event_type, source="", action="", severity="INFO", **kw):
        calls.append(
            dict(event_type=event_type, source=source, action=action,
                 severity=severity, **kw)
        )
        return 1

    mod.chain_log = chain_log
    monkeypatch.setitem(sys.modules, "opti_oignon.signed_audit_log", mod)
    return calls


def _fake_mode(monkeypatch, value):
    """Drive the egress gate through a fake security_mode module."""
    mod = types.ModuleType("opti_oignon.security_mode")
    if isinstance(value, BaseException) or (
        isinstance(value, type) and issubclass(value, BaseException)
    ):
        def get_current_mode():
            raise value if isinstance(value, BaseException) else value()
    else:
        def get_current_mode():
            return value

    mod.get_current_mode = get_current_mode
    monkeypatch.setitem(sys.modules, "opti_oignon.security_mode", mod)


# ---------------------------------------------------------------------------
# 1. THE BULBE REFUSAL SUITE (first, per the bloc contract)
# ---------------------------------------------------------------------------


class TestBulbeRefusalGate:
    """The binding-layer gate: strictly daily-only, fail-secure."""

    def test_daily_allows(self, monkeypatch):
        _fake_mode(monkeypatch, "daily")
        assert _eg.network_allowed() is True

    def test_bulbe_refuses(self, monkeypatch):
        _fake_mode(monkeypatch, "bulbe")
        assert _eg.network_allowed() is False

    @pytest.mark.parametrize("mode", ["", "unknown", "DAILY", "remote", None, 0])
    def test_unknown_or_unset_mode_is_bulbe(self, monkeypatch, mode):
        # STRICTER than the veilid guard's bulbe-equality test: anything
        # that is not exactly "daily" refuses (spec 8.3).
        _fake_mode(monkeypatch, mode)
        assert _eg.network_allowed() is False

    def test_mode_read_exception_is_bulbe(self, monkeypatch):
        _fake_mode(monkeypatch, RuntimeError("boom"))
        assert _eg.current_mode() == "bulbe"
        assert _eg.network_allowed() is False

    def test_security_mode_absent_is_bulbe(self, monkeypatch):
        # None in sys.modules makes the lazy import raise: fail-secure.
        monkeypatch.setitem(sys.modules, "opti_oignon.security_mode", None)
        assert _eg.current_mode() == "bulbe"
        assert _eg.network_allowed() is False

    def test_assert_raises_dedicated_exception(self, monkeypatch):
        _fake_mode(monkeypatch, "bulbe")
        with pytest.raises(_eg.SandboxNetworkDisabledInBulbe):
            _eg.assert_network_allowed()

    def test_assert_passes_in_daily(self, monkeypatch):
        _fake_mode(monkeypatch, "daily")
        _eg.assert_network_allowed()  # must not raise

    def test_exception_hierarchy(self):
        assert issubclass(
            _eg.SandboxNetworkDisabledInBulbe, _eg.SandboxEgressError
        )
        assert issubclass(_eg.ProvisionValidationError, _eg.SandboxEgressError)


class TestBulbeRefusalManager:
    """Toggle-on and every egress refused outside Daily, at the manager."""

    def test_toggle_on_refused_under_bulbe(self, manager, monkeypatch):
        manager.create_sandbox("ws-b1")
        _gate(monkeypatch, False)
        with pytest.raises(PermissionError) as exc:
            manager.set_network_enabled("ws-b1", True, actor="alice")
        assert "Bulbe" in str(exc.value)
        assert manager.get_session("ws-b1").network_enabled is False
        assert "network_refused" in _approval_actions(manager, "ws-b1")
        assert "network_on" not in _approval_actions(manager, "ws-b1")

    def test_toggle_off_allowed_under_bulbe(self, manager, monkeypatch):
        manager.create_sandbox("ws-b2")
        manager.get_session("ws-b2").network_enabled = True
        _gate(monkeypatch, False)
        assert manager.set_network_enabled("ws-b2", False) is False
        assert manager.get_session("ws-b2").network_enabled is False
        assert "network_off" in _approval_actions(manager, "ws-b2")

    def test_egress_refused_under_bulbe_even_with_flag_on(
        self, manager, monkeypatch
    ):
        # The "flag somehow on" case: forced directly, bypassing the
        # toggle; the executor must STILL refuse on the live gate.
        manager.create_sandbox("ws-b3")
        manager.get_session("ws-b3").network_enabled = True
        _gate(monkeypatch, False)
        result = manager.execute_provision_command("ws-b3", "echo x")
        assert result.blocked is True
        assert "Daily-only" in result.block_reason
        assert "provision_refused" in _approval_actions(manager, "ws-b3")

    def test_gate_resolver_fail_secure_when_module_absent(
        self, manager, monkeypatch
    ):
        # None in sys.modules makes the resolver's import raise: an
        # undeterminable gate refuses, it never permits.
        monkeypatch.setitem(sys.modules, "opti_oignon.sandbox_egress", None)
        assert _sm.SandboxManager._network_gate_allows() is False

    def test_gate_resolver_fail_secure_when_gate_errors(
        self, manager, monkeypatch
    ):
        broken = types.ModuleType("opti_oignon.sandbox_egress")

        def network_allowed():
            raise RuntimeError("gate broke")

        broken.network_allowed = network_allowed
        monkeypatch.setitem(
            sys.modules, "opti_oignon.sandbox_egress", broken
        )
        assert _sm.SandboxManager._network_gate_allows() is False

    def test_gate_resolver_consults_egress_module(self, monkeypatch):
        stub = types.ModuleType("opti_oignon.sandbox_egress")
        stub.network_allowed = lambda: True
        monkeypatch.setitem(sys.modules, "opti_oignon.sandbox_egress", stub)
        assert _sm.SandboxManager._network_gate_allows() is True
        stub.network_allowed = lambda: False
        assert _sm.SandboxManager._network_gate_allows() is False


class TestDefaultOffPinned:
    """Default-off, no config surface, no model trigger."""

    def test_dataclass_default_false(self):
        session = _sm.SandboxSession(session_id="x", workspace_path="/tmp/x")
        assert session.network_enabled is False

    def test_create_sandbox_default_false(self, manager):
        created = manager.create_sandbox("ws-d1")
        assert created.network_enabled is False
        assert manager.get_session("ws-d1").network_enabled is False

    def test_no_config_key_enables_network(self):
        # The ONE new key is a timeout; no SandboxConfig field can flip the
        # flag (spec 8.3: never a config default).
        import dataclasses

        names = {f.name for f in dataclasses.fields(SandboxConfig)}
        assert "provision_timeout_seconds" in names
        network_like = {
            n for n in names if "network" in n or "egress" in n
        }
        assert network_like == set()

    def test_list_sessions_surfaces_flag(self, manager):
        manager.create_sandbox("ws-d2")
        rows = {r["session_id"]: r for r in manager.list_sessions()}
        assert rows["ws-d2"]["network_enabled"] is False

    def test_dispatch_carries_no_network_surface(self):
        src = open(
            os.path.join(_AGENT, "dispatch.py"), encoding="utf-8"
        ).read()
        for needle in (
            "network", "provision", "egress",
            "set_network_enabled", "execute_provision_command",
        ):
            assert needle not in src, f"dispatch.py must not carry {needle!r}"
        assert "S213" not in src

    def test_sandbox_tools_carries_no_network_surface(self):
        src = open(
            os.path.join(_OO, "sandbox_tools.py"), encoding="utf-8"
        ).read()
        for needle in (
            "set_network_enabled", "execute_provision_command", "provision",
        ):
            assert needle not in src
        assert "S213" not in src


# ---------------------------------------------------------------------------
# 2. The toggle: audit on/off, who and when
# ---------------------------------------------------------------------------


class TestToggleAudit:
    def test_enable_in_daily_sets_flag_and_audits(self, manager, monkeypatch):
        manager.create_sandbox("ws-t1")
        _gate(monkeypatch, True)
        chain = _chain_recorder(monkeypatch)
        assert manager.set_network_enabled("ws-t1", True, actor="alice") is True
        assert manager.get_session("ws-t1").network_enabled is True
        rows = manager.audit.get_approval_log("ws-t1")
        on_rows = [r for r in rows if r["action"] == "network_on"]
        assert len(on_rows) == 1 and "actor=alice" in on_rows[0]["detail"]
        toggles = [c for c in chain if c["event_type"] == "sandbox_network_toggle"]
        assert len(toggles) == 1
        assert toggles[0]["severity"] == "WARNING"
        assert toggles[0]["enabled"] is True
        assert toggles[0]["actor"] == "alice"

    def test_disable_audits_off_row(self, manager, monkeypatch):
        manager.create_sandbox("ws-t2")
        _gate(monkeypatch, True)
        chain = _chain_recorder(monkeypatch)
        manager.set_network_enabled("ws-t2", True, actor="alice")
        manager.set_network_enabled("ws-t2", False, actor="alice")
        actions = _approval_actions(manager, "ws-t2")
        assert actions.count("network_on") == 1
        assert actions.count("network_off") == 1
        offs = [
            c for c in chain
            if c["event_type"] == "sandbox_network_toggle" and not c["enabled"]
        ]
        assert len(offs) == 1 and offs[0]["severity"] == "INFO"

    def test_unknown_session_valueerror(self, manager, monkeypatch):
        _gate(monkeypatch, True)
        with pytest.raises(ValueError):
            manager.set_network_enabled("ws-none", True)

    def test_toggle_touches_activity(self, manager, monkeypatch):
        manager.create_sandbox("ws-t3")
        _gate(monkeypatch, True)
        before = manager.get_session("ws-t3").last_activity
        import time as _time

        _time.sleep(0.01)
        manager.set_network_enabled("ws-t3", True)
        assert manager.get_session("ws-t3").last_activity >= before


# ---------------------------------------------------------------------------
# 3. The provision command shape and the requirements validation
# ---------------------------------------------------------------------------


class TestProvisionCommandShape:
    def test_command_contract(self):
        cmd = _eg.build_provision_command("requirements.txt", "deps/.venv")
        assert "--require-hashes" in cmd
        assert "--only-binary=:all:" in cmd
        assert "--no-cache-dir" in cmd
        assert "--no-input" in cmd
        assert "python3 -m venv --clear '/workspace/deps/.venv'" in cmd
        assert "-r '/workspace/requirements.txt'" in cmd
        assert "'/workspace/deps/.venv/bin/python' -m pip install" in cmd

    def test_default_venv_dir(self):
        cmd = _eg.build_provision_command("requirements.txt")
        assert "'/workspace/.venv'" in cmd

    @pytest.mark.parametrize("bad", [
        "/abs/req.txt", "../req.txt", "a/../../b", "", ".", "a\x00b", "a\\b",
    ])
    def test_bad_requirements_path_refused(self, bad):
        with pytest.raises(_eg.ProvisionValidationError):
            _eg.build_provision_command(bad, ".venv")

    @pytest.mark.parametrize("bad", ["/abs", "..", "x/../..", "", "v\x00"])
    def test_bad_venv_dir_refused(self, bad):
        with pytest.raises(_eg.ProvisionValidationError):
            _eg.build_provision_command("requirements.txt", bad)

    def test_refuse_rel_path_accepts_normal(self):
        assert _eg.refuse_rel_path("a/b/c.txt") is None
        assert _eg.refuse_rel_path("requirements.txt") is None

    def test_command_validator_accepts_shape(self, tmp_path):
        # Defense in depth retained: the server-built line passes the real
        # yaml blocklist (it contains no curl/wget/nc tokens).
        mgr = _make_manager(
            tmp_path,
            blocked_commands=["curl", "wget", "nc ", "ncat", "netcat"],
        )
        cmd = _eg.build_provision_command("requirements.txt")
        is_safe, reason = mgr._validator.validate(cmd)
        assert is_safe, reason


class TestRequirementsValidation:
    def test_pinned_with_hash_accepted(self):
        accepted, refused = _eg.validate_requirements_text(GOOD_LINE + "\n")
        assert accepted == ["requests"] and refused == []

    def test_multi_hash_and_extras_accepted(self):
        text = (
            f"uvicorn[standard]==0.30.1 --hash=sha256:{'b' * 64} "
            f"--hash=sha256:{'c' * 64}\n"
        )
        accepted, refused = _eg.validate_requirements_text(text)
        assert accepted == ["uvicorn"] and refused == []

    def test_continuation_lines_assembled(self):
        text = (
            "flask==3.0.0 \\\n"
            f"    --hash=sha256:{'d' * 64} \\\n"
            f"    --hash=sha256:{'e' * 64}\n"
        )
        accepted, refused = _eg.validate_requirements_text(text)
        assert accepted == ["flask"] and refused == []

    def test_comments_and_blanks_ignored(self):
        text = f"# pinned set\n\n{GOOD_LINE}\n"
        accepted, refused = _eg.validate_requirements_text(text)
        assert accepted == ["requests"] and refused == []

    @pytest.mark.parametrize("line,why", [
        ("requests>=2.0", "range"),
        ("requests", "bare name"),
        ("requests==2.31.0", "pinned but hashless"),
        ("-r other.txt", "nested file option"),
        ("--index-url https://evil.example", "index option"),
        ("-e .", "editable option"),
        ("--trusted-host evil.example", "trust option"),
        ("https://evil.example/pkg.whl", "direct URL"),
        ("git+https://evil.example/repo", "vcs URL"),
    ])
    def test_non_pinned_or_option_lines_refused(self, line, why):
        accepted, refused = _eg.validate_requirements_text(line + "\n")
        assert refused, why
        assert refused[0]["line"] == 1
        assert accepted == []

    def test_nul_byte_refused(self):
        accepted, refused = _eg.validate_requirements_text("a==1\x00 --hash=sha256:" + "f" * 64)
        assert refused and "NUL" in refused[0]["reason"]

    def test_partial_set_reports_every_refusal(self):
        text = f"{GOOD_LINE}\nflask\n--find-links /tmp\n"
        accepted, refused = _eg.validate_requirements_text(text)
        assert accepted == ["requests"]
        assert [r["line"] for r in refused] == [2, 3]

    def test_oversize_file_refused_whole(self):
        text = ("#" + "x" * 100 + "\n") * 4000
        accepted, refused = _eg.validate_requirements_text(text)
        assert accepted == [] and refused and refused[0]["line"] == 0


# ---------------------------------------------------------------------------
# 4. The argv contract: default byte-stable, the network delta exact
# ---------------------------------------------------------------------------


class TestArgvContract:
    def _argvs(self):
        cfg = SandboxConfig()
        base = _sm._build_bwrap_command("echo hi", "/tmp/ws", cfg, seccomp_fd=7)
        net = _sm._build_bwrap_command(
            "echo hi", "/tmp/ws", cfg, seccomp_fd=7, allow_network=True
        )
        return base, net

    def test_default_equals_omitted_kwarg(self):
        cfg = SandboxConfig()
        a = _sm._build_bwrap_command("echo hi", "/tmp/ws", cfg, seccomp_fd=7)
        b = _sm._build_bwrap_command(
            "echo hi", "/tmp/ws", cfg, seccomp_fd=7, allow_network=False
        )
        assert a == b

    def test_default_argv_shape_preserved(self):
        base, _ = self._argvs()
        # The S183/S209 pinned shape, re-asserted: clearenv first, net/pid
        # explicit, --size immediately before --tmpfs /tmp.
        assert base[0] == "bwrap" and base[1] == "--clearenv"
        assert base.count("--unshare-net") == 1
        for flag in ("--unshare-pid", "--unshare-ipc", "--unshare-uts",
                     "--unshare-cgroup", "--new-session", "--die-with-parent"):
            assert flag in base
        assert base[base.index("--tmpfs") + 1] == "/tmp"
        assert base[base.index("--size") + 2] == "--tmpfs"
        # No name-resolution bind ever appears on a task argv.
        assert "/etc/resolv.conf" not in base
        assert "/etc/hosts" not in base
        assert "/etc/nsswitch.conf" not in base

    def test_network_argv_delta_exact(self):
        base, net = self._argvs()
        assert "--unshare-net" not in net
        for flag in ("--unshare-pid", "--unshare-ipc", "--unshare-uts",
                     "--unshare-cgroup", "--new-session", "--die-with-parent"):
            assert flag in net
        # Reconstruct: removing the dns binds and re-inserting
        # --unshare-net before --unshare-pid must give back the base argv
        # exactly -- nothing else may differ.
        stripped = list(net)
        for ns_file in ("/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf"):
            while ns_file in stripped:
                i = stripped.index(ns_file)
                # the triplet is --ro-bind <src> <dest>; dest == ns_file and
                # src may be the realpath, so cut around the dest.
                j = i
                while stripped[j] != "--ro-bind":
                    j -= 1
                del stripped[j:j + 3]
        stripped.insert(stripped.index("--unshare-pid"), "--unshare-net")
        assert stripped == base

    def test_network_argv_binds_only_existing_files(self):
        _, net = self._argvs()
        # every name-resolution triple is --ro-bind <realpath(src)> <dest>
        ns_targets = ("/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf")
        seen = []
        for j, tok in enumerate(net):
            if tok == "--ro-bind" and net[j + 2] in ns_targets:
                src, dest = net[j + 1], net[j + 2]
                assert os.path.isfile(src)
                assert src == os.path.realpath(dest)
                seen.append(dest)
        # the existing ones in this environment are all bound, none twice
        expected = [p for p in ns_targets if os.path.isfile(os.path.realpath(p))]
        assert seen == expected

    def test_seccomp_flag_on_both(self):
        base, net = self._argvs()
        assert base[base.index("--seccomp") + 1] == "7"
        assert net[net.index("--seccomp") + 1] == "7"

    def test_run_bwrap_threads_allow_network(self, manager, monkeypatch):
        recorded = {}

        def fake_build(command, workspace, config, seccomp_fd=None,
                       *, allow_network=False):
            recorded["allow_network"] = allow_network
            return ["true"]

        monkeypatch.setattr(_sm, "_build_bwrap_command", fake_build)
        monkeypatch.setattr(
            manager, "_spawn_tracked",
            lambda argv, **kw: __import__("subprocess").CompletedProcess(
                argv, 0, b"", b""
            ),
        )
        manager._config.seccomp_enabled = False
        manager._run_bwrap("echo hi", "/tmp/ws", 5)
        assert recorded["allow_network"] is False
        manager._run_bwrap("echo hi", "/tmp/ws", 5, allow_network=True)
        assert recorded["allow_network"] is True

    def test_only_provision_seam_passes_allow_network_true(self):
        # Network-off-again BY CONSTRUCTION: by source, the only CODE site
        # that ever forwards allow_network=True is the provision executor
        # (comments and docstrings excluded).
        import ast

        path = os.path.join(_OO, "sandbox_manager.py")
        src = open(path, encoding="utf-8").read()
        tree = ast.parse(src)
        true_sites = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.keyword)
            and node.arg == "allow_network"
            and isinstance(node.value, ast.Constant)
            and node.value.value is True
        ]
        assert len(true_sites) == 1, [n.lineno for n in true_sites]

        segments = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in (
                "execute_command", "execute_provision_command",
            ):
                segments[node.name] = ast.get_source_segment(src, node)
        prov = segments["execute_provision_command"]
        assert "allow_network=True" in prov
        # execute_command (the task path) never mentions the grant
        assert "allow_network" not in segments["execute_command"]


# ---------------------------------------------------------------------------
# 5. The provision executor: refusals fail-secure, audited; the happy seam
# ---------------------------------------------------------------------------


class TestProvisionExecutor:
    def test_flag_off_refused(self, manager, monkeypatch):
        manager.create_sandbox("ws-p1")
        _gate(monkeypatch, True)
        result = manager.execute_provision_command("ws-p1", "echo x")
        assert result.blocked is True
        assert "not enabled" in result.block_reason
        assert "provision_refused" in _approval_actions(manager, "ws-p1")

    def test_bwrap_absence_refused_fail_secure(self, manager, monkeypatch):
        # The container posture: tempdir backend (or no bwrap) refuses --
        # a network-on tempdir run would be raw host execution.
        manager.create_sandbox("ws-p2")
        manager.get_session("ws-p2").network_enabled = True
        _gate(monkeypatch, True)
        result = manager.execute_provision_command("ws-p2", "echo x")
        assert result.blocked is True
        assert "bwrap" in result.block_reason
        assert "provision_refused" in _approval_actions(manager, "ws-p2")

    def test_validator_still_applies(self, manager, monkeypatch):
        manager.create_sandbox("ws-p3")
        manager.get_session("ws-p3").network_enabled = True
        _gate(monkeypatch, True)
        monkeypatch.setattr(
            manager, "_isolation_backend", _sm.IsolationBackend.BWRAP
        )
        monkeypatch.setattr(manager, "_bwrap_available", True)
        monkeypatch.setattr(
            manager._validator, "validate",
            lambda cmd: (False, "blocked for the test"),
        )
        result = manager.execute_provision_command("ws-p3", "echo x")
        assert result.blocked is True
        assert "validation" in result.block_reason
        assert "provision_refused" in _approval_actions(manager, "ws-p3")

    def test_blocklisted_token_refused_with_real_config(
        self, tmp_path, monkeypatch
    ):
        # Same seam with the yaml-like blocklist actually configured.
        mgr = _make_manager(tmp_path, blocked_commands=["curl", "wget"])
        mgr.create_sandbox("ws-p3b")
        mgr.get_session("ws-p3b").network_enabled = True
        _gate(monkeypatch, True)
        monkeypatch.setattr(
            mgr, "_isolation_backend", _sm.IsolationBackend.BWRAP
        )
        monkeypatch.setattr(mgr, "_bwrap_available", True)
        result = mgr.execute_provision_command("ws-p3b", "curl evil")
        assert result.blocked is True
        assert "validation" in result.block_reason

    def test_unknown_session_valueerror(self, manager, monkeypatch):
        _gate(monkeypatch, True)
        with pytest.raises(ValueError):
            manager.execute_provision_command("ws-none", "echo x")

    def test_happy_seam_runs_network_on_with_provision_timeout(
        self, manager, monkeypatch
    ):
        manager.create_sandbox("ws-p4")
        manager.get_session("ws-p4").network_enabled = True
        _gate(monkeypatch, True)
        monkeypatch.setattr(
            manager, "_isolation_backend", _sm.IsolationBackend.BWRAP
        )
        monkeypatch.setattr(manager, "_bwrap_available", True)
        chain = _chain_recorder(monkeypatch)
        recorded = {}

        def fake_run(command, workspace, timeout, *, session_id="",
                     allow_network=False):
            recorded.update(
                command=command, timeout=timeout,
                allow_network=allow_network,
            )
            return _sm.CommandResult(
                return_code=0, isolation_backend="bwrap"
            )

        monkeypatch.setattr(manager, "_run_bwrap", fake_run)
        cmd = _eg.build_provision_command("requirements.txt")
        result = manager.execute_provision_command("ws-p4", cmd)
        assert result.return_code == 0 and result.blocked is False
        assert recorded["allow_network"] is True
        assert recorded["timeout"] == manager._config.provision_timeout_seconds
        runs = [c for c in chain if c["event_type"] == "sandbox_provision_run"]
        assert len(runs) == 1 and runs[0]["return_code"] == 0
        assert manager.get_session("ws-p4").command_count == 1

    def test_explicit_timeout_overrides(self, manager, monkeypatch):
        manager.create_sandbox("ws-p5")
        manager.get_session("ws-p5").network_enabled = True
        _gate(monkeypatch, True)
        monkeypatch.setattr(
            manager, "_isolation_backend", _sm.IsolationBackend.BWRAP
        )
        monkeypatch.setattr(manager, "_bwrap_available", True)
        recorded = {}
        monkeypatch.setattr(
            manager, "_run_bwrap",
            lambda c, w, t, **kw: recorded.update(timeout=t)
            or _sm.CommandResult(return_code=0),
        )
        manager.execute_provision_command("ws-p5", "echo x", timeout=42)
        assert recorded["timeout"] == 42


class TestProvisionConfig:
    def test_default_and_clamps(self):
        assert SandboxConfig().provision_timeout_seconds == 600
        assert SandboxConfig(provision_timeout_seconds=0) \
            .provision_timeout_seconds == 30
        assert SandboxConfig(provision_timeout_seconds=10 ** 9) \
            .provision_timeout_seconds == 3600

    def test_yaml_carries_the_key(self):
        import yaml

        raw = yaml.safe_load(
            open(os.path.join(_OO, "config", "sandbox.yaml"), encoding="utf-8")
        )
        assert raw.get("provision_timeout_seconds") == 600
        # And the yaml never carries a network-on default.
        assert not any(
            "network" in str(k) for k in raw.keys()
        )


# ---------------------------------------------------------------------------
# 6. Routes: the code ladders via TestClient
# ---------------------------------------------------------------------------

_ROUTES_SANDBOX_CACHE: dict = {}


def _load_routes_sandbox():
    if "mod" in _ROUTES_SANDBOX_CACHE:
        return _ROUTES_SANDBOX_CACHE["mod"]
    schemas = _load_fresh(
        os.path.join("opti_oignon", "api", "schemas.py"),
        register="opti_oignon.api.schemas",
    )
    deps = types.ModuleType("opti_oignon.api.deps")
    deps.SANDBOX_AVAILABLE = True
    deps.sandbox_manager = None
    deps.FILE_TOOLS_AVAILABLE = False
    mod = _load_fresh(
        os.path.join("opti_oignon", "api", "routes_sandbox.py"),
        register="opti_oignon.api.routes_sandbox",
        bind={
            "opti_oignon.api.routes_auth": None,
            "opti_oignon.api.schemas": schemas,
            "opti_oignon.api.deps": deps,
            "opti_oignon.sandbox_manager": _sm,
            "opti_oignon.sandbox_workspace": _ws,
            "opti_oignon.sandbox_tools": _st,
            "opti_oignon.sandbox_egress": _eg,
        },
    )
    _ROUTES_SANDBOX_CACHE["mod"] = mod
    return mod


@pytest.fixture()
def api(manager, monkeypatch):
    rs = _load_routes_sandbox()
    monkeypatch.setattr(rs, "SANDBOX_AVAILABLE", True)
    monkeypatch.setattr(rs, "sandbox_manager", manager)
    monkeypatch.setattr(rs, "EGRESS_AVAILABLE", True)
    monkeypatch.setattr(rs, "_eg", _eg)
    app = fastapi.FastAPI()
    app.include_router(rs.router)
    return rs, TestClient(app), manager


class TestNetworkRoute:
    def test_404_unknown(self, api):
        _, client, _ = api
        resp = client.post("/api/sandbox/ws-none/network", json={"enabled": True})
        assert resp.status_code == 404

    def test_403_foreign_owner(self, api):
        _, client, manager = api
        manager.create_sandbox("ws-n1", owner_user_id="someone-else")
        resp = client.post("/api/sandbox/ws-n1/network", json={"enabled": True})
        assert resp.status_code == 403

    def test_403_bulbe_on_enable(self, api, monkeypatch):
        _, client, manager = api
        manager.create_sandbox("ws-n2")
        _gate(monkeypatch, False)
        resp = client.post("/api/sandbox/ws-n2/network", json={"enabled": True})
        assert resp.status_code == 403
        assert "Bulbe" in resp.json()["detail"]
        assert manager.get_session("ws-n2").network_enabled is False

    def test_200_both_directions_in_daily(self, api, monkeypatch):
        _, client, manager = api
        manager.create_sandbox("ws-n3")
        _gate(monkeypatch, True)
        on = client.post("/api/sandbox/ws-n3/network", json={"enabled": True})
        assert on.status_code == 200 and on.json()["network_enabled"] is True
        off = client.post("/api/sandbox/ws-n3/network", json={"enabled": False})
        assert off.status_code == 200 and off.json()["network_enabled"] is False

    def test_200_disable_under_bulbe(self, api, monkeypatch):
        _, client, manager = api
        manager.create_sandbox("ws-n4")
        manager.get_session("ws-n4").network_enabled = True
        _gate(monkeypatch, False)
        resp = client.post("/api/sandbox/ws-n4/network", json={"enabled": False})
        assert resp.status_code == 200
        assert resp.json()["network_enabled"] is False

    def test_503_when_egress_absent(self, api, monkeypatch):
        rs, client, manager = api
        manager.create_sandbox("ws-n5")
        monkeypatch.setattr(rs, "EGRESS_AVAILABLE", False)
        monkeypatch.setattr(rs, "_eg", None)
        resp = client.post("/api/sandbox/ws-n5/network", json={"enabled": True})
        assert resp.status_code == 503


class TestProvisionRoute:
    def _grant(self, manager, sid, monkeypatch):
        manager.create_sandbox(sid)
        manager.get_session(sid).network_enabled = True
        _gate(monkeypatch, True)

    def _write_req(self, manager, sid, text):
        ws = manager.get_active_workspace_path(sid)
        path = os.path.join(ws, "requirements.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def test_404_unknown(self, api):
        _, client, _ = api
        resp = client.post(
            "/api/sandbox/ws-none/provision",
            json={"requirements_path": "requirements.txt"},
        )
        assert resp.status_code == 404

    def test_403_foreign_owner(self, api):
        _, client, manager = api
        manager.create_sandbox("ws-q1", owner_user_id="someone-else")
        resp = client.post(
            "/api/sandbox/ws-q1/provision",
            json={"requirements_path": "requirements.txt"},
        )
        assert resp.status_code == 403

    def test_503_partial_build(self, api, monkeypatch):
        rs, client, manager = api
        manager.create_sandbox("ws-q2")
        monkeypatch.setattr(rs, "EGRESS_AVAILABLE", False)
        monkeypatch.setattr(rs, "_eg", None)
        resp = client.post(
            "/api/sandbox/ws-q2/provision",
            json={"requirements_path": "requirements.txt"},
        )
        assert resp.status_code == 503

    def test_403_bulbe_at_route_gate(self, api, monkeypatch):
        _, client, manager = api
        manager.create_sandbox("ws-q3")
        manager.get_session("ws-q3").network_enabled = True
        fake = types.SimpleNamespace(
            network_allowed=lambda: False,
            refuse_rel_path=_eg.refuse_rel_path,
            validate_requirements_text=_eg.validate_requirements_text,
            build_provision_command=_eg.build_provision_command,
        )
        rs = api[0]
        monkeypatch.setattr(rs, "_eg", fake)
        resp = client.post(
            "/api/sandbox/ws-q3/provision",
            json={"requirements_path": "requirements.txt"},
        )
        assert resp.status_code == 403
        assert "Bulbe" in resp.json()["detail"]
        assert "provision_refused" in _approval_actions(manager, "ws-q3")

    def test_403_route_gate_unreadable_fail_secure(self, api, monkeypatch):
        rs, client, manager = api
        manager.create_sandbox("ws-q4")
        manager.get_session("ws-q4").network_enabled = True

        def boom():
            raise RuntimeError("gate broke")

        fake = types.SimpleNamespace(network_allowed=boom)
        monkeypatch.setattr(rs, "_eg", fake)
        resp = client.post(
            "/api/sandbox/ws-q4/provision",
            json={"requirements_path": "requirements.txt"},
        )
        assert resp.status_code == 403

    def test_409_when_flag_off(self, api, monkeypatch):
        _, client, manager = api
        manager.create_sandbox("ws-q5")
        _gate(monkeypatch, True)
        resp = client.post(
            "/api/sandbox/ws-q5/provision",
            json={"requirements_path": "requirements.txt"},
        )
        assert resp.status_code == 409
        assert "provision_refused" in _approval_actions(manager, "ws-q5")

    def test_400_bad_path(self, api, monkeypatch):
        _, client, manager = api
        self._grant(manager, "ws-q6", monkeypatch)
        resp = client.post(
            "/api/sandbox/ws-q6/provision",
            json={"requirements_path": "../escape.txt"},
        )
        assert resp.status_code == 400

    def test_400_missing_file(self, api, monkeypatch):
        _, client, manager = api
        self._grant(manager, "ws-q7", monkeypatch)
        resp = client.post(
            "/api/sandbox/ws-q7/provision",
            json={"requirements_path": "requirements.txt"},
        )
        assert resp.status_code == 400

    def test_400_non_pinned_set_with_per_line_refusals(self, api, monkeypatch):
        _, client, manager = api
        self._grant(manager, "ws-q8", monkeypatch)
        self._write_req(manager, "ws-q8", f"{GOOD_LINE}\nflask\n")
        resp = client.post(
            "/api/sandbox/ws-q8/provision",
            json={"requirements_path": "requirements.txt"},
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "Nothing was installed" in detail["message"]
        assert detail["refused"][0]["line"] == 2
        # nothing-on-partial: the workspace gained no venv
        ws = manager.get_active_workspace_path("ws-q8")
        assert not os.path.exists(os.path.join(ws, ".venv"))

    def test_blocked_200_in_container(self, api, monkeypatch):
        # The container posture end to end: gate green, flag on, set
        # pinned -- the executor refuses on bwrap absence and the route
        # surfaces it honestly in the 200 body, audited.
        _, client, manager = api
        self._grant(manager, "ws-q9", monkeypatch)
        self._write_req(manager, "ws-q9", GOOD_LINE + "\n")
        resp = client.post(
            "/api/sandbox/ws-q9/provision",
            json={"requirements_path": "requirements.txt"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["blocked"] is True and "bwrap" in body["block_reason"]
        assert body["accepted_requirements"] == ["requests"]
        assert "--require-hashes" in body["command"]
        actions = _approval_actions(manager, "ws-q9")
        assert "provision_run" in actions
        assert "provision_refused" in actions

    def test_200_run_result_shape(self, api, monkeypatch):
        _, client, manager = api
        self._grant(manager, "ws-qa", monkeypatch)
        self._write_req(manager, "ws-qa", GOOD_LINE + "\n")
        monkeypatch.setattr(
            manager, "execute_provision_command",
            lambda sid, cmd, timeout=None: _sm.CommandResult(
                stdout="ok-tail", return_code=0, isolation_backend="bwrap"
            ),
        )
        resp = client.post(
            "/api/sandbox/ws-qa/provision",
            json={"requirements_path": "requirements.txt"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["return_code"] == 0 and body["blocked"] is False
        assert body["stdout_tail"] == "ok-tail"
        rows = manager.audit.get_approval_log("ws-qa")
        run_rows = [r for r in rows if r["action"] == "provision_run"]
        assert len(run_rows) == 1 and "rc=0" in run_rows[0]["detail"]


class TestStatusAndSessionsSurface:
    def test_status_carries_gate_and_caps(self, api, monkeypatch):
        rs, client, manager = api
        fake = types.SimpleNamespace(network_allowed=lambda: True)
        monkeypatch.setattr(rs, "_eg", fake)
        resp = client.get("/api/sandbox/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["network_allowed"] is True
        assert body["command_timeout_default"] == manager.config.command_timeout
        assert body["limit_memory_bytes"] == manager.config.limit_memory_bytes
        assert body["limit_nproc"] == manager.config.limit_nproc
        assert body["limit_cpu_seconds"] == manager.config.limit_cpu_seconds
        assert (
            body["disk_soft_limit_bytes"]
            == manager.config.disk_soft_limit_bytes
        )

    def test_status_gate_fail_secure(self, api, monkeypatch):
        rs, client, _ = api

        def boom():
            raise RuntimeError("gate broke")

        monkeypatch.setattr(rs, "_eg", types.SimpleNamespace(network_allowed=boom))
        resp = client.get("/api/sandbox/status")
        assert resp.json()["network_allowed"] is False
        monkeypatch.setattr(rs, "EGRESS_AVAILABLE", False)
        monkeypatch.setattr(rs, "_eg", None)
        assert client.get("/api/sandbox/status").json()["network_allowed"] is False

    def test_sessions_carry_has_cloned_baseline(self, api):
        _, client, manager = api
        manager.create_sandbox("ws-s1")
        manager.create_sandbox("ws-s2")
        _ws.get_workspace_manifests().record(
            "ws-s2", {"a.txt": "0" * 64}, cloned_root="/share/src",
            cloned_mount="src",
        )
        rows = {r["session_id"]: r for r in client.get("/api/sandbox/sessions").json()}
        assert rows["ws-s1"]["has_cloned_baseline"] is False
        assert rows["ws-s2"]["has_cloned_baseline"] is True


class TestDestroyLastContract:
    def test_new_routes_registered_before_final_delete(self, api):
        rs, _, _ = api
        paths = [
            (r.path, sorted(r.methods)) for r in rs.router.routes
        ]
        flat = [p for p, _ in paths]
        assert "/api/sandbox/{session_id}/network" in flat
        assert "/api/sandbox/{session_id}/provision" in flat
        # destroy is still the LAST registered route
        last_path, last_methods = paths[-1]
        assert last_path == "/api/sandbox/{session_id}"
        assert last_methods == ["DELETE"]


# ---------------------------------------------------------------------------
# 7. Registrations by source
# ---------------------------------------------------------------------------


def _read(relpath):
    return open(os.path.join(_ROOT, relpath), encoding="utf-8").read()


class TestRegistrations:
    def test_egress_module_conventions(self):
        src = _read(os.path.join("opti_oignon", "sandbox_egress.py"))
        assert "checkpoint_before_apply = True" in src
        assert _eg.checkpoint_before_apply is True
        # raw --share-net stays out of scope: the module never builds it
        assert "--share-net" not in src.replace(
            "Raw ``--share-net``", ""
        ).replace("``--share-net``", "")

    def test_proxy_mode_prepared_not_wired(self):
        # Detection helper exists; no route or manager call reaches it.
        assert callable(_eg.proxy_mode_available)
        assert isinstance(_eg.proxy_mode_available(), bool)
        routes_src = _read(os.path.join("opti_oignon", "api", "routes_sandbox.py"))
        manager_src = _read(os.path.join("opti_oignon", "sandbox_manager.py"))
        assert "proxy_mode_available" not in routes_src
        assert "proxy_mode_available" not in manager_src

    def test_spec_status_section(self):
        spec = _read("SANDBOX_WORKSPACE_SPEC.md")
        assert "### 8.5 Status (S213)" in spec
        assert "PROVISION PHASE" in spec
        assert "byte-identical" in spec

    def test_spec_cartography_and_tests_row(self):
        spec = _read("SANDBOX_WORKSPACE_SPEC.md")
        assert "LANDED S213" in spec
        assert "tests/test_s213_sandbox_bloc4.py" in spec
        assert "EXTENDED S213" in spec

    def test_frontend_redesign_rows(self):
        frd = _read("FRONTEND_REDESIGN_SPEC.md")
        assert "`SandboxSettingsStrip.svelte` | NEW | S213" in frd
        # the preserved rows note-extended, the pinned patterns intact
        assert "`SandboxPanel.svelte` | NEW | S210" in frd
        assert "S213 (Bloc 4): gains the Workspace-settings card" in frd
        assert "`SandboxWorkspaceList.svelte` | NEW | S210" in frd
        assert "live since S213" in frd

    def test_component_exists_balanced_and_token_only(self):
        src = _read(os.path.join(
            "frontend", "src", "lib", "components", "panels",
            "SandboxSettingsStrip.svelte",
        ))
        body = re.sub(r"<!--.*?-->", "", src, flags=re.S)
        body = re.sub(r"(<script[^>]*>).*?(</script>)", r"\1\2", body, flags=re.S)
        markup = re.sub(r"(<style[^>]*>).*?(</style>)", r"\1\2", body, flags=re.S)
        from collections import Counter

        opens = Counter(re.findall(r"<([A-Za-z][A-Za-z0-9.-]*)\b[^>]*?(?<!/)>", markup))
        closes = Counter(re.findall(r"</([A-Za-z][A-Za-z0-9.-]*)>", markup))
        for tag in set(opens) | set(closes):
            if tag.lower() in ("input", "br", "hr", "img"):
                continue
            assert opens[tag] == closes[tag], f"unbalanced <{tag}>"
        for kind in ("if", "each"):
            assert len(re.findall(r"\{#" + kind + r"\b", markup)) == len(
                re.findall(r"\{/" + kind + r"\}", markup)
            )
        # token hygiene: every hex sits inside var(--oo-*, #fallback)
        for m in re.finditer(r"#[0-9a-fA-F]{3,8}\b", src):
            ctx = src[max(0, m.start() - 60):m.start()]
            assert re.search(r"var\(\s*--oo-[a-z0-9-]+\s*,\s*$", ctx), (
                f"raw hex at offset {m.start()}"
            )
        # the honest surfaces
        assert "Bulbe" in src
        assert "approval gate" in src
        assert "Daily-only" in src

    def test_panel_hosts_the_strip(self):
        src = _read(os.path.join(
            "frontend", "src", "lib", "components", "panels",
            "SandboxPanel.svelte",
        ))
        assert "import SandboxSettingsStrip from './SandboxSettingsStrip.svelte'" in src
        assert "<SandboxSettingsStrip" in src
        assert "handleNetworkChanged" in src
        assert "Network stays off this cycle" not in src

    def test_client_and_types(self):
        ts = _read(os.path.join("frontend", "src", "lib", "api", "sandbox.ts"))
        assert "export async function setNetwork" in ts
        assert "export async function provisionWorkspace" in ts
        assert "/network" in ts and "/provision" in ts
        types_src = _read(os.path.join("frontend", "src", "lib", "types.ts"))
        for name in (
            "SandboxNetworkToggleRequest", "SandboxNetworkToggleResponse",
            "SandboxProvisionRequest", "SandboxProvisionResponse",
            "has_cloned_baseline", "network_allowed",
        ):
            assert name in types_src

    def test_schemas_registered(self):
        schemas = _load_routes_sandbox()  # ensures the schema copy loaded
        assert schemas is not None
        src = _read(os.path.join("opti_oignon", "api", "schemas.py"))
        for name in (
            "SandboxNetworkToggleRequest", "SandboxNetworkToggleResponse",
            "SandboxProvisionRequest", "SandboxProvisionResponse",
            "SandboxProvisionRefusedLine",
        ):
            assert f"class {name}" in src
        assert "network_allowed" in src
        assert "has_cloned_baseline" in src

    def test_odysseus_spec_untouched_by_s213(self):
        spec = _read("ODYSSEUS_SPEC.md")
        assert "S213" not in spec
