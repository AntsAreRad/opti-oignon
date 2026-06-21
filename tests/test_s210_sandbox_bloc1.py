#!/usr/bin/env python3
"""S210 (Sandbox Workspace, Bloc 1): lifecycle manager + conversation binding.

Container-deliverable assertions only; nothing here executes bwrap. Covered:
the SandboxSession lifecycle fields and the derived list view (age, running,
bounded approximate disk use); the stop path with a FAKED tracked subprocess
(the per-session running-process registry, the group SIGKILL, the honest
no-op); the deselect-plus-reassert block for the six S209 ``_run_bwrap``
wiring assertions superseded by the ``subprocess.run`` -> tracked ``Popen``
switch (same argv/kw assertions re-pinned against ``Popen``, plus the
fail-secure tripwire moved onto ``Popen``); the destroy/stop/bind routes'
status codes and the router auth parity by source; the conversation-binding
invariants and the dispatch injection proving ATL-02; the reconcile / lazy
idle TTL (bound exempt) / disk soft quota; and the spec / cartography /
FRONTEND_REDESIGN registrations with the FRD-03 mount by source. The rendered
UI walk is host territory (the shakedown owns it).
"""

import os
import re
import sys
import threading
import time
import types

# Guarded stub: in CI ollama is installed and this is a no-op; locally it lets
# the isolated module load resolve the opti_oignon import chain.
sys.modules.setdefault("ollama", types.ModuleType("ollama"))

import importlib.util
import subprocess as _subprocess_mod

import pytest

_ROOT = os.path.join(os.path.dirname(__file__), os.pardir)
_OO = os.path.join(_ROOT, "opti_oignon")
_API = os.path.join(_OO, "api")
_AGENT = os.path.join(_OO, "agent")


def _ensure_pkg(name: str, path: str) -> None:
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = [path]
        sys.modules[name] = mod


_ensure_pkg("opti_oignon", _OO)
_ensure_pkg("opti_oignon.api", _API)
_ensure_pkg("opti_oignon.agent", _AGENT)


def _load(name: str, relpath: str, register: str | None = None):
    """Load a module, REUSING an existing sys.modules entry when present.

    Reuse is only safe for modules whose cross-module identity this file
    does not assert (the agent stack); the sandbox family below goes
    through _load_fresh instead.
    """
    if register is not None and register in sys.modules:
        return sys.modules[register]
    path = os.path.join(_ROOT, relpath)
    spec = importlib.util.spec_from_file_location(register or name, path)
    mod = importlib.util.module_from_spec(spec)
    if register is not None:
        sys.modules[register] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_fresh(relpath: str, register: str, bind: dict | None = None):
    """ALWAYS exec this file's own copy; never reuse a pre-loaded module.

    Other suites in the sweep (notably test_file_tools, which imports the
    whole real opti_oignon.api.app chain) pre-load the canonical module
    names. Reusing those would split exception-class identity between this
    file's manager copy and the routes' except clauses. So: temporarily
    register `bind` plus the module's own name, exec the fresh copy (its
    guarded absolute imports then resolve against OUR copies), and restore
    every touched sys.modules entry afterwards. The returned module object
    stays fully usable; per-test resolution is re-pinned by the
    _bind_module_copies fixture.
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
_ssec = _load_fresh(
    os.path.join("opti_oignon", "sandbox_seccomp.py"),
    register="opti_oignon.sandbox_seccomp",
)

SandboxConfig = _sm.SandboxConfig


@pytest.fixture(autouse=True)
def _bind_module_copies(monkeypatch):
    """Bind THIS file's module copies for the duration of each test.

    Under the full sweep, an earlier test module may replace
    sys.modules["opti_oignon"] with a non-package stub (the friction class
    documented at S209), which breaks the source's lazy
    ``from opti_oignon import sandbox_seccomp`` inside ``_run_bwrap`` and can
    split exception-class identity (e.g. ``WorkspaceQuotaExceeded`` raised
    from one copy, caught against another in the routes). Binding our loaded
    copies per test makes both resolutions land on the same objects this
    file asserts against; monkeypatch restores everything afterwards.
    """
    pairs = {
        "opti_oignon.sandbox_manager": _sm,
        "opti_oignon.sandbox_tools": _st,
        "opti_oignon.sandbox_workspace": _ws,
        "opti_oignon.sandbox_seccomp": _ssec,
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
def _fresh_bindings():
    _ws.reset_workspace_bindings()
    yield
    _ws.reset_workspace_bindings()


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


# ---------------------------------------------------------------------------
# Lifecycle fields + the derived list view
# ---------------------------------------------------------------------------

class TestLifecycleFields:
    def test_session_field_defaults(self):
        s = _sm.SandboxSession(session_id="x", workspace_path="/tmp/x")
        assert s.label == ""
        assert s.owner_user_id == "local"
        assert s.bound_conversation_id is None
        assert s.network_enabled is False
        assert s.timeout_override is None
        assert s.last_activity > 0

    def test_create_sets_lifecycle_fields(self, manager):
        s = manager.create_sandbox(
            "ws-a", label="demo", owner_user_id="alice", timeout_override=7
        )
        assert s.label == "demo"
        assert s.owner_user_id == "alice"
        assert s.timeout_override == 7
        assert s.last_activity >= s.created_at

    def test_create_none_autogenerates_id(self, manager):
        s = manager.create_sandbox(None)
        assert s.session_id.startswith("ws-")
        assert s.session_id in {x["session_id"] for x in manager.list_sessions()}
        assert "None" not in os.path.basename(s.workspace_path)

    def test_list_view_has_manager_fields(self, manager):
        manager.create_sandbox("ws-v", label="lbl")
        view = manager.list_sessions()[0]
        for key in (
            "label",
            "owner_user_id",
            "bound_conversation_id",
            "network_enabled",
            "last_activity",
            "timeout_override",
            "age_seconds",
            "running",
            "disk_use_bytes",
        ):
            assert key in view
        assert view["network_enabled"] is False  # Bloc 4 flips it, not here
        assert view["running"] is False
        assert view["age_seconds"] >= 0

    def test_disk_use_counts_files_not_symlink_targets(self, manager, tmp_path):
        s = manager.create_sandbox("ws-d")
        for i in range(3):
            with open(os.path.join(s.workspace_path, f"f{i}.bin"), "wb") as fh:
                fh.write(b"x" * 1000)
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"y" * 50000)
        os.symlink(str(outside), os.path.join(s.workspace_path, "lnk"))
        use = _sm.SandboxManager._workspace_disk_use(s.workspace_path)
        assert 3000 <= use < 50000

    def test_disk_walk_is_bounded_on_entries(self, monkeypatch, tmp_path):
        monkeypatch.setattr(_sm, "_DISK_WALK_MAX_ENTRIES", 5)
        base = tmp_path / "many"
        base.mkdir()
        for i in range(20):
            (base / f"f{i}").write_bytes(b"z" * 10)
        use = _sm.SandboxManager._workspace_disk_use(str(base))
        assert use <= 5 * 10

    def test_timeout_resolution_order(self, manager, monkeypatch):
        captured = {}

        def _spawn(self_mgr, argv, *, timeout, **kw):
            captured["timeout"] = timeout
            return _subprocess_mod.CompletedProcess(argv, 0, b"ok", b"")

        monkeypatch.setattr(_sm.SandboxManager, "_spawn_tracked", _spawn)
        manager.create_sandbox("ws-t", timeout_override=7)
        manager.execute_command("ws-t", "echo hi")
        assert captured["timeout"] == 7  # session override beats config
        manager.execute_command("ws-t", "echo hi", timeout=3)
        assert captured["timeout"] == 3  # explicit call beats override

    def test_execute_updates_last_activity(self, manager, monkeypatch):
        monkeypatch.setattr(
            _sm.SandboxManager,
            "_spawn_tracked",
            lambda self, argv, *, timeout, **kw: _subprocess_mod.CompletedProcess(
                argv, 0, b"", b""
            ),
        )
        s = manager.create_sandbox("ws-act")
        before = s.last_activity
        time.sleep(0.01)
        manager.execute_command("ws-act", "echo hi")
        assert manager.get_session("ws-act").last_activity > before


# ---------------------------------------------------------------------------
# The stop path (faked / tracked subprocess; no bwrap executed)
# ---------------------------------------------------------------------------

class _FakePopen:
    """A Popen stand-in capturing argv/kw; communicate returns immediately."""

    _next_pid = 50000

    def __init__(self, argv, **kw):
        self.argv = argv
        self.kw = kw
        _FakePopen._next_pid += 1
        self.pid = _FakePopen._next_pid
        self.returncode = 0
        self.killed = False

    def communicate(self, timeout=None):
        return (b"ok", b"")

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.killed = True


class TestStopPath:
    def test_registry_filled_during_spawn_and_cleared_after(
        self, manager, monkeypatch
    ):
        seen = {}

        class _Probe(_FakePopen):
            def communicate(inner, timeout=None):
                seen["during"] = manager.is_running("ws-r")
                return (b"ok", b"")

        monkeypatch.setattr(_sm.subprocess, "Popen", _Probe)
        manager.create_sandbox("ws-r")
        res = manager.execute_command("ws-r", "echo hi")
        assert res.return_code == 0
        assert seen["during"] is True
        assert manager.is_running("ws-r") is False

    def test_stop_kills_the_process_group_and_reaps(self, manager, monkeypatch):
        manager.create_sandbox("ws-k")
        proc = _FakePopen(["bash", "-c", "sleep 999"])
        with manager._lock:
            manager._running_procs["ws-k"] = proc
        killed = {}
        monkeypatch.setattr(
            _sm.os, "killpg", lambda pgid, sig: killed.update(pgid=pgid, sig=sig)
        )
        assert manager.stop_command("ws-k") is True
        assert killed["pgid"] == proc.pid  # start_new_session: pid == pgid
        assert killed["sig"] == _sm.signal.SIGKILL
        assert manager.is_running("ws-k") is False
        # The workspace persists: the stop path never destroys it.
        assert manager.get_session("ws-k").active is True
        rows = manager.audit.get_approval_log("ws-k")
        assert any(r.get("action") == "workspace_stopped" for r in rows)

    def test_stop_idle_is_a_noop_not_an_error(self, manager):
        manager.create_sandbox("ws-i")
        assert manager.stop_command("ws-i") is False

    def test_stop_unknown_raises_for_an_honest_404(self, manager):
        with pytest.raises(ValueError):
            manager.stop_command("nope")

    def test_timeout_kills_the_whole_group(self, manager, monkeypatch):
        calls = {"n": 0}

        class _Hang(_FakePopen):
            def communicate(inner, timeout=None):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise _subprocess_mod.TimeoutExpired(inner.argv, timeout)
                return (b"", b"")

        killed = {}
        monkeypatch.setattr(_sm.subprocess, "Popen", _Hang)
        monkeypatch.setattr(
            _sm.os, "killpg", lambda pgid, sig: killed.update(pgid=pgid, sig=sig)
        )
        manager.create_sandbox("ws-to")
        res = manager.execute_command("ws-to", "sleep 999", timeout=1)
        assert res.timed_out is True
        assert killed["sig"] == _sm.signal.SIGKILL
        assert manager.is_running("ws-to") is False


# ---------------------------------------------------------------------------
# Deselect-plus-reassert: the six S209 _run_bwrap wiring assertions, re-pinned
# against the tracked Popen (the spawn the stop path requires). The originals
# in tests/test_s209_sandbox_bloc0.py are deselected in pyproject.toml, never
# edited or deleted. Same argv/kw assertions, plus start_new_session and the
# registry, plus the fail-secure tripwire moved onto Popen.
# ---------------------------------------------------------------------------

class TestRunBwrapWiringPopen:
    def _fake_popen(self, captured):
        def _factory(argv, **kw):
            captured["argv"] = argv
            captured["kw"] = kw
            return _FakePopen(argv, **kw)

        return _factory

    def test_rlimit_backend_wires_preexec_and_passes_seccomp_fd(
        self, manager, monkeypatch
    ):
        captured = {}
        monkeypatch.setattr(_sm.subprocess, "Popen", self._fake_popen(captured))
        manager._config.limits_enabled = True
        manager._config.resource_backend = "rlimit"
        manager._config.seccomp_enabled = True
        res = manager._run_bwrap("echo hi", str(manager._config.workspace_base), 5)
        assert res.blocked is False
        assert callable(captured["kw"]["preexec_fn"])
        assert captured["kw"]["start_new_session"] is True
        pass_fds = captured["kw"]["pass_fds"]
        assert len(pass_fds) == 1 and isinstance(pass_fds[0], int)
        argv = captured["argv"]
        assert "--seccomp" in argv
        assert argv[argv.index("--seccomp") + 1] == str(pass_fds[0])

    def test_limits_disabled_means_no_preexec(self, manager, monkeypatch):
        captured = {}
        monkeypatch.setattr(_sm.subprocess, "Popen", self._fake_popen(captured))
        manager._config.limits_enabled = False
        manager._run_bwrap("echo hi", str(manager._config.workspace_base), 5)
        assert captured["kw"]["preexec_fn"] is None

    def test_seccomp_disabled_means_no_flag(self, manager, monkeypatch):
        captured = {}
        monkeypatch.setattr(_sm.subprocess, "Popen", self._fake_popen(captured))
        manager._config.seccomp_enabled = False
        manager._run_bwrap("echo hi", str(manager._config.workspace_base), 5)
        assert "--seccomp" not in captured["argv"]
        assert captured["kw"]["pass_fds"] == ()

    def test_cgroup_backend_prefixes_and_drops_preexec(
        self, manager, monkeypatch
    ):
        captured = {}
        monkeypatch.setattr(_sm.subprocess, "Popen", self._fake_popen(captured))
        monkeypatch.setattr(_sm, "_detect_systemd_run", lambda: True)
        manager._config.limits_enabled = True
        manager._config.resource_backend = "cgroup"
        manager._run_bwrap("echo hi", str(manager._config.workspace_base), 5)
        assert captured["argv"][0] == "systemd-run"
        assert captured["kw"]["preexec_fn"] is None

    def test_cgroup_unavailable_falls_back_to_rlimit(
        self, manager, monkeypatch
    ):
        captured = {}
        monkeypatch.setattr(_sm.subprocess, "Popen", self._fake_popen(captured))
        monkeypatch.setattr(_sm, "_detect_systemd_run", lambda: False)
        manager._config.limits_enabled = True
        manager._config.resource_backend = "cgroup"
        manager._run_bwrap("echo hi", str(manager._config.workspace_base), 5)
        # Never disabled: it falls back to the rlimit preexec.
        assert captured["argv"][0] != "systemd-run"
        assert callable(captured["kw"]["preexec_fn"])

    def test_required_false_launches_unfiltered_with_warning(
        self, manager, monkeypatch, caplog
    ):
        def _raise(*a, **k):
            raise _ssec.SeccompUnavailable("forced for test")

        monkeypatch.setattr(_ssec, "build_filter_program", _raise)
        captured = {}
        monkeypatch.setattr(_sm.subprocess, "Popen", self._fake_popen(captured))
        manager._config.seccomp_enabled = True
        manager._config.seccomp_required = False
        with caplog.at_level("WARNING"):
            res = manager._run_bwrap(
                "echo hi", str(manager._config.workspace_base), 5
            )
        assert res.blocked is False
        assert "--seccomp" not in captured["argv"]
        assert captured["kw"]["pass_fds"] == ()
        assert any("seccomp" in r.message.lower() for r in caplog.records)

    def test_required_true_refuses_before_any_popen(self, manager, monkeypatch):
        # The fail-secure tripwire, moved onto the spawn that now exists.
        def _raise(*a, **k):
            raise _ssec.SeccompUnavailable("forced for test")

        monkeypatch.setattr(_ssec, "build_filter_program", _raise)

        def _boom(*a, **k):
            raise AssertionError("subprocess.Popen reached on fail-secure path")

        monkeypatch.setattr(_sm.subprocess, "Popen", _boom)
        manager._config.seccomp_enabled = True
        manager._config.seccomp_required = True
        res = manager._run_bwrap(
            "echo hi", str(manager._config.workspace_base), 5
        )
        assert res.blocked is True
        assert res.return_code == -1
        assert "fail-secure" in res.block_reason


# ---------------------------------------------------------------------------
# Conversation binding invariants (sandbox_workspace)
# ---------------------------------------------------------------------------

class TestWorkspaceBindings:
    def test_bind_and_write_through_mirror(self, manager):
        manager.create_sandbox("ws-a")
        b = _ws.WorkspaceBindings()
        b.bind("conv-1", "ws-a", user_id="local", manager=manager)
        assert b.get_sandbox_for("conv-1", manager=manager) == "ws-a"
        assert manager.get_session("ws-a").bound_conversation_id == "conv-1"
        assert b.get_conversation_for("ws-a") == "conv-1"

    def test_one_active_conversation_per_workspace(self, manager):
        manager.create_sandbox("ws-a")
        b = _ws.WorkspaceBindings()
        b.bind("conv-1", "ws-a", user_id="local", manager=manager)
        with pytest.raises(_ws.WorkspaceAlreadyBound):
            b.bind("conv-2", "ws-a", user_id="local", manager=manager)

    def test_owner_mismatch_refused(self, manager):
        manager.create_sandbox("ws-o", owner_user_id="other")
        b = _ws.WorkspaceBindings()
        with pytest.raises(_ws.WorkspaceOwnerMismatch):
            b.bind("conv-1", "ws-o", user_id="local", manager=manager)

    def test_unknown_workspace_refused(self, manager):
        b = _ws.WorkspaceBindings()
        with pytest.raises(_ws.WorkspaceNotFound):
            b.bind("conv-1", "nope", user_id="local", manager=manager)

    def test_rebind_atomically_releases_the_old(self, manager):
        manager.create_sandbox("ws-a")
        manager.create_sandbox("ws-b")
        b = _ws.WorkspaceBindings()
        b.bind("conv-1", "ws-a", user_id="local", manager=manager)
        b.bind("conv-1", "ws-b", user_id="local", manager=manager)
        assert b.get_sandbox_for("conv-1", manager=manager) == "ws-b"
        assert manager.get_session("ws-a").bound_conversation_id is None
        assert b.get_conversation_for("ws-a") is None
        # The released workspace is free for another conversation.
        b.bind("conv-2", "ws-a", user_id="local", manager=manager)

    def test_same_pair_rebind_is_idempotent(self, manager):
        manager.create_sandbox("ws-a")
        b = _ws.WorkspaceBindings()
        b.bind("conv-1", "ws-a", user_id="local", manager=manager)
        b.bind("conv-1", "ws-a", user_id="local", manager=manager)
        assert b.snapshot() == {"conv-1": "ws-a"}

    def test_resolution_self_heals_on_destroyed_session(self, manager):
        manager.create_sandbox("ws-a")
        b = _ws.WorkspaceBindings()
        b.bind("conv-1", "ws-a", user_id="local", manager=manager)
        manager.destroy_sandbox("ws-a")
        assert b.get_sandbox_for("conv-1", manager=manager) is None
        assert b.snapshot() == {}

    def test_unbind_is_a_noop_when_unbound(self, manager):
        b = _ws.WorkspaceBindings()
        assert b.unbind("conv-x", user_id="local", manager=manager) is False

    def test_bind_and_unbind_are_audited(self, manager):
        manager.create_sandbox("ws-a")
        b = _ws.WorkspaceBindings()
        b.bind("conv-1", "ws-a", user_id="local", manager=manager)
        b.unbind("conv-1", user_id="local", manager=manager)
        actions = {
            r.get("action") for r in manager.audit.get_approval_log("ws-a")
        }
        assert "conversation_bound" in actions
        assert "conversation_unbound" in actions

    def test_singleton_reset(self):
        first = _ws.get_workspace_bindings()
        assert _ws.get_workspace_bindings() is first
        _ws.reset_workspace_bindings()
        assert _ws.get_workspace_bindings() is not first

    def test_module_carries_the_sentinel(self):
        assert _ws.checkpoint_before_apply is True


# ---------------------------------------------------------------------------
# attach / detach (sandbox_tools) and the registry lockout invariant
# ---------------------------------------------------------------------------

class _FakeRegistry:
    def __init__(self):
        self.calls = []
        self._disabled_by_sandbox = set()

    def set_sandbox_mode(self, on: bool):
        self.calls.append(on)
        return ["execute_code"] if on else ["execute_code"]

    def get(self, name):
        return None


class TestAttachDetach:
    def test_attach_binds_existing_without_creating(self, manager):
        manager.create_sandbox("ws-a")
        reg = _FakeRegistry()
        sess = _st.SandboxToolSession(manager, reg)
        assert sess.attach("ws-a") == "ws-a"
        assert sess.active is True
        assert reg.calls == [True]  # the lockout invariant, unchanged

    def test_detach_releases_without_destroying(self, manager):
        manager.create_sandbox("ws-a")
        reg = _FakeRegistry()
        sess = _st.SandboxToolSession(manager, reg)
        sess.attach("ws-a")
        assert sess.detach() is True
        assert sess.active is False
        assert manager.get_session("ws-a").active is True  # never destroyed
        assert reg.calls == [True, False]

    def test_attach_unknown_or_inactive_refused(self, manager):
        sess = _st.SandboxToolSession(manager, _FakeRegistry())
        with pytest.raises(ValueError):
            sess.attach("nope")

    def test_attach_while_active_refused(self, manager):
        manager.create_sandbox("ws-a")
        manager.create_sandbox("ws-b")
        sess = _st.SandboxToolSession(manager, _FakeRegistry())
        sess.attach("ws-a")
        with pytest.raises(RuntimeError):
            sess.attach("ws-b")

    def test_start_and_stop_behaviour_unchanged(self, manager):
        reg = _FakeRegistry()
        sess = _st.SandboxToolSession(manager, reg)
        sid = sess.start("ws-s")
        assert reg.calls == [True]
        assert sess.stop() is True
        assert reg.calls == [True, False]
        assert manager.get_session(sid) is None  # stop still destroys

    def test_attach_session_for_conversation_seam(self, manager):
        manager.create_sandbox("ws-a")
        _ws.get_workspace_bindings().bind(
            "conv-9", "ws-a", user_id="local", manager=manager
        )
        got = _ws.attach_session_for_conversation("conv-9", manager=manager)
        assert got is not None and got.session_id == "ws-a"
        got.detach()
        assert (
            _ws.attach_session_for_conversation("conv-none", manager=manager)
            is None
        )


# ---------------------------------------------------------------------------
# ATL-02: the bound workspace's session reaches the agent run and dispatch
# ---------------------------------------------------------------------------

def _load_agent_stack():
    for mod_name in (
        "tool_parsing",
        "allowlists",
        "dispatch",
        "untrusted_context",
        "loop",
        "tools",
        "teacher",
    ):
        _load(
            mod_name,
            os.path.join("opti_oignon", "agent", f"{mod_name}.py"),
            register=f"opti_oignon.agent.{mod_name}",
        )
    _load(
        "skills",
        os.path.join("opti_oignon", "agent", "skills.py"),
        register="opti_oignon.agent.skills",
    )
    return _load(
        "routes_agent_s210",
        os.path.join("opti_oignon", "api", "routes_agent.py"),
        register="opti_oignon.api.routes_agent",
    )


class TestATL02Injection:
    def test_bound_conversation_injects_attached_session(
        self, manager, monkeypatch
    ):
        ra = _load_agent_stack()
        manager.create_sandbox("ws-run")
        _ws.get_workspace_bindings().bind(
            "conv-run", "ws-run", user_id="local", manager=manager
        )
        # The default manager inside the seam must be THIS test manager.
        monkeypatch.setattr(
            _ws.WorkspaceBindings,
            "_resolve_manager",
            staticmethod(lambda m: m if m is not None else manager),
        )
        captured = {}
        done = threading.Event()

        def _fake_run(**kwargs):
            sandbox = kwargs.get("sandbox")
            captured["sandbox"] = sandbox
            # session_id and active must be read DURING the run: the finally
            # detaches the session afterwards (that release is itself part of
            # the contract asserted below).
            captured["sid_during_run"] = getattr(sandbox, "session_id", None)
            captured["active_during_run"] = bool(
                getattr(sandbox, "active", False)
            )
            done.set()
            return types.SimpleNamespace(stop_reason="done", rounds=1)

        monkeypatch.setattr(ra.agent_loop, "run", _fake_run)
        mgr_run = ra.AgentRunManager()
        result = mgr_run.start(
            "do something",
            model_client=object(),
            mode="daily",
            conversation_id="conv-run",
        )
        assert result.get("started") is True
        assert done.wait(timeout=5)
        mgr_run.join(timeout=5)
        injected = captured["sandbox"]
        assert injected is not None
        assert captured["sid_during_run"] == "ws-run"
        assert captured["active_during_run"] is True
        # After the run: detached (released), workspace NEVER destroyed.
        assert injected.active is False
        assert manager.get_session("ws-run").active is True

    def test_unbound_conversation_injects_nothing(self, manager, monkeypatch):
        ra = _load_agent_stack()
        captured = {}
        done = threading.Event()

        def _fake_run(**kwargs):
            captured["sandbox"] = kwargs.get("sandbox")
            done.set()
            return types.SimpleNamespace(stop_reason="done", rounds=1)

        monkeypatch.setattr(ra.agent_loop, "run", _fake_run)
        mgr_run = ra.AgentRunManager()
        mgr_run.start(
            "do something",
            model_client=object(),
            mode="daily",
            conversation_id="conv-unbound",
        )
        assert done.wait(timeout=5)
        mgr_run.join(timeout=5)
        assert captured["sandbox"] is None  # explicit binding, no auto-create

    def test_dispatch_executes_through_the_attached_session(
        self, manager, monkeypatch
    ):
        _load_agent_stack()
        dispatch = sys.modules["opti_oignon.agent.dispatch"]
        manager.create_sandbox("ws-d")
        sess = _st.SandboxToolSession(manager, _FakeRegistry())
        sess.attach("ws-d")
        # bwrap is absent in the container; assert the honest refusal first.
        refused = dispatch.dispatch_tool_call(
            dispatch.ToolCall(name="bash", arguments={"command": "echo hi"}),
            mode="daily",
            sandbox=sess,
        )
        assert refused.executed is False
        assert refused.reason == dispatch.REASON_SANDBOX_UNAVAILABLE
        # With bwrap declared available, the call reaches the session method:
        # the only execution path is a method on the injected session object.
        monkeypatch.setattr(manager, "_bwrap_available", True)
        called = {}
        monkeypatch.setattr(
            sess, "bash", lambda command, timeout=30: called.update(c=command) or "ok"
        )
        result = dispatch.dispatch_tool_call(
            dispatch.ToolCall(name="bash", arguments={"command": "echo hi"}),
            mode="daily",
            sandbox=sess,
        )
        assert result.executed is True
        assert called["c"] == "echo hi"
        sess.detach()

    def test_dispatch_module_is_untouched_this_session(self):
        with open(
            os.path.join(_OO, "agent", "dispatch.py"), encoding="utf-8"
        ) as fh:
            src = fh.read()
        assert "S210" not in src  # the invariant landed with ZERO edits here


# ---------------------------------------------------------------------------
# Reconcile, lazy idle TTL (bound exempt), disk soft quota
# ---------------------------------------------------------------------------

class TestReconcileTtlQuota:
    def test_reconcile_reaps_orphan_directories_only(self, tmp_path):
        manager = _make_manager(tmp_path)
        live = manager.create_sandbox("ws-live")
        base = manager.config.workspace_base
        orphan = os.path.join(base, "sandbox-orphan-x")
        os.makedirs(orphan)
        stray_file = os.path.join(base, "keep.db")
        with open(stray_file, "w", encoding="utf-8") as fh:
            fh.write("x")
        unrelated_dir = os.path.join(base, "not-a-sandbox")
        os.makedirs(unrelated_dir)
        assert manager.reconcile_workspaces() == 1
        assert not os.path.isdir(orphan)
        assert os.path.isfile(stray_file)  # the audit DB class is never touched
        assert os.path.isdir(unrelated_dir)
        assert os.path.isdir(live.workspace_path)

    def test_reconcile_runs_at_startup(self, tmp_path):
        base = tmp_path / "sbx"
        base.mkdir()
        orphan = base / "sandbox-old-run"
        orphan.mkdir()
        _make_manager(tmp_path)
        assert not orphan.exists()

    def test_persistent_base_skips_the_reap(self, tmp_path):
        base = tmp_path / "sbx"
        base.mkdir()
        orphan = base / "sandbox-old-run"
        orphan.mkdir()
        manager = _make_manager(tmp_path, workspace_persistent=True)
        assert orphan.exists()
        assert manager.reconcile_workspaces() == 0

    def test_reconcile_on_start_false_skips(self, tmp_path):
        base = tmp_path / "sbx"
        base.mkdir()
        orphan = base / "sandbox-old-run"
        orphan.mkdir()
        _make_manager(tmp_path, reconcile_on_start=False)
        assert orphan.exists()

    def test_idle_ttl_reaps_only_idle_unbound(self, tmp_path):
        manager = _make_manager(tmp_path, idle_ttl_seconds=5)
        stale = manager.create_sandbox("ws-stale")
        bound = manager.create_sandbox("ws-bound")
        running = manager.create_sandbox("ws-running")
        fresh = manager.create_sandbox("ws-fresh")
        old = time.time() - 100
        stale.last_activity = old
        bound.last_activity = old
        running.last_activity = old
        manager.set_binding("ws-bound", "conv-keep")
        bound.last_activity = old  # set_binding touched it; re-age it
        with manager._lock:
            manager._running_procs["ws-running"] = _FakePopen(["x"])
        reaped = manager._sweep_idle_sessions()
        assert reaped == 1
        ids = {s["session_id"] for s in manager.list_sessions()}
        assert "ws-stale" not in ids
        assert {"ws-bound", "ws-running", "ws-fresh"} <= ids
        with manager._lock:
            manager._running_procs.pop("ws-running", None)

    def test_ttl_zero_disables_the_sweep(self, tmp_path):
        manager = _make_manager(tmp_path, idle_ttl_seconds=0)
        s = manager.create_sandbox("ws-z")
        s.last_activity = time.time() - 10**6
        assert manager._sweep_idle_sessions() == 0

    def test_ttl_clamped_into_range(self):
        cfg = SandboxConfig(idle_ttl_seconds=10**9)
        assert cfg.idle_ttl_seconds == _sm._TTL_SECONDS_MAX
        cfg = SandboxConfig(disk_soft_limit_bytes=1)
        assert cfg.disk_soft_limit_bytes == _sm._DISK_SOFT_BYTES_MIN

    def test_sweep_runs_before_the_concurrency_check(self, tmp_path):
        manager = _make_manager(
            tmp_path, idle_ttl_seconds=5, max_concurrent_sessions=1
        )
        s = manager.create_sandbox("ws-old")
        s.last_activity = time.time() - 100
        # Without the sweep this would raise the cap error.
        manager.create_sandbox("ws-new")
        ids = {x["session_id"] for x in manager.list_sessions()}
        assert ids == {"ws-new"}

    def test_quota_refuses_inject_files(self, manager, tmp_path):
        manager._config.disk_soft_limit_bytes = 1
        manager.create_sandbox("ws-q")
        big = tmp_path / "big.txt"
        big.write_bytes(b"x" * 4096)
        with pytest.raises(_sm.WorkspaceQuotaExceeded):
            manager.inject_files("ws-q", [str(big)])

    def test_quota_refuses_inject_directory_and_keeps_workspace(
        self, manager, tmp_path
    ):
        manager._config.disk_soft_limit_bytes = 1
        manager.create_sandbox("ws-qd")
        src = tmp_path / "tree"
        src.mkdir()
        (src / "f.bin").write_bytes(b"x" * 4096)
        with pytest.raises(_sm.WorkspaceQuotaExceeded):
            manager.inject_directory("ws-qd", str(src))
        assert manager.get_session("ws-qd").active is True

    def test_quota_allows_within_limit(self, manager, tmp_path):
        manager.create_sandbox("ws-ok")
        small = tmp_path / "small.txt"
        small.write_text("hello", encoding="utf-8")
        injected = manager.inject_files("ws-ok", [str(small)])
        assert len(injected) == 1


# ---------------------------------------------------------------------------
# Routes: codes (404/403/409/413), the stop/bind surface, auth parity
# ---------------------------------------------------------------------------

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

_ROUTES_SANDBOX_CACHE = {}


def _load_routes_sandbox():
    if "mod" in _ROUTES_SANDBOX_CACHE:
        return _ROUTES_SANDBOX_CACHE["mod"]
    # Fresh schemas first (never reuse a pre-loaded one), then the router.
    # None under routes_auth makes `from .routes_auth import ...` raise
    # ImportError, driving routes_sandbox onto its fallback auth stub
    # ({"sub": None} -> effective owner "local"), which is the isolated
    # posture these tests assert against. _load_fresh restores every touched
    # sys.modules entry, so other suites in the sweep see a clean table.
    schemas = _load_fresh(
        os.path.join("opti_oignon", "api", "schemas.py"),
        register="opti_oignon.api.schemas",
    )
    deps = types.ModuleType("opti_oignon.api.deps")
    deps.SANDBOX_AVAILABLE = True
    deps.sandbox_manager = None  # patched per test on the routes module
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
        },
    )
    _ROUTES_SANDBOX_CACHE["mod"] = mod
    return mod


@pytest.fixture()
def api(manager, monkeypatch):
    rs = _load_routes_sandbox()
    monkeypatch.setattr(rs, "SANDBOX_AVAILABLE", True)
    monkeypatch.setattr(rs, "sandbox_manager", manager)
    app = fastapi.FastAPI()
    app.include_router(rs.router)
    return rs, TestClient(app), manager


class TestRoutes:
    def test_create_with_label_and_timeout(self, api):
        rs, client, manager = api
        resp = client.post(
            "/api/sandbox/create", json={"label": "demo", "timeout": 7}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["label"] == "demo"
        assert body["session_id"].startswith("ws-")
        session = manager.get_session(body["session_id"])
        assert session.timeout_override == 7
        assert session.owner_user_id == "local"

    def test_sessions_view_carries_manager_fields(self, api):
        rs, client, manager = api
        manager.create_sandbox("ws-v", label="lbl")
        resp = client.get("/api/sandbox/sessions")
        assert resp.status_code == 200
        row = resp.json()[0]
        for key in ("label", "running", "disk_use_bytes", "age_seconds"):
            assert key in row

    def test_stop_unknown_404(self, api):
        rs, client, _ = api
        assert client.post("/api/sandbox/nope/stop").status_code == 404

    def test_stop_idle_is_200_false(self, api):
        rs, client, manager = api
        manager.create_sandbox("ws-i")
        resp = client.post("/api/sandbox/ws-i/stop")
        assert resp.status_code == 200
        assert resp.json() == {"session_id": "ws-i", "stopped": False}

    def test_stop_running_is_200_true(self, api, monkeypatch):
        rs, client, manager = api
        manager.create_sandbox("ws-r")
        with manager._lock:
            manager._running_procs["ws-r"] = _FakePopen(["x"])
        monkeypatch.setattr(_sm.os, "killpg", lambda *a, **k: None)
        resp = client.post("/api/sandbox/ws-r/stop")
        assert resp.status_code == 200
        assert resp.json()["stopped"] is True

    def test_stop_foreign_owner_403(self, api):
        rs, client, manager = api
        manager.create_sandbox("ws-f", owner_user_id="someone-else")
        assert client.post("/api/sandbox/ws-f/stop").status_code == 403

    def test_destroy_then_404_idempotent_effect(self, api):
        rs, client, manager = api
        manager.create_sandbox("ws-d")
        assert client.delete("/api/sandbox/ws-d").status_code == 200
        assert client.delete("/api/sandbox/ws-d").status_code == 404

    def test_destroy_foreign_owner_403(self, api):
        rs, client, manager = api
        manager.create_sandbox("ws-f2", owner_user_id="someone-else")
        assert client.delete("/api/sandbox/ws-f2").status_code == 403
        assert manager.get_session("ws-f2").active is True

    def test_bind_flow_and_codes(self, api):
        rs, client, manager = api
        manager.create_sandbox("ws-a")
        resp = client.post(
            "/api/sandbox/bind",
            json={"conversation_id": "conv-1", "session_id": "ws-a"},
        )
        assert resp.status_code == 200
        assert resp.json()["bound"] is True
        # held by another conversation -> 409
        resp = client.post(
            "/api/sandbox/bind",
            json={"conversation_id": "conv-2", "session_id": "ws-a"},
        )
        assert resp.status_code == 409
        # unknown workspace -> 404
        resp = client.post(
            "/api/sandbox/bind",
            json={"conversation_id": "conv-3", "session_id": "nope"},
        )
        assert resp.status_code == 404
        # foreign owner -> 403
        manager.create_sandbox("ws-x", owner_user_id="someone-else")
        resp = client.post(
            "/api/sandbox/bind",
            json={"conversation_id": "conv-4", "session_id": "ws-x"},
        )
        assert resp.status_code == 403
        # read it back, then unbind (no-op safe)
        resp = client.get("/api/sandbox/bind/conv-1")
        assert resp.json() == {
            "conversation_id": "conv-1",
            "session_id": "ws-a",
            "bound": True,
        }
        resp = client.delete("/api/sandbox/bind/conv-1")
        assert resp.status_code == 200 and resp.json()["bound"] is False
        resp = client.get("/api/sandbox/bind/conv-1")
        assert resp.json()["bound"] is False
        assert client.delete("/api/sandbox/bind/conv-1").status_code == 200

    def test_inject_over_quota_is_413(self, api, tmp_path):
        rs, client, manager = api
        manager._config.disk_soft_limit_bytes = 1
        manager.create_sandbox("ws-q")
        big = tmp_path / "big.txt"
        big.write_bytes(b"x" * 4096)
        resp = client.post(
            "/api/sandbox/inject",
            json={"session_id": "ws-q", "file_paths": [str(big)]},
        )
        assert resp.status_code == 413

    def test_router_auth_parity_by_source(self):
        with open(
            os.path.join(_API, "routes_sandbox.py"), encoding="utf-8"
        ) as fh:
            src = fh.read()
        assert "_auth_dep = [Depends(_get_current_user)]" in src
        assert (
            'APIRouter(prefix="/api/sandbox", tags=["sandbox"], '
            "dependencies=_auth_dep)" in src
        )
        # The new surface lives on that router (inherits the auth dep).
        assert '@router.post("/{session_id}/stop"' in src
        assert '@router.post("/bind"' in src
        assert '@router.delete("/bind/{conversation_id}"' in src
        assert '@router.get("/bind/{conversation_id}"' in src


# ---------------------------------------------------------------------------
# Registrations: spec, cartography, FRONTEND_REDESIGN, the FRD-03 mount
# ---------------------------------------------------------------------------

def _read(relpath: str) -> str:
    with open(os.path.join(_ROOT, relpath), encoding="utf-8") as fh:
        return fh.read()


class TestRegistrations:
    def test_spec_section_4_status_landed(self):
        spec = _read("SANDBOX_WORKSPACE_SPEC.md")
        assert "### 4.4 Status (S210)" in spec
        assert "ATL-02 is closed" in spec
        assert "FRD-03 closed" in spec

    def test_spec_section_12_registers_the_module(self):
        spec = _read("SANDBOX_WORKSPACE_SPEC.md")
        assert "binding\n  LANDED S210" in spec or "LANDED S210" in spec
        assert "reset_workspace_bindings" in spec

    def test_spec_section_15_gains_the_s210_row(self):
        spec = _read("SANDBOX_WORKSPACE_SPEC.md")
        assert "tests/test_s210_sandbox_bloc1.py" in spec

    def test_frontend_spec_registers_the_components(self):
        spec = _read("FRONTEND_REDESIGN_SPEC.md")
        assert re.search(r"SandboxPanel\.svelte`?\s*\|\s*NEW\s*\|\s*S210", spec)
        assert re.search(
            r"SandboxWorkspaceList\.svelte`?\s*\|\s*NEW\s*\|\s*S210", spec
        )
        # The S176 row survives (test_s176_agent_panel pins it) with the mount.
        assert re.search(r"AgentPanel\.svelte`?\s*\|\s*NEW\s*\|\s*S176", spec)
        assert "mounted S210" in spec

    def test_components_exist_tag_balanced_token_only(self):
        for rel in (
            os.path.join(
                "frontend", "src", "lib", "components", "panels",
                "SandboxPanel.svelte",
            ),
            os.path.join(
                "frontend", "src", "lib", "components", "panels",
                "SandboxWorkspaceList.svelte",
            ),
        ):
            src = _read(rel)
            for tag in ("script", "style", "section", "ul", "li", "div", "span"):
                opens = len(re.findall(rf"<{tag}[\s>]", src))
                closes = len(re.findall(rf"</{tag}>", src))
                selfc = len(re.findall(rf"<{tag}\b[^>]*/>", src, re.S))
                assert opens == closes + selfc, f"{rel}: <{tag}> unbalanced"
            for hex_match in re.finditer(r"#[0-9a-fA-F]{3,8}\b", src):
                prefix = src[max(0, hex_match.start() - 60):hex_match.start()]
                assert "var(--oo-" in prefix, (
                    f"{rel}: raw hex outside var(--oo-*) fallback"
                )

    def test_frd03_mount_by_source(self):
        layout = _read(
            os.path.join("frontend", "src", "routes", "chat", "+layout.svelte")
        )
        assert "AgentPanel" in layout and "SandboxPanel" in layout
        assert "$activePanel === 'agent'" in layout
        assert "$activePanel === 'sandbox'" in layout
        toggle = _read(
            os.path.join(
                "frontend", "src", "lib", "components", "panels",
                "PanelToggle.svelte",
            )
        )
        assert "togglePanel('agent')" in toggle
        assert "togglePanel('sandbox')" in toggle
        types_src = _read(os.path.join("frontend", "src", "lib", "types.ts"))
        assert "'agent'" in types_src.split("PanelType")[1].split(";")[0]
        assert "'sandbox'" in types_src.split("PanelType")[1].split(";")[0]

    def test_sandbox_ts_client_extended(self):
        src = _read(os.path.join("frontend", "src", "lib", "api", "sandbox.ts"))
        for fn in (
            "stopSandboxCommand",
            "bindConversation",
            "unbindConversation",
            "getConversationBinding",
        ):
            assert f"export async function {fn}" in src

    def test_pyproject_deselects_the_superseded_s209_assertions(self):
        src = _read("pyproject.toml")
        for test_id in (
            "test_s209_sandbox_bloc0.py::TestRunBwrapWiring::test_rlimit_backend_wires_preexec_and_passes_seccomp_fd",
            "test_s209_sandbox_bloc0.py::TestRunBwrapWiring::test_limits_disabled_means_no_preexec",
            "test_s209_sandbox_bloc0.py::TestRunBwrapWiring::test_seccomp_disabled_means_no_flag",
            "test_s209_sandbox_bloc0.py::TestRunBwrapWiring::test_cgroup_backend_prefixes_and_drops_preexec",
            "test_s209_sandbox_bloc0.py::TestRunBwrapWiring::test_cgroup_unavailable_falls_back_to_rlimit",
            "test_s209_sandbox_bloc0.py::TestSeccompFailSecure::test_required_false_launches_unfiltered_with_warning",
        ):
            assert test_id in src, f"missing deselect: {test_id}"

    def test_config_documents_the_lifecycle_keys(self):
        src = _read(os.path.join("opti_oignon", "config", "sandbox.yaml"))
        for key in (
            "workspace_persistent",
            "reconcile_on_start",
            "idle_ttl_seconds",
            "disk_soft_limit_bytes",
        ):
            assert key in src
