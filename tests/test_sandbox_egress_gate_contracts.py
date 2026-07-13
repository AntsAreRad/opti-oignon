#!/usr/bin/env python3
"""Sandbox egress gate contracts: network refused by default, Daily-only.

Outbound network for a workspace is a capability that must be granted, and
every layer of the grant fails toward refusal. The status strip reports the
live gate answer and reports False whenever the gate is absent or errors.
The per-workspace toggle refuses as unavailable without the gate module and
maps the gate's refusal to a forbidden answer. The provision run -- the one
scoped egress -- walks a refusal ladder before anything installs: mode gate
first (an unreadable mode counts as the isolated one), then the explicit
per-workspace flag, then the relative-path checks, then an exact-and-pinned
requirements validation; the command it finally runs is built server-side
only. This suite pins that behavior:

  * EG1 -- the status answer for network capability is False when the gate
    module is absent and False when the gate errors, True only when the
    live gate says so;
  * EG2 -- the network toggle refuses as unavailable without the gate,
    passes the caller's flag and identity to the manager, and maps the
    manager's refusal to a forbidden answer;
  * EG3 -- the provision run is refused as forbidden when the mode gate
    answers no, and equally refused when the gate is unreadable, both
    audited, and nothing executes;
  * EG4 -- with the mode gate open but the per-workspace flag off, the
    provision run is refused as a state conflict, audited, and nothing
    executes;
  * EG5 -- a refused relative path (requirements file or venv directory)
    stops the provision run with a caller error naming the field, and
    nothing executes;
  * EG6 -- a requirements set with any refused line installs nothing and
    surfaces the refusals; on the accepted path the executed command is
    exactly the server-built one and the run is audited with the actor.

Loads the facade module in isolation under a stand-in package; every
``opti_oignon.*`` entry plus the web-framework and model-client entries is
snapshotted and evicted first, and the seeds are deterministic recorders. A
meta-path guard refuses any project submodule that was not seeded, so the
load behaves identically whether or not the project is installed.
Local-only. Runs under pytest or the __main__ runner.
"""

import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"

_SCHEMA_NAMES = (
    "HostBrowseEntry", "HostBrowseResponse", "QuickSandboxSessionInfo",
    "QuickSandboxStatusResponse", "QuickSandboxToggleRequest",
    "QuickSandboxTTLRequest", "SandboxApplyEntry", "SandboxApplyRefusedEntry",
    "SandboxApplyRequest", "SandboxApplyResponse", "SandboxApprovalAuditEntry",
    "SandboxApprovalAuditResponse", "SandboxApprovalInfoResponse",
    "SandboxApproveRequest", "SandboxApproveResponse", "SandboxAuditEntry",
    "SandboxAuditResponse", "SandboxBindingResponse", "SandboxBindRequest",
    "SandboxCloneRequest", "SandboxCloneResponse",
    "SandboxConfirmDegradedResponse", "SandboxConfirmDeletionsRefused",
    "SandboxConfirmDeletionsRequest", "SandboxConfirmDeletionsResponse",
    "SandboxCopyOutEntry", "SandboxCopyOutResponse", "SandboxCreateRequest",
    "SandboxCreateResponse", "SandboxDestroyResponse", "SandboxDiffEntry",
    "SandboxDiffResponse", "SandboxExecuteRequest", "SandboxExecuteResponse",
    "SandboxFileEntry", "SandboxFilesResponse", "SandboxInjectRequest",
    "SandboxInjectResponse", "SandboxNetworkToggleRequest",
    "SandboxNetworkToggleResponse", "SandboxPreviewResponse",
    "SandboxProvisionRefusedLine", "SandboxProvisionRequest",
    "SandboxProvisionResponse", "SandboxRejectResponse", "SandboxSessionInfo",
    "SandboxStatusResponse", "SandboxStopResponse", "SandboxUploadRefused",
    "SandboxUploadResponse",
)


class _IsolationGuard:
    """Refuse every project submodule the test did not seed.

    A stand-in package whose ``__path__`` is empty isolates the tree only
    while the parent path is the sole way to resolve a submodule. That
    assumption breaks wherever the project is installed in editable mode:
    such an install registers a finder that answers on the module NAME and
    ignores the parent path, so a real submodule resolves behind the test's
    back -- silently importing live code. This guard sits ahead of every
    finder and refuses the names that were not seeded, so a load behaves
    identically whether the project is installed or not.
    """

    _PREFIX = "opti_oignon."

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(self._PREFIX):
            raise ModuleNotFoundError(
                f"not seeded in the isolation window: {fullname}",
                name=fullname,
            )
        return None


class _HTTPRefusal(Exception):
    """Framework stand-in refusal carrying the status code and detail."""

    def __init__(self, status_code, detail=""):
        super().__init__(f"{status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class _Router:
    """Framework stand-in router; decorators return the function as-is."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.registered = []

    def _decorate(self, method, path):
        def deco(fn):
            self.registered.append((method, path))
            return fn
        return deco

    def get(self, path, **kwargs):
        return self._decorate("GET", path)

    def post(self, path, **kwargs):
        return self._decorate("POST", path)

    def delete(self, path, **kwargs):
        return self._decorate("DELETE", path)


class _Model:
    """Schema stand-in: an attribute bag with a dump method."""

    def __init__(self, **kwargs):
        self._kwargs = dict(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def model_dump(self):
        return dict(self._kwargs)


class _Audit:
    """Recorder for the manager-side approval/audit journal."""

    def __init__(self):
        self.entries = []

    def log_approval(self, session_id, action="", paths=None, detail="",
                     dest_dir=""):
        self.entries.append({
            "session_id": session_id,
            "action": action,
            "paths": list(paths or []),
            "detail": detail,
        })

    def actions(self):
        return [e["action"] for e in self.entries]

    def details_for(self, action):
        return [e["detail"] for e in self.entries if e["action"] == action]


class _Manager:
    """Configurable recording manager standing behind the facade."""

    def __init__(self, workspace=None):
        self.audit = _Audit()
        self.sessions = {}
        self.workspace = workspace
        self.set_network_calls = []
        self.set_network_result = True
        self.set_network_error = None
        self.provision_calls = []
        self.provision_result = SimpleNamespace(
            return_code=0, blocked=False, block_reason="", timed_out=False,
            isolation_backend="isolated", stdout="installed", stderr="",
        )
        self.config = SimpleNamespace(
            enabled=True, max_concurrent_sessions=3, command_timeout=30,
            limit_memory_bytes=1024, limit_nproc=8, limit_cpu_seconds=60,
            disk_soft_limit_bytes=4096,
        )
        self.isolation_backend = SimpleNamespace(value="isolated")
        self.bwrap_available = True
        self.degraded_mode = False
        self.degraded_confirmed = False
        self.active_session_count = 0

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def get_active_workspace_path(self, session_id):
        if self.workspace is None:
            raise ValueError(f"Session not found: {session_id}")
        return self.workspace

    def set_network_enabled(self, session_id, enabled, actor=""):
        self.set_network_calls.append((session_id, enabled, actor))
        if self.set_network_error is not None:
            raise self.set_network_error
        return self.set_network_result

    def execute_provision_command(self, session_id, command):
        self.provision_calls.append((session_id, list(command)))
        return self.provision_result


def _session(owner="alice", network=False):
    return SimpleNamespace(
        owner_user_id=owner,
        network_enabled=network,
        approved_paths=set(),
        confirmed_deletions=set(),
    )


def _fail_unconfigured(*_args, **_kwargs):
    raise AssertionError("stub callable was not configured by the test")


def _load():
    """Load the sandbox REST facade under a stand-in package."""
    lateral = [
        k for k in list(sys.modules)
        if k == "fastapi" or k.startswith("fastapi.")
    ]
    keys = ["ollama"] + lateral + [
        k
        for k in list(sys.modules)
        if k == "opti_oignon" or k.startswith("opti_oignon.")
    ]
    saved = {k: sys.modules[k] for k in keys if k in sys.modules}
    for k in keys:
        sys.modules.pop(k, None)
    sys.modules["ollama"] = None  # no client import exists; drift fails loud

    framework = types.ModuleType("fastapi")
    framework.APIRouter = _Router
    framework.HTTPException = _HTTPRefusal
    framework.Depends = lambda fn=None: ("dependency", fn)
    framework.File = lambda default=None: default
    framework.Form = lambda default="": default
    framework.UploadFile = type("UploadFile", (), {})
    responses = types.ModuleType("fastapi.responses")
    responses.FileResponse = type("FileResponse", (), {})
    framework.responses = responses
    sys.modules["fastapi"] = framework
    sys.modules["fastapi.responses"] = responses

    root = types.ModuleType("opti_oignon")
    root.__path__ = []
    sys.modules["opti_oignon"] = root

    api = types.ModuleType("opti_oignon.api")
    api.__path__ = []
    sys.modules["opti_oignon.api"] = api
    root.api = api

    deps = types.ModuleType("opti_oignon.api.deps")
    deps.FILE_TOOLS_AVAILABLE = True
    deps.SANDBOX_AVAILABLE = True
    deps.sandbox_manager = None
    sys.modules["opti_oignon.api.deps"] = deps
    api.deps = deps

    schemas = types.ModuleType("opti_oignon.api.schemas")
    for name in _SCHEMA_NAMES:
        setattr(schemas, name, type(name, (_Model,), {}))
    sys.modules["opti_oignon.api.schemas"] = schemas
    api.schemas = schemas

    ws = types.ModuleType("opti_oignon.sandbox_workspace")
    for exc_name in (
        "WorkspaceNotFound", "WorkspaceOwnerMismatch", "WorkspaceAlreadyBound",
        "WorkspaceBindingError", "WorkspaceDiffBoundExceeded",
        "WorkspaceReviewDrift", "WorkspaceApplyTargetError",
    ):
        setattr(ws, exc_name, type(exc_name, (Exception,), {}))
    ws.get_workspace_bindings = _fail_unconfigured
    ws.get_workspace_manifests = _fail_unconfigured
    ws.generate_workspace_diff = _fail_unconfigured
    ws.apply_workspace_changes = _fail_unconfigured
    sys.modules["opti_oignon.sandbox_workspace"] = ws
    root.sandbox_workspace = ws

    eg = types.ModuleType("opti_oignon.sandbox_egress")
    eg.network_allowed = _fail_unconfigured
    eg.refuse_rel_path = _fail_unconfigured
    eg.validate_requirements_text = _fail_unconfigured
    eg.build_provision_command = _fail_unconfigured
    sys.modules["opti_oignon.sandbox_egress"] = eg
    root.sandbox_egress = eg

    stop = types.ModuleType("opti_oignon.emergency_stop")
    stop.guard_http = lambda: None
    sys.modules["opti_oignon.emergency_stop"] = stop
    root.emergency_stop = stop

    manager_mod = types.ModuleType("opti_oignon.sandbox_manager")
    manager_mod.WorkspaceQuotaExceeded = type(
        "WorkspaceQuotaExceeded", (Exception,), {}
    )
    manager_mod.validate_sandbox_path = _fail_unconfigured
    sys.modules["opti_oignon.sandbox_manager"] = manager_mod
    root.sandbox_manager = manager_mod

    isolation = types.ModuleType("opti_oignon.user_isolation")
    isolation.effective_user_id = (
        lambda user_id, single_user_mode=True: user_id or "local"
    )
    sys.modules["opti_oignon.user_isolation"] = isolation
    root.user_isolation = isolation

    guard = _IsolationGuard()
    sys.meta_path.insert(0, guard)

    def restore():
        try:
            sys.meta_path.remove(guard)
        except ValueError:
            pass
        for k in list(sys.modules):
            if k == "opti_oignon" or k.startswith("opti_oignon."):
                del sys.modules[k]
            elif k == "fastapi" or k.startswith("fastapi."):
                del sys.modules[k]
        sys.modules.pop("ollama", None)
        for k, v in saved.items():
            sys.modules[k] = v

    full = "opti_oignon.api.routes_sandbox"
    spec = importlib.util.spec_from_file_location(
        full, _OO / "api" / "routes_sandbox.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    api.routes_sandbox = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        restore()
        raise

    return SimpleNamespace(mod=mod, ws=ws, eg=eg, restore=restore)


def _expect_refusal(status, fn, *args, **kwargs):
    """Call the route and assert the stand-in refusal with the given code."""
    try:
        fn(*args, **kwargs)
    except _HTTPRefusal as exc:
        assert exc.status_code == status, (
            f"expected status {status}, got {exc.status_code}: {exc.detail!r}"
        )
        return exc
    raise AssertionError(f"expected a {status} refusal, nothing was raised")


def _wire(ctx, manager):
    ctx.mod.SANDBOX_AVAILABLE = True
    ctx.mod.sandbox_manager = manager


def _provision_request(requirements="requirements.txt", venv=".venv"):
    return SimpleNamespace(requirements_path=requirements, venv_dir=venv)


# ---------------------------------------------------------------------------
# EG1 -- the status network answer is fail-secure
# ---------------------------------------------------------------------------
def test_eg1_status_network_answer_is_fail_secure():
    ctx = _load()
    try:
        mgr = _Manager()
        _wire(ctx, mgr)

        ctx.mod.EGRESS_AVAILABLE = False
        ctx.mod._eg = None
        response = ctx.mod.get_sandbox_status()
        assert response.network_allowed is False, (
            "an absent gate module must surface network as refused"
        )

        ctx.mod.EGRESS_AVAILABLE = True
        ctx.mod._eg = ctx.eg

        def broken():
            raise RuntimeError("gate unreadable")

        ctx.eg.network_allowed = broken
        response = ctx.mod.get_sandbox_status()
        assert response.network_allowed is False, (
            "an erroring gate must surface network as refused, never open"
        )

        ctx.eg.network_allowed = lambda: True
        response = ctx.mod.get_sandbox_status()
        assert response.network_allowed is True, (
            "only the live gate saying yes may surface network as allowed"
        )
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# EG2 -- the toggle refuses without the gate and maps the gate refusal
# ---------------------------------------------------------------------------
def test_eg2_network_toggle_refuses_without_gate_and_maps_refusal():
    ctx = _load()
    try:
        mgr = _Manager()
        mgr.sessions["sid-1"] = _session(owner="alice")
        _wire(ctx, mgr)
        user = {"sub": "alice"}
        request = SimpleNamespace(enabled=True)

        ctx.mod.EGRESS_AVAILABLE = False
        ctx.mod._eg = None
        exc = _expect_refusal(
            503, ctx.mod.toggle_workspace_network, "sid-1", request,
            current_user=user,
        )
        assert "not available" in str(exc.detail), exc.detail
        assert mgr.set_network_calls == [], (
            "an absent gate must never reach the manager"
        )

        ctx.mod.EGRESS_AVAILABLE = True
        ctx.mod._eg = ctx.eg
        ctx.mod.toggle_workspace_network(
            "sid-1", request, current_user=user
        )
        assert mgr.set_network_calls == [("sid-1", True, "alice")], (
            "the caller's flag and identity must reach the manager, got "
            f"{mgr.set_network_calls}"
        )

        mgr.set_network_error = PermissionError("only the daily mode")
        _expect_refusal(
            403, ctx.mod.toggle_workspace_network, "sid-1", request,
            current_user=user,
        )
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# EG3 -- the provision mode gate refuses closed, readable or not
# ---------------------------------------------------------------------------
def test_eg3_provision_mode_gate_refuses_closed_and_unreadable():
    ctx = _load()
    try:
        mgr = _Manager(workspace="unused")
        mgr.sessions["sid-1"] = _session(owner="alice", network=True)
        _wire(ctx, mgr)
        user = {"sub": "alice"}

        ctx.eg.network_allowed = lambda: False
        exc = _expect_refusal(
            403, ctx.mod.provision_workspace, "sid-1", _provision_request(),
            current_user=user,
        )
        assert "Daily-only" in str(exc.detail), exc.detail

        def unreadable():
            raise RuntimeError("mode store lost")

        ctx.eg.network_allowed = unreadable
        exc = _expect_refusal(
            403, ctx.mod.provision_workspace, "sid-1", _provision_request(),
            current_user=user,
        )
        assert "could not be determined" in str(exc.detail), exc.detail

        refusals = mgr.audit.details_for("provision_refused")
        assert any("mode is not daily" in d for d in refusals), refusals
        assert any("gate unreadable" in d for d in refusals), refusals
        assert mgr.provision_calls == [], (
            "a refused mode gate must never execute anything"
        )
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# EG4 -- the per-workspace flag is a precondition even with the gate open
# ---------------------------------------------------------------------------
def test_eg4_provision_requires_the_explicit_per_workspace_flag():
    ctx = _load()
    try:
        mgr = _Manager(workspace="unused")
        mgr.sessions["sid-1"] = _session(owner="alice", network=False)
        _wire(ctx, mgr)
        ctx.eg.network_allowed = lambda: True

        exc = _expect_refusal(
            409, ctx.mod.provision_workspace, "sid-1",
            _provision_request(), current_user={"sub": "alice"},
        )
        assert "Network is not enabled" in str(exc.detail), exc.detail
        refusals = mgr.audit.details_for("provision_refused")
        assert any("network flag is off" in d for d in refusals), refusals
        assert mgr.provision_calls == [], (
            "an off flag must never execute anything"
        )
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# EG5 -- a refused relative path stops the run before anything executes
# ---------------------------------------------------------------------------
def test_eg5_refused_relative_paths_stop_the_provision_run():
    ctx = _load()
    try:
        mgr = _Manager(workspace="unused")
        mgr.sessions["sid-1"] = _session(owner="alice", network=True)
        _wire(ctx, mgr)
        ctx.eg.network_allowed = lambda: True
        seen = []

        def refuse_escapes(rel):
            seen.append(rel)
            return "escapes the workspace" if ".." in str(rel) else None

        ctx.eg.refuse_rel_path = refuse_escapes
        exc = _expect_refusal(
            400, ctx.mod.provision_workspace, "sid-1",
            _provision_request(requirements="../reqs.txt"),
            current_user={"sub": "alice"},
        )
        assert "requirements_path" in str(exc.detail), exc.detail

        exc = _expect_refusal(
            400, ctx.mod.provision_workspace, "sid-1",
            _provision_request(venv="../venv"),
            current_user={"sub": "alice"},
        )
        assert "venv_dir" in str(exc.detail), exc.detail
        assert "requirements.txt" in seen and "../venv" in seen, (
            f"both fields must be checked, saw {seen}"
        )
        assert mgr.provision_calls == [], (
            "a refused path must never execute anything"
        )
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# EG6 -- refused lines install nothing; the executed command is server-built
# ---------------------------------------------------------------------------
def test_eg6_requirements_refusals_install_nothing_and_command_is_server_built():
    ctx = _load()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "requirements.txt").write_text(
                "pkg==1.0\n", encoding="ascii"
            )
            mgr = _Manager(workspace=tmp)
            mgr.sessions["sid-1"] = _session(owner="alice", network=True)
            _wire(ctx, mgr)
            ctx.eg.network_allowed = lambda: True
            ctx.eg.refuse_rel_path = lambda rel: None
            user = {"sub": "alice"}

            ctx.eg.validate_requirements_text = lambda text: (
                [], [{"line": 1, "reason": "missing hash pin"}]
            )
            exc = _expect_refusal(
                400, ctx.mod.provision_workspace, "sid-1",
                _provision_request(), current_user=user,
            )
            assert "Nothing was installed" in exc.detail["message"]
            assert exc.detail["refused"] == [
                {"line": 1, "reason": "missing hash pin"}
            ]
            assert mgr.provision_calls == [], (
                "a partially refused requirements set must install nothing"
            )
            refusals = mgr.audit.details_for("provision_refused")
            assert any("nothing installed" in d for d in refusals), refusals

            ctx.eg.validate_requirements_text = lambda text: (
                ["pkg==1.0 --hash=sha256:aa"], []
            )
            marker = ["/usr/bin/env", "provision", "requirements.txt",
                      ".venv"]
            ctx.eg.build_provision_command = lambda req, venv: list(marker)
            response = ctx.mod.provision_workspace(
                "sid-1", _provision_request(), current_user=user
            )
            assert mgr.provision_calls == [("sid-1", marker)], (
                "the executed command must be exactly the server-built one, "
                f"got {mgr.provision_calls}"
            )
            assert response.command == marker
            runs = mgr.audit.details_for("provision_run")
            assert len(runs) == 1 and "actor=alice" in runs[0], runs
    finally:
        ctx.restore()


if __name__ == "__main__":
    _failures = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except Exception as _e:  # noqa: BLE001
                _failures += 1
                print(f"FAIL {_name}: {_e!r}")
    print(f"\n{'OK' if _failures == 0 else str(_failures) + ' FAILED'}")
    sys.exit(1 if _failures else 0)
