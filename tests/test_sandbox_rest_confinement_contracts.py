#!/usr/bin/env python3
"""Sandbox REST facade contracts: gated exits, owned writes, honest destroy.

The sandbox REST facade is the only path by which workspace bytes can ever
reach the host, and every one of those exits is a human action: a download
requires prior approval AND path validation, a batch copy-out is confined
to the data directory, the diff/apply cycle demands an echo of the reviewed
digest, and a deletion applies only from its own separately confirmed set.
Write-shaped routes are owner-checked, and the destructive single-segment
catch-all is registered last so it can never shadow a literal route. This
suite pins that behavior:

  * RC1 -- an unapproved download is refused before any workspace access,
    and approval alone never bypasses path validation;
  * RC2 -- the copy-out destination is confined to the data directory,
    traversal and absolute escapes refused, the validated real path is
    what the manager receives;
  * RC3 -- write surfaces answer 404 for an unknown workspace and 403 for
    a foreign owner, before the underlying operation runs; the caller
    identity flows through the effective-user seam;
  * RC4 -- with the workspace module absent, diff, deletion confirmation
    and apply all refuse as unavailable instead of degrading;
  * RC5 -- apply passes the caller's reviewed digest through unchanged,
    maps review drift to a conflict, and maps a bad target to a caller
    error;
  * RC6 -- deletion confirmation accepts only paths the current diff
    classifies as deleted, refusing live, unknown and empty paths each
    with its own honest reason;
  * RC7 -- destroy checks ownership, answers 404 on a second destroy,
    forgets the baseline manifest, and stays the last registered route.

Loads the facade module in isolation under a stand-in package; every
``opti_oignon.*`` entry plus the web-framework and model-client entries is
snapshotted and evicted first, and the seeds are deterministic recorders: a
minimal framework stand-in whose refusal type carries the status code, a
schema module of attribute bags, a workspace/egress pair of configurable
stubs, and a path validator with real containment semantics that journals
every call. A meta-path guard refuses any project submodule that was not
seeded, so the load behaves identically whether or not the project is
installed. Local-only. Runs under pytest or the __main__ runner.
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
    """Framework stand-in router journaling the registration order."""

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


class _FileHandle:
    """Framework stand-in file response capturing its arguments."""

    def __init__(self, path, filename=None, media_type=None):
        self.path = path
        self.filename = filename
        self.media_type = media_type


class _Model:
    """Schema stand-in: an attribute bag with a dump method."""

    def __init__(self, **kwargs):
        self._kwargs = dict(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def model_dump(self):
        return dict(self._kwargs)


class _PathValidator:
    """Recording path validator with real containment semantics."""

    def __init__(self):
        self.calls = []

    def __call__(self, workspace_root, requested_path):
        self.calls.append((str(workspace_root), str(requested_path)))
        if not requested_path:
            return False, "", "Empty path"
        workspace_real = os.path.realpath(workspace_root)
        if os.path.isabs(requested_path):
            return False, "", "absolute paths are refused by the stand-in"
        resolved = os.path.realpath(
            os.path.join(workspace_real, requested_path)
        )
        if (
            not resolved.startswith(workspace_real + os.sep)
            and resolved != workspace_real
        ):
            return False, "", "path escapes the workspace"
        return True, resolved, ""


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


class _Manager:
    """Configurable recording manager standing behind the facade."""

    def __init__(self, workspace=None):
        self.audit = _Audit()
        self.sessions = {}
        self.workspace = workspace
        self.workspace_calls = []
        self.approved = set()
        self.approval_checks = []
        self.copy_out_calls = []
        self.confirm_calls = []
        self.clone_calls = []
        self.clone_result = None
        self.stop_calls = []
        self.destroy_calls = []
        self.destroy_result = True

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def _get_active_workspace(self, session_id):
        self.workspace_calls.append(session_id)
        if self.workspace is None:
            raise ValueError(f"Session not found: {session_id}")
        return self.workspace

    def get_active_workspace_path(self, session_id):
        return self._get_active_workspace(session_id)

    def is_file_approved(self, session_id, path):
        self.approval_checks.append((session_id, path))
        return path in self.approved

    def copy_out_batch(self, session_id, paths, dest):
        self.copy_out_calls.append((session_id, list(paths), dest))
        return []

    def confirm_deletions(self, session_id, paths):
        self.confirm_calls.append((session_id, list(paths)))
        return list(paths)

    def clone_directory(self, session_id, src_path, dest_subdir=""):
        self.clone_calls.append((session_id, src_path, dest_subdir))
        if isinstance(self.clone_result, Exception):
            raise self.clone_result
        return self.clone_result

    def stop_command(self, session_id):
        self.stop_calls.append(session_id)
        return True

    def destroy_sandbox(self, session_id):
        self.destroy_calls.append(session_id)
        return self.destroy_result


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
    responses.FileResponse = _FileHandle
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

    validator = _PathValidator()
    manager_mod = types.ModuleType("opti_oignon.sandbox_manager")
    manager_mod.WorkspaceQuotaExceeded = type(
        "WorkspaceQuotaExceeded", (Exception,), {}
    )
    manager_mod.validate_sandbox_path = validator
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

    return SimpleNamespace(
        mod=mod, ws=ws, eg=eg, validator=validator, restore=restore
    )


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


# ---------------------------------------------------------------------------
# RC1 -- unapproved download refused first; approval never bypasses paths
# ---------------------------------------------------------------------------
def test_rc1_download_is_approval_gated_and_path_validated():
    ctx = _load()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "report.txt").write_text("payload", encoding="ascii")
            mgr = _Manager(workspace=tmp)
            _wire(ctx, mgr)

            exc = _expect_refusal(
                403, ctx.mod.download_sandbox_file, "sid-1", "report.txt"
            )
            assert "not approved" in str(exc.detail), exc.detail
            assert mgr.workspace_calls == [], (
                "an unapproved download must be refused before any "
                "workspace access"
            )
            assert ctx.validator.calls == [], (
                "an unapproved download must not even resolve the path"
            )

            mgr.approved.add("report.txt")
            handle = ctx.mod.download_sandbox_file("sid-1", "report.txt")
            workspace_real = os.path.realpath(tmp)
            assert handle.path.startswith(workspace_real + os.sep), (
                f"the served path must stay inside the workspace, "
                f"got {handle.path!r}"
            )
            assert handle.filename == "report.txt"

            mgr.approved.add("../outside.txt")
            exc = _expect_refusal(
                400, ctx.mod.download_sandbox_file, "sid-1", "../outside.txt"
            )
            assert "Invalid path" in str(exc.detail), (
                "approval alone must never bypass path validation"
            )
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# RC2 -- copy-out destination confined to the data directory
# ---------------------------------------------------------------------------
def test_rc2_copy_out_destination_is_confined_to_the_data_directory():
    ctx = _load()
    try:
        mgr = _Manager(workspace="unused")
        _wire(ctx, mgr)
        allowed_root = os.path.realpath(str(_REPO / "data"))
        assert ctx.mod._ALLOWED_EXPORT_ROOT == allowed_root, (
            "the export root must be the project data directory, got "
            f"{ctx.mod._ALLOWED_EXPORT_ROOT!r}"
        )

        for hostile in ("/etc/cron.d", str(_REPO / "data" / ".." / "docs")):
            exc = _expect_refusal(
                403,
                ctx.mod.copy_out_sandbox_files,
                "sid-1",
                SimpleNamespace(paths=["a.txt"], dest_dir=hostile),
            )
            assert "within the data/ directory" in str(exc.detail), exc.detail
        assert mgr.copy_out_calls == [], (
            "a refused destination must never reach the manager"
        )

        inside = str(_REPO / "data" / "sandbox_exports")
        ctx.mod.copy_out_sandbox_files(
            "sid-1", SimpleNamespace(paths=["a.txt"], dest_dir=inside)
        )
        assert len(mgr.copy_out_calls) == 1
        _sid, _paths, dest = mgr.copy_out_calls[0]
        assert dest == os.path.realpath(inside), (
            "the manager must receive the validated real destination"
        )
        assert dest.startswith(allowed_root + os.sep), dest
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# RC3 -- write surfaces are owner-checked before the operation runs
# ---------------------------------------------------------------------------
def test_rc3_write_surfaces_refuse_unknown_and_foreign_workspaces():
    ctx = _load()
    try:
        mgr = _Manager(workspace="unused")
        mgr.clone_result = {
            "dest": "cloned", "cloned_root": "root", "copied_files": 1,
            "copied_bytes": 1, "skipped_symlinks": 0, "skipped_special": 0,
            "manifest": {},
        }
        _wire(ctx, mgr)
        ctx.mod.WORKSPACE_BINDING_AVAILABLE = False
        ctx.mod._ws = None

        request = SimpleNamespace(src_path="/share/src", dest_subdir="")
        _expect_refusal(
            404, ctx.mod.clone_host_directory, "ghost", request,
            current_user={"sub": "alice"},
        )

        mgr.sessions["sid-1"] = _session(owner="alice")
        _expect_refusal(
            403, ctx.mod.clone_host_directory, "sid-1", request,
            current_user={"sub": "mallory"},
        )
        assert mgr.clone_calls == [], (
            "a foreign owner must be refused before the clone runs"
        )
        _expect_refusal(
            403, ctx.mod.stop_sandbox_command, "sid-1",
            current_user={"sub": "mallory"},
        )
        assert mgr.stop_calls == [], (
            "a foreign owner must be refused before the stop runs"
        )

        ctx.mod.clone_host_directory(
            "sid-1", request, current_user={"sub": "alice"}
        )
        assert mgr.clone_calls == [("sid-1", "/share/src", "")], (
            "the owning caller must reach the underlying operation"
        )

        mgr.sessions["sid-2"] = _session(owner="local")
        ctx.mod.stop_sandbox_command("sid-2", current_user={"sub": None})
        assert mgr.stop_calls == ["sid-2"], (
            "an anonymous caller must map to the local owner through the "
            "effective-user seam"
        )
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# RC4 -- diff, deletion confirmation and apply refuse without the module
# ---------------------------------------------------------------------------
def test_rc4_review_cycle_refuses_when_the_workspace_module_is_absent():
    ctx = _load()
    try:
        mgr = _Manager(workspace="unused")
        mgr.sessions["sid-1"] = _session(owner="alice")
        _wire(ctx, mgr)
        ctx.mod.WORKSPACE_BINDING_AVAILABLE = False
        ctx.mod._ws = None

        user = {"sub": "alice"}
        exc = _expect_refusal(
            503, ctx.mod.get_workspace_diff, "sid-1", current_user=user
        )
        assert "not available" in str(exc.detail), exc.detail
        _expect_refusal(
            503, ctx.mod.confirm_workspace_deletions, "sid-1",
            SimpleNamespace(paths=["x"]), current_user=user,
        )
        _expect_refusal(
            503, ctx.mod.apply_workspace_changes, "sid-1",
            SimpleNamespace(diff_hash="h", target_dir=None),
            current_user=user,
        )
        assert mgr.confirm_calls == [], (
            "an unavailable review cycle must never reach the manager"
        )
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# RC5 -- apply echoes the reviewed digest and maps drift to a conflict
# ---------------------------------------------------------------------------
def test_rc5_apply_echoes_the_reviewed_digest_and_maps_drift():
    ctx = _load()
    try:
        mgr = _Manager(workspace="unused")
        mgr.sessions["sid-1"] = _session(owner="alice")
        _wire(ctx, mgr)
        user = {"sub": "alice"}
        request = SimpleNamespace(diff_hash="digest-under-test",
                                  target_dir="/share/target")

        def drift(*_args, **_kwargs):
            raise ctx.ws.WorkspaceReviewDrift("workspace drifted")

        ctx.ws.apply_workspace_changes = drift
        _expect_refusal(
            409, ctx.mod.apply_workspace_changes, "sid-1", request,
            current_user=user,
        )

        def bad_target(*_args, **_kwargs):
            raise ctx.ws.WorkspaceApplyTargetError("no target")

        ctx.ws.apply_workspace_changes = bad_target
        _expect_refusal(
            400, ctx.mod.apply_workspace_changes, "sid-1", request,
            current_user=user,
        )

        received = {}

        def record(session_id, diff_hash, manager=None, target_dir=None):
            received.update(
                session_id=session_id, diff_hash=diff_hash,
                manager=manager, target_dir=target_dir,
            )
            return {
                "target": "/share/target", "applied": [], "deleted": [],
                "refused": [], "skipped_unapproved": 0,
                "skipped_unconfirmed": 0, "diff_hash": diff_hash,
            }

        ctx.ws.apply_workspace_changes = record
        response = ctx.mod.apply_workspace_changes(
            "sid-1", request, current_user=user
        )
        assert received["diff_hash"] == "digest-under-test", (
            "the caller's reviewed digest must be passed through unchanged"
        )
        assert received["manager"] is mgr
        assert received["target_dir"] == "/share/target"
        assert response.diff_hash == "digest-under-test"
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# RC6 -- deletion confirmation accepts only currently-deleted paths
# ---------------------------------------------------------------------------
def test_rc6_deletion_confirmation_filters_on_the_current_diff():
    ctx = _load()
    try:
        mgr = _Manager(workspace="unused")
        mgr.sessions["sid-1"] = _session(owner="alice")
        _wire(ctx, mgr)

        diff = SimpleNamespace(
            baseline_present=True, cloned_root=None, cloned_mount=None,
            entries=[
                SimpleNamespace(path="gone.txt", kind="deleted"),
                SimpleNamespace(path="live.txt", kind="modified"),
            ],
            unchanged=0, skipped_symlinks=0, skipped_special=0,
            diff_hash="h1",
        )
        ctx.ws.generate_workspace_diff = lambda sid, manager=None: diff

        response = ctx.mod.confirm_workspace_deletions(
            "sid-1",
            SimpleNamespace(paths=["gone.txt", "live.txt", "ghost.txt", ""]),
            current_user={"sub": "alice"},
        )
        assert mgr.confirm_calls == [("sid-1", ["gone.txt"])], (
            "only paths the current diff classifies as deleted may be "
            f"confirmed, manager saw {mgr.confirm_calls}"
        )
        assert response.confirmed == ["gone.txt"]
        reasons = {r.path: r.reason for r in response.refused}
        assert set(reasons) == {"live.txt", "ghost.txt", ""}, reasons
        assert reasons[""] == "empty path"
        assert "not classified as deleted" in reasons["live.txt"]
        assert "not classified as deleted" in reasons["ghost.txt"]
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# RC7 -- destroy: owned, honestly idempotent, forgetting, registered last
# ---------------------------------------------------------------------------
def test_rc7_destroy_is_owned_honest_forgetful_and_registered_last():
    ctx = _load()
    try:
        mgr = _Manager(workspace="unused")
        mgr.sessions["sid-1"] = _session(owner="alice")
        _wire(ctx, mgr)

        dropped = []
        ctx.ws.get_workspace_manifests = lambda: SimpleNamespace(
            drop=lambda sid: dropped.append(sid)
        )

        _expect_refusal(
            403, ctx.mod.destroy_sandbox, "sid-1",
            current_user={"sub": "mallory"},
        )
        assert mgr.destroy_calls == [], (
            "a foreign owner must be refused before anything is destroyed"
        )

        response = ctx.mod.destroy_sandbox(
            "sid-1", current_user={"sub": "alice"}
        )
        assert response.destroyed is True
        assert mgr.destroy_calls == ["sid-1"]
        assert dropped == ["sid-1"], (
            "a successful destroy must forget the baseline manifest"
        )

        mgr.destroy_result = False
        mgr.sessions.pop("sid-1")
        _expect_refusal(
            404, ctx.mod.destroy_sandbox, "sid-1",
            current_user={"sub": "alice"},
        )

        registered = ctx.mod.router.registered
        assert registered[-1] == ("DELETE", "/{session_id}"), (
            "the destructive single-segment catch-all must stay the last "
            f"registered route, order ends with {registered[-3:]}"
        )
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
