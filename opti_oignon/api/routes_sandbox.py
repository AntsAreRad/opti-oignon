#!/usr/bin/env python3
"""
API routes for Sandbox management -- S73/S116.

Provides endpoints for creating, managing, and executing tools
within sandboxed environments. All filesystem operations executed
through these endpoints run in isolation (bwrap or tempdir fallback).

S116: Copy-out + human approval workflow endpoints:
- preview: read file content from sandbox for display
- download: download a single approved file as binary
- approve: approve specific files for copy-out
- reject: reject all files, preventing copy-out
- copy-out: batch copy approved files to host
- approval info and audit trail
"""

import logging
import os

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .deps import (
    FILE_TOOLS_AVAILABLE,
    SANDBOX_AVAILABLE,
    sandbox_manager,
)
from .schemas import (
    HostBrowseEntry,
    HostBrowseResponse,
    QuickSandboxSessionInfo,
    # S117
    QuickSandboxStatusResponse,
    QuickSandboxToggleRequest,
    QuickSandboxTTLRequest,
    SandboxApplyEntry,
    SandboxApplyRefusedEntry,
    SandboxApplyRequest,
    SandboxApplyResponse,
    SandboxApprovalAuditEntry,
    SandboxApprovalAuditResponse,
    SandboxApprovalInfoResponse,
    SandboxApproveRequest,
    SandboxApproveResponse,
    SandboxAuditEntry,
    SandboxAuditResponse,
    SandboxBindingResponse,
    SandboxBindRequest,
    SandboxCloneRequest,
    SandboxCloneResponse,
    SandboxConfirmDegradedResponse,
    SandboxConfirmDeletionsRefused,
    SandboxConfirmDeletionsRequest,
    SandboxConfirmDeletionsResponse,
    SandboxCopyOutEntry,
    SandboxCopyOutResponse,
    SandboxCreateRequest,
    SandboxCreateResponse,
    SandboxDestroyResponse,
    # S212 (Bloc 3)
    SandboxDiffEntry,
    SandboxDiffResponse,
    SandboxExecuteRequest,
    SandboxExecuteResponse,
    SandboxFileEntry,
    SandboxFilesResponse,
    SandboxInjectRequest,
    SandboxInjectResponse,
    # S213 (Bloc 4)
    SandboxNetworkToggleRequest,
    SandboxNetworkToggleResponse,
    # S116
    SandboxPreviewResponse,
    SandboxProvisionRefusedLine,
    SandboxProvisionRequest,
    SandboxProvisionResponse,
    SandboxRejectResponse,
    SandboxSessionInfo,
    SandboxStatusResponse,
    # S210 (Bloc 1)
    SandboxStopResponse,
    # S211 (Bloc 2)
    SandboxUploadRefused,
    SandboxUploadResponse,
)

# S210 (Bloc 1): the conversation <-> workspace binding store; guarded so the
# router still loads if the module is absent in a partial build.
try:
    from opti_oignon import sandbox_workspace as _ws
    WORKSPACE_BINDING_AVAILABLE = True
except ImportError:
    _ws = None
    WORKSPACE_BINDING_AVAILABLE = False

# S213 (Bloc 4): the binding-layer egress gate; guarded the same way. The
# fail-secure direction is REFUSAL: with the module absent, network_allowed
# surfaces False on /status and the network/provision routes answer 503 --
# an unavailable gate never permits.
try:
    from opti_oignon import sandbox_egress as _eg
    EGRESS_AVAILABLE = True
except ImportError:
    _eg = None
    EGRESS_AVAILABLE = False

# S215: emergency-stop admission guard (a stopped system refuses honestly)
try:
    from opti_oignon import emergency_stop as _emergency_stop
except Exception:
    _emergency_stop = None

# S210: the disk soft-quota refusal raised by the inject paths.
try:
    from opti_oignon.sandbox_manager import (
        WorkspaceQuotaExceeded as _QuotaError,
    )
except ImportError:
    class _QuotaError(Exception):  # type: ignore[no-redef]
        """Placeholder when the manager module is absent."""

# S210: effective_user_id isolation pattern for workspace ownership; the
# fallback keeps the single-user posture when the module is absent.
try:
    from opti_oignon.user_isolation import effective_user_id as _effective_user_id
except ImportError:
    def _effective_user_id(user_id, single_user_mode: bool = True) -> str:  # type: ignore[misc]
        return user_id or "local"

# S117: Quick sandbox imports
try:
    from opti_oignon.quick_sandbox import (
        QUICK_SANDBOX_AVAILABLE,
    )
    from opti_oignon.quick_sandbox import (
        quick_sandbox_manager as _qs_manager,
    )
except ImportError:
    _qs_manager = None
    QUICK_SANDBOX_AVAILABLE = False

logger = logging.getLogger(__name__)

# S136 audit fix: require authentication for all endpoints
try:
    from .routes_auth import _get_current_user
    _auth_dep = [Depends(_get_current_user)]
except ImportError:
    _auth_dep = []

    def _get_current_user() -> dict:  # type: ignore[misc]
        return {"sub": None}


def _current_uid(user: dict) -> str:
    """The effective owner id of the calling user (S210)."""
    return _effective_user_id((user or {}).get("sub"))

router = APIRouter(prefix="/api/sandbox", tags=["sandbox"], dependencies=_auth_dep)

# Default export directory for copy-out
_DEFAULT_EXPORT_DIR = os.path.join("data", "sandbox_exports")

# S136 audit fix: restrict copy-out destination to the data directory.
# Without this, an API caller could write to arbitrary host paths
# (e.g. /etc/cron.d/, ~/.ssh/) which breaks the sandbox security model.
_ALLOWED_EXPORT_ROOT = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data")
)


def _validate_dest_dir(dest: str) -> str:
    """Validate that dest_dir resolves within the allowed export root.

    Returns the validated, real path. Raises HTTPException if the path
    escapes the allowed directory.
    """
    resolved = os.path.realpath(dest)
    if (
        not resolved.startswith(_ALLOWED_EXPORT_ROOT + os.sep)
        and resolved != _ALLOWED_EXPORT_ROOT
    ):
        raise HTTPException(
            status_code=403,
            detail=(
                f"Destination directory must be within the data/ directory. "
                f"Resolved path '{resolved}' is outside '{_ALLOWED_EXPORT_ROOT}'. "
                f"This restriction prevents sandbox copy-out to arbitrary host paths."
            ),
        )
    return resolved


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _require_sandbox():
    """Raise 503 if sandbox is not available."""
    if not SANDBOX_AVAILABLE or sandbox_manager is None:
        raise HTTPException(status_code=503, detail="Sandbox not available")


# ---------------------------------------------------------------------------
# System-level endpoints (no session_id in path)
# ---------------------------------------------------------------------------

@router.get("/status", response_model=SandboxStatusResponse)
def get_sandbox_status() -> dict:
    """Get overall sandbox system status."""
    if not SANDBOX_AVAILABLE or sandbox_manager is None:
        return SandboxStatusResponse(available=False)

    # S213: the live egress-gate answer (fail-secure False when the gate is
    # absent or errors) plus the configured caps for the settings strip.
    network_allowed = False
    if EGRESS_AVAILABLE and _eg is not None:
        try:
            network_allowed = bool(_eg.network_allowed())
        except Exception:
            network_allowed = False
    cfg = sandbox_manager.config
    return SandboxStatusResponse(
        available=True,
        enabled=sandbox_manager.config.enabled,
        isolation_backend=sandbox_manager.isolation_backend.value,
        bwrap_available=sandbox_manager.bwrap_available,
        degraded_mode=sandbox_manager.degraded_mode,
        degraded_confirmed=sandbox_manager.degraded_confirmed,
        active_sessions=sandbox_manager.active_session_count,
        max_sessions=sandbox_manager.config.max_concurrent_sessions,
        network_allowed=network_allowed,
        command_timeout_default=cfg.command_timeout,
        limit_memory_bytes=cfg.limit_memory_bytes,
        limit_nproc=cfg.limit_nproc,
        limit_cpu_seconds=cfg.limit_cpu_seconds,
        disk_soft_limit_bytes=cfg.disk_soft_limit_bytes,
    )


@router.post("/confirm-degraded", response_model=SandboxConfirmDegradedResponse)
def confirm_degraded_mode() -> dict:
    """Confirm willingness to run in degraded (tempdir) mode.

    Must be called before creating sandboxes when bwrap is unavailable
    and require_degraded_confirmation is True.
    """
    _require_sandbox()

    warning = sandbox_manager.confirm_degraded_mode()
    return SandboxConfirmDegradedResponse(
        confirmed=True,
        warning=warning,
    )


@router.post("/create", response_model=SandboxCreateResponse)
def create_sandbox(
    request: SandboxCreateRequest,
    current_user: dict = Depends(_get_current_user),
) -> dict:
    """Create a new sandbox session (workspace)."""
    if _emergency_stop is not None:
        _emergency_stop.guard_http()  # S215: refused, not hung
    _require_sandbox()

    try:
        session = sandbox_manager.create_sandbox(
            session_id=request.session_id or None,
            allow_degraded=request.allow_degraded,
            label=request.label,
            owner_user_id=_current_uid(current_user),
            timeout_override=request.timeout,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return SandboxCreateResponse(
        session_id=session.session_id,
        workspace_path=session.workspace_path,
        isolation_backend=session.isolation_backend.value,
        degraded=sandbox_manager.degraded_mode,
        label=session.label,
    )


@router.post("/inject", response_model=SandboxInjectResponse)
def inject_files(request: SandboxInjectRequest) -> dict:
    """Inject files from the host into a sandbox."""
    _require_sandbox()

    try:
        injected = sandbox_manager.inject_files(
            request.session_id, request.file_paths
        )
    except _QuotaError as exc:
        # S210: the disk soft quota refuses the copy-in (413), never kills
        # the workspace.
        raise HTTPException(status_code=413, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return SandboxInjectResponse(
        session_id=request.session_id,
        injected_count=len(injected),
        injected_paths=injected,
    )


@router.post("/execute", response_model=SandboxExecuteResponse)
def execute_sandbox_tool(request: SandboxExecuteRequest) -> dict:
    """Execute a tool within a sandbox session.

    Supported tool_name values: bash, view, create_file, str_replace.
    Arguments vary per tool -- see tool definitions.
    """
    if _emergency_stop is not None:
        _emergency_stop.guard_http()  # S215: refused, not hung
    _require_sandbox()
    if not FILE_TOOLS_AVAILABLE:
        raise HTTPException(
            status_code=503, detail="File tools not available"
        )

    from opti_oignon.file_tools import (
        _handle_sandbox_bash,
        _handle_sandbox_create_file,
        _handle_sandbox_str_replace,
        _handle_sandbox_view,
    )

    tool_map = {
        "bash": _handle_sandbox_bash,
        "sandbox_bash": _handle_sandbox_bash,
        "view": _handle_sandbox_view,
        "sandbox_view": _handle_sandbox_view,
        "create_file": _handle_sandbox_create_file,
        "sandbox_create_file": _handle_sandbox_create_file,
        "str_replace": _handle_sandbox_str_replace,
        "sandbox_str_replace": _handle_sandbox_str_replace,
    }

    handler = tool_map.get(request.tool_name)
    if handler is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown tool: {request.tool_name}. "
                f"Available: {list(tool_map.keys())}"
            ),
        )

    try:
        args = dict(request.arguments)
        args["session_id"] = request.session_id
        args["_sandbox_manager"] = sandbox_manager
        result_str = handler(**args)
    except TypeError as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid arguments: {exc}"
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # Parse result to detect blocked/timeout status
    blocked = "BLOCKED" in result_str
    timed_out = "TIMEOUT" in result_str
    block_reason = ""
    if blocked and "Reason:" in result_str:
        lines = result_str.split("\n")
        for line in lines:
            if line.startswith("Reason:"):
                block_reason = line[len("Reason:"):].strip()
                break

    return SandboxExecuteResponse(
        session_id=request.session_id,
        tool_name=request.tool_name,
        result=result_str,
        blocked=blocked,
        block_reason=block_reason,
        timed_out=timed_out,
        isolation_backend=sandbox_manager.isolation_backend.value,
    )


@router.get("/sessions", response_model=list[SandboxSessionInfo])
def list_sessions() -> list:
    """List all sandbox sessions.

    S213: each row is enriched with ``has_cloned_baseline`` from the
    manifests store (guarded; absent store means False) so the settings
    strip can sharpen the exfiltration warning when host files were
    cloned in.
    """
    if not SANDBOX_AVAILABLE or sandbox_manager is None:
        return []

    rows = sandbox_manager.list_sessions()
    if WORKSPACE_BINDING_AVAILABLE and _ws is not None:
        try:
            manifests = _ws.get_workspace_manifests()
            for row in rows:
                sid = row.get("session_id", "")
                row["has_cloned_baseline"] = bool(
                    sid and manifests.get_cloned_root(sid)
                )
        except Exception:  # pragma: no cover - enrichment must not break
            logger.exception("has_cloned_baseline enrichment failed")

    # bound_conversation_id (frontend contract): the explicit workspace
    # binding takes priority; otherwise fall back to the conversation that
    # created the quick-sandbox session (auto-created sessions are keyed by
    # conversation_id but never explicitly bound). None means detached.
    for row in rows:
        sid = row.get("session_id", "")
        bound: str | None = None
        if WORKSPACE_BINDING_AVAILABLE and _ws is not None:
            try:
                bound = _ws.get_conversation_for(sid)
            except Exception:  # pragma: no cover - enrichment must not break
                bound = None
        if (
            bound is None
            and QUICK_SANDBOX_AVAILABLE
            and _qs_manager is not None
        ):
            try:
                qs = _qs_manager.get_session(sid)
                bound = qs.conversation_id if qs is not None else None
            except Exception:  # pragma: no cover - enrichment must not break
                bound = None
        row["bound_conversation_id"] = bound

    return [SandboxSessionInfo(**s) for s in rows]


@router.get("/audit", response_model=SandboxAuditResponse)
def get_audit_log(session_id: str | None = None, limit: int = 100) -> dict:
    """Get sandbox audit log entries.

    If session_id is provided, returns only that session's entries.
    Otherwise returns the most recent entries across all sessions.
    """
    if not SANDBOX_AVAILABLE or sandbox_manager is None:
        return SandboxAuditResponse()

    if session_id:
        raw = sandbox_manager.audit.get_session_log(session_id)
    else:
        raw = sandbox_manager.audit.get_all_logs(limit=limit)

    entries = [
        SandboxAuditEntry(
            id=e.get("id", 0),
            session_id=e.get("session_id", ""),
            timestamp=e.get("timestamp", 0.0),
            command=e.get("command", ""),
            return_code=e.get("return_code"),
            blocked=bool(e.get("blocked", 0)),
            block_reason=e.get("block_reason", ""),
            timed_out=bool(e.get("timed_out", 0)),
            stdout_len=e.get("stdout_len", 0),
            stderr_len=e.get("stderr_len", 0),
            isolation_backend=e.get("isolation_backend", ""),
        )
        for e in raw
    ]
    return SandboxAuditResponse(entries=entries, count=len(entries))


# ---------------------------------------------------------------------------
# Session-level endpoints: file listing with approval status
# ---------------------------------------------------------------------------

@router.get("/files/{session_id}", response_model=SandboxFilesResponse)
def list_sandbox_files(session_id: str) -> dict:
    """List files in a sandbox workspace with approval status."""
    _require_sandbox()

    try:
        files = sandbox_manager.extract_files(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    approval_info = sandbox_manager.get_approval_info(session_id)

    entries = [
        SandboxFileEntry(
            path=f["path"],
            size=f["size"],
            modified=f["modified"],
            approved=sandbox_manager.is_file_approved(session_id, f["path"]),
        )
        for f in files
    ]
    return SandboxFilesResponse(
        session_id=session_id,
        files=entries,
        count=len(entries),
        approval_state=approval_info.get("approval_state", "pending"),
    )


# ---------------------------------------------------------------------------
# S116: Preview endpoint
# ---------------------------------------------------------------------------

@router.get(
    "/preview/{session_id}/{path:path}",
    response_model=SandboxPreviewResponse,
)
def preview_sandbox_file(session_id: str, path: str) -> dict:
    """Preview a file from a sandbox (text content, capped at 64KB).

    Binary files return a hex preview capped at 2KB.
    """
    _require_sandbox()

    try:
        preview = sandbox_manager.preview_file(session_id, path)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return SandboxPreviewResponse(
        session_id=session_id,
        path=preview["path"],
        content=preview["content"],
        size=preview["size"],
        truncated=preview["truncated"],
        is_binary=preview["is_binary"],
    )


# ---------------------------------------------------------------------------
# S116: Download endpoint (binary, requires approval)
# ---------------------------------------------------------------------------

@router.get("/download/{session_id}/{path:path}")
def download_sandbox_file(session_id: str, path: str) -> FileResponse:
    """Download a single approved file from the sandbox as binary.

    Returns 403 if the file has not been approved for copy-out.
    """
    _require_sandbox()

    if not sandbox_manager.is_file_approved(session_id, path):
        raise HTTPException(
            status_code=403,
            detail=(
                f"File not approved for download: {path}. "
                f"Approve it first via POST /api/sandbox/{{session_id}}/approve"
            ),
        )

    try:
        workspace = sandbox_manager._get_active_workspace(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    from opti_oignon.sandbox_manager import validate_sandbox_path

    valid, resolved, err = validate_sandbox_path(workspace, path)
    if not valid:
        raise HTTPException(status_code=400, detail=f"Invalid path: {err}")
    if not os.path.isfile(resolved):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    return FileResponse(
        path=resolved,
        filename=os.path.basename(path),
        media_type="application/octet-stream",
    )


# ---------------------------------------------------------------------------
# S116: Approve endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/{session_id}/approve",
    response_model=SandboxApproveResponse,
)
def approve_sandbox_files(session_id: str, request: SandboxApproveRequest) -> dict:
    """Approve specific files for copy-out from the sandbox.

    No auto-approve: only the paths listed in the request are approved.
    Approval is additive (calling twice adds more paths).
    """
    _require_sandbox()

    try:
        approved = sandbox_manager.approve_files(session_id, request.paths)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    info = sandbox_manager.get_approval_info(session_id)

    return SandboxApproveResponse(
        session_id=session_id,
        approved_paths=approved,
        approved_count=len(approved),
        approval_state=info["approval_state"],
    )


# ---------------------------------------------------------------------------
# S116: Copy-out endpoint (batch copy approved files to host)
# ---------------------------------------------------------------------------

@router.post(
    "/{session_id}/copy-out",
    response_model=SandboxCopyOutResponse,
)
def copy_out_sandbox_files(session_id: str, request: SandboxApproveRequest) -> dict:
    """Copy approved files from the sandbox to the host filesystem.

    S136 audit fix: dest_dir is validated to stay within the data/
    directory. Copies only files that have been approved.
    """
    _require_sandbox()

    dest = request.dest_dir or _DEFAULT_EXPORT_DIR
    dest = _validate_dest_dir(dest)

    try:
        results = sandbox_manager.copy_out_batch(
            session_id, request.paths, dest,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    entries = [
        SandboxCopyOutEntry(
            src_path=r["src_path"],
            dest_path=r["dest_path"],
            size=r["size"],
        )
        for r in results
    ]

    return SandboxCopyOutResponse(
        session_id=session_id,
        copied=entries,
        copied_count=len(entries),
        dest_dir=os.path.realpath(dest),
    )


# ---------------------------------------------------------------------------
# S116: Reject endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/{session_id}/reject",
    response_model=SandboxRejectResponse,
)
def reject_sandbox_files(session_id: str) -> dict:
    """Reject all files in a sandbox, preventing any copy-out."""
    _require_sandbox()

    try:
        sandbox_manager.reject_files(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return SandboxRejectResponse(
        session_id=session_id,
        rejected=True,
        approval_state="rejected",
    )


# ---------------------------------------------------------------------------
# S116: Approval info endpoint
# ---------------------------------------------------------------------------

@router.get(
    "/{session_id}/approval",
    response_model=SandboxApprovalInfoResponse,
)
def get_approval_info(session_id: str) -> dict:
    """Get approval state summary for a sandbox session."""
    _require_sandbox()

    info = sandbox_manager.get_approval_info(session_id)

    return SandboxApprovalInfoResponse(
        session_id=session_id,
        approval_state=info["approval_state"],
        approved_paths=info["approved_paths"],
        approved_at=info["approved_at"],
    )


# ---------------------------------------------------------------------------
# S116: Approval audit endpoint
# ---------------------------------------------------------------------------

@router.get(
    "/{session_id}/approval-audit",
    response_model=SandboxApprovalAuditResponse,
)
def get_approval_audit(session_id: str) -> dict:
    """Get approval audit trail for a sandbox session."""
    _require_sandbox()

    raw = sandbox_manager.audit.get_approval_log(session_id)

    entries = [
        SandboxApprovalAuditEntry(
            id=e.get("id", 0),
            session_id=e.get("session_id", ""),
            timestamp=e.get("timestamp", 0.0),
            action=e.get("action", ""),
            paths=e.get("paths", ""),
            dest_dir=e.get("dest_dir", ""),
            detail=e.get("detail", ""),
        )
        for e in raw
    ]
    return SandboxApprovalAuditResponse(
        entries=entries,
        count=len(entries),
    )


# ---------------------------------------------------------------------------
# S117: Quick Sandbox endpoints
# ---------------------------------------------------------------------------

@router.get("/quick/status", response_model=QuickSandboxStatusResponse)
def get_quick_sandbox_status() -> dict:
    """Get quick sandbox status and configuration."""
    if not QUICK_SANDBOX_AVAILABLE or _qs_manager is None:
        return QuickSandboxStatusResponse(
            enabled=False,
            available=False,
        )
    status = _qs_manager.get_status()
    return QuickSandboxStatusResponse(**status)


@router.post("/quick/toggle", response_model=QuickSandboxStatusResponse)
def toggle_quick_sandbox(request: QuickSandboxToggleRequest) -> dict:
    """Enable or disable quick sandbox mode."""
    if not QUICK_SANDBOX_AVAILABLE or _qs_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Quick sandbox not available (missing dependencies)",
        )
    _qs_manager.enabled = request.enabled
    logger.info("Quick sandbox toggled: enabled=%s", request.enabled)
    status = _qs_manager.get_status()
    return QuickSandboxStatusResponse(**status)


@router.get(
    "/quick/sessions", response_model=list[QuickSandboxSessionInfo]
)
def list_quick_sandbox_sessions() -> list:
    """List active quick sandbox sessions."""
    if not QUICK_SANDBOX_AVAILABLE or _qs_manager is None:
        return []
    sessions = _qs_manager.list_sessions()
    return [QuickSandboxSessionInfo(**s) for s in sessions]


@router.post("/quick/cleanup")
def cleanup_quick_sandbox_sessions() -> dict:
    """Destroy all expired quick sandbox sessions."""
    if not QUICK_SANDBOX_AVAILABLE or _qs_manager is None:
        return {"destroyed": 0}
    count = _qs_manager.cleanup_expired()
    return {"destroyed": count}


@router.delete("/quick/{session_id}")
def destroy_quick_sandbox_session(session_id: str) -> dict:
    """Destroy a specific quick sandbox session."""
    if not QUICK_SANDBOX_AVAILABLE or _qs_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Quick sandbox not available",
        )
    destroyed = _qs_manager.destroy_session(session_id)
    if not destroyed:
        raise HTTPException(
            status_code=404,
            detail=f"Quick sandbox session not found: {session_id}",
        )
    return {"session_id": session_id, "destroyed": True}


@router.post(
    "/quick/{session_id}/ttl", response_model=QuickSandboxSessionInfo
)
def set_quick_sandbox_ttl(
    session_id: str, request: QuickSandboxTTLRequest
) -> QuickSandboxSessionInfo:
    """Change a quick sandbox session's auto-destroy (inactivity) timeout.

    The activity clock is reset so the new window starts now.
    """
    if not QUICK_SANDBOX_AVAILABLE or _qs_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Quick sandbox not available",
        )
    try:
        updated = _qs_manager.set_session_auto_destroy(
            session_id, request.auto_destroy_minutes
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not updated:
        raise HTTPException(
            status_code=404,
            detail=f"Quick sandbox session not found: {session_id}",
        )
    for s in _qs_manager.list_sessions():
        if s["session_id"] == session_id:
            return QuickSandboxSessionInfo(**s)
    raise HTTPException(
        status_code=404,
        detail=f"Quick sandbox session not found: {session_id}",
    )


# ---------------------------------------------------------------------------
# Parametric catch-all (MUST be LAST to avoid shadowing literal routes)
# ---------------------------------------------------------------------------

@router.post("/{session_id}/stop", response_model=SandboxStopResponse)
def stop_sandbox_command(
    session_id: str,
    current_user: dict = Depends(_get_current_user),
) -> dict:
    """SIGKILL the workspace's running command; keep the workspace (S210).

    Honest semantics: 404 for an unknown session, 403 for a foreign owner,
    and stopped=False (200) when nothing was running -- a no-op, never an
    error that leaks state. Files persist for inspection.
    """
    _require_sandbox()

    session = sandbox_manager.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}",
        )
    if session.owner_user_id != _current_uid(current_user):
        raise HTTPException(
            status_code=403,
            detail="Workspace is not owned by this user",
        )

    try:
        stopped = sandbox_manager.stop_command(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return SandboxStopResponse(session_id=session_id, stopped=stopped)


# ---------------------------------------------------------------------------
# Conversation binding (S210, Bloc 1)
# ---------------------------------------------------------------------------

def _require_bindings():
    """Raise 503 if the binding store is not available."""
    if not WORKSPACE_BINDING_AVAILABLE or _ws is None:
        raise HTTPException(
            status_code=503, detail="Workspace binding not available"
        )
    return _ws.get_workspace_bindings()


@router.post("/bind", response_model=SandboxBindingResponse)
def bind_conversation(
    request: SandboxBindRequest,
    current_user: dict = Depends(_get_current_user),
) -> dict:
    """Bind a conversation to a workspace (rebind allowed, audited).

    409 when the workspace is held by another conversation (at most one
    active conversation per workspace, for an unambiguous audit trail);
    403 on owner mismatch; 404 on an unknown workspace.
    """
    _require_sandbox()
    bindings = _require_bindings()

    try:
        bindings.bind(
            request.conversation_id,
            request.session_id,
            user_id=_current_uid(current_user),
            manager=sandbox_manager,
        )
    except _ws.WorkspaceNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except _ws.WorkspaceOwnerMismatch as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except _ws.WorkspaceAlreadyBound as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except _ws.WorkspaceBindingError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return SandboxBindingResponse(
        conversation_id=request.conversation_id,
        session_id=request.session_id,
        bound=True,
    )


@router.delete("/bind/{conversation_id}", response_model=SandboxBindingResponse)
def unbind_conversation(
    conversation_id: str,
    current_user: dict = Depends(_get_current_user),
) -> dict:
    """Release a conversation's workspace binding (no-op when unbound)."""
    _require_sandbox()
    bindings = _require_bindings()

    try:
        bindings.unbind(
            conversation_id,
            user_id=_current_uid(current_user),
            manager=sandbox_manager,
        )
    except _ws.WorkspaceOwnerMismatch as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    return SandboxBindingResponse(
        conversation_id=conversation_id,
        session_id=None,
        bound=False,
    )


@router.get("/bind/{conversation_id}", response_model=SandboxBindingResponse)
def get_conversation_binding(conversation_id: str) -> dict:
    """The workspace currently bound to a conversation, if any."""
    _require_sandbox()
    bindings = _require_bindings()

    sid = bindings.get_sandbox_for(conversation_id, manager=sandbox_manager)
    return SandboxBindingResponse(
        conversation_id=conversation_id,
        session_id=sid,
        bound=sid is not None,
    )


# ---------------------------------------------------------------------------
# Copy-in (S211, Bloc 2): drag-and-drop upload, allowlisted host browse,
# symlink-safe host clone. All three are EXPLICIT user actions through the
# manager UI -- the model can trigger none of them (S73/S74). The baseline
# manifest (spec section 6.1, the seam Bloc 3's diff consumes) is recorded
# here after a successful manager operation; the manager computes the hashes
# on the fly and never imports the manifest store (no cycle).
# ---------------------------------------------------------------------------

def _require_owned_session(session_id: str, current_user: dict):
    """404 on an unknown session, 403 on a foreign owner (the S210 codes)."""
    session = sandbox_manager.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}",
        )
    if session.owner_user_id != _current_uid(current_user):
        raise HTTPException(
            status_code=403,
            detail="Workspace is not owned by this user",
        )
    return session


def _record_manifest(
    session_id: str,
    entries: dict,
    cloned_root: str | None = None,
    cloned_mount: str | None = None,
) -> int:
    """Record baseline-manifest entries; honest no-op when the store is
    absent (a partial build): copy-in still works, Bloc 3 then simply has
    no baseline -- the conservative failure mode."""
    if not entries:
        return 0
    if not WORKSPACE_BINDING_AVAILABLE or _ws is None:
        logger.warning(
            "Baseline manifest not recorded (store unavailable) for %s",
            session_id,
        )
        return 0
    try:
        manifests = _ws.get_workspace_manifests()
        manifests.record(
            session_id,
            entries,
            cloned_root=cloned_root,
            cloned_mount=cloned_mount,
        )
        return len(entries)
    except Exception:  # pragma: no cover - recording must not break copy-in
        logger.exception("Baseline manifest recording failed for %s", session_id)
        return 0


@router.post("/{session_id}/upload", response_model=SandboxUploadResponse)
def upload_workspace_files(
    session_id: str,
    files: list[UploadFile] = File(...),
    dest_subdir: str = Form(""),
    current_user: dict = Depends(_get_current_user),
) -> dict:
    """Multipart drag-and-drop copy-in (spec 5.1).

    No host filesystem path is ever enumerated or read by the server on
    this path -- the browser supplies the bytes. Sizes are summed BEFORE
    any write; an exceeded cap (request file count, per-file bytes, or the
    S210 workspace quota) refuses the WHOLE request with 413 and the
    workspace untouched. Individually invalid names and destination
    collisions are refused per file in the 200 response, never overwritten.
    """
    _require_sandbox()
    _require_owned_session(session_id, current_user)

    items = []
    for upload in files:
        try:
            upload.file.seek(0, os.SEEK_END)
            size = upload.file.tell()
            upload.file.seek(0)
        except OSError:
            raise HTTPException(
                status_code=400,
                detail=f"Unreadable upload stream: {upload.filename!r}",
            )
        items.append((upload.filename or "", upload.file, size))

    try:
        result = sandbox_manager.upload_files(
            session_id, items, dest_subdir=dest_subdir
        )
    except _QuotaError as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    except ValueError as exc:
        # The session was validated above; a remaining ValueError is an
        # invalid destination subdirectory.
        raise HTTPException(status_code=400, detail=str(exc))

    manifest_entries = {
        w["relative_path"]: w["sha256"] for w in result["written"]
    }
    manifest_files = _record_manifest(session_id, manifest_entries)

    return SandboxUploadResponse(
        session_id=session_id,
        uploaded_paths=[w["relative_path"] for w in result["written"]],
        refused=[
            SandboxUploadRefused(name=r["name"], reason=r["reason"])
            for r in result["refused"]
        ],
        uploaded_bytes=result["written_bytes"],
        manifest_files=manifest_files,
    )


@router.get("/host/browse", response_model=HostBrowseResponse)
def browse_host_directory(path: str | None = None) -> dict:
    """Allowlisted host directory listing (spec 5.2a).

    Confinement runs BEFORE any existence check: a path outside the
    allowlisted share roots answers 403 whether or not it exists (no
    existence leak); a missing or non-directory path INSIDE a root answers
    404. With no path, the roots themselves are listed. Auth rides the
    router dependency; every browse is audited by the manager.
    """
    _require_sandbox()

    try:
        listing = sandbox_manager.browse_host(path)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return HostBrowseResponse(
        path=listing["path"],
        roots=listing["roots"],
        entries=[HostBrowseEntry(**e) for e in listing["entries"]],
    )


@router.post("/{session_id}/clone", response_model=SandboxCloneResponse)
def clone_host_directory(
    session_id: str,
    request: SandboxCloneRequest,
    current_user: dict = Depends(_get_current_user),
) -> dict:
    """Clone an allowlisted host directory into the workspace (spec 5.2b).

    Symlink-safe (links skipped and counted, targets never exposed),
    specials skipped, caps and the S210 quota enforced by an exact
    pre-walk BEFORE any copy (413), destination collisions refused (409),
    sources outside the allowlisted roots refused (403). The baseline
    manifest (6.1) is recorded from the hashes computed during the copy;
    the cloned root is write-once -- Bloc 3's only implicit write-back
    target. Refusals are audited as host_clone_refused.
    """
    _require_sandbox()
    _require_owned_session(session_id, current_user)

    def _audit_refusal(detail: str) -> None:
        try:
            sandbox_manager.audit.log_approval(
                session_id,
                action="host_clone_refused",
                paths=[],
                detail=detail,
            )
        except Exception:  # pragma: no cover - audit must not mask the code
            logger.debug("clone refusal audit failed", exc_info=True)

    try:
        result = sandbox_manager.clone_directory(
            session_id, request.src_path, dest_subdir=request.dest_subdir
        )
    except PermissionError as exc:
        _audit_refusal(str(exc))
        raise HTTPException(status_code=403, detail=str(exc))
    except _QuotaError as exc:
        _audit_refusal(str(exc))
        raise HTTPException(status_code=413, detail=str(exc))
    except FileExistsError as exc:
        _audit_refusal(str(exc))
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        _audit_refusal(str(exc))
        # "Invalid destination" is a caller error on the workspace side
        # (400); any other ValueError is a missing/non-directory source
        # inside the allowlisted roots (404).
        code = 400 if "Invalid destination" in str(exc) else 404
        raise HTTPException(status_code=code, detail=str(exc))

    manifest_files = _record_manifest(
        session_id,
        result["manifest"],
        cloned_root=result["cloned_root"],
        cloned_mount=result["dest"],
    )

    return SandboxCloneResponse(
        session_id=session_id,
        dest=result["dest"],
        cloned_root=result["cloned_root"],
        copied_files=result["copied_files"],
        copied_bytes=result["copied_bytes"],
        skipped_symlinks=result["skipped_symlinks"],
        skipped_special=result["skipped_special"],
        manifest_files=manifest_files,
    )


# ---------------------------------------------------------------------------
# S212 (Bloc 3): diff-gated write-back -- diff, deletion confirmation, apply
# ---------------------------------------------------------------------------

def _require_workspace_module() -> None:
    """503 when the workspace module is absent (partial build).

    The diff and the apply ARE the feature here; unlike the S211 manifest
    recording (an honest no-op), they cannot degrade.
    """
    if not WORKSPACE_BINDING_AVAILABLE or _ws is None:
        raise HTTPException(
            status_code=503,
            detail="Workspace diff/apply not available in this build",
        )


def _compute_session_diff(session_id: str):
    """Run the workspace diff with the shared exception-to-code mapping."""
    try:
        return _ws.generate_workspace_diff(session_id, manager=sandbox_manager)
    except _ws.WorkspaceDiffBoundExceeded as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/{session_id}/diff", response_model=SandboxDiffResponse)
def get_workspace_diff(
    session_id: str,
    current_user: dict = Depends(_get_current_user),
) -> dict:
    """The live workspace classified against the baseline manifest (S212).

    Hash-driven added/modified/deleted against the recorded 6.1 baseline;
    matching hashes count as unchanged; symlinks and specials are skipped
    and counted (never followed -- following one would read host files
    into the review). No baseline: everything classifies "added",
    baseline_present is False, and there is no implicit write-back
    target. The diff_hash in the response is the review-integrity digest
    the apply request must echo. Exceeding the diff entry or depth bound
    refuses (413) -- the review presents the whole change set or nothing.
    Read-only: reads the workspace and the in-memory baseline, never the
    host tree, so it is not in the section 9 additionally-audited set.
    """
    _require_sandbox()
    _require_workspace_module()
    session = _require_owned_session(session_id, current_user)

    diff = _compute_session_diff(session_id)

    return SandboxDiffResponse(
        session_id=session_id,
        baseline_present=diff.baseline_present,
        cloned_root=diff.cloned_root,
        cloned_mount=diff.cloned_mount,
        entries=[SandboxDiffEntry(**c.to_dict()) for c in diff.entries],
        unchanged=diff.unchanged,
        skipped_symlinks=diff.skipped_symlinks,
        skipped_special=diff.skipped_special,
        diff_hash=diff.diff_hash,
        approved_paths=sorted(session.approved_paths),
        confirmed_deletions=sorted(session.confirmed_deletions),
    )


@router.post(
    "/{session_id}/confirm-deletions",
    response_model=SandboxConfirmDeletionsResponse,
)
def confirm_workspace_deletions(
    session_id: str,
    request: SandboxConfirmDeletionsRequest,
    current_user: dict = Depends(_get_current_user),
) -> dict:
    """Explicitly confirm deletions for apply-to-host (S212, 6.2).

    Distinct from approval by design: removing a host file requires its
    own confirmation and is never bundled into a blanket approve-all.
    Each requested path is validated against the CURRENT diff's deleted
    set; anything else (live files, unknown paths) is refused per path,
    honestly, in the 200 body. The confirmation is recorded and audited
    (deletion_confirm); the load-bearing enforcement stays in the apply
    writer, which deletes only confirmed AND currently-deleted paths.
    """
    _require_sandbox()
    _require_workspace_module()
    _require_owned_session(session_id, current_user)

    diff = _compute_session_diff(session_id)
    deleted_now = {c.path for c in diff.entries if c.kind == "deleted"}

    to_confirm: list[str] = []
    refused: list[SandboxConfirmDeletionsRefused] = []
    for path in request.paths:
        if not path:
            refused.append(SandboxConfirmDeletionsRefused(
                path=path, reason="empty path",
            ))
        elif path not in deleted_now:
            refused.append(SandboxConfirmDeletionsRefused(
                path=path,
                reason="not classified as deleted by the current diff",
            ))
        else:
            to_confirm.append(path)

    confirmed: list[str] = []
    if to_confirm:
        try:
            confirmed = sandbox_manager.confirm_deletions(
                session_id, to_confirm
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    return SandboxConfirmDeletionsResponse(
        session_id=session_id,
        confirmed=confirmed,
        refused=refused,
    )


@router.post("/{session_id}/apply", response_model=SandboxApplyResponse)
def apply_workspace_changes(
    session_id: str,
    request: SandboxApplyRequest,
    current_user: dict = Depends(_get_current_user),
) -> dict:
    """Write ONLY approved changes back to the host (S212, 6.2).

    The cycle's highest-risk route; a USER action through the manager UI
    -- the model can trigger nothing here (the dispatch invariant is
    unchanged). Target: the write-once cloned_root when present
    (re-validated against the current share roots; a conflicting explicit
    target is refused), else an explicit allowlisted target_dir, else 400
    -- never guess. The request must echo the reviewed diff_hash; any
    workspace drift since the review answers 409 (re-run the diff).
    Writes go through symlink-component validation and
    temp-file-plus-rename; deletions apply only from the
    separately-confirmed set. Per-file results -- applied, deleted,
    refused -- are returned honestly and audited per file (apply_write /
    apply_delete / apply_refused) with a closing apply_summary.
    """
    _require_sandbox()
    _require_workspace_module()
    _require_owned_session(session_id, current_user)

    try:
        result = _ws.apply_workspace_changes(
            session_id,
            request.diff_hash,
            manager=sandbox_manager,
            target_dir=request.target_dir,
        )
    except _ws.WorkspaceReviewDrift as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except _ws.WorkspaceApplyTargetError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except _ws.WorkspaceDiffBoundExceeded as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return SandboxApplyResponse(
        session_id=session_id,
        target=result["target"],
        applied=[SandboxApplyEntry(**e) for e in result["applied"]],
        deleted=[SandboxApplyEntry(**e) for e in result["deleted"]],
        refused=[SandboxApplyRefusedEntry(**e) for e in result["refused"]],
        skipped_unapproved=result["skipped_unapproved"],
        skipped_unconfirmed=result["skipped_unconfirmed"],
        diff_hash=result["diff_hash"],
    )


# -- S213 (Bloc 4): the per-workspace network gate and the provision run --


def _require_egress() -> None:
    """503 when the egress gate module is absent (partial build).

    The gate IS the feature here (the _require_workspace_module
    precedent): an absent gate must refuse the capability, never default
    it open.
    """
    if not EGRESS_AVAILABLE or _eg is None:
        raise HTTPException(
            status_code=503,
            detail="Sandbox egress gate not available in this build",
        )


@router.post(
    "/{session_id}/network", response_model=SandboxNetworkToggleResponse
)
def toggle_workspace_network(
    session_id: str,
    request: SandboxNetworkToggleRequest,
    current_user: dict = Depends(_get_current_user),
) -> dict:
    """Flip the per-workspace network flag (S213, spec 8.3).

    An explicit USER action -- never a config default, never
    model-triggerable (the dispatch invariant is unchanged; no tool
    surface reaches this). Enabling is Daily-only: the binding-layer gate
    refuses with the same 403 discipline the security-mode middleware
    uses, and an unset or unknown mode is treated as Bulbe (fail-secure).
    Disabling is permitted in any mode. Both directions are audited (the
    per-session log and the hash-chain).
    """
    _require_sandbox()
    _require_egress()
    _require_owned_session(session_id, current_user)
    try:
        new_state = sandbox_manager.set_network_enabled(
            session_id,
            bool(request.enabled),
            actor=_current_uid(current_user),
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return SandboxNetworkToggleResponse(
        session_id=session_id,
        network_enabled=new_state,
    )


@router.post(
    "/{session_id}/provision", response_model=SandboxProvisionResponse
)
def provision_workspace(
    session_id: str,
    request: SandboxProvisionRequest,
    current_user: dict = Depends(_get_current_user),
) -> dict:
    """Run the provision phase: the one scoped egress (S213, spec 8.4).

    A USER action. Ladder: 404/403 owner, 503 partial build, 403 under
    Bulbe (the binding-layer gate; unknown mode is Bulbe), 409 when the
    per-workspace network flag is off (a state precondition, the S211
    collision precedent), 400 on a bad path or a requirements set that is
    not exact-and-hash-pinned (per-line refusals listed honestly; nothing
    installs on a partial validation). The command is built server-side
    only; the run goes through the manager's single gated provision seam,
    where bwrap-absence and every other refusal come back blocked and
    audited -- surfaced honestly in the 200 body, the execute posture.
    """
    if _emergency_stop is not None:
        _emergency_stop.guard_http()  # S215: refused, not hung
    _require_sandbox()
    _require_egress()
    session = _require_owned_session(session_id, current_user)

    def _audit_refused(detail: str) -> None:
        try:
            sandbox_manager.audit.log_approval(
                session_id, action="provision_refused", detail=detail
            )
        except Exception:  # pragma: no cover - audit must not mask the code
            logger.exception("provision_refused audit failed")

    try:
        if not _eg.network_allowed():
            _audit_refused("route gate: mode is not daily (fail-secure)")
            raise HTTPException(
                status_code=403,
                detail=(
                    "Sandbox network egress is disabled in Bulbe mode: the "
                    "provision run is a Daily-only capability. Switch to "
                    "Daily mode."
                ),
            )
    except HTTPException:
        raise
    except Exception:
        _audit_refused("route gate unreadable (fail-secure)")
        raise HTTPException(
            status_code=403,
            detail=(
                "Sandbox network egress refused: the security mode could "
                "not be determined (treated as Bulbe, fail-secure)."
            ),
        )

    if not session.network_enabled:
        _audit_refused("network flag is off for this workspace")
        raise HTTPException(
            status_code=409,
            detail=(
                "Network is not enabled for this workspace. Enable it "
                "explicitly first (per-workspace flag, default off)."
            ),
        )

    for label, rel in (
        ("requirements_path", request.requirements_path),
        ("venv_dir", request.venv_dir),
    ):
        reason = _eg.refuse_rel_path(rel)
        if reason is not None:
            _audit_refused(f"{label} refused: {reason}")
            raise HTTPException(
                status_code=400, detail=f"{label}: {reason}"
            )

    workspace = sandbox_manager.get_active_workspace_path(session_id)
    req_host_path = os.path.join(workspace, request.requirements_path)
    if not os.path.isfile(req_host_path):
        _audit_refused("requirements file not found in workspace")
        raise HTTPException(
            status_code=400,
            detail=(
                "requirements_path does not name a file in the workspace: "
                f"{request.requirements_path}"
            ),
        )
    try:
        with open(req_host_path, encoding="utf-8", errors="replace") as fh:
            req_text = fh.read()
    except OSError as exc:
        _audit_refused(f"requirements file unreadable: {exc}")
        raise HTTPException(
            status_code=400, detail="requirements file could not be read"
        )

    accepted, refused_lines = _eg.validate_requirements_text(req_text)
    if refused_lines or not accepted:
        _audit_refused(
            f"requirements validation refused {len(refused_lines)} line(s); "
            "nothing installed"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Requirements set refused: every line must be an exact "
                    "name==version requirement carrying --hash=sha256: "
                    "pins; option lines are never accepted. Nothing was "
                    "installed."
                ),
                "refused": [
                    SandboxProvisionRefusedLine(**r).model_dump()
                    for r in refused_lines
                ],
            },
        )

    command = _eg.build_provision_command(
        request.requirements_path, request.venv_dir
    )
    result = sandbox_manager.execute_provision_command(session_id, command)

    sandbox_manager.audit.log_approval(
        session_id,
        action="provision_run",
        paths=[request.requirements_path, request.venv_dir],
        detail=(
            f"rc={result.return_code} blocked={result.blocked} "
            f"timed_out={result.timed_out} packages={len(accepted)} "
            f"actor={_current_uid(current_user)}"
        ),
    )

    return SandboxProvisionResponse(
        session_id=session_id,
        command=command,
        return_code=result.return_code,
        blocked=result.blocked,
        block_reason=result.block_reason,
        timed_out=result.timed_out,
        isolation_backend=result.isolation_backend,
        stdout_tail=(result.stdout or "")[-2000:],
        stderr_tail=(result.stderr or "")[-2000:],
        accepted_requirements=accepted,
    )


# DELETE /{session_id} stays the LAST registered route: the single-segment
# wildcard must not shadow the specific paths above (the catch-all ordering
# contract pinned by tests/test_sandbox_api.py::TestS116RoutesExist).
@router.delete("/{session_id}", response_model=SandboxDestroyResponse)
def destroy_sandbox(
    session_id: str,
    current_user: dict = Depends(_get_current_user),
) -> dict:
    """Destroy a sandbox session and remove all its files.

    S210: ownership is checked (403 on mismatch; unchanged behaviour in
    single-user mode where everything is "local"). A second destroy of the
    same id answers 404 -- the effect is idempotent and the code honest.
    """
    _require_sandbox()

    session = sandbox_manager.get_session(session_id)
    if session is not None and session.owner_user_id != _current_uid(
        current_user
    ):
        raise HTTPException(
            status_code=403,
            detail="Workspace is not owned by this user",
        )

    destroyed = sandbox_manager.destroy_sandbox(session_id)
    if not destroyed:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}",
        )

    # S211: forget the baseline manifest so a reused session id never
    # inherits a stale baseline (guarded; absence must not break destroy).
    if WORKSPACE_BINDING_AVAILABLE and _ws is not None:
        try:
            _ws.get_workspace_manifests().drop(session_id)
        except Exception:  # pragma: no cover - cleanup must not break destroy
            logger.debug("manifest drop failed", exc_info=True)

    return SandboxDestroyResponse(
        session_id=session_id,
        destroyed=True,
    )
