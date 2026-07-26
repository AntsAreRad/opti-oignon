#!/usr/bin/env python3
"""Conversation <-> workspace binding for the Sandbox Workspace cycle.

Bloc 1 of SANDBOX_WORKSPACE_SPEC.md (section 4.1): a workspace is a named,
user-owned sandbox with a stable id that outlives a single run. This module
owns the ``conversation_id -> sandbox_id`` binding so a whole conversation can
run against one workspace: the chat agent, when a workspace is bound, injects
that workspace's ``SandboxToolSession`` into ``dispatch.dispatch_tool_call``
instead of creating and destroying a session per run (this is the ATL-02
remediation: the sandboxed tools become reachable from ``/api/agent/run``).

Design decisions:

- The store is IN-MEMORY and thread-safe. Sandbox sessions themselves live in
  the manager's in-memory map and die with the process, so persisting a
  binding would outlive its target and lie at startup. Revisit only if
  persistent sessions ever land.
- The binding is EXPLICIT: a conversation has no workspace until the user
  creates or attaches one; there is no auto-create on first tool use.
- At most ONE active conversation per workspace (unambiguous audit trail);
  binding a workspace already bound to another conversation is refused.
  Rebinding a conversation to a different workspace atomically releases its
  previous binding.
- Owner scoping follows the ``effective_user_id`` isolation pattern: the
  workspace records its owner at create; bind/unbind refuse on a mismatch.
- Single source of truth with a write-through mirror: this store is the only
  mutation point; it writes the binding through to the session object
  (``SandboxManager.set_binding``) so ``list_sessions`` and the idle-TTL
  exemption see it without the manager importing this module (no cycle).
  Resolution lazily self-heals: a binding whose session is gone or inactive
  is dropped on lookup.

The execution boundary is untouched: this module never executes anything; it
only resolves which existing, isolated workspace a conversation uses. The
dispatch invariant is unchanged (the sandbox stays an injected session).

Kerckhoffs: the binding carries no secret; enforcement is the owner check and
the manager's session validation, both fully described here.
"""

from __future__ import annotations

import logging
import os
import secrets
import stat
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Module conventions.
checkpoint_before_apply = True
FEATURE_AVAILABLE = True

# Guarded heavy imports: the default manager and the tool-session wrapper are
# optional so this module loads under importlib isolation without the backend.
try:
    from opti_oignon.sandbox_manager import (
        sandbox_manager as _default_sandbox_manager,
    )
except Exception:  # pragma: no cover - defensive guard
    _default_sandbox_manager = None

try:
    from opti_oignon.sandbox_manager import (
        validate_sandbox_path as _validate_sandbox_path,
    )
except Exception:  # pragma: no cover - defensive guard
    _validate_sandbox_path = None

try:
    from opti_oignon.sandbox_tools import SandboxToolSession as _ToolSession

    SANDBOX_TOOLS_AVAILABLE = True
except Exception:  # pragma: no cover - defensive guard
    _ToolSession = None
    SANDBOX_TOOLS_AVAILABLE = False


class WorkspaceBindingError(Exception):
    """Base class for binding refusals."""


class WorkspaceNotFound(WorkspaceBindingError):
    """The target workspace does not exist or is inactive (route: 404)."""


class WorkspaceOwnerMismatch(WorkspaceBindingError):
    """The caller does not own the target workspace (route: 403)."""


class WorkspaceAlreadyBound(WorkspaceBindingError):
    """The workspace is bound to another active conversation (route: 409)."""


class WorkspaceBindings:
    """Thread-safe in-memory conversation <-> workspace binding store."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_conversation: dict[str, str] = {}
        self._by_sandbox: dict[str, str] = {}

    # -- internals ----------------------------------------------------

    @staticmethod
    def _resolve_manager(manager: Any) -> Any:
        mgr = manager if manager is not None else _default_sandbox_manager
        if mgr is None:
            raise WorkspaceBindingError("Sandbox manager not available")
        return mgr

    def _drop_locked(self, conversation_id: str) -> str | None:
        """Remove a conversation's binding from both maps. Lock held."""
        sid = self._by_conversation.pop(conversation_id, None)
        if sid is not None:
            self._by_sandbox.pop(sid, None)
        return sid

    @staticmethod
    def _audit(mgr: Any, session_id: str, action: str, detail: str) -> None:
        """Best-effort audit through the manager's approval log."""
        try:
            mgr.audit.log_approval(
                session_id, action=action, paths=[], detail=detail
            )
        except Exception:  # pragma: no cover - audit must not break binding
            logger.debug("binding audit failed", exc_info=True)

    # -- mutations ----------------------------------------------------

    def bind(
        self,
        conversation_id: str,
        sandbox_id: str,
        user_id: str = "local",
        manager: Any = None,
    ) -> None:
        """Bind a conversation to a workspace (rebind allowed).

        Raises:
            WorkspaceBindingError: empty conversation id or no manager.
            WorkspaceNotFound: unknown or inactive workspace (404).
            WorkspaceOwnerMismatch: caller is not the owner (403).
            WorkspaceAlreadyBound: another conversation holds it (409).
        """
        if not conversation_id:
            raise WorkspaceBindingError("conversation_id cannot be empty")
        mgr = self._resolve_manager(manager)
        session = mgr.get_session(sandbox_id)
        if session is None or not session.active:
            raise WorkspaceNotFound(f"Workspace not found: {sandbox_id}")
        if session.owner_user_id != user_id:
            raise WorkspaceOwnerMismatch(
                f"Workspace {sandbox_id} is not owned by this user"
            )

        with self._lock:
            holder = self._by_sandbox.get(sandbox_id)
            if holder is not None and holder != conversation_id:
                raise WorkspaceAlreadyBound(
                    f"Workspace {sandbox_id} is already bound to another "
                    "conversation"
                )
            if holder == conversation_id:
                return  # idempotent rebind of the same pair
            previous = self._drop_locked(conversation_id)
            self._by_conversation[conversation_id] = sandbox_id
            self._by_sandbox[sandbox_id] = conversation_id

        if previous is not None:
            mgr.set_binding(previous, None)
            self._audit(
                mgr,
                previous,
                "conversation_unbound",
                f"Conversation rebound away (conversation={conversation_id})",
            )
        mgr.set_binding(sandbox_id, conversation_id)
        self._audit(
            mgr,
            sandbox_id,
            "conversation_bound",
            f"Bound to conversation {conversation_id} (user={user_id})",
        )
        logger.info(
            "Workspace bound: conversation=%s sandbox=%s",
            conversation_id,
            sandbox_id,
        )

    def unbind(
        self,
        conversation_id: str,
        user_id: str = "local",
        manager: Any = None,
    ) -> bool:
        """Release a conversation's binding. No-op (False) when unbound.

        Raises:
            WorkspaceOwnerMismatch: caller is not the owner (403).
        """
        mgr = self._resolve_manager(manager)
        with self._lock:
            sid = self._by_conversation.get(conversation_id)
        if sid is None:
            return False
        session = mgr.get_session(sid)
        if session is not None and session.owner_user_id != user_id:
            raise WorkspaceOwnerMismatch(
                f"Workspace {sid} is not owned by this user"
            )
        with self._lock:
            self._drop_locked(conversation_id)
        mgr.set_binding(sid, None)
        self._audit(
            mgr,
            sid,
            "conversation_unbound",
            f"Unbound from conversation {conversation_id} (user={user_id})",
        )
        logger.info(
            "Workspace unbound: conversation=%s sandbox=%s",
            conversation_id,
            sid,
        )
        return True

    # -- resolution ---------------------------------------------------

    def get_sandbox_for(
        self, conversation_id: str, manager: Any = None
    ) -> str | None:
        """The workspace bound to a conversation, lazily self-healing.

        A binding whose session no longer exists (or is inactive) is stale:
        it is dropped from both maps and None is returned, so a destroyed
        workspace never resolves.
        """
        with self._lock:
            sid = self._by_conversation.get(conversation_id)
        if sid is None:
            return None
        try:
            mgr = self._resolve_manager(manager)
        except WorkspaceBindingError:
            return None
        session = mgr.get_session(sid)
        if session is None or not session.active:
            with self._lock:
                self._drop_locked(conversation_id)
            logger.info(
                "Stale binding dropped: conversation=%s sandbox=%s",
                conversation_id,
                sid,
            )
            return None
        return sid

    def get_conversation_for(self, sandbox_id: str) -> str | None:
        """The conversation holding a workspace, if any."""
        with self._lock:
            return self._by_sandbox.get(sandbox_id)

    def snapshot(self) -> dict[str, str]:
        """A copy of the conversation -> sandbox map (UI / tests)."""
        with self._lock:
            return dict(self._by_conversation)


# ---------------------------------------------------------------------------
# Module-level singleton (reset_* for tests, per the cartography invariant)
# ---------------------------------------------------------------------------

_BINDINGS: WorkspaceBindings | None = None
_BINDINGS_LOCK = threading.Lock()


def get_workspace_bindings() -> WorkspaceBindings:
    """The process-wide binding store."""
    global _BINDINGS
    with _BINDINGS_LOCK:
        if _BINDINGS is None:
            _BINDINGS = WorkspaceBindings()
        return _BINDINGS


def reset_workspace_bindings() -> None:
    """Drop the singleton (tests)."""
    global _BINDINGS
    with _BINDINGS_LOCK:
        _BINDINGS = None


# ---------------------------------------------------------------------------
# Baseline manifest (the section 6.1 seam the apply writer consumes)
# ---------------------------------------------------------------------------

def manifest_hash_file(path: str, chunk_size: int = 65536) -> str:
    """The per-file content hash of the baseline manifest (section 6.1).

    A streamed SHA-256 (the ``_compute_diffs_hash`` family in the coding
    agent), bounded chunks so a large file never lands in memory at once.
    Bloc 3's diff recomputes this over the live workspace and compares it
    against the recorded baseline.
    """
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class WorkspaceManifests:
    """Thread-safe in-memory baseline-manifest store.

    Per workspace: ``relative_path -> sha256`` for every file recorded at
    copy-in (upload AND clone, per the spec's 6.1 text), plus the
    ``cloned_root`` -- the realpath of the originally-cloned host directory,
    the only implicit write-back target Bloc 3 may use (6.2). The root is
    WRITE-ONCE: the first clone fixes it; later clones never move it
    ("writes only under the ORIGINALLY-cloned root"). Upload-only
    workspaces keep it None -- no implicit target.

    Design decisions:

    - IN-MEMORY, like the bindings: a manifest inside ``/workspace`` would
      be writable by the sandboxed code, which could then hide its own
      modifications from the Bloc 3 review -- unacceptable for a review-gate
      input. A host-side file would persist user-tree paths and hashes in
      plaintext (a privacy surface) and add a cleanup lifecycle. In-memory
      is tamper-proof from the sandbox and dies with the process.
    - Documented limitation: under ``workspace_persistent``, a restart
      loses the baseline. The failure mode is CONSERVATIVE: Bloc 3 without
      a baseline has no implicit write-back target and must treat the
      whole tree as unreviewed -- never less review, never a silent write.
    - Entries are recorded by the ROUTES after a successful manager
      operation on an active session (the manager computes the hashes on
      the fly during the copy; it never imports this module -- no cycle).
      The destroy route drops the entry so a reused session id never
      inherits a stale baseline.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, str]] = {}
        self._cloned_roots: dict[str, str] = {}
        self._cloned_mounts: dict[str, str] = {}

    def record(
        self,
        session_id: str,
        entries: dict[str, str],
        cloned_root: str | None = None,
        cloned_mount: str | None = None,
    ) -> int:
        """Merge manifest entries for a workspace; returns its new size.

        ``cloned_root`` is recorded only if none is set yet (write-once).
        ``cloned_mount`` is the clone's WORKSPACE-relative
        destination (e.g. ``src`` for a clone of ``/share/src``), recorded
        write-once TOGETHER with the root: the apply writer maps
        ``mount/<rel>`` onto ``cloned_root/<rel>`` so the round-trip lands
        on the original files, and refuses (per file, honestly) anything
        outside the cloned subtree -- a workspace-top-level addition has no
        original under the cloned root.
        """
        with self._lock:
            manifest = self._entries.setdefault(session_id, {})
            manifest.update(entries)
            if cloned_root and session_id not in self._cloned_roots:
                self._cloned_roots[session_id] = cloned_root
                if cloned_mount:
                    self._cloned_mounts[session_id] = cloned_mount
            return len(manifest)

    def get_manifest(self, session_id: str) -> dict[str, str] | None:
        """The recorded baseline (a copy), or None when nothing recorded."""
        with self._lock:
            manifest = self._entries.get(session_id)
            return dict(manifest) if manifest is not None else None

    def get_cloned_root(self, session_id: str) -> str | None:
        """The originally-cloned host root (Bloc 3's implicit target)."""
        with self._lock:
            return self._cloned_roots.get(session_id)

    def get_cloned_mount(self, session_id: str) -> str | None:
        """The clone's workspace-relative destination, if any."""
        with self._lock:
            return self._cloned_mounts.get(session_id)

    def drop(self, session_id: str) -> None:
        """Forget a workspace's baseline (called on destroy)."""
        with self._lock:
            self._entries.pop(session_id, None)
            self._cloned_roots.pop(session_id, None)
            self._cloned_mounts.pop(session_id, None)

    def snapshot(self) -> dict[str, int]:
        """Per-workspace manifest sizes (UI / tests)."""
        with self._lock:
            return {sid: len(m) for sid, m in self._entries.items()}


_MANIFESTS: WorkspaceManifests | None = None
_MANIFESTS_LOCK = threading.Lock()


def get_workspace_manifests() -> WorkspaceManifests:
    """The process-wide manifest store."""
    global _MANIFESTS
    with _MANIFESTS_LOCK:
        if _MANIFESTS is None:
            _MANIFESTS = WorkspaceManifests()
        return _MANIFESTS


def reset_workspace_manifests() -> None:
    """Drop the singleton (tests)."""
    global _MANIFESTS
    with _MANIFESTS_LOCK:
        _MANIFESTS = None


# ---------------------------------------------------------------------------
# Workspace diff + apply-to-host writer (spec section 6)
# ---------------------------------------------------------------------------

# The exact-walk refusal semantics: hitting a bound REFUSES, never
# undercounts -- a review computed on a partial walk is no review.
_WORKSPACE_WALK_MAX_DEPTH = 32
# Fallback when no manager config is reachable (tests, partial builds);
# the authoritative knob is SandboxConfig.diff_max_entries.
_DIFF_MAX_ENTRIES_FALLBACK = 50000
# Streamed write chunk size for the apply writer (the copy-in family).
_APPLY_CHUNK_BYTES = 65536
# Sane default mode when a workspace source carries no permission bits.
_APPLY_DEFAULT_MODE = 0o644


class WorkspaceDiffError(Exception):
    """Base class for diff/apply refusals."""


class WorkspaceDiffBoundExceeded(WorkspaceDiffError):
    """The diff exceeds its entry or depth bound (route: 413).

    A review gate must present the WHOLE change set or refuse: a truncated
    or paginated diff would let the user approve against an incomplete
    picture, so exceeding a bound refuses (the exact-prewalk
    semantics), never silently truncates.
    """


class WorkspaceReviewDrift(WorkspaceDiffError):
    """The workspace changed since the reviewed diff (route: 409).

    The CA-05 reviewed-diff binding: apply proceeds only for the exact
    change set the human saw; any drift requires a fresh diff and review.
    """


class WorkspaceApplyTargetError(WorkspaceDiffError):
    """No usable apply target, or a conflicting explicit one (route: 400)."""


@dataclass
class WorkspaceChange:
    """One classified change against the baseline manifest (section 6.1).

    The classification mirrors the coding agent's ``FileDiff`` semantics
    (added = is_new, modified, deleted = is_deleted), generalized to any
    workspace and HASH-DRIVEN: the baseline stores hashes only (never
    contents), so there is no stored original to produce a unified text
    diff against -- the per-file content preview rides the existing
    ``preview_file`` instead, which already hex-previews binaries (so
    binary files need no special casing here: classification is hash-only
    for every file by construction).
    """

    path: str
    kind: str  # "added" | "modified" | "deleted"
    size: int = 0
    baseline_hash: str = ""
    current_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "size": self.size,
            "baseline_hash": self.baseline_hash,
            "current_hash": self.current_hash,
        }


@dataclass
class WorkspaceDiff:
    """The classified workspace diff plus its review-integrity hash."""

    session_id: str
    baseline_present: bool
    cloned_root: str | None
    cloned_mount: str | None
    entries: list[WorkspaceChange]
    unchanged: int
    skipped_symlinks: int
    skipped_special: int
    diff_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "baseline_present": self.baseline_present,
            "cloned_root": self.cloned_root,
            "cloned_mount": self.cloned_mount,
            "entries": [c.to_dict() for c in self.entries],
            "unchanged": self.unchanged,
            "skipped_symlinks": self.skipped_symlinks,
            "skipped_special": self.skipped_special,
            "diff_hash": self.diff_hash,
        }


def compute_workspace_diff_hash(entries: list[WorkspaceChange]) -> str:
    """Deterministic review-integrity hash over a classified change set.

    The coding agent's ``_compute_diffs_hash`` recipe generalized
    hash-driven: per entry, path + kind + current content hash (empty for
    a deletion), NUL-separated, over the set sorted by path so the digest
    is independent of walk order. The apply writer recomputes the live
    diff and refuses on a digest mismatch (the CA-05 reviewed-diff
    binding): apply proceeds only for the exact change set the human saw.
    """
    import hashlib

    digest = hashlib.sha256()
    for change in sorted(entries, key=lambda c: c.path):
        digest.update(change.path.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(change.kind.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(change.current_hash.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _walk_workspace_hashes(
    workspace_root: str, max_entries: int
) -> tuple[dict[str, str], dict[str, int], int, int]:
    """lstat-driven hash walk of the live workspace (no follow, bounded).

    Symlinks are NEVER followed (and never hashed): the sandboxed code can
    create them, and following one during the review would read host files
    into the diff -- an exfiltration channel into the very gate meant to
    stop it. Symlinks and special files are skipped and COUNTED so the
    review can state them honestly (the clone discipline). Exceeding
    the file-count bound or the depth bound REFUSES
    (``WorkspaceDiffBoundExceeded``), never undercounts.

    Returns (relative_path -> sha256, relative_path -> size,
    skipped_symlinks, skipped_special).
    """
    hashes: dict[str, str] = {}
    sizes: dict[str, int] = {}
    skipped_symlinks = 0
    skipped_special = 0
    root_real = os.path.realpath(workspace_root)
    stack: list[tuple[str, int]] = [(root_real, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > _WORKSPACE_WALK_MAX_DEPTH:
            raise WorkspaceDiffBoundExceeded(
                f"Workspace exceeds the maximum diff walk depth "
                f"({_WORKSPACE_WALK_MAX_DEPTH}): {current}"
            )
        with os.scandir(current) as it:
            for entry in it:
                try:
                    if entry.is_symlink():
                        skipped_symlinks += 1
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        # Manifest rule (cross-cycle with the clone and
                        # apply layers): the workspace-root .agent/ prefix is
                        # agent-internal (spill files); the copy-out diff
                        # excludes it, so it can never classify as a change.
                        if depth == 0 and entry.name == ".agent":
                            continue
                        stack.append((entry.path, depth + 1))
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        skipped_special += 1
                        continue
                    size = entry.stat(follow_symlinks=False).st_size
                except OSError:
                    continue
                if len(hashes) >= max_entries:
                    raise WorkspaceDiffBoundExceeded(
                        f"Workspace exceeds the diff entry bound "
                        f"({max_entries} files); the review refuses rather "
                        f"than truncate"
                    )
                rel = os.path.relpath(entry.path, root_real)
                try:
                    hashes[rel] = manifest_hash_file(entry.path)
                except OSError:
                    continue
                sizes[rel] = size
    return hashes, sizes, skipped_symlinks, skipped_special


def generate_workspace_diff(
    session_id: str,
    manager: Any = None,
    max_entries: int | None = None,
) -> WorkspaceDiff:
    """Classify the live workspace against the recorded baseline (6.1).

    Recomputes per-file hashes over the live workspace
    (``manifest_hash_file``, the recompute function the status note
    names) and compares them against the ``WorkspaceManifests`` baseline:
    a live file absent from the baseline is "added", a hash mismatch is
    "modified", a baseline entry absent from the live tree is "deleted";
    matching hashes count as unchanged (uploaded-then-untouched files diff
    clean).

    No-baseline posture: when nothing is recorded (upload-only workspace
    after a restart, or a partial build), EVERYTHING live classifies
    "added", ``baseline_present`` is False, and there is no implicit
    write-back target -- the result states it; a baseline is never
    invented. The failure mode stays CONSERVATIVE: never less review.

    Raises:
        ValueError: Unknown/inactive session or missing workspace (404).
        WorkspaceDiffBoundExceeded: Entry or depth bound exceeded (413).
    """
    mgr = WorkspaceBindings._resolve_manager(manager)
    workspace = mgr.get_active_workspace_path(session_id)
    if max_entries is None:
        max_entries = getattr(
            getattr(mgr, "_config", None),
            "diff_max_entries",
            _DIFF_MAX_ENTRIES_FALLBACK,
        )
    store = get_workspace_manifests()
    baseline = store.get_manifest(session_id)
    cloned_root = store.get_cloned_root(session_id)
    cloned_mount = store.get_cloned_mount(session_id)

    live_hashes, live_sizes, skipped_symlinks, skipped_special = (
        _walk_workspace_hashes(workspace, max_entries)
    )

    baseline_present = baseline is not None
    base = baseline or {}
    entries: list[WorkspaceChange] = []
    unchanged = 0
    for rel in sorted(live_hashes):
        current_hash = live_hashes[rel]
        baseline_hash = base.get(rel)
        if baseline_hash is None:
            entries.append(WorkspaceChange(
                path=rel,
                kind="added",
                size=live_sizes.get(rel, 0),
                current_hash=current_hash,
            ))
        elif baseline_hash != current_hash:
            entries.append(WorkspaceChange(
                path=rel,
                kind="modified",
                size=live_sizes.get(rel, 0),
                baseline_hash=baseline_hash,
                current_hash=current_hash,
            ))
        else:
            unchanged += 1
    for rel in sorted(base):
        if rel not in live_hashes:
            entries.append(WorkspaceChange(
                path=rel,
                kind="deleted",
                baseline_hash=base[rel],
            ))

    if len(entries) > max_entries:
        raise WorkspaceDiffBoundExceeded(
            f"Diff exceeds the entry bound ({len(entries)} > {max_entries} "
            f"classified changes); the review refuses rather than truncate"
        )

    return WorkspaceDiff(
        session_id=session_id,
        baseline_present=baseline_present,
        cloned_root=cloned_root,
        cloned_mount=cloned_mount,
        entries=entries,
        unchanged=unchanged,
        skipped_symlinks=skipped_symlinks,
        skipped_special=skipped_special,
        diff_hash=compute_workspace_diff_hash(entries),
    )


def _refuse_request_path(rel: str) -> str | None:
    """Reject a client-supplied relative path before any filesystem touch.

    Returns the refusal reason, or None when the path shape is acceptable.
    Absolute paths, NUL bytes, empty paths, the bare target root, and any
    ``..`` segment after normalization are refused outright -- apply paths
    are client-controlled input, not trusted manifest keys.
    """
    if not rel or "\x00" in rel:
        return "empty path or NUL byte"
    if os.path.isabs(rel):
        return "absolute path refused"
    norm = os.path.normpath(rel)
    if norm == ".":
        return "path resolves to the target root"
    if norm == ".." or ".." in norm.split(os.sep):
        return "path escapes the target root ('..' segment)"
    return None


def _ensure_safe_parent(target_root: str, rel: str) -> tuple[str | None, str]:
    """Walk-and-create the destination's parent, refusing symlink components.

    From the (already realpath'd, allowlist-validated) target root down to
    the destination's parent: every EXISTING component must lstat as a
    real directory -- a symlink component, even one currently pointing
    inside the root, is refused (it could redirect the write, e.g. onto a
    dotfile in ``$HOME``); missing components are created one at a time
    (0o755) and re-checked. Belt-and-braces: the final parent's realpath
    must equal its lexical join, so any redirection that slipped between
    checks still refuses.

    Returns (refusal_reason_or_None, destination_abs_path).
    """
    norm = os.path.normpath(rel)
    dest = os.path.join(target_root, norm)
    rel_parent = os.path.dirname(norm)
    current = target_root
    if rel_parent:
        for part in rel_parent.split(os.sep):
            if not part:
                continue
            current = os.path.join(current, part)
            try:
                st = os.lstat(current)
            except FileNotFoundError:
                try:
                    os.mkdir(current, 0o755)
                    st = os.lstat(current)
                except OSError as exc:
                    return f"cannot create directory component: {exc}", dest
            except OSError as exc:
                return f"cannot inspect directory component: {exc}", dest
            if stat.S_ISLNK(st.st_mode):
                return f"symlinked path component refused: {current}", dest
            if not stat.S_ISDIR(st.st_mode):
                return f"non-directory path component: {current}", dest
    lexical_parent = os.path.dirname(dest)
    if os.path.realpath(lexical_parent) != lexical_parent:
        return "parent directory resolves elsewhere (symlink redirection)", dest
    return None, dest


def _write_file_into_target(src: str, dest: str, mode: int) -> int:
    """Stream a workspace file onto ``dest`` via temp-file-plus-rename.

    The temp file is created in the SAME directory as ``dest`` with
    ``O_CREAT|O_EXCL|O_NOFOLLOW`` (its creation can never be redirected
    through a symlink), filled in bounded chunks, ``fchmod``'d to the
    requested sane mode, then moved over with ``os.replace`` -- atomic on
    the same filesystem; a symlink at the final component is replaced AS
    THE LINK, never followed. The temp file is removed on any failure.

    Returns the byte count written.
    """
    parent = os.path.dirname(dest)
    tmp = os.path.join(parent, f".oo-apply-{secrets.token_hex(8)}.tmp")
    fd = os.open(
        tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600
    )
    written = 0
    try:
        try:
            with open(src, "rb") as fin:
                while True:
                    chunk = fin.read(_APPLY_CHUNK_BYTES)
                    if not chunk:
                        break
                    os.write(fd, chunk)
                    written += len(chunk)
            os.fchmod(fd, mode)
        finally:
            os.close(fd)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return written


def apply_workspace_changes(
    session_id: str,
    diff_hash: str,
    manager: Any = None,
    target_dir: str | None = None,
    max_entries: int | None = None,
) -> dict[str, Any]:
    """Write ONLY approved changes back to the host (6.2).

    The cycle's highest-risk component; every refusal is honest and
    audited. ``checkpoint_before_apply`` is exactly what this writer
    exists for: NOTHING reaches the host without prior human approval --
    per-change approval for writes, a SEPARATE explicit confirmation for
    deletions -- and apply itself is a user action through the manager UI;
    the model can trigger nothing here (the dispatch invariant is
    unchanged).

    Target rule: the write-once ``cloned_root`` when
    present, RE-VALIDATED at apply time against the CURRENT
    ``host_share_roots`` (an operator who narrowed the allowlist after
    the clone fails secure); a conflicting explicit ``target_dir`` is
    refused, never silently ignored. Without a cloned root an EXPLICIT
    user-chosen target is required, resolved through the
    confinement (403-before-existence). No cloned root and no explicit
    target: refuse; never guess. Path mapping under the cloned root: the
    clone lands at the workspace-relative ``cloned_mount`` (e.g. ``src``
    for a clone of ``/share/src``), so ``mount/<rel>`` maps onto
    ``cloned_root/<rel>`` -- the round-trip writes the original files,
    never ``cloned_root/<mount>/<rel>``; entries OUTSIDE the cloned
    subtree have no original under the root and are refused per file,
    honestly (apply them via an explicit target instead). Under an
    explicit target, workspace paths map 1:1 (the workspace layout is
    reproduced).

    Review binding (the CA-05 discipline): the live diff is recomputed
    and its digest compared against ``diff_hash`` -- the digest the
    client received with the diff the human reviewed. Any drift refuses
    (``WorkspaceReviewDrift``).

    Writes: only entries classified added/modified AND approved
    (``is_file_approved``), streamed via temp-file-plus-rename under the
    symlink-component validation. Deletions: only entries classified
    deleted AND separately confirmed (``get_confirmed_deletions``); a
    deletion target that lstat's as a symlink is removed AS THE LINK,
    never the target; an already-absent target reports honestly.

    Raises:
        WorkspaceReviewDrift: Missing or stale ``diff_hash`` (409).
        WorkspaceApplyTargetError: No target, or a conflicting one (400).
        PermissionError: Explicit target outside the allowlist (403).
        ValueError: Unknown/inactive session, or a target directory that
            no longer exists inside a root (404).
        WorkspaceDiffBoundExceeded: The recomputed diff exceeds its
            bounds (413).
    """
    if not diff_hash:
        raise WorkspaceReviewDrift(
            "Apply requires the reviewed diff_hash; run the diff first"
        )
    mgr = WorkspaceBindings._resolve_manager(manager)
    workspace = mgr.get_active_workspace_path(session_id)

    diff = generate_workspace_diff(
        session_id, manager=mgr, max_entries=max_entries
    )
    if diff.diff_hash != diff_hash:
        raise WorkspaceReviewDrift(
            "Workspace changed since the reviewed diff; re-run the diff "
            "and review again"
        )

    cloned_root = diff.cloned_root
    cloned_mount = diff.cloned_mount
    under_cloned_root = False
    if cloned_root:
        if target_dir:
            requested = os.path.realpath(os.path.expanduser(target_dir))
            if requested != cloned_root:
                raise WorkspaceApplyTargetError(
                    "Workspace has a write-once cloned root; a different "
                    "explicit target is refused"
                )
        target_root = mgr.resolve_share_target(cloned_root)
        under_cloned_root = True
    elif target_dir:
        target_root = mgr.resolve_share_target(target_dir)
    else:
        raise WorkspaceApplyTargetError(
            "No cloned root and no explicit target directory; refusing "
            "to guess an apply target"
        )

    audit = getattr(mgr, "audit", None)

    def _audit_row(action: str, paths: list[str], detail: str) -> None:
        if audit is None:
            return
        try:
            audit.log_approval(
                session_id,
                action=action,
                paths=paths,
                dest_dir=target_root,
                detail=detail,
            )
        except Exception:  # pragma: no cover - audit must not break apply
            logger.debug("apply audit failed", exc_info=True)

    confirmed = mgr.get_confirmed_deletions(session_id)
    applied: list[dict[str, Any]] = []
    deleted: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    skipped_unapproved = 0
    skipped_unconfirmed = 0

    for change in diff.entries:
        rel = change.path
        if change.kind in ("added", "modified"):
            if not mgr.is_file_approved(session_id, rel):
                skipped_unapproved += 1
                continue
        else:
            # The load-bearing deletion check: only paths BOTH classified
            # deleted by the recomputed diff AND separately confirmed are
            # ever removed -- confirming a live path can never delete it.
            if rel not in confirmed:
                skipped_unconfirmed += 1
                continue

        # Map the workspace path onto the host side. Under the cloned root,
        # only the cloned subtree round-trips (mount/<rel> -> root/<rel>);
        # anything outside it has no original under the root and refuses
        # honestly. Under an explicit target, paths map 1:1.
        if under_cloned_root:
            if not cloned_mount:
                reason = (
                    "cloned mount unknown (conservative refusal); "
                    "choose an explicit target directory"
                )
                refused.append({"path": rel, "error": reason})
                _audit_row("apply_refused", [rel], f"Refused: {reason}")
                continue
            if rel == cloned_mount or rel.startswith(cloned_mount + "/"):
                host_rel = rel[len(cloned_mount):].lstrip("/")
            else:
                reason = (
                    f"outside the cloned subtree ({cloned_mount}/); "
                    "apply it via an explicit target directory"
                )
                refused.append({"path": rel, "error": reason})
                _audit_row("apply_refused", [rel], f"Refused: {reason}")
                continue
        else:
            host_rel = rel

        reason = _refuse_request_path(host_rel)
        dest = ""
        if reason is None:
            reason, dest = _ensure_safe_parent(target_root, host_rel)
        if reason is not None:
            refused.append({"path": rel, "error": reason})
            _audit_row("apply_refused", [rel], f"Refused: {reason}")
            continue

        if change.kind == "deleted":
            try:
                st = os.lstat(dest)
            except FileNotFoundError:
                deleted.append({"path": rel, "action": "already_absent"})
                _audit_row(
                    "apply_delete", [rel], "Deletion target already absent"
                )
                continue
            except OSError as exc:
                refused.append({"path": rel, "error": str(exc)})
                _audit_row("apply_refused", [rel], f"Refused: {exc}")
                continue
            if stat.S_ISDIR(st.st_mode):
                refused.append(
                    {"path": rel, "error": "deletion target is a directory"}
                )
                _audit_row(
                    "apply_refused",
                    [rel],
                    "Refused: deletion target is a directory",
                )
                continue
            was_symlink = stat.S_ISLNK(st.st_mode)
            try:
                os.remove(dest)  # removes a symlink AS THE LINK, never the target
            except OSError as exc:
                refused.append({"path": rel, "error": str(exc)})
                _audit_row("apply_refused", [rel], f"Refused: {exc}")
                continue
            deleted.append({"path": rel, "action": "deleted"})
            _audit_row(
                "apply_delete",
                [rel],
                "Deleted on host"
                + (" (symlink removed as the link)" if was_symlink else ""),
            )
            continue

        # added/modified: stream the workspace source over a same-dir temp.
        if _validate_sandbox_path is None:
            refused.append({
                "path": rel,
                "error": "workspace path validation unavailable",
            })
            _audit_row(
                "apply_refused",
                [rel],
                "Refused: workspace path validation unavailable",
            )
            continue
        valid, src, err = _validate_sandbox_path(workspace, rel)
        if not valid or not os.path.isfile(src):
            detail = err or "not a regular file"
            refused.append(
                {"path": rel, "error": f"workspace source invalid: {detail}"}
            )
            _audit_row(
                "apply_refused", [rel], f"Refused: source invalid ({detail})"
            )
            continue
        try:
            src_mode = os.lstat(src).st_mode & 0o777
            written = _write_file_into_target(
                src, dest, src_mode or _APPLY_DEFAULT_MODE
            )
        except OSError as exc:
            refused.append({"path": rel, "error": str(exc)})
            _audit_row("apply_refused", [rel], f"Refused: {exc}")
            continue
        action = "created" if change.kind == "added" else "modified"
        applied.append({"path": rel, "action": action, "bytes": written})
        _audit_row("apply_write", [rel], f"{action} ({written} bytes)")

    try:
        mgr.touch_activity(session_id)
    except Exception:  # pragma: no cover - activity must not break apply
        logger.debug("apply touch_activity failed", exc_info=True)

    _audit_row(
        "apply_summary",
        [],
        f"Applied {len(applied)} write(s), {len(deleted)} deletion(s), "
        f"{len(refused)} refused, {skipped_unapproved} unapproved "
        f"skipped, {skipped_unconfirmed} unconfirmed skipped",
    )
    return {
        "session_id": session_id,
        "target": target_root,
        "applied": applied,
        "deleted": deleted,
        "refused": refused,
        "skipped_unapproved": skipped_unapproved,
        "skipped_unconfirmed": skipped_unconfirmed,
        "diff_hash": diff.diff_hash,
    }


def attach_session_for_conversation(
    conversation_id: str,
    manager: Any = None,
    tool_registry: Any = None,
) -> Any:
    """Build a SandboxToolSession ATTACHED to the conversation's workspace.

    The ATL-02 seam: called by the agent run wiring. Returns the attached
    session (registry lockout ON, per the unchanged ``set_sandbox_mode``
    invariant) or None when the conversation has no bound workspace -- the
    dispatch then refuses sandboxed tools exactly as before (explicit
    binding, no auto-create). The caller MUST ``detach()`` the returned
    session when the run ends; detach releases the lockout WITHOUT
    destroying the workspace.
    """
    if not conversation_id or not SANDBOX_TOOLS_AVAILABLE:
        return None
    bindings = get_workspace_bindings()
    sandbox_id = bindings.get_sandbox_for(conversation_id, manager=manager)
    if sandbox_id is None:
        return None
    try:
        mgr = WorkspaceBindings._resolve_manager(manager)
    except WorkspaceBindingError:
        return None
    try:
        session = (
            _ToolSession(mgr, tool_registry)
            if tool_registry is not None
            else _ToolSession(mgr)
        )
        session.attach(sandbox_id)
        return session
    except Exception:  # pragma: no cover - the run proceeds sandbox-less
        logger.exception(
            "Failed to attach workspace %s for conversation %s",
            sandbox_id,
            conversation_id,
        )
        return None
