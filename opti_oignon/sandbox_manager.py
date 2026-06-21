#!/usr/bin/env python3
"""
SANDBOX MANAGER - OPTI-OIGNON v2.1.0 (S73/S81/S116)
============================================

Manages fully isolated, disposable sandbox environments for LLM
filesystem and shell tool execution. Every LLM-initiated file or
command operation runs inside a sandbox with strict security boundaries.

S116: File copy-out with human approval workflow.
- ApprovalState enum: PENDING -> APPROVED -> copied out
- preview_file(): read file content for display (capped at 64KB)
- approve_files(): explicitly approve specific paths for copy-out
- reject_files(): reject all, preventing any copy-out
- copy_out_file(): copy a single approved file to host
- copy_out_batch(): copy multiple approved files to host
- Audit trail for all approval and copy-out events
- No auto-approve: files NEVER copied without explicit user action

ISOLATION BACKENDS:

1. **bwrap (bubblewrap)** -- RECOMMENDED / DEFAULT
   Uses Linux kernel namespaces via bubblewrap to create a true
   isolated environment:
   - Mount namespace: process sees ONLY /workspace + read-only system bins
   - Network namespace: zero network access (--unshare-net)
   - PID namespace: cannot see/signal host processes (--unshare-pid)
   - No privilege escalation (--new-session, --die-with-parent)
   - Clean environment (--clearenv): no host environment variable
     (including any secret) is inherited by sandboxed code
   Even rm -rf / only destroys the namespace. Host is invisible.

2. **tempdir** -- UNSAFE FALLBACK (requires explicit confirmation)
   Uses tempfile.mkdtemp with a command blocklist. The process runs
   with the same privileges as the backend. A determined or
   hallucinating LLM CAN escape this. Only for development/testing.

The command blocklist (blocked_commands + blocked_patterns) is applied
on ALL backends as defense-in-depth -- even with bwrap.

S81: Security audit hardening:
- CommandValidator inspects recently created file contents before
  allowing bash execution that references those files
- Extended base64-pipe-to-shell detection
- Extended Python subprocess pattern detection

Author: Leon
"""

import enum
import logging
import os
import re
import resource
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from opti_oignon.db_utils import safe_connect

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "config", "sandbox.yaml"
)

# Per-sandbox resource-cap bounds (S209, Bloc 0). Out-of-range configured
# values are CLAMPED into these ranges, never disabled: a cap that is set too
# low or too high is corrected, not switched off. Defaults are conservative.
_CAP_MEMORY_BYTES_DEFAULT = 2 * 1024 ** 3      # 2 GiB (RLIMIT_AS, per process)
_CAP_MEMORY_BYTES_MIN = 64 * 1024 ** 2         # 64 MiB
_CAP_MEMORY_BYTES_MAX = 16 * 1024 ** 3         # 16 GiB
_CAP_NPROC_DEFAULT = 4096                       # RLIMIT_NPROC (per real uid)
_CAP_NPROC_MIN = 16
_CAP_NPROC_MAX = 8192
_CAP_FSIZE_BYTES_DEFAULT = 1 * 1024 ** 3       # 1 GiB (RLIMIT_FSIZE, per file)
_CAP_FSIZE_BYTES_MIN = 1 * 1024 ** 2           # 1 MiB
_CAP_FSIZE_BYTES_MAX = 16 * 1024 ** 3          # 16 GiB
_CAP_CPU_SECONDS_DEFAULT = 300                  # RLIMIT_CPU (CPU-seconds)
_CAP_CPU_SECONDS_MIN = 5
_CAP_CPU_SECONDS_MAX = 86400
_CAP_TMPFS_BYTES_DEFAULT = 256 * 1024 ** 2     # 256 MiB (--size on the tmpfs)
_CAP_TMPFS_BYTES_MIN = 1 * 1024 ** 2           # 1 MiB
_CAP_TMPFS_BYTES_MAX = 8 * 1024 ** 3           # 8 GiB

# Workspace lifecycle bounds (S210, Bloc 1). Same clamp discipline as the
# S209 caps: out-of-range values are corrected, never disabled. idle TTL 0
# is the documented "disabled" value (no idle reaping).
_TTL_SECONDS_DEFAULT = 3600                     # 1 hour idle -> destroyed
_TTL_SECONDS_MIN = 0                            # 0 disables the idle TTL
_TTL_SECONDS_MAX = 30 * 86400                   # 30 days
_DISK_SOFT_BYTES_DEFAULT = 512 * 1024 ** 2     # 512 MiB per workspace
_DISK_SOFT_BYTES_MIN = 1 * 1024 ** 2           # 1 MiB
_DISK_SOFT_BYTES_MAX = 16 * 1024 ** 3          # 16 GiB

# Bounded disk-use walk (S210): the approximate per-workspace disk figure is
# a scandir walk capped on entries and depth so a pathological tree cannot
# stall the manager. The figure is approximate by design.
_DISK_WALK_MAX_ENTRIES = 10000
_DISK_WALK_MAX_DEPTH = 32

# Copy-in bounds (S211, Bloc 2). Same clamp discipline (correct, never
# disable). The per-request upload total and the clone total are ALSO bounded
# by the S210 per-workspace disk soft quota: the caps below bound a single
# request, the quota bounds the workspace as a whole -- both apply.
_UPLOAD_MAX_FILES_DEFAULT = 64
_UPLOAD_MAX_FILES_MIN = 1
_UPLOAD_MAX_FILES_MAX = 1024
_UPLOAD_MAX_FILE_BYTES_DEFAULT = 128 * 1024 ** 2   # 128 MiB per file
_UPLOAD_MAX_FILE_BYTES_MIN = 1 * 1024 ** 2         # 1 MiB
_UPLOAD_MAX_FILE_BYTES_MAX = 16 * 1024 ** 3        # 16 GiB
_CLONE_MAX_BYTES_DEFAULT = 512 * 1024 ** 2         # 512 MiB per clone
_CLONE_MAX_BYTES_MIN = 1 * 1024 ** 2               # 1 MiB
_CLONE_MAX_BYTES_MAX = 16 * 1024 ** 3              # 16 GiB
_CLONE_MAX_FILES_DEFAULT = 20000
_CLONE_MAX_FILES_MIN = 1
_CLONE_MAX_FILES_MAX = 1000000

# Exact pre-walk bound for the clone source (S211): unlike the approximate
# disk-use walk above, hitting this depth REFUSES the clone rather than
# silently undercounting -- a cap enforced on a partial figure is no cap.
_CLONE_WALK_MAX_DEPTH = 32

# Streamed copy/hash chunk size for the copy-in paths (S211).
_COPYIN_CHUNK_BYTES = 65536

# Diff review bound (S212, Bloc 3). A review gate must present the WHOLE
# change set or refuse: a truncated or paginated review would let the user
# approve against an incomplete picture. Exceeding this entry bound REFUSES
# the diff (the S211 exact-prewalk refusal semantics), never truncates.
_DIFF_MAX_ENTRIES_DEFAULT = 50000
_DIFF_MAX_ENTRIES_MIN = 1
_DIFF_MAX_ENTRIES_MAX = 1000000

# Provision-run timeout (S213, Bloc 4). The provision phase is the one
# network-on run (dependency installation into a workspace venv); a real
# pip install needs longer than the interactive command_timeout. This is a
# TIMEOUT key only -- there is no configuration key that turns the network
# on (spec 8.3: never a config default; the flag is per workspace and
# user-set only).
_PROVISION_TIMEOUT_DEFAULT = 600
_PROVISION_TIMEOUT_MIN = 30
_PROVISION_TIMEOUT_MAX = 3600

_RESOURCE_BACKENDS = ("rlimit", "cgroup")


def _clamp(value: int, low: int, high: int) -> int:
    """Clamp an integer into [low, high]; out-of-range corrects, never disables."""
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        ivalue = low
    if ivalue < low:
        return low
    if ivalue > high:
        return high
    return ivalue


def _normalize_share_roots(raw: list[str] | None) -> list[str]:
    """Normalize the host-share root allowlist (S211, Bloc 2).

    Each entry is expanduser'd and realpath'd at load time so the browse and
    clone confinement checks compare resolved paths against resolved roots.
    Filesystem root ("/") is NEVER an allowed share root and is dropped with
    a warning; entries that do not resolve to an existing directory are
    dropped (an operator typo must not silently widen or dangle the
    allowlist). An empty or unset list defaults to the user's home. The
    normalized list may legitimately end up EMPTY (e.g. home resolves to
    "/"); the browse/clone routes then refuse everything -- fail-secure.
    """
    fs_root = os.path.realpath(os.sep)
    candidates = list(raw) if raw else [os.path.expanduser("~")]
    roots: list[str] = []
    for entry in candidates:
        if not isinstance(entry, str) or not entry.strip():
            continue
        resolved = os.path.realpath(os.path.expanduser(entry.strip()))
        if resolved == fs_root:
            logger.warning(
                "host_share_roots: refusing filesystem root '/' as a "
                "share root (entry %r dropped)", entry,
            )
            continue
        if not os.path.isdir(resolved):
            logger.warning(
                "host_share_roots: entry %r does not resolve to an "
                "existing directory (dropped)", entry,
            )
            continue
        if resolved not in roots:
            roots.append(resolved)
    return roots

# Hardcoded paths that must NEVER be bound into the sandbox,
# regardless of what the YAML config says. Defense-in-depth.
_HARDCODED_NEVER_BIND = frozenset({
    "/home",
    "/root",
    "/var",
    "/etc/shadow",
    "/etc/passwd",
    "/etc/sudoers",
    "/etc/sudoers.d",
    "/etc/ssh",
    "/mnt",
    "/media",
    "/opt",
    "/srv",
    "/boot",
    "/run",
    "/snap",
    "/swapfile",
})

DEGRADED_WARNING = """
================================================================================
  WARNING: SANDBOX RUNNING IN DEGRADED MODE (tempdir only)
================================================================================

  Bubblewrap (bwrap) is NOT available on this system.
  The sandbox is using a temporary directory with a command blocklist ONLY.

  THIS IS NOT TRUE ISOLATION. The LLM process runs with YOUR user
  privileges and CAN access the entire host filesystem if it bypasses
  the blocklist (which is possible via encoding tricks, obscure tools,
  or language one-liners).

  RISKS:
  - LLM can read ANY file your user can read (SSH keys, configs, etc.)
  - LLM can write/delete ANY file your user can write
  - LLM can start network connections
  - LLM can interact with other processes

  TO FIX: Install bubblewrap:
    sudo apt install bubblewrap    (Debian/Ubuntu)
    sudo dnf install bubblewrap    (Fedora)
    sudo pacman -S bubblewrap      (Arch)

  Do you want to continue in degraded mode? This is NOT recommended
  for any use beyond development testing with trusted prompts.
================================================================================
"""


# ---------------------------------------------------------------------------
# Isolation backend
# ---------------------------------------------------------------------------

class IsolationBackend(enum.Enum):
    """Sandbox isolation mechanism."""
    BWRAP = "bwrap"      # Full kernel namespace isolation
    TEMPDIR = "tempdir"  # Tempdir only — NOT real isolation


class ApprovalState(enum.Enum):
    """Approval state for sandbox file copy-out (S116).

    Files created inside the sandbox MUST go through this state
    machine before they can be copied to the host filesystem.
    No auto-approve is ever allowed.
    """
    PENDING = "pending"      # Files exist but not yet reviewed
    APPROVED = "approved"    # User explicitly approved specific paths
    REJECTED = "rejected"    # User rejected copy-out


def _detect_bwrap() -> tuple[bool, str]:
    """Detect if bubblewrap is available and functional.

    Performs a real functional test (not just binary existence) because
    some container environments block the namespace syscalls bwrap needs.

    Returns:
        (available, message) — message is version string on success,
        or failure reason on failure.
    """
    try:
        # Step 1: check binary exists and get version
        ver_result = subprocess.run(
            ["bwrap", "--version"],
            capture_output=True,
            timeout=5,
        )
        if ver_result.returncode != 0:
            return False, "bwrap binary returned non-zero on --version"

        version = ver_result.stdout.decode("utf-8", errors="replace").strip()

        # Step 2: functional test — actually create a namespace
        # This catches environments where bwrap exists but namespaces
        # are blocked (e.g., unprivileged Docker containers).
        # We bind only the minimum needed to run /bin/echo (or /usr/bin/echo).
        test_cmd = ["bwrap"]
        for path in ["/usr", "/bin", "/lib", "/lib64"]:
            if os.path.exists(path):
                test_cmd.extend(["--ro-bind", path, path])
        test_cmd.extend([
            "--dev", "/dev",
            "--proc", "/proc",
            "--tmpfs", "/tmp",
            "--unshare-net",
            "--unshare-pid",
            "--new-session",
            "--die-with-parent",
            "--chdir", "/",
            "echo", "sandbox-functional-test-ok",
        ])

        test_result = subprocess.run(
            test_cmd,
            capture_output=True,
            timeout=10,
        )

        if test_result.returncode != 0:
            stderr = test_result.stderr.decode("utf-8", errors="replace")
            return False, (
                f"bwrap namespace test failed (rc={test_result.returncode}): "
                f"{stderr[:200]}"
            )

        stdout = test_result.stdout.decode("utf-8", errors="replace").strip()
        if "sandbox-functional-test-ok" not in stdout:
            return False, "bwrap functional test: unexpected output"

        return True, version

    except FileNotFoundError:
        return False, "bwrap binary not found in PATH"
    except subprocess.TimeoutExpired:
        return False, "bwrap detection timed out"
    except OSError as exc:
        return False, f"bwrap OS error: {exc}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SandboxConfig:
    """Configuration for the sandbox manager."""

    enabled: bool = True
    isolation_backend: str = "auto"  # "auto", "bwrap", "tempdir"
    require_degraded_confirmation: bool = True
    workspace_base: str = "/tmp/opti-oignon-sandboxes"
    command_timeout: int = 30
    max_output_bytes: int = 65536
    max_stderr_bytes: int = 16384
    max_concurrent_sessions: int = 5
    audit_db_path: str = "sandbox_audit.db"
    blocked_commands: list[str] = field(default_factory=list)
    blocked_patterns: list[str] = field(default_factory=list)
    bwrap_ro_binds: list[str] = field(default_factory=lambda: [
        "/usr", "/bin", "/lib", "/lib64",
        "/etc/ld.so.cache", "/etc/alternatives",
        "/etc/python3", "/etc/localtime", "/etc/ssl/certs",
    ])
    bwrap_never_bind: list[str] = field(default_factory=list)
    disable_web_search_in_sandbox: bool = False
    # S124: Strict mode — refuse ALL code execution if bwrap is unavailable
    strict_mode: bool = True
    # S209 (Bloc 0): per-sandbox resource caps. limits_enabled gates the caps;
    # resource_backend selects the mechanism. "rlimit" (default) installs
    # RLIMIT_* via a preexec hook (dependency-free, native, per-process AS and
    # per-uid NPROC semantics). "cgroup" wraps the launch in a transient
    # systemd --user scope (stronger aggregate accounting) when systemd-run is
    # available, and falls back to rlimit otherwise — it never disables caps.
    limits_enabled: bool = True
    resource_backend: str = "rlimit"
    limit_memory_bytes: int = _CAP_MEMORY_BYTES_DEFAULT
    limit_nproc: int = _CAP_NPROC_DEFAULT
    limit_fsize_bytes: int = _CAP_FSIZE_BYTES_DEFAULT
    limit_cpu_seconds: int = _CAP_CPU_SECONDS_DEFAULT
    tmpfs_size_bytes: int = _CAP_TMPFS_BYTES_DEFAULT
    # S209 (Bloc 0): seccomp denylist. seccomp_enabled builds and passes the
    # filter on every bwrap launch. seccomp_required makes a build/pass failure
    # REFUSE the launch (fail-secure) rather than run unfiltered; flip it off
    # only with the loud warning logged below.
    seccomp_enabled: bool = True
    seccomp_required: bool = True
    # S210 (Bloc 1): workspace lifecycle. Workspaces default ephemeral under
    # workspace_base (which is volatile under /tmp and does not survive a
    # reboot). workspace_persistent=True skips the startup reconcile so files
    # under a configured persistent base survive restarts -- the documented
    # trade-off is a widened disposable window (data persists on disk between
    # runs; write-back to the host stays approval-gated regardless, Bloc 3).
    # idle_ttl_seconds destroys idle, unbound workspaces lazily (0 disables).
    # disk_soft_limit_bytes is the per-workspace copy-in soft quota: a copy-in
    # that would exceed it is refused; the workspace itself is never killed.
    workspace_persistent: bool = False
    reconcile_on_start: bool = True
    idle_ttl_seconds: int = _TTL_SECONDS_DEFAULT
    disk_soft_limit_bytes: int = _DISK_SOFT_BYTES_DEFAULT
    # S211 (Bloc 2): copy-in. host_share_roots is the allowlist the host
    # browse and clone are confined to (normalized at load: expanduser +
    # realpath, "/" refused, non-directories dropped; empty/unset defaults to
    # the user's home). The upload caps bound a single multipart request
    # (count and per-file bytes); the clone caps bound a single clone (bytes
    # and file count). The request/clone TOTAL is additionally bounded by the
    # S210 per-workspace disk soft quota -- the caps and the quota both
    # apply, neither subsumes the other.
    host_share_roots: list[str] = field(default_factory=list)
    upload_max_files: int = _UPLOAD_MAX_FILES_DEFAULT
    upload_max_file_bytes: int = _UPLOAD_MAX_FILE_BYTES_DEFAULT
    clone_max_bytes: int = _CLONE_MAX_BYTES_DEFAULT
    clone_max_files: int = _CLONE_MAX_FILES_DEFAULT
    # S212 (Bloc 3): the diff review entry bound. A diff whose classified
    # entry count (baseline union live files) would exceed this REFUSES
    # instead of truncating or paginating: the review gate must present the
    # whole change set or nothing.
    diff_max_entries: int = _DIFF_MAX_ENTRIES_DEFAULT

    # S213 (Bloc 4): the provision-run timeout. NOT a network default --
    # the per-workspace network flag has no configuration surface at all.
    provision_timeout_seconds: int = _PROVISION_TIMEOUT_DEFAULT

    def __post_init__(self) -> None:
        # Clamp every cap into its documented range. Out-of-range values are
        # corrected, never disabled (S209). resource_backend falls back to the
        # dependency-free default if an unknown value is configured.
        self.limit_memory_bytes = _clamp(
            self.limit_memory_bytes, _CAP_MEMORY_BYTES_MIN, _CAP_MEMORY_BYTES_MAX
        )
        self.limit_nproc = _clamp(
            self.limit_nproc, _CAP_NPROC_MIN, _CAP_NPROC_MAX
        )
        self.limit_fsize_bytes = _clamp(
            self.limit_fsize_bytes, _CAP_FSIZE_BYTES_MIN, _CAP_FSIZE_BYTES_MAX
        )
        self.limit_cpu_seconds = _clamp(
            self.limit_cpu_seconds, _CAP_CPU_SECONDS_MIN, _CAP_CPU_SECONDS_MAX
        )
        self.tmpfs_size_bytes = _clamp(
            self.tmpfs_size_bytes, _CAP_TMPFS_BYTES_MIN, _CAP_TMPFS_BYTES_MAX
        )
        if self.resource_backend not in _RESOURCE_BACKENDS:
            self.resource_backend = "rlimit"
        # S210 lifecycle clamps (same discipline: correct, never disable).
        self.idle_ttl_seconds = _clamp(
            self.idle_ttl_seconds, _TTL_SECONDS_MIN, _TTL_SECONDS_MAX
        )
        self.disk_soft_limit_bytes = _clamp(
            self.disk_soft_limit_bytes,
            _DISK_SOFT_BYTES_MIN,
            _DISK_SOFT_BYTES_MAX,
        )
        # S211 copy-in clamps + share-root normalization (correct, never
        # disable; "/" is never an allowed share root).
        self.upload_max_files = _clamp(
            self.upload_max_files, _UPLOAD_MAX_FILES_MIN, _UPLOAD_MAX_FILES_MAX
        )
        self.upload_max_file_bytes = _clamp(
            self.upload_max_file_bytes,
            _UPLOAD_MAX_FILE_BYTES_MIN,
            _UPLOAD_MAX_FILE_BYTES_MAX,
        )
        self.clone_max_bytes = _clamp(
            self.clone_max_bytes, _CLONE_MAX_BYTES_MIN, _CLONE_MAX_BYTES_MAX
        )
        self.clone_max_files = _clamp(
            self.clone_max_files, _CLONE_MAX_FILES_MIN, _CLONE_MAX_FILES_MAX
        )
        # S212 diff review bound (same discipline: correct, never disable).
        self.diff_max_entries = _clamp(
            self.diff_max_entries, _DIFF_MAX_ENTRIES_MIN, _DIFF_MAX_ENTRIES_MAX
        )
        # S213 provision timeout (same discipline).
        self.provision_timeout_seconds = _clamp(
            self.provision_timeout_seconds,
            _PROVISION_TIMEOUT_MIN,
            _PROVISION_TIMEOUT_MAX,
        )
        self.host_share_roots = _normalize_share_roots(self.host_share_roots)


def _load_config() -> SandboxConfig:
    """Load sandbox configuration from YAML with safe defaults."""
    cfg = SandboxConfig()
    try:
        import yaml
        if os.path.isfile(_CONFIG_PATH):
            with open(_CONFIG_PATH, encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
            cfg = SandboxConfig(
                enabled=raw.get("enabled", True),
                isolation_backend=raw.get("isolation_backend", "auto"),
                require_degraded_confirmation=raw.get(
                    "require_degraded_confirmation", True
                ),
                workspace_base=raw.get(
                    "workspace_base", "/tmp/opti-oignon-sandboxes"
                ),
                command_timeout=raw.get("command_timeout", 30),
                max_output_bytes=raw.get("max_output_bytes", 65536),
                max_stderr_bytes=raw.get("max_stderr_bytes", 16384),
                max_concurrent_sessions=raw.get("max_concurrent_sessions", 5),
                audit_db_path=raw.get("audit_db_path", "sandbox_audit.db"),
                blocked_commands=raw.get("blocked_commands", []),
                blocked_patterns=raw.get("blocked_patterns", []),
                bwrap_ro_binds=raw.get("bwrap_ro_binds", [
                    "/usr", "/bin", "/lib", "/lib64",
                ]),
                bwrap_never_bind=raw.get("bwrap_never_bind", []),
                disable_web_search_in_sandbox=raw.get(
                    "disable_web_search_in_sandbox", False
                ),
                strict_mode=raw.get("strict_mode", True),
                limits_enabled=raw.get("limits_enabled", True),
                resource_backend=raw.get("resource_backend", "rlimit"),
                limit_memory_bytes=raw.get(
                    "limit_memory_bytes", _CAP_MEMORY_BYTES_DEFAULT
                ),
                limit_nproc=raw.get("limit_nproc", _CAP_NPROC_DEFAULT),
                limit_fsize_bytes=raw.get(
                    "limit_fsize_bytes", _CAP_FSIZE_BYTES_DEFAULT
                ),
                limit_cpu_seconds=raw.get(
                    "limit_cpu_seconds", _CAP_CPU_SECONDS_DEFAULT
                ),
                tmpfs_size_bytes=raw.get(
                    "tmpfs_size_bytes", _CAP_TMPFS_BYTES_DEFAULT
                ),
                seccomp_enabled=raw.get("seccomp_enabled", True),
                seccomp_required=raw.get("seccomp_required", True),
                workspace_persistent=raw.get("workspace_persistent", False),
                reconcile_on_start=raw.get("reconcile_on_start", True),
                idle_ttl_seconds=raw.get(
                    "idle_ttl_seconds", _TTL_SECONDS_DEFAULT
                ),
                disk_soft_limit_bytes=raw.get(
                    "disk_soft_limit_bytes", _DISK_SOFT_BYTES_DEFAULT
                ),
                host_share_roots=raw.get("host_share_roots", []),
                upload_max_files=raw.get(
                    "upload_max_files", _UPLOAD_MAX_FILES_DEFAULT
                ),
                upload_max_file_bytes=raw.get(
                    "upload_max_file_bytes", _UPLOAD_MAX_FILE_BYTES_DEFAULT
                ),
                clone_max_bytes=raw.get(
                    "clone_max_bytes", _CLONE_MAX_BYTES_DEFAULT
                ),
                clone_max_files=raw.get(
                    "clone_max_files", _CLONE_MAX_FILES_DEFAULT
                ),
                diff_max_entries=raw.get(
                    "diff_max_entries", _DIFF_MAX_ENTRIES_DEFAULT
                ),
                provision_timeout_seconds=raw.get(
                    "provision_timeout_seconds", _PROVISION_TIMEOUT_DEFAULT
                ),
            )
    except Exception as exc:
        logger.warning("Failed to load sandbox config: %s", exc)

    # S124: security.yaml strict_mode overrides sandbox.yaml
    try:
        import yaml
        sec_path = os.path.join(
            os.path.dirname(__file__), "config", "security.yaml"
        )
        if os.path.isfile(sec_path):
            with open(sec_path, encoding="utf-8") as fh:
                sec_raw = yaml.safe_load(fh) or {}
            sandbox_sec = sec_raw.get("sandbox", {})
            if isinstance(sandbox_sec, dict) and "strict_mode" in sandbox_sec:
                cfg.strict_mode = bool(sandbox_sec["strict_mode"])
    except Exception:
        pass

    return cfg


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SandboxSession:
    """Tracks a single sandbox session.

    S210 (Bloc 1): a session is a user-owned workspace with a stable id.
    Stored state lives here (label, owner, binding, activity); derived
    figures (age, running, approximate disk use) are computed by
    ``list_sessions`` so they are never stale on the object.
    """

    session_id: str
    workspace_path: str
    isolation_backend: IsolationBackend = IsolationBackend.TEMPDIR
    created_at: float = field(default_factory=time.time)
    active: bool = True
    command_count: int = 0
    # S116: Approval state machine for copy-out
    approval_state: ApprovalState = ApprovalState.PENDING
    approved_paths: set[str] = field(default_factory=set)
    approved_at: float | None = None
    # S210 (Bloc 1): workspace lifecycle fields. label is the optional human
    # name; owner_user_id follows the effective_user_id isolation pattern
    # (the workspace implies its owning user); bound_conversation_id mirrors
    # the sandbox_workspace binding store (write-through, the store is the
    # mutation point); network_enabled is the per-workspace network flag --
    # default False, flipped ONLY by set_network_enabled (S213, Bloc 4: an
    # explicit user action behind the Daily-only gate; never a config
    # default, never model-triggerable); last_activity drives the idle TTL;
    # timeout_override is the per-sandbox command timeout.
    label: str = ""
    owner_user_id: str = "local"
    bound_conversation_id: str | None = None
    network_enabled: bool = False
    last_activity: float = field(default_factory=time.time)
    timeout_override: int | None = None
    # S212 (Bloc 3): deletions confirmed for apply-to-host. PARALLEL to
    # approved_paths by design: approve_files validates os.path.isfile so a
    # deleted entry can never enter approved_paths, and a blanket approve-all
    # can never include a deletion. The load-bearing check is in the apply
    # writer: it deletes only paths that are BOTH confirmed here AND
    # classified "deleted" by the recomputed diff -- confirming a live path
    # can never delete it.
    confirmed_deletions: set[str] = field(default_factory=set)


class WorkspaceQuotaExceeded(Exception):
    """A copy-in would push the workspace past its disk soft quota.

    Raised by the inject paths (S210). Soft semantics: the copy-in is
    refused with this error; the workspace itself is never destroyed.
    """


@dataclass
class CommandResult:
    """Result of a sandboxed command execution."""

    stdout: str = ""
    stderr: str = ""
    return_code: int = -1
    timed_out: bool = False
    blocked: bool = False
    block_reason: str = ""
    truncated_stdout: bool = False
    truncated_stderr: bool = False
    isolation_backend: str = ""


# ---------------------------------------------------------------------------
# Audit log (SQLite)
# ---------------------------------------------------------------------------

class AuditLog:
    """SQLite-backed audit log for sandbox command history."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a SQLite connection.

        S136 audit fix: routes through get_encrypted_connection().
        """
        return safe_connect(self._db_path)

    def _init_db(self) -> None:
        """Create the audit table if it does not exist."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sandbox_audit (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        command TEXT NOT NULL,
                        return_code INTEGER,
                        blocked INTEGER DEFAULT 0,
                        block_reason TEXT DEFAULT '',
                        timed_out INTEGER DEFAULT 0,
                        stdout_len INTEGER DEFAULT 0,
                        stderr_len INTEGER DEFAULT 0,
                        isolation_backend TEXT DEFAULT ''
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_audit_session
                    ON sandbox_audit(session_id)
                """)
                conn.commit()
            finally:
                conn.close()

    def log_command(
        self,
        session_id: str,
        command: str,
        result: CommandResult,
    ) -> None:
        """Record a command execution in the audit log."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO sandbox_audit
                        (session_id, timestamp, command, return_code,
                         blocked, block_reason, timed_out,
                         stdout_len, stderr_len, isolation_backend)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        time.time(),
                        command,
                        result.return_code,
                        1 if result.blocked else 0,
                        result.block_reason,
                        1 if result.timed_out else 0,
                        len(result.stdout),
                        len(result.stderr),
                        result.isolation_backend,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def get_session_log(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieve all audit entries for a session."""
        with self._lock:
            conn = self._get_conn()
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    "SELECT * FROM sandbox_audit WHERE session_id = ? "
                    "ORDER BY id ASC",
                    (session_id,),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_all_logs(self, limit: int = 100) -> list[dict[str, Any]]:
        """Retrieve recent audit entries across all sessions."""
        with self._lock:
            conn = self._get_conn()
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    "SELECT * FROM sandbox_audit ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def clear(self) -> None:
        """Clear all audit log entries."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("DELETE FROM sandbox_audit")
                # S116: clear approval audit if table exists
                try:
                    conn.execute("DELETE FROM sandbox_approval_audit")
                except sqlite3.OperationalError:
                    pass  # Table may not exist yet
                conn.commit()
            finally:
                conn.close()

    def _ensure_approval_table(self) -> None:
        """Create the approval audit table if it does not exist (S116)."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sandbox_approval_audit (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        action TEXT NOT NULL,
                        paths TEXT DEFAULT '',
                        dest_dir TEXT DEFAULT '',
                        detail TEXT DEFAULT ''
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_approval_session
                    ON sandbox_approval_audit(session_id)
                """)
                conn.commit()
            finally:
                conn.close()

    def log_approval(
        self,
        session_id: str,
        action: str,
        paths: list[str] | None = None,
        dest_dir: str = "",
        detail: str = "",
    ) -> None:
        """Record an approval/rejection/copy-out event (S116)."""
        self._ensure_approval_table()
        import json
        paths_json = json.dumps(paths or [])
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO sandbox_approval_audit
                        (session_id, timestamp, action, paths, dest_dir, detail)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, time.time(), action, paths_json, dest_dir, detail),
                )
                conn.commit()
            finally:
                conn.close()

    def get_approval_log(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieve all approval audit entries for a session (S116)."""
        self._ensure_approval_table()
        with self._lock:
            conn = self._get_conn()
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    "SELECT * FROM sandbox_approval_audit WHERE session_id = ? "
                    "ORDER BY id ASC",
                    (session_id,),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()


# ---------------------------------------------------------------------------
# Command validation (defense-in-depth, applied on ALL backends)
# ---------------------------------------------------------------------------

class CommandValidator:
    """Validates commands against blocked lists and dangerous patterns.

    This is defense-in-depth: applied even when bwrap is the backend.
    A blocklist can never be exhaustive, but it catches the obvious
    cases and adds friction against accidental damage.

    S81 additions:
    - Tracks recently created files and inspects their content before
      allowing bash execution that references them (write-then-execute
      attack vector).
    - Extended base64-pipe-to-shell detection (covers more encoding
      bypass patterns).
    - Extended Python subprocess pattern detection.
    """

    # S209 boundary note: this regex command denylist is SECONDARY
    # defense-in-depth ONLY. It is bypassable -- base64/eval, here-docs,
    # write-then-execute, obscure interpreters -- and must never be relied on
    # as the security boundary. The boundary is the namespace isolation that
    # bwrap provides (hardened in Bloc 0: clean env, full namespace
    # cloistering, seccomp denylist, resource caps); these patterns only add
    # friction against the obvious and accidental cases.

    # Patterns indicating a command tries to execute a file by path
    _EXEC_FILE_PATTERNS = re.compile(
        r"(?:^|\s|&&|\|\||;)"
        r"(?:bash|sh|zsh|python[23]?|perl|ruby|node|chmod\s+\+x)\s+"
        r"['\"]?([^\s;|&'\"]+)",
        re.IGNORECASE,
    )

    # Patterns for ./script execution
    _DOT_SLASH_PATTERN = re.compile(
        r"(?:^|\s|&&|\|\||;)\./([^\s;|&'\"]+)",
    )

    # Dangerous content patterns to detect in created file contents
    _DANGEROUS_FILE_CONTENT = [
        re.compile(r"\b(curl|wget|nc|ncat|netcat)\b", re.IGNORECASE),
        re.compile(r"\b(socket|urllib|requests|httpx|aiohttp)\b", re.IGNORECASE),
        re.compile(r"\bsubprocess\b.*\b(Popen|call|run|check_output)\b", re.IGNORECASE),
        re.compile(r"\bos\.system\b", re.IGNORECASE),
        re.compile(r"\bexec\s*\(", re.IGNORECASE),
        re.compile(r"base64.*\|\s*(bash|sh|zsh|python)", re.IGNORECASE),
        re.compile(r"\bimport\s+(subprocess|socket|http|urllib|ftplib|smtplib)\b"),
        re.compile(r"\b(rm\s+-rf\s+/|dd\s+if=)"),
    ]

    def __init__(self, config: SandboxConfig):
        self._blocked_commands = [
            cmd.lower() for cmd in (config.blocked_commands or [])
        ]
        self._blocked_patterns: list[re.Pattern] = []
        for pattern_str in (config.blocked_patterns or []):
            try:
                self._blocked_patterns.append(
                    re.compile(pattern_str, re.IGNORECASE)
                )
            except re.error as exc:
                logger.warning(
                    "Invalid blocked pattern '%s': %s", pattern_str, exc
                )

        # S81: Track recently created files for write-then-execute detection
        self._recent_files: dict[str, str] = {}
        self._recent_files_lock = threading.Lock()

    def register_created_file(self, relative_path: str, content: str) -> None:
        """Register a file that was just created in the sandbox.

        This allows the validator to inspect the file content before
        allowing bash commands that reference it.

        Args:
            relative_path: Path relative to the sandbox workspace root.
            content: The file content that was written.
        """
        # Normalize path: strip /workspace/ prefix
        clean = relative_path
        if clean.startswith("/workspace/"):
            clean = clean[len("/workspace/"):]
        elif clean.startswith("/"):
            clean = clean[1:]
        with self._recent_files_lock:
            self._recent_files[clean] = content

    def clear_recent_files(self) -> None:
        """Clear the recent files registry (e.g., on session destroy)."""
        with self._recent_files_lock:
            self._recent_files.clear()

    def validate(self, command: str) -> tuple[bool, str]:
        """Validate a command. Returns (is_safe, reason).

        Returns (True, '') if the command passes all checks,
        or (False, reason) if it should be blocked.
        """
        if not command or not command.strip():
            return False, "Empty command"

        cmd_lower = command.strip().lower()

        # Check blocked command prefixes
        for blocked in self._blocked_commands:
            if cmd_lower.startswith(blocked.rstrip()):
                return False, f"Blocked command: {blocked.strip()}"

        # Check blocked patterns
        for pattern in self._blocked_patterns:
            if pattern.search(command):
                return False, f"Blocked pattern: {pattern.pattern}"

        # Additional hardcoded safety checks that must never be bypassed
        # regardless of config contents

        # Prevent rm targeting root or absolute paths outside workspace
        if re.search(r"\brm\b.*\s+/(?!workspace)", command):
            return False, "rm targeting paths outside /workspace"

        # Block eval/exec with network or OS calls
        if re.search(
            r"\b(eval|exec)\b.*\b(socket|urllib|requests|http)\b",
            command,
            re.IGNORECASE,
        ):
            return False, "Blocked eval/exec with network access"

        # Block python -c with network modules (defense-in-depth,
        # catches patterns the YAML regex might miss)
        if re.search(
            r"python[23]?\s+-c\s+.*\b(urllib|http\.|requests\.|"
            r"httpx|aiohttp|socket|ftplib|smtplib|xmlrpc|"
            r"telnetlib|poplib|imaplib|nntplib)",
            command,
            re.IGNORECASE,
        ):
            return False, "Blocked python -c with network module"

        # S81: Block python -c with subprocess (constructed args attack)
        if re.search(
            r"python[23]?\s+-c\s+.*\b(subprocess|os\.system|os\.popen|"
            r"os\.exec[a-z]*|pty\.spawn)\b",
            command,
            re.IGNORECASE,
        ):
            return False, "Blocked python -c with subprocess/os.exec"

        # S81: Block base64-encoded commands piped to execution
        # Extended: covers hex decode, xxd, and printf byte sequences
        if re.search(
            r"base64\s+(-d|--decode).*\|\s*(bash|sh|zsh|python|perl|ruby)",
            command,
            re.IGNORECASE,
        ):
            return False, "Blocked base64 decode piped to shell"

        # S81: Block echo with base64 piped to decode then execute
        if re.search(
            r"echo\s+['\"]?[A-Za-z0-9+/=]{20,}['\"]?\s*\|\s*base64\s+(-d|--decode)",
            command,
        ):
            return False, "Blocked echo of base64 payload to decode"

        # S81: Block xxd reverse piped to shell
        if re.search(
            r"xxd\s+-r.*\|\s*(bash|sh|zsh|python)",
            command,
            re.IGNORECASE,
        ):
            return False, "Blocked xxd reverse piped to shell"

        # S81: Block Python subprocess with shell=True or Popen
        if re.search(
            r"python[23]?\s+-c\s+.*\bsubprocess\.(Popen|call|run|"
            r"check_output|check_call)\b",
            command,
            re.IGNORECASE,
        ):
            return False, "Blocked python -c with subprocess calls"

        # S81: Check write-then-execute attack vector
        safe, reason = self._check_file_execution(command)
        if not safe:
            return False, reason

        return True, ""

    def _check_file_execution(self, command: str) -> tuple[bool, str]:
        """Check if command executes a recently created file with dangerous content.

        Detects patterns like:
          bash script.sh
          python script.py
          ./script.sh
          sh -c 'source script.sh'
          chmod +x script.sh && ./script.sh

        Args:
            command: The shell command to inspect.

        Returns:
            (True, '') if safe, (False, reason) if blocked.
        """
        with self._recent_files_lock:
            if not self._recent_files:
                return True, ""

        # Extract file references from the command
        referenced_files: set[str] = set()
        for match in self._EXEC_FILE_PATTERNS.finditer(command):
            referenced_files.add(match.group(1))
        for match in self._DOT_SLASH_PATTERN.finditer(command):
            referenced_files.add(match.group(1))

        if not referenced_files:
            return True, ""

        # Check each referenced file against recent creates
        with self._recent_files_lock:
            for ref_path in referenced_files:
                # Normalize: strip leading ./ or /workspace/
                clean = ref_path
                if clean.startswith("./"):
                    clean = clean[2:]
                if clean.startswith("/workspace/"):
                    clean = clean[len("/workspace/"):]

                content = self._recent_files.get(clean)
                if content is None:
                    continue

                # Inspect file content for dangerous patterns
                for pattern in self._DANGEROUS_FILE_CONTENT:
                    if pattern.search(content):
                        return False, (
                            f"Blocked execution of recently created file "
                            f"'{clean}': content matches dangerous pattern "
                            f"'{pattern.pattern}'"
                        )

        return True, ""


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------

def validate_sandbox_path(
    workspace_root: str,
    requested_path: str,
) -> tuple[bool, str, str]:
    """Validate that a path stays within the sandbox workspace.

    Returns (is_valid, resolved_path, error_message).
    The resolved path is absolute and guaranteed to be inside the workspace.
    """
    if not requested_path:
        return False, "", "Empty path"

    workspace_real = os.path.realpath(workspace_root)

    # If path is absolute, it must be under /workspace/ (sandbox virtual root)
    if os.path.isabs(requested_path):
        if requested_path.startswith("/workspace/"):
            # Map /workspace/X to actual workspace path
            relative = requested_path[len("/workspace/"):]
            candidate = os.path.join(workspace_real, relative)
        elif requested_path == "/workspace":
            candidate = workspace_real
        else:
            return False, "", (
                f"Absolute path '{requested_path}' is outside sandbox. "
                f"Use /workspace/ prefix or relative paths."
            )
    else:
        candidate = os.path.join(workspace_real, requested_path)

    # Resolve to real path (follows symlinks)
    resolved = os.path.realpath(candidate)

    # Strict containment check
    if (
        not resolved.startswith(workspace_real + os.sep)
        and resolved != workspace_real
    ):
        return False, "", (
            f"Path traversal detected: resolved path '{resolved}' "
            f"escapes workspace '{workspace_real}'"
        )

    # Check for symlink escape (defense-in-depth, already covered above)
    if os.path.islink(candidate):
        link_target = os.path.realpath(candidate)
        if (
            not link_target.startswith(workspace_real + os.sep)
            and link_target != workspace_real
        ):
            return False, "", (
                f"Symlink escape detected: '{candidate}' -> '{link_target}'"
            )

    return True, resolved, ""


# ---------------------------------------------------------------------------
# Bwrap command builder
# ---------------------------------------------------------------------------

def _build_bwrap_command(
    command: str,
    workspace: str,
    config: SandboxConfig,
    seccomp_fd: int | None = None,
    *,
    allow_network: bool = False,
) -> list[str]:
    """Build a bubblewrap command line for isolated execution.

    The resulting process sees:
    - /workspace (read-write) -> actual workspace directory
    - /usr, /bin, /lib, /lib64 (read-only) -> host system binaries
    - /dev (minimal: null, zero, urandom, etc.)
    - /proc (isolated PID namespace, only sandbox processes visible)
    - /tmp (isolated tmpfs)
    - A clean environment (--clearenv): only PATH, HOME, TMPDIR, PWD,
      LANG, LC_ALL are set; no host environment variable is inherited
    - NO network, NO host PIDs, NO host home/var/etc

    S213 (Bloc 4): ``allow_network`` is keyword-only and defaults to False,
    keeping the default argv byte-identical (pinned by test). When True --
    the PROVISION run only, never a task run -- ``--unshare-net`` is
    omitted and the name-resolution files (the realpath of
    /etc/resolv.conf, /etc/hosts, /etc/nsswitch.conf, each only if it
    exists) are ro-bound file-level so the pinned installer can resolve
    the index; /etc/ssl/certs is already in the standard ro-binds. Every
    other boundary (clearenv, pid/ipc/uts/cgroup namespaces, seccomp,
    rlimits) is unchanged on the network-on argv.

    Args:
        command: Shell command to execute.
        workspace: Host path to the sandbox workspace.
        config: Sandbox configuration.
        seccomp_fd: Inheritable fd carrying the seccomp filter, if any.
        allow_network: Provision-run network grant (S213); default False.

    Returns:
        Complete command list for subprocess.run().
    """
    # Merge configured never-bind with hardcoded list
    never_bind = _HARDCODED_NEVER_BIND | frozenset(
        config.bwrap_never_bind or []
    )

    cmd = ["bwrap"]

    # Clean environment (S-01 / C-01): clear the inherited environment so no
    # host secret -- an env-provided OPTI_ENCRYPTION_KEY, a search API key, or a
    # SQLCipher passphrase -- reaches untrusted, model-driven code running in the
    # sandbox, then set only the minimal variables the workspace needs.
    # --clearenv must precede --setenv: variables set afterwards survive the
    # clear. This mirrors the env scrub the tempdir fallback already performs.
    cmd.append("--clearenv")
    cmd.extend(["--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin"])
    cmd.extend(["--setenv", "HOME", "/workspace"])
    cmd.extend(["--setenv", "TMPDIR", "/tmp"])
    cmd.extend(["--setenv", "PWD", "/workspace"])
    cmd.extend(["--setenv", "LANG", "C.UTF-8"])
    cmd.extend(["--setenv", "LC_ALL", "C.UTF-8"])

    # Read-only system path bindings
    for bind_path in (config.bwrap_ro_binds or []):
        # Security: skip any path in the never-bind list
        skip = False
        for blocked in never_bind:
            if bind_path == blocked or bind_path.startswith(blocked + "/"):
                logger.warning(
                    "Refusing to bind blocked path into sandbox: %s",
                    bind_path,
                )
                skip = True
                break
        if skip:
            continue

        # Only bind paths that actually exist on the host
        if os.path.exists(bind_path):
            cmd.extend(["--ro-bind", bind_path, bind_path])

    # Read-write workspace mounted at /workspace
    cmd.extend(["--bind", workspace, "/workspace"])

    # Minimal /dev (null, zero, urandom, etc. — NOT the real /dev)
    cmd.extend(["--dev", "/dev"])

    # Isolated /proc (shows only sandbox PID namespace processes)
    cmd.extend(["--proc", "/proc"])

    # Isolated /tmp (tmpfs, separate from host) with a size cap (S209). The
    # --size option applies to the immediately following filesystem mount, so
    # it must precede --tmpfs. The read-write workspace is a real bind, not a
    # tmpfs: its disk is bounded by RLIMIT_FSIZE (per file) plus the Bloc 1
    # workspace quota, not by --size.
    cmd.extend([
        "--size", str(int(config.tmpfs_size_bytes)),
        "--tmpfs", "/tmp",
    ])

    # Namespace isolation flags. net and pid were already unshared; S209 adds
    # ipc, uts, and cgroup so the child shares no host SysV/POSIX IPC, sees no
    # host hostname/domainname, and has no host cgroup view. net and pid stay
    # listed explicitly (not folded into --unshare-all) so each isolation
    # boundary is individually pinned and the user namespace / uid-gid mapping
    # is left untouched.
    #
    # S213 (Bloc 4): --unshare-net is unconditional on every task run; ONLY
    # the provision run (allow_network=True, reachable solely through
    # execute_provision_command behind the Daily-only gate) omits it. Raw
    # --share-net does not exist as a grant here: bwrap networking is simply
    # the absence of --unshare-net, scoped by the fact that the provision
    # command is a fixed, server-built installer line -- arbitrary task code
    # never runs on this argv (spec 8.4).
    if not allow_network:
        cmd.append("--unshare-net")   # No network access whatsoever
    cmd.extend([
        "--unshare-pid",       # Isolated PID namespace
        "--unshare-ipc",       # No host SysV/POSIX IPC sharing (S209)
        "--unshare-uts",       # No host hostname/domainname leak (S209)
        "--unshare-cgroup",    # No host cgroup view (S209)
        "--new-session",       # No terminal/tty escape
        "--die-with-parent",   # Kill sandbox if parent dies
    ])

    # S213: name resolution for the provision run only. File-level ro-binds
    # (never directories); resolv.conf is bound via its realpath because on
    # systemd-resolved hosts /etc/resolv.conf is a symlink into /run, which
    # is (and stays) in the never-bind list. These files carry no secrets.
    if allow_network:
        for ns_file in ("/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf"):
            try:
                real = os.path.realpath(ns_file)
            except OSError:
                continue
            if os.path.isfile(real):
                cmd.extend(["--ro-bind", real, ns_file])

    # Seccomp denylist filter (S209). The fd is created and passed by
    # _run_bwrap; the filter reduces kernel syscall surface but is NOT the
    # boundary -- the namespaces are. Only emitted when a fd is supplied.
    if seccomp_fd is not None:
        cmd.extend(["--seccomp", str(int(seccomp_fd))])

    # Working directory
    cmd.extend(["--chdir", "/workspace"])

    # The actual command
    cmd.extend(["bash", "-c", command])

    return cmd


# ---------------------------------------------------------------------------
# Resource caps (S209, Bloc 0)
# ---------------------------------------------------------------------------

def _make_rlimit_preexec(config: SandboxConfig):
    """Build a preexec hook that installs RLIMIT_* on the bwrap process tree.

    The returned callable runs in the forked child between fork and exec, so
    the limits are inherited by bwrap and the bash it spawns. It only calls
    resource.setrlimit (no allocation, no logging, no locks), which keeps it
    safe to run after fork in a threaded server.

    Honest limits of this mechanism: RLIMIT_AS bounds address space PER
    PROCESS, not the aggregate of the tree (a fork bomb is bounded by
    RLIMIT_NPROC instead). RLIMIT_NPROC counts processes of the real uid
    system-wide, so it is a coarse guard on a single-user host. The cgroup
    backend is the stronger aggregate accounting where a systemd --user
    session exists.
    """
    mem = int(config.limit_memory_bytes)
    nproc = int(config.limit_nproc)
    fsize = int(config.limit_fsize_bytes)
    cpu = int(config.limit_cpu_seconds)

    def _apply() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return _apply


def _detect_systemd_run() -> bool:
    """Return True if the systemd-run binary is present.

    Presence only; a working systemd --user session is host-verified. The
    cgroup backend falls back to rlimit when this is False, and a transient
    scope that fails to start at runtime surfaces as a normal launch error.
    """
    return shutil.which("systemd-run") is not None


def _systemd_run_prefix(config: SandboxConfig) -> list[str]:
    """Build the transient systemd --user scope prefix for the cgroup backend.

    Bounds memory and PID count via cgroup accounting (MemoryMax, TasksMax).
    CPU in this path is bounded by the wall-clock timeout; RLIMIT_CPU is the
    rlimit backend's CPU bound.
    """
    return [
        "systemd-run", "--user", "--scope", "--quiet",
        "-p", f"MemoryMax={int(config.limit_memory_bytes)}",
        "-p", f"TasksMax={int(config.limit_nproc)}",
        "--",
    ]


# ---------------------------------------------------------------------------
# Sandbox Manager
# ---------------------------------------------------------------------------

class SandboxManager:
    """Manages isolated sandbox environments for LLM tool execution.

    Prefers bubblewrap (bwrap) for true kernel namespace isolation.
    Falls back to tempdir with explicit confirmation if bwrap is
    unavailable. The command blocklist is applied on ALL backends.
    """

    def __init__(self, config: SandboxConfig | None = None):
        self._config = config or _load_config()
        self._sessions: dict[str, SandboxSession] = {}
        self._lock = threading.Lock()
        self._validator = CommandValidator(self._config)
        self._degraded_confirmed = False
        self._unshare_available: bool | None = None  # Lazy-detected
        # S210 (Bloc 1): per-session running-process registry. The stop path
        # needs a handle on the running child, so spawns register their Popen
        # here (under the lock) for the duration of the command and deregister
        # in a finally. A session with an entry is "running"; without, "idle".
        self._running_procs: dict[str, subprocess.Popen] = {}

        # Detect isolation backend
        self._bwrap_available, self._bwrap_info = _detect_bwrap()
        self._isolation_backend = self._resolve_backend()

        # Ensure workspace base exists
        os.makedirs(self._config.workspace_base, mode=0o700, exist_ok=True)

        # S210 (Bloc 1): startup reconcile. Sessions are in-memory and die
        # with the process, so any sandbox-* directory found under
        # workspace_base at startup is an orphan; reap it (reset_*-style)
        # unless the user opted into a persistent base, where surviving
        # directories are deliberate (logged, inert; write-back stays gated).
        if self._config.reconcile_on_start:
            try:
                self.reconcile_workspaces()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Workspace reconcile failed: %s", exc)

        # Initialize audit log
        audit_path = os.path.join(
            self._config.workspace_base, self._config.audit_db_path
        )
        self._audit = AuditLog(audit_path)

        # Log isolation status
        if self._isolation_backend == IsolationBackend.BWRAP:
            logger.info(
                "Sandbox isolation: bwrap (%s)", self._bwrap_info
            )
        else:
            logger.warning(
                "Sandbox isolation: DEGRADED (tempdir only). "
                "Reason: %s", self._bwrap_info
            )

    def _resolve_backend(self) -> IsolationBackend:
        """Determine which isolation backend to use.

        S124: In strict_mode, refuse tempdir fallback entirely.
        """
        preference = self._config.isolation_backend.lower()

        if preference == "bwrap":
            if not self._bwrap_available:
                raise RuntimeError(
                    f"Configuration requires bwrap but it is not available: "
                    f"{self._bwrap_info}. Install bubblewrap or set "
                    f"isolation_backend to 'auto'."
                )
            return IsolationBackend.BWRAP

        if preference == "tempdir":
            if self._config.strict_mode:
                logger.warning(
                    "Sandbox strict_mode is ON but isolation_backend is "
                    "'tempdir'. Allowing tempdir as explicitly requested, "
                    "but this provides NO real isolation."
                )
            return IsolationBackend.TEMPDIR

        # "auto": prefer bwrap, fall back to tempdir (unless strict)
        if self._bwrap_available:
            return IsolationBackend.BWRAP

        if self._config.strict_mode:
            logger.error(
                "Sandbox strict_mode is ON but bwrap is not available: %s. "
                "Code execution will be BLOCKED. Install bubblewrap "
                "(apt install bubblewrap) or set strict_mode: false in "
                "security.yaml.",
                self._bwrap_info,
            )
            # Still return TEMPDIR as the "backend" but execution will
            # be blocked by the strict mode check in execute_command
            return IsolationBackend.TEMPDIR

        return IsolationBackend.TEMPDIR

    @property
    def config(self) -> SandboxConfig:
        """Access the sandbox configuration."""
        return self._config

    @property
    def audit(self) -> AuditLog:
        """Access the audit log."""
        return self._audit

    @property
    def isolation_backend(self) -> IsolationBackend:
        """Current isolation backend."""
        return self._isolation_backend

    @property
    def bwrap_available(self) -> bool:
        """Whether bubblewrap is available on this system."""
        return self._bwrap_available

    @property
    def degraded_mode(self) -> bool:
        """Whether running in degraded (tempdir-only) mode."""
        return self._isolation_backend == IsolationBackend.TEMPDIR

    @property
    def strict_mode(self) -> bool:
        """Whether strict mode is enabled (S124)."""
        return self._config.strict_mode

    @property
    def execution_blocked(self) -> bool:
        """Whether code execution is blocked (strict_mode + no bwrap) (S124)."""
        return (
            self._config.strict_mode
            and not self._bwrap_available
            and self._isolation_backend != IsolationBackend.BWRAP
        )

    def get_isolation_status(self) -> dict[str, Any]:
        """Return comprehensive isolation status for health checks (S124).

        Returns a dict suitable for inclusion in /api/health responses.
        """
        if self._bwrap_available:
            level = "bwrap"
        elif self._config.strict_mode:
            level = "blocked"
        else:
            level = "tempdir"

        return {
            "isolation_level": level,
            "bwrap_available": self._bwrap_available,
            "bwrap_info": self._bwrap_info,
            "strict_mode": self._config.strict_mode,
            "execution_blocked": self.execution_blocked,
            "backend": self._isolation_backend.value,
        }

    @property
    def degraded_confirmed(self) -> bool:
        """Whether the user has confirmed degraded mode."""
        return self._degraded_confirmed

    @property
    def active_session_count(self) -> int:
        """Count of currently active sandbox sessions."""
        with self._lock:
            return sum(1 for s in self._sessions.values() if s.active)

    def confirm_degraded_mode(self) -> str:
        """Explicitly confirm willingness to run in degraded mode.

        Must be called before any sandbox operations when bwrap is
        unavailable and require_degraded_confirmation is True.

        Returns:
            The warning message that was acknowledged.
        """
        self._degraded_confirmed = True
        logger.warning(
            "User confirmed degraded sandbox mode (tempdir only)"
        )
        return DEGRADED_WARNING

    def get_degraded_warning(self) -> str:
        """Get the degraded mode warning message."""
        return DEGRADED_WARNING

    def create_sandbox(
        self,
        session_id: str | None,
        allow_degraded: bool = False,
        label: str = "",
        owner_user_id: str = "local",
        timeout_override: int | None = None,
    ) -> SandboxSession:
        """Create a fresh isolated sandbox workspace.

        Args:
            session_id: Unique identifier for this sandbox session, or None
                to auto-generate one (S210; previously a None leaked into
                the key and the directory prefix).
            allow_degraded: If True, allow tempdir mode without prior
                confirmation. If False (default), degraded mode requires
                confirm_degraded_mode() to have been called first.
            label: Optional human label for the workspace manager (S210).
            owner_user_id: Owning user per the effective_user_id isolation
                pattern (S210); defaults to the single-user "local".
            timeout_override: Optional per-sandbox command timeout in
                seconds (S210); None uses the config default.

        Returns:
            SandboxSession with workspace path set.

        Raises:
            ValueError: If session_id already exists or limits exceeded.
            RuntimeError: If sandbox is disabled, or degraded mode is not
                confirmed when required.
        """
        if not self._config.enabled:
            raise RuntimeError("Sandbox is disabled in configuration")

        # Enforce degraded mode confirmation
        if self.degraded_mode:
            if self._config.require_degraded_confirmation:
                if not allow_degraded and not self._degraded_confirmed:
                    raise RuntimeError(
                        "Sandbox is in DEGRADED mode (no bwrap). "
                        "True filesystem isolation is NOT available. "
                        f"Reason: {self._bwrap_info}. "
                        "Call confirm_degraded_mode() to acknowledge the "
                        "risks, or install bubblewrap for real isolation."
                    )

        # S210: reap idle, unbound workspaces past the TTL before the
        # concurrency check, so a stale workspace never blocks a new one.
        self._sweep_idle_sessions()

        if session_id is None:
            session_id = f"ws-{uuid.uuid4().hex[:12]}"

        with self._lock:
            if session_id in self._sessions:
                raise ValueError(
                    f"Session '{session_id}' already exists"
                )
            active = sum(1 for s in self._sessions.values() if s.active)
            if active >= self._config.max_concurrent_sessions:
                raise ValueError(
                    f"Maximum concurrent sessions reached "
                    f"({self._config.max_concurrent_sessions})"
                )

        # Create isolated temp directory with strict permissions
        workspace = tempfile.mkdtemp(
            prefix=f"sandbox-{session_id}-",
            dir=self._config.workspace_base,
        )
        os.chmod(workspace, 0o700)

        now = time.time()
        session = SandboxSession(
            session_id=session_id,
            workspace_path=workspace,
            isolation_backend=self._isolation_backend,
            created_at=now,
            label=label,
            owner_user_id=owner_user_id,
            last_activity=now,
            timeout_override=timeout_override,
        )

        with self._lock:
            self._sessions[session_id] = session

        logger.info(
            "Sandbox created: session=%s, path=%s, isolation=%s",
            session_id,
            workspace,
            self._isolation_backend.value,
        )
        return session

    def destroy_sandbox(self, session_id: str) -> bool:
        """Destroy a sandbox and remove all its files.

        Args:
            session_id: Session to destroy.

        Returns:
            True if destroyed, False if session not found.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            session.active = False
            # S116: Log pending approved paths that were never copied out
            pending_approved = sorted(session.approved_paths)

        # Remove workspace directory tree
        workspace = session.workspace_path
        if os.path.isdir(workspace):
            shutil.rmtree(workspace, ignore_errors=True)

        with self._lock:
            self._sessions.pop(session_id, None)

        # S81: Clear file tracking for this session
        self._validator.clear_recent_files()

        # S116: Audit the destruction with approval context
        if pending_approved:
            self._audit.log_approval(
                session_id,
                action="session_destroyed",
                paths=pending_approved,
                detail=f"Session destroyed with {len(pending_approved)} approved path(s)",
            )

        logger.info("Sandbox destroyed: session=%s", session_id)
        return True

    def get_session(self, session_id: str) -> SandboxSession | None:
        """Retrieve a session by ID."""
        with self._lock:
            return self._sessions.get(session_id)

    def get_workspace_path(self, session_id: str) -> str | None:
        """Return the workspace root for a session.

        Returns None if the session does not exist or is inactive.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or not session.active:
                return None
            return session.workspace_path

    # -- S210 (Bloc 1): workspace lifecycle ------------------------------

    def is_running(self, session_id: str) -> bool:
        """Whether a command is currently executing in this workspace."""
        with self._lock:
            return session_id in self._running_procs

    def stop_command(self, session_id: str) -> bool:
        """SIGKILL the workspace's running command; keep the workspace.

        The stop path of spec section 4.2: kills the tracked process group
        (the bwrap child and its bash tree die together) and reaps it, then
        marks the workspace idle. Files persist for inspection; the
        workspace is NOT destroyed. Fail-secure no-op semantics: stopping a
        workspace with nothing running returns False and never errors or
        leaks state.

        Args:
            session_id: Target workspace.

        Returns:
            True if a running command was killed, False if nothing was
            running (no-op).

        Raises:
            ValueError: If the session does not exist (the route maps this
                to an honest 404).
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ValueError(f"Session not found: {session_id}")
            proc = self._running_procs.pop(session_id, None)

        if proc is None:
            return False

        try:
            # The child was spawned with start_new_session=True, so its pid
            # is the process-group id: killpg reaches bwrap and its tree.
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            # Already gone or unreachable: fall through to the direct kill
            # and the reap; stopping must never raise past this point.
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=5)
        except Exception:  # pragma: no cover - defensive reap
            pass

        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.last_activity = time.time()

        self._audit.log_approval(
            session_id,
            action="workspace_stopped",
            paths=[],
            detail="Running command killed (SIGKILL, process group)",
        )
        logger.info("Workspace command stopped: session=%s", session_id)
        return True

    def set_binding(self, session_id: str, conversation_id: str | None) -> None:
        """Write-through mirror of the conversation binding (S210).

        The sandbox_workspace binding store is the single mutation point;
        it calls this so the session object (and list_sessions) reflects
        the binding without the manager importing the store (no cycle).
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.bound_conversation_id = conversation_id
                session.last_activity = time.time()

    def touch_activity(self, session_id: str) -> None:
        """Record activity on a workspace (drives the idle TTL)."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.last_activity = time.time()

    def reconcile_workspaces(self) -> int:
        """Reap orphaned workspace directories under workspace_base.

        Sessions are in-memory, so at startup every sandbox-* directory is
        an orphan of a previous process. Only direct child DIRECTORIES whose
        name starts with "sandbox-" and which no tracked session owns are
        removed; files (notably the audit database living in
        workspace_base) are never touched. In persistent mode the reap is
        skipped entirely and survivors are logged as detached (inert until
        a future cycle re-attaches them; write-back stays gated).

        Returns:
            Count of directories removed (0 in persistent mode).
        """
        base = self._config.workspace_base
        if not os.path.isdir(base):
            return 0

        with self._lock:
            tracked = {
                s.workspace_path for s in self._sessions.values()
            }

        orphans: list[str] = []
        try:
            with os.scandir(base) as entries:
                for entry in entries:
                    if not entry.name.startswith("sandbox-"):
                        continue
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    if entry.path in tracked:
                        continue
                    orphans.append(entry.path)
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning("Workspace reconcile scan failed: %s", exc)
            return 0

        if self._config.workspace_persistent:
            if orphans:
                logger.info(
                    "Persistent workspace_base: %d detached workspace "
                    "directorie(s) left in place (not reaped): %s",
                    len(orphans),
                    ", ".join(os.path.basename(p) for p in orphans),
                )
            return 0

        removed = 0
        for path in orphans:
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
        if removed:
            logger.info(
                "Workspace reconcile: reaped %d orphaned director(ies) "
                "under %s",
                removed,
                base,
            )
        return removed

    def _sweep_idle_sessions(self) -> int:
        """Destroy idle, unbound workspaces past the idle TTL (lazy sweep).

        Called from create_sandbox (so a stale workspace never blocks the
        concurrency cap) and list_sessions. A workspace is reaped when ALL
        of: the TTL is enabled (> 0), nothing is running in it, it is not
        bound to a conversation (an explicit binding is user intent and
        exempts it), and last_activity is older than the TTL.

        Returns:
            Count of workspaces destroyed.
        """
        ttl = self._config.idle_ttl_seconds
        if ttl <= 0:
            return 0
        now = time.time()
        with self._lock:
            stale = [
                s.session_id
                for s in self._sessions.values()
                if s.active
                and s.bound_conversation_id is None
                and s.session_id not in self._running_procs
                and (now - s.last_activity) > ttl
            ]
        reaped = 0
        for sid in stale:
            if self.destroy_sandbox(sid):
                reaped += 1
                logger.info(
                    "Idle TTL: workspace %s destroyed after %ds idle",
                    sid,
                    ttl,
                )
        return reaped

    @staticmethod
    def _workspace_disk_use(path: str) -> int:
        """Approximate disk use of a workspace tree, bounded.

        A scandir walk capped on entries and depth (S210 constants) so a
        pathological tree cannot stall the manager; symlinks are not
        followed. The figure is approximate by design and recomputed per
        list call, never cached on the session.
        """
        total = 0
        entries_seen = 0
        stack: list[tuple[str, int]] = [(path, 0)]
        while stack:
            current, depth = stack.pop()
            if depth > _DISK_WALK_MAX_DEPTH:
                continue
            try:
                with os.scandir(current) as it:
                    for entry in it:
                        entries_seen += 1
                        if entries_seen > _DISK_WALK_MAX_ENTRIES:
                            return total
                        try:
                            if entry.is_file(follow_symlinks=False):
                                total += entry.stat(
                                    follow_symlinks=False
                                ).st_size
                            elif entry.is_dir(follow_symlinks=False):
                                stack.append((entry.path, depth + 1))
                        except OSError:
                            continue
            except OSError:
                continue
        return total

    def register_created_file(
        self, relative_path: str, content: str,
    ) -> None:
        """Register a file creation for write-then-execute detection (S81).

        Called by file_tools after creating a file in the sandbox.
        The CommandValidator will inspect this file's content before
        allowing bash commands that reference it.

        Args:
            relative_path: File path relative to workspace root.
            content: The content that was written.
        """
        self._validator.register_created_file(relative_path, content)

    def inject_files(
        self,
        session_id: str,
        file_paths: list[str],
    ) -> list[str]:
        """Copy files from the host into the sandbox workspace.

        Only existing, readable files are copied. Files are placed
        in the workspace root, preserving only the filename (no
        directory structure from the source).

        Args:
            session_id: Target sandbox session.
            file_paths: List of host file paths to inject.

        Returns:
            List of paths (inside workspace) of successfully copied files.

        Raises:
            ValueError: If session not found or inactive.
        """
        workspace = self._get_active_workspace(session_id)
        injected: list[str] = []

        # S210: per-workspace disk soft quota. Compute the incoming size up
        # front and refuse the whole copy-in when it would push the
        # workspace past the soft limit; the workspace is never destroyed.
        incoming = 0
        for src_path in file_paths:
            src_real = os.path.realpath(src_path)
            if os.path.isfile(src_real):
                try:
                    incoming += os.path.getsize(src_real)
                except OSError:
                    continue
        self._check_disk_quota(session_id, workspace, incoming)

        for src_path in file_paths:
            src_real = os.path.realpath(src_path)
            if not os.path.isfile(src_real):
                logger.warning("Inject skipped (not a file): %s", src_path)
                continue
            if not os.access(src_real, os.R_OK):
                logger.warning("Inject skipped (not readable): %s", src_path)
                continue

            filename = os.path.basename(src_real)
            # Prevent overwriting with a name that escapes
            valid, resolved, err = validate_sandbox_path(workspace, filename)
            if not valid:
                logger.warning("Inject blocked (path validation): %s", err)
                continue

            shutil.copy2(src_real, resolved)
            injected.append(resolved)
            logger.debug("Injected file: %s -> %s", src_path, resolved)

        return injected

    def inject_directory(
        self,
        session_id: str,
        src_dir: str,
        dest_subdir: str = "",
    ) -> int:
        """Copy a directory tree into the sandbox workspace.

        Args:
            session_id: Target sandbox session.
            src_dir: Host directory to copy.
            dest_subdir: Subdirectory within workspace (default: root).

        Returns:
            Number of files copied.

        Raises:
            ValueError: If session not found, inactive, or path invalid.
        """
        workspace = self._get_active_workspace(session_id)

        if not os.path.isdir(src_dir):
            raise ValueError(f"Source is not a directory: {src_dir}")

        if dest_subdir:
            valid, dest_path, err = validate_sandbox_path(
                workspace, dest_subdir
            )
            if not valid:
                raise ValueError(f"Invalid destination: {err}")
        else:
            dest_path = workspace

        # S210: per-workspace disk soft quota on the incoming tree.
        incoming = self._workspace_disk_use(src_dir)
        self._check_disk_quota(session_id, workspace, incoming)

        # Copy tree
        dest_final = os.path.join(dest_path, os.path.basename(src_dir))
        shutil.copytree(src_dir, dest_final, dirs_exist_ok=True)

        self.touch_activity(session_id)

        # Count files
        count = sum(1 for _, _, files in os.walk(dest_final) for _ in files)
        logger.debug(
            "Injected directory: %s -> %s (%d files)",
            src_dir,
            dest_final,
            count,
        )
        return count

    def _check_disk_quota(
        self, session_id: str, workspace: str, incoming_bytes: int
    ) -> None:
        """Refuse a copy-in that would exceed the workspace soft quota.

        Soft semantics (S210): raises WorkspaceQuotaExceeded; the caller's
        copy-in is refused and the workspace is left untouched. The tmpfs
        --size from Bloc 0 caps /tmp inside the sandbox; this quota covers
        the workspace bind on the host side.
        """
        limit = self._config.disk_soft_limit_bytes
        current = self._workspace_disk_use(workspace)
        if current + incoming_bytes > limit:
            raise WorkspaceQuotaExceeded(
                f"Copy-in of {incoming_bytes} bytes would exceed the "
                f"workspace disk soft limit ({current} bytes in use, "
                f"limit {limit} bytes) for session {session_id}"
            )

    # -----------------------------------------------------------------
    # Copy-in (S211, Bloc 2): drag-and-drop upload, allowlisted host
    # browse, and the symlink-safe host clone. All three are EXPLICIT,
    # user-initiated actions through the manager UI -- the model can
    # trigger none of them (S73/S74); the agent only ever sees /workspace.
    # -----------------------------------------------------------------

    @staticmethod
    def sanitize_upload_filename(name: str) -> tuple[bool, str]:
        """Sanitize a client-supplied upload filename (S211).

        Returns (ok, clean_name_or_error). The client controls this string,
        so it is reduced to a basename and refused outright when empty,
        dot/dot-dot, or carrying path separators or NUL -- the collision
        and containment checks downstream then only ever see a plain name.
        """
        if not isinstance(name, str) or not name.strip():
            return False, "Empty filename"
        candidate = name.strip()
        if "\x00" in candidate:
            return False, "Filename contains NUL"
        if "/" in candidate or "\\" in candidate:
            return False, "Filename contains a path separator"
        base = os.path.basename(candidate)
        if base in ("", ".", ".."):
            return False, "Filename resolves to a reserved name"
        return True, base

    def upload_files(
        self,
        session_id: str,
        items: list[tuple[str, Any, int]],
        dest_subdir: str = "",
    ) -> dict[str, Any]:
        """Write user-uploaded file streams into the workspace (S211, 5.1).

        ``items`` is a list of (filename, binary stream, size_bytes); sizes
        are summed BEFORE any write. Caps: at most ``upload_max_files`` per
        request and ``upload_max_file_bytes`` per file; the request total is
        bounded by the S210 disk soft quota. Any exceeded cap refuses the
        WHOLE request (WorkspaceQuotaExceeded -> the route's 413) with the
        workspace untouched. Individually invalid names and destination
        collisions are refused PER FILE (never overwritten, never renamed)
        and reported honestly; valid files are written in bounded chunks
        with an on-the-fly sha256 so the caller can record the baseline
        manifest (section 6.1) without a second read.

        Returns {"written": [{path, name, size, sha256}], "refused":
        [{name, reason}], "written_bytes": int}.
        """
        workspace = self._get_active_workspace(session_id)
        cfg = self._config

        if len(items) > cfg.upload_max_files:
            raise WorkspaceQuotaExceeded(
                f"Upload of {len(items)} files exceeds the per-request "
                f"file cap ({cfg.upload_max_files}) for session {session_id}"
            )
        for name, _stream, size in items:
            if size > cfg.upload_max_file_bytes:
                raise WorkspaceQuotaExceeded(
                    f"Uploaded file '{name}' ({size} bytes) exceeds the "
                    f"per-file cap ({cfg.upload_max_file_bytes} bytes) "
                    f"for session {session_id}"
                )
        incoming = sum(size for _n, _s, size in items)
        self._check_disk_quota(session_id, workspace, incoming)

        if dest_subdir:
            valid, dest_root, err = validate_sandbox_path(
                workspace, dest_subdir
            )
            if not valid:
                raise ValueError(f"Invalid destination: {err}")
            os.makedirs(dest_root, exist_ok=True)
        else:
            dest_root = workspace

        written: list[dict[str, Any]] = []
        refused: list[dict[str, str]] = []
        written_bytes = 0
        import hashlib

        for name, stream, _size in items:
            ok, clean = self.sanitize_upload_filename(name)
            if not ok:
                refused.append({"name": name, "reason": clean})
                continue
            rel = (
                os.path.join(dest_subdir, clean) if dest_subdir else clean
            )
            valid, resolved, err = validate_sandbox_path(workspace, rel)
            if not valid:
                refused.append({"name": name, "reason": err})
                continue
            if os.path.lexists(resolved):
                refused.append({
                    "name": name,
                    "reason": "Destination already exists (not overwritten)",
                })
                continue
            digest = hashlib.sha256()
            size_written = 0
            with open(resolved, "wb") as out:
                while True:
                    chunk = stream.read(_COPYIN_CHUNK_BYTES)
                    if not chunk:
                        break
                    out.write(chunk)
                    digest.update(chunk)
                    size_written += len(chunk)
            written_bytes += size_written
            written.append({
                "path": resolved,
                "relative_path": rel.replace(os.sep, "/"),
                "name": clean,
                "size": size_written,
                "sha256": digest.hexdigest(),
            })

        self.touch_activity(session_id)
        self._audit.log_approval(
            session_id,
            action="workspace_upload",
            paths=[w["relative_path"] for w in written],
            detail=(
                f"Uploaded {len(written)} file(s), {written_bytes} bytes; "
                f"refused {len(refused)}"
            ),
        )
        logger.info(
            "Upload into workspace: session=%s, written=%d, refused=%d",
            session_id, len(written), len(refused),
        )
        return {
            "written": written,
            "refused": refused,
            "written_bytes": written_bytes,
        }

    def _resolve_share_path(self, path: str) -> str:
        """Resolve a host path and confine it to the share-root allowlist.

        Confinement runs BEFORE any existence check so a request outside the
        allowlisted roots learns nothing about the host tree (PermissionError
        -> the route's 403, whether or not the path exists). Inside a root,
        a missing or non-directory path raises ValueError (-> 404).
        """
        roots = self._config.host_share_roots
        resolved = os.path.realpath(os.path.expanduser(path))
        confined = any(
            resolved == root or resolved.startswith(root + os.sep)
            for root in roots
        )
        if not confined:
            raise PermissionError(
                f"Path is outside the allowlisted share roots: {resolved}"
            )
        if not os.path.isdir(resolved):
            raise ValueError(f"Not an existing directory: {resolved}")
        return resolved

    def browse_host(self, path: str | None = None) -> dict[str, Any]:
        """List a host directory's IMMEDIATE entries, allowlist-confined
        (S211, 5.2a).

        With no path, the allowlisted roots themselves are returned as the
        entry set (the explorer's entry points). Symlinks are displayed
        WITHOUT following (type "symlink", no target disclosure, size 0);
        hidden entries are listed with a flag rather than omitted -- hiding
        them would lie about what a clone of this directory copies. Sizes
        are the entry's own stat, never a tree walk.
        """
        roots = self._config.host_share_roots
        if path is None or not str(path).strip():
            entries = [
                {"name": r, "type": "dir", "size": 0, "hidden": False}
                for r in roots
            ]
            self._audit.log_approval(
                "host-browse",
                action="host_browse",
                paths=[],
                detail="Listed share roots",
            )
            return {"path": "", "roots": list(roots), "entries": entries}

        resolved = self._resolve_share_path(str(path))
        entries = []
        with os.scandir(resolved) as it:
            for entry in it:
                try:
                    if entry.is_symlink():
                        etype, size = "symlink", 0
                    elif entry.is_dir(follow_symlinks=False):
                        etype, size = "dir", 0
                    elif entry.is_file(follow_symlinks=False):
                        etype = "file"
                        size = entry.stat(follow_symlinks=False).st_size
                    else:
                        etype, size = "special", 0
                except OSError:
                    continue
                entries.append({
                    "name": entry.name,
                    "type": etype,
                    "size": size,
                    "hidden": entry.name.startswith("."),
                })
        entries.sort(key=lambda e: (e["type"] != "dir", e["name"].lower()))
        self._audit.log_approval(
            "host-browse",
            action="host_browse",
            paths=[],
            detail=f"Browsed {resolved} ({len(entries)} entries)",
        )
        return {"path": resolved, "roots": list(roots), "entries": entries}

    def _prewalk_clone_source(
        self, src: str, remaining_quota: int
    ) -> tuple[int, int]:
        """Exact bounded pre-walk of a clone source (S211, 5.2b).

        Counts the bytes and regular files a symlink-safe clone would copy
        (symlinks and special files are NOT counted: the clone skips them).
        Unlike the approximate disk-use walk, exceeding ANY bound here --
        the clone byte/file caps, the remaining workspace quota, or the
        walk depth -- REFUSES the clone (WorkspaceQuotaExceeded) instead of
        returning a partial figure: a cap enforced on an undercount is no
        cap. Runs BEFORE any copy; the workspace is untouched on refusal.
        """
        cfg = self._config
        max_bytes = min(cfg.clone_max_bytes, max(remaining_quota, 0))
        total_bytes = 0
        total_files = 0
        stack: list[tuple[str, int]] = [(src, 0)]
        while stack:
            current, depth = stack.pop()
            if depth > _CLONE_WALK_MAX_DEPTH:
                raise WorkspaceQuotaExceeded(
                    f"Clone source exceeds the maximum directory depth "
                    f"({_CLONE_WALK_MAX_DEPTH}): {current}"
                )
            with os.scandir(current) as it:
                for entry in it:
                    if entry.is_symlink():
                        continue
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append((entry.path, depth + 1))
                        elif entry.is_file(follow_symlinks=False):
                            total_files += 1
                            total_bytes += entry.stat(
                                follow_symlinks=False
                            ).st_size
                    except OSError:
                        continue
                    if total_files > cfg.clone_max_files:
                        raise WorkspaceQuotaExceeded(
                            f"Clone exceeds the file-count cap "
                            f"({cfg.clone_max_files} files)"
                        )
                    if total_bytes > max_bytes:
                        raise WorkspaceQuotaExceeded(
                            f"Clone exceeds the byte cap "
                            f"({total_bytes} > {max_bytes} bytes: the "
                            f"per-clone cap is {cfg.clone_max_bytes}, the "
                            f"remaining workspace quota {remaining_quota})"
                        )
        return total_bytes, total_files

    def clone_directory(
        self,
        session_id: str,
        src_path: str,
        dest_subdir: str = "",
    ) -> dict[str, Any]:
        """Clone an allowlisted host directory into the workspace (S211,
        5.2b), symlink-safe, with the section 6.1 baseline hashes computed
        on the fly.

        The source must resolve under an allowlisted share root (the same
        confinement as the browse). The exact pre-walk enforces the clone
        caps AND the remaining S210 quota before any copy. The copy itself
        never follows a symlink (every symlink is skipped and counted; its
        target is never read, copied, or exposed) and skips device/special
        files. Regular files are copied in bounded chunks with an
        on-the-fly sha256 per file -- the returned ``manifest`` maps each
        relative path (under the clone destination) to its content hash,
        the seam Bloc 3's diff consumes. The destination
        ``<workspace>/[dest_subdir/]<basename(src)>`` must not already
        exist (FileExistsError -> the route's 409): explicit, never merged.

        ``inject_directory`` (S116 semantics, copytree) is deliberately
        left byte-identical; this path owns the symlink-safe discipline.
        """
        workspace = self._get_active_workspace(session_id)
        src = self._resolve_share_path(src_path)

        if dest_subdir:
            valid, dest_parent, err = validate_sandbox_path(
                workspace, dest_subdir
            )
            if not valid:
                raise ValueError(f"Invalid destination: {err}")
        else:
            dest_parent = workspace
        dest_final = os.path.join(dest_parent, os.path.basename(src))
        if os.path.lexists(dest_final):
            raise FileExistsError(
                f"Clone destination already exists: {dest_final}"
            )

        current_use = self._workspace_disk_use(workspace)
        remaining = self._config.disk_soft_limit_bytes - current_use
        total_bytes, total_files = self._prewalk_clone_source(src, remaining)

        import hashlib

        copied_files = 0
        copied_bytes = 0
        skipped_symlinks = 0
        skipped_special = 0
        manifest: dict[str, str] = {}
        dest_rel_root = os.path.relpath(dest_final, workspace)

        os.makedirs(dest_parent, exist_ok=True)
        stack: list[tuple[str, str]] = [(src, dest_final)]
        while stack:
            cur_src, cur_dst = stack.pop()
            os.makedirs(cur_dst, exist_ok=True)
            with os.scandir(cur_src) as it:
                for entry in it:
                    target = os.path.join(cur_dst, entry.name)
                    try:
                        if entry.is_symlink():
                            skipped_symlinks += 1
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append((entry.path, target))
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            skipped_special += 1
                            continue
                        digest = hashlib.sha256()
                        with open(entry.path, "rb") as fin, \
                                open(target, "wb") as fout:
                            while True:
                                chunk = fin.read(_COPYIN_CHUNK_BYTES)
                                if not chunk:
                                    break
                                fout.write(chunk)
                                digest.update(chunk)
                                copied_bytes += len(chunk)
                        shutil.copystat(entry.path, target)
                        copied_files += 1
                        rel = os.path.relpath(target, workspace)
                        manifest[rel.replace(os.sep, "/")] = (
                            digest.hexdigest()
                        )
                    except OSError as exc:
                        logger.warning(
                            "Clone skipped (read/write error): %s (%s)",
                            entry.path, exc,
                        )
                        continue

        self.touch_activity(session_id)
        self._audit.log_approval(
            session_id,
            action="host_clone",
            paths=[dest_rel_root.replace(os.sep, "/")],
            detail=(
                f"Cloned {src} -> {dest_rel_root}: {copied_files} file(s), "
                f"{copied_bytes} bytes; skipped {skipped_symlinks} "
                f"symlink(s), {skipped_special} special file(s); "
                f"prewalk {total_files} file(s), {total_bytes} bytes"
            ),
        )
        logger.info(
            "Cloned host directory: %s -> %s (%d files, %d bytes)",
            src, dest_final, copied_files, copied_bytes,
        )
        return {
            "dest": dest_rel_root.replace(os.sep, "/"),
            "cloned_root": src,
            "copied_files": copied_files,
            "copied_bytes": copied_bytes,
            "skipped_symlinks": skipped_symlinks,
            "skipped_special": skipped_special,
            "manifest": manifest,
        }

    def extract_files(self, session_id: str) -> list[dict[str, Any]]:
        """List all files available for extraction from the sandbox.

        Returns a list of dicts with 'path' (relative to workspace),
        'size', and 'modified' keys.

        Raises:
            ValueError: If session not found or inactive.
        """
        workspace = self._get_active_workspace(session_id)
        files: list[dict[str, Any]] = []

        for root, _dirs, filenames in os.walk(workspace):
            for fname in sorted(filenames):
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, workspace)
                try:
                    stat = os.stat(full)
                    files.append({
                        "path": rel,
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                    })
                except OSError:
                    continue

        return files

    # -- S116: File preview, approval, and copy-out --

    def preview_file(
        self,
        session_id: str,
        path: str,
        max_bytes: int = 65536,
    ) -> dict[str, Any]:
        """Preview a file's content from the sandbox (S116).

        Returns text content capped at max_bytes for display.
        Binary files return a truncated hex preview.

        Args:
            session_id: Sandbox session.
            path: Relative path within the workspace.
            max_bytes: Maximum bytes to read (default 64KB).

        Returns:
            Dict with 'path', 'content', 'size', 'truncated', 'is_binary'.

        Raises:
            ValueError: If session not found, inactive, or path invalid.
        """
        workspace = self._get_active_workspace(session_id)
        valid, resolved, err = validate_sandbox_path(workspace, path)
        if not valid:
            raise ValueError(f"Invalid path: {err}")
        if not os.path.isfile(resolved):
            raise ValueError(f"File not found: {path}")

        file_size = os.path.getsize(resolved)
        truncated = file_size > max_bytes
        is_binary = False

        try:
            with open(resolved, encoding="utf-8") as fh:
                content = fh.read(max_bytes)
        except UnicodeDecodeError:
            is_binary = True
            with open(resolved, "rb") as fh:
                raw = fh.read(min(max_bytes, 2048))
            content = raw.hex()
            truncated = file_size > 2048

        return {
            "path": path,
            "content": content,
            "size": file_size,
            "truncated": truncated,
            "is_binary": is_binary,
        }

    def approve_files(
        self,
        session_id: str,
        paths: list[str],
    ) -> list[str]:
        """Approve specific files for copy-out (S116).

        Each path is validated against the workspace. Only valid files
        within the workspace are approved. Approval is additive.

        Args:
            session_id: Sandbox session.
            paths: List of relative paths to approve.

        Returns:
            List of paths actually approved.

        Raises:
            ValueError: If session not found or inactive.
        """
        workspace = self._get_active_workspace(session_id)
        approved: list[str] = []

        for p in paths:
            valid, resolved, err = validate_sandbox_path(workspace, p)
            if not valid:
                logger.warning("Approval denied (path validation): %s", err)
                continue
            if not os.path.isfile(resolved):
                logger.warning("Approval denied (not a file): %s", p)
                continue
            approved.append(p)

        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ValueError(f"Session not found: {session_id}")
            session.approved_paths.update(approved)
            if approved:
                session.approval_state = ApprovalState.APPROVED
                session.approved_at = time.time()

        # Audit
        self._audit.log_approval(
            session_id,
            action="approve",
            paths=approved,
            detail=f"Approved {len(approved)} file(s)",
        )

        logger.info(
            "Files approved for copy-out: session=%s, count=%d",
            session_id, len(approved),
        )
        return approved

    def reject_files(self, session_id: str) -> None:
        """Reject all files, preventing any copy-out (S116).

        Args:
            session_id: Sandbox session.

        Raises:
            ValueError: If session not found or inactive.
        """
        self._get_active_workspace(session_id)  # validate session

        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ValueError(f"Session not found: {session_id}")
            session.approval_state = ApprovalState.REJECTED
            session.approved_paths.clear()
            # S212: a full reject also withdraws every confirmed deletion --
            # reject means "apply nothing", writes and deletions alike.
            session.confirmed_deletions.clear()
            session.approved_at = time.time()

        self._audit.log_approval(
            session_id,
            action="reject",
            detail="All files rejected for copy-out",
        )
        logger.info("Files rejected: session=%s", session_id)

    def is_file_approved(self, session_id: str, path: str) -> bool:
        """Check if a specific file is approved for copy-out (S116)."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            return path in session.approved_paths

    # -- S212 (Bloc 3): diff-gated write-back support --

    def get_active_workspace_path(self, session_id: str) -> str:
        """Public wrapper for the active-workspace lookup (S212).

        The workspace diff and the apply writer live in
        ``sandbox_workspace.py`` (the section 12 cartography); this thin
        wrapper lets them resolve a session's workspace without reaching
        into a private member across modules.

        Raises:
            ValueError: If the session is unknown, inactive, or its
                workspace directory is missing.
        """
        return self._get_active_workspace(session_id)

    def resolve_share_target(self, path: str) -> str:
        """Public wrapper over the share-root confinement (S212).

        The apply writer validates its target through the exact S211
        ``_resolve_share_path`` discipline: confinement BEFORE existence
        (PermissionError -> 403 whether or not the path exists; inside a
        root, missing/non-directory -> ValueError -> 404).
        """
        return self._resolve_share_path(path)

    # -- S213 (Bloc 4): the per-workspace network flag and the provision run

    @staticmethod
    def _network_gate_allows() -> bool:
        """Consult the binding-layer egress gate, fail-secure.

        Resolves ``opti_oignon.sandbox_egress`` lazily (sys.modules first,
        then a real import) and asks ``network_allowed()`` -- True only when
        the live mode is exactly "daily". ANY failure -- the module absent
        in a partial build, the mode unreadable, an unexpected error --
        answers False: an undeterminable gate refuses, it never permits.
        """
        try:
            import importlib
            import sys as _sys

            egress = _sys.modules.get("opti_oignon.sandbox_egress")
            if egress is None:
                egress = importlib.import_module("opti_oignon.sandbox_egress")
            return bool(egress.network_allowed())
        except Exception:
            logger.warning(
                "Sandbox egress gate unavailable; refusing network "
                "(fail-secure)."
            )
            return False

    def set_network_enabled(
        self, session_id: str, enabled: bool, actor: str = ""
    ) -> bool:
        """Flip the per-workspace network flag -- an explicit USER action.

        S213 (Bloc 4, spec 8.3). Enabling is Daily-only: the binding-layer
        gate is consulted live and an unset, unknown, or undeterminable
        mode refuses (fail-secure); the refusal is audited and raised as
        PermissionError for the route's 403. DISABLING is permitted in any
        mode -- moving toward less capability is never refused, and keeping
        a stale on-flag would be the dishonest outcome. Both directions
        (who and when) are recorded in the per-session audit AND the
        hash-chain log. There is no configuration default and no tool
        surface: the only callers are the manager route and tests.

        Args:
            session_id: Sandbox session.
            enabled: Desired flag state.
            actor: The acting user id, for the audit rows.

        Returns:
            The new flag state.

        Raises:
            ValueError: If the session is unknown or inactive.
            PermissionError: If enabling is requested outside Daily mode.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or not session.active:
                raise ValueError(f"Unknown or inactive session: {session_id}")

        if enabled and not self._network_gate_allows():
            self._audit.log_approval(
                session_id,
                action="network_refused",
                detail=(
                    "enable refused: Daily-only capability and the live "
                    f"mode is not daily (actor={actor or 'unknown'})"
                ),
            )
            raise PermissionError(
                "Sandbox network egress is disabled in Bulbe mode: the "
                "workspace network is a Daily-only, user-activated "
                "capability. Switch to Daily mode to enable it."
            )

        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or not session.active:
                raise ValueError(f"Unknown or inactive session: {session_id}")
            session.network_enabled = bool(enabled)
            session.last_activity = time.time()

        action = "network_on" if enabled else "network_off"
        self._audit.log_approval(
            session_id,
            action=action,
            detail=f"actor={actor or 'unknown'}",
        )
        try:
            from opti_oignon.signed_audit_log import chain_log

            chain_log(
                event_type="sandbox_network_toggle",
                source="sandbox_manager",
                action=action,
                severity="WARNING" if enabled else "INFO",
                session_id=session_id,
                enabled=bool(enabled),
                actor=actor or "unknown",
            )
        except Exception:  # pragma: no cover - the chain is best-effort here
            logger.debug("Hash-chain audit unavailable for %s", action)
        logger.info(
            "Workspace %s network %s by %s",
            session_id,
            "ENABLED" if enabled else "disabled",
            actor or "unknown",
        )
        return bool(enabled)

    def execute_provision_command(
        self,
        session_id: str,
        command: str,
        timeout: int | None = None,
    ) -> CommandResult:
        """Run the ONE network-on step: the provision install (S213, 8.4).

        The single seam that ever reaches ``_run_bwrap`` with
        ``allow_network=True``. Refusals are fail-secure and audited with
        the ``_refused`` discipline:

        - the binding-layer gate is re-asserted live (defense in depth on
          top of the route's 403): not Daily -> blocked, even when the
          per-workspace flag is somehow on;
        - the per-workspace flag must be on: the user's explicit grant is a
          precondition, not implied by the call;
        - bwrap-only: the tempdir backend has no scoped network (its
          isolation IS ``unshare --net``), so a provision there would be a
          raw host-network run -- refused. In a container without bwrap
          this is the honest posture: the gate, the shapes, and the audit
          are proven; the live run is host territory;
        - the command passes the standard validator (defense in depth; the
          line is server-built from validated relative paths, never model
          text).

        After the run the network is off again BY CONSTRUCTION: every task
        run keeps the unconditional ``--unshare-net`` argv; nothing latches.

        Returns:
            CommandResult; refusals come back blocked with an honest
            reason, mirroring the execute path's posture.

        Raises:
            ValueError: If the session is unknown or inactive.
        """
        workspace = self._get_active_workspace(session_id)
        with self._lock:
            session = self._sessions.get(session_id)
            network_on = bool(session and session.network_enabled)

        def _refused(reason: str) -> CommandResult:
            self._audit.log_approval(
                session_id, action="provision_refused", detail=reason
            )
            result = CommandResult(
                blocked=True,
                block_reason=reason,
                isolation_backend="blocked",
            )
            self._audit.log_command(session_id, command, result)
            return result

        if not self._network_gate_allows():
            return _refused(
                "Provision refused: sandbox network egress is Daily-only "
                "and the live mode is not daily (fail-secure; an unset or "
                "unknown mode is treated as Bulbe)."
            )
        if not network_on:
            return _refused(
                "Provision refused: network is not enabled for this "
                "workspace. Enable it explicitly first (the flag is "
                "per-workspace, default off)."
            )
        if (
            self._isolation_backend != IsolationBackend.BWRAP
            or not self._bwrap_available
        ):
            return _refused(
                "Provision refused: the provision run requires the bwrap "
                "backend (the tempdir fallback has no scoped network). "
                "Install bubblewrap: apt install bubblewrap"
            )

        is_safe, reason = self._validator.validate(command)
        if not is_safe:
            return _refused(f"Provision command failed validation: {reason}")

        effective_timeout = timeout or self._config.provision_timeout_seconds
        result = self._run_bwrap(
            command,
            workspace,
            effective_timeout,
            session_id=session_id,
            allow_network=True,
        )
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.command_count += 1
                session.last_activity = time.time()
        self._audit.log_command(session_id, command, result)
        try:
            from opti_oignon.signed_audit_log import chain_log

            chain_log(
                event_type="sandbox_provision_run",
                source="sandbox_manager",
                action="provision_run",
                severity="WARNING",
                session_id=session_id,
                return_code=result.return_code,
                timed_out=result.timed_out,
                blocked=result.blocked,
            )
        except Exception:  # pragma: no cover - the chain is best-effort here
            logger.debug("Hash-chain audit unavailable for provision_run")
        return result

    def confirm_deletions(self, session_id: str, paths: list[str]) -> list[str]:
        """Record explicit deletion confirmations for apply-to-host (S212).

        Deliberately PARALLEL to ``approve_files`` (spec 6.2): a "deleted"
        change removes a HOST file, so it carries its own confirmation,
        distinct from added/modified, and is never bundled into a blanket
        approve-all. This method only records and audits; the load-bearing
        enforcement sits in the apply writer, which deletes only paths that
        are BOTH confirmed here AND classified "deleted" by the recomputed
        diff. Confirmation is additive, like approval.

        Args:
            session_id: Sandbox session.
            paths: Relative paths (as classified by the diff) to confirm.

        Returns:
            The list of recorded paths.

        Raises:
            ValueError: If the session is unknown or inactive.
        """
        self._get_active_workspace(session_id)  # validate session
        recorded = [p for p in paths if p]
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ValueError(f"Session not found: {session_id}")
            session.confirmed_deletions.update(recorded)

        self._audit.log_approval(
            session_id,
            action="deletion_confirm",
            paths=recorded,
            detail=f"Confirmed {len(recorded)} deletion(s) for apply",
        )
        logger.info(
            "Deletions confirmed for apply: session=%s, count=%d",
            session_id, len(recorded),
        )
        return recorded

    def get_confirmed_deletions(self, session_id: str) -> set[str]:
        """The deletion paths confirmed for apply (a copy; S212)."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return set()
            return set(session.confirmed_deletions)

    def copy_out_file(
        self,
        session_id: str,
        path: str,
        dest_dir: str,
    ) -> dict[str, Any]:
        """Copy a single approved file from the sandbox to the host (S116).

        The file MUST have been approved via approve_files() first.
        No auto-approve. No bypass.

        Args:
            session_id: Sandbox session.
            path: Relative path within the workspace.
            dest_dir: Host directory to copy the file to.

        Returns:
            Dict with 'src_path', 'dest_path', 'size'.

        Raises:
            ValueError: If session not found, inactive, or path invalid.
            PermissionError: If file not approved.
        """
        workspace = self._get_active_workspace(session_id)

        # Check approval
        if not self.is_file_approved(session_id, path):
            self._audit.log_approval(
                session_id,
                action="copy_out_denied",
                paths=[path],
                detail="File not approved for copy-out",
            )
            raise PermissionError(
                f"File not approved for copy-out: {path}. "
                f"Approve it first via approve_files()."
            )

        # Validate source path
        valid, resolved_src, err = validate_sandbox_path(workspace, path)
        if not valid:
            raise ValueError(f"Invalid source path: {err}")
        if not os.path.isfile(resolved_src):
            raise ValueError(f"Source file not found: {path}")

        # Validate and create destination
        dest_dir_real = os.path.realpath(dest_dir)
        os.makedirs(dest_dir_real, mode=0o755, exist_ok=True)

        dest_file = os.path.join(dest_dir_real, os.path.basename(path))
        shutil.copy2(resolved_src, dest_file)
        file_size = os.path.getsize(dest_file)

        self._audit.log_approval(
            session_id,
            action="copy_out",
            paths=[path],
            dest_dir=dest_dir_real,
            detail=f"Copied {file_size} bytes",
        )

        logger.info(
            "File copied out: session=%s, %s -> %s (%d bytes)",
            session_id, path, dest_file, file_size,
        )
        return {
            "src_path": path,
            "dest_path": dest_file,
            "size": file_size,
        }

    def copy_out_batch(
        self,
        session_id: str,
        paths: list[str],
        dest_dir: str,
    ) -> list[dict[str, Any]]:
        """Copy multiple approved files from the sandbox to the host (S116).

        Only approved files are copied. Non-approved files are skipped
        with a warning (not an error, to allow partial copy-out).

        Args:
            session_id: Sandbox session.
            paths: List of relative paths to copy.
            dest_dir: Host directory to copy files to.

        Returns:
            List of dicts with 'src_path', 'dest_path', 'size' per copied file.

        Raises:
            ValueError: If session not found or inactive.
        """
        results: list[dict[str, Any]] = []
        skipped: list[str] = []

        for p in paths:
            if not self.is_file_approved(session_id, p):
                skipped.append(p)
                logger.warning(
                    "Batch copy-out skipped (not approved): %s", p,
                )
                continue
            try:
                result = self.copy_out_file(session_id, p, dest_dir)
                results.append(result)
            except (ValueError, OSError) as exc:
                logger.warning(
                    "Batch copy-out failed for '%s': %s", p, exc,
                )
                skipped.append(p)

        if skipped:
            self._audit.log_approval(
                session_id,
                action="copy_out_batch_partial",
                paths=skipped,
                detail=f"Skipped {len(skipped)} file(s) (not approved or error)",
            )

        return results

    def get_approval_info(self, session_id: str) -> dict[str, Any]:
        """Get approval state summary for a session (S116).

        Returns:
            Dict with approval_state, approved_paths, approved_at.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return {
                    "approval_state": "unknown",
                    "approved_paths": [],
                    "approved_at": None,
                }
            return {
                "approval_state": session.approval_state.value,
                "approved_paths": sorted(session.approved_paths),
                "approved_at": session.approved_at,
            }

    def execute_command(
        self,
        session_id: str,
        command: str,
        timeout: int | None = None,
    ) -> CommandResult:
        """Execute a command inside the sandbox with security enforcement.

        On bwrap backend: command runs in a fully isolated Linux namespace
        with no network, no host filesystem visibility, no host PIDs.

        On tempdir backend: command runs with a restricted environment and
        command blocklist, but without true kernel isolation.

        The command blocklist is applied on BOTH backends (defense-in-depth).

        Args:
            session_id: Sandbox session to execute in.
            command: Shell command to run.
            timeout: Override timeout (seconds). Uses config default if None.

        Returns:
            CommandResult with stdout, stderr, return code, and status flags.

        Raises:
            ValueError: If session not found or inactive.
        """
        workspace = self._get_active_workspace(session_id)
        # S210: timeout resolution order is explicit call > per-sandbox
        # override > config default.
        if timeout is not None:
            effective_timeout = timeout
        else:
            with self._lock:
                _session = self._sessions.get(session_id)
                _override = (
                    _session.timeout_override if _session is not None else None
                )
            effective_timeout = _override or self._config.command_timeout
        backend = self._isolation_backend

        # S124: Strict mode — refuse execution if bwrap is not available
        if (
            self._config.strict_mode
            and not self._bwrap_available
            and backend != IsolationBackend.BWRAP
        ):
            result = CommandResult(
                blocked=True,
                block_reason=(
                    "Sandbox strict_mode is ON but bubblewrap (bwrap) is "
                    "not available. Code execution is BLOCKED for security. "
                    "Install bubblewrap: apt install bubblewrap"
                ),
                isolation_backend="blocked",
            )
            self._audit.log_command(session_id, command, result)
            logger.warning(
                "Execution blocked in session %s: strict_mode + no bwrap",
                session_id,
            )
            return result

        # Validate command (defense-in-depth on ALL backends)
        is_safe, reason = self._validator.validate(command)
        if not is_safe:
            result = CommandResult(
                blocked=True,
                block_reason=reason,
                isolation_backend=backend.value,
            )
            self._audit.log_command(session_id, command, result)
            logger.warning(
                "Command blocked in session %s: %s (reason: %s)",
                session_id,
                command[:100],
                reason,
            )
            return result

        # Execute with the appropriate backend
        if backend == IsolationBackend.BWRAP:
            result = self._run_bwrap(
                command, workspace, effective_timeout, session_id=session_id
            )
        else:
            result = self._run_tempdir(
                command, workspace, effective_timeout, session_id=session_id
            )

        result.isolation_backend = backend.value

        # Update session stats
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.command_count += 1
                session.last_activity = time.time()

        # Audit log
        self._audit.log_command(session_id, command, result)
        return result

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sandbox sessions with their status.

        S210 (Bloc 1): runs the lazy idle-TTL sweep first, then returns the
        manager view: the stored lifecycle fields plus the derived figures
        (age, running/idle from the process registry, approximate disk use
        from the bounded walk). network_enabled is surfaced and stays False
        this cycle (Bloc 4 flips it).
        """
        self._sweep_idle_sessions()
        now = time.time()
        with self._lock:
            sessions = list(self._sessions.values())
            running_ids = set(self._running_procs.keys())
        return [
            {
                "session_id": s.session_id,
                "workspace_path": s.workspace_path,
                "isolation_backend": s.isolation_backend.value,
                "created_at": s.created_at,
                "active": s.active,
                "command_count": s.command_count,
                "approval_state": s.approval_state.value,
                "approved_paths": sorted(s.approved_paths),
                "approved_at": s.approved_at,
                "label": s.label,
                "owner_user_id": s.owner_user_id,
                "bound_conversation_id": s.bound_conversation_id,
                "network_enabled": s.network_enabled,
                "last_activity": s.last_activity,
                "timeout_override": s.timeout_override,
                "age_seconds": max(0.0, now - s.created_at),
                "running": s.session_id in running_ids,
                "disk_use_bytes": self._workspace_disk_use(
                    s.workspace_path
                ) if os.path.isdir(s.workspace_path) else 0,
            }
            for s in sessions
        ]

    def cleanup_all(self) -> int:
        """Destroy all active sandbox sessions. Returns count destroyed."""
        with self._lock:
            session_ids = list(self._sessions.keys())

        count = 0
        for sid in session_ids:
            if self.destroy_sandbox(sid):
                count += 1
        return count

    # -- Private helpers --

    def _get_active_workspace(self, session_id: str) -> str:
        """Get workspace path for an active session or raise."""
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        if not session.active:
            raise ValueError(f"Session is inactive: {session_id}")
        if not os.path.isdir(session.workspace_path):
            raise ValueError(
                f"Workspace directory missing: {session.workspace_path}"
            )
        return session.workspace_path

    def _spawn_tracked(
        self,
        argv: list[str],
        *,
        timeout: int,
        session_id: str = "",
        preexec_fn=None,
        pass_fds: tuple = (),
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        """Spawn a command in its own process group, tracked for the stop path.

        S210 (Bloc 1): replaces the blocking ``subprocess.run`` so the stop
        path has a handle on the running child. The child is launched with
        ``start_new_session=True`` (its pid is the process-group id), the
        ``Popen`` is registered in the per-session running-process registry
        for the duration of the command, and ``communicate(timeout)``
        preserves the blocking semantics and the output capture. On timeout
        the WHOLE group is SIGKILLed and reaped (stronger than ``run()``,
        which killed only the direct child), then ``TimeoutExpired``
        propagates to the caller's existing handler. The registry entry is
        always removed in the ``finally``.
        """
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            preexec_fn=preexec_fn,
            pass_fds=pass_fds,
            start_new_session=True,
            cwd=cwd,
            env=env,
        )
        if session_id:
            with self._lock:
                self._running_procs[session_id] = proc
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return subprocess.CompletedProcess(
                argv, proc.returncode, stdout, stderr
            )
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                proc.communicate(timeout=5)
            except Exception:  # pragma: no cover - defensive reap
                pass
            raise
        finally:
            if session_id:
                with self._lock:
                    self._running_procs.pop(session_id, None)

    def _run_bwrap(
        self,
        command: str,
        workspace: str,
        timeout: int,
        *,
        session_id: str = "",
        allow_network: bool = False,
    ) -> CommandResult:
        """Run a command inside a bubblewrap sandbox.

        S209: a seccomp denylist filter is built and passed on every launch
        (fail-secure -- a build or fd failure refuses the launch when
        seccomp_required is True, rather than running unfiltered), and
        per-sandbox resource caps are applied via the configured backend
        (rlimit preexec by default; a transient systemd --user cgroup scope
        when resource_backend is "cgroup" and systemd-run is available, with a
        rlimit fallback that never disables the caps).

        S210: the spawn goes through ``_spawn_tracked`` (Popen in its own
        process group, registered per session) so the stop path can SIGKILL
        the running command; the timeout and output-truncation behaviour are
        preserved. ``session_id`` is keyword-only and optional so existing
        positional callers are unchanged.

        S213: ``allow_network`` is keyword-only, defaults to False (the argv
        stays byte-identical for every existing caller), and is forwarded to
        ``_build_bwrap_command`` only by ``execute_provision_command`` -- the
        single, gated provision seam. Seccomp and the resource caps apply to
        the network-on run unchanged.
        """
        config = self._config
        seccomp_file = None
        seccomp_fd: int | None = None

        # Build and stage the seccomp filter onto an inheritable fd.
        if config.seccomp_enabled:
            try:
                from opti_oignon import sandbox_seccomp

                blob = sandbox_seccomp.build_filter_program()
                seccomp_file = tempfile.TemporaryFile()
                seccomp_file.write(blob)
                seccomp_file.flush()
                seccomp_file.seek(0)
                seccomp_fd = seccomp_file.fileno()
            except Exception as exc:
                if seccomp_file is not None:
                    seccomp_file.close()
                    seccomp_file = None
                if config.seccomp_required:
                    return CommandResult(
                        stderr=(
                            "Sandbox seccomp filter could not be built and "
                            f"seccomp_required is True: {exc}. Refusing to "
                            "launch unfiltered."
                        ),
                        block_reason="seccomp filter unavailable (fail-secure)",
                        blocked=True,
                        return_code=-1,
                        isolation_backend="bwrap",
                    )
                logger.warning(
                    "SECURITY: seccomp filter could not be built (%s); "
                    "launching the sandbox WITHOUT a seccomp filter because "
                    "seccomp_required is False. Kernel syscall surface is not "
                    "reduced for this launch.",
                    exc,
                )

        try:
            argv = _build_bwrap_command(
                command, workspace, config, seccomp_fd=seccomp_fd,
                allow_network=allow_network,
            )

            preexec = None
            if config.limits_enabled:
                if (
                    config.resource_backend == "cgroup"
                    and _detect_systemd_run()
                ):
                    argv = _systemd_run_prefix(config) + argv
                else:
                    preexec = _make_rlimit_preexec(config)

            pass_fds = (seccomp_fd,) if seccomp_fd is not None else ()
            proc = self._spawn_tracked(
                argv,
                timeout=timeout,
                session_id=session_id,
                preexec_fn=preexec,
                pass_fds=pass_fds,
            )
            return self._process_output(proc, timeout)

        except subprocess.TimeoutExpired:
            return CommandResult(
                stderr=f"Command timed out after {timeout}s (bwrap)",
                timed_out=True,
                return_code=-1,
            )
        except Exception as exc:
            return CommandResult(
                stderr=f"bwrap execution error: {exc}",
                return_code=-1,
            )
        finally:
            if seccomp_file is not None:
                seccomp_file.close()

    def _run_tempdir(
        self,
        command: str,
        workspace: str,
        timeout: int,
        *,
        session_id: str = "",
    ) -> CommandResult:
        """Run a command in tempdir mode (degraded, limited isolation).

        When available, wraps the command in `unshare --user --net`
        to block network access even without bwrap. This prevents
        the LLM from creating a script with network code and running
        it (the main attack vector in tempdir mode).

        S210: spawns through ``_spawn_tracked`` so the stop path works on
        this backend too (same registry, same group kill).
        """
        # Build restricted environment: minimal PATH, no dangerous vars
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": workspace,
            "TMPDIR": workspace,
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }

        # Try to use unshare for network isolation (works without root)
        cmd_parts = self._build_tempdir_command(command)

        try:
            proc = self._spawn_tracked(
                cmd_parts,
                timeout=timeout,
                session_id=session_id,
                cwd=workspace,
                env=env,
            )
            return self._process_output(proc, timeout)

        except subprocess.TimeoutExpired:
            return CommandResult(
                stderr=f"Command timed out after {timeout}s",
                timed_out=True,
                return_code=-1,
            )
        except Exception as exc:
            return CommandResult(
                stderr=f"Execution error: {exc}",
                return_code=-1,
            )

    def _build_tempdir_command(self, command: str) -> list[str]:
        """Build command list for tempdir execution.

        Wraps in `unshare --user --net` when available to provide
        network isolation even without bubblewrap. Falls back to
        plain bash if unshare is not available.

        Returns:
            Command list for subprocess.run().
        """
        if self._unshare_available is None:
            self._unshare_available = self._detect_unshare()

        if self._unshare_available:
            return ["unshare", "--user", "--net", "--",
                    "bash", "-c", command]

        return ["bash", "-c", command]

    @staticmethod
    def _detect_unshare() -> bool:
        """Detect if unshare --user --net works on this system."""
        try:
            result = subprocess.run(
                ["unshare", "--user", "--net", "--",
                 "echo", "unshare-ok"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and "unshare-ok" in result.stdout:
                logger.info(
                    "Tempdir mode: unshare --user --net available "
                    "(network isolation enabled)"
                )
                return True
        except Exception:
            pass

        logger.warning(
            "Tempdir mode: unshare not available, "
            "NO network isolation in degraded mode"
        )
        return False

    def _process_output(
        self,
        proc: subprocess.CompletedProcess,
        timeout: int,
    ) -> CommandResult:
        """Process subprocess output with size caps."""
        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")

        truncated_stdout = False
        truncated_stderr = False

        if len(stdout.encode("utf-8")) > self._config.max_output_bytes:
            stdout = stdout[: self._config.max_output_bytes]
            stdout += "\n... [OUTPUT TRUNCATED]"
            truncated_stdout = True

        if len(stderr.encode("utf-8")) > self._config.max_stderr_bytes:
            stderr = stderr[: self._config.max_stderr_bytes]
            stderr += "\n... [STDERR TRUNCATED]"
            truncated_stderr = True

        return CommandResult(
            stdout=stdout,
            stderr=stderr,
            return_code=proc.returncode,
            truncated_stdout=truncated_stdout,
            truncated_stderr=truncated_stderr,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

SANDBOX_AVAILABLE = True
sandbox_manager: SandboxManager | None = None

try:
    _config = _load_config()
    if _config.enabled:
        sandbox_manager = SandboxManager(_config)
    else:
        logger.info("Sandbox disabled in configuration")
except Exception as _exc:
    SANDBOX_AVAILABLE = False
    logger.warning("Sandbox manager unavailable: %s", _exc)
