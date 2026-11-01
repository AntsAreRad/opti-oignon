#!/usr/bin/env python3
"""
SANDBOX MANAGER - OPTI-OIGNON
============================================

Manages fully isolated, disposable sandbox environments for LLM
filesystem and shell tool execution. Every LLM-initiated file or
command operation runs inside a sandbox with strict security boundaries.

File copy-out with human approval workflow.
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

Security audit hardening:
- CommandValidator inspects recently created file contents before
  allowing bash execution that references those files
- Extended base64-pipe-to-shell detection
- Extended Python subprocess pattern detection

Author: Leon
"""

import contextlib
import enum
import errno
import functools
import logging
import os
import re
import resource
import secrets
import shutil
import signal
import sqlite3
import stat
import subprocess
import tempfile
import threading
import time
import uuid
import weakref
from dataclasses import dataclass, field, replace
from typing import Any, Callable

try:
    import fcntl
except ImportError:  # not POSIX: no workspace can be told live from orphaned (reconcile_workspaces)
    fcntl = None

from opti_oignon.db_utils import safe_connect

logger = logging.getLogger(__name__)


# The class of opti_oignon/joined_pattern.py, copied: this module is loaded
# alone in contract windows, where a new import would not resolve. A contract
# holds the two copies equal.
class JoinedPattern:
    """``head``, then anything on the same line, then ``tail``: what a compiled ``head.*tail`` answers to
    ``search``, ``finditer`` and ``findall``, in time linear in the text."""

    def __init__(self, head, tail):
        if head.flags != tail.flags or head.flags & (re.DOTALL | re.VERBOSE):
            raise ValueError("head and tail take the same flags, neither DOTALL nor VERBOSE")
        if not (self._joinable(head.pattern, True, False) and self._joinable(tail.pattern, False, head.groups > 0)):
            raise ValueError("a head or a tail the join would read otherwise")
        self.flags = head.flags
        self.pattern = head.pattern + ".*" + tail.pattern
        self._head, self._tail = head, tail
        self._whole = re.compile(self.pattern, self.flags)

    def finditer(self, text):
        pos, reach = 0, {}
        while True:
            head = self._head.search(text, pos)
            if head is None:
                return
            start, end = head.span()
            if start == end:
                raise ValueError("head matched the empty string")
            stop = text.find("\n", end)
            match = None
            if self._reaches(text, end, len(text) if stop < 0 else stop, reach):
                match = self._whole.match(text, start)
            if match is None:
                pos = start + 1
                continue
            yield match
            pos = match.end()

    def _reaches(self, text, end, stop, reach):
        """True when ``tail`` matches from a place between ``end`` and ``stop``, the end of its line. A line is
        tried from its end back, once: ``reach[stop]`` is the lowest place tried and the highest place ``tail``
        matched from, or -1."""
        low, high = reach.get(stop, (stop + 1, -1))
        while high < 0 and low > end:
            low -= 1
            if self._tail.match(text, low):
                high = low
        reach[stop] = (low, high)
        return high >= end

    @staticmethod
    def _joinable(pattern, first, numbered):
        """True when ``pattern`` reads the same as a part of the join: no alternative at its top level, no group of
        global flags unless ``first`` and at its start, and, when ``numbered``, no numbered back-reference or
        condition (the head's groups would come before the ones it names)."""
        depth, at, inside = 0, 0, False
        while at < len(pattern):
            char = pattern[at]
            if char == "\\":
                if numbered and not inside and pattern[at + 1:at + 2] in tuple("123456789"):
                    return False
                at += 2
                continue
            if inside:
                inside = char != "]"
            elif char == "[":
                inside = True
                at += 2 if pattern[at + 1:at + 2] == "]" else 3 if pattern[at + 1:at + 3] == "^]" else 1
                continue
            elif char == "(":
                if pattern.startswith("(?#", at):
                    close = pattern.find(")", at)
                    at = len(pattern) if close < 0 else close + 1
                    continue
                if re.match(r"\(\?[aiLmsu]*x[aiLmsux]*(?:-[imsx]*)?:", pattern[at:]):
                    return False  # a verbose group: its comments run to a line break the scan does not read
                group = re.match(r"\(\?(?:([aiLmsux]+)\)|\((\d))", pattern[at:])
                if group and group.group(1) and not (first and at == 0):
                    return False
                if group and group.group(2) and numbered:
                    return False
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "|" and depth == 0:
                return False
            at += 1
        return True

    def search(self, text):
        return next(self.finditer(text), None)

    def findall(self, text):
        groups = self._whole.groups
        if groups == 0:
            return [match.group() for match in self.finditer(text)]
        if groups == 1:
            return [match.groups("")[0] for match in self.finditer(text)]
        return [match.groups("") for match in self.finditer(text)]


class ChainedPattern:
    """The parts a blocked pattern joins with ``.*``, searched in time linear in the text: whether the first part
    matches, then each next part from a place between where the one before it ends and the end of that line. A line
    is tried from its end back once for each part after the first, and what was found is kept. Each part is taken by
    the first way it matches from a place: the chain is made only where each part joins the rest exactly
    (``linear_pattern.joins_exactly``), and only of parts that each read a long text in linear time."""

    def __init__(self, pattern, parts):
        self.pattern, self._parts = pattern, parts

    def search(self, text):
        reach = [{} for _ in self._parts[1:]]
        pos = 0
        while pos <= len(text):
            head = self._parts[0].search(text, pos)
            if head is None:
                return None
            if self._reaches(text, 0, head.end(), reach):
                return head
            pos = head.start() + 1
        return None

    def _reaches(self, text, index, end, reach):
        """True when part ``index + 1``, and each after it, matches from a place between ``end`` and the end of its
        line. ``reach[index][stop]`` is the lowest place tried on the line that ends at ``stop`` and the highest
        place found, or -1."""
        stop = text.find("\n", end)
        stop = len(text) if stop < 0 else stop
        low, high = reach[index].get(stop, (stop + 1, -1))
        part = self._parts[index + 1]
        while high < 0 and low > end:
            low -= 1
            found = part.match(text, low)
            if found is not None and (index + 2 == len(self._parts)
                                      or self._reaches(text, index + 1, found.end(), reach)):
                high = low
        reach[index][stop] = (low, high)
        return high >= end


def _joined_parts(source):
    """The parts ``source`` joins with ``.*`` or ``.*?`` at its top level -- outside classes and groups, never an
    escaped dot -- or None when it joins none."""
    parts, depth, at, inside, begin = [], 0, 0, False, 0
    while at < len(source):
        char = source[at]
        if char == "\\":
            at += 2
            continue
        if inside:
            inside = char != "]"
        elif char == "[":
            inside = True
            at += 2 if source[at + 1:at + 2] == "]" else 3 if source[at + 1:at + 3] == "^]" else 1
            continue
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0 and source.startswith(".*", at):
            parts.append(source[begin:at])
            at += 3 if source.startswith(".*?", at) else 2
            begin = at
            continue
        at += 1
    return parts + [source[begin:]] if parts else None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "config", "sandbox.yaml"
)

# Per-sandbox resource-cap bounds. Out-of-range configured
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

# Workspace lifecycle bounds. Same clamp discipline as the
# caps: out-of-range values are corrected, never disabled. idle TTL 0
# is the documented "disabled" value (no idle reaping).
_TTL_SECONDS_DEFAULT = 3600                     # 1 hour idle -> destroyed
_TTL_SECONDS_MIN = 0                            # 0 disables the idle TTL
_TTL_SECONDS_MAX = 30 * 86400                   # 30 days
_DISK_SOFT_BYTES_DEFAULT = 512 * 1024 ** 2     # 512 MiB per workspace
_DISK_SOFT_BYTES_MIN = 1 * 1024 ** 2           # 1 MiB
_DISK_SOFT_BYTES_MAX = 16 * 1024 ** 3          # 16 GiB

# Bounded disk-use walk: the approximate per-workspace disk figure is
# a scandir walk capped on entries and depth so a pathological tree cannot
# stall the manager. The figure is approximate by design.
_DISK_WALK_MAX_ENTRIES = 10000
_DISK_WALK_MAX_DEPTH = 32

# Copy-in bounds. Same clamp discipline (correct, never
# disable). The per-request upload total and the clone total are ALSO bounded
# by the per-workspace disk soft quota: the caps below bound a single
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

# Exact pre-walk bound for the clone source: unlike the approximate
# disk-use walk above, hitting this depth REFUSES the clone rather than
# silently undercounting -- a cap enforced on a partial figure is no cap.
_CLONE_WALK_MAX_DEPTH = 32

# Streamed copy/hash chunk size for the copy-in paths.
_COPYIN_CHUNK_BYTES = 65536

# Diff review bound. A review gate must present the WHOLE
# change set or refuse: a truncated or paginated review would let the user
# approve against an incomplete picture. Exceeding this entry bound REFUSES
# the diff (the exact-prewalk refusal semantics), never truncates.
_DIFF_MAX_ENTRIES_DEFAULT = 50000
_DIFF_MAX_ENTRIES_MIN = 1
_DIFF_MAX_ENTRIES_MAX = 1000000

# Provision-run timeout. The provision phase is the one
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
    """Normalize the host-share root allowlist.

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
    TEMPDIR = "tempdir"  # Tempdir only -- NOT real isolation


class ApprovalState(enum.Enum):
    """Approval state for sandbox file copy-out.

    Files created inside the sandbox MUST go through this state
    machine before they can be copied to the host filesystem.
    No auto-approve is ever allowed.
    """
    PENDING = "pending"      # Files exist but not yet reviewed
    APPROVED = "approved"    # User explicitly approved specific paths
    REJECTED = "rejected"    # User rejected copy-out


# ---------------------------------------------------------------------------
# Workspace ownership
# ---------------------------------------------------------------------------
# A live workspace's directory is held by its manager under an exclusive
# flock, on a descriptor opened on the directory itself and not inherited by
# the commands run in it, for the session's life. A flock belongs to its open
# file description and goes with its process, so another manager -- in another
# process, or another one in this process -- tells a live workspace from an
# orphan: it removes only a directory whose lock it can take.
def _hold_workspace(path: str) -> int | None:
    """A descriptor holding the directory ``path`` names under an exclusive lock; ``None`` when another holds it.

    Raises OSError when the directory cannot be opened, and FileNotFoundError when ``path`` no longer names the
    directory locked (removed or replaced meanwhile).
    """
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    except BaseException:
        os.close(fd)
        raise
    try:
        held, named = os.fstat(fd), os.lstat(path)
    except BaseException:
        os.close(fd)
        raise
    if (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino):
        os.close(fd)
        raise FileNotFoundError(f"{path} no longer names the directory locked")
    return fd


# What flock answers on a file system that cannot lock a directory: NFS
# emulates it with a write lock, refused (EBADF) on a descriptor opened
# read-only; others answer EINVAL, ENOLCK or EOPNOTSUPP.
_LOCK_REFUSED = frozenset({errno.EBADF, errno.EINVAL, errno.ENOLCK, errno.EOPNOTSUPP})


def _base_locks(base: str) -> bool:
    """Whether ``base``'s file system holds an exclusive flock on a directory: asked once, on ``base`` itself.

    False without fcntl, or where flock is refused; any other failure is raised, never read as a refusal.
    """
    if fcntl is None:
        return False
    fd = os.open(base, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True  # held by another: the file system locks
    except OSError as exc:
        if exc.errno in _LOCK_REFUSED:
            return False
        raise
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    finally:
        os.close(fd)


# The tempdir backend's wrappers, the strongest first: a user and network namespace with a PID namespace of its own
# (_PID_INIT is its first process), then the user and network namespace alone
_UNSHARE_FORMS = (
    ("unshare", "--user", "--net", "--pid", "--fork", "--kill-child", "--"),
    ("unshare", "--user", "--net", "--"),
)

# The first process of a tempdir command's PID namespace: a shell that runs the command as its foreground child (PID
# 2, as under bubblewrap: the same signals, traps and stdin as before) and answers its status, its own stderr sent to
# /dev/null so that a command dying of a signal adds no line of the shell's. The command cannot make this shell leave
# the process group a stop or a destroy kills, nor clear its death signal (where ptrace reaches only a process's
# descendants, as Yama has it by default); whatever the command does to it, this shell's end -- the command
# returned, the group killed, or the shell signalled -- makes the kernel kill every process left in the namespace: a
# background job, one moved out of its group with ``setsid``, one that cleared its own death signal
_PID_INIT = 'exec 4>&2 2>/dev/null; bash -c "$1" 2>&4 4>&-; exit "$?"'


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL the process group ``proc`` leads and reap it; never raises.

    The child was spawned with ``start_new_session=True``, so its pid is the process-group id: ``killpg`` reaches
    bwrap and its tree. A leader not yet reaped holds its pid, and so its group's id, whatever it ran: it is signalled;
    one already reaped (``returncode`` set) is not, its pid free to be another process's (the race left is the
    check's own instant, as ``Popen.send_signal``'s). Already gone or unreachable: the direct kill and the reap
    follow; stopping must never raise past this point. A process that outlives its command -- a background job
    (``&``) once the command returned, or one moved out of its group (``setsid``) -- dies with it in a PID namespace:
    bubblewrap's, or the tempdir backend's where ``unshare --pid`` works (``_UNSHARE_FORMS``); where it does not, it is
    out of reach, as it is of the stop button.
    """
    if proc.returncode is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:
                pass
    try:
        proc.wait(timeout=5)
    except Exception:  # pragma: no cover - defensive reap
        pass


def _release_holds(holds: dict[str, int | None]) -> None:
    """Close every descriptor of ``holds``: the workspaces they held are orphans the next start removes."""
    for fd in list(holds.values()):
        if fd is None:
            continue
        try:
            os.close(fd)
        except OSError:
            pass
    holds.clear()


def _detect_bwrap() -> tuple[bool, str]:
    """Detect if bubblewrap is available and functional.

    Performs a real functional test (not just binary existence) because
    some container environments block the namespace syscalls bwrap needs.

    Returns:
        (available, message) -- message is version string on success,
        or failure reason on failure.
    """
    from opti_oignon.process_secrets import child_env

    try:
        # Step 1: check binary exists and get version
        ver_result = subprocess.run(
            ["bwrap", "--version"],
            capture_output=True,
            timeout=5,
            env=child_env(),
        )
        if ver_result.returncode != 0:
            return False, "bwrap binary returned non-zero on --version"

        version = ver_result.stdout.decode("utf-8", errors="replace").strip()

        # Step 2: functional test -- actually create a namespace
        # This catches environments where bwrap exists but namespaces
        # are blocked (e.g., unprivileged Docker containers).
        # We bind only the minimum needed to run /bin/echo (or /usr/bin/echo),
        # in an environment cleared as every sandbox run's is.
        test_cmd = ["bwrap", "--clearenv"]
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
            env=child_env(),
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
    # Strict mode -- refuse ALL code execution if bwrap is unavailable
    strict_mode: bool = True
    # Per-sandbox resource caps. limits_enabled gates the caps;
    # resource_backend selects the mechanism. "rlimit" (default) installs
    # RLIMIT_* via a preexec hook (dependency-free, native, per-process AS and
    # per-uid NPROC semantics). "cgroup" wraps the launch in a transient
    # systemd --user scope (stronger aggregate accounting) when systemd-run is
    # available, and falls back to rlimit otherwise -- it never disables caps.
    limits_enabled: bool = True
    resource_backend: str = "rlimit"
    limit_memory_bytes: int = _CAP_MEMORY_BYTES_DEFAULT
    limit_nproc: int = _CAP_NPROC_DEFAULT
    limit_fsize_bytes: int = _CAP_FSIZE_BYTES_DEFAULT
    limit_cpu_seconds: int = _CAP_CPU_SECONDS_DEFAULT
    tmpfs_size_bytes: int = _CAP_TMPFS_BYTES_DEFAULT
    # Seccomp denylist. seccomp_enabled builds and passes the
    # filter on every bwrap launch. seccomp_required makes a build/pass failure
    # REFUSE the launch (fail-secure) rather than run unfiltered; flip it off
    # only with the loud warning logged below.
    seccomp_enabled: bool = True
    seccomp_required: bool = True
    # Workspace lifecycle. Workspaces default ephemeral under
    # workspace_base (which is volatile under /tmp and does not survive a
    # reboot). workspace_persistent=True skips the startup reconcile so files
    # under a configured persistent base survive restarts -- the documented
    # trade-off is a widened disposable window (data persists on disk between
    # runs; write-back to the host stays approval-gated regardless).
    # idle_ttl_seconds destroys idle, unbound workspaces lazily (0 disables).
    # disk_soft_limit_bytes is the per-workspace copy-in soft quota: a copy-in
    # that would exceed it is refused; the workspace itself is never killed.
    workspace_persistent: bool = False
    reconcile_on_start: bool = True
    idle_ttl_seconds: int = _TTL_SECONDS_DEFAULT
    disk_soft_limit_bytes: int = _DISK_SOFT_BYTES_DEFAULT
    # Copy-in. host_share_roots is the allowlist the host
    # browse and clone are confined to (normalized at load: expanduser +
    # realpath, "/" refused, non-directories dropped; empty/unset defaults to
    # the user's home). The upload caps bound a single multipart request
    # (count and per-file bytes); the clone caps bound a single clone (bytes
    # and file count). The request/clone TOTAL is additionally bounded by the
    # per-workspace disk soft quota -- the caps and the quota both
    # apply, neither subsumes the other.
    host_share_roots: list[str] = field(default_factory=list)
    upload_max_files: int = _UPLOAD_MAX_FILES_DEFAULT
    upload_max_file_bytes: int = _UPLOAD_MAX_FILE_BYTES_DEFAULT
    clone_max_bytes: int = _CLONE_MAX_BYTES_DEFAULT
    clone_max_files: int = _CLONE_MAX_FILES_DEFAULT
    # The diff review entry bound. A diff whose classified
    # entry count (baseline union live files) would exceed this REFUSES
    # instead of truncating or paginating: the review gate must present the
    # whole change set or nothing.
    diff_max_entries: int = _DIFF_MAX_ENTRIES_DEFAULT

    # The provision-run timeout. NOT a network default --
    # the per-workspace network flag has no configuration surface at all.
    provision_timeout_seconds: int = _PROVISION_TIMEOUT_DEFAULT

    def __post_init__(self) -> None:
        # Clamp every cap into its documented range. Out-of-range values are
        # corrected, never disabled. resource_backend falls back to the
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
        # Lifecycle clamps (same discipline: correct, never disable).
        self.idle_ttl_seconds = _clamp(
            self.idle_ttl_seconds, _TTL_SECONDS_MIN, _TTL_SECONDS_MAX
        )
        self.disk_soft_limit_bytes = _clamp(
            self.disk_soft_limit_bytes,
            _DISK_SOFT_BYTES_MIN,
            _DISK_SOFT_BYTES_MAX,
        )
        # Copy-in clamps + share-root normalization (correct, never
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
        # Diff review bound (same discipline: correct, never disable).
        self.diff_max_entries = _clamp(
            self.diff_max_entries, _DIFF_MAX_ENTRIES_MIN, _DIFF_MAX_ENTRIES_MAX
        )
        # Provision timeout (same discipline).
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

    # Security.yaml strict_mode overrides sandbox.yaml
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

    A session is a user-owned workspace with a stable id.
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
    # Approval state machine for copy-out
    approval_state: ApprovalState = ApprovalState.PENDING
    approved_paths: set[str] = field(default_factory=set)
    approved_at: float | None = None
    # Workspace lifecycle fields. label is the optional human
    # name; owner_user_id follows the effective_user_id isolation pattern
    # (the workspace implies its owning user); bound_conversation_id mirrors
    # the sandbox_workspace binding store (write-through, the store is the
    # mutation point); network_enabled is the per-workspace network flag --
    # default False, flipped ONLY by set_network_enabled (an
    # explicit user action behind the Daily-only gate; never a config
    # default, never model-triggerable); last_activity drives the idle TTL;
    # timeout_override is the per-sandbox command timeout.
    label: str = ""
    owner_user_id: str = "local"
    bound_conversation_id: str | None = None
    # The conversation the session runs for, given by whoever made it for one (a chat coding session, a
    # conversation's code sandbox): the audit keeps it on each row, which the conversation's deletion forgets.
    conversation_id: str | None = None
    network_enabled: bool = False
    last_activity: float = field(default_factory=time.time)
    # The host's operations in flight on the workspace (SandboxManager.in_use):
    # a copy into it, an apply or a copy out of it. The idle sweep never takes
    # the session while one runs. Under the manager's lock.
    host_operations: int = 0
    timeout_override: int | None = None
    # Deletions confirmed for apply-to-host. PARALLEL to
    # approved_paths by design: approve_files validates os.path.isfile so a
    # deleted entry can never enter approved_paths, and a blanket approve-all
    # can never include a deletion. The load-bearing check is in the apply
    # writer: it deletes only paths that are BOTH confirmed here AND
    # classified "deleted" by the recomputed diff -- confirming a live path
    # can never delete it.
    confirmed_deletions: set[str] = field(default_factory=set)


class WorkspaceQuotaExceeded(Exception):
    """A copy-in would push the workspace past its disk soft quota.

    Raised by the inject paths. Soft semantics: the copy-in is
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

# What a forgotten command becomes.
_FORGOTTEN_WORDS = "[forgotten with its conversation]"
# A chat coding session's id: this prefix, the conversation's first characters, a dash.
_CODING_PREFIX = "cc-"
_CODING_SPAN = 12
# The conversations a deletion forgot in this process (``forget_conversation``): a command of one is recorded
# without its words, until a refused deletion hands it back.
_FORGOTTEN: set[str] = set()
# Held while a mark is checked and a row written, and while a deletion marks and replaces.
_MARKS_LOCK = threading.Lock()


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """The columns ``table`` has in the file, asked of the table itself; empty when it has no such table."""
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


class AuditLog:
    """SQLite-backed audit log for sandbox command history.

    ``conversation_of`` maps a session id to the conversation the session runs
    for (the manager's sessions); without it every row is of no conversation.
    """

    def __init__(self, db_path: str, conversation_of: Callable[[str], str | None] | None = None):
        self._db_path = db_path
        self._conversation_of = conversation_of
        self._lock = threading.Lock()
        # The schema is built at the first connection, not here. This log is
        # constructed at module scope, so building it here opened a database
        # on every import of the package.
        self._schema_ready = False

    def _get_conn(self) -> sqlite3.Connection:
        """Get a SQLite connection.

        Audit fix: routes through get_encrypted_connection().
        """
        conn = safe_connect(self._db_path)
        # What a write frees -- a deletion's words, a clear -- is overwritten, never left in free pages.
        conn.execute("PRAGMA secure_delete = ON")
        if not self._schema_ready:
            # Built on the connection just opened, without taking the lock:
            # every caller of this helper already holds it.
            self._schema_ready = True
            self._create_schema(conn)
        return conn

    def _init_db(self) -> None:
        """Create the audit table if it does not exist."""
        with self._lock:
            self._schema_ready = True
            conn = self._get_conn()
            try:
                self._create_schema(conn)
            finally:
                conn.close()

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        """Build the schema on a connection the caller already holds.

        Deliberately takes no lock. Every method in this class takes the
        instance lock and then asks for a connection; if building the schema
        reacquired it, the first of them to run before the schema existed
        would hang rather than fail -- the worst way for a suite to tell you,
        because it tells you last.
        """
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
                isolation_backend TEXT DEFAULT '',
                conversation_id TEXT NOT NULL DEFAULT ''
            )
        """)
        # An older file's rows did not keep their conversation: they are of none, and only a chat coding session's
        # id still names one. The table is asked, never a duplicate caught: the asking holds whatever the driver raises.
        if "conversation_id" not in _columns(conn, "sandbox_audit"):
            conn.execute("ALTER TABLE sandbox_audit ADD COLUMN conversation_id TEXT NOT NULL DEFAULT ''")
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_session
            ON sandbox_audit(session_id)
        """)
        conn.commit()

    def _conversation(self, session_id: str) -> str:
        """The conversation ``session_id`` runs for, ``''`` for none; a lookup that fails is said, and the row is of
        none."""
        resolve = self._conversation_of
        if resolve is None:
            return ""
        try:
            return resolve(session_id) or ""
        except Exception as exc:  # noqa: BLE001 - a row is never lost to the lookup
            logger.warning("sandbox audit: the conversation of session %s could not be read (%s); its row is of none",
                           session_id, type(exc).__name__)
            return ""

    def log_command(
        self,
        session_id: str,
        command: str,
        result: CommandResult,
    ) -> None:
        """Record a command execution in the audit log; a command of a conversation this process forgot is recorded
        without its words."""
        conversation = self._conversation(session_id)
        with self._lock, _MARKS_LOCK:
            forgotten = bool(conversation) and conversation in _FORGOTTEN
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO sandbox_audit
                        (session_id, timestamp, command, return_code,
                         blocked, block_reason, timed_out,
                         stdout_len, stderr_len, isolation_backend,
                         conversation_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        time.time(),
                        _FORGOTTEN_WORDS if forgotten else command,
                        result.return_code,
                        1 if result.blocked else 0,
                        "" if forgotten else result.block_reason,
                        1 if result.timed_out else 0,
                        len(result.stdout),
                        len(result.stderr),
                        result.isolation_backend,
                        conversation,
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
                # The approval audit, if its table exists yet: asked of the file, never caught from a driver.
                if _columns(conn, "sandbox_approval_audit"):
                    conn.execute("DELETE FROM sandbox_approval_audit")
                conn.commit()
            finally:
                conn.close()

    def _ensure_approval_table(self) -> None:
        """Create the approval audit table if it does not exist."""
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
                        detail TEXT DEFAULT '',
                        conversation_id TEXT NOT NULL DEFAULT ''
                    )
                """)
                if "conversation_id" not in _columns(conn, "sandbox_approval_audit"):
                    conn.execute(
                        "ALTER TABLE sandbox_approval_audit ADD COLUMN conversation_id TEXT NOT NULL DEFAULT ''")
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
        """Record an approval/rejection/copy-out event; one of a conversation this process forgot is recorded without
        its words."""
        self._ensure_approval_table()
        import json
        conversation = self._conversation(session_id)
        with self._lock, _MARKS_LOCK:
            forgotten = bool(conversation) and conversation in _FORGOTTEN
            paths_json = json.dumps([] if forgotten else (paths or []))
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO sandbox_approval_audit
                        (session_id, timestamp, action, paths, dest_dir, detail, conversation_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, time.time(), action, paths_json, "" if forgotten else dest_dir,
                     "" if forgotten else detail, conversation),
                )
                conn.commit()
            finally:
                conn.close()

    def forget_conversation(self, conversation_id: str) -> int:
        """Forget the words a deleted conversation's sessions left, and keep the record of what ran; how many rows
        lost words.

        In both tables, a command and its block reason, an event's paths, destination and detail are replaced; the
        time, the outcome, the sizes and the backend stay. Rows an older version wrote, which did not keep their
        conversation, are reached through a chat coding session's id (``cc-`` and the conversation's first twelve
        characters); the other sessions' older rows cannot be. A log that cannot be opened raises: a deletion that
        asks must never take it for one with nothing to forget. A file that is not there holds nothing, and is not
        created; the schema is never changed here. What is replaced is overwritten. From then on, a command of the
        conversation is recorded without its words in this process, unless the forgetting fails.
        """
        try:
            with _MARKS_LOCK:
                # Marked under the writes' lock: a row written first loses its words below, one written after is
                # written without them.
                _FORGOTTEN.add(conversation_id)
                return self._forget_rows(conversation_id)
        except BaseException:
            _FORGOTTEN.discard(conversation_id)
            raise

    def _forget_rows(self, conversation_id: str | None) -> int:
        """Replace the words of the conversation's rows -- every row's when ``conversation_id`` is None -- in both
        tables; how many rows lost words."""
        if not os.path.exists(self._db_path):
            return 0
        conn = safe_connect(self._db_path)
        try:
            conn.execute("PRAGMA secure_delete = ON")
            forgotten = 0
            for table, words, kept, values in (
                ("sandbox_audit", "command = ?, block_reason = ''", "command != ?", (_FORGOTTEN_WORDS,)),
                ("sandbox_approval_audit", "paths = '[]', dest_dir = '', detail = ''",
                 "(paths != '[]' OR dest_dir != '' OR detail != '')", ()),
            ):
                columns = _columns(conn, table)
                if not columns:
                    continue
                reach, asked = "1", ()
                if conversation_id is not None:
                    # A row an older version wrote names its conversation only through a chat coding session's id.
                    prefix = f"{_CODING_PREFIX}{conversation_id[:_CODING_SPAN]}-"
                    reach, asked = "substr(session_id, 1, ?) = ?", (len(prefix), prefix)
                    if "conversation_id" in columns:
                        reach = f"conversation_id = ? OR (conversation_id = '' AND {reach})"
                        asked = (conversation_id, *asked)
                forgotten += conn.execute(f"UPDATE {table} SET {words} WHERE ({reach}) AND {kept}",
                                          (*values, *asked, *values)).rowcount
            conn.commit()
            return forgotten
        finally:
            conn.close()

    def forget_every_word(self) -> int:
        """The user's wipe: every row loses its words, the record of what ran kept; how many rows lost words. A log
        that cannot be opened raises; a file that is not there holds nothing, and is not created."""
        with _MARKS_LOCK:
            return self._forget_rows(None)

    def get_approval_log(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieve all approval audit entries for a session."""
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

    Additions:
    - Tracks recently created files and inspects their content before
      allowing bash execution that references them (write-then-execute
      attack vector).
    - Extended base64-pipe-to-shell detection (covers more encoding
      bypass patterns).
    - Extended Python subprocess pattern detection.
    """

    # Boundary note: this regex command denylist is SECONDARY
    # defense-in-depth ONLY. It is bypassable -- base64/eval, here-docs,
    # write-then-execute, obscure interpreters -- and must never be relied on
    # as the security boundary. The boundary is the namespace isolation that
    # bwrap provides (hardened with a clean env, full namespace
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
        JoinedPattern(re.compile(r"\bsubprocess\b", re.IGNORECASE),
                      re.compile(r"\b(Popen|call|run|check_output)\b", re.IGNORECASE)),
        re.compile(r"\bos\.system\b", re.IGNORECASE),
        re.compile(r"\bexec\s*\(", re.IGNORECASE),
        JoinedPattern(re.compile(r"base64", re.IGNORECASE), re.compile(r"\|\s*(bash|sh|zsh|python)", re.IGNORECASE)),
        re.compile(r"\bimport\s+(subprocess|socket|http|urllib|ftplib|smtplib)\b"),
        re.compile(r"\b(rm\s+-rf\s+/|dd\s+if=)"),
    ]

    # Hardcoded command checks of the form head.*tail, read once per line
    # (JoinedPattern), so a line that repeats the head costs no more than once.
    _EVAL_NETWORK = JoinedPattern(re.compile(r"\b(eval|exec)\b", re.IGNORECASE),
                                  re.compile(r"\b(socket|urllib|requests|http)\b", re.IGNORECASE))
    _PYTHON_C = re.compile(r"python[23]?\s+-c\s+", re.IGNORECASE)
    _PYTHON_C_NETWORK = JoinedPattern(_PYTHON_C, re.compile(
        r"\b(urllib|http\.|requests\.|httpx|aiohttp|socket|ftplib|smtplib|xmlrpc|"
        r"telnetlib|poplib|imaplib|nntplib)",
        re.IGNORECASE,
    ))
    _PYTHON_C_PROCESS = JoinedPattern(_PYTHON_C, re.compile(
        r"\b(subprocess|os\.system|os\.popen|os\.exec[a-z]*|pty\.spawn)\b", re.IGNORECASE))
    _BASE64_TO_SHELL = JoinedPattern(re.compile(r"base64\s+(-d|--decode)", re.IGNORECASE),
                                     re.compile(r"\|\s*(bash|sh|zsh|python|perl|ruby)", re.IGNORECASE))
    _XXD_TO_SHELL = JoinedPattern(re.compile(r"xxd\s+-r", re.IGNORECASE),
                                  re.compile(r"\|\s*(bash|sh|zsh|python)", re.IGNORECASE))
    _PYTHON_C_SUBPROCESS = JoinedPattern(_PYTHON_C, re.compile(
        r"\bsubprocess\.(Popen|call|run|check_output|check_call)\b", re.IGNORECASE))

    def __init__(self, config: SandboxConfig):
        self._blocked_commands = [
            cmd.lower() for cmd in (config.blocked_commands or [])
        ]
        # Made at the first command (reading what a pattern costs is not done at import): see _blocked.
        self._blocked_sources = list(config.blocked_patterns or [])
        self._blocked_patterns: list | None = None
        self._blocked_refused: list[str] = []
        self._blocked_lock = threading.Lock()

        # Track recently created files for write-then-execute detection
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

    def _blocked(self):
        """(patterns, refused): each configured pattern that reads a long command in linear time, as written, or
        as a chain of the parts it joins with ``.*`` where each part does and joins the rest exactly, or as the
        engine without backtracking reads it; and every other one, which refuses the configuration. A rule of refusal never falls silently: while one is refused,
        every command is refused, naming it. A pattern the engine refuses as invalid is left out with a warning; one
        too large for it to read is refused as a costly one is."""
        with self._blocked_lock:
            if self._blocked_patterns is None and not self._blocked_sources:
                self._blocked_patterns = []
            if self._blocked_patterns is None:
                from opti_oignon.linear_pattern import CostlyPattern, compile_linear, joins_exactly
                patterns, refused = [], []
                for source in self._blocked_sources:
                    try:
                        patterns.append(compile_linear(source, re.IGNORECASE, engine=False))
                        continue
                    except CostlyPattern:
                        pass
                    except re.error as exc:
                        logger.warning("Invalid blocked pattern '%s': %s", source, exc)
                        continue
                    try:
                        parts = _joined_parts(source)
                        flags = re.compile(source, re.IGNORECASE).flags
                        if parts is None or not all(parts) or flags & (re.DOTALL | re.VERBOSE) or not all(
                                joins_exactly(parts[at], ".*".join(parts[at + 1:]), flags)
                                for at in range(len(parts) - 1)):
                            raise CostlyPattern("no exact chain")
                        patterns.append(ChainedPattern(source, [compile_linear(part, flags, engine=False)
                                                                for part in parts]))
                        continue
                    except (re.error, RecursionError, OverflowError, ValueError, TypeError):
                        pass
                    try:
                        patterns.append(compile_linear(source, re.IGNORECASE))
                    except (re.error, RecursionError, OverflowError, ValueError, TypeError):
                        refused.append(source)
                if refused:
                    logger.error("Sandbox configuration refused: blocked pattern(s) %s may take more than linear "
                                 "time on a long command; every command is refused until they are rewritten",
                                 refused)
                self._blocked_patterns, self._blocked_refused = patterns, refused
            return self._blocked_patterns, self._blocked_refused

    def validate(self, command: str) -> tuple[bool, str]:
        """Validate a command. Returns (is_safe, reason).

        Returns (True, '') if the command passes all checks,
        or (False, reason) if it should be blocked.
        """
        if not command or not command.strip():
            return False, "Empty command"

        patterns, refused = self._blocked()
        if refused:
            return False, (f"Sandbox configuration refused: the blocked pattern {refused[0]!r} may take more than "
                           f"linear time on a long command")

        cmd_lower = command.strip().lower()

        # Check blocked command prefixes
        for blocked in self._blocked_commands:
            if cmd_lower.startswith(blocked.rstrip()):
                return False, f"Blocked command: {blocked.strip()}"

        # Check blocked patterns
        for pattern in patterns:
            if pattern.search(command):
                return False, f"Blocked pattern: {pattern.pattern}"

        # Additional hardcoded safety checks that must never be bypassed
        # regardless of config contents

        # Prevent rm targeting root or absolute paths outside workspace
        # Linear on a long run: the blanks before the slash are read from the start of their run. Read from the first
        # rm of each line: any later rm of that line that reaches a slash, the first one reaches too.
        if re.search(r"^(?:(?!\brm\b).)*\brm\b(?:.*\S)?\s+/(?!workspace)", command, re.MULTILINE):
            return False, "rm targeting paths outside /workspace"

        # Block eval/exec with network or OS calls
        if self._EVAL_NETWORK.search(command):
            return False, "Blocked eval/exec with network access"

        # Block python -c with network modules (defense-in-depth,
        # catches patterns the YAML regex might miss)
        if self._PYTHON_C_NETWORK.search(command):
            return False, "Blocked python -c with network module"

        # Block Python subprocess with shell=True or Popen. Read before the check below, which covers it: after that
        # check it could never refuse anything, and its reason was never given.
        if self._PYTHON_C_SUBPROCESS.search(command):
            return False, "Blocked python -c with subprocess calls"

        # Block python -c with subprocess (constructed args attack)
        if self._PYTHON_C_PROCESS.search(command):
            return False, "Blocked python -c with subprocess/os.exec"

        # Block base64-encoded commands piped to execution
        # Extended: covers hex decode, xxd, and printf byte sequences
        if self._BASE64_TO_SHELL.search(command):
            return False, "Blocked base64 decode piped to shell"

        # Block echo with base64 piped to decode then execute
        if re.search(
            r"echo\s+['\"]?[A-Za-z0-9+/=]{20,}['\"]?\s*\|\s*base64\s+(-d|--decode)",
            command,
        ):
            return False, "Blocked echo of base64 payload to decode"

        # Block xxd reverse piped to shell
        if self._XXD_TO_SHELL.search(command):
            return False, "Blocked xxd reverse piped to shell"

        # Check write-then-execute attack vector
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


# An export's free name: counted for its first tries, then drawn at random, so that no number of earlier exports of
# one name -- the folder is shared and never emptied -- leaves it none. A file name holds at most this many bytes.
_COUNTED_TRIES = 100
_DRAWN_TRIES = 32
_NAME_MAX = 255


def _fit_name(head: str, stem: str, tail: str) -> str:
    """``head + stem + tail``, cut a whole character at a time so the name holds in ``_NAME_MAX`` bytes: the stem from
    its end, then, when a tail is longer than a name leaves room for, the tail from its end. Never a byte of a
    character: a name is cut where its characters end."""
    room = _NAME_MAX - len(os.fsencode(head))
    while stem and len(os.fsencode(stem + tail)) > room:
        stem = stem[:-1]
    while tail and len(os.fsencode(stem + tail)) > room:
        tail = tail[:-1]
    return head + stem + tail


def _create_free_export(dest_dir: str, name: str) -> tuple[str, int]:
    """A new file for an export in ``dest_dir``, named ``name``, else ``<stem>-<n><suffix>``, then
    ``<stem>-<random><suffix>`` (the stem cut to fit a file name): its path, and a descriptor open for writing. Never
    a file already there, nor through a link at the name (``O_EXCL``): an export never overwrites another file,
    whoever's it is."""
    stem, suffix = os.path.splitext(name)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    marks = [f"-{n}" for n in range(1, _COUNTED_TRIES)] + [f"-{secrets.token_hex(6)}" for _ in range(_DRAWN_TRIES)]
    for candidate in [name] + [_fit_name("", stem, mark + suffix) for mark in marks]:
        path = os.path.join(dest_dir, candidate)
        try:
            return path, os.open(path, flags, 0o600)
        except FileExistsError:
            continue
    raise FileExistsError(f"no free name for {name} in {dest_dir}")


def _open_beneath(root: str, relative: str) -> int:
    """A descriptor of the file ``relative`` names under ``root``, opened one component at a time without following
    a link (``dir_fd``, ``O_NOFOLLOW``): a link anywhere on the way -- one the sandbox put there after a check -- is
    refused, never followed, and so is a file that is not regular with one name (``OSError``). The last open does not
    wait (``O_NONBLOCK``): a FIFO put in a file's place is refused at once, never waited on."""
    parts = [part for part in os.path.normpath(relative).split(os.sep) if part not in ("", ".")]
    if not parts or ".." in parts:
        raise OSError(errno.EINVAL, f"not a path beneath the workspace: {relative}")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    # A folder is walked, not read: O_PATH asks no permission to list it, as a path walk does not
    directory = getattr(os, "O_PATH", os.O_RDONLY) | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(root, directory)
    try:
        for part in parts[:-1]:
            inner = os.open(part, directory | nofollow, dir_fd=fd)
            os.close(fd)
            fd = inner
        handle = os.open(parts[-1], os.O_RDONLY | nofollow | getattr(os, "O_NONBLOCK", 0), dir_fd=fd)
    finally:
        os.close(fd)
    st = os.fstat(handle)
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
        os.close(handle)
        why = "not a regular file" if not stat.S_ISREG(st.st_mode) else "a second name of a file (a hard link)"
        raise OSError(errno.EPERM, why, relative)
    return handle


def _why_not_read(exc: OSError) -> str:
    """What a refusal of ``_open_beneath`` says of a path, to whoever asked for it: not there, a link met, not a
    regular file, a second name of a file, a file that may not be read."""
    if isinstance(exc, (FileNotFoundError, NotADirectoryError)):
        return "File not found"
    if exc.errno == errno.ELOOP:
        return "A link stands there, not followed"
    if exc.errno == errno.EPERM and exc.strerror:
        return exc.strerror[:1].upper() + exc.strerror[1:]
    return exc.strerror or str(exc)


def _workspace_parts(requested: str) -> list[str]:
    """The components of a path the user names in a workspace, relative or under ``/workspace``: ``""`` and ``.``
    dropped, ``..`` refused (``OSError``) -- a path is walked as written, never folded first."""
    if requested == "/workspace" or requested.startswith("/workspace/"):
        requested = requested[len("/workspace"):]
    elif os.path.isabs(requested):
        raise OSError(errno.EINVAL, f"not a path beneath the workspace: {requested}")
    parts = [part for part in requested.split(os.sep) if part not in ("", ".")]
    if ".." in parts:
        raise OSError(errno.EINVAL, f"not a path beneath the workspace: {requested}")
    return parts


def _folder_beneath(root: str, parts: list[str], make: bool = False) -> int:
    """A descriptor (``O_PATH``) of the folder ``parts`` name under ``root``, reached one component at a time without
    following a link: a link or anything but a folder on the way is refused (``OSError``), never followed -- even a
    link that stays inside the workspace. With ``make``, a missing folder is made beneath its parent, as ``makedirs``
    made it (mode 0o777 under the umask)."""
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_PATH", os.O_RDONLY) | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(root, directory)
    try:
        for part in parts:
            if make:
                try:
                    os.mkdir(part, 0o777, dir_fd=fd)
                except FileExistsError:
                    pass
            inner = os.open(part, directory | nofollow, dir_fd=fd)
            os.close(fd)
            fd = inner
    except BaseException:
        os.close(fd)
        raise
    return fd


# A removal holds one descriptor per folder level down to this depth and never enters a folder deeper: that one is
# moved up to the top of the tree under a fresh name and removed from there, or left standing when it cannot move
_REMOVE_DEPTH = 32


def _remove_tree(path: str) -> bool:
    """Remove the workspace tree at ``path``; whether it is gone.

    The tree is walked by descriptors, one folder at a time, and never through a link: a link is removed as the entry
    it is, whatever it names. A folder that denies its own listing or emptying -- one the sandbox's code froze
    (``chmod a-w``, ``0o000``) -- is given back its owner's rights (``u+rwx``) through the descriptor that holds it
    (``_entered``). A folder below ``_REMOVE_DEPTH`` levels is never entered there: it is moved up to the top first,
    or left standing when it cannot move (``_past_depth``): the walk never holds more descriptors than that, whatever
    depth the sandbox's code built or however it stops the moves. An entry that cannot be removed is passed over and
    the walk goes on: what still stands -- a file the owner cannot undo, a folder a running command keeps filling --
    is left, and the answer says so.
    """
    try:
        top = os.open(os.path.dirname(path), getattr(os, "O_PATH", os.O_RDONLY) | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return not os.path.lexists(path)
    # The folders being emptied, the deepest last: (its parent, its name there, its descriptor, its names left)
    frames: list[tuple[int, str, int, list[str]]] = []
    try:
        entered = _entered(top, os.path.basename(path))
        if entered is not None:
            frames.append((top, os.path.basename(path), *entered))
        while frames:
            parent, name, folder, left = frames[-1]
            if left:
                inner = left.pop()
                if len(frames) >= _REMOVE_DEPTH:
                    _past_depth(folder, inner, frames[0])
                    continue
                entered = _entered(folder, inner)
                if entered is not None:
                    frames.append((folder, inner, *entered))
                continue
            frames.pop()
            os.close(folder)
            with contextlib.suppress(OSError):
                os.rmdir(name, dir_fd=parent)
    finally:
        for _parent, _name, folder, _left in frames:
            os.close(folder)
        os.close(top)
    return not os.path.lexists(path)


def _entered(parent: int, name: str) -> tuple[int, list[str]] | None:
    """The entry ``name`` of the folder ``parent`` (a descriptor): removed if it is no folder (left if it cannot be),
    and ``None``; if it is a folder, its descriptor (``O_PATH``) and its names, its owner's rights given back first
    where it lacked them. Never through a link: a link is no folder, and a folder swapped for anything else between
    its listing and its opening is passed over (``None``), its mode untouched."""
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_PATH", os.O_RDONLY) | getattr(os, "O_DIRECTORY", 0)
    try:
        before = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if not stat.S_ISDIR(before.st_mode):
            os.unlink(name, dir_fd=parent)
            return None
        folder = os.open(name, directory | nofollow, dir_fd=parent)
    except OSError:
        return None
    try:
        now = os.fstat(folder)
        if not stat.S_ISDIR(now.st_mode) or (now.st_dev, now.st_ino) != (before.st_dev, before.st_ino):
            raise OSError(errno.ENOTDIR, "no longer the folder listed", name)
        mode = stat.S_IMODE(now.st_mode)
        if mode & stat.S_IRWXU != stat.S_IRWXU:
            _give_owner_rights(folder, mode)
        listing = os.open(os.curdir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0), dir_fd=folder)
        try:
            names = os.listdir(listing)
        finally:
            os.close(listing)
    except OSError:
        os.close(folder)
        return None
    return folder, names


def _past_depth(folder: int, name: str, top: tuple[int, str, int, list[str]]) -> None:
    """The entry ``name`` of ``folder`` (a descriptor) past ``_REMOVE_DEPTH`` levels, never entered there: anything
    but a folder is unlinked where it is; a folder is moved to the top of the tree being removed (``top``: its frame
    in ``_remove_tree``) under a fresh name the walk reaches there, or passed over when it cannot move -- left
    standing, the tree with it. A rename moves the entry itself, never what a link names. A folder moved to another
    parent rewrites its own ``..``: one its sandbox froze is given back its owner's rights first, through a
    descriptor."""
    try:
        before = os.stat(name, dir_fd=folder, follow_symlinks=False)
        if not stat.S_ISDIR(before.st_mode):
            os.unlink(name, dir_fd=folder)
            return
        if stat.S_IMODE(before.st_mode) & stat.S_IRWXU != stat.S_IRWXU:
            nofollow = getattr(os, "O_NOFOLLOW", 0)
            directory = getattr(os, "O_PATH", os.O_RDONLY) | getattr(os, "O_DIRECTORY", 0)
            moving = os.open(name, directory | nofollow, dir_fd=folder)
            try:
                now = os.fstat(moving)
                if not stat.S_ISDIR(now.st_mode) or (now.st_dev, now.st_ino) != (before.st_dev, before.st_ino):
                    return
                _give_owner_rights(moving, stat.S_IMODE(now.st_mode))
            finally:
                os.close(moving)
        fresh = f".removed-{secrets.token_hex(8)}"
        os.rename(name, fresh, src_dir_fd=folder, dst_dir_fd=top[2])
    except OSError:
        return
    top[3].append(fresh)


def _give_owner_rights(folder: int, mode: int) -> None:
    """Give the folder ``folder`` (a descriptor, ``O_PATH`` where there is one) its owner's rights (``u+rwx``) beside
    ``mode``, never through a name: through its own entry in ``/proc`` (an ``O_PATH`` descriptor takes no
    ``fchmod``) or, where ``/proc`` cannot serve, through the folder itself opened from that descriptor (``.``) --
    which only a folder its owner may still read and search allows (``chmod a-w``, not ``0o000``). Raises OSError
    when neither can."""
    try:
        os.chmod(f"/proc/self/fd/{folder}", mode | stat.S_IRWXU)
        return
    except OSError:
        pass
    itself = os.open(os.curdir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0), dir_fd=folder)
    try:
        os.fchmod(itself, mode | stat.S_IRWXU)
    finally:
        os.close(itself)


def _hold_shut_workspace(path: str) -> int | None:
    """``_hold_workspace``, a workspace folder its sandbox shut (no read or search) given back its owner's rights first
    (``_give_owner_rights``, the folder reached without following a link): live or orphaned, it can be checked."""
    try:
        return _hold_workspace(path)
    except PermissionError:
        pass
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    folder = os.open(path, getattr(os, "O_PATH", os.O_RDONLY) | getattr(os, "O_DIRECTORY", 0) | nofollow)
    try:
        _give_owner_rights(folder, stat.S_IMODE(os.fstat(folder).st_mode))
    finally:
        os.close(folder)
    return _hold_workspace(path)


def _plain_name(name: str) -> str:
    """``name`` when it names an entry of one folder, else ``OSError``."""
    if not name or name in (".", "..") or os.sep in name or "\x00" in name:
        raise OSError(errno.EINVAL, f"not a file name: {name!r}")
    return name


def _new_file_beneath(folder: int, name: str, mode: int = 0o666) -> int:
    """A descriptor open for writing on a new file ``name`` in ``folder`` (a descriptor): never a file already there,
    nor through a link at the name (``O_EXCL``, ``O_NOFOLLOW``)."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    return os.open(_plain_name(name), flags, mode, dir_fd=folder)


def _pour(source: int, out: int, st: os.stat_result) -> int:
    """Copy every byte of the descriptor ``source`` to the descriptor ``out``, then give ``out`` the permission bits
    of ``st`` (without setuid, setgid or sticky) and its times; the bytes written."""
    written = 0
    while True:
        chunk = os.read(source, _COPYIN_CHUNK_BYTES)
        if not chunk:
            break
        view = memoryview(chunk)
        while view:
            view = view[os.write(out, view):]
        written += len(chunk)
    os.fchmod(out, stat.S_IMODE(st.st_mode) & 0o777)
    os.utime(out, ns=(st.st_atime_ns, st.st_mtime_ns))
    return written


def _open_in_place(folder: int, name: str) -> int | None:
    """A descriptor open for writing on the file at ``name`` in ``folder`` (a descriptor) when it is a regular file
    with one name, opened without following a link and without waiting; else None -- nothing there, a link, a
    folder, a FIFO, a second name of another file, or a file its owner may not write."""
    flags = os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        out = os.open(name, flags, dir_fd=folder)
    except OSError:
        return None
    st = os.fstat(out)
    if stat.S_ISREG(st.st_mode) and st.st_nlink == 1:
        return out
    os.close(out)
    return None


def _replace_beneath(folder: int, name: str, source: str) -> int:
    """Copy the host file ``source`` to ``name`` in ``folder`` (a descriptor); the bytes written, with the source's
    permission bits (without setuid, setgid or sticky) and times. A regular file with one name already at ``name`` is
    written in place, as ``copytree`` wrote over it (a folder its owner may not write still takes it), through a
    descriptor opened without following a link -- unless it is the source itself, refused as ``copy2`` refused it.
    Anything else -- nothing, a link, a second name of a host file, a file its owner may not write, a FIFO -- is
    replaced as an entry: the copy is written under a fresh name beside ``name``, then renamed over it, never written
    through what stood there. A folder at the name is refused (``OSError``), and so is a source that is not a regular
    file -- a FIFO, a device -- at once (``O_NONBLOCK``: never waited on). A fresh copy that fails is removed and the
    name left as it was; a file written in place is left as far as the write got, as ``copytree`` left it."""
    _plain_name(name)
    handle = os.open(source, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0))
    try:
        st = os.fstat(handle)
        if not stat.S_ISREG(st.st_mode):
            raise OSError(errno.EINVAL, f"not a regular file: {source}")
        out = _open_in_place(folder, name)
        if out is not None:
            try:
                there = os.fstat(out)
                if (there.st_dev, there.st_ino) == (st.st_dev, st.st_ino):
                    # Truncated, the source would be read empty
                    raise OSError(errno.EINVAL, f"the source is the file at its name: {source}")
                os.ftruncate(out, 0)
                return _pour(handle, out, st)
            finally:
                os.close(out)
        for _ in range(_DRAWN_TRIES):
            fresh = f".copy-in-{secrets.token_hex(8)}"
            try:
                out = _new_file_beneath(folder, fresh, 0o600)
                break
            except FileExistsError:
                continue
        else:
            raise FileExistsError(f"no free name beside {name}")
        try:
            try:
                written = _pour(handle, out, st)
            finally:
                os.close(out)
            os.replace(fresh, name, src_dir_fd=folder, dst_dir_fd=folder)
        except BaseException:
            try:
                os.unlink(fresh, dir_fd=folder)
            except OSError:
                pass
            raise
        return written
    finally:
        os.close(handle)


def _copy_tree_beneath(src: str, root: str, parts: list[str]) -> tuple[int, list[tuple[str, str, str]]]:
    """Copy the host folder ``src`` to the folder ``parts`` name under ``root``, merged into what is there; the files
    written, and the failures. The source is read as ``shutil.copytree`` read it (a link in it followed: a file link
    copied as its content, a folder link as the folder; a dangling one a failure); the destination is written only
    beneath ``root`` and through no link (``_folder_beneath``, ``_replace_beneath``): a link at a folder's place
    refuses that folder, one at a file's name is replaced by the file. A source folder reached again below itself
    through a link is refused, where ``copytree`` recursed until the path failed. Each failure is recorded as
    ``copytree`` recorded it -- (source, destination, reason) -- and the rest copied; a folder's permission bits and
    times are set after its content, as ``copystat`` set them."""
    written = 0
    errors: list[tuple[str, str, str]] = []
    # (source folder, its destination's parts, the source folders above it, its content done)
    stack: list[tuple[str, list[str], tuple[tuple[int, int], ...], bool]] = [(src, parts, (), False)]
    while stack:
        current, here, above, done = stack.pop()
        shown = os.path.join(root, *here)
        if done:
            try:
                st = os.stat(current)
                folder = _folder_beneath(root, here)
                try:
                    own = os.open(".", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0), dir_fd=folder)
                finally:
                    os.close(folder)
                try:
                    os.fchmod(own, stat.S_IMODE(st.st_mode) & 0o777)
                    os.utime(own, ns=(st.st_atime_ns, st.st_mtime_ns))
                finally:
                    os.close(own)
            except OSError as why:
                errors.append((current, shown, str(why)))
            continue
        try:
            st = os.stat(current)
            identity = (st.st_dev, st.st_ino)
            if identity in above:
                raise OSError(errno.ELOOP, f"a link back to a folder above it: {current}")
            with os.scandir(current) as it:
                entries = list(it)
            folder = _folder_beneath(root, here, make=True)
        except OSError as why:
            errors.append((current, shown, str(why)))
            continue
        try:
            stack.append((current, here, above, True))
            for entry in entries:
                target = os.path.join(shown, entry.name)
                try:
                    if entry.is_dir():
                        stack.append((entry.path, here + [entry.name], above + (identity,), False))
                    else:
                        _replace_beneath(folder, entry.name, entry.path)
                        written += 1
                except OSError as why:
                    errors.append((entry.path, target, str(why)))
        finally:
            os.close(folder)
    return written, errors


# A listing of a workspace goes no deeper than this many folders: its walk holds one descriptor per level.
_LIST_MAX_DEPTH = 128


def _walk_beneath(root: str):
    """Every entry under the folder ``root`` that is neither a folder nor a link, as (relative path, lstat), in
    ``os.walk``'s top-down order -- a folder's files by name, then its folders' trees one after the other, the folders
    taken by name where ``os.walk`` took them as the folder listed them -- walked through descriptors: each folder opened from its parent's descriptor without following a link,
    each entry read itself (lstat beneath its folder), a link never followed nor listed, a folder that cannot be
    listed skipped as ``os.walk`` skipped it. Folders deeper than ``_LIST_MAX_DEPTH`` are not walked; the walk logs
    how many."""
    directory = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    cut = 0

    def entries(folder: int) -> tuple[list[tuple[str, os.stat_result]], list[str]]:
        """A folder's files (anything but a folder or a link) with their lstat, and its folders, by name."""
        try:
            with os.scandir(folder) as it:
                names = sorted(entry.name for entry in it)
        except OSError:
            return [], []
        files, folders = [], []
        for name in names:
            try:
                st = os.stat(name, dir_fd=folder, follow_symlinks=False)
            except OSError:
                continue
            if stat.S_ISDIR(st.st_mode):
                folders.append(name)
            elif not stat.S_ISLNK(st.st_mode):
                files.append((name, st))
        return files, folders

    def walk(folder: int, here: str, depth: int):
        nonlocal cut
        files, folders = entries(folder)
        for name, st in files:
            yield os.path.join(here, name) if here else name, st
        for name in folders:
            if depth >= _LIST_MAX_DEPTH:
                cut += 1
                continue
            try:
                inner = os.open(name, directory | nofollow, dir_fd=folder)
            except OSError:
                continue
            try:
                yield from walk(inner, os.path.join(here, name) if here else name, depth + 1)
            finally:
                os.close(inner)

    try:
        top = os.open(root, directory)
    except OSError:
        return
    try:
        yield from walk(top, "", 0)
    finally:
        os.close(top)
        if cut:
            logger.warning("workspace listing left %d folder(s) deeper than %d levels", cut, _LIST_MAX_DEPTH)


# ---------------------------------------------------------------------------
# Bwrap command builder
# ---------------------------------------------------------------------------

def _tempdir_env(workspace: str) -> dict[str, str]:
    """The environment of a command run without bubblewrap: a minimal PATH and
    the workspace, nothing of this process's -- no secret, no variable a shell
    runs code from (S-01 / C-01)."""
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": workspace,
        "TMPDIR": workspace,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }


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

    ``allow_network`` is keyword-only and defaults to False,
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
        allow_network: Provision-run network grant; default False.

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

    # Minimal /dev (null, zero, urandom, etc. -- NOT the real /dev)
    cmd.extend(["--dev", "/dev"])

    # Isolated /proc (shows only sandbox PID namespace processes)
    cmd.extend(["--proc", "/proc"])

    # Isolated /tmp (tmpfs, separate from host) with a size cap. The
    # --size option applies to the immediately following filesystem mount, so
    # it must precede --tmpfs. The read-write workspace is a real bind, not a
    # tmpfs: its disk is bounded by RLIMIT_FSIZE (per file) plus the
    # workspace disk quota, not by --size.
    cmd.extend([
        "--size", str(int(config.tmpfs_size_bytes)),
        "--tmpfs", "/tmp",
    ])

    # Namespace isolation flags. net and pid were already unshared; this adds
    # ipc, uts, and cgroup so the child shares no host SysV/POSIX IPC, sees no
    # host hostname/domainname, and has no host cgroup view. net and pid stay
    # listed explicitly (not folded into --unshare-all) so each isolation
    # boundary is individually pinned and the user namespace / uid-gid mapping
    # is left untouched.
    #
    # --unshare-net is unconditional on every task run; ONLY
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
        "--unshare-ipc",       # No host SysV/POSIX IPC sharing
        "--unshare-uts",       # No host hostname/domainname leak
        "--unshare-cgroup",    # No host cgroup view
        "--new-session",       # No terminal/tty escape
        "--die-with-parent",   # Kill sandbox if parent dies
    ])

    # Name resolution for the provision run only. File-level ro-binds
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

    # Seccomp denylist filter. The fd is created and passed by
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
# Resource caps
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


def _copying_in(copy):
    """A copy into a session's workspace, ``copy(self, session_id, ...)``, run with its session in use
    (``SandboxManager.in_use``): the idle sweep never takes the workspace while it is written. An explicit destroy is
    not held off; when one ended the session meanwhile, the copy answers so (``ValueError``), whether it returned --
    with files that went with the workspace -- or failed for the removal."""

    @functools.wraps(copy)
    def copied(self, session_id, *args, **kwargs):
        with self.in_use(session_id) as session:
            try:
                done = copy(self, session_id, *args, **kwargs)
            except Exception as exc:
                # The destroy removed the workspace under the copy, which
                # failed for it: the destroy is the answer, not its trace
                if not session.active:
                    raise _destroyed_during(session_id) from exc
                raise
            if not session.active:
                raise _destroyed_during(session_id)
            return done

    return copied


def _destroyed_during(session_id: str) -> ValueError:
    return ValueError(f"Session {session_id} was destroyed while files were copied into it")


def _copying_out(copy):
    """A copy out of a session's workspace to the host, ``copy(self, session_id, ...)``, run with its session in use
    (``SandboxManager.in_use``): the idle sweep never takes the workspace while it is read. An explicit destroy is not
    held off, and what was copied is on the host: the answer stands."""

    @functools.wraps(copy)
    def copied(self, session_id, *args, **kwargs):
        with self.in_use(session_id):
            return copy(self, session_id, *args, **kwargs)

    return copied


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
        # The unshare wrapper the tempdir backend runs a command in, the
        # strongest that works here (_UNSHARE_FORMS; empty: plain bash)
        self._unshare_prefix: list[str] | None = None
        # Per-session running-process registry. The stop path
        # needs a handle on the running children, so spawns register their
        # Popen here (under the lock) for the duration of the command and
        # deregister in a finally; a session may run several commands at
        # once, each kept. A session with an entry is "running"; without
        # (an empty set is dropped), "idle".
        self._running_procs: dict[str, set[subprocess.Popen]] = {}
        # Each live session's hold on its workspace (_hold_workspace), released by destroy_sandbox, or when this
        # manager is dropped with sessions still live.
        self._holds: dict[str, int] = {}
        self._release = weakref.finalize(self, _release_holds, self._holds)
        # The workspaces a destroy could not remove whole, by path, with their
        # holds: counted against the limit until gone, removed again by the
        # idle sweep and cleanup_all, released with the manager.
        self._leftovers: dict[str, int | None] = {}
        self._release_leftovers = weakref.finalize(self, _release_holds, self._leftovers)

        # Detect isolation backend
        self._bwrap_available, self._bwrap_info = _detect_bwrap()
        self._configured_backend = self._resolve_backend()

        # Ensure workspace base exists
        os.makedirs(self._config.workspace_base, mode=0o700, exist_ok=True)
        # Whether its workspaces can be held. Where they cannot, they are
        # made unheld and no start removes any: an orphan stays until it is
        # removed by hand.
        self._base_locks = _base_locks(self._config.workspace_base)
        if not self._base_locks:
            logger.warning(
                "Sandbox workspace base %s cannot be locked (%s): workspaces "
                "are made unheld, no start can tell one live from an orphan "
                "and none removes one; orphans stay until removed by hand",
                self._config.workspace_base,
                "no flock here" if fcntl is None else "its file system refuses flock",
            )

        # Startup reconcile. Sessions are in-memory and die with their
        # manager, so a sandbox-* directory under workspace_base that no
        # manager holds is an orphan; reap it (reset_*-style) unless the user
        # opted into a persistent base, where surviving directories are
        # deliberate (logged, inert; write-back stays gated). A directory
        # another manager holds -- the server's, seen from a command line, or
        # another manager of this process -- is live and left.
        if self._config.reconcile_on_start:
            try:
                self.reconcile_workspaces()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Workspace reconcile failed: %s", exc)

        # Initialize audit log
        audit_path = os.path.join(
            self._config.workspace_base, self._config.audit_db_path
        )
        # The audit asks the manager through a weak reference: a bound method
        # would hold the manager in a cycle, and a manager dropped with live
        # sessions would keep their holds until a cycle collection.
        manager = weakref.ref(self)

        def conversation_of(session_id: str) -> str | None:
            held = manager()
            return held._conversation_of(session_id) if held is not None else None

        self._audit = AuditLog(audit_path, conversation_of=conversation_of)

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

        In strict_mode, refuse tempdir fallback entirely.
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
                logger.error(
                    "Sandbox strict_mode is ON but isolation_backend is "
                    "'tempdir', which gives no real isolation: code "
                    "execution will be BLOCKED. Set isolation_backend to "
                    "'auto' or 'bwrap', or strict_mode: false."
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

    def _bulbe_pins(self) -> bool:
        """Whether Bulbe pins the strict settings at this moment.

        The security mode is read at each use, so a switch to Bulbe pins
        them without a restart. A mode that cannot be read pins them; without
        the security-mode module there is no Bulbe to pin for.
        """
        try:
            from opti_oignon.security_mode import is_bulbe
        except ImportError:
            return False
        try:
            return bool(is_bulbe())
        except Exception:  # noqa: BLE001 - an unreadable mode pins the strict settings
            return True

    @property
    def effective_config(self) -> SandboxConfig:
        """The configuration in force.

        Daily keeps every setting as configured. Under Bulbe each switch that
        lowers the bubblewrap path reads at its strict value, whatever
        sandbox.yaml or security.yaml say: strict mode on, a tempdir backend
        given way to auto, the seccomp filter on and required, the resource
        limits on, and a degraded sandbox confirmed by the user.
        """
        if not self._bulbe_pins():
            return self._config
        backend = self._config.isolation_backend
        return replace(
            self._config,
            strict_mode=True,
            isolation_backend="auto" if backend.lower() == "tempdir" else backend,
            seccomp_enabled=True,
            seccomp_required=True,
            limits_enabled=True,
            require_degraded_confirmation=True,
        )

    @property
    def config(self) -> SandboxConfig:
        """Access the sandbox configuration in force (see effective_config)."""
        return self.effective_config

    @property
    def _isolation_backend(self) -> IsolationBackend:
        """The backend in force: under Bulbe a configured tempdir gives way to
        bubblewrap wherever bubblewrap runs."""
        if (
            self._configured_backend == IsolationBackend.TEMPDIR
            and self._bwrap_available
            and self.effective_config.isolation_backend.lower() != "tempdir"
        ):
            return IsolationBackend.BWRAP
        return self._configured_backend

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
    def bwrap_in_use(self) -> bool:
        """Whether commands run under bubblewrap: bwrap is the resolved
        backend and it is present. Installed is not in use: a configured
        tempdir backend runs without it."""
        return (
            self._isolation_backend == IsolationBackend.BWRAP
            and self._bwrap_available
        )

    @property
    def degraded_mode(self) -> bool:
        """Whether running in degraded (tempdir-only) mode."""
        return self._isolation_backend == IsolationBackend.TEMPDIR

    @property
    def strict_mode(self) -> bool:
        """Whether strict mode is in force (always under Bulbe)."""
        return self.effective_config.strict_mode

    @property
    def execution_blocked(self) -> bool:
        """Whether code execution is blocked (strict_mode, bwrap not in use)."""
        return self.strict_mode and not self.bwrap_in_use

    def get_isolation_status(self) -> dict[str, Any]:
        """Return comprehensive isolation status for health checks.

        Returns a dict suitable for inclusion in /api/health responses.
        """
        strict = self.strict_mode
        if self.bwrap_in_use:
            level = "bwrap"
        elif strict:
            level = "blocked"
        else:
            level = "tempdir"

        return {
            "isolation_level": level,
            "bwrap_available": self._bwrap_available,
            "bwrap_info": self._bwrap_info,
            "strict_mode": strict,
            "execution_blocked": self.execution_blocked,
            "backend": self._isolation_backend.value,
        }

    @property
    def degraded_confirmed(self) -> bool:
        """Whether the user has confirmed degraded mode."""
        return self._degraded_confirmed

    @property
    def active_session_count(self) -> int:
        """Count of currently active sandbox sessions, with the destroyed
        workspaces not yet removed: the places the limit counts."""
        with self._lock:
            return sum(1 for s in self._sessions.values() if s.active) + len(self._leftovers)

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
        label: str = "",
        owner_user_id: str = "local",
        timeout_override: int | None = None,
    ) -> SandboxSession:
        """Create a fresh isolated sandbox workspace.

        Args:
            session_id: Unique identifier for this sandbox session, or None
                to auto-generate one (previously a None leaked into
                the key and the directory prefix).
            label: Optional human label for the workspace manager.
            owner_user_id: Owning user per the effective_user_id isolation
                pattern; defaults to the single-user "local".
            timeout_override: Optional per-sandbox command timeout in
                seconds; None uses the config default.

        Returns:
            SandboxSession with workspace path set.

        Raises:
            ValueError: If session_id already exists or limits exceeded.
            RuntimeError: If sandbox is disabled, or degraded mode is not
                confirmed when required.
        """
        if not self._config.enabled:
            raise RuntimeError("Sandbox is disabled in configuration")

        # Enforce degraded mode confirmation. Only the configuration and the
        # user's own confirmation decide it; no caller can ask past it.
        if self.degraded_mode:
            if self.effective_config.require_degraded_confirmation:
                if not self._degraded_confirmed:
                    raise RuntimeError(
                        "Sandbox is in DEGRADED mode (no bwrap). "
                        "True filesystem isolation is NOT available. "
                        f"Reason: {self._bwrap_info}. "
                        "Call confirm_degraded_mode() to acknowledge the "
                        "risks, or install bubblewrap for real isolation."
                    )

        # Reap idle, unbound workspaces past the TTL before the
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
            if active + len(self._leftovers) >= self._config.max_concurrent_sessions:
                raise self._limit_reached()

        # Create isolated temp directory with strict permissions, held from
        # its making on
        workspace, hold = self._make_workspace(session_id)

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
            taken = session_id in self._sessions
            # Counted again under the lock that registers: creations that all
            # passed the first count while others made their workspaces
            # never register past the limit together
            full = not taken and (
                sum(1 for s in self._sessions.values() if s.active)
                + len(self._leftovers)
                >= self._config.max_concurrent_sessions
            )
            refusal = self._limit_reached() if full else None
            if not taken and not full:
                self._sessions[session_id] = session
                if hold is not None:
                    self._holds[session_id] = hold
        if taken or full:
            # Another create of the same id, or enough others to reach the
            # limit, finished while this one made its workspace: this one's
            # workspace goes, and its hold with it.
            shutil.rmtree(workspace, ignore_errors=True)
            if hold is not None:
                os.close(hold)
            if taken:
                raise ValueError(f"Session '{session_id}' already exists")
            raise refusal

        logger.info(
            "Sandbox created: session=%s, path=%s, isolation=%s",
            session_id,
            workspace,
            self._isolation_backend.value,
        )
        return session

    def _limit_reached(self) -> ValueError:
        """The refusal of a creation past the limit, naming the destroyed
        workspaces it still counts (taken under the lock)."""
        message = (
            f"Maximum concurrent sessions reached "
            f"({self._config.max_concurrent_sessions})"
        )
        if self._leftovers:
            message += (
                f": {len(self._leftovers)} destroyed workspace(s) not yet "
                f"removed count against it"
            )
        return ValueError(message)

    def _make_workspace(self, session_id: str) -> tuple[str, int | None]:
        """A new workspace directory under workspace_base and its hold, taken before anything is put in it.

        Another manager's reconciliation may remove the empty directory between its making and its hold: it is
        dropped and another made. Where the base cannot be locked there is no hold (``None``); where it can, a hold
        that fails otherwise (descriptors exhausted) removes the directory and is raised: a workspace is never left
        unheld where another start would remove it.
        """
        for _attempt in range(3):
            workspace = tempfile.mkdtemp(
                prefix=f"sandbox-{session_id}-",
                dir=self._config.workspace_base,
            )
            if not self._base_locks:
                os.chmod(workspace, 0o700)
                return workspace, None
            try:
                hold = _hold_workspace(workspace)
            except FileNotFoundError:
                continue
            except BaseException:
                shutil.rmtree(workspace, ignore_errors=True)
                raise
            if hold is None:
                continue
            try:
                os.chmod(workspace, 0o700)
            except BaseException:
                os.close(hold)
                raise
            return workspace, hold
        raise RuntimeError("No workspace could be held: another manager removed each one as it was made")

    def destroy_sandbox(self, session_id: str, *, idle_for: float | None = None) -> bool:
        """Destroy a sandbox and remove all its files.

        The session's running commands are killed first (``_kill_group``):
        none runs on in a destroyed workspace, nor refills it while it is
        removed (a process that outlived its command dies with it in a PID
        namespace; without one, it is out of reach: see ``_kill_group``). A folder the sandbox froze is given back its owner's rights
        on the way (``_remove_tree``). A workspace that still stands is kept, with
        its hold: it counts against the limit until the idle sweep or
        ``cleanup_all`` removes it. The session is destroyed either way.

        Args:
            session_id: Session to destroy.
            idle_for: The idle sweep's TTL: the session is destroyed only if,
                under the lock that ends it, it is still unbound, running
                nothing, in use by no host operation (``in_use``) and idle
                for longer than this (its decision was taken on a snapshot; a
                command or a copy may have started since).

        Returns:
            True if destroyed, False if session not found (or, with
            ``idle_for``, no longer idle).
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            if idle_for is not None and (
                not session.active
                or session.bound_conversation_id is not None
                or self._running_procs.get(session_id)
                or session.host_operations
                or time.time() - session.last_activity <= idle_for
            ):
                return False
            session.active = False
            # Its running commands, killed before the workspace is removed:
            # none refills a tree being removed, none runs on in it.
            # They stay registered until they end, so that a second destroy
            # racing this one kills them too before it removes
            running = list(self._running_procs.get(session_id, ()))
            # Log pending approved paths that were never copied out
            pending_approved = sorted(session.approved_paths)

        for proc in running:
            _kill_group(proc)
        if running:
            logger.info(
                "Sandbox destroyed: session=%s, %d running command(s) killed first",
                session_id,
                len(running),
            )

        # Remove workspace directory tree
        workspace = session.workspace_path
        gone = True
        if os.path.isdir(workspace):
            gone = _remove_tree(workspace)
            if not gone:
                logger.warning(
                    "Sandbox destroyed: session=%s, but its workspace %s "
                    "could not be removed whole: it counts against the limit "
                    "until the idle sweep removes it",
                    session_id,
                    workspace,
                )

        with self._lock:
            ended = self._sessions.pop(session_id, None) is not None
            self._running_procs.pop(session_id, None)
            hold = self._holds.pop(session_id, None)
            if not gone and ended:
                # Its place is still taken: kept, hold and all (by the one
                # destroy that ended the session, when two raced)
                self._leftovers[workspace] = hold
                hold = None
        if hold is not None:
            os.close(hold)

        # Clear file tracking for this session
        self._validator.clear_recent_files()

        # Audit the destruction with approval context
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

    # -- Workspace lifecycle ---------------------------------------------

    def is_running(self, session_id: str) -> bool:
        """Whether a command is currently executing in this workspace."""
        with self._lock:
            return bool(self._running_procs.get(session_id))

    def stop_command(self, session_id: str) -> bool:
        """SIGKILL the workspace's running commands; keep the workspace.

        The stop path of spec section 4.2: kills every tracked process group
        of the session (each bwrap child and its bash tree die together) and
        reaps them, then marks the workspace idle. Files persist for inspection; the
        workspace is NOT destroyed. Fail-secure no-op semantics: stopping a
        workspace with nothing running returns False and never errors or
        leaks state.

        Args:
            session_id: Target workspace.

        Returns:
            True if running commands were killed, False if nothing was
            running (no-op).

        Raises:
            ValueError: If the session does not exist (the route maps this
                to an honest 404).
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ValueError(f"Session not found: {session_id}")
            # Left registered until they end, as a destroy leaves them: a destroy
            # racing this stop still kills them before it removes
            procs = list(self._running_procs.get(session_id, ()))

        if not procs:
            return False

        for proc in procs:
            _kill_group(proc)

        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.last_activity = time.time()

        self._audit.log_approval(
            session_id,
            action="workspace_stopped",
            paths=[],
            detail=f"{len(procs)} running command(s) killed (SIGKILL, process group)",
        )
        logger.info("Workspace command(s) stopped: session=%s, %d killed", session_id, len(procs))
        return True

    def set_conversation(self, session_id: str, conversation_id: str | None) -> None:
        """The conversation a session runs for, set by whoever made it for one: its audit rows keep it."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.conversation_id = conversation_id

    def _conversation_of(self, session_id: str) -> str | None:
        """The conversation a session's audit rows keep: the one it runs for, else its workspace's bound one."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        return session.conversation_id or session.bound_conversation_id

    def set_binding(self, session_id: str, conversation_id: str | None) -> None:
        """Write-through mirror of the conversation binding.

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

    @contextlib.contextmanager
    def in_use(self, session_id: str):
        """Count a host operation on an active session's workspace while it runs.

        Yields the session (its ``workspace_path`` is the workspace) and
        counts the operation on it (``host_operations``): while it runs the
        idle sweep never takes the session -- ``destroy_sandbox``'s check
        made again with ``idle_for`` declines it, and the sweep's snapshot
        skips it --, so a copy into the workspace, an apply or a copy out of
        it that runs past what was left of the idle TTL is never cut by the
        removal of its workspace. The count is the session's own: a session
        made again under the same id is not held by an operation on the old
        one. Being in use is not activity: the operation counts its own (one
        refused whole counts none). An explicit destroy is not held off: it
        ends the session as before (``session.active`` turns False).

        Raises:
            ValueError: If the session is unknown or inactive, or its
                workspace directory is missing (``_get_active_workspace``);
                nothing then stays counted.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            counted = session is not None and session.active
            if counted:
                session.host_operations += 1
        try:
            self._get_active_workspace(session_id)
            yield session
        finally:
            if counted:
                with self._lock:
                    session.host_operations -= 1

    def reconcile_workspaces(self) -> int:
        """Reap orphaned workspace directories under workspace_base.

        Sessions are in-memory and die with their manager, so a sandbox-*
        directory no manager holds (_hold_workspace) is an orphan. Only
        direct child DIRECTORIES whose name starts with "sandbox-", which no
        tracked session owns and whose lock this manager takes, are removed,
        under that lock; one another manager holds is live and left, one
        that cannot be opened is left as not checked, one that cannot be
        removed whole is left, each said, and only those removed whole are
        counted. Files
        (notably the audit database living in workspace_base) are never
        touched. Without flock nothing tells a live workspace from an
        orphan, and nothing is removed; so where the base's file system
        refuses flock. In persistent mode the reap is
        skipped entirely and survivors are logged as detached (inert until
        a future cycle re-attaches them; write-back stays gated).

        Returns:
            Count of directories removed (0 in persistent mode).
        """
        base = self._config.workspace_base
        if not os.path.isdir(base):
            return 0

        with self._lock:
            # A workspace a destroy left standing is this manager's, removed
            # by its idle sweep: never taken for another manager's
            tracked = {
                s.workspace_path for s in self._sessions.values()
            } | set(self._leftovers)

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

        if not self._base_locks:
            if orphans:
                logger.warning(
                    "Workspace reconcile: %s cannot be locked, so no live "
                    "workspace can be told from an orphan: %d director(ies) "
                    "left, not checked: %s",
                    base,
                    len(orphans),
                    ", ".join(os.path.basename(p) for p in orphans),
                )
            return 0

        removed = 0
        live: list[str] = []
        unchecked: list[str] = []
        stuck: list[str] = []
        for path in orphans:
            try:
                hold = _hold_shut_workspace(path)
            except FileNotFoundError:
                continue  # gone meanwhile: its owner destroyed it
            except OSError as exc:
                unchecked.append(f"{os.path.basename(path)} ({exc.strerror or exc})")
                continue
            if hold is None:
                live.append(os.path.basename(path))
                continue
            try:
                gone = _remove_tree(path)
            finally:
                os.close(hold)
            if not gone:
                stuck.append(os.path.basename(path))
            else:
                removed += 1
        if live:
            logger.info(
                "Workspace reconcile: %d workspace(s) another manager holds "
                "left in place: %s",
                len(live),
                ", ".join(live),
            )
        if unchecked:
            logger.warning(
                "Workspace reconcile: %d director(ies) could not be opened "
                "or locked and were left, not checked: %s",
                len(unchecked),
                ", ".join(unchecked),
            )
        if stuck:
            logger.warning(
                "Workspace reconcile: %d orphaned director(ies) could not be "
                "removed whole and were left: %s",
                len(stuck),
                ", ".join(stuck),
            )
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
        of: the TTL is enabled (> 0), nothing is running in it, the host is
        not copying into it, applying from it nor copying out of it
        (``in_use``; the check made again under the lock declines one), it
        is not bound to a conversation (an explicit binding is user intent
        and exempts it), and last_activity is older than the TTL.

        The workspaces a destroy could not remove whole are removed again
        first, whatever the TTL (``_remove_leftovers``).

        Returns:
            Count of workspaces destroyed.
        """
        self._remove_leftovers()
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
                and not s.host_operations
                and (now - s.last_activity) > ttl
            ]
        reaped = 0
        for sid in stale:
            if self.destroy_sandbox(sid, idle_for=ttl):
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

        A scandir walk capped on entries and depth so a
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
        """Register a file creation for write-then-execute detection.

        Called by file_tools after creating a file in the sandbox.
        The CommandValidator will inspect this file's content before
        allowing bash commands that reference it.

        Args:
            relative_path: File path relative to workspace root.
            content: The content that was written.
        """
        self._validator.register_created_file(relative_path, content)

    @_copying_in
    def inject_files(
        self,
        session_id: str,
        file_paths: list[str],
    ) -> list[str]:
        """Copy files from the host into the sandbox workspace.

        Only existing, readable files are copied. Files are placed
        in the workspace root, preserving only the filename (no
        directory structure from the source). A regular file with one
        name already at the name is written in place, through no link;
        any other entry -- a link or a second name the sandbox put there
        -- is replaced by a copy renamed over it, never written through.
        A folder at the name, a source that is that very file, or a copy
        that fails, is skipped and logged.

        Args:
            session_id: Target sandbox session.
            file_paths: List of host file paths to inject.

        Returns:
            List of paths (inside workspace) of successfully copied files.

        Raises:
            ValueError: If session not found or inactive, or destroyed while
                the files were copied (``_copying_in``).
        """
        workspace = self._get_active_workspace(session_id)
        injected: list[str] = []

        # Per-workspace disk soft quota. Compute the incoming size up
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
            # A file of the workspace written in place; a link the
            # sandbox put at the name, or a second name of a host file,
            # replaced by a copy renamed over it, never written through
            try:
                folder = _folder_beneath(workspace, [])
                try:
                    _replace_beneath(folder, filename, src_real)
                finally:
                    os.close(folder)
            except OSError as exc:
                logger.warning("Inject refused: %s (%s)", src_path, exc)
                continue
            resolved = os.path.join(os.path.realpath(workspace), filename)
            injected.append(resolved)
            logger.debug("Injected file: %s -> %s", src_path, resolved)

        self.touch_activity(session_id)
        return injected

    @_copying_in
    def inject_directory(
        self,
        session_id: str,
        src_dir: str,
        dest_subdir: str = "",
    ) -> int:
        """Copy a directory tree into the sandbox workspace.

        The tree lands at ``[dest_subdir/]<basename(src_dir)>``, merged
        into what is there, as ``shutil.copytree(dirs_exist_ok=True)``
        landed it: a link in the HOST source is followed, each failure
        is recorded while the rest is copied, and ``shutil.Error``
        names them at the end. The WORKSPACE side is walked without
        following a link (``_copy_tree_beneath``): a link the sandbox
        put at a folder's place refuses that folder, one at a file's
        name is replaced by the file, and nothing is written outside
        the workspace. Where they part: a device in the source (or a
        link to one), which ``copytree`` read, is refused like a FIFO
        or a socket; a link in the source back to a folder above it is
        refused; a file the workspace's owner may not write is replaced
        rather than refused; a source path ending in ``..`` or a
        ``dest_subdir`` with ``..`` is refused.

        Args:
            session_id: Target sandbox session.
            src_dir: Host directory to copy.
            dest_subdir: Subdirectory within workspace (default: root).

        Returns:
            Number of files copied.

        Raises:
            ValueError: If session not found, inactive, or path invalid, or
                destroyed while the tree was copied (``_copying_in``).
            shutil.Error: If some entries could not be copied.
        """
        workspace = self._get_active_workspace(session_id)

        if not os.path.isdir(src_dir):
            raise ValueError(f"Source is not a directory: {src_dir}")

        if dest_subdir:
            valid, _dest_path, err = validate_sandbox_path(
                workspace, dest_subdir
            )
            if not valid:
                raise ValueError(f"Invalid destination: {err}")
        # Walked as written: a source named ``x/..`` would land beside
        # the workspace
        try:
            base = _workspace_parts(dest_subdir)
            top = _workspace_parts(os.path.basename(src_dir))
        except OSError as exc:
            raise ValueError(f"Invalid destination: {exc}") from exc

        # Per-workspace disk soft quota on the incoming tree.
        incoming = self._workspace_disk_use(src_dir)
        self._check_disk_quota(session_id, workspace, incoming)

        try:
            os.close(_folder_beneath(workspace, base, make=True))
        except OSError as exc:
            raise ValueError(f"Invalid destination: {exc}") from exc
        count, errors = _copy_tree_beneath(src_dir, workspace, base + top)
        if errors:
            raise shutil.Error(errors)

        self.touch_activity(session_id)

        dest_final = os.path.join(workspace, *(base + top))
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

        Soft semantics: raises WorkspaceQuotaExceeded; the caller's
        copy-in is refused and the workspace is left untouched. The
        tmpfs_size_bytes setting bounds /tmp inside the sandbox; this quota
        covers the workspace bind on the host side.
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
    # Copy-in: drag-and-drop upload, allowlisted host
    # browse, and the symlink-safe host clone. All three are EXPLICIT,
    # user-initiated actions through the manager UI -- the model can
    # trigger none of them; the agent only ever sees /workspace.
    # -----------------------------------------------------------------

    @staticmethod
    def sanitize_upload_filename(name: str) -> tuple[bool, str]:
        """Sanitize a client-supplied upload filename.

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

    @_copying_in
    def upload_files(
        self,
        session_id: str,
        items: list[tuple[str, Any, int]],
        dest_subdir: str = "",
    ) -> dict[str, Any]:
        """Write user-uploaded file streams into the workspace (5.1).

        ``items`` is a list of (filename, binary stream, size_bytes); sizes
        are summed BEFORE any write. Caps: at most ``upload_max_files`` per
        request and ``upload_max_file_bytes`` per file; the request total is
        bounded by the disk soft quota. Any exceeded cap refuses the
        WHOLE request (WorkspaceQuotaExceeded -> the route's 413) with the
        workspace untouched. Individually invalid names and destination
        collisions are refused PER FILE (never overwritten, never renamed)
        and reported honestly; valid files are written in bounded chunks
        with an on-the-fly sha256 so the caller can record the baseline
        manifest (section 6.1) without a second read. The destination
        folder is reached as written, without following a link: a link
        on the way, or ``..``, refuses the upload (ValueError); each file
        is created only where nothing stands (``O_EXCL``), so a link the
        sandbox puts at its name is refused, never followed.

        Returns {"written": [{path, name, size, sha256}], "refused":
        [{name, reason}], "written_bytes": int}; a destroy that ended the
        session meanwhile is answered ``ValueError`` (``_copying_in``).
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
            valid, _dest_root, err = validate_sandbox_path(
                workspace, dest_subdir
            )
            if not valid:
                raise ValueError(f"Invalid destination: {err}")
        # The folder is reached without following a link and each file
        # created where no entry stands (O_EXCL): a link the sandbox
        # puts at a name after any check is refused, never followed
        try:
            base = _workspace_parts(dest_subdir)
            folder = _folder_beneath(workspace, base, make=True)
        except OSError as exc:
            raise ValueError(f"Invalid destination: {exc}") from exc
        dest_root = os.path.join(os.path.realpath(workspace), *base)

        written: list[dict[str, Any]] = []
        refused: list[dict[str, str]] = []
        written_bytes = 0
        import hashlib

        try:
            for name, stream, _size in items:
                ok, clean = self.sanitize_upload_filename(name)
                if not ok:
                    refused.append({"name": name, "reason": clean})
                    continue
                rel = (
                    os.path.join(dest_subdir, clean) if dest_subdir else clean
                )
                resolved = os.path.join(dest_root, clean)
                digest = hashlib.sha256()
                try:
                    handle = _new_file_beneath(folder, clean)
                except FileExistsError:
                    refused.append({
                        "name": name,
                        "reason": "Destination already exists (not overwritten)",
                    })
                    continue
                except OSError as exc:
                    refused.append({"name": name, "reason": str(exc)})
                    continue
                size_written = 0
                with os.fdopen(handle, "wb") as out:
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
        finally:
            os.close(folder)

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
        (5.2a).

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
        """Exact bounded pre-walk of a clone source (5.2b).

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

    @_copying_in
    def clone_directory(
        self,
        session_id: str,
        src_path: str,
        dest_subdir: str = "",
    ) -> dict[str, Any]:
        """Clone an allowlisted host directory into the workspace (spec
        5.2b), symlink-safe, with the section 6.1 baseline hashes computed
        on the fly.

        The source must resolve under an allowlisted share root (the same
        confinement as the browse). The exact pre-walk enforces the clone
        caps AND the remaining quota before any copy. The copy itself
        never follows a symlink (every symlink is skipped and counted; its
        target is never read, copied, or exposed) and skips device/special
        files. Regular files are copied in bounded chunks with an
        on-the-fly sha256 per file -- the returned ``manifest`` maps each
        relative path (under the clone destination) to its content hash,
        the seam the diff review consumes. The destination
        ``<workspace>/[dest_subdir/]<basename(src)>`` must not already
        exist (FileExistsError -> the route's 409): explicit, never merged.
        The workspace side is written through no link: each folder is
        reached beneath the workspace without following one, the
        destination made only where nothing stands, each file created
        only where nothing stands (``O_EXCL``) -- a link or a file the
        sandbox puts at a file's name in a folder of the clone while it
        runs is refused and the file skipped; one put at a folder's
        place refuses the clone (FileExistsError), its target never
        written. A ``dest_subdir`` with ``..`` is refused, even one that
        stays inside the workspace.

        ``inject_directory`` keeps ``copytree``'s source semantics (a
        link in the host source is followed); this path skips them. A
        destroy that ended the session meanwhile is answered
        ``ValueError`` (``_copying_in``).
        """
        workspace = self._get_active_workspace(session_id)
        src = self._resolve_share_path(src_path)

        if dest_subdir:
            valid, _dest_parent, err = validate_sandbox_path(
                workspace, dest_subdir
            )
            if not valid:
                raise ValueError(f"Invalid destination: {err}")
        try:
            base = _workspace_parts(dest_subdir)
        except OSError as exc:
            raise ValueError(f"Invalid destination: {exc}") from exc
        dest_parent = os.path.join(os.path.realpath(workspace), *base)
        top = os.path.basename(src)
        dest_final = os.path.join(dest_parent, top)
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
        dest_rel_root = "/".join(base + [top])

        try:
            parent = _folder_beneath(workspace, base, make=True)
        except OSError as exc:
            raise ValueError(f"Invalid destination: {exc}") from exc
        try:
            # Made only where nothing stands, a link included
            os.mkdir(top, 0o777, dir_fd=parent)
        finally:
            os.close(parent)
        stack: list[tuple[str, list[str]]] = [(src, base + [top])]
        while stack:
            cur_src, here = stack.pop()
            try:
                folder = _folder_beneath(workspace, here, make=True)
            except NotADirectoryError as exc:
                # Something the sandbox put at a folder's place while the
                # clone ran: refused, never followed -- the route's 409,
                # as when makedirs met a file there
                raise FileExistsError(
                    f"Clone destination folder replaced while the clone "
                    f"ran: {'/'.join(here)}"
                ) from exc
            try:
                with os.scandir(cur_src) as it:
                    for entry in it:
                        rel = "/".join(here + [entry.name])
                        try:
                            if entry.is_symlink():
                                skipped_symlinks += 1
                                continue
                            if entry.is_dir(follow_symlinks=False):
                                stack.append((entry.path, here + [entry.name]))
                                continue
                            if not entry.is_file(follow_symlinks=False):
                                skipped_special += 1
                                continue
                            digest = hashlib.sha256()
                            with open(entry.path, "rb") as fin, \
                                    os.fdopen(
                                        _new_file_beneath(folder, entry.name),
                                        "wb",
                                    ) as fout:
                                while True:
                                    chunk = fin.read(_COPYIN_CHUNK_BYTES)
                                    if not chunk:
                                        break
                                    fout.write(chunk)
                                    digest.update(chunk)
                                    copied_bytes += len(chunk)
                                # The status set on the copy itself,
                                # after its last byte
                                fout.flush()
                                st = os.fstat(fin.fileno())
                                os.fchmod(
                                    fout.fileno(),
                                    stat.S_IMODE(st.st_mode) & 0o777,
                                )
                                os.utime(
                                    fout.fileno(),
                                    ns=(st.st_atime_ns, st.st_mtime_ns),
                                )
                            copied_files += 1
                            manifest[rel] = digest.hexdigest()
                        except OSError as exc:
                            logger.warning(
                                "Clone skipped (read/write error): %s (%s)",
                                entry.path, exc,
                            )
                            continue
            finally:
                os.close(folder)

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
        'size', and 'modified' keys. Links are left out, never followed.

        Raises:
            ValueError: If session not found or inactive.
        """
        workspace = self._get_active_workspace(session_id)
        files: list[dict[str, Any]] = []

        # Walked through descriptors, each entry read itself: a link is neither followed nor listed (a copy out
        # refuses it), so no size or time of a file outside the workspace is shown
        for rel, st in _walk_beneath(workspace):
            files.append({
                "path": rel,
                "size": st.st_size,
                "modified": st.st_mtime,
            })

        return files

    # -- File preview, approval, and copy-out --

    def preview_file(
        self,
        session_id: str,
        path: str,
        max_bytes: int = 65536,
    ) -> dict[str, Any]:
        """Preview a file's content from the sandbox.

        Returns text content capped at max_bytes for display.
        Binary files return a truncated hex preview. The file is read
        through one descriptor opened beneath the workspace without
        following a link; anything but a regular file with one name is
        refused, saying what it is.

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
        # The checked path is opened once beneath the workspace, through no link and without waiting, as an export
        # reads its source; its size and its bytes come from that descriptor
        root = os.path.realpath(workspace)
        try:
            fd = _open_beneath(root, os.path.relpath(resolved, root))
        except OSError as exc:
            raise ValueError(f"{_why_not_read(exc)}: {path}") from None

        try:
            file_size = os.fstat(fd).st_size
            truncated = file_size > max_bytes
            is_binary = False

            try:
                with open(fd, encoding="utf-8", closefd=False) as fh:
                    content = fh.read(max_bytes)
            except UnicodeDecodeError:
                is_binary = True
                raw = os.pread(fd, min(max_bytes, 2048), 0)
                content = raw.hex()
                truncated = file_size > 2048
        finally:
            os.close(fd)

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
        """Approve specific files for copy-out.

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
        """Reject all files, preventing any copy-out.

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
            # A full reject also withdraws every confirmed deletion --
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
        """Check if a specific file is approved for copy-out."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            return path in session.approved_paths

    # -- Diff-gated write-back support --

    def get_active_workspace_path(self, session_id: str) -> str:
        """Public wrapper for the active-workspace lookup.

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
        """Public wrapper over the share-root confinement.

        The apply writer validates its target through the exact
        ``_resolve_share_path`` discipline: confinement BEFORE existence
        (PermissionError -> 403 whether or not the path exists; inside a
        root, missing/non-directory -> ValueError -> 404).
        """
        return self._resolve_share_path(path)

    # -- The per-workspace network flag and the provision run

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

        Enabling is Daily-only (spec 8.3): the binding-layer
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
        """Run the ONE network-on step: the provision install (8.4).

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
        """Record explicit deletion confirmations for apply-to-host.

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
        """The deletion paths confirmed for apply (a copy)."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return set()
            return set(session.confirmed_deletions)

    @_copying_out
    def copy_out_file(
        self,
        session_id: str,
        path: str,
        dest_dir: str,
    ) -> dict[str, Any]:
        """Copy a single approved file from the sandbox to the host.

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

        # The source checked above, read through no link whatever the sandbox did since (_open_beneath); the export
        # never over a file already there, nor through a link at its name: one of the same name keeps its own, and
        # this one takes a free name. Content, then the source's times and mode, set on the export's own descriptor --
        # the mode without setuid, setgid or sticky, nor write for others than its owner; an export that fails is
        # removed.
        root = os.path.realpath(workspace)
        src_fd = _open_beneath(root, os.path.relpath(resolved_src, root))
        with os.fdopen(src_fd, "rb") as src:
            st = os.fstat(src.fileno())
            dest_file, out = _create_free_export(dest_dir_real, os.path.basename(os.path.normpath(path)))
            try:
                with os.fdopen(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                    dst.flush()
                    os.fchmod(dst.fileno(), stat.S_IMODE(st.st_mode) & 0o755)
                    os.utime(dst.fileno(), ns=(st.st_atime_ns, st.st_mtime_ns))
                    file_size = os.fstat(dst.fileno()).st_size
            except BaseException:
                try:
                    os.unlink(dest_file)
                except OSError:
                    pass
                raise

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

    @_copying_out
    def copy_out_batch(
        self,
        session_id: str,
        paths: list[str],
        dest_dir: str,
    ) -> list[dict[str, Any]]:
        """Copy multiple approved files from the sandbox to the host.

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
        """Get approval state summary for a session.

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
        # Timeout resolution order is explicit call > per-sandbox
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

        # Strict mode -- refuse execution unless bwrap is the backend in use
        if self.execution_blocked:
            if self._bwrap_available:
                reason = (
                    "Sandbox strict_mode is ON but the configured isolation "
                    "backend is 'tempdir', which gives no real isolation. "
                    "Code execution is BLOCKED. Set isolation_backend to "
                    "'auto' or 'bwrap'."
                )
            else:
                reason = (
                    "Sandbox strict_mode is ON but bubblewrap (bwrap) is "
                    "not available. Code execution is BLOCKED for security. "
                    "Install bubblewrap: apt install bubblewrap"
                )
            result = CommandResult(
                blocked=True,
                block_reason=reason,
                isolation_backend="blocked",
            )
            self._audit.log_command(session_id, command, result)
            logger.warning(
                "Execution blocked in session %s: strict_mode, bwrap not in use",
                session_id,
            )
            return result

        # Without bubblewrap the command shares this process's view of /proc:
        # a secret still in the environment the backend was started with is
        # readable there, whatever environment the command is given.
        if backend != IsolationBackend.BWRAP:
            from opti_oignon.process_secrets import unwiped_refusal

            refusal = unwiped_refusal()
            if refusal:
                result = CommandResult(
                    blocked=True,
                    block_reason=f"Code execution without bubblewrap is BLOCKED: {refusal}",
                    isolation_backend="blocked",
                )
                self._audit.log_command(session_id, command, result)
                logger.warning(
                    "Execution blocked in session %s: a secret is still in this process's initial environment",
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

        with self._lock:
            # A command starting to run is activity (a refused one is not):
            # the idle sweep, which checks it again under the lock that ends a
            # session, never takes a session whose command is about to run
            _starting = self._sessions.get(session_id)
            if _starting is not None:
                _starting.last_activity = time.time()

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

        Runs the lazy idle-TTL sweep first, then returns the
        manager view: the stored lifecycle fields plus the derived figures
        (age, running/idle from the process registry, approximate disk use
        from the bounded walk). network_enabled is surfaced and reflects the
        per-workspace flag (default False, flipped only by set_network_enabled).
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
        """Destroy all active sandbox sessions, and remove again the
        workspaces a destroy could not remove whole. Returns count destroyed."""
        with self._lock:
            session_ids = list(self._sessions.keys())

        count = 0
        for sid in session_ids:
            if self.destroy_sandbox(sid):
                count += 1
        self._remove_leftovers()
        return count

    def _remove_leftovers(self) -> int:
        """Remove again the workspaces a destroy could not remove whole; each
        one gone leaves the count and its hold is closed. Returns how many."""
        with self._lock:
            pending = list(self._leftovers)
        removed = 0
        for workspace in pending:
            if os.path.lexists(workspace) and not _remove_tree(workspace):
                continue
            with self._lock:
                if workspace not in self._leftovers:
                    continue  # another sweep removed it first
                hold = self._leftovers.pop(workspace)
            if hold is not None:
                os.close(hold)
            removed += 1
            logger.info("Sandbox workspace %s removed after its destroy", workspace)
        return removed

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

        Replaces the blocking ``subprocess.run`` so the stop
        path has a handle on the running child. The child is launched with
        ``start_new_session=True`` (its pid is the process-group id), the
        ``Popen`` is registered in the per-session running-process registry
        for the duration of the command, and ``communicate(timeout)``
        preserves the blocking semantics and the output capture. On timeout
        the WHOLE group is SIGKILLed and reaped (stronger than ``run()``,
        which killed only the direct child), then ``TimeoutExpired``
        propagates to the caller's existing handler. The registry entry is
        always removed in the ``finally``. A session a destroy ended after its
        caller checked it gets nothing: the process just started is killed
        and ``ValueError`` raised, which the runners answer as an execution
        error.
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
                session = self._sessions.get(session_id)
                live = session is not None and session.active
                if live:
                    self._running_procs.setdefault(session_id, set()).add(proc)
            if not live:
                # A destroy ended the session after its caller checked it:
                # nothing runs on a workspace being removed
                _kill_group(proc)
                raise ValueError(f"Session '{session_id}' was destroyed")
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return subprocess.CompletedProcess(
                argv, proc.returncode, stdout, stderr
            )
        except subprocess.TimeoutExpired:
            # A destroy may have killed and reaped the command meanwhile: its
            # pid is then never signalled again
            _kill_group(proc)
            try:
                proc.communicate(timeout=5)
            except Exception:  # pragma: no cover - defensive reap
                pass
            raise
        finally:
            if session_id:
                with self._lock:
                    running = self._running_procs.get(session_id)
                    if running is not None:
                        running.discard(proc)
                        if not running:
                            del self._running_procs[session_id]

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

        A seccomp denylist filter is built and passed on every launch
        (fail-secure -- a build or fd failure refuses the launch when
        seccomp_required is True, rather than running unfiltered), and
        per-sandbox resource caps are applied via the configured backend
        (rlimit preexec by default; a transient systemd --user cgroup scope
        when resource_backend is "cgroup" and systemd-run is available, with a
        rlimit fallback that never disables the caps).

        The spawn goes through ``_spawn_tracked`` (Popen in its own
        process group, registered per session) so the stop path can SIGKILL
        the running command; the timeout and output-truncation behaviour are
        preserved. ``session_id`` is keyword-only and optional so existing
        positional callers are unchanged.

        ``allow_network`` is keyword-only, defaults to False (the argv
        stays byte-identical for every existing caller), and is forwarded to
        ``_build_bwrap_command`` only by ``execute_provision_command`` -- the
        single, gated provision seam. Seccomp and the resource caps apply to
        the network-on run unchanged.
        """
        from opti_oignon.process_secrets import child_env

        config = self.effective_config
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
                # bwrap and systemd-run themselves: the command inside starts
                # from the cleared environment _build_bwrap_command sets.
                env=child_env(),
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

        When available, wraps the command in `unshare --user --net` --
        with a PID namespace of its own where the system allows one
        (``_build_tempdir_command``) -- to block network access even
        without bwrap. This prevents the LLM from creating a script with
        network code and running it (the main attack vector in tempdir
        mode).

        Spawns through ``_spawn_tracked`` so the stop path works on
        this backend too (same registry, same group kill).
        """
        env = _tempdir_env(workspace)

        # Try to use unshare for network and PID isolation (works without root)
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

        Wraps in the strongest form of `unshare` that works here
        (_UNSHARE_FORMS): a user and network namespace with a PID
        namespace of its own, whose first process (_PID_INIT) runs the
        command and takes down with it every process the command leaves
        -- a background job, one moved out of its group, one that cleared
        its death signal; else the user and network namespace alone
        (network isolation, but a process that outlives its command runs
        on); else plain bash.

        Returns:
            Command list for subprocess.run().
        """
        if self._unshare_prefix is None:
            self._unshare_prefix = self._detect_unshare()
            self._unshare_available = bool(self._unshare_prefix)

        if "--pid" in self._unshare_prefix:
            return [*self._unshare_prefix, "bash", "-c", _PID_INIT, "sandbox-init", command]
        return [*self._unshare_prefix, "bash", "-c", command]

    @staticmethod
    def _detect_unshare() -> list[str]:
        """The strongest unshare form that works on this system
        (_UNSHARE_FORMS), as the prefix of a command; empty if none."""
        from opti_oignon.process_secrets import child_env

        def answers(probe) -> bool:
            try:
                result = probe()
            except Exception:
                return False
            return result.returncode == 0 and "unshare-ok" in result.stdout

        # Each probe's argv is written out, so that the egress census reads
        # the program it runs (echo)
        if answers(lambda: subprocess.run(
            ["unshare", "--user", "--net", "--pid", "--fork", "--kill-child",
             "--", "echo", "unshare-ok"],
            capture_output=True,
            text=True,
            timeout=5,
            env=child_env(),
        )):
            logger.info(
                "Tempdir mode: unshare --user --net --pid available "
                "(network isolation enabled; every process of a "
                "command dies with it)"
            )
            return list(_UNSHARE_FORMS[0])
        if answers(lambda: subprocess.run(
            ["unshare", "--user", "--net", "--", "echo", "unshare-ok"],
            capture_output=True,
            text=True,
            timeout=5,
            env=child_env(),
        )):
            logger.warning(
                "Tempdir mode: unshare --user --net available "
                "(network isolation enabled) but no PID namespace: "
                "a process that outlives its command runs on, out "
                "of reach of a stop or a destroy"
            )
            return list(_UNSHARE_FORMS[1])

        logger.warning(
            "Tempdir mode: unshare not available, "
            "NO network isolation in degraded mode"
        )
        return []

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

def forget_conversation(conversation_id: str) -> int:
    """Forget the words a deleted conversation's sandbox sessions left in this process's audit log, the record of
    what ran kept (``AuditLog.forget_conversation``); how many rows. With the sandbox off, the log its settings name,
    if one is there."""
    audit = sandbox_manager.audit if sandbox_manager is not None else None
    if audit is None:
        config = _load_config()
        audit = AuditLog(os.path.join(config.workspace_base, config.audit_db_path))
    return audit.forget_conversation(conversation_id)


def forget_every_word() -> int:
    """The user's wipe: every row of this process's audit log loses its words, the record of what ran kept; with
    the sandbox off, the log its settings name, if one is there."""
    audit = sandbox_manager.audit if sandbox_manager is not None else None
    if audit is None:
        config = _load_config()
        audit = AuditLog(os.path.join(config.workspace_base, config.audit_db_path))
    return audit.forget_every_word()


def unforget_conversation(conversation_id: str) -> None:
    """Record the conversation's commands whole again: its deletion was refused, and it stays."""
    _FORGOTTEN.discard(conversation_id)


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
