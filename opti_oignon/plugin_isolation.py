#!/usr/bin/env python3
"""Where a plugin's worker runs, and inside which walls.

A plugin is third-party code. Its worker starts under bubblewrap whenever
the sandbox runs its own commands there: the same namespaces, the same
seccomp filter, and a filesystem holding only what the worker needs -- the
interpreter, the worker file and the plugin's folder read-only and, for a
plugin whose manifest may write, its private data folder. A protected
folder (the server's data and configuration) that lies inside one of those
trees is masked. The network stays outside unless the manifest declares
``network_outbound``. Host and worker talk over a socket pair the worker
inherits: no socket file, and no writable folder for one.

The posture follows the sandbox:

  * ``bwrap`` -- the sandbox runs its commands under bubblewrap;
  * ``blocked`` -- it does not, and strict mode is on: no plugin starts,
    and the reason names the cause and the remedy;
  * ``direct`` -- it does not, and strict mode is off: the worker starts as
    a plain process, and the reason says it runs without isolation.

A sandbox whose state cannot be read gives ``blocked``. The limits are the
server's: a manifest asks, the ceilings in config/plugins.yaml cap, and the
launcher installs them on the process before it executes, never above the
hard limits the server itself has.

Importing this module has no side effect. The sandbox manager, whose import
builds its singleton, is reached only when a posture is resolved.
"""

from __future__ import annotations

import importlib
import logging
import os
import resource
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Callable, Iterable

logger = logging.getLogger(__name__)

checkpoint_before_apply = True

MODE_BWRAP = "bwrap"
MODE_DIRECT = "direct"
MODE_BLOCKED = "blocked"

# Manifest permissions that grant the private data folder, and the network.
WRITE_PERMISSIONS = frozenset({"filesystem_plugin_dir", "filesystem_write"})
NETWORK_PERMISSION = "network_outbound"

_REMEDY = (
    "plugins run isolated only under the sandbox's bubblewrap: keep the sandbox "
    "enabled, install bubblewrap (apt install bubblewrap) and keep "
    "isolation_backend on auto or bwrap; or set strict_mode: false to run "
    "plugins without isolation"
)

# No bind may cover these, nor the user's own home.
_TOO_WIDE = ("/", "/home", "/root")

_DEFAULT_TMPFS_BYTES = 256 * 1024 ** 2
# The empty tmpfs laid over a protected folder: room for the mount points
# bubblewrap creates there, not a place a plugin could fill memory with.
_MASK_TMPFS_BYTES = 64 * 1024

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "plugins.yaml"


class PluginIsolationError(Exception):
    """A plugin's worker cannot be started inside the walls it needs."""


@dataclass(frozen=True)
class PluginPosture:
    """Where plugin workers run, and the walls they get there."""

    mode: str
    strict: bool
    reason: str = ""
    ro_binds: tuple[str, ...] = ()
    never_bind: frozenset[str] = frozenset()
    tmpfs_size_bytes: int = _DEFAULT_TMPFS_BYTES
    seccomp_enabled: bool = True
    seccomp_required: bool = True
    protected_dirs: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Posture
# ---------------------------------------------------------------------------

def resolve_posture(sandbox_module: Any = None) -> PluginPosture:
    """Read the plugin posture off the sandbox; an unreadable one is blocked."""
    try:
        if sandbox_module is None:
            sandbox_module = importlib.import_module("opti_oignon.sandbox_manager")
        never_bind = frozenset(getattr(sandbox_module, "_HARDCODED_NEVER_BIND", ()))
        manager = getattr(sandbox_module, "sandbox_manager", None)
        if manager is not None:
            return _from_manager(manager, never_bind)
        return _without_manager(getattr(sandbox_module, "_config", None), never_bind)
    except Exception as exc:  # noqa: BLE001 - an unreadable sandbox fails closed
        return PluginPosture(
            mode=MODE_BLOCKED,
            strict=True,
            reason=f"plugins are not started: the sandbox's state cannot be read ({exc}); {_REMEDY}",
        )


def _from_manager(manager: Any, never_bind: frozenset[str]) -> PluginPosture:
    config = manager.config
    walls = _walls(config, never_bind)
    strict = bool(manager.strict_mode)
    if manager.bwrap_in_use:
        return PluginPosture(
            mode=MODE_BWRAP, strict=strict, reason="plugin workers run under bubblewrap", **walls,
        )
    if str(getattr(config, "isolation_backend", "")).lower() == "tempdir":
        cause = "the sandbox is configured with isolation_backend: tempdir, which runs nothing under bubblewrap"
    else:
        detail = manager.get_isolation_status().get("bwrap_info") or "no detail"
        cause = f"bubblewrap is not available ({detail})"
    return _lacking(cause, strict, walls)


def _without_manager(config: Any, never_bind: frozenset[str]) -> PluginPosture:
    if config is None:
        return PluginPosture(
            mode=MODE_BLOCKED,
            strict=True,
            reason=f"plugins are not started: the sandbox manager is unavailable; {_REMEDY}",
        )
    if getattr(config, "enabled", True) is False:
        cause = "the sandbox is switched off (enabled: false in config/sandbox.yaml)"
    else:
        cause = "the sandbox manager did not start"
    return _lacking(cause, bool(getattr(config, "strict_mode", True)), _walls(config, never_bind))


def _lacking(cause: str, strict: bool, walls: dict[str, Any]) -> PluginPosture:
    if strict:
        return PluginPosture(
            mode=MODE_BLOCKED, strict=True,
            reason=f"plugins are not started: {cause}; {_REMEDY}", **walls,
        )
    return PluginPosture(
        mode=MODE_DIRECT, strict=False,
        reason=f"plugins run WITHOUT isolation: {cause}, and strict_mode is off", **walls,
    )


def _walls(config: Any, never_bind: frozenset[str]) -> dict[str, Any]:
    return {
        "ro_binds": tuple(getattr(config, "bwrap_ro_binds", None) or ()),
        "never_bind": frozenset(never_bind) | frozenset(getattr(config, "bwrap_never_bind", None) or ()),
        "tmpfs_size_bytes": int(getattr(config, "tmpfs_size_bytes", _DEFAULT_TMPFS_BYTES)),
        "seccomp_enabled": bool(getattr(config, "seccomp_enabled", True)),
        "seccomp_required": bool(getattr(config, "seccomp_required", True)),
        "protected_dirs": _protected_dirs(),
    }


def _protected_dirs() -> tuple[str, ...]:
    """The server's own data and configuration folders, wherever they live."""
    # The package exposes a ``config`` object that shadows the module.
    oo_config = importlib.import_module("opti_oignon.config")

    found: list[str] = []
    for path in (
        oo_config.DATA_DIR,
        oo_config.CONFIG_DIR,
        Path(oo_config.PROJECT_ROOT).parent / "data",
    ):
        real = os.path.realpath(path)
        if os.path.isdir(real) and real not in found:
            found.append(real)
    return tuple(found)


def posture_status(posture: PluginPosture) -> dict[str, Any]:
    """The posture as /api/health reports it."""
    return {"mode": posture.mode, "strict": posture.strict, "reason": posture.reason}


def isolation_status() -> dict[str, Any]:
    """The current plugin isolation, for /api/health."""
    return posture_status(resolve_posture())


# ---------------------------------------------------------------------------
# Walls
# ---------------------------------------------------------------------------

def _inside(path: str, tree: str) -> bool:
    """Whether ``path`` is ``tree`` or lies under it."""
    tree = tree.rstrip("/") or "/"
    return path == tree or tree == "/" or path.startswith(tree + "/")


def _refuse_wide(path: str) -> None:
    home = os.path.realpath(os.path.expanduser("~"))
    for wide in _TOO_WIDE + (home,):
        if _inside(wide, path):
            raise PluginIsolationError(
                f"refusing to bind {path} into a plugin's sandbox: it would expose {wide}"
            )


def _system_binds(posture: PluginPosture) -> list[str]:
    binds: list[str] = []
    for path in posture.ro_binds:
        if any(path == blocked or path.startswith(blocked + "/") for blocked in posture.never_bind):
            logger.warning("Refusing to bind blocked path into a plugin's sandbox: %s", path)
            continue
        if os.path.exists(path):
            binds.append(path)
    return binds


def _interpreter_roots(interpreter: str, system: list[str]) -> list[str]:
    """The installation trees the interpreter needs, beyond the system binds."""
    roots: list[str] = []
    for raw in (sys.prefix, sys.base_prefix, os.path.dirname(os.path.dirname(os.path.realpath(interpreter)))):
        real = os.path.realpath(raw)
        if not os.path.isdir(real):
            continue
        if any(_inside(real, bound) for bound in system + roots):
            continue
        roots = [root for root in roots if not _inside(root, real)] + [real]
    return roots


def build_bwrap_argv(
    posture: PluginPosture,
    *,
    interpreter: str,
    worker_script: str,
    plugin_dir: str,
    data_dir: str | None,
    allow_network: bool,
    seccomp_fd: int | None = None,
) -> list[str]:
    """The bubblewrap command line that runs a plugin's worker.

    Every path is bound at its own place, so nothing needs translating
    between the host and the sandbox. Raises PluginIsolationError when a
    bind would expose the root, a home, or a protected folder itself.
    """
    worker = os.path.realpath(worker_script)
    plugin = os.path.realpath(plugin_dir)
    data = os.path.realpath(data_dir) if data_dir else None
    own = [worker, plugin] + ([data] if data else [])
    for path in own:
        _refuse_wide(path)
        if path in posture.protected_dirs:
            raise PluginIsolationError(f"refusing to bind protected folder {path} into a plugin's sandbox")
        if not os.path.exists(path):
            raise PluginIsolationError(f"{path} does not exist")
    system = _system_binds(posture)
    roots = _interpreter_roots(interpreter, system)
    for root in roots:
        _refuse_wide(root)

    argv = ["bwrap"]
    for path in system:
        argv += ["--ro-bind", path, path]
    # /tmp is a fresh, capped tmpfs; every bind that may lie under it follows.
    argv += [
        "--dev", "/dev",
        "--proc", "/proc",
        "--size", str(int(posture.tmpfs_size_bytes)), "--tmpfs", "/tmp",
    ]
    for root in roots:
        argv += ["--ro-bind", root, root]
    argv += ["--ro-bind", worker, worker]
    argv += ["--ro-bind", plugin, plugin]
    trees = roots + [plugin]
    for protected in posture.protected_dirs:
        if any(_inside(protected, tree) for tree in trees):
            argv += ["--size", str(_MASK_TMPFS_BYTES), "--tmpfs", protected]
    if data:
        argv += ["--bind", data, data]
    if not allow_network:
        argv.append("--unshare-net")
    argv += [
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup",
        "--new-session",
        "--die-with-parent",
    ]
    if allow_network:
        # resolv.conf is bound through its realpath: on systemd-resolved hosts
        # it is a link into /run, which stays out of every sandbox.
        for ns_file in ("/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf"):
            real = os.path.realpath(ns_file)
            if os.path.isfile(real):
                argv += ["--ro-bind", real, ns_file]
    if seccomp_fd is not None:
        argv += ["--seccomp", str(int(seccomp_fd))]
    argv += ["--chdir", plugin, interpreter, worker]
    return argv


def stage_seccomp(posture: PluginPosture, plugin_name: str) -> IO[bytes] | None:
    """The sandbox's seccomp filter on a file the launch can pass on.

    A filter that cannot be built refuses the launch when it is required;
    otherwise the worker starts without one, said loudly.
    """
    if not posture.seccomp_enabled:
        return None
    try:
        from opti_oignon import sandbox_seccomp

        blob = sandbox_seccomp.build_filter_program()
    except Exception as exc:  # noqa: BLE001 - every failure reads as no filter
        if posture.seccomp_required:
            raise PluginIsolationError(
                f"Plugin '{plugin_name}' not started: the seccomp filter cannot be built "
                f"({exc}) and seccomp_required is on"
            ) from exc
        logger.warning(
            "SECURITY: plugin '%s' starts under bubblewrap WITHOUT a seccomp filter (%s); "
            "seccomp_required is off", plugin_name, exc,
        )
        return None
    staged = tempfile.TemporaryFile()
    staged.write(blob)
    staged.flush()
    staged.seek(0)
    return staged


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProcessLimits:
    """Resource limits of one plugin process (or the ceilings over them)."""

    cpu_time_seconds: int = 30
    memory_bytes: int = 256 * 1024 ** 2
    max_file_descriptors: int = 64
    max_processes: int = 4096
    max_file_size_bytes: int = 256 * 1024 ** 2


# Configured ceilings are clamped into these ranges, never switched off.
_CEILING_BOUNDS = {
    "cpu_time_seconds": (1, 86400),
    "memory_bytes": (64 * 1024 ** 2, 16 * 1024 ** 3),
    "max_file_descriptors": (16, 65536),
    "max_processes": (16, 8192),
    "max_file_size_bytes": (1024 ** 2, 16 * 1024 ** 3),
}

# What a manifest may ask for; the processes and the file size are the server's.
_ASKABLE = ("cpu_time_seconds", "memory_bytes", "max_file_descriptors")

# Limit -> (resource, environment variable announcing it to the worker).
_RLIMITS = (
    ("cpu_time_seconds", resource.RLIMIT_CPU, "OO_RLIMIT_CPU"),
    ("memory_bytes", resource.RLIMIT_AS, "OO_RLIMIT_MEM"),
    ("max_file_descriptors", resource.RLIMIT_NOFILE, "OO_RLIMIT_NOFILE"),
    ("max_processes", resource.RLIMIT_NPROC, "OO_RLIMIT_NPROC"),
    ("max_file_size_bytes", resource.RLIMIT_FSIZE, "OO_RLIMIT_FSIZE"),
)


def load_ceilings(path: Path | None = None) -> ProcessLimits:
    """The ceilings of config/plugins.yaml (``subprocess.limit_ceilings``), clamped."""
    defaults = ProcessLimits()
    try:
        import yaml

        raw = yaml.safe_load(Path(path or _DEFAULT_CONFIG_PATH).read_text(encoding="utf-8")) or {}
        section = (raw.get("subprocess") or {}).get("limit_ceilings") or {}
    except Exception as exc:  # noqa: BLE001 - unreadable ceilings fall back to the shipped ones
        logger.warning("Plugin limit ceilings unreadable (%s): using the shipped ones", exc)
        return defaults
    values: dict[str, int] = {}
    for name, (low, high) in _CEILING_BOUNDS.items():
        value = section.get(name, getattr(defaults, name))
        if isinstance(value, bool) or not isinstance(value, int):
            logger.warning("Plugin limit ceiling %s is not an integer: using %d", name, getattr(defaults, name))
            value = getattr(defaults, name)
        values[name] = min(max(value, low), high)
    return ProcessLimits(**values)


def effective_limits(requested: Any, ceilings: ProcessLimits) -> ProcessLimits:
    """The lower of what the manifest asks and the ceiling, limit by limit."""
    values: dict[str, int] = {}
    for name, _rid, _variable in _RLIMITS:
        ceiling = getattr(ceilings, name)
        asked = getattr(requested, name, None) if name in _ASKABLE else None
        if isinstance(asked, bool) or not isinstance(asked, (int, float)) or asked < 0:
            values[name] = ceiling
        else:
            values[name] = min(int(asked), ceiling)
    return ProcessLimits(**values)


def limits_env(limits: ProcessLimits) -> dict[str, str]:
    """The limits as the worker reads them back, to check they are in place."""
    return {variable: str(getattr(limits, name)) for name, _rid, variable in _RLIMITS}


def limits_preexec(limits: ProcessLimits) -> Callable[[], None]:
    """A preexec hook that installs the limits on the launched process.

    It runs in the forked child before exec, so bubblewrap and the worker
    inherit the limits. A limit never goes above the hard limit the server
    already has; any failure aborts the launch. Like the sandbox's hook, it
    only calls getrlimit and setrlimit, safe after fork in a threaded server.
    RLIMIT_NPROC counts the processes of the real user, so it is a coarse
    guard; a cgroup would account a plugin's processes exactly.
    """
    pairs = [(rid, int(getattr(limits, name))) for name, rid, _variable in _RLIMITS]

    def _apply() -> None:
        for rid, value in pairs:
            hard = resource.getrlimit(rid)[1]
            target = value if hard == resource.RLIM_INFINITY else min(value, hard)
            resource.setrlimit(rid, (target, target))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return _apply


# ---------------------------------------------------------------------------
# Permissions and the data folder
# ---------------------------------------------------------------------------

def grants_data_dir(permissions: Iterable[str]) -> bool:
    """Whether the manifest's permissions grant a private data folder."""
    return bool(WRITE_PERMISSIONS & set(permissions))


def grants_network(permissions: Iterable[str]) -> bool:
    """Whether the manifest's permissions keep the network."""
    return NETWORK_PERMISSION in set(permissions)


def default_data_root() -> Path:
    """Where plugins' private data folders live: under the server's data folder."""
    from opti_oignon.config import DATA_DIR

    return Path(DATA_DIR) / "plugin_data"


def prepare_data_dir(root: Path | str, plugin_name: str) -> Path:
    """Create the plugin's private data folder (0700), refusing a link."""
    path = Path(root) / plugin_name
    if path.is_symlink():
        raise PluginIsolationError(
            f"Plugin '{plugin_name}' not started: its data folder {path} is a link"
        )
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        raise PluginIsolationError(
            f"Plugin '{plugin_name}' not started: its data folder {path} cannot be made ({exc})"
        ) from exc
    os.chmod(path, 0o700)
    return path
