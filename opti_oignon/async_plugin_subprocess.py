#!/usr/bin/env python3
"""
Async plugin subprocess communication (S160).

Provides non-blocking plugin subprocess management using
``asyncio.create_subprocess_exec`` with JSON message passing over
stdin/stdout pipes.  Each plugin runs in its own subprocess; messages
are length-prefixed JSON frames.

Key features:
- Fully async lifecycle: start, communicate, shutdown
- Configurable per-call timeout (default 30 s)
- Graceful shutdown: SIGTERM followed by SIGKILL after grace period
- Output capture on stderr (routed to logging)
- Plugin registry with concurrent-safe access via asyncio.Lock

This module complements (does not replace) the existing
``plugin_subprocess.py`` which uses Unix domain sockets with HMAC.
The pipe-based approach here is lighter weight and better suited to
short-lived plugin invocations that benefit from async I/O.
"""

import asyncio
import json
import logging
import os
import signal
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Hardcoded, never overridable
checkpoint_before_apply = True

# -- Constants ---------------------------------------------------------------

# Length-prefix format: 4-byte big-endian unsigned int
_LENGTH_FMT = "!I"
_LENGTH_SIZE = struct.calcsize(_LENGTH_FMT)

# Maximum message payload (4 MB)
MAX_ASYNC_MESSAGE_SIZE: int = 4 * 1024 * 1024

# Default timeouts
DEFAULT_CALL_TIMEOUT_S: float = 30.0
DEFAULT_STARTUP_TIMEOUT_S: float = 15.0
DEFAULT_SHUTDOWN_GRACE_S: float = 5.0
DEFAULT_SIGKILL_WAIT_S: float = 2.0

FEATURE_AVAILABLE = True


# Environment variables forwarded to a plugin subprocess. Plugin code runs
# untrusted inside the worker, so the host environment (which may carry
# OPTI_ENCRYPTION_KEY, an SQLCipher passphrase, or search API keys) is NOT
# inherited. Only this minimal, secret-free base is forwarded; callers add what
# a plugin needs via env_extra. Mirrors the sandbox --clearenv discipline.
_FORWARDED_ENV_VARS: tuple[str, ...] = (
    "PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TZ",
)


def _build_plugin_env(extra: dict[str, str]) -> dict[str, str]:
    """Build a minimal, secret-free environment for a plugin subprocess.

    The host environment is not inherited; only a small allowlist of benign
    variables is forwarded, plus the explicit ``extra`` mapping.
    """
    env = {k: os.environ[k] for k in _FORWARDED_ENV_VARS if k in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env.update(extra)
    return env


# -- Exceptions --------------------------------------------------------------

class AsyncPluginError(Exception):
    """Base exception for async plugin subprocess operations."""


class AsyncPluginTimeout(AsyncPluginError):
    """Raised when a plugin call exceeds its timeout."""


class AsyncPluginIPCError(AsyncPluginError):
    """Raised when IPC communication fails."""


class AsyncPluginNotRunning(AsyncPluginError):
    """Raised when attempting to call a plugin that is not running."""


# -- Wire protocol -----------------------------------------------------------

def encode_message(payload: dict[str, Any]) -> bytes:
    """Encode a dict as a length-prefixed JSON frame.

    Wire format: [4-byte big-endian length][JSON payload bytes]
    """
    raw = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
    if len(raw) > MAX_ASYNC_MESSAGE_SIZE:
        raise AsyncPluginIPCError(
            f"Message size {len(raw)} exceeds limit {MAX_ASYNC_MESSAGE_SIZE}"
        )
    header = struct.pack(_LENGTH_FMT, len(raw))
    return header + raw


def decode_message(data: bytes) -> dict[str, Any]:
    """Decode a length-prefixed JSON frame.

    Parameters
    ----------
    data : bytes
        Raw bytes: [4-byte length header][JSON payload].

    Returns
    -------
    dict
        Parsed JSON payload.
    """
    if len(data) < _LENGTH_SIZE:
        raise AsyncPluginIPCError("Data too short for length header")
    length = struct.unpack(_LENGTH_FMT, data[:_LENGTH_SIZE])[0]
    raw = data[_LENGTH_SIZE:]
    if len(raw) != length:
        raise AsyncPluginIPCError(
            f"Payload length mismatch: header says {length}, got {len(raw)}"
        )
    try:
        return json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AsyncPluginIPCError(f"JSON decode error: {exc}") from exc


async def async_read_message(reader: asyncio.StreamReader) -> dict[str, Any]:
    """Read a single length-prefixed JSON message from an async reader.

    Parameters
    ----------
    reader : asyncio.StreamReader
        The subprocess stdout stream.

    Returns
    -------
    dict
        Parsed JSON message.

    Raises
    ------
    AsyncPluginIPCError
        On communication failure or malformed message.
    """
    header = await reader.readexactly(_LENGTH_SIZE)
    length = struct.unpack(_LENGTH_FMT, header)[0]
    if length > MAX_ASYNC_MESSAGE_SIZE:
        raise AsyncPluginIPCError(
            f"Message length {length} exceeds limit {MAX_ASYNC_MESSAGE_SIZE}"
        )
    raw = await reader.readexactly(length)
    try:
        return json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AsyncPluginIPCError(f"JSON decode error: {exc}") from exc


async def async_write_message(
    writer: asyncio.StreamWriter,
    payload: dict[str, Any],
) -> None:
    """Write a length-prefixed JSON message to an async writer.

    Parameters
    ----------
    writer : asyncio.StreamWriter
        The subprocess stdin stream.
    payload : dict
        Message dict to serialize and send.
    """
    data = encode_message(payload)
    writer.write(data)
    await writer.drain()


# -- Process handle ----------------------------------------------------------

@dataclass
class AsyncPluginProcess:
    """Tracks a running async plugin subprocess."""

    plugin_name: str
    process: asyncio.subprocess.Process
    worker_script: str
    call_timeout: float = DEFAULT_CALL_TIMEOUT_S
    started_at: float = field(default_factory=lambda: __import__("time").time())
    call_count: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def is_alive(self) -> bool:
        """Check if the subprocess is still running."""
        return self.process.returncode is None

    @property
    def pid(self) -> int | None:
        """Return the PID of the subprocess, or None if exited."""
        return self.process.pid if self.is_alive else None

    def to_dict(self) -> dict[str, Any]:
        """Serialize process info to a dict."""
        import time
        return {
            "plugin_name": self.plugin_name,
            "pid": self.pid,
            "alive": self.is_alive,
            "uptime_s": round(time.time() - self.started_at, 2),
            "call_count": self.call_count,
            "call_timeout": self.call_timeout,
        }


# -- Stderr capture task -----------------------------------------------------

async def _stderr_capture(
    plugin_name: str,
    stderr: asyncio.StreamReader,
) -> None:
    """Read stderr lines from a plugin subprocess and log them."""
    try:
        while True:
            line = await stderr.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                logger.debug("[%s:stderr] %s", plugin_name, text)
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.debug("[%s:stderr] capture stopped: %s", plugin_name, exc)


# -- Manager -----------------------------------------------------------------

class AsyncPluginSubprocessManager:
    """Manages async plugin subprocesses with pipe-based IPC.

    Parameters
    ----------
    worker_script : Path or str or None
        Path to the worker script executed by each plugin subprocess.
        Defaults to ``plugin_worker.py`` in the same directory.
    default_call_timeout : float
        Default timeout for individual plugin calls (seconds).
    startup_timeout : float
        Maximum time to wait for a plugin to send its ready message.
    shutdown_grace : float
        Grace period (seconds) between SIGTERM and SIGKILL during shutdown.
    """

    def __init__(
        self,
        *,
        worker_script: Path | str | None = None,
        default_call_timeout: float = DEFAULT_CALL_TIMEOUT_S,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT_S,
        shutdown_grace: float = DEFAULT_SHUTDOWN_GRACE_S,
    ) -> None:
        if worker_script:
            self._worker_script = str(Path(worker_script).resolve())
        else:
            self._worker_script = str(
                Path(__file__).parent / "plugin_worker.py"
            )
        self._default_call_timeout = default_call_timeout
        self._startup_timeout = startup_timeout
        self._shutdown_grace = shutdown_grace

        self._plugins: dict[str, AsyncPluginProcess] = {}
        self._stderr_tasks: dict[str, asyncio.Task] = {}  # type: ignore[type-arg]
        self._lock = asyncio.Lock()

    # -- Properties ----------------------------------------------------------

    @property
    def running_plugins(self) -> list[str]:
        """List names of currently running plugins."""
        return [n for n, p in self._plugins.items() if p.is_alive]

    @property
    def default_call_timeout(self) -> float:
        """Default per-call timeout in seconds."""
        return self._default_call_timeout

    # -- Lifecycle -----------------------------------------------------------

    async def start_plugin(
        self,
        plugin_name: str,
        plugin_dir: str | Path,
        entry_point: str,
        *,
        call_timeout: float | None = None,
        env_extra: dict[str, str] | None = None,
    ) -> AsyncPluginProcess:
        """Launch a plugin in a new async subprocess.

        Parameters
        ----------
        plugin_name : str
            Unique plugin identifier.
        plugin_dir : str or Path
            Working directory for the subprocess.
        entry_point : str
            Relative path to the plugin entry-point script.
        call_timeout : float, optional
            Per-call timeout override for this plugin.
        env_extra : dict, optional
            Additional environment variables for the subprocess.

        Returns
        -------
        AsyncPluginProcess
            Handle to the running subprocess.

        Raises
        ------
        AsyncPluginError
            If the subprocess fails to start or initialize.
        """
        async with self._lock:
            if plugin_name in self._plugins:
                await self._kill_plugin_unlocked(plugin_name)

        timeout = call_timeout if call_timeout is not None else self._default_call_timeout

        env = _build_plugin_env({
            "OO_PLUGIN_NAME": plugin_name,
            "OO_PLUGIN_DIR": str(Path(plugin_dir).resolve()),
            "OO_PLUGIN_ENTRY": entry_point,
        })
        if env_extra:
            env.update(env_extra)

        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", self._worker_script,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(Path(plugin_dir).resolve()),
                env=env,
            )
        except OSError as exc:
            raise AsyncPluginError(
                f"Failed to start subprocess for plugin '{plugin_name}': {exc}"
            ) from exc

        app = AsyncPluginProcess(
            plugin_name=plugin_name,
            process=proc,
            worker_script=self._worker_script,
            call_timeout=timeout,
        )

        # Start stderr capture
        if proc.stderr:
            task = asyncio.create_task(
                _stderr_capture(plugin_name, proc.stderr),
                name=f"oo-async-{plugin_name}-stderr",
            )
            self._stderr_tasks[plugin_name] = task

        # Wait for ready message
        try:
            ready = await asyncio.wait_for(
                async_read_message(proc.stdout),
                timeout=self._startup_timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise AsyncPluginError(
                f"Plugin '{plugin_name}' did not send ready message "
                f"within {self._startup_timeout}s"
            )
        except (asyncio.IncompleteReadError, AsyncPluginIPCError) as exc:
            proc.kill()
            raise AsyncPluginError(
                f"Plugin '{plugin_name}' startup communication failed: {exc}"
            ) from exc

        if ready.get("status") != "ready":
            proc.kill()
            raise AsyncPluginError(
                f"Plugin '{plugin_name}' sent unexpected startup message: {ready}"
            )

        async with self._lock:
            self._plugins[plugin_name] = app

        logger.info(
            "Started async plugin '%s' (pid=%d, timeout=%.1fs)",
            plugin_name, proc.pid, timeout,
        )
        return app

    async def stop_plugin(
        self,
        plugin_name: str,
        grace: float | None = None,
    ) -> bool:
        """Gracefully stop a plugin subprocess.

        Sends a shutdown message, waits for *grace* seconds, then sends
        SIGTERM.  If the process still has not exited after another
        ``SIGKILL_WAIT_S`` seconds, it is killed with SIGKILL.

        Returns True if a plugin was running and was stopped.
        """
        async with self._lock:
            app = self._plugins.pop(plugin_name, None)
            stderr_task = self._stderr_tasks.pop(plugin_name, None)

        if app is None:
            return False

        grace = grace if grace is not None else self._shutdown_grace
        await self._shutdown_process(app, grace)

        if stderr_task and not stderr_task.done():
            stderr_task.cancel()
            try:
                await stderr_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped async plugin '%s'", plugin_name)
        return True

    async def stop_all(self, grace: float | None = None) -> int:
        """Stop all running plugin subprocesses. Returns count stopped."""
        async with self._lock:
            names = list(self._plugins.keys())
        count = 0
        for name in names:
            if await self.stop_plugin(name, grace=grace):
                count += 1
        return count

    async def is_running(self, plugin_name: str) -> bool:
        """Check if a plugin subprocess is alive."""
        async with self._lock:
            app = self._plugins.get(plugin_name)
        return app is not None and app.is_alive

    async def get_status(self, plugin_name: str) -> dict[str, Any] | None:
        """Return status dict for a plugin, or None if not found."""
        async with self._lock:
            app = self._plugins.get(plugin_name)
        if app is None:
            return None
        return app.to_dict()

    async def list_plugins(self) -> list[dict[str, Any]]:
        """Return status dicts for all tracked plugins."""
        async with self._lock:
            plugins = list(self._plugins.values())
        return [p.to_dict() for p in plugins]

    # -- RPC -----------------------------------------------------------------

    async def call_plugin(
        self,
        plugin_name: str,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send an RPC-style call to a plugin and await its response.

        Parameters
        ----------
        plugin_name : str
            Target plugin.
        method : str
            Method name to invoke on the plugin side.
        params : dict, optional
            Parameters for the method call.
        timeout : float, optional
            Timeout override (default: plugin's ``call_timeout``).

        Returns
        -------
        dict
            Result payload from the plugin.

        Raises
        ------
        AsyncPluginNotRunning
            If the plugin is not currently running.
        AsyncPluginTimeout
            If the call exceeds the timeout.
        AsyncPluginIPCError
            If communication fails.
        """
        async with self._lock:
            app = self._plugins.get(plugin_name)

        if app is None or not app.is_alive:
            raise AsyncPluginNotRunning(
                f"Plugin '{plugin_name}' is not running"
            )

        t = timeout if timeout is not None else app.call_timeout

        request = {
            "method": method,
            "params": params or {},
        }

        async with app._lock:
            try:
                await asyncio.wait_for(
                    async_write_message(app.process.stdin, request),
                    timeout=t,
                )
            except asyncio.TimeoutError:
                raise AsyncPluginTimeout(
                    f"Timeout writing to plugin '{plugin_name}'"
                )
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                raise AsyncPluginIPCError(
                    f"Write failed for plugin '{plugin_name}': {exc}"
                ) from exc

            try:
                response = await asyncio.wait_for(
                    async_read_message(app.process.stdout),
                    timeout=t,
                )
            except asyncio.TimeoutError:
                raise AsyncPluginTimeout(
                    f"Timeout reading from plugin '{plugin_name}' "
                    f"after {t}s"
                )
            except asyncio.IncompleteReadError as exc:
                raise AsyncPluginIPCError(
                    f"Plugin '{plugin_name}' closed connection: {exc}"
                ) from exc

        app.call_count += 1

        if "error" in response:
            err = response["error"]
            raise AsyncPluginError(
                f"Plugin '{plugin_name}' error: {err}"
            )

        return response.get("result", {})

    async def ping(self, plugin_name: str, timeout: float = 3.0) -> bool:
        """Ping a plugin to check health. Returns True if responsive."""
        try:
            result = await self.call_plugin(
                plugin_name, "ping", {}, timeout=timeout,
            )
            return result.get("status") == "pong"
        except Exception:
            return False

    # -- Internal ------------------------------------------------------------

    async def _shutdown_process(
        self,
        app: AsyncPluginProcess,
        grace: float,
    ) -> None:
        """Gracefully shut down a plugin subprocess."""
        proc = app.process
        if not app.is_alive:
            return

        # Step 1: send shutdown message
        try:
            await asyncio.wait_for(
                async_write_message(proc.stdin, {"method": "shutdown", "params": {}}),
                timeout=min(grace, 2.0),
            )
        except Exception:
            pass

        # Step 2: wait for voluntary exit
        try:
            await asyncio.wait_for(proc.wait(), timeout=grace)
            return
        except asyncio.TimeoutError:
            pass

        # Step 3: SIGTERM
        if app.is_alive:
            try:
                proc.send_signal(signal.SIGTERM)
            except (ProcessLookupError, OSError):
                return
            try:
                await asyncio.wait_for(proc.wait(), timeout=DEFAULT_SIGKILL_WAIT_S)
                return
            except asyncio.TimeoutError:
                pass

        # Step 4: SIGKILL
        if app.is_alive:
            logger.warning(
                "Plugin '%s' did not exit after SIGTERM, sending SIGKILL",
                app.plugin_name,
            )
            try:
                proc.kill()
                await asyncio.wait_for(proc.wait(), timeout=DEFAULT_SIGKILL_WAIT_S)
            except Exception:
                pass

    async def _kill_plugin_unlocked(self, plugin_name: str) -> None:
        """Kill a plugin subprocess (caller must hold self._lock)."""
        app = self._plugins.pop(plugin_name, None)
        if app is None:
            return
        stderr_task = self._stderr_tasks.pop(plugin_name, None)
        if stderr_task and not stderr_task.done():
            stderr_task.cancel()
        if app.is_alive:
            try:
                app.process.kill()
                await asyncio.wait_for(
                    app.process.wait(), timeout=DEFAULT_SIGKILL_WAIT_S,
                )
            except Exception:
                pass


# -- Module-level singleton --------------------------------------------------

ASYNC_PLUGIN_SUBPROCESS_AVAILABLE = True

_default_manager: AsyncPluginSubprocessManager | None = None


def get_async_plugin_manager(
    **kwargs: Any,
) -> AsyncPluginSubprocessManager:
    """Return the module-level singleton (created on first call)."""
    global _default_manager
    if _default_manager is None:
        _default_manager = AsyncPluginSubprocessManager(**kwargs)
    return _default_manager


def reset_async_plugin_manager() -> None:
    """Reset the singleton (for test isolation)."""
    global _default_manager
    _default_manager = None
