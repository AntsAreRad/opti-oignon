#!/usr/bin/env python3
"""
Plugin out-of-process isolation for Opti-Oignon (S143).

Runs each plugin in its own subprocess communicating via JSON-RPC over
Unix domain sockets.  HMAC-signed messages, resource limits, watchdog
timers, and stdout/stderr capture ensure that a misbehaving plugin
cannot compromise the host process.
"""

import hashlib
import hmac
import json
import logging
import os
import secrets
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum message size (4 MB)
MAX_MESSAGE_SIZE: int = 4 * 1024 * 1024

# Header: 4 bytes length + 32 bytes HMAC-SHA256
HEADER_SIZE: int = 4 + 32

# Default resource limits (can be overridden per-plugin in manifest)
DEFAULT_RESOURCE_LIMITS: dict[str, int] = {
    "cpu_time_seconds": 30,       # RLIMIT_CPU
    "memory_bytes": 256 * 1024 * 1024,  # RLIMIT_AS (256 MB)
    "max_file_descriptors": 64,   # RLIMIT_NOFILE
}

# Watchdog defaults
DEFAULT_WATCHDOG_INTERVAL_S: float = 5.0
DEFAULT_HOOK_TIMEOUT_S: float = 10.0
DEFAULT_STARTUP_TIMEOUT_S: float = 15.0

# Log rotation defaults
DEFAULT_LOG_MAX_BYTES: int = 5 * 1024 * 1024  # 5 MB
DEFAULT_LOG_BACKUP_COUNT: int = 3

# JSON-RPC error codes
JSONRPC_PARSE_ERROR = -32700
JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INTERNAL_ERROR = -32603

# Environment variables forwarded to a plugin subprocess. Plugin code runs
# untrusted (possibly third-party) inside the worker, so the host environment --
# which may carry OPTI_ENCRYPTION_KEY, an SQLCipher passphrase, or search API
# keys -- is NOT inherited. Only this minimal, secret-free base is forwarded;
# callers add anything a plugin genuinely needs via env_extra. This mirrors the
# sandbox --clearenv discipline (S-01/C-01) on the plugin execution path.
_FORWARDED_ENV_VARS: tuple[str, ...] = (
    "PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TZ",
)


def _build_plugin_env(extra: dict[str, str]) -> dict[str, str]:
    """Build a minimal, secret-free environment for a plugin subprocess.

    The host environment is not inherited; only a small allowlist of benign
    variables (``_FORWARDED_ENV_VARS``) is forwarded, plus the explicit
    ``extra`` mapping. PATH is backfilled to a safe default if absent.
    """
    env = {k: os.environ[k] for k in _FORWARDED_ENV_VARS if k in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env.update(extra)
    return env


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PluginSubprocessError(Exception):
    """Raised when a plugin subprocess operation fails."""


class PluginSubprocessTimeout(PluginSubprocessError):
    """Raised when a plugin subprocess times out."""


class PluginIPCError(PluginSubprocessError):
    """Raised when IPC communication fails."""


class PluginHMACError(PluginIPCError):
    """Raised when HMAC verification fails."""


# ---------------------------------------------------------------------------
# HMAC-signed message protocol
# ---------------------------------------------------------------------------

def _compute_hmac(key: bytes, data: bytes) -> bytes:
    """Compute HMAC-SHA256 for a message payload."""
    return hmac.new(key, data, hashlib.sha256).digest()


def _verify_hmac(key: bytes, data: bytes, expected: bytes) -> bool:
    """Verify HMAC-SHA256 signature (constant-time comparison)."""
    computed = hmac.new(key, data, hashlib.sha256).digest()
    return hmac.compare_digest(computed, expected)


def pack_message(key: bytes, payload: dict[str, Any]) -> bytes:
    """Serialize a dict to JSON, sign with HMAC, and pack with length header.

    Wire format: [4-byte big-endian length][32-byte HMAC-SHA256][JSON payload]
    """
    raw = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
    if len(raw) > MAX_MESSAGE_SIZE:
        raise PluginIPCError(
            f"Message size {len(raw)} exceeds limit {MAX_MESSAGE_SIZE}"
        )
    mac = _compute_hmac(key, raw)
    header = struct.pack("!I", len(raw)) + mac
    return header + raw


def unpack_message(key: bytes, data: bytes) -> dict[str, Any]:
    """Unpack and verify a wire-format message.

    Parameters
    ----------
    key : bytes
        HMAC secret key.
    data : bytes
        Raw bytes containing header + payload.

    Returns
    -------
    dict
        Parsed JSON payload.

    Raises
    ------
    PluginHMACError
        If HMAC verification fails.
    PluginIPCError
        If the message is malformed.
    """
    if len(data) < HEADER_SIZE:
        raise PluginIPCError("Message too short for header")
    length = struct.unpack("!I", data[:4])[0]
    mac = data[4:36]
    raw = data[36:]
    if len(raw) != length:
        raise PluginIPCError(
            f"Payload length mismatch: header says {length}, got {len(raw)}"
        )
    if not _verify_hmac(key, raw, mac):
        raise PluginHMACError("HMAC verification failed — message tampered or wrong key")
    try:
        return json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PluginIPCError(f"JSON decode error: {exc}") from exc


def recv_message(sock: socket.socket, key: bytes, timeout: float = 10.0) -> dict[str, Any]:
    """Read a single framed message from a Unix socket.

    Parameters
    ----------
    sock : socket.socket
        Connected Unix domain socket.
    key : bytes
        HMAC secret key.
    timeout : float
        Read timeout in seconds.

    Returns
    -------
    dict
        Parsed JSON-RPC message.
    """
    sock.settimeout(timeout)
    try:
        # Read length + HMAC header
        header = _recv_exact(sock, HEADER_SIZE)
        length = struct.unpack("!I", header[:4])[0]
        if length > MAX_MESSAGE_SIZE:
            raise PluginIPCError(
                f"Message length {length} exceeds limit {MAX_MESSAGE_SIZE}"
            )
        mac = header[4:36]
        # Read payload
        raw = _recv_exact(sock, length)
        if not _verify_hmac(key, raw, mac):
            raise PluginHMACError("HMAC verification failed")
        return json.loads(raw.decode("utf-8"))
    except socket.timeout:
        raise PluginSubprocessTimeout("Timed out reading from plugin socket")
    except (ConnectionResetError, BrokenPipeError) as exc:
        raise PluginIPCError(f"Connection lost: {exc}") from exc


def send_message(sock: socket.socket, key: bytes, payload: dict[str, Any]) -> None:
    """Send a single framed message over a Unix socket."""
    data = pack_message(key, payload)
    try:
        sock.sendall(data)
    except (ConnectionResetError, BrokenPipeError, OSError) as exc:
        raise PluginIPCError(f"Failed to send message: {exc}") from exc


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly *n* bytes from a socket."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise PluginIPCError("Connection closed while reading")
        buf.extend(chunk)
    return bytes(buf)


# ---------------------------------------------------------------------------
# Resource limit helpers
# ---------------------------------------------------------------------------

@dataclass
class PluginResourceLimits:
    """Configurable resource limits for a plugin subprocess."""

    cpu_time_seconds: int = DEFAULT_RESOURCE_LIMITS["cpu_time_seconds"]
    memory_bytes: int = DEFAULT_RESOURCE_LIMITS["memory_bytes"]
    max_file_descriptors: int = DEFAULT_RESOURCE_LIMITS["max_file_descriptors"]

    @classmethod
    def from_manifest(cls, manifest_data: dict[str, Any]) -> "PluginResourceLimits":
        """Create resource limits from plugin manifest 'resource_limits' section."""
        rl = manifest_data.get("resource_limits", {})
        return cls(
            cpu_time_seconds=int(rl.get(
                "cpu_time_seconds",
                DEFAULT_RESOURCE_LIMITS["cpu_time_seconds"],
            )),
            memory_bytes=int(rl.get(
                "memory_bytes",
                DEFAULT_RESOURCE_LIMITS["memory_bytes"],
            )),
            max_file_descriptors=int(rl.get(
                "max_file_descriptors",
                DEFAULT_RESOURCE_LIMITS["max_file_descriptors"],
            )),
        )

    def to_dict(self) -> dict[str, int]:
        """Serialize to dict for transmission to worker."""
        return {
            "cpu_time_seconds": self.cpu_time_seconds,
            "memory_bytes": self.memory_bytes,
            "max_file_descriptors": self.max_file_descriptors,
        }


# ---------------------------------------------------------------------------
# Plugin log capture
# ---------------------------------------------------------------------------

def setup_plugin_logger(
    plugin_name: str,
    log_dir: Path,
    max_bytes: int = DEFAULT_LOG_MAX_BYTES,
    backup_count: int = DEFAULT_LOG_BACKUP_COUNT,
) -> logging.Logger:
    """Create a rotating file logger for a plugin's stdout/stderr.

    Returns a Logger instance writing to ``log_dir/<plugin_name>.log``
    with automatic rotation.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{plugin_name}.log"
    plugin_logger = logging.getLogger(f"opti.plugin.{plugin_name}")
    plugin_logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on reload
    if not plugin_logger.handlers:
        handler = RotatingFileHandler(
            str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        plugin_logger.addHandler(handler)

    return plugin_logger


# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------

def make_rpc_request(
    method: str,
    params: Optional[dict[str, Any]] = None,
    request_id: Optional[str] = None,
) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 request dict."""
    return {
        "jsonrpc": "2.0",
        "method": method,
        "params": params or {},
        "id": request_id or secrets.token_hex(8),
    }


def make_rpc_response(
    request_id: str,
    result: Any = None,
    error: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 response dict."""
    resp: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        resp["error"] = error
    else:
        resp["result"] = result
    return resp


def make_rpc_error(
    request_id: Optional[str],
    code: int,
    message: str,
    data: Any = None,
) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 error response."""
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Subprocess handle
# ---------------------------------------------------------------------------

@dataclass
class PluginProcess:
    """Tracks a running plugin subprocess."""

    plugin_name: str
    process: subprocess.Popen  # type: ignore[type-arg]
    socket_path: str
    hmac_key: bytes
    conn: Optional[socket.socket] = None
    plugin_logger: Optional[logging.Logger] = None
    started_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def is_alive(self) -> bool:
        """Check if the subprocess is still running."""
        return self.process.poll() is None

    def elapsed_since_heartbeat(self) -> float:
        """Seconds since last successful communication."""
        return time.time() - self.last_heartbeat

    def touch_heartbeat(self) -> None:
        """Update the heartbeat timestamp."""
        self.last_heartbeat = time.time()


# ---------------------------------------------------------------------------
# Output capture threads
# ---------------------------------------------------------------------------

def _stream_reader(
    stream: Any,
    plugin_logger: logging.Logger,
    stream_name: str,
) -> None:
    """Read lines from a subprocess stream and log them."""
    try:
        for line in iter(stream.readline, b""):
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                plugin_logger.info("[%s] %s", stream_name, text)
    except (ValueError, OSError):
        pass  # Stream closed


# ---------------------------------------------------------------------------
# PluginSubprocessManager
# ---------------------------------------------------------------------------

class PluginSubprocessManager:
    """Manages plugin subprocesses with IPC, resource limits, and watchdog.

    Parameters
    ----------
    socket_dir : Path or str or None
        Directory for Unix domain sockets (default: temp dir).
    log_dir : Path or str or None
        Directory for plugin log files (default: data/plugin_logs).
    worker_script : Path or str or None
        Path to the worker script (default: auto-detect).
    watchdog_interval : float
        Seconds between watchdog checks.
    default_hook_timeout : float
        Default timeout for hook RPC calls in seconds.
    startup_timeout : float
        Timeout waiting for subprocess to become ready.
    """

    def __init__(
        self,
        *,
        socket_dir: Optional[Path | str] = None,
        log_dir: Optional[Path | str] = None,
        worker_script: Optional[Path | str] = None,
        watchdog_interval: float = DEFAULT_WATCHDOG_INTERVAL_S,
        default_hook_timeout: float = DEFAULT_HOOK_TIMEOUT_S,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT_S,
    ) -> None:
        self._socket_dir: Path
        if socket_dir:
            self._socket_dir = Path(socket_dir)
        else:
            self._socket_dir = Path(tempfile.mkdtemp(prefix="oo_plugin_"))

        self._log_dir: Path
        if log_dir:
            self._log_dir = Path(log_dir)
        else:
            self._log_dir = Path("data") / "plugin_logs"

        self._worker_script: Path
        if worker_script:
            self._worker_script = Path(worker_script)
        else:
            self._worker_script = Path(__file__).parent / "plugin_worker.py"

        self._watchdog_interval = watchdog_interval
        self._default_hook_timeout = default_hook_timeout
        self._startup_timeout = startup_timeout

        self._processes: dict[str, PluginProcess] = {}
        self._lock = threading.Lock()
        self._watchdog_thread: Optional[threading.Thread] = None
        self._watchdog_stop = threading.Event()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def running_plugins(self) -> list[str]:
        """List of currently running plugin names."""
        with self._lock:
            return [n for n, p in self._processes.items() if p.is_alive()]

    @property
    def socket_dir(self) -> Path:
        """Directory used for Unix domain sockets."""
        return self._socket_dir

    @property
    def log_dir(self) -> Path:
        """Directory used for plugin log files."""
        return self._log_dir

    # ------------------------------------------------------------------
    # Subprocess lifecycle
    # ------------------------------------------------------------------

    def start_plugin(
        self,
        plugin_name: str,
        plugin_dir: str | Path,
        entry_point: str,
        *,
        resource_limits: Optional[PluginResourceLimits] = None,
        env_extra: Optional[dict[str, str]] = None,
    ) -> PluginProcess:
        """Launch a plugin in a new subprocess.

        Parameters
        ----------
        plugin_name : str
            Unique plugin identifier.
        plugin_dir : str or Path
            Directory containing the plugin files.
        entry_point : str
            Relative path to the plugin entry point file.
        resource_limits : PluginResourceLimits, optional
            Resource limits for the subprocess.
        env_extra : dict, optional
            Additional environment variables for the subprocess.

        Returns
        -------
        PluginProcess
            Handle to the running subprocess.
        """
        with self._lock:
            # Kill existing process if any
            if plugin_name in self._processes:
                self._kill_plugin_unlocked(plugin_name)

        rlimits = resource_limits or PluginResourceLimits()

        # Generate per-plugin HMAC key
        hmac_key = secrets.token_bytes(32)

        # Socket path
        self._socket_dir.mkdir(parents=True, exist_ok=True)
        sock_path = str(self._socket_dir / f"{plugin_name}.sock")
        # Clean up stale socket
        if os.path.exists(sock_path):
            os.unlink(sock_path)

        # Setup plugin logger
        plugin_log = setup_plugin_logger(plugin_name, self._log_dir)

        # Build subprocess environment. Plugin code runs untrusted in the
        # worker, so the host environment is NOT inherited (see
        # _build_plugin_env); only a minimal, secret-free base plus the explicit
        # OO_* variables is forwarded.
        env = _build_plugin_env({
            "OO_PLUGIN_NAME": plugin_name,
            "OO_PLUGIN_DIR": str(Path(plugin_dir).resolve()),
            "OO_PLUGIN_ENTRY": entry_point,
            "OO_SOCKET_PATH": sock_path,
            "OO_HMAC_KEY": hmac_key.hex(),
            "OO_RLIMIT_CPU": str(rlimits.cpu_time_seconds),
            "OO_RLIMIT_MEM": str(rlimits.memory_bytes),
            "OO_RLIMIT_NOFILE": str(rlimits.max_file_descriptors),
        })
        if env_extra:
            env.update(env_extra)

        # Launch subprocess
        try:
            proc = subprocess.Popen(
                [
                    # PSB-04: launch the worker with the host's own
                    # interpreter (ui.py convention); a bare "python3"
                    # resolves via PATH and can differ in venv/conda
                    # setups.
                    sys.executable,
                    str(self._worker_script),
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(Path(plugin_dir).resolve()),
            )
        except OSError as exc:
            raise PluginSubprocessError(
                f"Failed to start subprocess for plugin '{plugin_name}': {exc}"
            ) from exc

        pp = PluginProcess(
            plugin_name=plugin_name,
            process=proc,
            socket_path=sock_path,
            hmac_key=hmac_key,
            plugin_logger=plugin_log,
        )

        # Start stdout/stderr capture threads
        if proc.stdout:
            t_out = threading.Thread(
                target=_stream_reader,
                args=(proc.stdout, plugin_log, "stdout"),
                daemon=True,
                name=f"oo-plugin-{plugin_name}-stdout",
            )
            t_out.start()
        if proc.stderr:
            t_err = threading.Thread(
                target=_stream_reader,
                args=(proc.stderr, plugin_log, "stderr"),
                daemon=True,
                name=f"oo-plugin-{plugin_name}-stderr",
            )
            t_err.start()

        # Wait for socket to appear (worker creates it)
        if not self._wait_for_socket(sock_path, proc):
            rc = proc.poll()
            raise PluginSubprocessError(
                f"Plugin '{plugin_name}' subprocess did not create socket "
                f"within {self._startup_timeout}s (exit code: {rc})"
            )

        # Connect to the plugin socket
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            conn.connect(sock_path)
        except OSError as exc:
            proc.kill()
            raise PluginSubprocessError(
                f"Failed to connect to plugin '{plugin_name}' socket: {exc}"
            ) from exc

        pp.conn = conn

        # Send init handshake
        try:
            self._rpc_call(
                pp, "initialize", {"plugin_name": plugin_name},
                timeout=self._startup_timeout,
            )
        except Exception as exc:
            conn.close()
            proc.kill()
            raise PluginSubprocessError(
                f"Plugin '{plugin_name}' failed initialization handshake: {exc}"
            ) from exc

        pp.touch_heartbeat()

        with self._lock:
            self._processes[plugin_name] = pp

        # PSB-05: lazily start the watchdog with the first worker --
        # nothing in the product ever started it, so dead-process
        # reaping never ran.  start_watchdog() is idempotent.
        self.start_watchdog()

        logger.info(
            "Started plugin '%s' subprocess (pid=%d, socket=%s)",
            plugin_name, proc.pid, sock_path,
        )
        return pp

    def stop_plugin(self, plugin_name: str, timeout: float = 5.0) -> bool:
        """Gracefully stop a plugin subprocess.

        Sends a 'shutdown' RPC, waits for exit, then force-kills if needed.
        Returns True if the plugin was running and stopped.
        """
        with self._lock:
            pp = self._processes.pop(plugin_name, None)
        if pp is None:
            return False

        self._shutdown_process(pp, timeout)
        return True

    def stop_all(self, timeout: float = 5.0) -> int:
        """Stop all running plugin subprocesses.

        Returns the number of plugins stopped.
        """
        self.stop_watchdog()
        with self._lock:
            names = list(self._processes.keys())
        count = 0
        for name in names:
            if self.stop_plugin(name, timeout=timeout):
                count += 1
        return count

    def is_running(self, plugin_name: str) -> bool:
        """Check if a plugin subprocess is alive."""
        with self._lock:
            pp = self._processes.get(plugin_name)
        return pp is not None and pp.is_alive()

    def get_process(self, plugin_name: str) -> Optional[PluginProcess]:
        """Get the PluginProcess handle for a plugin, or None."""
        with self._lock:
            return self._processes.get(plugin_name)

    # ------------------------------------------------------------------
    # RPC interface
    # ------------------------------------------------------------------

    def call_hook(
        self,
        plugin_name: str,
        hook_name: str,
        context_data: dict[str, Any],
        *,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        """Call a hook on a plugin via RPC.

        Parameters
        ----------
        plugin_name : str
            Target plugin.
        hook_name : str
            Hook point name.
        context_data : dict
            Hook context data to pass.
        timeout : float, optional
            RPC timeout (default: ``default_hook_timeout``).

        Returns
        -------
        dict
            Result from the plugin hook.

        Raises
        ------
        PluginSubprocessError
            If the plugin is not running or RPC fails.
        PluginSubprocessTimeout
            If the call times out.
        """
        with self._lock:
            pp = self._processes.get(plugin_name)
        if pp is None or not pp.is_alive():
            raise PluginSubprocessError(
                f"Plugin '{plugin_name}' is not running"
            )

        t = timeout if timeout is not None else self._default_hook_timeout
        try:
            result = self._rpc_call(
                pp, "execute_hook",
                {"hook_name": hook_name, "data": context_data},
                timeout=t,
            )
        except (PluginSubprocessTimeout, PluginIPCError) as exc:
            # PSB-02: a timed-out or failed RPC leaves the single
            # request/response channel desynchronized -- the worker's
            # late reply would be read by the NEXT call, producing a
            # permanent ID-mismatch on every subsequent hook.  Tear the
            # worker down so the plugin is cleanly "not running" (and
            # can be re-enabled) instead of poisoning the channel.
            # A clean JSON-RPC error response (PluginSubprocessError
            # without IPC failure) keeps the channel in sync and does
            # NOT kill the worker.
            logger.warning(
                "call_hook '%s' on plugin '%s' failed (%s); "
                "terminating the worker to avoid channel desync",
                hook_name, plugin_name, exc,
            )
            with self._lock:
                self._kill_plugin_unlocked(plugin_name)
            raise
        pp.touch_heartbeat()
        return result

    def ping(self, plugin_name: str, timeout: float = 3.0) -> bool:
        """Ping a plugin subprocess to check health.

        Returns True if the plugin responds.
        """
        with self._lock:
            pp = self._processes.get(plugin_name)
        if pp is None or not pp.is_alive():
            return False
        try:
            result = self._rpc_call(pp, "ping", {}, timeout=timeout)
            pp.touch_heartbeat()
            return result.get("status") == "pong"
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Watchdog
    # ------------------------------------------------------------------

    def start_watchdog(self) -> None:
        """Start the background watchdog thread that monitors subprocesses."""
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            daemon=True,
            name="oo-plugin-watchdog",
        )
        self._watchdog_thread.start()
        logger.info("Plugin watchdog started (interval=%.1fs)", self._watchdog_interval)

    def stop_watchdog(self) -> None:
        """Stop the watchdog thread."""
        self._watchdog_stop.set()
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=self._watchdog_interval + 1)
            self._watchdog_thread = None
            logger.info("Plugin watchdog stopped")

    def _watchdog_loop(self) -> None:
        """Watchdog background loop: check subprocess health."""
        while not self._watchdog_stop.is_set():
            self._watchdog_check()
            self._watchdog_stop.wait(self._watchdog_interval)

    def _watchdog_check(self) -> None:
        """Single watchdog pass: detect dead subprocesses."""
        with self._lock:
            items = list(self._processes.items())

        for name, pp in items:
            if not pp.is_alive():
                logger.warning(
                    "Watchdog: plugin '%s' subprocess died (exit=%s)",
                    name, pp.process.returncode,
                )
                if pp.plugin_logger:
                    pp.plugin_logger.warning(
                        "Subprocess exited unexpectedly (code=%s)",
                        pp.process.returncode,
                    )
                with self._lock:
                    self._processes.pop(name, None)
                self._cleanup_socket(pp.socket_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rpc_call(
        self,
        pp: PluginProcess,
        method: str,
        params: dict[str, Any],
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """Send an RPC request and wait for the response."""
        if pp.conn is None:
            raise PluginIPCError(f"No connection to plugin '{pp.plugin_name}'")

        req = make_rpc_request(method, params)
        request_id = req["id"]

        with pp._lock:
            send_message(pp.conn, pp.hmac_key, req)
            resp = recv_message(pp.conn, pp.hmac_key, timeout=timeout)

        # Validate response
        if resp.get("id") != request_id:
            raise PluginIPCError(
                f"Response ID mismatch: expected {request_id}, got {resp.get('id')}"
            )

        if "error" in resp:
            err = resp["error"]
            raise PluginSubprocessError(
                f"Plugin RPC error [{err.get('code')}]: {err.get('message')}"
            )

        return resp.get("result", {})

    def _wait_for_socket(
        self,
        sock_path: str,
        proc: subprocess.Popen,  # type: ignore[type-arg]
    ) -> bool:
        """Wait for a socket file to appear, with timeout."""
        deadline = time.time() + self._startup_timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                return False  # Process exited
            if os.path.exists(sock_path):
                return True
            time.sleep(0.05)
        return False

    def _shutdown_process(self, pp: PluginProcess, timeout: float = 5.0) -> None:
        """Gracefully shutdown a plugin subprocess."""
        # Try graceful RPC shutdown
        if pp.conn and pp.is_alive():
            try:
                self._rpc_call(pp, "shutdown", {}, timeout=min(timeout, 3.0))
            except Exception:
                pass

        # Close socket connection
        if pp.conn:
            try:
                pp.conn.close()
            except OSError:
                pass

        # Wait for process to exit
        if pp.is_alive():
            try:
                pp.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Plugin '%s' did not exit gracefully, killing",
                    pp.plugin_name,
                )
                pp.process.kill()
                pp.process.wait(timeout=2.0)

        self._cleanup_socket(pp.socket_path)
        logger.info("Stopped plugin '%s' subprocess", pp.plugin_name)

    def _kill_plugin_unlocked(self, plugin_name: str) -> None:
        """Kill a plugin subprocess (caller must hold self._lock)."""
        pp = self._processes.pop(plugin_name, None)
        if pp is None:
            return
        if pp.conn:
            try:
                pp.conn.close()
            except OSError:
                pass
        if pp.is_alive():
            pp.process.kill()
            try:
                pp.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
        self._cleanup_socket(pp.socket_path)

    @staticmethod
    def _cleanup_socket(sock_path: str) -> None:
        """Remove a Unix socket file."""
        try:
            if os.path.exists(sock_path):
                os.unlink(sock_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

PLUGIN_SUBPROCESS_AVAILABLE = True

try:
    from opti_oignon.config import DATA_DIR as _DATA_DIR

    _log_dir = Path(_DATA_DIR) / "plugin_logs"
    subprocess_manager = PluginSubprocessManager(log_dir=_log_dir)
except Exception as _exc:
    logger.debug("PluginSubprocessManager singleton init deferred: %s", _exc)
    subprocess_manager = None  # type: ignore[assignment]
