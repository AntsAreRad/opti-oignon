#!/usr/bin/env python3
"""
Plugin worker process for Opti-Oignon.

This script is launched by PluginSubprocessManager as a child process,
under bubblewrap whenever the sandbox runs there. It reads its
configuration from environment variables, checks that the resource limits
the host installed before it executed are in place -- and refuses to serve
when one is not -- then loads the plugin entry point and serves JSON-RPC
requests from the host over the socket it inherited.

Environment variables (set by the host):
    OO_PLUGIN_NAME     -- plugin identifier
    OO_PLUGIN_DIR      -- absolute path to plugin directory
    OO_PLUGIN_ENTRY    -- relative path to entry point file
    OO_SOCKET_FD       -- inherited descriptor of the worker's socket end
    OO_HMAC_KEY        -- hex-encoded 32-byte HMAC key
    OO_RLIMIT_CPU      -- CPU time limit in seconds
    OO_RLIMIT_MEM      -- address space limit in bytes
    OO_RLIMIT_NOFILE   -- max open file descriptors
    OO_RLIMIT_NPROC    -- max processes of the user
    OO_RLIMIT_FSIZE    -- max size of a written file in bytes
    OO_PLUGIN_DATA_DIR -- the plugin's private data folder, when it may write
"""

import hashlib
import hmac as _hmac
import importlib.util
import json
import logging
import os
import resource
import signal
import socket
import struct
import sys
import time
import types
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging (to stderr, captured by host)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("plugin_worker")

# ---------------------------------------------------------------------------
# Wire protocol constants (must match plugin_subprocess.py)
# ---------------------------------------------------------------------------

MAX_MESSAGE_SIZE: int = 4 * 1024 * 1024
HEADER_SIZE: int = 4 + 32  # 4-byte length + 32-byte HMAC-SHA256


# ---------------------------------------------------------------------------
# HMAC helpers
# ---------------------------------------------------------------------------

def _compute_hmac(key: bytes, data: bytes) -> bytes:
    return _hmac.new(key, data, hashlib.sha256).digest()


def _verify_hmac(key: bytes, data: bytes, expected: bytes) -> bool:
    computed = _hmac.new(key, data, hashlib.sha256).digest()
    return _hmac.compare_digest(computed, expected)


# ---------------------------------------------------------------------------
# Wire protocol
# ---------------------------------------------------------------------------

def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly *n* bytes from a socket."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed while reading")
        buf.extend(chunk)
    return bytes(buf)


def recv_message(sock: socket.socket, key: bytes, timeout: float = 30.0) -> dict[str, Any]:
    """Read a single framed message from a socket."""
    sock.settimeout(timeout)
    header = recv_exact(sock, HEADER_SIZE)
    length = struct.unpack("!I", header[:4])[0]
    if length > MAX_MESSAGE_SIZE:
        raise ValueError(f"Message length {length} exceeds limit {MAX_MESSAGE_SIZE}")
    mac = header[4:36]
    raw = recv_exact(sock, length)
    if not _verify_hmac(key, raw, mac):
        raise ValueError("HMAC verification failed")
    return json.loads(raw.decode("utf-8"))


def send_message(sock: socket.socket, key: bytes, payload: dict[str, Any]) -> None:
    """Send a single framed message over a socket."""
    raw = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
    if len(raw) > MAX_MESSAGE_SIZE:
        raise ValueError(f"Message size {len(raw)} exceeds limit {MAX_MESSAGE_SIZE}")
    mac = _compute_hmac(key, raw)
    header = struct.pack("!I", len(raw)) + mac
    sock.sendall(header + raw)


# ---------------------------------------------------------------------------
# Resource limits
# ---------------------------------------------------------------------------

# Exit status of a worker that refuses to serve without its limits.
EXIT_LIMITS_MISSING = 3

# Each limit the host installs, and the variable that announces it.
_ANNOUNCED_LIMITS = (
    ("OO_RLIMIT_CPU", resource.RLIMIT_CPU, "RLIMIT_CPU"),
    ("OO_RLIMIT_MEM", resource.RLIMIT_AS, "RLIMIT_AS"),
    ("OO_RLIMIT_NOFILE", resource.RLIMIT_NOFILE, "RLIMIT_NOFILE"),
    ("OO_RLIMIT_NPROC", resource.RLIMIT_NPROC, "RLIMIT_NPROC"),
    ("OO_RLIMIT_FSIZE", resource.RLIMIT_FSIZE, "RLIMIT_FSIZE"),
)


def missing_resource_limits(environ: Any) -> list[str]:
    """Name each limit the host announced and did not install.

    The host sets the limits before this process executes; the worker only
    checks them. Setting them here would hide a launcher that forgot to.
    """
    missing: list[str] = []
    for variable, rid, name in _ANNOUNCED_LIMITS:
        try:
            expected = int(environ[variable])
        except (KeyError, ValueError):
            missing.append(f"{name} is not announced ({variable})")
            continue
        hard = resource.getrlimit(rid)[1]
        if hard == resource.RLIM_INFINITY or hard > expected:
            shown = "unlimited" if hard == resource.RLIM_INFINITY else str(hard)
            missing.append(
                f"{name} is not in place (hard limit {shown}, expected at most {expected})"
            )
    return missing


# ---------------------------------------------------------------------------
# Plugin loading (simplified, no sandbox -- isolation is via process boundary)
# ---------------------------------------------------------------------------

class _HostPackageGuard:
    """Meta-path finder that refuses the host package inside the worker.

    The worker runs plugin code behind a process boundary with a
    minimal, secret-free environment: no PYTHONPATH, no encryption
    keys, no search credentials. The host package cannot work there,
    and an attempt to import it must fail fast and deterministically --
    not hang the initialization handshake on whatever the host install
    layout happens to drag in. Plugins are expected to catch the
    ImportError and engage their standalone fallbacks.
    """

    _BLOCKED_TOP_LEVEL = "opti_oignon"

    def find_spec(self, fullname: str, path=None, target=None):
        if fullname.split(".")[0] == self._BLOCKED_TOP_LEVEL:
            raise ImportError(
                f"'{fullname}' is not importable inside the plugin "
                "isolation boundary (the worker forwards no host "
                "package, no PYTHONPATH and no secrets)"
            )
        return None


def install_host_package_guard() -> None:
    """Install the host-package import guard (idempotent)."""
    if any(isinstance(f, _HostPackageGuard) for f in sys.meta_path):
        return
    sys.meta_path.insert(0, _HostPackageGuard())


def load_plugin_module(
    plugin_name: str,
    plugin_dir: str,
    entry_point: str,
) -> types.ModuleType:
    """Load a plugin's entry point as a Python module.

    Parameters
    ----------
    plugin_name : str
        Unique plugin name.
    plugin_dir : str
        Absolute path to plugin directory.
    entry_point : str
        Relative filename of the entry point script.

    Returns
    -------
    types.ModuleType
        The loaded plugin module.
    """
    entry_path = Path(plugin_dir) / entry_point
    if not entry_path.exists():
        raise FileNotFoundError(f"Entry point not found: {entry_path}")

    # The plugin's module-level imports run inside exec_module below:
    # the boundary must be in place before any of them can reach for
    # the host package.
    install_host_package_guard()

    module_name = f"_oo_worker_plugin_{plugin_name}"
    spec = importlib.util.spec_from_file_location(module_name, str(entry_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {entry_path}")

    module = importlib.util.module_from_spec(spec)
    module.__plugin_name__ = plugin_name  # type: ignore[attr-defined]
    module.__plugin_dir__ = plugin_dir  # type: ignore[attr-defined]
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Hook execution
# ---------------------------------------------------------------------------

class WorkerHookContext:
    """Context object handed to plugin hook callbacks in the worker.

    Mirrors the public surface of the host-side hook context (the seven
    fields plus ``get``/``set``) so plugin code written against it runs
    unchanged across the process boundary. The host RPC proxy serializes
    that object into a seven-field wire payload; this class rebuilds the
    object the plugins expect. Mutating the context (including ``set()``)
    affects only this local view: to propagate changes downstream a hook
    must RETURN a dict, exactly as in-process.
    """

    def __init__(
        self,
        wire: dict[str, Any],
        *,
        hook_name: str,
        plugin_name: str,
    ) -> None:
        self.hook_name = str(wire.get("hook_name") or hook_name)
        self.plugin_name = str(wire.get("plugin_name") or plugin_name)
        self.conversation_id = wire.get("conversation_id")
        self.model = wire.get("model")
        data = wire.get("data")
        self.data: dict[str, Any] = dict(data) if isinstance(data, dict) else {}
        config = wire.get("config")
        self.config: dict[str, Any] = (
            dict(config) if isinstance(config, dict) else {}
        )
        metadata = wire.get("metadata")
        self.metadata: dict[str, Any] = (
            dict(metadata) if isinstance(metadata, dict) else {}
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the data dict."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in this hook's LOCAL data view (never propagated)."""
        self.data[key] = value


def execute_hook(
    module: types.ModuleType,
    hook_name: str,
    data: dict[str, Any],
    *,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """Execute a hook function on the loaded plugin module.

    Looks for either a ``HOOKS`` dict mapping or ``hook_<name>`` function.
    ``data`` is the wire context payload built by the host RPC proxy; the
    callback receives it rebuilt as a :class:`WorkerHookContext`. Its
    ``metadata["data_dir"]`` is the worker's own fact: the plugin's private
    data folder when it has one, absent otherwise, whatever the wire said.

    Returns
    -------
    dict
        Result data (or empty dict if hook returned None).
    """
    # Try HOOKS dict first
    hooks_dict = getattr(module, "HOOKS", None)
    callback = None
    if isinstance(hooks_dict, dict):
        callback = hooks_dict.get(hook_name)

    # Fall back to hook_<name> function
    if callback is None:
        fn_name = f"hook_{hook_name}"
        callback = getattr(module, fn_name, None)

    if callback is None or not callable(callback):
        return {"status": "no_handler", "hook_name": hook_name}

    context = WorkerHookContext(
        data,
        hook_name=hook_name,
        plugin_name=getattr(module, "__plugin_name__", ""),
    )
    context.metadata.pop("data_dir", None)
    if data_dir:
        context.metadata["data_dir"] = data_dir
    result = callback(context)
    if isinstance(result, dict):
        return result
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# JSON-RPC response builders
# ---------------------------------------------------------------------------

def make_response(request_id: str | None, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def make_error(
    request_id: str | None,
    code: int,
    message: str,
    data: Any = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


# ---------------------------------------------------------------------------
# Main server loop
# ---------------------------------------------------------------------------

class PluginWorkerServer:
    """Serves JSON-RPC from the host over the socket the worker inherited."""

    def __init__(
        self,
        plugin_name: str,
        plugin_dir: str,
        entry_point: str,
        hmac_key: bytes,
        *,
        conn: socket.socket | None = None,
        data_dir: str | None = None,
    ) -> None:
        self.plugin_name = plugin_name
        self.plugin_dir = plugin_dir
        self.entry_point = entry_point
        self.hmac_key = hmac_key
        self.conn = conn
        self.data_dir = data_dir
        self.module: types.ModuleType | None = None
        self._running = False

    def start(self) -> None:
        """Serve the host on the inherited connection until it ends."""
        if self.conn is None:
            raise RuntimeError("no connection to serve")
        logger.info("Worker for '%s' serving the host", self.plugin_name)
        self._running = True
        try:
            self._serve(self.conn)
        except Exception as exc:
            logger.error("Worker loop error: %s", exc)
        finally:
            self.conn.close()
            logger.info("Worker '%s' cleaned up", self.plugin_name)

    def _serve(self, conn: socket.socket) -> None:
        """Read JSON-RPC requests and dispatch them."""
        while self._running:
            try:
                msg = recv_message(conn, self.hmac_key, timeout=60.0)
            except TimeoutError:
                continue
            except (ConnectionError, ValueError) as exc:
                logger.error("Receive error: %s", exc)
                break

            method = msg.get("method", "")
            params = msg.get("params", {})
            request_id = msg.get("id")

            try:
                response = self._dispatch(method, params, request_id)
            except Exception as exc:
                logger.error("Dispatch error for '%s': %s", method, exc)
                response = make_error(
                    request_id, -32603,
                    f"Internal error: {exc}",
                )

            try:
                send_message(conn, self.hmac_key, response)
            except (ConnectionError, OSError) as exc:
                logger.error("Send error: %s", exc)
                break

    def _dispatch(
        self,
        method: str,
        params: dict[str, Any],
        request_id: str | None,
    ) -> dict[str, Any]:
        """Route a JSON-RPC method to the appropriate handler."""
        if method == "initialize":
            return self._handle_initialize(params, request_id)
        elif method == "execute_hook":
            return self._handle_execute_hook(params, request_id)
        elif method == "ping":
            return self._handle_ping(request_id)
        elif method == "shutdown":
            return self._handle_shutdown(request_id)
        else:
            return make_error(
                request_id, -32601,
                f"Method not found: {method}",
            )

    def _handle_initialize(
        self, params: dict[str, Any], request_id: str | None,
    ) -> dict[str, Any]:
        """Load the plugin module and call init() if present."""
        try:
            self.module = load_plugin_module(
                self.plugin_name, self.plugin_dir, self.entry_point,
            )
            # Call plugin's init() if it exists
            init_fn = getattr(self.module, "init", None)
            if callable(init_fn):
                init_fn()

            logger.info("Plugin '%s' initialized successfully", self.plugin_name)
            return make_response(request_id, {
                "status": "ok",
                "plugin_name": self.plugin_name,
            })
        except Exception as exc:
            logger.error("Plugin initialization failed: %s", exc)
            return make_error(
                request_id, -32603,
                f"Initialization failed: {exc}",
            )

    def _handle_execute_hook(
        self, params: dict[str, Any], request_id: str | None,
    ) -> dict[str, Any]:
        """Execute a hook on the loaded plugin."""
        if self.module is None:
            return make_error(
                request_id, -32603,
                "Plugin not initialized",
            )

        hook_name = params.get("hook_name", "")
        data = params.get("data", {})

        try:
            result = execute_hook(self.module, hook_name, data, data_dir=self.data_dir)
            return make_response(request_id, result)
        except Exception as exc:
            logger.error(
                "Hook '%s' execution failed: %s", hook_name, exc,
            )
            return make_error(
                request_id, -32603,
                f"Hook execution failed: {exc}",
            )

    def _handle_ping(self, request_id: str | None) -> dict[str, Any]:
        """Respond to a health check."""
        return make_response(request_id, {
            "status": "pong",
            "plugin_name": self.plugin_name,
            "uptime": time.time(),
        })

    def _handle_shutdown(self, request_id: str | None) -> dict[str, Any]:
        """Gracefully shut down the worker."""
        logger.info("Shutdown requested for '%s'", self.plugin_name)
        self._running = False

        # Call plugin shutdown() if present
        if self.module is not None:
            shutdown_fn = getattr(self.module, "shutdown", None)
            if callable(shutdown_fn):
                try:
                    shutdown_fn()
                except Exception as exc:
                    logger.warning("Plugin shutdown() error: %s", exc)

        return make_response(request_id, {"status": "ok"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point for the plugin worker subprocess."""
    # Read configuration from environment
    plugin_name = os.environ.get("OO_PLUGIN_NAME", "")
    plugin_dir = os.environ.get("OO_PLUGIN_DIR", "")
    entry_point = os.environ.get("OO_PLUGIN_ENTRY", "")
    socket_fd = os.environ.get("OO_SOCKET_FD", "")
    hmac_key_hex = os.environ.get("OO_HMAC_KEY", "")

    if not all([plugin_name, plugin_dir, entry_point, socket_fd, hmac_key_hex]):
        logger.error("Missing required environment variables")
        sys.exit(1)

    hmac_key = bytes.fromhex(hmac_key_hex)

    # The host installed the limits before this process executed; serve
    # only once each of them is seen in place.
    missing = missing_resource_limits(os.environ)
    if missing:
        for line in missing:
            logger.error("Resource limit %s: refusing to serve", line)
        sys.exit(EXIT_LIMITS_MISSING)

    # Ignore SIGINT (host handles signals)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Start the worker server
    server = PluginWorkerServer(
        plugin_name=plugin_name,
        plugin_dir=plugin_dir,
        entry_point=entry_point,
        hmac_key=hmac_key,
        conn=socket.socket(fileno=int(socket_fd)),
        data_dir=os.environ.get("OO_PLUGIN_DATA_DIR") or None,
    )

    try:
        server.start()
    except Exception as exc:
        logger.error("Worker failed: %s", exc)
        sys.exit(1)

    logger.info("Worker '%s' exiting", plugin_name)


if __name__ == "__main__":
    main()
