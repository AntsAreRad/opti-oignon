#!/usr/bin/env python3
"""
Plugin worker process for Opti-Oignon (S143).

This script is launched by PluginSubprocessManager as a child process.
It reads configuration from environment variables, applies resource
limits, loads the plugin entry point, creates a Unix domain socket,
and serves JSON-RPC requests from the host.

Environment variables (set by the host):
    OO_PLUGIN_NAME   — plugin identifier
    OO_PLUGIN_DIR    — absolute path to plugin directory
    OO_PLUGIN_ENTRY  — relative path to entry point file
    OO_SOCKET_PATH   — Unix socket path for IPC
    OO_HMAC_KEY      — hex-encoded 32-byte HMAC key
    OO_RLIMIT_CPU    — CPU time limit in seconds
    OO_RLIMIT_MEM    — memory limit in bytes
    OO_RLIMIT_NOFILE — max open file descriptors
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
from typing import Any, Optional

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

def apply_resource_limits(
    cpu_seconds: int,
    memory_bytes: int,
    max_fds: int,
) -> dict[str, Any]:
    """Apply OS-level resource limits to this process.

    Returns a dict of applied limits for logging.
    """
    applied: dict[str, Any] = {}

    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        applied["RLIMIT_CPU"] = cpu_seconds
    except (ValueError, OSError) as exc:
        logger.warning("Failed to set RLIMIT_CPU: %s", exc)

    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        applied["RLIMIT_AS"] = memory_bytes
    except (ValueError, OSError) as exc:
        logger.warning("Failed to set RLIMIT_AS: %s", exc)

    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (max_fds, max_fds))
        applied["RLIMIT_NOFILE"] = max_fds
    except (ValueError, OSError) as exc:
        logger.warning("Failed to set RLIMIT_NOFILE: %s", exc)

    return applied


# ---------------------------------------------------------------------------
# Plugin loading (simplified, no sandbox — isolation is via process boundary)
# ---------------------------------------------------------------------------

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

def execute_hook(
    module: types.ModuleType,
    hook_name: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Execute a hook function on the loaded plugin module.

    Looks for either a ``HOOKS`` dict mapping or ``hook_<name>`` function.

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

    result = callback(data)
    if isinstance(result, dict):
        return result
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# JSON-RPC response builders
# ---------------------------------------------------------------------------

def make_response(request_id: Optional[str], result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def make_error(
    request_id: Optional[str],
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
    """Unix domain socket server handling JSON-RPC from the host process."""

    def __init__(
        self,
        plugin_name: str,
        plugin_dir: str,
        entry_point: str,
        socket_path: str,
        hmac_key: bytes,
    ) -> None:
        self.plugin_name = plugin_name
        self.plugin_dir = plugin_dir
        self.entry_point = entry_point
        self.socket_path = socket_path
        self.hmac_key = hmac_key
        self.module: Optional[types.ModuleType] = None
        self._running = False
        self._server_sock: Optional[socket.socket] = None

    def start(self) -> None:
        """Create the listening socket, accept a connection, and serve."""
        # Create listening socket
        self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self._server_sock.bind(self.socket_path)
        except OSError as exc:
            logger.error("Failed to bind socket %s: %s", self.socket_path, exc)
            raise

        self._server_sock.listen(1)
        self._server_sock.settimeout(30.0)  # Accept timeout
        logger.info(
            "Worker for '%s' listening on %s",
            self.plugin_name, self.socket_path,
        )

        try:
            conn, _ = self._server_sock.accept()
        except socket.timeout:
            logger.error("No connection received within timeout")
            return

        logger.info("Host connected to worker '%s'", self.plugin_name)
        self._running = True

        try:
            self._serve(conn)
        except Exception as exc:
            logger.error("Worker loop error: %s", exc)
        finally:
            conn.close()
            self._cleanup()

    def _serve(self, conn: socket.socket) -> None:
        """Read JSON-RPC requests and dispatch them."""
        while self._running:
            try:
                msg = recv_message(conn, self.hmac_key, timeout=60.0)
            except socket.timeout:
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
        request_id: Optional[str],
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
        self, params: dict[str, Any], request_id: Optional[str],
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
        self, params: dict[str, Any], request_id: Optional[str],
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
            result = execute_hook(self.module, hook_name, data)
            return make_response(request_id, result)
        except Exception as exc:
            logger.error(
                "Hook '%s' execution failed: %s", hook_name, exc,
            )
            return make_error(
                request_id, -32603,
                f"Hook execution failed: {exc}",
            )

    def _handle_ping(self, request_id: Optional[str]) -> dict[str, Any]:
        """Respond to a health check."""
        return make_response(request_id, {
            "status": "pong",
            "plugin_name": self.plugin_name,
            "uptime": time.time(),
        })

    def _handle_shutdown(self, request_id: Optional[str]) -> dict[str, Any]:
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

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self._server_sock:
            try:
                self._server_sock.close()
            except OSError:
                pass
        # Socket file cleanup
        try:
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
        except OSError:
            pass
        logger.info("Worker '%s' cleaned up", self.plugin_name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point for the plugin worker subprocess."""
    # Read configuration from environment
    plugin_name = os.environ.get("OO_PLUGIN_NAME", "")
    plugin_dir = os.environ.get("OO_PLUGIN_DIR", "")
    entry_point = os.environ.get("OO_PLUGIN_ENTRY", "")
    socket_path = os.environ.get("OO_SOCKET_PATH", "")
    hmac_key_hex = os.environ.get("OO_HMAC_KEY", "")

    if not all([plugin_name, plugin_dir, entry_point, socket_path, hmac_key_hex]):
        logger.error("Missing required environment variables")
        sys.exit(1)

    hmac_key = bytes.fromhex(hmac_key_hex)

    # Apply resource limits
    cpu_limit = int(os.environ.get("OO_RLIMIT_CPU", "30"))
    mem_limit = int(os.environ.get("OO_RLIMIT_MEM", str(256 * 1024 * 1024)))
    fd_limit = int(os.environ.get("OO_RLIMIT_NOFILE", "64"))

    applied = apply_resource_limits(cpu_limit, mem_limit, fd_limit)
    logger.info("Resource limits applied: %s", applied)

    # Ignore SIGINT (host handles signals)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Start the worker server
    server = PluginWorkerServer(
        plugin_name=plugin_name,
        plugin_dir=plugin_dir,
        entry_point=entry_point,
        socket_path=socket_path,
        hmac_key=hmac_key,
    )

    try:
        server.start()
    except Exception as exc:
        logger.error("Worker failed: %s", exc)
        sys.exit(1)

    logger.info("Worker '%s' exiting", plugin_name)


if __name__ == "__main__":
    main()
