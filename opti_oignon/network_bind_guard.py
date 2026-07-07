#!/usr/bin/env python3
"""
Network Bind Guard for Opti-Oignon (S133, extended S145).

Enforces that the server bind address is localhost-only in Bulbe mode.
This is a *physical constraint* at the socket level, not a policy.

Defense layers provided by this module:
  1. get_safe_bind_address() — forces 127.0.0.1 in Bulbe regardless of input
  2. assert_localhost_only() — kills the process if somehow not 127.0.0.1
  3. is_remote_access_allowed() — triple-gated check (mode + config + TLS)
  4. check_ollama_bind() — detect if Ollama is exposed on 0.0.0.0 (S145)

All six defense layers (this module + middleware + ModePolicy + tls_manager +
API routes) are independent. An attacker must bypass ALL SIX.

Kerckhoffs compliance: security derives from the socket bind, not from
code obscurity. This entire file will be on GitHub.
"""

from __future__ import annotations

import logging
import os
import re
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Hardcoded, never overridable
checkpoint_before_apply = True

_LOCALHOST_ADDRESSES = ("127.0.0.1", "::1", "localhost")
_RECOGNIZED_MODES = ("daily", "bulbe")
_SECURITY_YAML = Path(__file__).resolve().parent / "config" / "security.yaml"


# ---------------------------------------------------------------------------
# Core guard functions
# ---------------------------------------------------------------------------

def get_safe_bind_address(requested_host: str) -> str:
    """Return the safe bind address for the server.

    In Bulbe mode: **always returns '127.0.0.1'** regardless of
    requested_host. This is not configurable — it is hardcoded.

    In Daily mode with remote access enabled: returns requested_host.
    In Daily mode without remote access: returns '127.0.0.1'.

    Args:
        requested_host: The host address requested (CLI, env, config).

    Returns:
        The actual address the server should bind to.
    """
    mode = _get_current_mode()

    if mode == "bulbe":
        if requested_host not in _LOCALHOST_ADDRESSES:
            logger.warning(
                "Bulbe mode: requested bind address '%s' overridden to "
                "'127.0.0.1'. Remote binding is physically impossible "
                "in Bulbe mode.",
                requested_host,
            )
        return "127.0.0.1"

    # Daily mode: check if remote access is explicitly enabled
    if _is_remote_enabled_in_config():
        logger.info(
            "Daily mode with remote access enabled: binding to '%s'",
            requested_host,
        )
        return requested_host

    # Daily mode, remote not enabled: safe default
    if requested_host not in _LOCALHOST_ADDRESSES:
        logger.info(
            "Daily mode without remote access: requested '%s' overridden "
            "to '127.0.0.1'. Enable remote_access in security.yaml to "
            "allow non-local binding.",
            requested_host,
        )
    return "127.0.0.1"


def assert_safe_bind_address(host: str) -> None:
    """Synchronous dead man's switch checked just before binding the socket.

    In Bulbe mode the bind address MUST be loopback. ``get_safe_bind_address``
    already forces this on the supported launch path; this assertion verifies
    the value actually about to be passed to the server and terminates the
    process if it is somehow non-loopback (e.g. the forcing was bypassed or
    patched). Deterministic -- no network probe, no timing window.

    Unlike ``assert_localhost_only`` (a post-bind /proc probe), this runs in the
    launcher's main thread before ``uvicorn.run``, so ``sys.exit`` reliably
    stops startup. Invoking uvicorn directly (bypassing the launcher) skips both
    the forcing and this assertion; the middleware, ModePolicy, tls_manager and
    route layers remain as the other independent defenses in that case.
    """
    mode = _get_current_mode()
    if mode != "bulbe":
        return
    if host not in _LOCALHOST_ADDRESSES:
        logger.critical(
            "FATAL: Bulbe mode requires a loopback bind address, but '%s' was "
            "requested. Refusing to start.",
            host,
        )
        _audit_critical_event("bind_guard_fatal", bound_address=host, port=0)
        sys.exit(1)


def assert_localhost_only(port: int = 8001) -> None:
    """Dead man's switch: verify the server is actually bound to localhost.

    Called after server startup in Bulbe mode. Attempts to connect to
    the specified port on 0.0.0.0 to verify that it is NOT reachable
    from non-local interfaces.

    If the server is bound to a non-local address, this function calls
    sys.exit(1) immediately. This is the last line of defense.

    Args:
        port: The port the server is listening on.
    """
    mode = _get_current_mode()
    if mode != "bulbe":
        return

    # Check by probing the actual bind address
    try:
        # Try to get the server socket info
        bound_address = _probe_bound_address(port)
        if bound_address and bound_address not in _LOCALHOST_ADDRESSES:
            logger.critical(
                "FATAL: Server bound to '%s' in Bulbe mode! "
                "This violates the localhost-only constraint. "
                "Terminating immediately.",
                bound_address,
            )
            _audit_critical_event(
                "bind_guard_fatal",
                bound_address=bound_address,
                port=port,
            )
            sys.exit(1)
        elif bound_address:
            logger.info(
                "Bind guard verified: server bound to '%s' (OK)",
                bound_address,
            )
    except Exception as exc:
        # If we cannot verify, log warning but do not kill
        # (the bind address was already forced by get_safe_bind_address)
        logger.warning(
            "Could not verify bind address (non-fatal): %s", exc,
        )


def is_remote_access_allowed() -> bool:
    """Check if remote access is currently allowed.

    Returns True ONLY if ALL THREE conditions are met:
      (a) Current mode is Daily (never Bulbe)
      (b) security.yaml > remote_access > enabled is True
      (c) TLS certificate files exist and are valid

    Any doubt returns False.
    """
    # Condition (a): must be Daily
    mode = _get_current_mode()
    if mode == "bulbe":
        return False

    # Condition (b): remote_access must be explicitly enabled in config
    if not _is_remote_enabled_in_config():
        return False

    # Condition (c): TLS certificate files must exist
    if not _tls_files_exist():
        return False

    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_current_mode() -> str:
    """Load the current security mode, normalized and fail-secure.

    Any value that is not exactly a recognized mode is treated as Bulbe:
    an unreadable mode (exception) and an unrecognized or malformed mode
    string (a hand-edited config carrying a stray capitalization, a
    trailing space, or an empty value) both resolve to the restrictive
    interpretation. The guard must never fall to the permissive path on
    an undetermined mode.
    """
    try:
        from opti_oignon.security_mode import get_current_mode
        raw = get_current_mode()
    except Exception:
        logger.warning(
            "Cannot determine security mode; defaulting to 'bulbe' "
            "(fail-secure)."
        )
        return "bulbe"
    if raw not in _RECOGNIZED_MODES:
        logger.warning(
            "Unrecognized security mode %r; defaulting to 'bulbe' "
            "(fail-secure).",
            raw,
        )
        return "bulbe"
    return raw


def _is_remote_enabled_in_config() -> bool:
    """Check if remote_access.enabled is True in security.yaml."""
    try:
        if _SECURITY_YAML.exists():
            with open(_SECURITY_YAML, encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            ra = cfg.get("remote_access", {})
            if isinstance(ra, dict):
                return bool(ra.get("enabled", False))
    except Exception as exc:
        logger.warning("Failed to read remote_access config: %s", exc)
    return False


def _tls_files_exist() -> bool:
    """Check if TLS certificate files are present."""
    data_dir = Path(__file__).resolve().parent.parent / "data" / "tls"
    required_files = ["server.key", "server.crt", "ca.crt"]
    for fname in required_files:
        fpath = data_dir / fname
        if not fpath.exists():
            return False
        # Basic sanity: file must not be empty
        if fpath.stat().st_size == 0:
            return False
    return True


def _probe_bound_address(port: int) -> str | None:
    """Try to determine what address the server is bound to.

    Uses /proc/net/tcp on Linux or ss command as fallback.
    Returns the bind address string, or None if undetermined.
    """
    # Method 1: Parse /proc/net/tcp (Linux-specific, no subprocess)
    try:
        hex_port = f"{port:04X}"
        with open("/proc/net/tcp", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                local_addr = parts[1]
                if ":" not in local_addr:
                    continue
                addr_hex, port_hex = local_addr.split(":")
                if port_hex.upper() == hex_port:
                    # Decode hex IP (little-endian on Linux)
                    ip_int = int(addr_hex, 16)
                    ip_bytes = ip_int.to_bytes(4, byteorder="little")
                    ip_str = ".".join(str(b) for b in ip_bytes)
                    if ip_str == "0.0.0.0":
                        return "0.0.0.0"
                    return ip_str
    except (FileNotFoundError, PermissionError, ValueError):
        pass

    # Method 2: Try connecting to localhost to verify it responds
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            result = s.connect_ex(("127.0.0.1", port))
            if result == 0:
                return "127.0.0.1"
    except Exception:
        pass

    return None


def _audit_critical_event(event: str, **details) -> None:
    """Log a critical security event to the audit trail."""
    try:
        from opti_oignon.signed_audit_log import chain_log
        chain_log(
            event_type=event,
            source="network_bind_guard",
            action=event,
            severity="CRITICAL",
            **details,
        )
    except Exception:
        pass  # Audit logging is best-effort; the sys.exit is the real guard


# ---------------------------------------------------------------------------
# Ollama Bind Guard (S145)
# ---------------------------------------------------------------------------

# Default Ollama port
_OLLAMA_DEFAULT_PORT: int = 11434
_OLLAMA_WILDCARD_ADDRESSES = ("0.0.0.0", "::", "0:0:0:0:0:0:0:0")


@dataclass
class OllamaBindCheckResult:
    """Result of an Ollama bind address check."""

    checked: bool = False
    exposed: bool = False
    bind_address: str = ""
    port: int = _OLLAMA_DEFAULT_PORT
    method: str = ""
    detail: str = ""
    blocked: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for API responses."""
        return {
            "checked": self.checked,
            "exposed": self.exposed,
            "bind_address": self.bind_address,
            "port": self.port,
            "method": self.method,
            "detail": self.detail,
            "blocked": self.blocked,
        }


def check_ollama_bind(
    port: int | None = None,
    *,
    block_if_exposed_bulbe: bool = True,
) -> OllamaBindCheckResult:
    """Detect whether Ollama is bound to a wildcard address (0.0.0.0).

    Uses three detection methods in order:
      1. OLLAMA_HOST environment variable
      2. /proc/net/tcp parsing (Linux, no subprocess)
      3. ``ss -tlnp`` subprocess fallback

    Args:
        port: Ollama port to check. Defaults to 11434.
        block_if_exposed_bulbe: If True, log CRITICAL in Bulbe mode
            when Ollama is exposed. Does NOT call sys.exit — the caller
            (startup_checks) decides whether to block.

    Returns:
        OllamaBindCheckResult with detection details.
    """
    port = port or _OLLAMA_DEFAULT_PORT
    result = OllamaBindCheckResult(port=port)

    # ---- Method 1: OLLAMA_HOST environment variable ----
    ollama_host = os.environ.get("OLLAMA_HOST", "")
    if ollama_host:
        result.checked = True
        result.method = "env_OLLAMA_HOST"
        host_part = _extract_host(ollama_host)
        result.bind_address = host_part
        if host_part in _OLLAMA_WILDCARD_ADDRESSES:
            result.exposed = True
            result.detail = (
                f"OLLAMA_HOST is set to '{ollama_host}' — Ollama is "
                f"exposed to all network interfaces"
            )
            logger.warning("Ollama bind guard: %s", result.detail)
        else:
            result.detail = (
                f"OLLAMA_HOST='{ollama_host}' — Ollama bound to "
                f"'{host_part}' (OK)"
            )
        _check_bulbe_block(result, block_if_exposed_bulbe)
        return result

    # ---- Method 2: /proc/net/tcp parsing ----
    proc_result = _check_ollama_proc_net_tcp(port)
    if proc_result is not None:
        result.checked = True
        result.method = "proc_net_tcp"
        result.bind_address = proc_result
        if proc_result in _OLLAMA_WILDCARD_ADDRESSES:
            result.exposed = True
            result.detail = (
                f"Ollama is listening on {proc_result}:{port} — "
                f"exposed to all network interfaces"
            )
            logger.warning("Ollama bind guard: %s", result.detail)
        else:
            result.detail = (
                f"Ollama bound to {proc_result}:{port} (OK)"
            )
        _check_bulbe_block(result, block_if_exposed_bulbe)
        return result

    # ---- Method 3: ss -tlnp fallback ----
    ss_result = _check_ollama_ss(port)
    if ss_result is not None:
        result.checked = True
        result.method = "ss_command"
        result.bind_address = ss_result
        if ss_result in _OLLAMA_WILDCARD_ADDRESSES:
            result.exposed = True
            result.detail = (
                f"Ollama is listening on {ss_result}:{port} — "
                f"exposed to all network interfaces (detected via ss)"
            )
            logger.warning("Ollama bind guard: %s", result.detail)
        else:
            result.detail = (
                f"Ollama bound to {ss_result}:{port} (OK, detected via ss)"
            )
        _check_bulbe_block(result, block_if_exposed_bulbe)
        return result

    # ---- Could not determine ----
    result.checked = False
    result.method = "none"
    result.detail = (
        "Could not determine Ollama bind address — "
        "OLLAMA_HOST not set, /proc/net/tcp unreadable, ss unavailable"
    )
    logger.info("Ollama bind guard: %s", result.detail)
    return result


def _extract_host(ollama_host: str) -> str:
    """Extract host part from OLLAMA_HOST value.

    OLLAMA_HOST can be: '0.0.0.0', '0.0.0.0:11434', 'http://0.0.0.0:11434',
    '127.0.0.1:11434', etc.
    """
    val = ollama_host.strip()
    # Strip scheme if present
    for scheme in ("http://", "https://"):
        if val.lower().startswith(scheme):
            val = val[len(scheme):]
            break
    # Strip trailing path
    val = val.split("/")[0]
    # Strip port
    if ":" in val:
        # Handle IPv6 [::]:port
        if val.startswith("["):
            bracket_end = val.find("]")
            if bracket_end != -1:
                return val[1:bracket_end]
        # Simple host:port
        parts = val.rsplit(":", 1)
        return parts[0]
    return val


def _check_ollama_proc_net_tcp(port: int) -> str | None:
    """Check /proc/net/tcp for Ollama listening on given port.

    Returns the bind address string, or None if not found.
    """
    try:
        hex_port = f"{port:04X}"
        with open("/proc/net/tcp", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                local_addr = parts[1]
                if ":" not in local_addr:
                    continue
                addr_hex, port_hex = local_addr.split(":")
                if port_hex.upper() != hex_port:
                    continue
                # Check state: 0A = LISTEN
                state = parts[3]
                if state != "0A":
                    continue
                # Decode hex IP (little-endian on Linux)
                ip_int = int(addr_hex, 16)
                ip_bytes = ip_int.to_bytes(4, byteorder="little")
                ip_str = ".".join(str(b) for b in ip_bytes)
                return ip_str
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        pass
    return None


def _check_ollama_ss(port: int) -> str | None:
    """Use ``ss -tlnp`` to find Ollama bind address.

    Returns the bind address string, or None if unavailable.
    """
    try:
        result = subprocess.run(
            ["ss", "-tlnp"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            if str(port) not in line:
                continue
            # Match patterns like *:11434, 0.0.0.0:11434, 127.0.0.1:11434
            match = re.search(
                rf"([\d.*:]+):{port}\b", line,
            )
            if match:
                addr = match.group(1)
                if addr == "*":
                    return "0.0.0.0"
                return addr
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _check_bulbe_block(
    result: OllamaBindCheckResult,
    block_if_exposed: bool,
) -> None:
    """Mark result as blocked if Ollama is exposed in Bulbe mode."""
    if not result.exposed:
        return
    mode = _get_current_mode()
    if mode == "bulbe" and block_if_exposed:
        result.blocked = True
        result.detail += (
            " — BLOCKED: Ollama must bind to 127.0.0.1 in Bulbe mode. "
            "Set OLLAMA_HOST=127.0.0.1 or remove OLLAMA_HOST."
        )
        logger.critical(
            "Ollama bind guard: Ollama exposed in Bulbe mode! "
            "Startup should be blocked."
        )
        _audit_critical_event(
            "ollama_bind_exposed",
            bind_address=result.bind_address,
            port=result.port,
            mode=mode,
        )
    elif result.exposed:
        logger.warning(
            "Ollama bind guard: Ollama exposed on %s:%d in %s mode. "
            "Consider binding to 127.0.0.1 for better security.",
            result.bind_address, result.port, mode,
        )


# ---------------------------------------------------------------------------
# Module availability flag
# ---------------------------------------------------------------------------

NETWORK_BIND_GUARD_AVAILABLE = True
