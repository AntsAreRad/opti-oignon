#!/usr/bin/env python3
"""
Network Hardening Checks for Opti-Oignon.

Provides advisory checks for network security posture:

  1. **DNS encryption** -- detect DoH/DoT via systemd-resolved or
     resolv.conf configuration.
  2. **Proxy configuration** -- verify SOCKS5 proxy availability
     (e.g. Tor) for Bulbe mode with web search.
  3. **Listening ports** -- report unexpected open ports on the host
     that could expose the application.

All checks are **advisory** (non-blocking) except: in Bulbe mode with
web search re-enabled, the SOCKS5 proxy is enforced for search requests.

Configuration (security.yaml)
------------------------------

.. code-block:: yaml

   network:
     socks_proxy: "socks5://127.0.0.1:9050"
"""

from __future__ import annotations

import logging
import os
import re
import socket
import subprocess
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DnsCheckResult:
    """Result of DNS encryption detection."""
    encrypted: bool = False
    protocol: str = "unknown"  # "doh", "dot", "plain", "unknown"
    resolver: str = ""
    details: str = ""
    error: str = ""


@dataclass
class ProxyCheckResult:
    """Result of SOCKS5 proxy check."""
    configured: bool = False
    proxy_url: str = ""
    reachable: bool = False
    error: str = ""


@dataclass
class PortInfo:
    """Information about an open listening port."""
    port: int = 0
    protocol: str = "tcp"
    address: str = ""
    process: str = ""
    expected: bool = True


# ---------------------------------------------------------------------------
# DNS Encryption Check
# ---------------------------------------------------------------------------

def check_dns_encryption() -> DnsCheckResult:
    """Detect whether DNS queries are encrypted (DoH or DoT).

    Checks, in order:
      1. systemd-resolved status (``resolvectl status``)
      2. ``/etc/resolv.conf`` for known encrypted resolvers
      3. NetworkManager DNS configuration

    Returns
    -------
    DnsCheckResult
        Detection result with protocol and resolver info.
    """
    result = DnsCheckResult()

    # Strategy 1: systemd-resolved
    try:
        proc = subprocess.run(
            ["resolvectl", "status"],
            capture_output=True, text=True, timeout=5,
        )
        output = proc.stdout.lower()
        if "dns over tls" in output and "yes" in output:
            result.encrypted = True
            result.protocol = "dot"
            result.resolver = "systemd-resolved"
            result.details = "DNS-over-TLS enabled via systemd-resolved"
            return result
        if "dnsovertls" in output.replace(" ", ""):
            result.encrypted = True
            result.protocol = "dot"
            result.resolver = "systemd-resolved"
            result.details = "DNSOverTLS detected in resolved config"
            return result
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Strategy 2: Check /etc/resolv.conf
    try:
        if os.path.isfile("/etc/resolv.conf"):
            with open("/etc/resolv.conf", encoding="utf-8") as fh:
                resolv = fh.read()
            # Known DoH/DoT stub resolvers
            stub_resolvers = {
                "127.0.0.53": "systemd-resolved (stub)",
                "127.0.0.1": "local resolver (possibly encrypted)",
            }
            for line in resolv.split("\n"):
                line = line.strip()
                if line.startswith("nameserver"):
                    parts = line.split()
                    if len(parts) >= 2:
                        ns = parts[1]
                        result.resolver = ns
                        if ns in stub_resolvers:
                            result.details = stub_resolvers[ns]
                            # Stub resolver likely encrypts, but cannot confirm
                            result.protocol = "stub"
                            break
    except PermissionError:
        result.error = "Cannot read /etc/resolv.conf"

    # Strategy 3: Check for known DNS-over-HTTPS services
    try:
        proc = subprocess.run(
            ["systemctl", "is-active", "dnscrypt-proxy"],
            capture_output=True, text=True, timeout=5,
        )
        if proc.stdout.strip() == "active":
            result.encrypted = True
            result.protocol = "doh"
            result.resolver = "dnscrypt-proxy"
            result.details = "dnscrypt-proxy is active"
            return result
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Strategy 4: Check for stubby (DoT client)
    try:
        proc = subprocess.run(
            ["systemctl", "is-active", "stubby"],
            capture_output=True, text=True, timeout=5,
        )
        if proc.stdout.strip() == "active":
            result.encrypted = True
            result.protocol = "dot"
            result.resolver = "stubby"
            result.details = "Stubby DoT resolver is active"
            return result
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if not result.encrypted and not result.error:
        result.details = "No DNS encryption detected (plain DNS)"
        result.protocol = "plain"

    return result


# ---------------------------------------------------------------------------
# Proxy Configuration Check
# ---------------------------------------------------------------------------

def _load_proxy_config() -> str:
    """Load SOCKS proxy URL from security.yaml."""
    try:
        import yaml
        config_path = os.path.join(
            os.path.dirname(__file__), "config", "security.yaml"
        )
        if os.path.isfile(config_path):
            with open(config_path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            return data.get("network", {}).get("socks_proxy", "")
    except Exception:
        pass
    return ""


def check_proxy_config() -> ProxyCheckResult:
    """Check SOCKS5 proxy configuration and reachability.

    Reads the proxy URL from ``security.yaml > network > socks_proxy``
    and attempts a TCP connection to verify it is reachable.

    Returns
    -------
    ProxyCheckResult
        Proxy configuration and connectivity status.
    """
    result = ProxyCheckResult()

    proxy_url = _load_proxy_config()
    if not proxy_url:
        result.configured = False
        result.error = "No SOCKS proxy configured in security.yaml"
        return result

    result.configured = True
    result.proxy_url = proxy_url

    # Parse proxy URL
    try:
        parsed = urlparse(proxy_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 9050
    except Exception as exc:
        result.error = f"Invalid proxy URL: {exc}"
        return result

    # TCP connect test
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        sock.connect((host, port))
        sock.close()
        result.reachable = True
    except OSError as exc:
        result.reachable = False
        result.error = f"Proxy unreachable at {host}:{port}: {exc}"

    return result


# ---------------------------------------------------------------------------
# Listening Ports Check
# ---------------------------------------------------------------------------

# Ports expected for Opti-Oignon operation
_EXPECTED_PORTS = {
    8000: "Opti-Oignon API",
    5173: "SvelteKit dev server",
    11434: "Ollama",
}


def check_listening_ports() -> list[PortInfo]:
    """Report TCP listening ports on the host.

    Uses ``ss -tlnp`` (Linux) to enumerate listening sockets.
    Each port is flagged as expected or unexpected based on
    known Opti-Oignon services.

    Returns
    -------
    list[PortInfo]
        List of detected listening ports.
    """
    ports: list[PortInfo] = []

    try:
        proc = subprocess.run(
            ["ss", "-tlnp"],
            capture_output=True, text=True, timeout=5,
        )
        lines = proc.stdout.strip().split("\n")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # ss not available; try netstat
        try:
            proc = subprocess.run(
                ["netstat", "-tlnp"],
                capture_output=True, text=True, timeout=5,
            )
            lines = proc.stdout.strip().split("\n")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return ports

    # Parse ss/netstat output
    # ss format: State  Recv-Q  Send-Q  Local Address:Port  Peer Address:Port  Process
    for line in lines[1:]:  # Skip header
        parts = line.split()
        if len(parts) < 5:
            continue

        # Extract local address:port
        local = parts[3] if parts[0] in ("LISTEN", "State") else parts[3]
        # Handle ss format where state is first column
        if parts[0] == "LISTEN":
            local = parts[3]
        elif len(parts) >= 4:
            # Try to find address:port pattern
            for p in parts:
                if re.match(r".*:\d+$", p):
                    local = p
                    break

        # Parse port
        match = re.search(r":(\d+)$", local)
        if not match:
            continue

        port_num = int(match.group(1))
        addr = local[:match.start()]

        # Extract process info
        process = ""
        for p in parts:
            if "users:" in p:
                proc_match = re.search(r'"([^"]+)"', p)
                if proc_match:
                    process = proc_match.group(1)
                break

        info = PortInfo(
            port=port_num,
            protocol="tcp",
            address=addr,
            process=process,
            expected=port_num in _EXPECTED_PORTS,
        )
        ports.append(info)

    return ports


# ---------------------------------------------------------------------------
# Combined Status
# ---------------------------------------------------------------------------

def get_full_network_status() -> dict[str, Any]:
    """Return combined network hardening status for the API.

    Returns
    -------
    dict[str, Any]
        Dictionary with dns, proxy, ports, and overall assessment.
    """
    dns = check_dns_encryption()
    proxy = check_proxy_config()
    ports = check_listening_ports()

    unexpected_ports = [p for p in ports if not p.expected]

    # Overall assessment
    warnings: list[str] = []
    if not dns.encrypted and dns.protocol == "plain":
        warnings.append(
            "DNS queries are not encrypted. Consider enabling DoH or DoT."
        )
    if not proxy.configured:
        warnings.append(
            "No SOCKS proxy configured. Recommended for Bulbe mode with web search."
        )
    elif not proxy.reachable:
        warnings.append(
            f"SOCKS proxy configured but unreachable: {proxy.error}"
        )
    if unexpected_ports:
        port_list = ", ".join(
            f"{p.port} ({p.process or 'unknown'})" for p in unexpected_ports
        )
        warnings.append(f"Unexpected listening ports: {port_list}")

    return {
        "available": True,
        "dns": {
            "encrypted": dns.encrypted,
            "protocol": dns.protocol,
            "resolver": dns.resolver,
            "details": dns.details,
        },
        "proxy": {
            "configured": proxy.configured,
            "proxy_url": proxy.proxy_url,
            "reachable": proxy.reachable,
            "error": proxy.error,
        },
        "ports": {
            "total": len(ports),
            "unexpected": len(unexpected_ports),
            "details": [
                {
                    "port": p.port,
                    "address": p.address,
                    "process": p.process,
                    "expected": p.expected,
                }
                for p in ports
            ],
        },
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# API endpoint addition in routes_security.py
# ---------------------------------------------------------------------------

def get_network_hardening_endpoint_data() -> dict[str, Any]:
    """Convenience wrapper for the GET /api/security/hardening/network
    endpoint.  Returns the same data as ``get_full_network_status()``.
    """
    return get_full_network_status()


# ---------------------------------------------------------------------------
# Module-level feature flag
# ---------------------------------------------------------------------------

NETWORK_HARDENING_AVAILABLE = True
