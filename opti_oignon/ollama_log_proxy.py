#!/usr/bin/env python3
"""
Ollama Log Sanitization Proxy for Opti-Oignon (S131).

Ollama logs prompt content at ``debug`` level, which may include
sensitive user messages, API keys, or PII.  This module:

  1. Detects the current Ollama logging configuration.
  2. Provides recommended environment variables per security mode.
  3. Sanitizes log lines to strip potential PII/secrets.
  4. Surfaces warnings in the health/security status endpoints when
     Ollama logging is unnecessarily verbose.

This module does **not** modify the Ollama process directly -- it
provides advisory recommendations and a sanitization filter that can
be applied to captured log output.

Configuration (security.yaml)
------------------------------

.. code-block:: yaml

   ollama:
     log_sanitization: true
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OllamaLogConfig:
    """Current Ollama logging configuration and recommendations."""
    log_level: str = "unknown"
    debug_enabled: bool = False
    sanitization_enabled: bool = True
    keep_alive: str = "unknown"
    is_verbose: bool = False
    recommendations: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def _detect_ollama_env() -> dict[str, str]:
    """Read Ollama-related environment variables from the current process
    and, if possible, from the running Ollama server process.

    Returns a dict of env var name -> value.
    """
    env_vars = {}

    # Check current process environment (Ollama inherits from shell)
    for key in (
        "OLLAMA_LOG_LEVEL", "OLLAMA_DEBUG", "OLLAMA_KEEP_ALIVE",
        "OLLAMA_HOST", "OLLAMA_MODELS", "OLLAMA_TMPDIR",
    ):
        val = os.environ.get(key)
        if val is not None:
            env_vars[key] = val

    # Try to read from the Ollama process via /proc (Linux only)
    if os.path.isdir("/proc"):
        try:
            result = subprocess.run(
                ["pgrep", "-f", "ollama serve"],
                capture_output=True, text=True, timeout=5,
            )
            pids = result.stdout.strip().split()
            for pid in pids[:1]:  # Only check the first match
                environ_path = f"/proc/{pid}/environ"
                if os.path.isfile(environ_path):
                    try:
                        with open(environ_path, "r", encoding="utf-8",
                                  errors="replace") as fh:
                            raw = fh.read()
                        for entry in raw.split("\x00"):
                            if "=" in entry and entry.startswith("OLLAMA_"):
                                k, v = entry.split("=", 1)
                                env_vars[k] = v
                    except PermissionError:
                        # Common: cannot read other user's /proc/PID/environ
                        pass
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return env_vars


def check_ollama_log_config() -> OllamaLogConfig:
    """Detect current Ollama logging configuration.

    Reads environment variables and infers the effective log level.
    Returns an ``OllamaLogConfig`` with recommendations.
    """
    config = OllamaLogConfig()

    # Load sanitization setting from security.yaml
    try:
        import yaml
        yaml_path = os.path.join(
            os.path.dirname(__file__), "config", "security.yaml"
        )
        if os.path.isfile(yaml_path):
            with open(yaml_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            ollama_cfg = data.get("ollama", {})
            config.sanitization_enabled = ollama_cfg.get(
                "log_sanitization", True
            )
    except Exception:
        pass

    env = _detect_ollama_env()

    # Log level
    log_level = env.get("OLLAMA_LOG_LEVEL", "").lower()
    if log_level:
        config.log_level = log_level
    else:
        # Ollama defaults to info if not set
        config.log_level = "info (default)"

    # Debug flag
    debug_val = env.get("OLLAMA_DEBUG", "")
    config.debug_enabled = debug_val.lower() in ("1", "true", "yes")
    if config.debug_enabled:
        config.log_level = "debug"

    # Keep alive
    config.keep_alive = env.get("OLLAMA_KEEP_ALIVE", "5m (default)")

    # Verbosity assessment
    config.is_verbose = config.log_level in ("debug", "trace") or config.debug_enabled

    if config.is_verbose:
        config.warnings.append(
            "Ollama is running with verbose logging. "
            "Prompt content may appear in logs."
        )

    return config


def get_ollama_env_recommendations(mode: str = "daily") -> dict[str, str]:
    """Return recommended Ollama environment variables for the given
    security mode.

    Parameters
    ----------
    mode : str
        Either ``"daily"`` or ``"bulbe"``.

    Returns
    -------
    dict[str, str]
        Mapping of env var name to recommended value.
    """
    if mode.lower() == "bulbe":
        return {
            "OLLAMA_LOG_LEVEL": "error",
            "OLLAMA_DEBUG": "0",
            "OLLAMA_KEEP_ALIVE": "0",
            "OLLAMA_TMPDIR": "/tmp/ollama-secure",
        }
    else:
        # Daily mode: moderate logging
        return {
            "OLLAMA_LOG_LEVEL": "warn",
            "OLLAMA_DEBUG": "0",
            "OLLAMA_KEEP_ALIVE": "5m",
        }


# ---------------------------------------------------------------------------
# Log sanitization
# ---------------------------------------------------------------------------

# Patterns that likely contain sensitive content
_SENSITIVE_PATTERNS = [
    # API keys and tokens
    re.compile(r"(api[_-]?key|token|secret|password|auth)\s*[:=]\s*\S+", re.IGNORECASE),
    # Email addresses
    re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    # Credit card-like numbers (13-19 digits)
    re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{1,7}\b"),
    # SSN-like patterns
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # Phone numbers (various formats)
    re.compile(r"\b\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b"),
    # Long base64 strings (potential keys/tokens, >= 32 chars)
    re.compile(r"\b[A-Za-z0-9+/]{32,}={0,2}\b"),
    # Bearer tokens
    re.compile(r"Bearer\s+\S+", re.IGNORECASE),
]

# Prompt content markers in Ollama logs
_PROMPT_MARKERS = [
    re.compile(r'"prompt"\s*:\s*"[^"]*"', re.IGNORECASE),
    re.compile(r'"content"\s*:\s*"[^"]*"', re.IGNORECASE),
    re.compile(r'"system"\s*:\s*"[^"]*"', re.IGNORECASE),
]

_REDACTION = "[REDACTED]"


def sanitize_ollama_prompt_log(text: str) -> str:
    """Strip PII, secrets, and prompt content from Ollama log lines.

    Parameters
    ----------
    text : str
        Raw log line or block from Ollama.

    Returns
    -------
    str
        Sanitized text with sensitive content replaced by ``[REDACTED]``.
    """
    if not text:
        return text

    result = text

    # Redact prompt content fields
    for pattern in _PROMPT_MARKERS:
        result = pattern.sub(_REDACTION, result)

    # Redact sensitive patterns
    for pattern in _SENSITIVE_PATTERNS:
        result = pattern.sub(_REDACTION, result)

    return result


# ---------------------------------------------------------------------------
# Module-level feature flag
# ---------------------------------------------------------------------------

OLLAMA_LOG_PROXY_AVAILABLE = True
