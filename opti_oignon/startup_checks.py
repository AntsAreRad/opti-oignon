#!/usr/bin/env python3
"""
Startup Security Checklist for Opti-Oignon (S145).

Unified module that runs all security guards at application startup
and caches results. Combines:
  - Code signing verification (scripts exist)
  - Ollama bind guard (0.0.0.0 detection)
  - LUKS full-disk encryption detection
  - Existing security score checks
  - Resource governor Ollama limits advisory (R-03, S226)

Results are cached for the lifetime of the process and exposed via
``GET /api/security/startup-checks``.

This module is the single entry point for all startup-time security
verification. Individual guards remain independently testable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_SIGN_SCRIPT = _SCRIPTS_DIR / "sign_release.sh"
_VERIFY_SCRIPT = _SCRIPTS_DIR / "verify_release.sh"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CheckItem:
    """A single check result in the startup checklist."""

    name: str
    passed: bool
    severity: str  # "critical", "warning", "info"
    detail: str
    score_impact: int = 0  # negative = deduction from max score
    tips: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API response."""
        d: dict[str, Any] = {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "detail": self.detail,
            "score_impact": self.score_impact,
        }
        if self.tips:
            d["tips"] = list(self.tips)
        return d


@dataclass
class StartupCheckResult:
    """Aggregated result of all startup security checks."""

    timestamp: float = 0.0
    checks: list[CheckItem] = field(default_factory=list)
    all_passed: bool = False
    blocked: bool = False
    block_reason: str = ""
    total_score_impact: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API response."""
        return {
            "timestamp": self.timestamp,
            "checks": [c.to_dict() for c in self.checks],
            "all_passed": self.all_passed,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "total_score_impact": self.total_score_impact,
            "check_count": len(self.checks),
            "passed_count": sum(1 for c in self.checks if c.passed),
            "failed_count": sum(1 for c in self.checks if not c.passed),
        }


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_cached_result: StartupCheckResult | None = None


def get_cached_result() -> StartupCheckResult | None:
    """Return the cached startup check result, or None if not yet run."""
    return _cached_result


def clear_cache() -> None:
    """Clear the cached result (for testing or re-run)."""
    global _cached_result
    _cached_result = None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_startup_checks(*, force: bool = False) -> StartupCheckResult:
    """Run all startup security checks and cache the result.

    Args:
        force: If True, re-run even if already cached.

    Returns:
        StartupCheckResult with all check details.
    """
    global _cached_result

    if _cached_result is not None and not force:
        return _cached_result

    result = StartupCheckResult(timestamp=time.time())

    # --- Check 1: Code signing scripts ---
    result.checks.append(_check_code_signing_scripts())

    # --- Check 2: Ollama bind guard ---
    result.checks.append(_check_ollama_bind())

    # --- Check 3: LUKS encryption ---
    result.checks.append(_check_luks())

    # --- Check 4: Security mode ---
    result.checks.append(_check_security_mode())

    # --- Check 5: Encrypted swap ---
    result.checks.append(_check_encrypted_swap())

    # --- Check 6: Resource governor Ollama limits advisory (S226) ---
    result.checks.append(_check_governor_ollama_limits())

    # --- Aggregate ---
    result.all_passed = all(c.passed for c in result.checks)
    result.total_score_impact = sum(c.score_impact for c in result.checks)
    result.blocked = any(
        not c.passed and c.severity == "critical" for c in result.checks
    )
    if result.blocked:
        critical_failures = [
            c.name for c in result.checks
            if not c.passed and c.severity == "critical"
        ]
        result.block_reason = (
            f"Critical check(s) failed: {', '.join(critical_failures)}"
        )
        logger.critical("Startup blocked: %s", result.block_reason)
    elif not result.all_passed:
        warnings = [c.name for c in result.checks if not c.passed]
        logger.warning(
            "Startup checks: %d warning(s): %s",
            len(warnings), ", ".join(warnings),
        )
    else:
        logger.info("All startup security checks passed")

    _cached_result = result
    return result


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_code_signing_scripts() -> CheckItem:
    """Verify that code signing scripts exist and are executable."""
    missing: list[str] = []
    for script, label in [
        (_SIGN_SCRIPT, "sign_release.sh"),
        (_VERIFY_SCRIPT, "verify_release.sh"),
    ]:
        if not script.exists():
            missing.append(label)
        elif not script.stat().st_mode & 0o111:
            missing.append(f"{label} (not executable)")

    if not missing:
        return CheckItem(
            name="code_signing_scripts",
            passed=True,
            severity="info",
            detail="Release signing scripts present and executable",
            score_impact=0,
        )
    return CheckItem(
        name="code_signing_scripts",
        passed=False,
        severity="info",
        detail=f"Missing or non-executable: {', '.join(missing)}",
        score_impact=-2,
        tips=[
            "Run: chmod +x scripts/sign_release.sh scripts/verify_release.sh",
            "Code signing ensures release integrity — see SECURITY.md",
        ],
    )


def _check_ollama_bind() -> CheckItem:
    """Check if Ollama is exposed on a wildcard address."""
    try:
        from opti_oignon.network_bind_guard import check_ollama_bind

        result = check_ollama_bind()

        if not result.checked:
            return CheckItem(
                name="ollama_bind",
                passed=True,
                severity="info",
                detail=result.detail,
                score_impact=0,
            )

        if result.blocked:
            return CheckItem(
                name="ollama_bind",
                passed=False,
                severity="critical",
                detail=result.detail,
                score_impact=-15,
                tips=[
                    "Set OLLAMA_HOST=127.0.0.1 before starting Ollama",
                    "Or remove OLLAMA_HOST to use the default (localhost)",
                    "Ollama must NOT be accessible from the network in Bulbe mode",
                ],
            )

        if result.exposed:
            return CheckItem(
                name="ollama_bind",
                passed=False,
                severity="warning",
                detail=result.detail,
                score_impact=-10,
                tips=[
                    "Ollama is listening on all interfaces — any device on "
                    "your network can send prompts to your LLM",
                    "Set OLLAMA_HOST=127.0.0.1 to restrict to localhost",
                ],
            )

        return CheckItem(
            name="ollama_bind",
            passed=True,
            severity="info",
            detail=result.detail,
            score_impact=0,
        )
    except Exception as exc:
        logger.warning("Ollama bind check failed: %s", exc)
        return CheckItem(
            name="ollama_bind",
            passed=True,
            severity="info",
            detail=f"Ollama bind check unavailable: {exc}",
            score_impact=0,
        )


def _check_luks() -> CheckItem:
    """Check for full-disk encryption."""
    try:
        from opti_oignon.luks_detector import check_luks_encryption

        result = check_luks_encryption()

        if not result.checked:
            return CheckItem(
                name="luks_encryption",
                passed=False,
                severity="warning",
                detail=result.detail,
                score_impact=-5,
                tips=result.tips,
            )

        if result.encrypted:
            return CheckItem(
                name="luks_encryption",
                passed=True,
                severity="info",
                detail=result.detail,
                score_impact=0,
            )

        return CheckItem(
            name="luks_encryption",
            passed=False,
            severity="warning",
            detail=result.detail,
            score_impact=-5,
            tips=result.tips,
        )
    except Exception as exc:
        logger.warning("LUKS check failed: %s", exc)
        return CheckItem(
            name="luks_encryption",
            passed=False,
            severity="warning",
            detail=f"LUKS detection unavailable: {exc}",
            score_impact=-5,
            tips=[
                "Install lsblk for reliable encryption detection",
            ],
        )


def _check_security_mode() -> CheckItem:
    """Verify that a security mode is explicitly configured."""
    try:
        from opti_oignon.security_mode import get_current_mode
        mode = get_current_mode()
        return CheckItem(
            name="security_mode",
            passed=True,
            severity="info",
            detail=f"Security mode: {mode}",
            score_impact=0,
        )
    except Exception:
        return CheckItem(
            name="security_mode",
            passed=True,
            severity="info",
            detail="Security mode defaults to 'bulbe' (fail-secure)",
            score_impact=0,
        )


def _check_encrypted_swap() -> CheckItem:
    """Check if swap is encrypted (advisory)."""
    try:
        swap_path = Path("/proc/swaps")
        if not swap_path.exists():
            return CheckItem(
                name="encrypted_swap",
                passed=True,
                severity="info",
                detail="No swap partitions detected",
                score_impact=0,
            )

        with open(swap_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # First line is header
        swap_entries = [l.strip() for l in lines[1:] if l.strip()]

        if not swap_entries:
            return CheckItem(
                name="encrypted_swap",
                passed=True,
                severity="info",
                detail="No active swap partitions",
                score_impact=0,
            )

        # Check if swap devices are on dm-crypt
        has_unencrypted = False
        for entry in swap_entries:
            device = entry.split()[0] if entry.split() else ""
            if "/dm-" not in device and "/mapper/" not in device:
                has_unencrypted = True
                break

        if has_unencrypted:
            return CheckItem(
                name="encrypted_swap",
                passed=False,
                severity="warning",
                detail="Swap partition may not be encrypted — RAM contents "
                       "could leak to disk",
                score_impact=-3,
                tips=[
                    "Use encrypted swap or disable swap: sudo swapoff -a",
                    "Sensitive data in RAM (keys, conversations) can be "
                    "written to unencrypted swap by the kernel",
                ],
            )

        return CheckItem(
            name="encrypted_swap",
            passed=True,
            severity="info",
            detail="Swap appears to be on an encrypted device",
            score_impact=0,
        )
    except (OSError, PermissionError) as exc:
        return CheckItem(
            name="encrypted_swap",
            passed=True,
            severity="info",
            detail=f"Swap check unavailable: {exc}",
            score_impact=0,
        )


# ---------------------------------------------------------------------------
# Module availability flag
# ---------------------------------------------------------------------------

def _check_governor_ollama_limits() -> CheckItem:
    """R-03 external-Ollama limits advisory (S226, spec Section 6).

    ADVISORY-ONLY in all modes, never blocking startup: the S145
    LUKS-detector precedent applied verbatim -- this check NEVER
    returns severity "critical", so it can never set the blocked
    flag. Honoured config switches: ``ollama_limits.external_advisory``
    disables the advisory; an "unknown" observation (the documented
    systemd case: OLLAMA_* not visible from this process) is the
    normal external posture and reports as a passing info line with
    actionable tips, never a failure. Only a visible MISMATCH between
    the configured limits and the environment surfaces as a warning.
    Never raises (fail-open on any import or computation error).
    """
    try:
        from opti_oignon.resource_governor import (
            compute_ollama_limits_advisory,
            load_config,
        )

        advisory = compute_ollama_limits_advisory(load_config())

        if not advisory.get("external_advisory", True):
            return CheckItem(
                name="governor_ollama_limits",
                passed=True,
                severity="info",
                detail=(
                    "Ollama limits advisory disabled by config "
                    "(ollama_limits.external_advisory: false)"
                ),
                score_impact=0,
            )

        status = advisory.get("status")
        detail = str(advisory.get("detail", ""))

        if status == "mismatch":
            return CheckItem(
                name="governor_ollama_limits",
                passed=False,
                severity="warning",
                detail=detail,
                score_impact=-3,
                tips=[
                    "Align the environment with resource_governor.yaml: "
                    "set the OLLAMA_* variables where the Ollama server "
                    "starts (for systemd, an Environment= line in the "
                    "ollama unit drop-in)",
                    "Or update ollama_limits in "
                    "config/resource_governor.yaml to match the running "
                    "server",
                    "See scripts/ollama_cgroup_limits.sh for the unit "
                    "drop-in recipe",
                ],
            )

        if status == "unknown":
            return CheckItem(
                name="governor_ollama_limits",
                passed=True,
                severity="info",
                detail=detail,
                score_impact=0,
                tips=[
                    "Configured limits only apply where the Ollama "
                    "server reads them: set OLLAMA_MAX_LOADED_MODELS / "
                    "OLLAMA_NUM_PARALLEL / OLLAMA_MAX_QUEUE in the "
                    "server's environment (systemd unit drop-in)",
                    "See scripts/ollama_cgroup_limits.sh for the recipe",
                ],
            )

        # "match" and "not_configured" both pass clean.
        return CheckItem(
            name="governor_ollama_limits",
            passed=True,
            severity="info",
            detail=detail or "Ollama limits advisory unavailable",
            score_impact=0,
        )
    except Exception as exc:
        logger.warning("Governor Ollama limits advisory failed: %s", exc)
        return CheckItem(
            name="governor_ollama_limits",
            passed=True,
            severity="info",
            detail=f"Ollama limits advisory unavailable: {exc}",
            score_impact=0,
        )


# ---------------------------------------------------------------------------
# Module availability flag (sentinel)
# ---------------------------------------------------------------------------

STARTUP_CHECKS_AVAILABLE = True
