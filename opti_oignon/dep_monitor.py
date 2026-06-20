#!/usr/bin/env python3
"""
Dependency Monitor -- Periodic pip-audit integration (S158).

Provides DependencyMonitor for running pip-audit, parsing JSON output,
filtering by severity threshold, and storing results with timestamps.

Designed to be called by SecurityScheduler or manually via API.
"""

__all__ = [
    "DependencyMonitor",
    "AuditResult",
    "VulnerabilityRecord",
]

checkpoint_before_apply = True

import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Ordered from lowest to highest severity
SEVERITY_ORDER = ["low", "medium", "high", "critical"]


@dataclass
class VulnerabilityRecord:
    """A single vulnerability finding from pip-audit."""

    package_name: str
    installed_version: str
    vuln_id: str
    description: str
    severity: str = "unknown"
    fix_versions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_name": self.package_name,
            "installed_version": self.installed_version,
            "vuln_id": self.vuln_id,
            "description": self.description,
            "severity": self.severity,
            "fix_versions": self.fix_versions,
        }


@dataclass
class AuditResult:
    """Result of a pip-audit run."""

    status: str  # "completed" | "parse_error" | "pip_audit_not_found" | "timeout" | "error"
    timestamp: str
    all_findings: list[VulnerabilityRecord] = field(default_factory=list)
    filtered_findings: list[VulnerabilityRecord] = field(default_factory=list)
    severity_threshold: str = "high"
    total_raw: int = 0
    error: Optional[str] = None

    @property
    def filtered_count(self) -> int:
        return len(self.filtered_findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "timestamp": self.timestamp,
            "total_raw": self.total_raw,
            "filtered_count": self.filtered_count,
            "severity_threshold": self.severity_threshold,
            "vulnerabilities": [v.to_dict() for v in self.filtered_findings],
            "error": self.error,
        }


class DependencyMonitor:
    """Runs pip-audit and parses results with severity filtering.

    Parameters
    ----------
    severity_threshold : str
        Minimum severity to include in filtered results.
        One of: low, medium, high, critical.
    timeout : int
        Maximum seconds to wait for pip-audit subprocess.
    clock : callable or None
        Injectable clock for testing.
    """

    def __init__(
        self,
        severity_threshold: str = "high",
        timeout: int = 300,
        clock: Any = None,
    ) -> None:
        if severity_threshold not in SEVERITY_ORDER:
            raise ValueError(
                f"Invalid severity threshold: {severity_threshold!r} "
                f"(valid: {SEVERITY_ORDER})"
            )
        self._severity_threshold = severity_threshold
        self._timeout = timeout
        self._clock = clock or (lambda: datetime.now(timezone.utc))

        # State: history of audit results
        self._results: list[AuditResult] = []

    @property
    def severity_threshold(self) -> str:
        return self._severity_threshold

    @property
    def last_result(self) -> Optional[AuditResult]:
        """Most recent audit result, or None."""
        return self._results[-1] if self._results else None

    @property
    def results_history(self) -> list[AuditResult]:
        """All stored audit results."""
        return list(self._results)

    @property
    def audit_count(self) -> int:
        return len(self._results)

    def run_audit(self) -> AuditResult:
        """Execute pip-audit and return parsed, filtered results.

        Returns
        -------
        AuditResult
            Parsed and severity-filtered audit result.
        """
        timestamp = self._clock().isoformat()

        try:
            proc = subprocess.run(
                ["pip-audit", "--format", "json", "--output", "-"],
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            return self._parse_output(proc.stdout, timestamp)

        except FileNotFoundError:
            result = AuditResult(
                status="pip_audit_not_found",
                timestamp=timestamp,
                severity_threshold=self._severity_threshold,
                error="pip-audit is not installed",
            )
            self._results.append(result)
            logger.warning("pip-audit not installed, skipping audit")
            return result

        except subprocess.TimeoutExpired:
            result = AuditResult(
                status="timeout",
                timestamp=timestamp,
                severity_threshold=self._severity_threshold,
                error=f"pip-audit timed out after {self._timeout}s",
            )
            self._results.append(result)
            logger.warning("pip-audit timed out after %ds", self._timeout)
            return result

        except Exception as exc:
            result = AuditResult(
                status="error",
                timestamp=timestamp,
                severity_threshold=self._severity_threshold,
                error=str(exc),
            )
            self._results.append(result)
            logger.error("Dependency audit failed: %s", exc)
            return result

    def run_audit_from_json(self, json_str: str) -> AuditResult:
        """Parse pip-audit JSON output directly (useful for testing).

        Parameters
        ----------
        json_str : str
            Raw JSON string from pip-audit.

        Returns
        -------
        AuditResult
            Parsed and severity-filtered result.
        """
        timestamp = self._clock().isoformat()
        return self._parse_output(json_str, timestamp)

    def _parse_output(self, stdout: str, timestamp: str) -> AuditResult:
        """Parse pip-audit JSON output and filter by severity.

        Parameters
        ----------
        stdout : str
            Raw stdout from pip-audit.
        timestamp : str
            ISO timestamp for this audit.

        Returns
        -------
        AuditResult
            Parsed result with severity filtering applied.
        """
        if not stdout.strip():
            result = AuditResult(
                status="completed",
                timestamp=timestamp,
                severity_threshold=self._severity_threshold,
            )
            self._results.append(result)
            return result

        try:
            raw = json.loads(stdout)
        except json.JSONDecodeError as exc:
            result = AuditResult(
                status="parse_error",
                timestamp=timestamp,
                severity_threshold=self._severity_threshold,
                error=f"JSON parse error: {exc}",
            )
            self._results.append(result)
            logger.warning("Failed to parse pip-audit JSON output")
            return result

        # pip-audit outputs either a dict with "dependencies" key or a list
        dependencies: list[dict[str, Any]] = []
        if isinstance(raw, dict):
            dependencies = raw.get("dependencies", [])
        elif isinstance(raw, list):
            dependencies = raw

        all_findings: list[VulnerabilityRecord] = []
        for dep in dependencies:
            pkg_name = dep.get("name", "")
            pkg_version = dep.get("version", "")
            vulns = dep.get("vulns", [])
            for vuln in vulns:
                record = VulnerabilityRecord(
                    package_name=pkg_name,
                    installed_version=pkg_version,
                    vuln_id=vuln.get("id", ""),
                    description=vuln.get("description", ""),
                    severity=vuln.get("severity", "unknown"),
                    fix_versions=vuln.get("fix_versions", []),
                )
                all_findings.append(record)

        # Filter by severity threshold
        threshold_idx = SEVERITY_ORDER.index(self._severity_threshold)
        filtered = []
        for record in all_findings:
            sev_lower = record.severity.lower()
            if sev_lower in SEVERITY_ORDER:
                if SEVERITY_ORDER.index(sev_lower) >= threshold_idx:
                    filtered.append(record)
            elif sev_lower == "unknown":
                # Include unknown severity (conservative approach)
                filtered.append(record)

        result = AuditResult(
            status="completed",
            timestamp=timestamp,
            all_findings=all_findings,
            filtered_findings=filtered,
            severity_threshold=self._severity_threshold,
            total_raw=len(all_findings),
        )
        self._results.append(result)

        if filtered:
            logger.warning(
                "Dependency audit found %d vulnerabilities at or above %s",
                len(filtered),
                self._severity_threshold,
            )
        else:
            logger.info("Dependency audit clean (threshold: %s)", self._severity_threshold)

        return result

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of the monitor state.

        Returns
        -------
        dict
            Summary with last result and history count.
        """
        last = self.last_result
        return {
            "audit_count": self.audit_count,
            "severity_threshold": self._severity_threshold,
            "last_audit": last.to_dict() if last else None,
        }
