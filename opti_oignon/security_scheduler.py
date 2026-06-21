#!/usr/bin/env python3
"""
Security Scheduler -- Automated red team runs and dependency monitoring (S158).

Provides SecurityScheduler singleton that runs periodic security tasks:
- Red team campaigns at configured intervals (daily/weekly)
- Dependency vulnerability audits via pip-audit
- Regression detection with alert storage
- Quiet hours enforcement

All scheduling uses threading.Timer for lightweight, non-blocking operation.
"""

__all__ = [
    "SecurityScheduler",
    "SchedulerAlert",
    "get_scheduler",
]

checkpoint_before_apply = True

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SchedulerAlert:
    """An alert raised by the scheduler."""

    alert_type: str  # "regression" | "vulnerability" | "error"
    message: str
    timestamp: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_type": self.alert_type,
            "message": self.message,
            "timestamp": self.timestamp,
            "details": self.details,
        }


class SecurityScheduler:
    """Automated security task scheduler.

    Manages periodic red team campaigns and dependency audits with
    quiet hours enforcement and regression detection.

    Parameters
    ----------
    config : SchedulerConfig or None
        Scheduler configuration. Uses defaults if None.
    clock : callable or None
        Injectable clock for testing (returns datetime). Defaults to
        datetime.now(timezone.utc).
    """

    def __init__(
        self,
        config: Any = None,
        clock: Any = None,
    ) -> None:
        if config is None:
            from opti_oignon.redteam.config import SchedulerConfig
            config = SchedulerConfig()

        self._config = config
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()

        # Dependency monitor delegate
        from opti_oignon.dep_monitor import DependencyMonitor
        self._dep_monitor = DependencyMonitor(
            severity_threshold=config.dep_severity_threshold,
            clock=self._clock,
        )

        # Timers
        self._redteam_timer: threading.Timer | None = None
        self._dep_audit_timer: threading.Timer | None = None

        # State
        self._running = False
        self._last_redteam_run: str | None = None
        self._last_redteam_result: dict[str, Any] | None = None
        self._next_redteam_run: str | None = None
        self._alerts: list[SchedulerAlert] = []
        self._run_count: int = 0

    @property
    def config(self) -> Any:
        return self._config

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start the scheduler timers."""
        if not self._config.enabled:
            logger.info("Security scheduler disabled in configuration")
            return

        with self._lock:
            if self._running:
                logger.warning("Security scheduler already running")
                return
            self._running = True

        # Schedule red team runs (skip on-deploy, that is trigger-only)
        interval = self._config.interval_seconds
        if interval > 0:
            self._schedule_redteam(interval)
            logger.info(
                "Red team scheduler started (interval=%s, %ds)",
                self._config.interval,
                interval,
            )

        # Schedule dependency audits
        dep_interval = self._config.dep_audit_interval_seconds
        if dep_interval > 0:
            self._schedule_dep_audit(dep_interval)
            logger.info(
                "Dependency audit scheduler started (interval=%s, %ds)",
                self._config.dep_audit_interval,
                dep_interval,
            )

    def stop(self) -> None:
        """Stop all scheduler timers."""
        with self._lock:
            self._running = False

        if self._redteam_timer is not None:
            self._redteam_timer.cancel()
            self._redteam_timer = None

        if self._dep_audit_timer is not None:
            self._dep_audit_timer.cancel()
            self._dep_audit_timer = None

        self._next_redteam_run = None
        logger.info("Security scheduler stopped")

    def trigger_redteam(self) -> dict[str, Any]:
        """Manually trigger a red team run (bypasses quiet hours).

        Returns
        -------
        dict
            Run result summary.
        """
        return self._execute_redteam(force=True)

    def trigger_dep_audit(self) -> dict[str, Any]:
        """Manually trigger a dependency audit.

        Returns
        -------
        dict
            Audit result summary.
        """
        return self._execute_dep_audit()

    def get_status(self) -> dict[str, Any]:
        """Return full scheduler status."""
        dep_summary = self._dep_monitor.get_summary()
        with self._lock:
            return {
                "enabled": self._config.enabled,
                "running": self._running,
                "redteam": {
                    "interval": self._config.interval,
                    "last_run": self._last_redteam_run,
                    "last_result": self._last_redteam_result,
                    "next_run": self._next_redteam_run,
                    "run_count": self._run_count,
                },
                "dep_audit": {
                    "interval": self._config.dep_audit_interval,
                    "severity_threshold": self._config.dep_severity_threshold,
                    "last_audit": dep_summary.get("last_audit"),
                    "audit_count": dep_summary.get("audit_count", 0),
                },
                "quiet_hours": {
                    "start": self._config.quiet_hours_start,
                    "end": self._config.quiet_hours_end,
                    "active": self._is_quiet_hours(),
                },
                "alerts": [a.to_dict() for a in self._alerts[-20:]],
                "alerts_total": len(self._alerts),
            }

    def get_alerts(self) -> list[dict[str, Any]]:
        """Return all alerts."""
        with self._lock:
            return [a.to_dict() for a in self._alerts]

    def clear_alerts(self) -> int:
        """Clear all alerts. Returns count cleared."""
        with self._lock:
            count = len(self._alerts)
            self._alerts.clear()
            return count

    # -- Internal scheduling --------------------------------------------------

    def _schedule_redteam(self, interval: int) -> None:
        """Schedule the next red team run."""
        now = self._clock()
        next_time = now.timestamp() + interval
        next_dt = datetime.fromtimestamp(next_time, tz=timezone.utc)

        with self._lock:
            self._next_redteam_run = next_dt.isoformat()

        self._redteam_timer = threading.Timer(interval, self._redteam_tick)
        self._redteam_timer.daemon = True
        self._redteam_timer.start()

    def _redteam_tick(self) -> None:
        """Timer callback for red team runs."""
        if not self._running:
            return

        if self._is_quiet_hours():
            logger.info("Skipping scheduled red team run (quiet hours active)")
            # Reschedule for after quiet hours end
            retry_seconds = self._seconds_until_quiet_end()
            if retry_seconds > 0:
                self._redteam_timer = threading.Timer(
                    retry_seconds, self._redteam_tick
                )
                self._redteam_timer.daemon = True
                self._redteam_timer.start()
            return

        try:
            self._execute_redteam(force=False)
        except Exception as exc:
            logger.error("Scheduled red team run failed: %s", exc)
            self._add_alert(SchedulerAlert(
                alert_type="error",
                message=f"Scheduled red team run failed: {exc}",
            ))

        # Reschedule
        interval = self._config.interval_seconds
        if interval > 0 and self._running:
            self._schedule_redteam(interval)

    def _schedule_dep_audit(self, interval: int) -> None:
        """Schedule the next dependency audit."""
        self._dep_audit_timer = threading.Timer(interval, self._dep_audit_tick)
        self._dep_audit_timer.daemon = True
        self._dep_audit_timer.start()

    def _dep_audit_tick(self) -> None:
        """Timer callback for dependency audits."""
        if not self._running:
            return

        try:
            self._execute_dep_audit()
        except Exception as exc:
            logger.error("Scheduled dependency audit failed: %s", exc)
            self._add_alert(SchedulerAlert(
                alert_type="error",
                message=f"Dependency audit failed: {exc}",
            ))

        # Reschedule
        dep_interval = self._config.dep_audit_interval_seconds
        if dep_interval > 0 and self._running:
            self._schedule_dep_audit(dep_interval)

    # -- Execution -------------------------------------------------------------

    def _execute_redteam(self, force: bool = False) -> dict[str, Any]:
        """Run a red team campaign and store results.

        Parameters
        ----------
        force : bool
            If True, bypass quiet hours check.

        Returns
        -------
        dict
            Summary of the run result.
        """
        if not force and self._is_quiet_hours():
            return {"status": "skipped", "reason": "quiet_hours"}

        now = self._clock()
        timestamp = now.isoformat()

        logger.info("Starting scheduled red team campaign")

        # Capture previous bypass rate for regression detection
        prev_bypass_rate = None
        with self._lock:
            if self._last_redteam_result:
                prev_bypass_rate = self._last_redteam_result.get(
                    "bypass_rate"
                )

        result: dict[str, Any] = {
            "status": "completed",
            "timestamp": timestamp,
            "bypass_rate": 0.0,
            "detection_rate": 0.0,
            "total_attacks": 0,
            "bypasses": 0,
        }

        try:
            from opti_oignon.redteam.config import load_redteam_config
            from opti_oignon.redteam.runner import RedTeamRunner
            from opti_oignon.redteam.scoring import aggregate_scores, score_result

            config = load_redteam_config()
            runner = RedTeamRunner(config=config)
            campaign = runner.run_campaign()

            scores = []
            for attack, strategy_name, target_result in campaign.results:
                sc = score_result(
                    target_result,
                    category=attack.category,
                    strategy=strategy_name,
                    payload_hash=attack.hash,
                    bypass_threshold=config.bypass_threshold,
                    flag_threshold=config.flag_threshold,
                )
                scores.append(sc)

            campaign_score = aggregate_scores(scores)
            score_dict = campaign_score.to_dict()

            result["bypass_rate"] = score_dict.get("overall_bypass_rate", 0.0)
            result["detection_rate"] = score_dict.get(
                "overall_detection_rate", 0.0
            )
            result["total_attacks"] = score_dict.get("total_attacks", 0)
            result["bypasses"] = score_dict.get("total_bypasses", 0)

            # Store in report store
            try:
                from opti_oignon.api.routes_security import (
                    _redteam_report_store,
                )
                global _redteam_report_counter_ref
                report_counter = len(_redteam_report_store) + 1
                report_id = f"rt-{report_counter:04d}"
                _redteam_report_store[report_id] = {
                    "id": report_id,
                    "timestamp": timestamp,
                    "campaign": campaign.to_dict(),
                    "score": score_dict,
                    "source": "scheduler",
                }
                result["report_id"] = report_id
            except Exception as exc:
                logger.warning("Failed to store scheduled report: %s", exc)

        except Exception as exc:
            result["status"] = "error"
            result["error"] = str(exc)
            logger.error("Red team campaign execution failed: %s", exc)

        with self._lock:
            self._last_redteam_run = timestamp
            self._last_redteam_result = result
            self._run_count += 1

        # Regression detection
        if (
            prev_bypass_rate is not None
            and result["status"] == "completed"
            and result["bypass_rate"] > prev_bypass_rate + 0.05
        ):
            regression_msg = (
                f"Red team regression detected: bypass rate increased "
                f"from {prev_bypass_rate:.1%} to {result['bypass_rate']:.1%}"
            )
            logger.warning(regression_msg)
            self._add_alert(SchedulerAlert(
                alert_type="regression",
                message=regression_msg,
                details={
                    "previous_bypass_rate": prev_bypass_rate,
                    "current_bypass_rate": result["bypass_rate"],
                },
            ))

        return result

    def _execute_dep_audit(self) -> dict[str, Any]:
        """Run pip-audit via DependencyMonitor and process results.

        Returns
        -------
        dict
            Audit result summary with vulnerability count.
        """
        result = self._dep_monitor.run_audit()

        # Generate alert if vulnerabilities found
        if result.filtered_count > 0:
            self._add_alert(SchedulerAlert(
                alert_type="vulnerability",
                message=(
                    f"Dependency audit found {result.filtered_count} "
                    f"vulnerabilities at or above "
                    f"{self._config.dep_severity_threshold} severity"
                ),
                details={
                    "count": result.filtered_count,
                    "threshold": self._config.dep_severity_threshold,
                    "vulnerabilities": [
                        v.vuln_id for v in result.filtered_findings[:10]
                    ],
                },
            ))

        return result.to_dict()

    # -- Quiet hours -----------------------------------------------------------

    def _is_quiet_hours(self) -> bool:
        """Check if current time is within quiet hours window."""
        now = self._clock()
        current_minutes = now.hour * 60 + now.minute

        start_h, start_m = (
            int(self._config.quiet_hours_start[:2]),
            int(self._config.quiet_hours_start[3:]),
        )
        end_h, end_m = (
            int(self._config.quiet_hours_end[:2]),
            int(self._config.quiet_hours_end[3:]),
        )

        start_minutes = start_h * 60 + start_m
        end_minutes = end_h * 60 + end_m

        if start_minutes <= end_minutes:
            # Simple range (e.g. 00:00 to 06:00)
            return start_minutes <= current_minutes < end_minutes
        else:
            # Wraps around midnight (e.g. 22:00 to 06:00)
            return current_minutes >= start_minutes or current_minutes < end_minutes

    def _seconds_until_quiet_end(self) -> int:
        """Compute seconds from now until quiet hours end."""
        now = self._clock()
        current_minutes = now.hour * 60 + now.minute

        end_h, end_m = (
            int(self._config.quiet_hours_end[:2]),
            int(self._config.quiet_hours_end[3:]),
        )
        end_minutes = end_h * 60 + end_m

        diff = end_minutes - current_minutes
        if diff <= 0:
            diff += 1440  # next day

        return diff * 60

    # -- Alerts ----------------------------------------------------------------

    def _add_alert(self, alert: SchedulerAlert) -> None:
        """Store an alert (thread-safe)."""
        with self._lock:
            self._alerts.append(alert)
            # Cap at 100 alerts
            if len(self._alerts) > 100:
                self._alerts = self._alerts[-100:]


# -- Module-level singleton ----------------------------------------------------

_scheduler: SecurityScheduler | None = None
_scheduler_lock = threading.Lock()


def get_scheduler(config: Any = None) -> SecurityScheduler:
    """Get or create the security scheduler singleton.

    Parameters
    ----------
    config : SchedulerConfig or None
        Configuration for the scheduler. Only used on first call.

    Returns
    -------
    SecurityScheduler
        The singleton instance.
    """
    global _scheduler
    with _scheduler_lock:
        if _scheduler is None:
            _scheduler = SecurityScheduler(config=config)
        return _scheduler
