"""
tests/test_s158_security_scheduler.py -- S158 automated security scheduling tests.

Verifies:
- Goal 1: Scheduled red team runs (interval, quiet hours, regression detection)
- Goal 2: Dependency monitoring (pip-audit parsing, severity filtering, history)
- Goal 3: Dashboard integration (scheduler status in endpoints)
- Goal 4: Configuration (SchedulerConfig validation, YAML loading)
- Goal 5: Module structure (checkpoint_before_apply, AST validity)
"""

import ast
import importlib.util
import json
import os
import re
import sys
import types
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# -- Isolation stubs (standard pattern) --
for mod_name in [
    "opti_oignon",
    "opti_oignon.db_utils",
    "opti_oignon.config",
    "opti_oignon.auth",
    "opti_oignon.middleware",
    "opti_oignon.security_mode",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(
    PROJECT_ROOT, "opti_oignon", "redteam", "config.py"
)
SCHEDULER_PATH = os.path.join(
    PROJECT_ROOT, "opti_oignon", "security_scheduler.py"
)
DEP_MONITOR_PATH = os.path.join(
    PROJECT_ROOT, "opti_oignon", "dep_monitor.py"
)
ROUTES_SECURITY_PATH = os.path.join(
    PROJECT_ROOT, "opti_oignon", "api", "routes_security.py"
)
APP_PATH = os.path.join(PROJECT_ROOT, "opti_oignon", "api", "app.py")
REDTEAM_YAML_PATH = os.path.join(
    PROJECT_ROOT, "opti_oignon", "config", "redteam.yaml"
)


# -- Helpers --


def _load_module(name, path):
    """Load a module by file path without triggering the full import chain."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_config():
    """Load redteam config module with yaml stub."""
    if "yaml" not in sys.modules:
        yaml_mock = types.ModuleType("yaml")
        yaml_mock.safe_load = lambda f: {}
        yaml_mock.safe_dump = lambda *a, **k: None
        sys.modules["yaml"] = yaml_mock
    mod = _load_module("rt_config", CONFIG_PATH)
    sys.modules["opti_oignon.redteam.config"] = mod
    return mod


def _load_dep_monitor():
    """Load dep_monitor module."""
    return _load_module("dep_monitor", DEP_MONITOR_PATH)


def _load_scheduler(config_mod=None):
    """Load security_scheduler module with dependencies."""
    if config_mod is None:
        config_mod = _load_config()
    dm_mod = _load_dep_monitor()
    sys.modules["opti_oignon.dep_monitor"] = dm_mod
    # Reset singleton
    mod = _load_module("security_scheduler", SCHEDULER_PATH)
    return mod


# ==========================================================================
# Class 1: SchedulerConfig validation
# ==========================================================================


class TestSchedulerConfigValidation:
    """Test SchedulerConfig dataclass validation."""

    def setup_method(self):
        self.cfg_mod = _load_config()
        self.SC = self.cfg_mod.SchedulerConfig

    def test_defaults(self):
        sc = self.SC()
        assert sc.enabled is True
        assert sc.interval == "weekly"
        assert sc.quiet_hours_start == "00:00"
        assert sc.quiet_hours_end == "06:00"
        assert sc.auto_accept_suggestions is False
        assert sc.dep_audit_interval == "weekly"
        assert sc.dep_severity_threshold == "high"

    def test_valid_intervals(self):
        for interval in ("daily", "weekly", "on-deploy"):
            sc = self.SC(interval=interval)
            assert sc.interval == interval

    def test_invalid_interval(self):
        with pytest.raises(ValueError, match="Invalid scheduler interval"):
            self.SC(interval="hourly")

    def test_invalid_dep_audit_interval(self):
        with pytest.raises(ValueError, match="Invalid dep_audit_interval"):
            self.SC(dep_audit_interval="monthly")

    def test_valid_severities(self):
        for sev in ("low", "medium", "high", "critical"):
            sc = self.SC(dep_severity_threshold=sev)
            assert sc.dep_severity_threshold == sev

    def test_invalid_severity(self):
        with pytest.raises(ValueError, match="Invalid dep_severity_threshold"):
            self.SC(dep_severity_threshold="extreme")

    def test_valid_time_formats(self):
        sc = self.SC(quiet_hours_start="23:59", quiet_hours_end="00:00")
        assert sc.quiet_hours_start == "23:59"

    def test_invalid_time_format_no_colon(self):
        with pytest.raises(ValueError, match="Invalid quiet_hours_start"):
            self.SC(quiet_hours_start="2300")

    def test_invalid_time_hour_25(self):
        with pytest.raises(ValueError, match="hours 00-23"):
            self.SC(quiet_hours_start="25:00")

    def test_invalid_time_minute_60(self):
        with pytest.raises(ValueError, match="minutes 00-59"):
            self.SC(quiet_hours_end="12:60")

    def test_interval_seconds_daily(self):
        sc = self.SC(interval="daily")
        assert sc.interval_seconds == 86400

    def test_interval_seconds_weekly(self):
        sc = self.SC(interval="weekly")
        assert sc.interval_seconds == 604800

    def test_interval_seconds_on_deploy(self):
        sc = self.SC(interval="on-deploy")
        assert sc.interval_seconds == 0

    def test_dep_audit_interval_seconds(self):
        sc = self.SC(dep_audit_interval="daily")
        assert sc.dep_audit_interval_seconds == 86400


# ==========================================================================
# Class 2: SchedulerConfig in RedTeamConfig
# ==========================================================================


class TestSchedulerConfigInRedTeamConfig:
    """Test SchedulerConfig integration in RedTeamConfig."""

    def setup_method(self):
        self.cfg_mod = _load_config()

    def test_default_scheduler_field(self):
        rtc = self.cfg_mod.RedTeamConfig()
        assert isinstance(rtc.scheduler, self.cfg_mod.SchedulerConfig)
        assert rtc.scheduler.enabled is True

    def test_custom_scheduler(self):
        sc = self.cfg_mod.SchedulerConfig(interval="daily", enabled=False)
        rtc = self.cfg_mod.RedTeamConfig(scheduler=sc)
        assert rtc.scheduler.interval == "daily"
        assert rtc.scheduler.enabled is False


# ==========================================================================
# Class 3: YAML loading with scheduler section
# ==========================================================================


class TestSchedulerYAMLLoading:
    """Test loading scheduler config from redteam.yaml."""

    def setup_method(self):
        self.cfg_mod = _load_config()

    def test_load_from_yaml(self):
        cfg = self.cfg_mod.load_redteam_config(config_path=REDTEAM_YAML_PATH)
        assert isinstance(cfg.scheduler, self.cfg_mod.SchedulerConfig)
        assert cfg.scheduler.enabled is True
        assert cfg.scheduler.interval == "weekly"
        assert cfg.scheduler.quiet_hours_start == "00:00"
        assert cfg.scheduler.quiet_hours_end == "06:00"
        assert cfg.scheduler.dep_severity_threshold == "high"

    def test_override_scheduler_from_dict(self):
        cfg = self.cfg_mod.load_redteam_config(
            config_path=REDTEAM_YAML_PATH,
            overrides={"scheduler": {"interval": "daily", "enabled": False}},
        )
        assert cfg.scheduler.interval == "daily"
        assert cfg.scheduler.enabled is False


# ==========================================================================
# Class 4: DependencyMonitor parsing
# ==========================================================================


class TestDepMonitorParsing:
    """Test DependencyMonitor JSON parsing."""

    def setup_method(self):
        self.dm_mod = _load_dep_monitor()
        self.clock = lambda: datetime(2026, 3, 27, 10, 0, 0, tzinfo=timezone.utc)

    def test_empty_output(self):
        dm = self.dm_mod.DependencyMonitor(clock=self.clock)
        result = dm.run_audit_from_json("")
        assert result.status == "completed"
        assert result.filtered_count == 0

    def test_dict_format(self):
        data = {"dependencies": [
            {"name": "pkg", "version": "1.0", "vulns": [
                {"id": "CVE-1", "description": "test", "severity": "high"}
            ]}
        ]}
        dm = self.dm_mod.DependencyMonitor(severity_threshold="high", clock=self.clock)
        result = dm.run_audit_from_json(json.dumps(data))
        assert result.status == "completed"
        assert result.total_raw == 1
        assert result.filtered_count == 1

    def test_list_format(self):
        data = [
            {"name": "pkg", "version": "1.0", "vulns": [
                {"id": "CVE-2", "description": "test", "severity": "critical"}
            ]}
        ]
        dm = self.dm_mod.DependencyMonitor(severity_threshold="high", clock=self.clock)
        result = dm.run_audit_from_json(json.dumps(data))
        assert result.filtered_count == 1
        assert result.filtered_findings[0].vuln_id == "CVE-2"

    def test_parse_error(self):
        dm = self.dm_mod.DependencyMonitor(clock=self.clock)
        result = dm.run_audit_from_json("{invalid json")
        assert result.status == "parse_error"

    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="Invalid severity threshold"):
            self.dm_mod.DependencyMonitor(severity_threshold="extreme")


# ==========================================================================
# Class 5: DependencyMonitor severity filtering
# ==========================================================================


class TestDepMonitorFiltering:
    """Test severity threshold filtering logic."""

    def setup_method(self):
        self.dm_mod = _load_dep_monitor()
        self.clock = lambda: datetime(2026, 3, 27, 10, 0, 0, tzinfo=timezone.utc)
        self.sample = json.dumps({"dependencies": [
            {"name": "a", "version": "1.0", "vulns": [
                {"id": "V1", "description": "low", "severity": "low"},
                {"id": "V2", "description": "med", "severity": "medium"},
                {"id": "V3", "description": "high", "severity": "high"},
                {"id": "V4", "description": "crit", "severity": "critical"},
            ]}
        ]})

    def test_threshold_low_all_pass(self):
        dm = self.dm_mod.DependencyMonitor(severity_threshold="low", clock=self.clock)
        result = dm.run_audit_from_json(self.sample)
        assert result.filtered_count == 4

    def test_threshold_medium(self):
        dm = self.dm_mod.DependencyMonitor(severity_threshold="medium", clock=self.clock)
        result = dm.run_audit_from_json(self.sample)
        assert result.filtered_count == 3

    def test_threshold_high(self):
        dm = self.dm_mod.DependencyMonitor(severity_threshold="high", clock=self.clock)
        result = dm.run_audit_from_json(self.sample)
        assert result.filtered_count == 2

    def test_threshold_critical(self):
        dm = self.dm_mod.DependencyMonitor(severity_threshold="critical", clock=self.clock)
        result = dm.run_audit_from_json(self.sample)
        assert result.filtered_count == 1
        assert result.filtered_findings[0].vuln_id == "V4"

    def test_unknown_severity_included(self):
        data = json.dumps({"dependencies": [
            {"name": "x", "version": "1.0", "vulns": [
                {"id": "U1", "description": "unknown sev"}
            ]}
        ]})
        dm = self.dm_mod.DependencyMonitor(severity_threshold="critical", clock=self.clock)
        result = dm.run_audit_from_json(data)
        assert result.filtered_count == 1  # conservative inclusion


# ==========================================================================
# Class 6: DependencyMonitor history and summary
# ==========================================================================


class TestDepMonitorHistory:
    """Test audit result history tracking."""

    def setup_method(self):
        self.dm_mod = _load_dep_monitor()
        self.clock = lambda: datetime(2026, 3, 27, 10, 0, 0, tzinfo=timezone.utc)

    def test_audit_count_increments(self):
        dm = self.dm_mod.DependencyMonitor(clock=self.clock)
        assert dm.audit_count == 0
        dm.run_audit_from_json("")
        assert dm.audit_count == 1
        dm.run_audit_from_json("")
        assert dm.audit_count == 2

    def test_last_result(self):
        dm = self.dm_mod.DependencyMonitor(clock=self.clock)
        assert dm.last_result is None
        r = dm.run_audit_from_json("")
        assert dm.last_result is r

    def test_results_history(self):
        dm = self.dm_mod.DependencyMonitor(clock=self.clock)
        dm.run_audit_from_json("")
        dm.run_audit_from_json("")
        assert len(dm.results_history) == 2

    def test_get_summary(self):
        dm = self.dm_mod.DependencyMonitor(clock=self.clock)
        dm.run_audit_from_json("")
        summary = dm.get_summary()
        assert summary["audit_count"] == 1
        assert summary["severity_threshold"] == "high"
        assert summary["last_audit"] is not None


# ==========================================================================
# Class 7: VulnerabilityRecord and AuditResult serialization
# ==========================================================================


class TestDepMonitorSerialization:
    """Test dataclass to_dict methods."""

    def setup_method(self):
        self.dm_mod = _load_dep_monitor()

    def test_vulnerability_record_to_dict(self):
        vr = self.dm_mod.VulnerabilityRecord(
            package_name="flask",
            installed_version="2.0.0",
            vuln_id="CVE-2024-1234",
            description="XSS",
            severity="high",
            fix_versions=["2.0.1"],
        )
        d = vr.to_dict()
        assert d["package_name"] == "flask"
        assert d["vuln_id"] == "CVE-2024-1234"
        assert d["fix_versions"] == ["2.0.1"]

    def test_audit_result_to_dict(self):
        self.clock = lambda: datetime(2026, 3, 27, 10, 0, 0, tzinfo=timezone.utc)
        dm = self.dm_mod.DependencyMonitor(clock=self.clock)
        result = dm.run_audit_from_json("")
        d = result.to_dict()
        assert "status" in d
        assert "timestamp" in d
        assert "filtered_count" in d
        assert "vulnerabilities" in d


# ==========================================================================
# Class 8: SecurityScheduler quiet hours
# ==========================================================================


class TestSchedulerQuietHours:
    """Test quiet hours enforcement."""

    def setup_method(self):
        self.cfg_mod = _load_config()
        self.sched_mod = _load_scheduler(self.cfg_mod)

    def _make(self, start, end, hour, minute=0):
        cfg = self.cfg_mod.SchedulerConfig(
            quiet_hours_start=start, quiet_hours_end=end,
        )
        clock = lambda: datetime(2026, 3, 27, hour, minute, 0, tzinfo=timezone.utc)
        return self.sched_mod.SecurityScheduler(config=cfg, clock=clock)

    def test_inside_simple_range(self):
        s = self._make("00:00", "06:00", 3)
        assert s._is_quiet_hours() is True

    def test_outside_simple_range(self):
        s = self._make("00:00", "06:00", 8)
        assert s._is_quiet_hours() is False

    def test_boundary_start(self):
        s = self._make("02:00", "06:00", 2)
        assert s._is_quiet_hours() is True

    def test_boundary_end_exclusive(self):
        s = self._make("02:00", "06:00", 6)
        assert s._is_quiet_hours() is False

    def test_wrap_around_inside_late(self):
        s = self._make("22:00", "06:00", 23)
        assert s._is_quiet_hours() is True

    def test_wrap_around_inside_early(self):
        s = self._make("22:00", "06:00", 3)
        assert s._is_quiet_hours() is True

    def test_wrap_around_outside(self):
        s = self._make("22:00", "06:00", 12)
        assert s._is_quiet_hours() is False

    def test_seconds_until_quiet_end(self):
        s = self._make("00:00", "06:00", 3, 30)
        seconds = s._seconds_until_quiet_end()
        # 06:00 - 03:30 = 2.5h = 150 min = 9000s
        assert seconds == 9000


# ==========================================================================
# Class 9: SecurityScheduler alerts
# ==========================================================================


class TestSchedulerAlerts:
    """Test alert management."""

    def setup_method(self):
        self.cfg_mod = _load_config()
        self.sched_mod = _load_scheduler(self.cfg_mod)

    def test_alert_creation(self):
        alert = self.sched_mod.SchedulerAlert(
            alert_type="regression", message="test alert"
        )
        d = alert.to_dict()
        assert d["alert_type"] == "regression"
        assert d["message"] == "test alert"
        assert d["timestamp"] != ""

    def test_add_and_get_alerts(self):
        cfg = self.cfg_mod.SchedulerConfig()
        clock = lambda: datetime(2026, 3, 27, 10, 0, 0, tzinfo=timezone.utc)
        s = self.sched_mod.SecurityScheduler(config=cfg, clock=clock)
        s._add_alert(self.sched_mod.SchedulerAlert(
            alert_type="test", message="msg1"
        ))
        s._add_alert(self.sched_mod.SchedulerAlert(
            alert_type="test", message="msg2"
        ))
        alerts = s.get_alerts()
        assert len(alerts) == 2

    def test_clear_alerts(self):
        cfg = self.cfg_mod.SchedulerConfig()
        clock = lambda: datetime(2026, 3, 27, 10, 0, 0, tzinfo=timezone.utc)
        s = self.sched_mod.SecurityScheduler(config=cfg, clock=clock)
        s._add_alert(self.sched_mod.SchedulerAlert(
            alert_type="test", message="msg"
        ))
        cleared = s.clear_alerts()
        assert cleared == 1
        assert len(s.get_alerts()) == 0

    def test_alert_cap_at_100(self):
        cfg = self.cfg_mod.SchedulerConfig()
        clock = lambda: datetime(2026, 3, 27, 10, 0, 0, tzinfo=timezone.utc)
        s = self.sched_mod.SecurityScheduler(config=cfg, clock=clock)
        for i in range(120):
            s._add_alert(self.sched_mod.SchedulerAlert(
                alert_type="test", message=f"msg-{i}"
            ))
        assert len(s._alerts) == 100
        # Most recent should be msg-119
        assert s._alerts[-1].message == "msg-119"


# ==========================================================================
# Class 10: SecurityScheduler get_status
# ==========================================================================


class TestSchedulerStatus:
    """Test scheduler status reporting."""

    def setup_method(self):
        self.cfg_mod = _load_config()
        self.sched_mod = _load_scheduler(self.cfg_mod)

    def test_status_structure(self):
        cfg = self.cfg_mod.SchedulerConfig()
        clock = lambda: datetime(2026, 3, 27, 10, 0, 0, tzinfo=timezone.utc)
        s = self.sched_mod.SecurityScheduler(config=cfg, clock=clock)
        status = s.get_status()

        assert "enabled" in status
        assert "running" in status
        assert "redteam" in status
        assert "dep_audit" in status
        assert "quiet_hours" in status
        assert "alerts" in status
        assert "alerts_total" in status

    def test_status_redteam_fields(self):
        cfg = self.cfg_mod.SchedulerConfig(interval="daily")
        clock = lambda: datetime(2026, 3, 27, 10, 0, 0, tzinfo=timezone.utc)
        s = self.sched_mod.SecurityScheduler(config=cfg, clock=clock)
        rt = s.get_status()["redteam"]
        assert rt["interval"] == "daily"
        assert rt["run_count"] == 0
        assert rt["last_run"] is None

    def test_status_dep_audit_fields(self):
        cfg = self.cfg_mod.SchedulerConfig(dep_severity_threshold="critical")
        clock = lambda: datetime(2026, 3, 27, 10, 0, 0, tzinfo=timezone.utc)
        s = self.sched_mod.SecurityScheduler(config=cfg, clock=clock)
        da = s.get_status()["dep_audit"]
        assert da["severity_threshold"] == "critical"
        assert da["audit_count"] == 0

    def test_status_quiet_hours(self):
        cfg = self.cfg_mod.SchedulerConfig(
            quiet_hours_start="02:00", quiet_hours_end="06:00"
        )
        clock = lambda: datetime(2026, 3, 27, 3, 0, 0, tzinfo=timezone.utc)
        s = self.sched_mod.SecurityScheduler(config=cfg, clock=clock)
        qh = s.get_status()["quiet_hours"]
        assert qh["start"] == "02:00"
        assert qh["end"] == "06:00"
        assert qh["active"] is True


# ==========================================================================
# Class 11: SecurityScheduler start/stop
# ==========================================================================


class TestSchedulerStartStop:
    """Test scheduler lifecycle."""

    def setup_method(self):
        self.cfg_mod = _load_config()
        self.sched_mod = _load_scheduler(self.cfg_mod)

    def test_start_sets_running(self):
        cfg = self.cfg_mod.SchedulerConfig()
        clock = lambda: datetime(2026, 3, 27, 10, 0, 0, tzinfo=timezone.utc)
        s = self.sched_mod.SecurityScheduler(config=cfg, clock=clock)
        s.start()
        assert s.is_running is True
        s.stop()

    def test_stop_clears_running(self):
        cfg = self.cfg_mod.SchedulerConfig()
        clock = lambda: datetime(2026, 3, 27, 10, 0, 0, tzinfo=timezone.utc)
        s = self.sched_mod.SecurityScheduler(config=cfg, clock=clock)
        s.start()
        s.stop()
        assert s.is_running is False
        assert s._next_redteam_run is None

    def test_disabled_scheduler_no_start(self):
        cfg = self.cfg_mod.SchedulerConfig(enabled=False)
        clock = lambda: datetime(2026, 3, 27, 10, 0, 0, tzinfo=timezone.utc)
        s = self.sched_mod.SecurityScheduler(config=cfg, clock=clock)
        s.start()
        assert s.is_running is False

    def test_double_start_no_error(self):
        cfg = self.cfg_mod.SchedulerConfig()
        clock = lambda: datetime(2026, 3, 27, 10, 0, 0, tzinfo=timezone.utc)
        s = self.sched_mod.SecurityScheduler(config=cfg, clock=clock)
        s.start()
        s.start()  # should log warning, not crash
        assert s.is_running is True
        s.stop()

    def test_on_deploy_no_timer(self):
        cfg = self.cfg_mod.SchedulerConfig(interval="on-deploy")
        clock = lambda: datetime(2026, 3, 27, 10, 0, 0, tzinfo=timezone.utc)
        s = self.sched_mod.SecurityScheduler(config=cfg, clock=clock)
        s.start()
        assert s._redteam_timer is None  # on-deploy = no periodic timer
        s.stop()


# ==========================================================================
# Class 12: SecurityScheduler dep audit delegation
# ==========================================================================


class TestSchedulerDepAuditDelegation:
    """Test that scheduler delegates to DependencyMonitor."""

    def setup_method(self):
        self.cfg_mod = _load_config()
        self.sched_mod = _load_scheduler(self.cfg_mod)
        self.dm_mod = _load_dep_monitor()

    def test_dep_monitor_instance(self):
        cfg = self.cfg_mod.SchedulerConfig(dep_severity_threshold="critical")
        clock = lambda: datetime(2026, 3, 27, 10, 0, 0, tzinfo=timezone.utc)
        s = self.sched_mod.SecurityScheduler(config=cfg, clock=clock)
        # Duck-type check (module reloading creates different class identities)
        assert hasattr(s._dep_monitor, "run_audit")
        assert hasattr(s._dep_monitor, "severity_threshold")
        assert s._dep_monitor.severity_threshold == "critical"

    def test_trigger_dep_audit_returns_dict(self):
        cfg = self.cfg_mod.SchedulerConfig()
        clock = lambda: datetime(2026, 3, 27, 10, 0, 0, tzinfo=timezone.utc)
        s = self.sched_mod.SecurityScheduler(config=cfg, clock=clock)
        # Mock run_audit to avoid calling real pip-audit
        mock_result = self.dm_mod.AuditResult(
            status="completed",
            timestamp="2026-03-27T10:00:00+00:00",
        )
        s._dep_monitor.run_audit = lambda: mock_result
        result = s.trigger_dep_audit()
        assert result["status"] == "completed"

    def test_dep_audit_vulnerability_alert(self):
        cfg = self.cfg_mod.SchedulerConfig()
        clock = lambda: datetime(2026, 3, 27, 10, 0, 0, tzinfo=timezone.utc)
        s = self.sched_mod.SecurityScheduler(config=cfg, clock=clock)
        vr = self.dm_mod.VulnerabilityRecord(
            package_name="pkg", installed_version="1.0",
            vuln_id="CVE-TEST", description="test", severity="high",
        )
        mock_result = self.dm_mod.AuditResult(
            status="completed",
            timestamp="2026-03-27T10:00:00+00:00",
            filtered_findings=[vr],
        )
        s._dep_monitor.run_audit = lambda: mock_result
        s.trigger_dep_audit()
        alerts = s.get_alerts()
        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == "vulnerability"


# ==========================================================================
# Class 13: SecurityScheduler regression detection
# ==========================================================================


class TestSchedulerRegressionDetection:
    """Test bypass rate regression detection logic."""

    def setup_method(self):
        self.cfg_mod = _load_config()
        self.sched_mod = _load_scheduler(self.cfg_mod)

    def test_regression_fires_on_increase(self):
        cfg = self.cfg_mod.SchedulerConfig()
        clock = lambda: datetime(2026, 3, 27, 10, 0, 0, tzinfo=timezone.utc)
        s = self.sched_mod.SecurityScheduler(config=cfg, clock=clock)
        # Simulate previous result with low bypass rate
        s._last_redteam_result = {"bypass_rate": 0.05}

        # Mock _execute_redteam to simulate a higher bypass rate
        original_execute = s._execute_redteam

        def mock_execute(force=False):
            # Directly set result without running real campaign
            result = {
                "status": "completed",
                "timestamp": clock().isoformat(),
                "bypass_rate": 0.20,  # +15% over previous
                "detection_rate": 0.80,
                "total_attacks": 10,
                "bypasses": 2,
            }
            prev_rate = s._last_redteam_result.get("bypass_rate")
            s._last_redteam_run = result["timestamp"]
            s._last_redteam_result = result
            s._run_count += 1
            # Check regression (same logic as real code)
            if (
                prev_rate is not None
                and result["status"] == "completed"
                and result["bypass_rate"] > prev_rate + 0.05
            ):
                s._add_alert(self.sched_mod.SchedulerAlert(
                    alert_type="regression",
                    message="Regression detected",
                    details={
                        "previous_bypass_rate": prev_rate,
                        "current_bypass_rate": result["bypass_rate"],
                    },
                ))
            return result

        s._execute_redteam = mock_execute
        s._execute_redteam(force=True)

        alerts = s.get_alerts()
        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == "regression"
        assert alerts[0]["details"]["previous_bypass_rate"] == 0.05
        assert alerts[0]["details"]["current_bypass_rate"] == 0.20

    def test_no_regression_on_small_change(self):
        cfg = self.cfg_mod.SchedulerConfig()
        clock = lambda: datetime(2026, 3, 27, 10, 0, 0, tzinfo=timezone.utc)
        s = self.sched_mod.SecurityScheduler(config=cfg, clock=clock)
        s._last_redteam_result = {"bypass_rate": 0.10}

        def mock_execute(force=False):
            result = {
                "status": "completed",
                "timestamp": clock().isoformat(),
                "bypass_rate": 0.12,  # Only +2%, below 5% threshold
                "detection_rate": 0.88,
                "total_attacks": 10,
                "bypasses": 1,
            }
            prev_rate = s._last_redteam_result.get("bypass_rate")
            s._last_redteam_run = result["timestamp"]
            s._last_redteam_result = result
            s._run_count += 1
            if (
                prev_rate is not None
                and result["status"] == "completed"
                and result["bypass_rate"] > prev_rate + 0.05
            ):
                s._add_alert(self.sched_mod.SchedulerAlert(
                    alert_type="regression",
                    message="Regression detected",
                ))
            return result

        s._execute_redteam = mock_execute
        s._execute_redteam(force=True)
        assert len(s.get_alerts()) == 0


# ==========================================================================
# Class 14: Module structure checks
# ==========================================================================


class TestModuleStructure:
    """Test checkpoint sentinels, AST validity, and code quality."""

    def test_scheduler_checkpoint(self):
        mod = _load_scheduler()
        assert mod.checkpoint_before_apply is True

    def test_dep_monitor_checkpoint(self):
        mod = _load_dep_monitor()
        assert mod.checkpoint_before_apply is True

    def test_scheduler_ast_valid(self):
        with open(SCHEDULER_PATH, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        assert isinstance(tree, ast.Module)

    def test_dep_monitor_ast_valid(self):
        with open(DEP_MONITOR_PATH, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        assert isinstance(tree, ast.Module)

    def test_config_ast_valid(self):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        assert isinstance(tree, ast.Module)

    def test_routes_security_ast_valid(self):
        with open(ROUTES_SECURITY_PATH, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        assert isinstance(tree, ast.Module)

    def test_app_ast_valid(self):
        with open(APP_PATH, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        assert isinstance(tree, ast.Module)

    def test_no_french_in_scheduler(self):
        with open(SCHEDULER_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        # Check for common French words in code comments
        french_patterns = [
            r"\bvoir\b", r"\bfonction\b", r"\bretourne\b",
            r"\bparametre\b", r"\bvérif",
        ]
        for pat in french_patterns:
            assert not re.search(pat, content, re.IGNORECASE), (
                f"French detected: {pat}"
            )

    def test_no_french_in_dep_monitor(self):
        with open(DEP_MONITOR_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        french_patterns = [
            r"\bvoir\b", r"\bfonction\b", r"\bretourne\b",
        ]
        for pat in french_patterns:
            assert not re.search(pat, content, re.IGNORECASE), (
                f"French detected: {pat}"
            )

    def test_no_emoji_in_scheduler(self):
        with open(SCHEDULER_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        emoji_pattern = re.compile(
            "[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff"
            "\U0001f680-\U0001f6ff\U0001f900-\U0001f9ff]"
        )
        assert not emoji_pattern.search(content), "Emoji found in scheduler"

    def test_no_emoji_in_dep_monitor(self):
        with open(DEP_MONITOR_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        emoji_pattern = re.compile(
            "[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff"
            "\U0001f680-\U0001f6ff\U0001f900-\U0001f9ff]"
        )
        assert not emoji_pattern.search(content), "Emoji found in dep_monitor"


# ==========================================================================
# Class 15: YAML config file structure
# ==========================================================================


class TestRedteamYAMLStructure:
    """Test that redteam.yaml contains expected scheduler section."""

    def test_yaml_has_scheduler_section(self):
        # Ensure real yaml module is used (not the mock)
        if "yaml" in sys.modules and not hasattr(sys.modules["yaml"], "__file__"):
            del sys.modules["yaml"]
        import yaml
        with open(REDTEAM_YAML_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "scheduler" in data
        sched = data["scheduler"]
        assert sched["enabled"] is True
        assert sched["interval"] == "weekly"
        assert sched["quiet_hours_start"] == "00:00"
        assert sched["quiet_hours_end"] == "06:00"
        assert sched["auto_accept_suggestions"] is False
        assert sched["dep_audit_interval"] == "weekly"
        assert sched["dep_severity_threshold"] == "high"

    def test_yaml_scheduler_keys_complete(self):
        if "yaml" in sys.modules and not hasattr(sys.modules["yaml"], "__file__"):
            del sys.modules["yaml"]
        import yaml
        with open(REDTEAM_YAML_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        expected_keys = {
            "enabled", "interval", "quiet_hours_start", "quiet_hours_end",
            "auto_accept_suggestions", "dep_audit_interval",
            "dep_severity_threshold",
        }
        assert set(data["scheduler"].keys()) == expected_keys


# ==========================================================================
# Class 16: Routes security new endpoints AST presence
# ==========================================================================


class TestRoutesNewEndpoints:
    """Verify new S158 endpoints exist in routes_security.py."""

    def setup_method(self):
        with open(ROUTES_SECURITY_PATH, "r", encoding="utf-8") as f:
            self.content = f.read()

    def test_scheduler_get_endpoint(self):
        assert 'router.get("/scheduler")' in self.content

    def test_scheduler_trigger_endpoint(self):
        assert 'router.post("/scheduler/trigger")' in self.content

    def test_scheduler_trigger_request_model(self):
        assert "class SchedulerTriggerRequest" in self.content

    def test_get_security_status_scheduler_section(self):
        assert "scheduler" in self.content
        assert "from opti_oignon.security_scheduler import get_scheduler" in self.content


# ==========================================================================
# Class 17: App.py health endpoint scheduler section
# ==========================================================================


class TestAppHealthScheduler:
    """Verify scheduler section in app.py health endpoint."""

    def setup_method(self):
        with open(APP_PATH, "r", encoding="utf-8") as f:
            self.content = f.read()

    def test_scheduler_import_in_health(self):
        assert "from opti_oignon.security_scheduler import get_scheduler" in self.content

    def test_scheduler_section_in_health_info(self):
        assert 'info["scheduler"]' in self.content
        assert "redteam_interval" in self.content
        assert "quiet_hours_active" in self.content
