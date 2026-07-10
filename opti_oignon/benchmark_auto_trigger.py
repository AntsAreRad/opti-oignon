#!/usr/bin/env python3
"""
Benchmark Auto-Trigger

Watches Ollama model list for changes (new models, updated digests,
removed models) and auto-triggers benchmark runs when changes are
detected.

STRICTLY OPT-IN: disabled by default. User must explicitly enable
via the UI toggle or the API. Never runs benchmarks without consent.
Local machines have limited GPU/RAM -- unsolicited runs would degrade
performance.

Lifecycle:
  1. Background polling thread checks ollama.list() at configurable interval
  2. Compares current model set against last-known snapshot
  3. On new/updated model detection, triggers benchmark_runner.start_run
  4. Debounce cooldown prevents re-triggering within configurable window
  5. Optional resource guard refuses to trigger under high system load
  6. Results feed into recommendations engine automatically
"""

import fnmatch
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# YAML config loading
# ---------------------------------------------------------------------------
_CONFIG_DIR = Path(__file__).parent / "config"
_CONFIG_PATH = _CONFIG_DIR / "benchmark_auto_trigger.yaml"


def _load_config(path: Path | None = None) -> dict:
    """Load auto-trigger configuration from YAML."""
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not available, using default config")
        return {}
    target = path or _CONFIG_PATH
    if not target.exists():
        logger.debug("Auto-trigger config not found: %s", target)
        return {}
    with open(target, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _save_config(data: dict, path: Path | None = None) -> bool:
    """Persist auto-trigger configuration to YAML."""
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not available, cannot save config")
        return False
    target = path or _CONFIG_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)
        return True
    except OSError as exc:
        logger.error("Failed to save auto-trigger config: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelSnapshot:
    """A point-in-time view of installed Ollama models."""
    models: dict[str, str] = field(default_factory=dict)
    timestamp: float = 0.0

    def diff(self, other: "ModelSnapshot") -> "ModelDiff":
        """Compare this snapshot against an older one."""
        added: list[str] = []
        removed: list[str] = []
        updated: list[str] = []

        for name, digest in self.models.items():
            if name not in other.models:
                added.append(name)
            elif other.models[name] != digest:
                updated.append(name)

        for name in other.models:
            if name not in self.models:
                removed.append(name)

        return ModelDiff(
            added=added,
            removed=removed,
            updated=updated,
            has_changes=bool(added or removed or updated),
        )


@dataclass
class ModelDiff:
    """Difference between two model snapshots."""
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    has_changes: bool = False


@dataclass
class TriggerEvent:
    """Record of an auto-trigger event."""
    event_id: str = ""
    timestamp: float = 0.0
    trigger_type: str = ""
    models: list[str] = field(default_factory=list)
    run_id: str = ""
    profile: str = ""
    skipped: bool = False
    skip_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "trigger_type": self.trigger_type,
            "models": self.models,
            "run_id": self.run_id,
            "profile": self.profile,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


# ---------------------------------------------------------------------------
# Auto-Trigger Engine
# ---------------------------------------------------------------------------

class AutoTrigger:
    """Watches Ollama model list and triggers benchmarks on changes.

    Thread-safe. The polling loop runs in a daemon thread that cleanly
    shuts down when stop() is called or the process exits.
    """

    def __init__(
        self,
        config_path: Path | None = None,
        benchmark_runner: Any = None,
        ollama_list_fn: Any = None,
    ):
        self._config_path = config_path
        self._config = _load_config(config_path)
        self._benchmark_runner = benchmark_runner
        self._ollama_list_fn = ollama_list_fn

        # State
        self._enabled: bool = self._config.get("enabled", False)
        self._poll_interval: float = float(
            self._config.get("poll_interval_seconds", 120)
        )
        self._cooldown: float = float(
            self._config.get("cooldown_seconds", 1800)
        )
        self._trigger_profile: str = self._config.get(
            "trigger_profile", "all_round"
        )
        self._trigger_models: str | list[str] = self._config.get(
            "trigger_models", "all_new"
        )
        self._resource_guard_max: float = float(
            self._config.get("resource_guard_load_max", 0.0)
        )
        self._use_judge: bool = self._config.get("use_judge", False)
        self._judge_model: str = self._config.get("judge_model", "")

        # Runtime state
        self._last_snapshot: ModelSnapshot = ModelSnapshot()
        self._last_trigger_time: float = 0.0
        self._events: list[TriggerEvent] = []
        self._max_events: int = 100

        # Thread management
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    # -- Properties --

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def config(self) -> dict:
        with self._lock:
            return {
                "enabled": self._enabled,
                "poll_interval_seconds": self._poll_interval,
                "cooldown_seconds": self._cooldown,
                "trigger_profile": self._trigger_profile,
                "trigger_models": self._trigger_models,
                "resource_guard_load_max": self._resource_guard_max,
                "use_judge": self._use_judge,
                "judge_model": self._judge_model,
            }

    @property
    def status(self) -> dict:
        with self._lock:
            now = time.time()
            remaining = 0.0
            if self._last_trigger_time > 0:
                elapsed = now - self._last_trigger_time
                if elapsed < self._cooldown:
                    remaining = self._cooldown - elapsed

            resource_guard_active = self._resource_guard_max > 0.0

            return {
                "enabled": self._enabled,
                "running": self.running,
                "poll_interval_seconds": self._poll_interval,
                "cooldown_seconds": self._cooldown,
                "cooldown_remaining": round(remaining, 1),
                "trigger_profile": self._trigger_profile,
                "last_trigger_time": self._last_trigger_time,
                "known_models": len(self._last_snapshot.models),
                "recent_events": len(self._events),
                "resource_guard_active": resource_guard_active,
                "resource_guard_load_max": self._resource_guard_max,
            }

    @property
    def events(self) -> list[dict]:
        with self._lock:
            return [e.to_dict() for e in self._events]

    # -- Enable/Disable --

    def enable(self) -> bool:
        """Enable auto-trigger and start the polling thread."""
        with self._lock:
            self._enabled = True
            self._persist_enabled(True)
        self.start()
        return True

    def disable(self) -> bool:
        """Disable auto-trigger and stop the polling thread."""
        with self._lock:
            self._enabled = False
            self._persist_enabled(False)
        self.stop()
        return True

    def _persist_enabled(self, value: bool) -> None:
        """Save the enabled state to disk."""
        cfg = _load_config(self._config_path)
        cfg["enabled"] = value
        _save_config(cfg, self._config_path)

    # -- Configuration update --

    def update_config(self, updates: dict) -> dict:
        """Update configuration fields and persist.

        Args:
            updates: Dict of field names to new values. Only known
                fields are accepted.

        Returns:
            Updated full config dict.
        """
        allowed = {
            "enabled", "poll_interval_seconds", "cooldown_seconds",
            "trigger_profile", "trigger_models",
            "resource_guard_load_max", "use_judge", "judge_model",
        }
        with self._lock:
            for key, value in updates.items():
                if key not in allowed:
                    continue
                if key == "enabled":
                    self._enabled = bool(value)
                elif key == "poll_interval_seconds":
                    self._poll_interval = max(10.0, float(value))
                elif key == "cooldown_seconds":
                    self._cooldown = max(0.0, float(value))
                elif key == "trigger_profile":
                    self._trigger_profile = str(value)
                elif key == "trigger_models":
                    self._trigger_models = value
                elif key == "resource_guard_load_max":
                    self._resource_guard_max = max(0.0, float(value))
                elif key == "use_judge":
                    self._use_judge = bool(value)
                elif key == "judge_model":
                    self._judge_model = str(value)

            # Persist
            cfg = _load_config(self._config_path)
            cfg.update({
                "enabled": self._enabled,
                "poll_interval_seconds": self._poll_interval,
                "cooldown_seconds": self._cooldown,
                "trigger_profile": self._trigger_profile,
                "trigger_models": self._trigger_models,
                "resource_guard_load_max": self._resource_guard_max,
                "use_judge": self._use_judge,
                "judge_model": self._judge_model,
            })
            _save_config(cfg, self._config_path)

            # Handle enable/disable side effect
            if "enabled" in updates:
                if self._enabled and not self.running:
                    # Will start after lock release
                    pass
                elif not self._enabled and self.running:
                    self._stop_event.set()

        if self._enabled and not self.running:
            self.start()

        return self.config

    # -- Snapshot --

    def take_snapshot(self) -> ModelSnapshot:
        """Take a snapshot of currently installed Ollama models."""
        models: dict[str, str] = {}
        try:
            list_fn = self._ollama_list_fn or _default_ollama_list
            raw_models = list_fn()
            for m in raw_models:
                name = ""
                digest = ""
                if hasattr(m, "model"):
                    name = m.model
                elif hasattr(m, "name"):
                    name = m.name
                elif isinstance(m, dict):
                    name = m.get("model", m.get("name", ""))

                if hasattr(m, "digest"):
                    digest = m.digest
                elif isinstance(m, dict):
                    digest = m.get("digest", "")

                if name:
                    models[name] = digest
        except Exception as exc:
            logger.debug("Failed to list Ollama models: %s", exc)

        return ModelSnapshot(models=models, timestamp=time.time())

    def _filter_trigger_models(self, model_names: list[str]) -> list[str]:
        """Filter model names against trigger_models config."""
        with self._lock:
            trigger = self._trigger_models

        if trigger == "all_new":
            return model_names

        if isinstance(trigger, list):
            result = []
            for name in model_names:
                for pattern in trigger:
                    if fnmatch.fnmatch(name, pattern):
                        result.append(name)
                        break
            return result

        return model_names

    # -- Resource guard --

    def _check_resource_guard(self) -> tuple[bool, str]:
        """Check if system load is below threshold.

        Returns:
            Tuple of (allowed, reason). allowed=True means OK to trigger.
        """
        with self._lock:
            max_load = self._resource_guard_max

        if max_load <= 0.0:
            return True, ""

        try:
            load_1min = os.getloadavg()[0]
            if load_1min > max_load:
                return False, f"System load {load_1min:.2f} exceeds threshold {max_load:.2f}"
        except (OSError, AttributeError):
            pass

        return True, ""

    # -- Cooldown --

    def _check_cooldown(self) -> tuple[bool, str]:
        """Check if enough time has passed since last trigger.

        Returns:
            Tuple of (allowed, reason).
        """
        with self._lock:
            elapsed = time.time() - self._last_trigger_time
            cooldown = self._cooldown

        if self._last_trigger_time > 0 and elapsed < cooldown:
            remaining = cooldown - elapsed
            return False, f"Cooldown active, {remaining:.0f}s remaining"

        return True, ""

    # -- Trigger execution --

    def _do_trigger(self, models: list[str], trigger_type: str) -> TriggerEvent:
        """Execute a benchmark trigger for the given models.

        Args:
            models: List of model names to benchmark.
            trigger_type: One of 'new_model', 'updated_model'.

        Returns:
            TriggerEvent recording what happened.
        """
        import uuid

        event = TriggerEvent(
            event_id=f"evt-{uuid.uuid4().hex[:10]}",
            timestamp=time.time(),
            trigger_type=trigger_type,
            models=models,
        )

        # Check cooldown
        allowed, reason = self._check_cooldown()
        if not allowed:
            event.skipped = True
            event.skip_reason = reason
            self._record_event(event)
            logger.info("Auto-trigger skipped (cooldown): %s", reason)
            return event

        # Check resource guard
        allowed, reason = self._check_resource_guard()
        if not allowed:
            event.skipped = True
            event.skip_reason = reason
            self._record_event(event)
            logger.info("Auto-trigger skipped (resource guard): %s", reason)
            return event

        # Get config under lock
        with self._lock:
            profile = self._trigger_profile
            use_judge = self._use_judge
            judge_model = self._judge_model

        event.profile = profile

        # Try to start run
        runner = self._benchmark_runner
        if runner is None:
            try:
                from opti_oignon.benchmark_runner import benchmark_runner as _runner
                runner = _runner
            except ImportError:
                event.skipped = True
                event.skip_reason = "Benchmark runner not available"
                self._record_event(event)
                return event

        if runner is None:
            event.skipped = True
            event.skip_reason = "Benchmark runner is None"
            self._record_event(event)
            return event

        # Check if runner is busy (identity check avoids MagicMock truthy)
        runner_busy = getattr(runner, "is_busy", None)
        if runner_busy is True:
            event.skipped = True
            event.skip_reason = "Runner busy with another benchmark"
            self._record_event(event)
            logger.info("Auto-trigger skipped (runner busy)")
            return event

        try:
            run_id = runner.start_run(
                profile=profile,
                models=models,
                use_judge=use_judge,
                judge_model=judge_model if use_judge else "",
            )
            event.run_id = run_id
            with self._lock:
                self._last_trigger_time = time.time()
            logger.info(
                "Auto-trigger started run %s for models %s (profile=%s)",
                run_id, models, profile,
            )
        except Exception as exc:
            event.skipped = True
            event.skip_reason = f"Runner error: {exc}"
            logger.error("Auto-trigger runner error: %s", exc)

        self._record_event(event)
        return event

    def _record_event(self, event: TriggerEvent) -> None:
        """Record a trigger event in the history buffer."""
        with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]

    # -- Polling loop --

    def _poll_once(self) -> ModelDiff | None:
        """Run a single poll cycle.

        Returns:
            ModelDiff if changes were detected, None otherwise.
        """
        snapshot = self.take_snapshot()
        if not snapshot.models:
            return None

        with self._lock:
            old = self._last_snapshot

        # First poll -- just record baseline, do not trigger
        if not old.models:
            with self._lock:
                self._last_snapshot = snapshot
            logger.debug(
                "Auto-trigger baseline: %d models", len(snapshot.models)
            )
            return None

        diff = snapshot.diff(old)
        if not diff.has_changes:
            with self._lock:
                self._last_snapshot = snapshot
            return None

        logger.info(
            "Model changes detected: +%d -%d ~%d",
            len(diff.added), len(diff.removed), len(diff.updated),
        )

        # Trigger for new and updated models
        trigger_candidates = diff.added + diff.updated
        filtered = self._filter_trigger_models(trigger_candidates)

        event = None
        if filtered:
            trigger_type = "new_model" if diff.added else "updated_model"
            event = self._do_trigger(filtered, trigger_type)

        # Only commit the new snapshot once the change has been
        # acted on. If the trigger was skipped (cooldown / runner busy /
        # resource guard) keep the old snapshot so the model is re-detected on
        # the next poll once the skip condition clears; previously the change
        # was consumed and the model was never benchmarked. No candidates
        # (removal-only or filtered out) is safe to commit.
        if event is not None and event.skipped:
            return diff
        with self._lock:
            self._last_snapshot = snapshot

        return diff

    def _polling_loop(self) -> None:
        """Main polling loop -- runs in a daemon thread."""
        logger.info("Auto-trigger polling loop started")

        # Take initial baseline snapshot
        baseline = self.take_snapshot()
        with self._lock:
            self._last_snapshot = baseline

        while not self._stop_event.is_set():
            with self._lock:
                if not self._enabled:
                    break
                interval = self._poll_interval

            try:
                self._poll_once()
            except Exception as exc:
                logger.error("Auto-trigger poll error: %s", exc)

            self._stop_event.wait(timeout=interval)

        logger.info("Auto-trigger polling loop stopped")

    # -- Thread lifecycle --

    def start(self) -> bool:
        """Start the polling thread if enabled and not already running."""
        if not self.enabled:
            logger.debug("Auto-trigger not enabled, not starting")
            return False

        if self.running:
            logger.debug("Auto-trigger already running")
            return True

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._polling_loop,
            daemon=True,
            name="benchmark-auto-trigger",
        )
        self._thread.start()
        return True

    def stop(self) -> bool:
        """Stop the polling thread gracefully."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None
        return True

    def test_poll(self) -> dict:
        """Run a single poll without triggering any benchmark.

        Useful for verifying connectivity and checking whether changes
        would be detected. Does NOT modify the stored snapshot.

        Returns:
            Dict with snapshot_models count, diff info, and connection ok.
        """
        try:
            snapshot = self.take_snapshot()
        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc),
                "snapshot_models": 0,
                "diff": None,
            }

        if not snapshot.models:
            return {
                "ok": False,
                "error": "No models returned from Ollama",
                "snapshot_models": 0,
                "diff": None,
            }

        with self._lock:
            old = self._last_snapshot

        diff_result = None
        if old.models:
            diff = snapshot.diff(old)
            diff_result = {
                "added": diff.added,
                "removed": diff.removed,
                "updated": diff.updated,
                "has_changes": diff.has_changes,
            }

        return {
            "ok": True,
            "error": "",
            "snapshot_models": len(snapshot.models),
            "model_names": sorted(snapshot.models.keys()),
            "diff": diff_result,
        }

    def reset_snapshot(self) -> None:
        """Reset the known model snapshot (useful after manual changes)."""
        snapshot = self.take_snapshot()
        with self._lock:
            self._last_snapshot = snapshot
            self._events.clear()


def _default_ollama_list() -> list:
    """Default function to list Ollama models."""
    try:
        import ollama
        response = ollama.list()
        if hasattr(response, "models"):
            return response.models or []
        if isinstance(response, dict):
            return response.get("models", [])
        return list(response) if response else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

try:
    auto_trigger = AutoTrigger()
    AUTO_TRIGGER_AVAILABLE = True
except Exception as e:
    logger.warning("AutoTrigger init failed: %s", e)
    auto_trigger = None  # type: ignore[assignment]
    AUTO_TRIGGER_AVAILABLE = False
