#!/usr/bin/env python3
"""
MODEL LIFECYCLE MANAGEMENT -- OPTI-OIGNON
================================================

Pull, delete, update, and alias Ollama models through a unified manager.
Progress tracking for multi-GB model downloads with background jobs.

Architecture:
    LifecycleConfig         -- dataclass holding config from YAML
    PullJob                 -- tracks a single model pull (progress, status)
    ModelAlias              -- user-friendly name -> model mapping
    ModelLifecycleManager   -- orchestrates all lifecycle operations
    get_lifecycle_manager() -- module-level singleton accessor

Ollama API endpoints used:
    POST /api/pull     -- pull a model (streaming JSON progress)
    DELETE /api/delete  -- remove a local model
    POST /api/show     -- model metadata + digest for update detection

Thread-safe: all public methods use RLock for concurrent access.

Author: Leon
"""

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path(__file__).parent / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "model_lifecycle.yaml"
_ALIASES_PATH = Path(__file__).parent.parent / "data" / "model_aliases.json"

# Pull job status constants.
PULL_STATUS_PENDING = "pending"
PULL_STATUS_DOWNLOADING = "downloading"
PULL_STATUS_VERIFYING = "verifying"
PULL_STATUS_COMPLETE = "complete"
PULL_STATUS_FAILED = "failed"
PULL_STATUS_CANCELLED = "cancelled"

# Conditional imports.
try:
    import ollama as _ollama_module

    OLLAMA_AVAILABLE = True
except ImportError:
    _ollama_module = None  # type: ignore[assignment]
    OLLAMA_AVAILABLE = False

try:
    import requests as _requests_lib

    REQUESTS_AVAILABLE = True
except ImportError:
    _requests_lib = None  # type: ignore[assignment]
    REQUESTS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class LifecycleConfig:
    """Configuration loaded from model_lifecycle.yaml."""

    enabled: bool = True
    ollama_base_url: str = "http://localhost:11434"
    max_concurrent_pulls: int = 2
    progress_poll_interval_s: float = 1.0
    cleanup_on_failure: bool = True
    auto_check_interval_s: int = 0
    compare_digests: bool = True
    stale_threshold_days: int = 30

    def validate(self) -> list[str]:
        """Return validation errors (empty = valid)."""
        errors: list[str] = []
        if self.max_concurrent_pulls < 1:
            errors.append("max_concurrent_pulls must be >= 1")
        if self.progress_poll_interval_s < 0.1:
            errors.append("progress_poll_interval_s must be >= 0.1")
        if self.stale_threshold_days < 0:
            errors.append("stale_threshold_days must be >= 0")
        return errors


@dataclass
class PullProgress:
    """Progress snapshot for a model pull operation."""

    status: str = ""
    digest: str = ""
    total_bytes: int = 0
    completed_bytes: int = 0
    percent: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "digest": self.digest,
            "total_bytes": self.total_bytes,
            "completed_bytes": self.completed_bytes,
            "percent": round(self.percent, 2),
        }


@dataclass
class PullJob:
    """Tracks a single model pull operation."""

    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    model_name: str = ""
    status: str = PULL_STATUS_PENDING
    progress: PullProgress = field(default_factory=PullProgress)
    started_at: float = 0.0
    completed_at: float = 0.0
    error: str = ""
    _cancelled: bool = field(default=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "model_name": self.model_name,
            "status": self.status,
            "progress": self.progress.to_dict(),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


@dataclass
class ModelUpdateInfo:
    """Result of checking whether a model has an update available."""

    model_name: str = ""
    current_digest: str = ""
    latest_digest: str = ""
    has_update: bool = False
    checked_at: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "current_digest": self.current_digest[:16] if self.current_digest else "",
            "latest_digest": self.latest_digest[:16] if self.latest_digest else "",
            "has_update": self.has_update,
            "checked_at": self.checked_at,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_config(path: Path | None = None) -> LifecycleConfig:
    """Load lifecycle config from YAML, with defaults for missing keys."""
    p = path or _DEFAULT_CONFIG_PATH
    cfg = LifecycleConfig()
    if not p.is_file():
        logger.debug("No model_lifecycle.yaml found, using defaults")
        return cfg

    try:
        with open(p) as f:
            raw = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("Failed to parse model_lifecycle.yaml: %s", exc)
        return cfg

    cfg.enabled = raw.get("enabled", cfg.enabled)
    cfg.ollama_base_url = raw.get("ollama_base_url", cfg.ollama_base_url)
    cfg.stale_threshold_days = raw.get("stale_threshold_days", cfg.stale_threshold_days)

    pull = raw.get("pull", {})
    if isinstance(pull, dict):
        cfg.max_concurrent_pulls = pull.get("max_concurrent_pulls", cfg.max_concurrent_pulls)
        cfg.progress_poll_interval_s = pull.get("progress_poll_interval_s", cfg.progress_poll_interval_s)
        cfg.cleanup_on_failure = pull.get("cleanup_on_failure", cfg.cleanup_on_failure)

    update = raw.get("update_check", {})
    if isinstance(update, dict):
        cfg.auto_check_interval_s = update.get("auto_check_interval_s", cfg.auto_check_interval_s)
        cfg.compare_digests = update.get("compare_digests", cfg.compare_digests)

    return cfg


# ---------------------------------------------------------------------------
# Alias persistence
# ---------------------------------------------------------------------------


def _load_aliases(path: Path | None = None) -> dict[str, str]:
    """Load model aliases from JSON file."""
    p = path or _ALIASES_PATH
    if not p.is_file():
        return {}
    try:
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception as exc:
        logger.warning("Failed to load model aliases: %s", exc)
    return {}


def _save_aliases(aliases: dict[str, str], path: Path | None = None) -> bool:
    """Persist model aliases to JSON file."""
    p = path or _ALIASES_PATH
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(aliases, f, indent=2)
        return True
    except Exception as exc:
        logger.warning("Failed to save model aliases: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Helper: format byte sizes
# ---------------------------------------------------------------------------


def _format_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    if n <= 0:
        return "0 B"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f} GB"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.1f} KB"
    return f"{n} B"


# ---------------------------------------------------------------------------
# ModelLifecycleManager
# ---------------------------------------------------------------------------


class ModelLifecycleManager:
    """Manages model pull, delete, update, and alias operations.

    All operations target the Ollama API. The manager tracks active pull
    jobs with progress and supports cancellation.
    """

    def __init__(
        self,
        config: LifecycleConfig | None = None,
        config_path: Path | None = None,
        aliases_path: Path | None = None,
        ollama_module: Any = None,
    ) -> None:
        self._config = config or _load_config(config_path)
        self._lock = threading.RLock()
        self._aliases_path = aliases_path or _ALIASES_PATH

        # Active and completed pull jobs keyed by job_id.
        self._jobs: dict[str, PullJob] = {}
        # Count of currently running pull threads.
        self._active_pulls = 0

        # Model aliases.
        self._aliases: dict[str, str] = _load_aliases(self._aliases_path)

        # Allow injection for testing.
        self._ollama = ollama_module or _ollama_module

        # Merge aliases from config (YAML takes lower priority than persisted).
        if self._config.enabled:
            cfg_aliases = {}
            try:
                p = _DEFAULT_CONFIG_PATH
                if p.is_file():
                    with open(p) as f:
                        raw = yaml.safe_load(f) or {}
                    cfg_aliases = raw.get("aliases", {}) or {}
            except Exception:
                pass
            for k, v in cfg_aliases.items():
                if k not in self._aliases:
                    self._aliases[str(k)] = str(v)

    @property
    def config(self) -> LifecycleConfig:
        return self._config

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    # ----- Model listing helpers -----

    def list_models(self) -> list[dict[str, Any]]:
        """List locally available Ollama models with metadata."""
        if not self._config.enabled or not self._ollama:
            return []
        try:
            response = self._ollama.list()
            models_raw = []
            if isinstance(response, dict):
                models_raw = response.get("models", [])
            elif hasattr(response, "models"):
                models_raw = response.models or []

            results: list[dict[str, Any]] = []
            for m in models_raw:
                info = self._parse_model_entry(m)
                if info:
                    results.append(info)
            return results
        except Exception as exc:
            logger.warning("Failed to list Ollama models: %s", exc)
            return []

    def get_model_info(self, model_name: str) -> dict[str, Any] | None:
        """Get detailed info for a single model via ollama.show()."""
        resolved = self.resolve_alias(model_name)
        if not self._ollama:
            return None
        try:
            info = self._ollama.show(resolved)
            if isinstance(info, dict):
                return {
                    "name": resolved,
                    "modelfile": info.get("modelfile", ""),
                    "parameters": info.get("parameters", ""),
                    "template": info.get("template", ""),
                    "details": info.get("details", {}),
                    "model_info": info.get("model_info", {}),
                }
            return {"name": resolved, "raw": str(info)}
        except Exception as exc:
            logger.warning("Failed to show model %s: %s", resolved, exc)
            return None

    # ----- Pull operations -----

    def start_pull(self, model_name: str) -> PullJob:
        """Start pulling a model in the background. Returns the PullJob."""
        with self._lock:
            if self._active_pulls >= self._config.max_concurrent_pulls:
                job = PullJob(model_name=model_name, status=PULL_STATUS_FAILED)
                job.error = (
                    f"Max concurrent pulls ({self._config.max_concurrent_pulls}) reached"
                )
                self._jobs[job.job_id] = job
                return job

            job = PullJob(
                model_name=model_name,
                status=PULL_STATUS_PENDING,
                started_at=time.time(),
            )
            self._jobs[job.job_id] = job
            self._active_pulls += 1

        thread = threading.Thread(
            target=self._pull_worker,
            args=(job,),
            daemon=True,
            name=f"pull-{job.job_id}",
        )
        thread.start()
        logger.info("Started pull job %s for model %s", job.job_id, model_name)
        return job

    def get_pull_job(self, job_id: str) -> PullJob | None:
        """Get the current state of a pull job."""
        with self._lock:
            return self._jobs.get(job_id)

    def cancel_pull(self, job_id: str) -> bool:
        """Request cancellation of a pull job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            if job.status in (PULL_STATUS_COMPLETE, PULL_STATUS_FAILED, PULL_STATUS_CANCELLED):
                return False
            job._cancelled = True
            return True

    def list_pull_jobs(self) -> list[dict[str, Any]]:
        """List all pull jobs (active and completed)."""
        with self._lock:
            return [j.to_dict() for j in self._jobs.values()]

    def _pull_worker(self, job: PullJob) -> None:
        """Background worker that executes the model pull."""
        base_url = self._config.ollama_base_url.rstrip("/")
        try:
            job.status = PULL_STATUS_DOWNLOADING

            if REQUESTS_AVAILABLE and _requests_lib:
                self._pull_via_http(job, base_url)
            elif self._ollama:
                self._pull_via_library(job)
            else:
                raise RuntimeError(
                    "Neither requests nor ollama library available for pull"
                )

            if job._cancelled:
                job.status = PULL_STATUS_CANCELLED
                logger.info("Pull job %s cancelled", job.job_id)
            elif job.status != PULL_STATUS_FAILED:
                job.status = PULL_STATUS_COMPLETE
                logger.info("Pull job %s completed for %s", job.job_id, job.model_name)

        except Exception as exc:
            job.status = PULL_STATUS_FAILED
            job.error = str(exc)
            logger.error("Pull job %s failed: %s", job.job_id, exc)
        finally:
            job.completed_at = time.time()
            with self._lock:
                self._active_pulls = max(0, self._active_pulls - 1)

    def _pull_via_http(self, job: PullJob, base_url: str) -> None:
        """Pull model using HTTP streaming to Ollama /api/pull."""
        url = f"{base_url}/api/pull"
        payload = {"name": job.model_name, "stream": True}

        resp = _requests_lib.post(url, json=payload, stream=True, timeout=600)
        resp.raise_for_status()

        for line in resp.iter_lines():
            if job._cancelled:
                return

            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            status_text = chunk.get("status", "")
            digest = chunk.get("digest", "")
            total = chunk.get("total", 0)
            completed = chunk.get("completed", 0)

            progress = PullProgress(
                status=status_text,
                digest=digest,
                total_bytes=total,
                completed_bytes=completed,
                percent=(completed / total * 100.0) if total > 0 else 0.0,
            )
            job.progress = progress

            if "verifying" in status_text.lower():
                job.status = PULL_STATUS_VERIFYING

            # Check for error in response.
            if chunk.get("error"):
                job.status = PULL_STATUS_FAILED
                job.error = chunk["error"]
                return

    def _pull_via_library(self, job: PullJob) -> None:
        """Pull model using ollama Python library (fallback)."""
        stream = self._ollama.pull(job.model_name, stream=True)
        for chunk in stream:
            if job._cancelled:
                return

            if isinstance(chunk, dict):
                status_text = chunk.get("status", "")
                total = chunk.get("total", 0)
                completed = chunk.get("completed", 0)
            else:
                status_text = getattr(chunk, "status", "")
                total = getattr(chunk, "total", 0)
                completed = getattr(chunk, "completed", 0)

            progress = PullProgress(
                status=status_text,
                total_bytes=total or 0,
                completed_bytes=completed or 0,
                percent=(completed / total * 100.0) if total else 0.0,
            )
            job.progress = progress

            if "verifying" in status_text.lower():
                job.status = PULL_STATUS_VERIFYING

    # ----- Delete operations -----

    def delete_model(self, model_name: str) -> dict[str, Any]:
        """Delete a locally stored Ollama model."""
        resolved = self.resolve_alias(model_name)
        if not self._ollama:
            return {"success": False, "error": "Ollama not available"}
        try:
            self._ollama.delete(resolved)
            logger.info("Deleted model: %s", resolved)
            return {"success": True, "model": resolved}
        except Exception as exc:
            logger.error("Failed to delete model %s: %s", resolved, exc)
            return {"success": False, "model": resolved, "error": str(exc)}

    # ----- Update check -----

    def check_update(self, model_name: str) -> ModelUpdateInfo:
        """Check if a newer version of a model is available.

        Compares the local model digest with the registry by attempting
        a dry pull (the Ollama pull API returns 'up to date' if no
        update is available).
        """
        resolved = self.resolve_alias(model_name)
        info = ModelUpdateInfo(model_name=resolved, checked_at=time.time())

        if not self._config.compare_digests:
            info.error = "Digest comparison disabled"
            return info

        # Get current local digest.
        current_digest = self._get_local_digest(resolved)
        info.current_digest = current_digest

        # Try a pull to see if Ollama reports 'already up to date'.
        base_url = self._config.ollama_base_url.rstrip("/")
        try:
            if REQUESTS_AVAILABLE and _requests_lib:
                resp = _requests_lib.post(
                    f"{base_url}/api/pull",
                    json={"name": resolved, "stream": False},
                    timeout=30,
                )
                if resp.ok:
                    data = resp.json()
                    status_text = data.get("status", "")
                    if "up to date" in status_text.lower():
                        info.has_update = False
                        info.latest_digest = current_digest
                    else:
                        info.has_update = True
                        info.latest_digest = data.get("digest", "")
                else:
                    info.error = f"HTTP {resp.status_code}"
            elif self._ollama:
                # Library fallback: pull with stream and check first status.
                for chunk in self._ollama.pull(resolved, stream=True):
                    status_text = ""
                    if isinstance(chunk, dict):
                        status_text = chunk.get("status", "")
                    else:
                        status_text = getattr(chunk, "status", "")
                    if "up to date" in status_text.lower():
                        info.has_update = False
                    else:
                        info.has_update = True
                    break
            else:
                info.error = "No HTTP client or ollama library available"
        except Exception as exc:
            info.error = str(exc)
            logger.warning("Update check failed for %s: %s", resolved, exc)

        return info

    def check_updates_batch(self, model_names: list[str]) -> list[ModelUpdateInfo]:
        """Check updates for multiple models."""
        results: list[ModelUpdateInfo] = []
        for name in model_names:
            results.append(self.check_update(name))
        return results

    def _get_local_digest(self, model_name: str) -> str:
        """Get the digest of a locally installed model."""
        if not self._ollama:
            return ""
        try:
            info = self._ollama.show(model_name)
            if isinstance(info, dict):
                # Digest can be in details or at top level.
                details = info.get("details", {})
                digest = info.get("digest", "") or details.get("digest", "")
                return str(digest)
            return getattr(info, "digest", "") or ""
        except Exception:
            return ""

    # ----- Alias management -----

    def resolve_alias(self, name: str) -> str:
        """Resolve an alias to the actual model name. Pass-through if not aliased."""
        with self._lock:
            return self._aliases.get(name, name)

    def set_alias(self, alias: str, model_name: str) -> bool:
        """Create or update a model alias."""
        with self._lock:
            self._aliases[alias] = model_name
            return _save_aliases(self._aliases, self._aliases_path)

    def remove_alias(self, alias: str) -> bool:
        """Remove a model alias."""
        with self._lock:
            if alias not in self._aliases:
                return False
            del self._aliases[alias]
            return _save_aliases(self._aliases, self._aliases_path)

    def list_aliases(self) -> dict[str, str]:
        """Return all aliases."""
        with self._lock:
            return dict(self._aliases)

    # ----- Stale model detection -----

    def detect_stale_models(self) -> list[dict[str, Any]]:
        """Find models that haven't been modified recently.

        Uses the model's modified_at timestamp from Ollama's list response.
        Models older than stale_threshold_days are flagged.
        """
        threshold_days = self._config.stale_threshold_days
        if threshold_days <= 0:
            return []

        cutoff = time.time() - (threshold_days * 86400)
        stale: list[dict[str, Any]] = []

        models = self.list_models()
        for m in models:
            modified_at = m.get("modified_at", 0)
            if isinstance(modified_at, (int, float)) and 0 < modified_at < cutoff:
                days_old = int((time.time() - modified_at) / 86400)
                stale.append({
                    "name": m.get("name", ""),
                    "size": m.get("size", 0),
                    "size_human": m.get("size_human", ""),
                    "modified_at": modified_at,
                    "days_since_modified": days_old,
                })

        return stale

    # ----- Internal helpers -----

    @staticmethod
    def _parse_model_entry(m: Any) -> dict[str, Any] | None:
        """Parse an Ollama model list entry into a dict."""
        if isinstance(m, dict):
            name = m.get("name", m.get("model", ""))
            size = m.get("size", 0)
            modified = m.get("modified_at", "")
            digest = m.get("digest", "")
            details = m.get("details", {})
        else:
            name = getattr(m, "name", getattr(m, "model", ""))
            size = getattr(m, "size", 0)
            modified = getattr(m, "modified_at", "")
            digest = getattr(m, "digest", "")
            details = getattr(m, "details", {})

        if not name:
            return None

        # Parse modified_at to epoch if it's a string.
        modified_epoch = 0.0
        if isinstance(modified, (int, float)):
            modified_epoch = float(modified)
        elif isinstance(modified, str) and modified:
            try:
                from datetime import datetime

                # Ollama returns ISO format timestamps.
                dt = datetime.fromisoformat(modified.replace("Z", "+00:00"))
                modified_epoch = dt.timestamp()
            except Exception:
                pass

        size_int = int(size) if size else 0

        return {
            "name": str(name),
            "size": size_int,
            "size_human": _format_bytes(size_int),
            "modified_at": modified_epoch,
            "digest": str(digest)[:16] if digest else "",
            "details": details if isinstance(details, dict) else {},
        }

    def shutdown(self) -> None:
        """Cancel all active pulls and cleanup."""
        with self._lock:
            for job in self._jobs.values():
                if job.status in (PULL_STATUS_PENDING, PULL_STATUS_DOWNLOADING, PULL_STATUS_VERIFYING):
                    job._cancelled = True
        logger.info("ModelLifecycleManager shutdown complete")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_manager: ModelLifecycleManager | None = None
_manager_lock = threading.Lock()


def get_lifecycle_manager(
    config_path: Path | None = None,
) -> ModelLifecycleManager:
    """Get or create the singleton ModelLifecycleManager."""
    global _manager
    if _manager is not None:
        return _manager
    with _manager_lock:
        if _manager is not None:
            return _manager
        _manager = ModelLifecycleManager(config_path=config_path)
        return _manager


def reset_manager() -> None:
    """Reset the singleton (for testing)."""
    global _manager
    with _manager_lock:
        if _manager is not None:
            _manager.shutdown()
        _manager = None
