#!/usr/bin/env python3
"""
INFERENCE AUTO-TUNER -- OPTI-OIGNON S110 + S111
=================================================

Automatically finds optimal inference parameters (batch size, threads,
GPU layers, flash attention) for the user's hardware. Runs on demand,
persists best config per model. Inspired by llama-optimus.

Architecture:
    TunerConfig         -- dataclass holding tuner settings
    ParameterSpace      -- defines the search grid
    BenchmarkResult     -- single benchmark run result
    TunerProfile        -- best params for a model + hardware fingerprint
    AutoTuner           -- orchestrates parameter sweep + hill climbing
    AutoTunerManager    -- singleton managing tuner state + persistence

S111 additions:
    create_ollama_benchmark_fn()    -- real benchmark via Ollama API
    create_llamacpp_benchmark_fn()  -- real benchmark via llama-cpp-python

No external optimizer dependency (no Optuna). Uses simple parameter
sweep with hill-climbing refinement for robustness on consumer hardware.
"""

import hashlib
import json
import logging
import os
import platform
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path(__file__).parent / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "auto_tuner.yaml"
_RESULTS_PATH = Path(__file__).parent.parent / "data" / "tuner_results.json"

# Default parameter search space.
_DEFAULT_PARAM_SPACE: dict[str, list] = {
    "batch_size": [512, 1024, 2048, 4096],
    "ubatch_size": [256, 512, 1024],
    "threads": [2, 4, 6, 8],
    "flash_attention": [True, False],
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TunerConfig:
    """Configuration for the auto-tuner."""

    enabled: bool = True
    warmup_runs: int = 3
    benchmark_tokens: int = 128
    benchmark_prompt_tokens: int = 128
    trials_per_param: int = 3
    auto_apply: bool = False

    def validate(self) -> list[str]:
        """Return validation errors (empty = valid)."""
        errors: list[str] = []
        if self.warmup_runs < 0:
            errors.append("warmup_runs must be >= 0")
        if self.benchmark_tokens < 1:
            errors.append("benchmark_tokens must be >= 1")
        if self.benchmark_prompt_tokens < 1:
            errors.append("benchmark_prompt_tokens must be >= 1")
        if self.trials_per_param < 1:
            errors.append("trials_per_param must be >= 1")
        return errors

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "enabled": self.enabled,
            "warmup_runs": self.warmup_runs,
            "benchmark_tokens": self.benchmark_tokens,
            "benchmark_prompt_tokens": self.benchmark_prompt_tokens,
            "trials_per_param": self.trials_per_param,
            "auto_apply": self.auto_apply,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TunerConfig":
        """Create from dict, ignoring unknown keys."""
        known = {
            "enabled", "warmup_runs", "benchmark_tokens",
            "benchmark_prompt_tokens", "trials_per_param", "auto_apply",
        }
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class ParameterSpace:
    """Defines the grid of parameters to search."""

    batch_size: list[int] = field(default_factory=lambda: [512, 1024, 2048, 4096])
    ubatch_size: list[int] = field(default_factory=lambda: [256, 512, 1024])
    threads: list[int] = field(default_factory=lambda: [2, 4, 6, 8])
    flash_attention: list[bool] = field(default_factory=lambda: [True, False])

    def total_combinations(self) -> int:
        """Total number of parameter combinations in the grid."""
        return (
            len(self.batch_size)
            * len(self.ubatch_size)
            * len(self.threads)
            * len(self.flash_attention)
        )

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "batch_size": self.batch_size,
            "ubatch_size": self.ubatch_size,
            "threads": self.threads,
            "flash_attention": self.flash_attention,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ParameterSpace":
        """Create from dict."""
        return cls(
            batch_size=data.get("batch_size", [512, 1024, 2048, 4096]),
            ubatch_size=data.get("ubatch_size", [256, 512, 1024]),
            threads=data.get("threads", [2, 4, 6, 8]),
            flash_attention=data.get("flash_attention", [True, False]),
        )


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    params: dict = field(default_factory=dict)
    tokens_per_second_tg: float = 0.0  # Token generation speed
    tokens_per_second_pp: float = 0.0  # Prompt processing speed
    total_time_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "params": self.params,
            "tokens_per_second_tg": round(self.tokens_per_second_tg, 2),
            "tokens_per_second_pp": round(self.tokens_per_second_pp, 2),
            "total_time_ms": round(self.total_time_ms, 2),
            "error": self.error,
        }


@dataclass
class TunerProfile:
    """Best parameters for a specific model on specific hardware."""

    model_name: str = ""
    best_params: dict = field(default_factory=dict)
    best_tg_speed: float = 0.0
    best_pp_speed: float = 0.0
    baseline_tg_speed: float = 0.0
    baseline_pp_speed: float = 0.0
    speedup_factor: float = 1.0
    hardware_fingerprint: str = ""
    timestamp: float = 0.0
    all_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "model_name": self.model_name,
            "best_params": self.best_params,
            "best_tg_speed": round(self.best_tg_speed, 2),
            "best_pp_speed": round(self.best_pp_speed, 2),
            "baseline_tg_speed": round(self.baseline_tg_speed, 2),
            "baseline_pp_speed": round(self.baseline_pp_speed, 2),
            "speedup_factor": round(self.speedup_factor, 2),
            "hardware_fingerprint": self.hardware_fingerprint,
            "timestamp": self.timestamp,
            "all_results": self.all_results,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TunerProfile":
        """Create from dict."""
        return cls(
            model_name=data.get("model_name", ""),
            best_params=data.get("best_params", {}),
            best_tg_speed=data.get("best_tg_speed", 0.0),
            best_pp_speed=data.get("best_pp_speed", 0.0),
            baseline_tg_speed=data.get("baseline_tg_speed", 0.0),
            baseline_pp_speed=data.get("baseline_pp_speed", 0.0),
            speedup_factor=data.get("speedup_factor", 1.0),
            hardware_fingerprint=data.get("hardware_fingerprint", ""),
            timestamp=data.get("timestamp", 0.0),
            all_results=data.get("all_results", []),
        )


@dataclass
class TunerJob:
    """Represents a running or completed tuner job."""

    job_id: str = ""
    model_name: str = ""
    status: str = "pending"  # pending, running, completed, failed, cancelled
    progress: float = 0.0  # 0.0 to 1.0
    current_step: str = ""
    total_steps: int = 0
    completed_steps: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    result: Optional[TunerProfile] = None
    error: str = ""

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "job_id": self.job_id,
            "model_name": self.model_name,
            "status": self.status,
            "progress": round(self.progress, 3),
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result.to_dict() if self.result else None,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Hardware fingerprint
# ---------------------------------------------------------------------------

def get_hardware_fingerprint() -> str:
    """Generate a fingerprint of the current hardware.

    Includes CPU info, thread count, and platform. Used to detect
    when tuning results may be stale due to hardware changes.
    """
    parts = [
        platform.machine(),
        platform.processor() or "unknown_cpu",
        str(os.cpu_count() or 0),
        platform.system(),
    ]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Tuner Recommendations (S112)
# ---------------------------------------------------------------------------


@dataclass
class TunerRecommendation:
    """A single actionable optimization recommendation.

    Generated by analyzing a TunerProfile's results: comparing
    baseline vs. tuned, identifying which parameters matter most,
    and producing human-readable advice.
    """

    title: str = ""
    description: str = ""
    parameter: str = ""
    current_value: Any = None
    recommended_value: Any = None
    estimated_speedup: float = 1.0
    confidence: str = "medium"  # low, medium, high
    category: str = "performance"  # performance, memory, quality
    applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "title": self.title,
            "description": self.description,
            "parameter": self.parameter,
            "current_value": self.current_value,
            "recommended_value": self.recommended_value,
            "estimated_speedup": round(self.estimated_speedup, 2),
            "confidence": self.confidence,
            "category": self.category,
            "applied": self.applied,
        }


def generate_recommendations(profile: TunerProfile) -> list[TunerRecommendation]:
    """Analyze a tuning profile and generate actionable recommendations.

    Examines baseline vs. best parameters, identifies the most impactful
    changes, and produces human-readable advice sorted by estimated
    speedup (highest first).

    Args:
        profile: A completed TunerProfile with all_results populated.

    Returns:
        List of TunerRecommendation objects, sorted by estimated speedup.
    """
    recommendations: list[TunerRecommendation] = []

    if not profile.best_params or not profile.all_results:
        return recommendations

    best = profile.best_params
    baseline_speed = profile.baseline_tg_speed
    best_speed = profile.best_tg_speed

    # Overall speedup recommendation.
    if best_speed > 0 and baseline_speed > 0:
        overall_speedup = best_speed / baseline_speed
        if overall_speedup > 1.05:
            confidence = "high" if len(profile.all_results) >= 5 else "medium"
            recommendations.append(TunerRecommendation(
                title="Apply tuned parameters",
                description=(
                    f"Tuning found a {overall_speedup:.1f}x speedup "
                    f"({baseline_speed:.1f} -> {best_speed:.1f} tok/s). "
                    f"Apply the optimized configuration for best performance."
                ),
                parameter="all",
                current_value="default",
                recommended_value=best,
                estimated_speedup=overall_speedup,
                confidence=confidence,
                category="performance",
            ))

    # Per-parameter analysis: identify which changes matter most.
    _analyze_threads(profile, recommendations)
    _analyze_batch_size(profile, recommendations)
    _analyze_flash_attention(profile, recommendations)
    _analyze_gpu_layers(profile, recommendations)

    # Sort by estimated speedup descending.
    recommendations.sort(key=lambda r: r.estimated_speedup, reverse=True)

    return recommendations


def _analyze_threads(
    profile: TunerProfile, recs: list[TunerRecommendation]
) -> None:
    """Check if thread count significantly impacts performance."""
    best_threads = profile.best_params.get("threads")
    if best_threads is None:
        return

    # Find results with different thread counts, holding other params fixed.
    thread_speeds: dict[int, list[float]] = {}
    for r in profile.all_results:
        params = r.get("params", {})
        tg = r.get("tokens_per_second_tg", 0.0)
        t = params.get("threads")
        if t is not None and tg > 0:
            thread_speeds.setdefault(int(t), []).append(tg)

    if len(thread_speeds) < 2:
        return

    # Average speed per thread count.
    avg_by_threads = {
        t: sum(speeds) / len(speeds) for t, speeds in thread_speeds.items()
    }
    best_t = max(avg_by_threads, key=lambda t: avg_by_threads[t])
    worst_t = min(avg_by_threads, key=lambda t: avg_by_threads[t])

    if avg_by_threads[worst_t] > 0:
        speedup = avg_by_threads[best_t] / avg_by_threads[worst_t]
        if speedup > 1.1 and best_t != worst_t:
            cpu_count = os.cpu_count() or 0
            recs.append(TunerRecommendation(
                title=f"Set threads to {best_t}",
                description=(
                    f"Thread count of {best_t} gives {speedup:.1f}x more throughput "
                    f"than {worst_t} threads on your {cpu_count}-core system."
                ),
                parameter="threads",
                current_value=worst_t,
                recommended_value=best_t,
                estimated_speedup=speedup,
                confidence="high" if len(thread_speeds[best_t]) >= 3 else "medium",
                category="performance",
            ))


def _analyze_batch_size(
    profile: TunerProfile, recs: list[TunerRecommendation]
) -> None:
    """Check if batch size significantly impacts performance."""
    best_batch = profile.best_params.get("batch_size")
    if best_batch is None:
        return

    batch_speeds: dict[int, list[float]] = {}
    for r in profile.all_results:
        params = r.get("params", {})
        tg = r.get("tokens_per_second_tg", 0.0)
        b = params.get("batch_size")
        if b is not None and tg > 0:
            batch_speeds.setdefault(int(b), []).append(tg)

    if len(batch_speeds) < 2:
        return

    avg_by_batch = {
        b: sum(speeds) / len(speeds) for b, speeds in batch_speeds.items()
    }
    best_b = max(avg_by_batch, key=lambda b: avg_by_batch[b])
    worst_b = min(avg_by_batch, key=lambda b: avg_by_batch[b])

    if avg_by_batch[worst_b] > 0:
        speedup = avg_by_batch[best_b] / avg_by_batch[worst_b]
        if speedup > 1.1:
            recs.append(TunerRecommendation(
                title=f"Use batch size {best_b}",
                description=(
                    f"Batch size {best_b} is {speedup:.1f}x faster than {worst_b}. "
                    f"Larger batches improve throughput if you have enough memory."
                ),
                parameter="batch_size",
                current_value=worst_b,
                recommended_value=best_b,
                estimated_speedup=speedup,
                confidence="medium",
                category="performance",
            ))


def _analyze_flash_attention(
    profile: TunerProfile, recs: list[TunerRecommendation]
) -> None:
    """Check if flash attention helps."""
    fa_speeds: dict[bool, list[float]] = {}
    for r in profile.all_results:
        params = r.get("params", {})
        tg = r.get("tokens_per_second_tg", 0.0)
        fa = params.get("flash_attention")
        if fa is not None and tg > 0:
            fa_speeds.setdefault(bool(fa), []).append(tg)

    if True not in fa_speeds or False not in fa_speeds:
        return

    avg_on = sum(fa_speeds[True]) / len(fa_speeds[True])
    avg_off = sum(fa_speeds[False]) / len(fa_speeds[False])

    if avg_off > 0:
        if avg_on > avg_off * 1.05:
            speedup = avg_on / avg_off
            recs.append(TunerRecommendation(
                title="Enable flash attention",
                description=(
                    f"Flash attention provides {speedup:.1f}x speedup "
                    f"and reduces memory usage for long contexts."
                ),
                parameter="flash_attention",
                current_value=False,
                recommended_value=True,
                estimated_speedup=speedup,
                confidence="high",
                category="performance",
            ))
        elif avg_off > avg_on * 1.05:
            speedup = avg_off / avg_on
            recs.append(TunerRecommendation(
                title="Disable flash attention",
                description=(
                    f"Flash attention is {speedup:.1f}x slower on your hardware. "
                    f"Your GPU may not benefit from this feature."
                ),
                parameter="flash_attention",
                current_value=True,
                recommended_value=False,
                estimated_speedup=speedup,
                confidence="medium",
                category="performance",
            ))


def _analyze_gpu_layers(
    profile: TunerProfile, recs: list[TunerRecommendation]
) -> None:
    """Check if GPU layer count significantly affects performance."""
    gl_speeds: dict[int, list[float]] = {}
    for r in profile.all_results:
        params = r.get("params", {})
        tg = r.get("tokens_per_second_tg", 0.0)
        gl = params.get("gpu_layers")
        if gl is not None and tg > 0:
            gl_speeds.setdefault(int(gl), []).append(tg)

    if len(gl_speeds) < 2:
        return

    avg_by_gl = {
        gl: sum(speeds) / len(speeds) for gl, speeds in gl_speeds.items()
    }
    best_gl = max(avg_by_gl, key=lambda g: avg_by_gl[g])
    worst_gl = min(avg_by_gl, key=lambda g: avg_by_gl[g])

    if avg_by_gl[worst_gl] > 0:
        speedup = avg_by_gl[best_gl] / avg_by_gl[worst_gl]
        if speedup > 1.15:
            recs.append(TunerRecommendation(
                title=f"Set GPU layers to {best_gl}",
                description=(
                    f"Offloading {best_gl} layers to GPU gives {speedup:.1f}x speedup "
                    f"over {worst_gl} layers. More layers on GPU = faster inference."
                ),
                parameter="gpu_layers",
                current_value=worst_gl,
                recommended_value=best_gl,
                estimated_speedup=speedup,
                confidence="high" if len(gl_speeds[best_gl]) >= 2 else "medium",
                category="performance",
            ))


# ---------------------------------------------------------------------------
# Auto-Tuner Engine
# ---------------------------------------------------------------------------

class AutoTuner:
    """Parameter sweep engine with hill-climbing refinement.

    Runs benchmarks across a parameter grid, measures tokens/sec,
    and identifies the fastest configuration. No external optimizer
    dependencies (no Optuna, no scipy).

    The tuner works with a benchmark function that accepts a parameter
    dict and returns a BenchmarkResult. This allows it to be used with
    any inference backend.
    """

    def __init__(
        self,
        config: TunerConfig,
        param_space: ParameterSpace,
        benchmark_fn: Optional[Callable[[dict], BenchmarkResult]] = None,
        progress_fn: Optional[Callable[[TunerJob], None]] = None,
    ):
        self._config = config
        self._param_space = param_space
        self._benchmark_fn = benchmark_fn
        self._progress_fn = progress_fn
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation of the current tuning run."""
        self._cancelled = True

    def run(self, model_name: str, job: TunerJob) -> TunerProfile:
        """Execute the full tuning process.

        Args:
            model_name: Name of the model being tuned.
            job: TunerJob to update with progress.

        Returns:
            TunerProfile with best parameters found.

        Raises:
            RuntimeError: If no benchmark function is set.
            ValueError: If cancelled during execution.
        """
        if self._benchmark_fn is None:
            raise RuntimeError("No benchmark function provided")

        self._cancelled = False
        job.status = "running"
        job.started_at = time.time()
        job.model_name = model_name

        # Calculate total steps.
        # Phase 1: warmup runs
        # Phase 2: parameter sweep (smart subset, not full grid)
        # Phase 3: best-of refinement
        sweep_combos = self._build_smart_sweep()
        total = (
            self._config.warmup_runs
            + len(sweep_combos) * self._config.trials_per_param
            + self._config.trials_per_param  # refinement of best
        )
        job.total_steps = total
        self._report_progress(job)

        try:
            # Phase 1: Warmup
            job.current_step = "Warming up..."
            self._report_progress(job)
            default_params = self._default_params()
            for i in range(self._config.warmup_runs):
                self._check_cancelled()
                self._benchmark_fn(default_params)
                job.completed_steps += 1
                job.progress = job.completed_steps / max(job.total_steps, 1)
                self._report_progress(job)

            # Baseline measurement
            job.current_step = "Measuring baseline..."
            self._report_progress(job)
            baseline = self._run_averaged(default_params)

            # Phase 2: Parameter sweep
            all_results: list[BenchmarkResult] = [baseline]
            best = baseline

            for idx, params in enumerate(sweep_combos):
                self._check_cancelled()
                job.current_step = f"Testing config {idx + 1}/{len(sweep_combos)}"
                job.progress = job.completed_steps / max(job.total_steps, 1)
                self._report_progress(job)

                result = self._run_averaged(params)
                all_results.append(result)

                if not result.error and result.tokens_per_second_tg > best.tokens_per_second_tg:
                    best = result

                job.completed_steps += self._config.trials_per_param
                job.progress = job.completed_steps / max(job.total_steps, 1)
                self._report_progress(job)

            # Phase 3: Refinement — re-confirm the best with extra runs
            job.current_step = "Confirming best configuration..."
            self._report_progress(job)
            confirmed = self._run_averaged(
                best.params, extra_trials=self._config.trials_per_param
            )
            if not confirmed.error:
                best = confirmed
            job.completed_steps = job.total_steps
            job.progress = 1.0

            # Build profile
            speedup = (
                best.tokens_per_second_tg / baseline.tokens_per_second_tg
                if baseline.tokens_per_second_tg > 0
                else 1.0
            )

            profile = TunerProfile(
                model_name=model_name,
                best_params=best.params,
                best_tg_speed=best.tokens_per_second_tg,
                best_pp_speed=best.tokens_per_second_pp,
                baseline_tg_speed=baseline.tokens_per_second_tg,
                baseline_pp_speed=baseline.tokens_per_second_pp,
                speedup_factor=speedup,
                hardware_fingerprint=get_hardware_fingerprint(),
                timestamp=time.time(),
                all_results=[r.to_dict() for r in all_results],
            )

            job.status = "completed"
            job.result = profile
            job.finished_at = time.time()
            job.current_step = f"Done! Best: {best.tokens_per_second_tg:.1f} tok/s ({speedup:.2f}x)"
            self._report_progress(job)

            return profile

        except ValueError as exc:
            # Cancelled
            job.status = "cancelled"
            job.error = str(exc)
            job.finished_at = time.time()
            self._report_progress(job)
            raise
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.finished_at = time.time()
            self._report_progress(job)
            raise

    def _build_smart_sweep(self) -> list[dict]:
        """Build a smart subset of parameter combinations.

        Instead of testing the full cartesian product (which can be
        huge), we test each parameter axis independently while keeping
        others at defaults. This reduces from O(n^4) to O(n) while
        still finding near-optimal configs in practice.
        """
        defaults = self._default_params()
        combos: list[dict] = []
        seen: set[str] = set()

        # Sweep each axis independently.
        for bs in self._param_space.batch_size:
            p = {**defaults, "batch_size": bs}
            key = _param_key(p)
            if key not in seen:
                combos.append(p)
                seen.add(key)

        for ubs in self._param_space.ubatch_size:
            p = {**defaults, "ubatch_size": ubs}
            key = _param_key(p)
            if key not in seen:
                combos.append(p)
                seen.add(key)

        for t in self._param_space.threads:
            p = {**defaults, "threads": t}
            key = _param_key(p)
            if key not in seen:
                combos.append(p)
                seen.add(key)

        for fa in self._param_space.flash_attention:
            p = {**defaults, "flash_attention": fa}
            key = _param_key(p)
            if key not in seen:
                combos.append(p)
                seen.add(key)

        return combos

    def _default_params(self) -> dict:
        """Return default (middle-of-range) parameters."""
        def mid(lst: list) -> Any:
            return lst[len(lst) // 2] if lst else None

        return {
            "batch_size": mid(self._param_space.batch_size) or 1024,
            "ubatch_size": mid(self._param_space.ubatch_size) or 512,
            "threads": mid(self._param_space.threads) or 4,
            "flash_attention": True,
        }

    def _run_averaged(
        self, params: dict, extra_trials: int = 0
    ) -> BenchmarkResult:
        """Run the benchmark multiple times and return averaged result."""
        trials = self._config.trials_per_param + extra_trials
        tg_speeds: list[float] = []
        pp_speeds: list[float] = []
        total_times: list[float] = []
        last_error = ""

        for _ in range(trials):
            self._check_cancelled()
            result = self._benchmark_fn(params)
            if result.error:
                last_error = result.error
                continue
            tg_speeds.append(result.tokens_per_second_tg)
            pp_speeds.append(result.tokens_per_second_pp)
            total_times.append(result.total_time_ms)

        if not tg_speeds:
            return BenchmarkResult(
                params=params,
                error=last_error or "All trials failed",
            )

        return BenchmarkResult(
            params=params,
            tokens_per_second_tg=sum(tg_speeds) / len(tg_speeds),
            tokens_per_second_pp=sum(pp_speeds) / len(pp_speeds),
            total_time_ms=sum(total_times) / len(total_times),
        )

    def _check_cancelled(self) -> None:
        """Raise ValueError if cancellation was requested."""
        if self._cancelled:
            raise ValueError("Tuning cancelled by user")

    def _report_progress(self, job: TunerJob) -> None:
        """Report progress via callback if available."""
        if self._progress_fn is not None:
            try:
                self._progress_fn(job)
            except Exception as exc:
                logger.debug("Progress callback error: %s", exc)


# ---------------------------------------------------------------------------
# Auto-Tuner Manager (singleton)
# ---------------------------------------------------------------------------

class AutoTunerManager:
    """Manages tuner configuration, job execution, and result persistence.

    This is the main entry point for the auto-tuner feature. It handles
    configuration loading, result storage, job lifecycle, and provides
    the API surface used by route handlers.
    """

    def __init__(self, config_path: Optional[str] = None):
        self._config = TunerConfig()
        self._param_space = ParameterSpace()
        self._profiles: dict[str, TunerProfile] = {}
        self._active_jobs: dict[str, TunerJob] = {}
        self._active_tuners: dict[str, AutoTuner] = {}
        self._lock = threading.RLock()
        self._load_config(config_path)

    def _load_config(self, config_path: Optional[str] = None) -> None:
        """Load configuration from YAML."""
        p = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        if not p.is_file():
            logger.debug("No auto_tuner.yaml found at %s", p)
            return

        try:
            with open(p, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning("Failed to load auto_tuner.yaml: %s", exc)
            return

        at_cfg = raw.get("auto_tuner", {})
        if isinstance(at_cfg, dict):
            self._config = TunerConfig.from_dict(at_cfg)

        ps_cfg = raw.get("parameter_space", {})
        if isinstance(ps_cfg, dict):
            self._param_space = ParameterSpace.from_dict(ps_cfg)

        self._load_results()

        logger.info(
            "Auto-tuner config loaded: enabled=%s, warmup=%d, trials=%d",
            self._config.enabled, self._config.warmup_runs,
            self._config.trials_per_param,
        )

    def _load_results(self) -> None:
        """Load persisted tuning results from disk."""
        if not _RESULTS_PATH.is_file():
            return
        try:
            with open(_RESULTS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for key, val in data.items():
                    if isinstance(val, dict):
                        self._profiles[key] = TunerProfile.from_dict(val)
        except Exception as exc:
            logger.debug("Failed to load tuner results: %s", exc)

    def _save_results(self) -> None:
        """Persist tuning results to disk."""
        try:
            _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {k: v.to_dict() for k, v in self._profiles.items()}
            with open(_RESULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            logger.debug("Failed to save tuner results: %s", exc)

    # -- Public API --

    @property
    def config(self) -> TunerConfig:
        """Current configuration."""
        with self._lock:
            return TunerConfig.from_dict(self._config.to_dict())

    @property
    def param_space(self) -> ParameterSpace:
        """Current parameter search space."""
        with self._lock:
            return ParameterSpace.from_dict(self._param_space.to_dict())

    def get_status(self) -> dict:
        """Get tuner status including config and active jobs."""
        with self._lock:
            return {
                "config": self._config.to_dict(),
                "param_space": self._param_space.to_dict(),
                "active_jobs": {
                    k: v.to_dict() for k, v in self._active_jobs.items()
                },
                "saved_profiles": list(self._profiles.keys()),
                "available": True,
            }

    def list_results(self) -> dict[str, dict]:
        """List all tuning results (per model)."""
        with self._lock:
            return {k: v.to_dict() for k, v in self._profiles.items()}

    def get_result(self, model_name: str) -> Optional[TunerProfile]:
        """Get best config for a specific model."""
        with self._lock:
            profile = self._profiles.get(model_name)
            if profile is None:
                return None
            return TunerProfile.from_dict(profile.to_dict())

    def delete_result(self, model_name: str) -> bool:
        """Delete tuning data for a model."""
        with self._lock:
            if model_name not in self._profiles:
                return False
            del self._profiles[model_name]
            self._save_results()
            return True

    def start_tuning(
        self,
        model_name: str,
        benchmark_fn: Callable[[dict], BenchmarkResult],
        progress_fn: Optional[Callable[[TunerJob], None]] = None,
    ) -> TunerJob:
        """Start a tuning session for a model.

        Args:
            model_name: Model to tune.
            benchmark_fn: Function that runs a benchmark with given params.
            progress_fn: Optional callback for progress updates.

        Returns:
            TunerJob that will be updated as tuning progresses.

        Raises:
            ValueError: If tuning is already running for this model.
        """
        with self._lock:
            # Check for active job.
            existing = self._active_jobs.get(model_name)
            if existing and existing.status == "running":
                raise ValueError(
                    f"Tuning already running for model: {model_name}"
                )

            job = TunerJob(
                job_id=str(uuid.uuid4()),
                model_name=model_name,
                status="pending",
            )
            self._active_jobs[model_name] = job

            tuner = AutoTuner(
                config=self._config,
                param_space=self._param_space,
                benchmark_fn=benchmark_fn,
                progress_fn=progress_fn,
            )
            self._active_tuners[model_name] = tuner

        # Run in a background thread.
        thread = threading.Thread(
            target=self._run_tuning_thread,
            args=(model_name, tuner, job),
            daemon=True,
            name=f"tuner-{model_name}",
        )
        thread.start()

        return job

    def cancel_tuning(self, model_name: str) -> bool:
        """Cancel an active tuning session."""
        with self._lock:
            tuner = self._active_tuners.get(model_name)
            if tuner is None:
                return False
            tuner.cancel()
            return True

    def get_job(self, model_name: str) -> Optional[TunerJob]:
        """Get the current/last job for a model."""
        with self._lock:
            job = self._active_jobs.get(model_name)
            if job is None:
                return None
            # Return a snapshot.
            return job

    def apply_result(self, model_name: str) -> Optional[dict]:
        """Get the best params for a model (for manual application).

        Returns the best_params dict, or None if no profile exists.
        """
        with self._lock:
            profile = self._profiles.get(model_name)
            if profile is None:
                return None
            return dict(profile.best_params)

    def _run_tuning_thread(
        self, model_name: str, tuner: AutoTuner, job: TunerJob
    ) -> None:
        """Thread target for running tuning."""
        try:
            profile = tuner.run(model_name, job)
            with self._lock:
                self._profiles[model_name] = profile
                self._save_results()
        except ValueError:
            # Cancelled — job already updated by tuner.
            pass
        except Exception as exc:
            logger.error("Tuning failed for %s: %s", model_name, exc)
            job.status = "failed"
            job.error = str(exc)
            job.finished_at = time.time()
        finally:
            with self._lock:
                self._active_tuners.pop(model_name, None)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_manager: Optional[AutoTunerManager] = None
_init_lock = threading.Lock()


def get_auto_tuner_manager(
    config_path: Optional[str] = None,
) -> AutoTunerManager:
    """Get or create the module-level singleton manager."""
    global _manager
    if _manager is not None:
        return _manager
    with _init_lock:
        if _manager is not None:
            return _manager
        _manager = AutoTunerManager(config_path=config_path)
        return _manager


def reset_manager() -> None:
    """Reset the singleton (for testing)."""
    global _manager
    with _init_lock:
        _manager = None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _param_key(params: dict) -> str:
    """Create a hashable key from a parameter dict."""
    parts = sorted(f"{k}={v}" for k, v in params.items())
    return "|".join(parts)


def create_mock_benchmark_fn(
    base_speed: float = 30.0,
    variance: float = 5.0,
) -> Callable[[dict], BenchmarkResult]:
    """Create a mock benchmark function for testing.

    Returns a function that simulates benchmark results with some
    variance based on parameter values. Higher batch sizes and thread
    counts give slightly better results (up to a point).
    """
    import random

    def _mock_benchmark(params: dict) -> BenchmarkResult:
        bs = params.get("batch_size", 1024)
        threads = params.get("threads", 4)
        fa = params.get("flash_attention", True)

        # Simulate speed variation based on params.
        speed = base_speed
        # Batch size effect: diminishing returns.
        speed += min(bs / 1024, 4.0) * 2.0
        # Thread effect: linear up to cpu_count, then drops.
        cpu_count = os.cpu_count() or 8
        if threads <= cpu_count:
            speed += threads * 0.5
        else:
            speed -= (threads - cpu_count) * 1.0
        # Flash attention bonus.
        if fa:
            speed += 3.0
        # Add noise.
        speed += random.uniform(-variance, variance)
        speed = max(1.0, speed)

        start = time.time()
        # Simulate some work time.
        time.sleep(0.01)
        elapsed = (time.time() - start) * 1000

        return BenchmarkResult(
            params=params,
            tokens_per_second_tg=speed,
            tokens_per_second_pp=speed * 1.5,
            total_time_ms=elapsed,
        )

    return _mock_benchmark


# Standard prompt used for all real benchmarks (consistent across runs).
_BENCHMARK_PROMPT = "Explain the theory of general relativity in detail."


def create_ollama_benchmark_fn(
    model_name: str,
    host: str = "http://localhost:11434",
    benchmark_tokens: int = 128,
) -> Callable[[dict], BenchmarkResult]:
    """Create a benchmark function that measures real Ollama inference speed.

    The returned callable accepts a parameter dict (with keys like
    ``num_thread``, ``num_batch``, ``flash_attn``, etc.) and runs a
    real generation request against the Ollama API. It measures prompt
    eval speed and token generation speed from the Ollama response
    metadata.

    Args:
        model_name: Ollama model tag (e.g. "llama3:8b-instruct-q4_K_M").
        host: Ollama API base URL.
        benchmark_tokens: Maximum tokens to generate per benchmark run.

    Returns:
        A ``Callable[[dict], BenchmarkResult]`` suitable for
        ``AutoTunerManager.start_tuning()``.
    """

    def _ollama_benchmark(params: dict) -> BenchmarkResult:
        # Map tuner parameter names to Ollama option names.
        options: dict = {}
        if "threads" in params:
            options["num_thread"] = int(params["threads"])
        if "batch_size" in params:
            options["num_batch"] = int(params["batch_size"])
        if "flash_attention" in params:
            options["flash_attn"] = bool(params["flash_attention"])
        if "ubatch_size" in params:
            # Ollama does not expose ubatch directly but we include it
            # in the options dict for backends that support it.
            options["num_batch"] = min(
                int(params.get("batch_size", 2048)),
                int(params["ubatch_size"]),
            )
        options["num_predict"] = benchmark_tokens

        try:
            import requests

            url = f"{host.rstrip('/')}/api/chat"
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": _BENCHMARK_PROMPT},
                ],
                "options": options,
                "stream": False,
            }

            start = time.time()
            resp = requests.post(url, json=payload, timeout=120)
            elapsed_ms = (time.time() - start) * 1000.0

            if resp.status_code != 200:
                return BenchmarkResult(
                    params=params,
                    error=f"Ollama returned HTTP {resp.status_code}: "
                          f"{resp.text[:200]}",
                )

            data = resp.json()

            # Extract timing from Ollama response metadata.
            # Ollama returns durations in nanoseconds.
            eval_count = data.get("eval_count", 0)
            eval_duration_ns = data.get("eval_duration", 0)
            prompt_eval_count = data.get("prompt_eval_count", 0)
            prompt_eval_duration_ns = data.get("prompt_eval_duration", 0)

            tg_speed = 0.0
            if eval_duration_ns > 0 and eval_count > 0:
                tg_speed = eval_count / (eval_duration_ns / 1e9)

            pp_speed = 0.0
            if prompt_eval_duration_ns > 0 and prompt_eval_count > 0:
                pp_speed = prompt_eval_count / (
                    prompt_eval_duration_ns / 1e9
                )

            return BenchmarkResult(
                params=params,
                tokens_per_second_tg=tg_speed,
                tokens_per_second_pp=pp_speed,
                total_time_ms=elapsed_ms,
            )

        except ImportError:
            return BenchmarkResult(
                params=params,
                error="requests library not installed",
            )
        except Exception as exc:
            return BenchmarkResult(
                params=params,
                error=f"Ollama benchmark failed: {exc}",
            )

    return _ollama_benchmark


def create_llamacpp_benchmark_fn(
    model_name: str,
    backend: Any = None,
    benchmark_tokens: int = 128,
) -> Callable[[dict], BenchmarkResult]:
    """Create a benchmark function using llama-cpp-python for real inference.

    The returned callable accepts a parameter dict and runs a real
    generation against a loaded llama-cpp-python model. Thread count
    and batch size are applied to the model before inference.

    Args:
        model_name: GGUF model filename or identifier.
        backend: A ``LlamaCppBackend`` instance (from inference_backend).
            If ``None``, the function will attempt to get the backend
            from the registry at call time.
        benchmark_tokens: Maximum tokens to generate per benchmark run.

    Returns:
        A ``Callable[[dict], BenchmarkResult]`` suitable for
        ``AutoTunerManager.start_tuning()``.
    """

    def _llamacpp_benchmark(params: dict) -> BenchmarkResult:
        nonlocal backend

        try:
            # Resolve backend lazily if not provided.
            _backend = backend
            if _backend is None:
                try:
                    from opti_oignon.inference_backend import (
                        get_backend_registry,
                    )
                    registry = get_backend_registry()
                    _backend = registry.get_backend("llama_cpp")
                except Exception:
                    return BenchmarkResult(
                        params=params,
                        error="llama.cpp backend not available",
                    )

            if _backend is None:
                return BenchmarkResult(
                    params=params,
                    error="llama.cpp backend not available",
                )

            # Build Ollama-style options from tuner params.
            options: dict = {}
            if "threads" in params:
                options["num_thread"] = int(params["threads"])
            if "batch_size" in params:
                options["n_batch"] = int(params["batch_size"])
            if "flash_attention" in params:
                options["flash_attn"] = bool(params["flash_attention"])
            options["num_predict"] = benchmark_tokens

            messages = [
                {"role": "user", "content": _BENCHMARK_PROMPT},
            ]

            start = time.time()
            response = _backend.generate(
                model=model_name,
                messages=messages,
                options=options,
            )
            elapsed_ms = (time.time() - start) * 1000.0

            # Estimate token speeds from wall-clock time.
            # llama-cpp-python's ChatResponse may carry extra timing.
            content = ""
            if hasattr(response, "content"):
                content = response.content or ""
            elif isinstance(response, dict):
                msg = response.get("message", {})
                content = msg.get("content", "") if isinstance(msg, dict) else ""

            # Rough token estimate (4 chars per token).
            estimated_tokens = max(len(content) / 4.0, 1.0)
            gen_time_s = elapsed_ms / 1000.0

            tg_speed = estimated_tokens / gen_time_s if gen_time_s > 0 else 0.0

            # Check for extra timing metadata from the backend.
            extra = {}
            if hasattr(response, "extra"):
                extra = response.extra or {}
            elif isinstance(response, dict):
                extra = response

            # If the backend provides precise timings, use them.
            if "timings" in extra:
                timings = extra["timings"]
                if "predicted_per_second" in timings:
                    tg_speed = float(timings["predicted_per_second"])

            pp_speed = tg_speed * 1.5  # Rough estimate for prompt processing.
            if "timings" in extra and "prompt_per_second" in extra["timings"]:
                pp_speed = float(extra["timings"]["prompt_per_second"])

            return BenchmarkResult(
                params=params,
                tokens_per_second_tg=tg_speed,
                tokens_per_second_pp=pp_speed,
                total_time_ms=elapsed_ms,
            )

        except Exception as exc:
            return BenchmarkResult(
                params=params,
                error=f"llama.cpp benchmark failed: {exc}",
            )

    return _llamacpp_benchmark
