#!/usr/bin/env python3
"""
INFERENCE AUTO-TUNER -- OPTI-OIGNON
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

Later additions:
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
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

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

# Where a reported rate came from. Three benchmark paths produce rates and
# they are not equally trustworthy: one reads counters the server reported,
# one derives a token count from a character count and a prompt rate from a
# constant multiple, and one invents both. Without a label they are
# indistinguishable once stored, which is how a fabricated figure becomes a
# measurement by the time someone reads it back.
SOURCE_MEASURED = "measured"
SOURCE_ESTIMATED = "estimated"
SOURCE_SIMULATED = "simulated"
SOURCE_UNKNOWN = "unknown"

# Least trustworthy first. Aggregation keeps the weakest source present, so a
# single invented trial cannot be laundered by measured neighbours, and a
# label this module does not recognise is treated as unknown rather than
# trusted.
_SOURCE_RANK = {
    SOURCE_UNKNOWN: 0,
    SOURCE_SIMULATED: 1,
    SOURCE_ESTIMATED: 2,
    SOURCE_MEASURED: 3,
}


def weakest_source(sources) -> str:
    """Return the least trustworthy source among ``sources``.

    An empty collection measured nothing, so it is unknown rather than
    measured: the absence of evidence never aggregates into evidence.
    """
    ranked = [s if s in _SOURCE_RANK else SOURCE_UNKNOWN for s in sources]
    if not ranked:
        return SOURCE_UNKNOWN
    return min(ranked, key=lambda s: _SOURCE_RANK[s])


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

def _positive_int(value):
    """A whole number above zero, or None; a boolean is not a number here."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    return value if isinstance(value, int) and value >= 1 else None


def _positive_seconds(value):
    """A finite number of seconds above zero, as a float, or None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if isinstance(value, (int, float)) and 0 < value < float("inf"):
        return float(value)
    return None


@dataclass
class TunerConfig:
    """Configuration for the auto-tuner."""

    enabled: bool = True
    warmup_runs: int = 3
    benchmark_tokens: int = 128
    benchmark_prompt_tokens: int = 128
    trials_per_param: int = 3
    auto_apply: bool = False
    benchmark_timeout_s: float = 120.0

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
        if _positive_seconds(self.benchmark_timeout_s) is None:
            errors.append("benchmark_timeout_s must be a number of seconds > 0")
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
            "benchmark_timeout_s": self.benchmark_timeout_s,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TunerConfig":
        """Create from dict, ignoring unknown keys.

        The benchmark's token budget and timeout are read alone: a value
        that cannot be read keeps its own default, and the fallback is
        logged by name rather than passed to the engine.
        """
        known = {
            "enabled", "warmup_runs", "benchmark_tokens",
            "benchmark_prompt_tokens", "trials_per_param", "auto_apply",
            "benchmark_timeout_s",
        }
        filtered = {k: v for k, v in data.items() if k in known}
        for key, parse in (("benchmark_tokens", _positive_int), ("benchmark_timeout_s", _positive_seconds)):
            if key not in filtered:
                continue
            value = parse(filtered[key])
            if value is None:
                logger.warning("auto_tuner.yaml %s %r cannot be read; its default is used", key, filtered.pop(key))
            else:
                filtered[key] = value
        return cls(**filtered)


_AUTO = "auto"
_DEFAULT_THREAD_FRACTIONS = (0.5, 0.75)


def _thread_counts(value: Any) -> tuple[list[int], bool]:
    """(counts, auto) for the file's ``threads`` setting.

    "auto", or no setting, takes the counts from the resource governor's
    plan at each run. A list is swept as written: whole numbers of one or
    more, each once, in order. Anything else, or a list with no such number,
    cannot be read and falls back to "auto", by name in the log.
    """
    if value is None or (isinstance(value, str) and value.strip().lower() == _AUTO):
        return [], True
    if isinstance(value, list):
        counts: list[int] = []
        for item in value:
            if isinstance(item, int) and not isinstance(item, bool) and item >= 1 and item not in counts:
                counts.append(item)
        if counts:
            return counts, False
    logger.warning("auto_tuner.yaml parameter_space.threads %r cannot be read; 'auto' is used", value)
    return [], True


def _thread_fractions(value: Any) -> list[float]:
    """The fractions of the plan's count an "auto" sweep measures besides the
    plan itself: numbers above 0 and under 1, each once. None, or a value
    with no such number, takes the default, by name in the log when one was
    written."""
    if value is None:
        return list(_DEFAULT_THREAD_FRACTIONS)
    fractions: list[float] = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, (int, float)) and not isinstance(item, bool) and 0 < item < 1:
            if float(item) not in fractions:
                fractions.append(float(item))
    if fractions:
        return fractions
    logger.warning(
        "auto_tuner.yaml parameter_space.threads_fractions %r cannot be read; %s is used",
        value,
        list(_DEFAULT_THREAD_FRACTIONS),
    )
    return list(_DEFAULT_THREAD_FRACTIONS)


@dataclass
class ParameterSpace:
    """Defines the grid of parameters to search.

    ``threads_auto`` (the file's ``threads: auto``, its default) takes the
    thread counts from the resource governor's plan for the machine at each
    run (``AutoTuner._thread_axis``): the plan's count, each of
    ``threads_fractions`` of it rounded up, and the fastest class's core
    count, never past the plan; ``threads`` is then unused. A space built in
    code names its own lists.
    """

    batch_size: list[int] = field(default_factory=lambda: [512, 1024, 2048, 4096])
    ubatch_size: list[int] = field(default_factory=lambda: [256, 512, 1024])
    threads: list[int] = field(default_factory=lambda: [2, 4, 6, 8])
    flash_attention: list[bool] = field(default_factory=lambda: [True, False])
    threads_auto: bool = False
    threads_fractions: list[float] = field(default_factory=lambda: list(_DEFAULT_THREAD_FRACTIONS))

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
            "threads": _AUTO if self.threads_auto else self.threads,
            "threads_fractions": list(self.threads_fractions),
            "flash_attention": self.flash_attention,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ParameterSpace":
        """Create from dict; ``threads`` is "auto" unless it names a list."""
        threads, auto = _thread_counts(data.get("threads"))
        return cls(
            batch_size=data.get("batch_size", [512, 1024, 2048, 4096]),
            ubatch_size=data.get("ubatch_size", [256, 512, 1024]),
            threads=threads,
            flash_attention=data.get("flash_attention", [True, False]),
            threads_auto=auto,
            threads_fractions=_thread_fractions(data.get("threads_fractions")),
        )


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    params: dict = field(default_factory=dict)
    tokens_per_second_tg: float = 0.0  # Token generation speed
    tokens_per_second_pp: float = 0.0  # Prompt processing speed
    total_time_ms: float = 0.0
    error: str = ""
    # Defaults to unknown so a result built without saying where its rates
    # came from cannot pass for a measurement.
    source: str = SOURCE_UNKNOWN
    # The engine that served the trial, where the model computed (the
    # admission's placement: "cpu", "split:<n>", or None where the card held
    # it whole or the split's layers were not counted), and whether the
    # engine applied the thread count asked. None and False claim nothing.
    engine: str | None = None
    placement: str | None = None
    threads_applied: bool = False

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "params": self.params,
            "tokens_per_second_tg": round(self.tokens_per_second_tg, 2),
            "tokens_per_second_pp": round(self.tokens_per_second_pp, 2),
            "total_time_ms": round(self.total_time_ms, 2),
            "error": self.error,
            "source": self.source,
            "engine": self.engine,
            "placement": self.placement,
            "threads_applied": self.threads_applied,
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
    # The weakest provenance among the results this profile was built from.
    source: str = SOURCE_UNKNOWN
    # The thread count the sweep kept for the governor to plan from
    # ({"engine", "placement", "threads", "threads_batch", "tg", "base_tg"}),
    # or None, with why in ``threads_note``.
    threads_optimum: dict | None = None
    threads_note: str = ""

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
            "source": self.source,
            "threads_optimum": dict(self.threads_optimum) if self.threads_optimum else None,
            "threads_note": self.threads_note,
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
            # A record written before provenance existed carries no claim, so
            # it rehydrates as unknown rather than being promoted.
            source=data.get("source", SOURCE_UNKNOWN),
            threads_optimum=(
                dict(data["threads_optimum"]) if isinstance(data.get("threads_optimum"), dict) else None
            ),
            threads_note=data.get("threads_note", "") if isinstance(data.get("threads_note"), str) else "",
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
    result: TunerProfile | None = None
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
# Tuner Recommendations
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

    ``thread_plan`` answers (base, candidates, source) for the model being
    tuned, as the resource governor plans its CPU threads
    (``ResourceGovernor.thread_candidates``): the baseline's thread count
    and, with "auto", the counts swept. Without it a list is swept from its
    middle, as before, and "auto" measures no count.
    """

    def __init__(
        self,
        config: TunerConfig,
        param_space: ParameterSpace,
        benchmark_fn: Callable[[dict], BenchmarkResult] | None = None,
        progress_fn: Callable[[TunerJob], None] | None = None,
        thread_plan: Callable[[], tuple[int | None, list[int], str] | None] | None = None,
    ):
        self._config = config
        self._param_space = param_space
        self._benchmark_fn = benchmark_fn
        self._progress_fn = progress_fn
        self._thread_plan = thread_plan
        # Resolved once per run: (counts swept, baseline count, source).
        self._threads: tuple[list[int], int | None, str] | None = None
        # What each point's successful trials said: who served them and
        # where the model computed, and whether every one applied its count.
        self._labels: dict[str, set[tuple[str | None, str | None]]] = {}
        self._applied: dict[str, bool] = {}
        self._last_params: dict | None = None
        # The engines the run's trials reached, warm-ups included, and the
        # count the governor held pinned for the model as the run began.
        self._engines: set[str] = set()
        self._pin_at_start: tuple[int, int] | None = None
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
        self._threads = None
        self._labels = {}
        self._applied = {}
        self._last_params = None
        self._engines = set()
        self._pin_at_start = self._read_pin(model_name)
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
                self._note_engine(self._benchmark_fn(default_params))
                job.completed_steps += 1
                job.progress = job.completed_steps / max(job.total_steps, 1)
                self._report_progress(job)

            # Baseline measurement
            job.current_step = "Measuring baseline..."
            self._report_progress(job)
            baseline = self._run_averaged(default_params)
            if baseline.error:
                # Nothing to compare a configuration with: the run ends here,
                # by the reason, rather than keep a profile of failures.
                raise RuntimeError(f"the baseline could not be measured: {baseline.error}")

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

            # Phase 3: Refinement -- re-confirm the best with extra runs
            job.current_step = "Confirming best configuration..."
            self._report_progress(job)
            confirmed = self._run_averaged(
                best.params, extra_trials=self._config.trials_per_param
            )
            if not confirmed.error:
                best = confirmed

            # Phase 4: the thread count the governor may plan from, and the
            # model left at the count its next decisions carry.
            job.current_step = "Confirming the best thread count..."
            self._report_progress(job)
            threads_optimum, threads_note = self._keep_threads(all_results, baseline, confirmed)
            if threads_optimum is None:
                self._settle()
                self._pin_plan(model_name)
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
                # Every result that fed this profile, including the one it
                # kept: the profile can claim no more than its weakest input.
                source=weakest_source(
                    [r.source for r in all_results] + [best.source]
                ),
                threads_optimum=threads_optimum,
                threads_note=threads_note,
            )

            job.status = "completed"
            job.result = profile
            job.finished_at = time.time()
            job.current_step = f"Done! Best: {best.tokens_per_second_tg:.1f} tok/s ({speedup:.2f}x)"
            self._report_progress(job)

            return profile

        except ValueError as exc:
            # Cancelled
            self._pin_plan(model_name)
            job.status = "cancelled"
            job.error = str(exc)
            job.finished_at = time.time()
            self._report_progress(job)
            raise
        except Exception as exc:
            self._pin_plan(model_name)
            job.status = "failed"
            job.error = str(exc)
            job.finished_at = time.time()
            self._report_progress(job)
            raise

    def _pin_plan(self, model_name: str) -> None:
        """A run whose last trial to reach the engine was not at the count a
        fresh load would get -- cancelled, refused or failed, or settled by
        a trial the governor held -- leaves the model loaded at that trial's
        count, which the governor pinned. When the plan gave the baseline
        its count, the governor is told to pin what a fresh load of the
        model would get instead (its plan for the engine the trials ran on
        and the placement the resident holds, so a kept count wins), and the
        next call reloads the model once at it, rather than every call
        keeping a count nobody plans. Decided on the pin itself, not on the
        last trial tried: a trial the governor held never reached the
        engine. No pin, a run whose trials reached no engine (whatever moved
        the pin, it did not), the pin the run found, one at the fresh count
        already, or no governor: nothing to do."""
        _axis, base, source = self._thread_axis()
        if source != "plan" or base is None:
            return
        governor = _running_governor()
        pinned = getattr(governor, "pinned_threads", None)
        pin = getattr(governor, "pin_threads", None)
        if not callable(pinned) or not callable(pin):
            return
        if not self._engines:
            # No trial reached an engine: whatever moved the pin, this run
            # did not, and with no engine named a plan would ignore a kept
            # count.
            return
        try:
            current = pinned(model_name)
            if current is None or current == self._pin_at_start:
                return
            target = self._fresh_count(governor, model_name, base)
            if current != target:
                pin(model_name, *target)
        except Exception as exc:  # noqa: BLE001 - the run has already ended
            logger.debug("pinning %s back at its planned thread count failed: %s", model_name, exc)

    def _fresh_count(self, governor: Any, model_name: str, base: int) -> tuple[int, int]:
        """The (threads, threads_batch) a fresh load of ``model_name`` would
        get: the governor's plan for the one engine the trials ran on and
        the placement the resident holds (a count kept for them wins), else
        the plan's own count."""
        plan = getattr(governor, "plan_threads", None)
        if not callable(plan):
            return base, base
        engines = self._engines
        placement_of = getattr(governor, "pinned_placement", None)
        placement = placement_of(model_name) if callable(placement_of) else None
        threads, threads_batch, _source = plan(model_name, next(iter(engines)) if len(engines) == 1 else None, placement)
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            return base, base
        if isinstance(threads_batch, bool) or not isinstance(threads_batch, int) or threads_batch < 1:
            threads_batch = threads
        return threads, threads_batch

    def _read_pin(self, model_name: str) -> tuple[int, int] | None:
        """The count the governor holds pinned for ``model_name``, or None
        (no pin, or no governor to ask)."""
        pinned = getattr(_running_governor(), "pinned_threads", None)
        if not callable(pinned):
            return None
        try:
            return pinned(model_name)
        except Exception as exc:  # noqa: BLE001 - an unread pin is no pin
            logger.debug("reading the pin of %s failed: %s", model_name, exc)
            return None

    def _note_engine(self, result: Any) -> None:
        """Note the engine a trial reached, warm-ups included: a trial that
        failed, or that the governor held, reached none."""
        engine = getattr(result, "engine", None)
        if engine and not getattr(result, "error", ""):
            self._engines.add(str(engine))

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

        for t in self._thread_axis()[0]:
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
        """Return default (middle-of-range) parameters, the thread count
        being the baseline's (``_thread_axis``): absent when there is none,
        and the engine picks its own."""
        def mid(lst: list) -> Any:
            return lst[len(lst) // 2] if lst else None

        params = {
            "batch_size": mid(self._param_space.batch_size) or 1024,
            "ubatch_size": mid(self._param_space.ubatch_size) or 512,
        }
        base = self._thread_axis()[1]
        if base is not None:
            params["threads"] = base
        params["flash_attention"] = True
        return params

    def _thread_axis(self) -> tuple[list[int], int | None, str]:
        """(counts swept, baseline count, source), resolved once per run.

        "auto" sweeps the candidates of the plan (``thread_plan``) from the
        plan's count, so the gain is told against what the plan would do;
        without a plan it measures no count, and the engine picks its own as
        it would for the calls it serves. A list is swept as written, from
        the plan's count when there is one, else from its middle as before.
        """
        if self._threads is not None:
            return self._threads
        base: int | None = None
        candidates: list[int] = []
        source = "none"
        if self._thread_plan is not None:
            try:
                planned = self._thread_plan()
            except Exception as exc:  # noqa: BLE001 - no plan is an answer
                logger.debug("thread plan unavailable to the tuner: %s", exc)
                planned = None
            if planned is not None:
                base, candidates, source = planned
        if self._param_space.threads_auto:
            axis = list(candidates) if base is not None else []
        else:
            axis = list(self._param_space.threads)
            if base is None:
                base = (axis[len(axis) // 2] if axis else None) or 4
        self._threads = (axis, base, source)
        return self._threads

    def _keep_threads(
        self, results: list[BenchmarkResult], baseline: BenchmarkResult, confirmed: BenchmarkResult
    ) -> tuple[dict | None, str]:
        """The thread count this sweep leaves for the governor to plan from,
        or None and why.

        The thread trials are the points that differ from the baseline in
        their thread count alone, the baseline among them. The fastest, the
        lowest count on a tie since it leaves the most cores, is confirmed
        with extra runs (``confirmed`` when it is the best configuration).
        It is kept only when the plan gave the baseline its count, and
        only if it holds: the confirmation ran; every thread trial was
        measured; by an engine that applied the count asked; on one engine
        and one placement, known exactly; at no more than the plan's count;
        and confirmed no slower than the baseline.
        """
        _axis, base, source = self._thread_axis()
        if source == "override":
            return None, "the resource governor's file names this model's thread count: none is kept"
        if source != "plan" or base is None:
            return None, "no thread plan for this machine: no count is measured against one"
        others = {k: v for k, v in baseline.params.items() if k != "threads"}
        trials = [r for r in results if {k: v for k, v in r.params.items() if k != "threads"} == others]
        ran = [r for r in trials if not r.error]
        if not ran:
            return None, "no thread trial ran"
        top = min(ran, key=lambda r: (-r.tokens_per_second_tg, r.params.get("threads", 0)))
        if _param_key(top.params) == _param_key(confirmed.params):
            check = confirmed
        else:
            check = self._run_averaged(top.params, extra_trials=self._config.trials_per_param)
        keys = {_param_key(r.params) for r in trials} | {_param_key(check.params)}
        labels: set[tuple[str | None, str | None]] = set()
        for key in keys:
            labels |= self._labels.get(key, set())
        threads = top.params.get("threads")
        if check.error:
            return None, f"the confirmation of the best count failed: {check.error}"
        if any(r.source != SOURCE_MEASURED for r in trials + [check]):
            return None, "a thread trial was not measured"
        if not all(self._applied.get(key, False) for key in keys):
            return None, "the engine does not apply a thread count per call"
        if len(labels) != 1:
            return None, "the engine or the placement changed during the sweep"
        engine, placement = next(iter(labels))
        if not engine or not placement:
            return None, "placement unknown: the card held the model whole, or its split was not counted"
        if not isinstance(threads, int) or isinstance(threads, bool) or not 1 <= threads <= base:
            return None, "the best count is past the plan"
        if check.tokens_per_second_tg < baseline.tokens_per_second_tg:
            return None, "the best count's confirmation was slower than the baseline"
        return {
            "engine": engine,
            "placement": placement,
            "threads": threads,
            "threads_batch": threads,
            "tg": check.tokens_per_second_tg,
            "base_tg": baseline.tokens_per_second_tg,
        }, ""

    def _settle(self) -> None:
        """One more trial at the plan's count, every other knob at its
        default, when the plan gave the baseline its count and the last
        trial ran at another: the model is left at the count its next
        decisions carry, not at one nobody plans. Its result measures
        nothing and joins no profile; its failure fails nothing."""
        _axis, base, source = self._thread_axis()
        if source != "plan" or base is None or self._last_params is None:
            return
        if self._last_params.get("threads") == base:
            return
        self._check_cancelled()
        defaults = self._default_params()
        self._last_params = defaults
        try:
            self._benchmark_fn(defaults)
        except Exception as exc:  # noqa: BLE001 - the sweep is already done
            logger.debug("settling trial at the plan's thread count failed: %s", exc)

    def _run_averaged(
        self, params: dict, extra_trials: int = 0
    ) -> BenchmarkResult:
        """Run the benchmark multiple times and return averaged result."""
        trials = self._config.trials_per_param + extra_trials
        tg_speeds: list[float] = []
        pp_speeds: list[float] = []
        total_times: list[float] = []
        sources: list[str] = []
        labels: set[tuple[str | None, str | None]] = set()
        applied: list[bool] = []
        last_error = ""

        for _ in range(trials):
            self._check_cancelled()
            self._last_params = params
            result = self._benchmark_fn(params)
            if result.error:
                last_error = result.error
                continue
            tg_speeds.append(result.tokens_per_second_tg)
            pp_speeds.append(result.tokens_per_second_pp)
            total_times.append(result.total_time_ms)
            sources.append(result.source)
            labels.add((result.engine, result.placement))
            self._note_engine(result)
            applied.append(bool(result.threads_applied))

        key = _param_key(params)
        self._labels.setdefault(key, set()).update(labels)
        if applied:
            self._applied[key] = self._applied.get(key, True) and all(applied)

        if not tg_speeds:
            return BenchmarkResult(
                params=params,
                error=last_error or "All trials failed",
            )

        engine, placement = next(iter(labels)) if len(labels) == 1 else (None, None)
        return BenchmarkResult(
            params=params,
            tokens_per_second_tg=sum(tg_speeds) / len(tg_speeds),
            tokens_per_second_pp=sum(pp_speeds) / len(pp_speeds),
            total_time_ms=sum(total_times) / len(total_times),
            # An average is only as good as the weakest trial inside it.
            source=weakest_source(sources),
            engine=engine,
            placement=placement,
            threads_applied=all(applied),
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

    def __init__(self, config_path: str | None = None):
        self._config = TunerConfig()
        self._param_space = ParameterSpace()
        self._profiles: dict[str, TunerProfile] = {}
        self._active_jobs: dict[str, TunerJob] = {}
        self._active_tuners: dict[str, AutoTuner] = {}
        self._lock = threading.RLock()
        self._load_config(config_path)

    def _load_config(self, config_path: str | None = None) -> None:
        """Load configuration from YAML."""
        p = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        if not p.is_file():
            logger.debug("No auto_tuner.yaml found at %s", p)
            return

        try:
            with open(p, encoding="utf-8") as f:
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
            with open(_RESULTS_PATH, encoding="utf-8") as f:
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

    def get_result(self, model_name: str) -> TunerProfile | None:
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
        progress_fn: Callable[[TunerJob], None] | None = None,
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
                thread_plan=self._thread_plan_for(model_name),
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

    def get_job(self, model_name: str) -> TunerJob | None:
        """Get the current/last job for a model."""
        with self._lock:
            job = self._active_jobs.get(model_name)
            if job is None:
                return None
            # Return a snapshot.
            return job

    def apply_result(self, model_name: str) -> dict | None:
        """Get the best params for a model (for manual application).

        Returns the best_params dict, or None if no profile exists.
        """
        with self._lock:
            profile = self._profiles.get(model_name)
            if profile is None:
                return None
            return dict(profile.best_params)

    def _thread_plan_for(self, model_name: str) -> Callable[[], tuple[int | None, list[int], str] | None]:
        """What a run for ``model_name`` asks the resource governor for its
        thread counts (``ResourceGovernor.thread_candidates``), with the
        fractions of the file; None without a governor to ask."""
        fractions = list(self._param_space.threads_fractions)

        def _plan() -> tuple[int | None, list[int], str] | None:
            governor = _tuner_governor()
            candidates = getattr(governor, "thread_candidates", None) if governor is not None else None
            if not callable(candidates):
                return None
            return candidates(model_name, fractions)

        return _plan

    def _record_threads_optimum(self, model_name: str, profile: TunerProfile) -> None:
        """Hand the count the run kept to the resource governor, which plans
        from it (``ResourceGovernor.record_thread_optimum``); the profile
        says so when the governor did not keep it."""
        kept = profile.threads_optimum
        if not kept:
            return
        governor = _tuner_governor()
        record = getattr(governor, "record_thread_optimum", None) if governor is not None else None
        written = False
        if callable(record):
            try:
                written = bool(
                    record(
                        model_name,
                        kept["engine"],
                        kept["placement"],
                        threads=kept["threads"],
                        threads_batch=kept["threads_batch"],
                        tg=kept["tg"],
                        base_tg=kept["base_tg"],
                    )
                )
            except Exception as exc:  # noqa: BLE001 - the profile says so
                logger.warning("the resource governor could not keep the thread count for %s: %s", model_name, exc)
        if not written:
            profile.threads_note = "the resource governor did not keep the count"

    def _run_tuning_thread(
        self, model_name: str, tuner: AutoTuner, job: TunerJob
    ) -> None:
        """Thread target for running tuning."""
        try:
            profile = tuner.run(model_name, job)
            self._record_threads_optimum(model_name, profile)
            with self._lock:
                self._profiles[model_name] = profile
                self._save_results()
        except ValueError:
            # Cancelled -- job already updated by tuner.
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

_manager: AutoTunerManager | None = None
_init_lock = threading.Lock()


def get_auto_tuner_manager(
    config_path: str | None = None,
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
            # No inference happened. Both rates come from a formula and a
            # random term, and the result says so.
            source=SOURCE_SIMULATED,
        )

    return _mock_benchmark


# Standard prompt used for all real benchmarks (consistent across runs).
_BENCHMARK_PROMPT = "Explain the theory of general relativity in detail."


def _registry_backend(name: str) -> Any:
    """The registry's backend under ``name``, or None when there is no registry or no such backend."""
    try:
        from opti_oignon.inference_backend import get_backend_registry
    except Exception as exc:  # noqa: BLE001 - absence is an answer
        logger.debug("inference registry unavailable to the tuner: %s", exc)
        return None
    try:
        return get_backend_registry().get(name)
    except Exception as exc:  # noqa: BLE001 - a broken registry is absence
        logger.debug("inference registry could not answer for %s: %s", name, exc)
        return None


# The caller a tuner's trial is admitted as: nobody waits on a sweep, so the
# governor's caller table makes it the background -- it waits for a quiet
# machine and never evicts or splits a model to make room.
TUNER_CALLER = "tuner"


class TunerRefused(RuntimeError):
    """The governor refused a trial for good: no wait can lift the refusal,
    and the run ends by it."""


def _tuner_governor() -> Any:
    """The resource governor, or None when there is none to ask: a trial
    then runs unadmitted, as every funnel fails open."""
    try:
        from opti_oignon.resource_governor import get_resource_governor

        return get_resource_governor()
    except Exception as exc:  # noqa: BLE001 - absence is an answer here
        logger.debug("resource governor unavailable to the tuner: %s", exc)
        return None


def _running_governor() -> Any:
    """The resource governor if one runs in this process, else None: the
    pins a run reads and restores live there, and reading one never starts
    a governor for nothing."""
    try:
        from opti_oignon.resource_governor import running_governor

        return running_governor()
    except Exception as exc:  # noqa: BLE001 - no governor running: no pin
        logger.debug("resource governor unavailable to the tuner's pins: %s", exc)
        return None


def _admit_trial(model_name: str, engine: str) -> tuple[Any, str]:
    """(decision, error) for one trial of ``model_name`` on ``engine``.

    The trial asks its own ticket as the tuner, naming its engine, so the
    decision says who serves it and where the model computes. A refusal a
    wait may lift is the trial's error, by its reason, and nothing runs; one
    no wait can lift raises TunerRefused. No governor: (None, "").
    """
    governor = _tuner_governor()
    if governor is None:
        return None, ""
    try:
        decision = governor.admit_or_wait(model_name, caller=TUNER_CALLER, engine=engine)
    except Exception as exc:  # noqa: BLE001 - the trial fails by it
        return None, f"resource governor admission failed: {exc}"
    if decision is None or getattr(decision, "admitted", True):
        return decision, ""
    reason = getattr(decision, "reason", "") or "refused"
    try:
        from opti_oignon.resource_governor import refusal_is_final

        final = refusal_is_final(decision)
    except Exception:  # noqa: BLE001 - unknown is not final
        final = False
    if final:
        raise TunerRefused(f"refused by the resource governor: {reason}")
    return None, f"held by the resource governor: {reason}"


@contextmanager
def _holding(decision: Any):
    """Hold the trial's own ticket around the engine call, so the engine
    gate accounts the load the governor admitted instead of admitting it
    again as a user call, which may evict."""
    if decision is None:
        yield
        return
    from opti_oignon.resource_governor import ticket_scope

    with ticket_scope(decision):
        yield


def _at_admitted_ctx(options: dict, decision: Any) -> dict:
    """``options`` with the context the admission priced, so the engine
    loads or keeps the model at that context; unchanged without one."""
    ctx = getattr(decision, "num_ctx", None)
    if isinstance(ctx, int) and not isinstance(ctx, bool) and ctx > 0:
        return {**options, "num_ctx": ctx}
    return options


def _trial_labels(decision: Any, backend: Any, default_engine: str, params: dict) -> dict:
    """What a trial's result says of itself: the engine that served it (the
    admission's, else the backend's own name), where the model computed (the
    admission's placement), and whether the engine applied the thread count
    asked (``threads_per_call``, as the engine declares it)."""
    engine = getattr(decision, "engine", None) or str(getattr(backend, "name", "") or default_engine)
    return {
        "engine": engine,
        "placement": getattr(decision, "placement", None),
        "threads_applied": "threads" in params and bool(getattr(backend, "threads_per_call", False)),
    }


def create_ollama_benchmark_fn(
    model_name: str,
    host: str = "http://localhost:11434",
    benchmark_tokens: int = 128,
    backend: Any = None,
    timeout_s: float = 120.0,
) -> Callable[[dict], BenchmarkResult]:
    """Create a benchmark function that measures real Ollama inference speed.

    The returned callable maps the tuner's parameters to engine options
    and asks the registry's Ollama backend for one generation, so every
    sweep point is admitted by the governor like any other request. The
    rates come from the counters the backend reports on ``extra``; a
    reply without them is not a measurement.

    Args:
        model_name: Ollama model tag (e.g. "llama3:8b-instruct-q4_K_M").
        host: Accepted for the former signature and unused: where the
            model is served is the registry's to know.
        benchmark_tokens: Maximum tokens to generate per benchmark run.
        backend: The registry's Ollama backend. If ``None``, the
            registry is asked at call time.
        timeout_s: Bound on one benchmark request.

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
            # Ollama exposes no micro-batch option of its own, so it travels
            # under its own key for backends that read one. It must not land
            # on num_batch: the sweep always carries both keys, so writing
            # min(batch, ubatch) there made every batch-size point send a
            # byte-identical request, and the batch recommendation that
            # compared them was reading run-to-run noise.
            options["num_ubatch"] = int(params["ubatch_size"])
        options["num_predict"] = benchmark_tokens
        options["timeout"] = timeout_s

        _backend = backend if backend is not None else _registry_backend("ollama")
        if _backend is None:
            return BenchmarkResult(
                params=params,
                error="no ollama backend in the inference registry: nothing was run",
            )

        engine = str(getattr(_backend, "name", "") or "ollama")
        decision, refused = _admit_trial(model_name, engine)
        if refused:
            return BenchmarkResult(params=params, error=refused, engine=engine)

        try:
            start = time.time()
            with _holding(decision):
                response = _backend.generate(
                    model=model_name,
                    messages=[{"role": "user", "content": _BENCHMARK_PROMPT}],
                    options=_at_admitted_ctx(options, decision),
                )
            elapsed_ms = (time.time() - start) * 1000.0

            # The counters the engine reported, in nanoseconds; absent
            # rather than zero when nothing was reported.
            extra = getattr(response, "extra", None) or {}
            eval_count = extra.get("eval_count") or 0
            eval_duration_ns = extra.get("eval_duration") or 0
            prompt_eval_count = extra.get("prompt_eval_count") or 0
            prompt_eval_duration_ns = extra.get("prompt_eval_duration") or 0

            tg_speed = 0.0
            tg_counted = eval_duration_ns > 0 and eval_count > 0
            if tg_counted:
                tg_speed = eval_count / (eval_duration_ns / 1e9)

            pp_speed = 0.0
            pp_counted = prompt_eval_duration_ns > 0 and prompt_eval_count > 0
            if pp_counted:
                pp_speed = prompt_eval_count / (
                    prompt_eval_duration_ns / 1e9
                )

            # Measured only when both rates came from counters the server
            # itself reported. A missing counter leaves a zero behind, and a
            # zero that nobody measured is not a measurement of zero.
            return BenchmarkResult(
                params=params,
                tokens_per_second_tg=tg_speed,
                tokens_per_second_pp=pp_speed,
                total_time_ms=elapsed_ms,
                source=(
                    SOURCE_MEASURED if tg_counted and pp_counted
                    else SOURCE_UNKNOWN
                ),
                **_trial_labels(decision, _backend, "ollama", params),
            )

        except Exception as exc:
            # A governor refusal arrives here too, in its own words.
            return BenchmarkResult(
                params=params,
                error=f"Ollama benchmark failed: {exc}",
                engine=engine,
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

        # Resolve backend lazily if not provided.
        _backend = backend
        if _backend is None:
            _backend = _registry_backend("llama_cpp")

        if _backend is None:
            return BenchmarkResult(
                params=params,
                error="llama.cpp backend not available",
            )

        engine = str(getattr(_backend, "name", "") or "llama_cpp")
        decision, refused = _admit_trial(model_name, engine)
        if refused:
            return BenchmarkResult(params=params, error=refused, engine=engine)

        try:
            # Build Ollama-style options from tuner params.
            options: dict = {}
            if "threads" in params:
                options["num_thread"] = int(params["threads"])
            if "batch_size" in params:
                options["n_batch"] = int(params["batch_size"])
            if "ubatch_size" in params:
                # The mirror of the Ollama defect: without this the whole
                # micro-batch axis left as one identical request, so the
                # values the sweep compared were never actually different.
                options["n_ubatch"] = int(params["ubatch_size"])
            if "flash_attention" in params:
                options["flash_attn"] = bool(params["flash_attention"])
            options["num_predict"] = benchmark_tokens

            messages = [
                {"role": "user", "content": _BENCHMARK_PROMPT},
            ]

            start = time.time()
            with _holding(decision):
                response = _backend.generate(
                    model=model_name,
                    messages=messages,
                    options=_at_admitted_ctx(options, decision),
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
            # Both rates start as estimates and are only promoted below, when
            # the backend turns out to have reported real counters.
            tg_counted = False
            pp_counted = False

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
                    tg_counted = True

            pp_speed = tg_speed * 1.5  # Rough estimate for prompt processing.
            if "timings" in extra and "prompt_per_second" in extra["timings"]:
                pp_speed = float(extra["timings"]["prompt_per_second"])
                pp_counted = True

            # A token count inferred from characters and a prompt rate taken
            # as a constant multiple of another rate are estimates. They are
            # measurements only when the backend reported both counters.
            return BenchmarkResult(
                params=params,
                tokens_per_second_tg=tg_speed,
                tokens_per_second_pp=pp_speed,
                total_time_ms=elapsed_ms,
                source=(
                    SOURCE_MEASURED if tg_counted and pp_counted
                    else SOURCE_ESTIMATED
                ),
                **_trial_labels(decision, _backend, "llama_cpp", params),
            )

        except Exception as exc:
            return BenchmarkResult(
                params=params,
                error=f"llama.cpp benchmark failed: {exc}",
                engine=engine,
            )

    return _llamacpp_benchmark
