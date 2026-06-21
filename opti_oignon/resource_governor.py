#!/usr/bin/env python3
"""Resource Governor -- Blocs 0-3: measurement, admission, backpressure, limits.

RESOURCE_GOVERNOR_SPEC.md Section 3 implemented verbatim at S223: a cached
ResourceSnapshot assembled from ranked, individually-optional sources with
honest provenance; the measure-and-adapt store (Section 3.2); the YAML
config loader (Section 10 keys). Bloc 1 (S224) adds the admission gate on
top: admit() and the AdmissionDecision ticket (Section 4.4), the 4.2 fit
math on the cached snapshot (get_snapshot_fast), the per-caller ctx ladder
and floors (4.3, benchmark/AGT never downsized), the R-04 emergency-stop
honour (4.5: the flag checked FIRST through the existing seams only, the
refusal built from refusal_payload()), the thread-local ticket pass-through
(ticket_scope, the S224 gate arbitration) and the mechanical backend gate
(backend_admission_gate) consumed by the four generate/stream heads.
Bloc 2 (S225) adds runtime backpressure (Section 5, the escalation order
verbatim): the pressure signal (in_use over effective capacity against the
config soft/hard thresholds, plus a bounded refusal-rate window), the
per-decision keep_alive override under soft pressure with the
sustained-write-then-restore discipline on the warmup's existing settable
property (one-way: warmup never knows the governor; its keepalive thread
is never stopped here), the targeted per-model eviction honouring
conditional-on-eviction admissions (audit-chained off the hot path,
degrading to Ollama's own LRU on any failure, the Section 12 posture),
and the bounded opt-in queue (per-caller enrollment, depth and wait
bounds, re-admission on wake, the estop never bypassed). Bloc 3 (S226)
adds limit management (Section 6, R-03): the pure spawn-path env
construction (build_ollama_spawn_env, the contract for any future
Ollama spawner -- none exists in-app today), the external-Ollama
advisory (compute_ollama_limits_advisory and the governor method,
consumed by the startup security checklist through the S145
advisory-only precedent: never blocking startup in any mode), and the
optional, off-by-default process-wide rlimits applier
(apply_llamacpp_rlimits, consumed by the llama.cpp load seam BEFORE the
first in-process load). The API/frontend
surfaces remain Bloc 4 territory.

Ranked sources (Section 3, decision D2):

- S1: the Ollama /api/ps view through model_warmup.get_loaded_models().
  The CC-01 dual-form handling (dict vs typed-object client responses)
  lives THERE and is consumed, not duplicated. Truth for what is loaded
  and what it actually costs (size_vram). Provenance note: the consumed
  seam answers an empty list for "no models", "client error" and "server
  down" alike (its documented fail-soft contract), so the "S1" provenance
  label means the read path was importable and answered; it is not a
  server liveness probe. The package-absent case is detected through the
  home module's OLLAMA_AVAILABLE flag.
- S2: the backend registry's own state -- an in-process backend's loaded
  set (the LlamaCppBackend ``_loaded_models`` idiom, invisible to Ollama)
  plus model_info() metadata as the estimation basis for backend-resident
  models. Read-only, defensive (getattr), no signature touched.
- S3: static estimation for not-yet-loaded models --
  speculative_decoding._VRAM_PER_BILLION_PARAMS through
  VRAMBudgetCalculator.estimate_model_vram(), reused BY IMPORT (the table
  is not moved and not duplicated; the s110 pins on its home module keep
  holding), plus the KV-cache increment implemented HERE as a function of
  the requested num_ctx (the config-tunable ``kv_coefficient``).
- S4: capacity and host memory -- total VRAM capacity is a CONFIGURED
  value (``total_vram_gb``, null by default meaning unknown), refined
  downward by the learned ceiling (Section 3.2); host RAM comes from
  /proc/meminfo MemAvailable with the psutil fallback. The RAM read is a
  deliberate LOCAL EQUIVALENT of the S171 smart_router idiom (read-gate
  decision DI-1): the S171 pre-flight is NOT moved out of smart_router
  and its private helper is not imported across modules; the few lines
  are replicated by decision so this module stays standalone-loadable
  and zero existing files are edited this bloc.

Design decisions (S223 read gate, arbitrated):

- DI-2: the default DB path follows the benchmark ResultsStore precedent
  (``opti_oignon/data/resource_governor.db``); ``db_path`` injectable.
- DI-4: ``kv_coefficient`` is GiB of KV cache per 1024 tokens of the
  requested num_ctx (layers folded into one conservative, deliberately
  high-side coefficient; refined later by measure-and-adapt).
- DI-5: the TTL cache exposes refresh() (synchronous build),
  get_snapshot() (stale -> synchronous refresh) and get_snapshot_fast()
  (returns the cached snapshot even when stale and triggers a single
  background refresh -- the primitive the Bloc 1 admission fast path will
  consume; the current decision uses the cached values conservatively).
- DI-8: ceiling learning is fast-down / slow-up. Fast-down immediately on
  a reported load failure to max(floor, observed_in_use - safety_margin);
  slow-up by _CEILING_RELAX_STEP_GB toward the configured capacity after
  _CEILING_RELAX_AFTER_SUCCESSES consecutive successes above the learned
  ceiling. The two relax knobs are module constants, not config keys, so
  the Section 10 contract is not extended without a spec touch.
- DI-9: invalidate_on_load() records a pending attribution; the next
  refresh that sees the model in the S1 view with a positive size_vram
  writes the learned per-model cost (keyed name+digest when the digest is
  present). The other hooks (evict / estop-drain / resume) only
  invalidate. None of the four is wired to a caller this bloc: Bloc 1
  wires the callers.

Conservative defaults and fail-open (Section 3.1): capacity unknown
(configured null AND no learned ceiling) -> the VRAM half reports
vram_status="disabled_capacity_unknown" with a logged warning and the RAM
half still applies; an unknown model is never treated as too large; a
source erroring is the same as a source absent (log at debug, degrade to
the next source, never raise into the request path). No audit-chain append
happens anywhere in this module (the chain is reserved for evictions,
config changes and ceiling-learning surfacing, all Bloc 1+ and off the hot
path).

Kerckhoffs: nothing here is secret; the measurement chain, the learning
rules and the config surface are fully described. The store holds derived,
regenerable state only (ATREST disposition: single-user, pending-scoping,
backup excluded).
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import yaml

logger = logging.getLogger(__name__)

# Module conventions.
checkpoint_before_apply = True
FEATURE_AVAILABLE = True

# ---------------------------------------------------------------------------
# Conditional imports (a source erroring == a source absent, Section 3.1)
# ---------------------------------------------------------------------------

try:
    from opti_oignon.db_utils import safe_connect as _safe_connect

    DB_UTILS_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only on broken installs
    import sqlite3 as _sq3

    def _safe_connect(p: Any, **kw: Any) -> Any:
        return _sq3.connect(str(p), **kw)

    DB_UTILS_AVAILABLE = False

try:
    from opti_oignon.model_warmup import model_warmup as _default_warmup

    MODEL_WARMUP_AVAILABLE = True
except Exception:
    _default_warmup = None
    MODEL_WARMUP_AVAILABLE = False

try:
    from opti_oignon.inference_backend import (
        get_backend_registry as _get_backend_registry,
    )

    INFERENCE_BACKEND_AVAILABLE = True
except Exception:
    _get_backend_registry = None
    INFERENCE_BACKEND_AVAILABLE = False

try:
    # S3 reuse BY IMPORT: estimate_model_vram() reads the
    # _VRAM_PER_BILLION_PARAMS table in its home module. The table is not
    # moved and not duplicated here (the s110 pins keep holding).
    from opti_oignon.speculative_decoding import (
        VRAMBudgetCalculator as _VRAMBudgetCalculator,
    )

    SPECULATIVE_AVAILABLE = True
except Exception:
    _VRAMBudgetCalculator = None
    SPECULATIVE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths and module constants
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path(__file__).parent / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "resource_governor.yaml"
_DATA_DIR = Path(__file__).parent / "data"
_DEFAULT_DB_PATH = _DATA_DIR / "resource_governor.db"

# Ceiling-learning relax knobs (DI-8): module constants, not config keys,
# so the Section 10 config contract is not extended without a spec touch.
_CEILING_RELAX_AFTER_SUCCESSES = 5
_CEILING_RELAX_STEP_GB = 1.0

# Bloc 2 (S225) constants. The queue waits in bounded real-time slices so
# a fake injected clock can drive deadline math in container tests while
# notify-based wakes stay immediate. The refusal-rate rule raises the
# pressure level to AT LEAST soft when at least
# _REFUSAL_RATE_MIN_DECISIONS recorded decisions fall inside the config
# window and the refused fraction reaches _REFUSAL_RATE_SOFT; the rate
# alone never reaches hard (DI-S225-2).
_QUEUE_WAIT_SLICE_S = 0.5
_REFUSAL_RATE_SOFT = 0.5
_REFUSAL_RATE_MIN_DECISIONS = 3

_BYTES_PER_GIB = 1024.0 ** 3

# Sentinel distinguishing "not passed" from an explicit None injection.
_UNSET: Any = object()

_SIZE_STR_RE = re.compile(r"^([\d.]+)\s*(GB|MB|B)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _read_available_ram_mb(meminfo_path: str | Path = "/proc/meminfo") -> float:
    """Return available system RAM in MB, or 0.0 when undeterminable.

    Deliberate local equivalent of the S171 smart_router idiom (read-gate
    decision DI-1): /proc/meminfo MemAvailable first, the psutil fallback
    second, 0.0 fail-open last (a 0.0 result means the RAM half of the
    snapshot is unknown and must never exclude anything). The smart_router
    pre-flight and its helper stay where they are.
    """
    p = Path(meminfo_path)
    if p.is_file():
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0  # kB -> MB
        except Exception:
            pass
    try:
        import psutil

        return float(psutil.virtual_memory().available) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def _parse_parameter_size_b(value: Any) -> float:
    """Parse a parameter-size label ("7B", "3.2b", 7.0) to billions.

    Returns 0.0 when missing or unparseable (fail-open: an unknown size is
    never treated as too large -- the S171 idiom restated by Section 3.1).
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().upper().replace(" ", "")
        match = re.match(r"^([\d.]+)B?$", text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return 0.0
    return 0.0


def _gb_from_size_string(size: Any) -> float | None:
    """Parse a human size string ("20.5GB", "512.0MB", "100B") to GB.

    Mirrors the formats emitted by inference_backend._parse_gguf_filename.
    Returns None when unparseable.
    """
    if not isinstance(size, str):
        return None
    match = _SIZE_STR_RE.match(size.strip())
    if not match:
        return None
    try:
        number = float(match.group(1))
    except ValueError:
        return None
    unit = match.group(2).upper()
    if unit == "GB":
        return number
    if unit == "MB":
        return number / 1024.0
    return number / _BYTES_PER_GIB


def _s1_backend_reachable(warmup: Any) -> bool:
    """Best-effort honesty check for the S1 provenance label.

    True when the warmup object's home module reports its Ollama client
    importable (OLLAMA_AVAILABLE). Objects without the flag (test fakes,
    foreign implementations) count as reachable; the deeper server-down
    ambiguity is inherited from the consumed seam and documented in the
    module docstring.
    """
    try:
        import sys as _sys

        mod = _sys.modules.get(type(warmup).__module__)
        return bool(getattr(mod, "OLLAMA_AVAILABLE", True))
    except Exception:
        return True


def _resolve_emergency_stop() -> Any:
    """Lazy estop resolver (spec 4.5): the existing seams only, fail-open.

    sys.modules is consulted first so a standalone-loaded or test-seeded
    module is reused as-is (the order-independent harness idiom).
    """
    try:
        import sys as _sys

        mod = _sys.modules.get("opti_oignon.emergency_stop")
        if mod is None:
            from opti_oignon import emergency_stop as mod  # type: ignore
        return mod
    except Exception:
        return None


def _resolve_context_manager() -> Any:
    """Lazy ModelLimits seam (the 4.2 clamp authority), fail-open."""
    try:
        import sys as _sys

        mod = _sys.modules.get("opti_oignon.context_manager")
        if mod is None:
            from opti_oignon import context_manager as mod  # type: ignore
        return mod
    except Exception:
        return None


def _parse_duration_s(value: Any) -> float | None:
    """Parse a keep_alive-style duration ('30m', '1h', '90s', 300) to
    seconds; None when unparseable or non-positive (conservative)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    try:
        text = str(value).strip().lower()
        if not text:
            return None
        unit = 1.0
        if text.endswith("h"):
            unit, text = 3600.0, text[:-1]
        elif text.endswith("m"):
            unit, text = 60.0, text[:-1]
        elif text.endswith("s"):
            text = text[:-1]
        seconds = float(text) * unit
        return seconds if seconds > 0 else None
    except Exception:
        return None


def _coerce_epoch_s(value: Any) -> float | None:
    """Coerce an S1 expiry (epoch number, datetime, ISO string) to epoch
    seconds; None when it cannot be interpreted (conservative). Expiries
    are wall-clock stamps, so callers compare against time.time(), never
    against the snapshot's (monotonic by default) ``taken_at``."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        if isinstance(value, datetime):
            return value.timestamp()
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()
    except Exception:
        return None


def _as_bool(value: Any, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_opt_float(value: Any, default: float | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        if isinstance(value, bool):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_kv_override_map(value: Any) -> dict[str, float]:
    """Coerce one ``kv_overrides`` sub-mapping to {lowercase name: GiB
    per 1024 tokens} (S259).

    Conservative by construction: a non-mapping yields the EMPTY table
    (the global coefficient then answers everything, fail-secure); an
    entry whose value does not coerce to a positive float is dropped
    with a warning, never guessed.
    """
    out: dict[str, float] = {}
    if not isinstance(value, dict):
        if value is not None:
            logger.warning(
                "kv_overrides sub-table is not a mapping; ignored"
            )
        return out
    for key, raw_value in value.items():
        try:
            coeff = float(raw_value)
        except (TypeError, ValueError):
            logger.warning(
                "kv_overrides entry %r=%r is not numeric; dropped",
                key,
                raw_value,
            )
            continue
        if coeff <= 0:
            logger.warning(
                "kv_overrides entry %r=%r is not positive; dropped",
                key,
                raw_value,
            )
            continue
        out[str(key).lower()] = coeff
    return out


def _as_weights_override_map(value: Any) -> dict[str, float]:
    """Coerce one ``weights_overrides`` sub-mapping to {lowercase name:
    weights-residency GiB} (S261).

    The S259 kv coercer mirrored as a sibling so the kv path stays
    byte-identical: a non-mapping yields the EMPTY table (the estimator
    chain then answers everything, fail-secure); an entry whose value
    does not coerce to a positive float is dropped with a warning,
    never guessed.
    """
    out: dict[str, float] = {}
    if not isinstance(value, dict):
        if value is not None:
            logger.warning(
                "weights_overrides sub-table is not a mapping; ignored"
            )
        return out
    for key, raw_value in value.items():
        try:
            gib = float(raw_value)
        except (TypeError, ValueError):
            logger.warning(
                "weights_overrides entry %r=%r is not numeric; dropped",
                key,
                raw_value,
            )
            continue
        if gib <= 0:
            logger.warning(
                "weights_overrides entry %r=%r is not positive; dropped",
                key,
                raw_value,
            )
            continue
        out[str(key).lower()] = gib
    return out


def _as_opt_int(value: Any, default: int | None) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Configuration (RESOURCE_GOVERNOR_SPEC.md Section 10)
# ---------------------------------------------------------------------------


@dataclass
class GovernorConfig:
    """Section 10 keys with spec defaults.

    Bloc 0 consumes the measurement subset (enabled, total_vram_gb,
    safety_margin_gb, snapshot_ttl_s, kv_coefficient, ceiling_floor_gb,
    decisions_ring_size); the remaining keys are carried so the file and
    the loader are written once and the later blocs only consume.
    ``enabled`` gates the FUTURE admission behaviour (Bloc 1); measurement
    itself stays available regardless.
    """

    enabled: bool = True
    total_vram_gb: float | None = None
    safety_margin_gb: float = 1.5
    snapshot_ttl_s: float = 2.0
    # GiB of KV cache per 1024 tokens of requested num_ctx (DI-4).
    kv_coefficient: float = 0.5
    # S259 per-model KV overrides (resolution: exact model name >
    # longest matching family substring > the global coefficient above,
    # which stays the fail-secure answer for any model neither table
    # names). Keys are normalised to lowercase at load.
    kv_override_models: dict[str, float] = field(default_factory=dict)
    kv_override_families: dict[str, float] = field(default_factory=dict)
    # S261 per-model weights-residency overrides in GiB (the MoE gap:
    # active-params residency differs from file size). Resolution:
    # exact model name > longest matching family substring > None,
    # which leaves the estimator chain untouched -- an unknown model
    # prices exactly as today. Keys are normalised to lowercase at load.
    weights_override_models: dict[str, float] = field(default_factory=dict)
    weights_override_families: dict[str, float] = field(default_factory=dict)
    # The learned ceiling never drops below this floor (Section 3.2).
    ceiling_floor_gb: float = 4.0
    # Bounded recent-decisions ring, pruned by count (Section 3.2).
    decisions_ring_size: int = 200
    # Later-bloc keys (carried, not consumed in Bloc 0):
    ctx_ladder: list[int] = field(
        default_factory=lambda: [32768, 16384, 8192, 4096]
    )
    ctx_floor: dict[str, int] = field(
        default_factory=lambda: {"chat": 4096, "pipeline": 4096}
    )
    idle_evict_threshold_s: float = 600.0
    pressure_soft_threshold: float = 0.85
    pressure_hard_threshold: float = 0.95
    pressure_keep_alive: str = "5m"
    # Bloc 2 (S225): soft-or-worse pressure must persist this long before
    # the governor writes the warmup keep_alive through its settable
    # property; the original is restored at the first clear observation.
    pressure_sustain_s: float = 60.0
    # Bloc 2 (S225): the bounded window the refusal-rate signal reads.
    pressure_refusal_window_s: float = 60.0
    queue_enabled_per_caller: dict[str, bool] = field(default_factory=dict)
    queue_depth: int = 2
    queue_wait_s: float = 30.0
    rlimits_enabled: bool = False
    rlimits_as_gb: float | None = None
    rlimits_data_gb: float | None = None
    ollama_max_loaded_models: int | None = None
    ollama_num_parallel: int | None = None
    ollama_max_queue: int | None = None
    ollama_spawn_applies: bool = True
    ollama_external_advisory: bool = True


def load_config(config_path: str | Path | None = None) -> GovernorConfig:
    """Load resource_governor.yaml with defaults for missing keys.

    Follows the telemetry/sandbox merge idiom: a dataclass of defaults
    merged key by key with type tolerance; a missing or unparseable file
    yields the defaults with a debug/warning log, never an exception.
    """
    p = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    cfg = GovernorConfig()
    if not p.is_file():
        logger.debug("No resource_governor.yaml found, using defaults")
        return cfg

    try:
        with open(p, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("Failed to parse resource_governor.yaml: %s", exc)
        return cfg

    if not isinstance(raw, dict):
        logger.warning(
            "resource_governor.yaml root is not a mapping; using defaults"
        )
        return cfg

    cfg.enabled = _as_bool(raw.get("enabled"), cfg.enabled)
    cfg.total_vram_gb = _as_opt_float(
        raw.get("total_vram_gb", cfg.total_vram_gb), cfg.total_vram_gb
    )
    cfg.safety_margin_gb = _as_float(
        raw.get("safety_margin_gb"), cfg.safety_margin_gb
    )
    cfg.snapshot_ttl_s = _as_float(raw.get("snapshot_ttl_s"), cfg.snapshot_ttl_s)
    cfg.kv_coefficient = _as_float(raw.get("kv_coefficient"), cfg.kv_coefficient)
    overrides = raw.get("kv_overrides")
    if isinstance(overrides, dict):
        cfg.kv_override_models = _as_kv_override_map(overrides.get("models"))
        cfg.kv_override_families = _as_kv_override_map(
            overrides.get("families")
        )
    elif overrides is not None:
        logger.warning(
            "kv_overrides is not a mapping; per-model KV overrides ignored"
        )
    w_overrides = raw.get("weights_overrides")
    if isinstance(w_overrides, dict):
        cfg.weights_override_models = _as_weights_override_map(
            w_overrides.get("models")
        )
        cfg.weights_override_families = _as_weights_override_map(
            w_overrides.get("families")
        )
    elif w_overrides is not None:
        logger.warning(
            "weights_overrides is not a mapping; weight overrides ignored"
        )
    cfg.ceiling_floor_gb = _as_float(
        raw.get("ceiling_floor_gb"), cfg.ceiling_floor_gb
    )
    cfg.decisions_ring_size = max(
        1, _as_int(raw.get("decisions_ring_size"), cfg.decisions_ring_size)
    )

    ladder = raw.get("ctx_ladder")
    if isinstance(ladder, list) and ladder:
        parsed = [_as_int(v, 0) for v in ladder]
        if all(v > 0 for v in parsed):
            cfg.ctx_ladder = parsed

    floors = raw.get("ctx_floor")
    if isinstance(floors, dict):
        merged = dict(cfg.ctx_floor)
        for key, value in floors.items():
            merged[str(key)] = _as_int(value, merged.get(str(key), 4096))
        cfg.ctx_floor = merged

    cfg.idle_evict_threshold_s = _as_float(
        raw.get("idle_evict_threshold_s"), cfg.idle_evict_threshold_s
    )

    pressure = raw.get("pressure")
    if isinstance(pressure, dict):
        cfg.pressure_soft_threshold = _as_float(
            pressure.get("soft_threshold"), cfg.pressure_soft_threshold
        )
        cfg.pressure_hard_threshold = _as_float(
            pressure.get("hard_threshold"), cfg.pressure_hard_threshold
        )
        cfg.pressure_sustain_s = _as_float(
            pressure.get("sustain_s"), cfg.pressure_sustain_s
        )
        cfg.pressure_refusal_window_s = _as_float(
            pressure.get("refusal_window_s"), cfg.pressure_refusal_window_s
        )
    raw_keep = raw.get("pressure_keep_alive")
    if isinstance(raw_keep, str) and raw_keep:
        cfg.pressure_keep_alive = raw_keep

    queue = raw.get("queue")
    if isinstance(queue, dict):
        enabled_pc = queue.get("enabled_per_caller")
        if isinstance(enabled_pc, dict):
            cfg.queue_enabled_per_caller = {
                str(k): bool(v) for k, v in enabled_pc.items()
            }
        cfg.queue_depth = _as_int(queue.get("depth"), cfg.queue_depth)
        cfg.queue_wait_s = _as_float(queue.get("wait_s"), cfg.queue_wait_s)

    rlimits = raw.get("rlimits")
    if isinstance(rlimits, dict):
        cfg.rlimits_enabled = _as_bool(rlimits.get("enabled"), cfg.rlimits_enabled)
        cfg.rlimits_as_gb = _as_opt_float(
            rlimits.get("as_gb", cfg.rlimits_as_gb), cfg.rlimits_as_gb
        )
        cfg.rlimits_data_gb = _as_opt_float(
            rlimits.get("data_gb", cfg.rlimits_data_gb), cfg.rlimits_data_gb
        )

    ollama_limits = raw.get("ollama_limits")
    if isinstance(ollama_limits, dict):
        cfg.ollama_max_loaded_models = _as_opt_int(
            ollama_limits.get("max_loaded_models", cfg.ollama_max_loaded_models),
            cfg.ollama_max_loaded_models,
        )
        cfg.ollama_num_parallel = _as_opt_int(
            ollama_limits.get("num_parallel", cfg.ollama_num_parallel),
            cfg.ollama_num_parallel,
        )
        cfg.ollama_max_queue = _as_opt_int(
            ollama_limits.get("max_queue", cfg.ollama_max_queue),
            cfg.ollama_max_queue,
        )
        cfg.ollama_spawn_applies = _as_bool(
            ollama_limits.get("spawn_applies"), cfg.ollama_spawn_applies
        )
        cfg.ollama_external_advisory = _as_bool(
            ollama_limits.get("external_advisory"), cfg.ollama_external_advisory
        )

    return cfg


# ---------------------------------------------------------------------------
# R-03 limit management (Bloc 3, Section 6)
# ---------------------------------------------------------------------------

# The three Ollama limit knobs: payload key, GovernorConfig attribute,
# environment variable.
_OLLAMA_LIMIT_ENV = (
    (
        "max_loaded_models",
        "ollama_max_loaded_models",
        "OLLAMA_MAX_LOADED_MODELS",
    ),
    ("num_parallel", "ollama_num_parallel", "OLLAMA_NUM_PARALLEL"),
    ("max_queue", "ollama_max_queue", "OLLAMA_MAX_QUEUE"),
)


def build_ollama_spawn_env(config: GovernorConfig) -> dict[str, str]:
    """Posture (a): the OLLAMA_* env dict for a spawn path (Section 6).

    Pure function from the config to the environment a spawner must
    merge into the Ollama child's environment. Only the configured
    (non-null) keys are emitted, values stringified;
    ``ollama_limits.spawn_applies: false`` yields an empty dict.

    This is the spawn-path CONTRACT: no in-app Ollama spawner exists
    today (verified at the S226 read -- emergency_stop only warms up an
    already running server), so whoever spawns first (a launcher script
    or a future process manager) consumes this helper instead of
    rebuilding the mapping. The wiring entry lives on the standing list.
    """
    if not config.ollama_spawn_applies:
        return {}
    env: dict[str, str] = {}
    for _key, attr, var in _OLLAMA_LIMIT_ENV:
        value = getattr(config, attr, None)
        if value is not None:
            try:
                env[var] = str(int(value))
            except (TypeError, ValueError):
                continue
    return env


def compute_ollama_limits_advisory(
    config: GovernorConfig, env: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Posture (b): compare the configured ollama_limits to what is visible.

    Honesty note on "visible" (Section 6): the only environment this
    process can read without privileged inspection is its OWN
    ``os.environ``, which is not the external server's environment in
    the documented systemd case. A configured key with no visible env
    var therefore reports "unknown" -- values unknown, config not
    enforced externally -- never a guess. ``env`` is injectable for
    tests and defaults to ``os.environ``.

    Returns the status-API shape the Bloc 4 surface will reuse. status
    is one of "not_configured" | "match" | "mismatch" | "unknown";
    mixed observations resolve mismatch > unknown > match; a visible
    value that does not parse as an integer counts as a mismatch.
    Advisory-only by contract: the consumer (the startup security
    checklist) never blocks startup on it, in any mode (the S145
    precedent). Never raises.
    """
    source: Mapping[str, str] = os.environ if env is None else env
    configured: dict[str, int | None] = {}
    visible: dict[str, str | None] = {}
    mismatches: list[dict[str, Any]] = []
    unknown_keys: list[str] = []

    for key, attr, var in _OLLAMA_LIMIT_ENV:
        conf_value = getattr(config, attr, None)
        configured[key] = conf_value
        try:
            raw = source.get(var)
        except Exception:
            raw = None
        visible[var] = raw
        if conf_value is None:
            continue
        if raw is None:
            unknown_keys.append(key)
            continue
        try:
            equal = int(str(raw).strip()) == int(conf_value)
        except (TypeError, ValueError):
            equal = False
        if not equal:
            mismatches.append(
                {
                    "key": key,
                    "env_var": var,
                    "configured": int(conf_value),
                    "visible": raw,
                }
            )

    any_configured = any(v is not None for v in configured.values())
    if not any_configured:
        status = "not_configured"
        detail = "No Ollama limits configured (ollama_limits keys are null)"
    elif mismatches:
        status = "mismatch"
        parts = ", ".join(
            "{var}={vis!r} (configured {conf})".format(
                var=m["env_var"], vis=m["visible"], conf=m["configured"]
            )
            for m in mismatches
        )
        detail = (
            "Configured Ollama limits differ from the visible "
            "environment: " + parts
        )
    elif unknown_keys:
        status = "unknown"
        detail = (
            "Configured Ollama limits are not visible from this process "
            "(OLLAMA_* unset here); values unknown, config not enforced "
            "externally"
        )
    else:
        status = "match"
        detail = "Configured Ollama limits match the visible environment"

    return {
        "status": status,
        "configured": configured,
        "visible": visible,
        "mismatches": mismatches,
        "unknown_keys": unknown_keys,
        "spawn_applies": bool(config.ollama_spawn_applies),
        "external_advisory": bool(config.ollama_external_advisory),
        "detail": detail,
    }


# Once-per-process outcome of the optional rlimits applier (Section 6).
# The FIRST call latches whatever it decided (applied or skipped); the
# llama.cpp load seam may call the applier on every load and stays
# once-effective by construction. Distinct configurations are honestly
# observable only in separate processes (the child-process test idiom).
_RLIMITS_OUTCOME: dict[str, Any] | None = None
_RLIMITS_LOCK = threading.Lock()


def apply_llamacpp_rlimits(
    config: GovernorConfig | None = None,
) -> dict[str, Any]:
    """Optional, off-by-default rlimits for the in-process backend.

    PROCESS-WIDE HONESTY CAVEAT (Section 6, stated wherever the knob is
    documented): ``resource.setrlimit`` caps the ENTIRE Opti-Oignon
    process, not the llama.cpp backend alone -- that is WHY the knob is
    optional and off by default. Admission-side accounting (R-01)
    remains the primary control; this is a hard backstop for the user
    who explicitly asks for one.

    Applies RLIMIT_AS from ``rlimits.as_gb`` and RLIMIT_DATA from
    ``rlimits.data_gb`` (each independently optional), lowering only
    the SOFT limit and never above the existing hard limit. Latches its
    outcome ONCE per process and returns the recorded dict on every
    later call. Fail-open on every unavailability: the ``resource``
    module absent (non-POSIX), ``setrlimit`` raising, null or
    non-coercible values -- never raises, never blocks a load.
    """
    global _RLIMITS_OUTCOME
    with _RLIMITS_LOCK:
        if _RLIMITS_OUTCOME is not None:
            return _RLIMITS_OUTCOME

        try:
            cfg = config if config is not None else load_config()
        except Exception:
            cfg = GovernorConfig()
        outcome: dict[str, Any] = {
            "applied": False,
            "reason": "",
            "as_bytes": None,
            "data_bytes": None,
        }

        if not cfg.rlimits_enabled:
            outcome["reason"] = "disabled"
            _RLIMITS_OUTCOME = outcome
            return outcome

        try:
            import resource as _resource
        except Exception as exc:
            outcome["reason"] = f"resource module unavailable: {exc}"
            _RLIMITS_OUTCOME = outcome
            return outcome

        targets = (
            ("as_bytes", "RLIMIT_AS", cfg.rlimits_as_gb),
            ("data_bytes", "RLIMIT_DATA", cfg.rlimits_data_gb),
        )
        applied_any = False
        reasons: list[str] = []
        for field_name, limit_name, gb in targets:
            if gb is None:
                continue
            try:
                limit = getattr(_resource, limit_name)
                target = int(float(gb) * (1024 ** 3))
                if target <= 0:
                    reasons.append(
                        f"{limit_name}: non-positive value skipped"
                    )
                    continue
                _soft, hard = _resource.getrlimit(limit)
                new_soft = target
                if hard != _resource.RLIM_INFINITY:
                    new_soft = min(target, hard)
                _resource.setrlimit(limit, (new_soft, hard))
                outcome[field_name] = new_soft
                applied_any = True
            except Exception as exc:
                reasons.append(f"{limit_name}: {exc}")

        outcome["applied"] = applied_any
        if applied_any:
            outcome["reason"] = "applied" + (
                " ({})".format("; ".join(reasons)) if reasons else ""
            )
            logger.warning(
                "Process-wide rlimits applied (caps the ENTIRE process, "
                "not the backend alone): AS=%s DATA=%s",
                outcome["as_bytes"],
                outcome["data_bytes"],
            )
        else:
            outcome["reason"] = (
                "; ".join(reasons) if reasons else "no limit values configured"
            )
        _RLIMITS_OUTCOME = outcome
        return outcome


# ---------------------------------------------------------------------------
# Snapshot views (Section 3)
# ---------------------------------------------------------------------------


@dataclass
class LoadedModelView:
    """One S1 entry: a model the Ollama ps view reports as loaded."""

    name: str
    size_vram_bytes: int = 0
    expires_at: float | None = None
    context_length: int | None = None
    digest: str | None = None

    @property
    def size_vram_gb(self) -> float:
        return self.size_vram_bytes / _BYTES_PER_GIB if self.size_vram_bytes else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size_vram_bytes": self.size_vram_bytes,
            "size_vram_gb": round(self.size_vram_gb, 3),
            "expires_at": self.expires_at,
            "context_length": self.context_length,
            "digest": self.digest,
        }


@dataclass
class BackendResidentView:
    """One S2 entry: an in-process (backend-resident) model not in S1.

    ``basis`` names how ``estimated_gb`` was obtained: "learned" (the
    adapt store), "static_table" (the S3 import), "file_size" (GGUF
    weight size as a floor) or "unknown" (no estimate; never treated as
    too large, Section 3.1).
    """

    name: str
    backend: str
    estimated_gb: float | None = None
    basis: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "estimated_gb": (
                round(self.estimated_gb, 3) if self.estimated_gb is not None else None
            ),
            "basis": self.basis,
        }


@dataclass
class ResourceSnapshot:
    """The assembled measurement view (Section 3).

    ``vram_available_gb`` is RAW capacity minus in-use: the safety margin
    is deliberately NOT subtracted here; applying it belongs to the Bloc 1
    fit computation (spec Section 4.2). ``sources`` is the honest
    provenance list naming exactly which read paths contributed.
    ``taken_at`` is on the governor's clock (monotonic by default).
    """

    taken_at: float
    ttl_s: float
    loaded: list[LoadedModelView] = field(default_factory=list)
    backend_resident: list[BackendResidentView] = field(default_factory=list)
    capacity_gb: float | None = None
    capacity_source: str = "unknown"
    vram_in_use_gb: float = 0.0
    vram_available_gb: float | None = None
    vram_status: str = "ok"
    ram_available_mb: float = 0.0
    sources: list[str] = field(default_factory=list)

    def age_s(self, now: float) -> float:
        return max(0.0, now - self.taken_at)

    def is_stale(self, now: float) -> bool:
        return self.age_s(now) > self.ttl_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "taken_at": self.taken_at,
            "ttl_s": self.ttl_s,
            "loaded": [m.to_dict() for m in self.loaded],
            "backend_resident": [m.to_dict() for m in self.backend_resident],
            "capacity_gb": self.capacity_gb,
            "capacity_source": self.capacity_source,
            "vram_in_use_gb": round(self.vram_in_use_gb, 3),
            "vram_available_gb": (
                round(self.vram_available_gb, 3)
                if self.vram_available_gb is not None
                else None
            ),
            "vram_status": self.vram_status,
            "ram_available_mb": round(self.ram_available_mb, 1),
            "sources": list(self.sources),
        }


# ---------------------------------------------------------------------------
# Bloc 1: the admission ticket (Section 4.4) and the typed refusal
# ---------------------------------------------------------------------------


@dataclass
class AdmissionDecision:
    """The admission ticket (spec Section 4.4).

    The first nine fields are the contract shape ({admitted, model,
    num_ctx, num_gpu or None, keep_alive override or None, action in
    {admit, downsize, refuse, queue}, reason, snapshot provenance, ticket
    id}); the trailing fields are internal companions (accounting,
    testability, payload capture) and not part of the minimum surface.
    num_gpu stays None (conservative: full offload when the fit
    holds means no option is sent; computed partial offload stays
    deferred behind a flag); keep_alive carries the Bloc 2 soft-pressure
    override when the pressure signal fills it (S225).
    """

    admitted: bool
    model: str
    num_ctx: int | None = None
    num_gpu: int | None = None
    keep_alive: str | None = None
    action: str = "admit"
    reason: str = ""
    provenance: list[str] = field(default_factory=list)
    ticket_id: str = ""
    # -- internal companions (not part of the 4.4 minimum shape) ----------
    caller: str = "chat"
    requested_ctx: int | None = None
    load_expected: bool = False
    conditional_on_eviction: bool = False
    shortfall_gb: float | None = None
    is_estop: bool = False
    payload: dict[str, Any] = field(default_factory=dict)

    def refusal_payload(self) -> dict[str, Any]:
        """The honest refusal body, mirroring the estop idiom (D3).

        The estop case returns the payload captured at decision time from
        emergency_stop.refusal_payload() (spec 4.5); the resource case
        names the model, the shortfall and the options.
        """
        if self.payload:
            return dict(self.payload)
        shortfall = (
            f" (short by {self.shortfall_gb:.1f} GB)"
            if self.shortfall_gb is not None
            else ""
        )
        return {
            "error": "resource_admission_refused",
            "message": (
                f"Not enough resources to load {self.model}{shortfall}:"
                " evict idle models, pick a smaller model, or lower the"
                " context."
            ),
            "model": self.model,
            "shortfall_gb": self.shortfall_gb,
            "options": [
                "evict idle models",
                "pick a smaller model",
                "lower context",
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "model": self.model,
            "num_ctx": self.num_ctx,
            "num_gpu": self.num_gpu,
            "keep_alive": self.keep_alive,
            "action": self.action,
            "reason": self.reason,
            "provenance": list(self.provenance),
            "ticket_id": self.ticket_id,
            "caller": self.caller,
            "requested_ctx": self.requested_ctx,
            "load_expected": self.load_expected,
            "conditional_on_eviction": self.conditional_on_eviction,
            "shortfall_gb": self.shortfall_gb,
        }


class GovernorRefusal(RuntimeError):
    """Typed refusal raised by the mechanical backend gate (4.1/4.4)."""

    def __init__(self, decision: AdmissionDecision):
        super().__init__(
            decision.refusal_payload().get(
                "message", "resource admission refused"
            )
        )
        self.decision = decision


_ticket_local = threading.local()


def get_active_ticket() -> AdmissionDecision | None:
    """The calling thread's active admission ticket, if a funnel set one."""
    return getattr(_ticket_local, "ticket", None)


def set_active_ticket(decision: AdmissionDecision | None) -> None:
    """Set the thread-local ticket the backend gate will see (4.4)."""
    _ticket_local.ticket = decision


def clear_active_ticket() -> None:
    """Drop the thread-local ticket."""
    _ticket_local.ticket = None


@contextmanager
def ticket_scope(decision: AdmissionDecision | None):
    """Hold an admission ticket around a backend call (Section 4.4).

    The pass-through mechanism arbitrated at the S224 read gate: a thread
    local, never an options key (an options sidecar would leak a private
    key to the transport on the direct-ollama fallback paths). The hook at
    the generate/stream heads reads it through get_active_ticket(). A
    None decision is a no-op scope so call sites stay unconditional.
    Streaming note: a generator head executes at first iteration, so the
    scope (or set_active_ticket) must live on the consuming thread.
    """
    if decision is None:
        yield
        return
    previous = get_active_ticket()
    set_active_ticket(decision)
    try:
        yield
    finally:
        set_active_ticket(previous)


# ---------------------------------------------------------------------------
# Measure-and-adapt store (Section 3.2)
# ---------------------------------------------------------------------------


class AdaptStore:
    """Persistent measure-and-adapt state (data/resource_governor.db).

    Standard per-feature DB pattern (the benchmark ResultsStore
    precedent): safe_connect, schema init under a threading.Lock,
    open-use-close connections per operation, parameterized SQL only.
    Holds derived, regenerable state: learned per-model VRAM cost (keyed
    name+digest when the digest is present), the learned capacity ceiling
    (fast down, slow up, config floor) and the bounded recent-decisions
    ring (schema and prune-by-count land in this bloc; Bloc 1 writes the
    rows on the admission path).
    """

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = str(db_path or _DEFAULT_DB_PATH)
        parent = os.path.dirname(self._db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> Any:
        return _safe_connect(self._db_path, check_same_thread=False)

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS model_costs (
                        name TEXT NOT NULL,
                        digest TEXT NOT NULL DEFAULT '',
                        size_vram_bytes INTEGER NOT NULL,
                        num_ctx INTEGER,
                        observed_at REAL NOT NULL,
                        PRIMARY KEY (name, digest)
                    );

                    CREATE TABLE IF NOT EXISTS ceiling (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        learned_ceiling_gb REAL,
                        successes_above INTEGER NOT NULL DEFAULT 0,
                        updated_at REAL NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS decisions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts REAL NOT NULL,
                        caller TEXT NOT NULL,
                        model TEXT NOT NULL,
                        requested_ctx INTEGER,
                        admitted_ctx INTEGER,
                        decision TEXT NOT NULL,
                        reason TEXT NOT NULL DEFAULT ''
                    );

                    CREATE INDEX IF NOT EXISTS idx_costs_name
                        ON model_costs(name);
                    CREATE INDEX IF NOT EXISTS idx_decisions_ts
                        ON decisions(ts);
                    """
                )
                conn.commit()
            finally:
                conn.close()

    # -- learned per-model cost --------------------------------------------

    def record_model_cost(
        self,
        name: str,
        digest: str | None,
        size_vram_bytes: int,
        num_ctx: int | None = None,
        observed_at: float | None = None,
    ) -> None:
        """Persist an observed per-model VRAM cost (supersedes statics)."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO model_costs
                       (name, digest, size_vram_bytes, num_ctx, observed_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        name,
                        digest or "",
                        int(size_vram_bytes),
                        num_ctx,
                        observed_at if observed_at is not None else time.time(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def get_model_cost(
        self, name: str, digest: str | None = None
    ) -> dict[str, Any] | None:
        """Latest learned cost for a model; exact digest row preferred."""
        with self._lock:
            conn = self._connect()
            try:
                row = None
                if digest:
                    row = conn.execute(
                        """SELECT name, digest, size_vram_bytes, num_ctx,
                                  observed_at
                           FROM model_costs WHERE name = ? AND digest = ?""",
                        (name, digest),
                    ).fetchone()
                if row is None:
                    row = conn.execute(
                        """SELECT name, digest, size_vram_bytes, num_ctx,
                                  observed_at
                           FROM model_costs WHERE name = ?
                           ORDER BY observed_at DESC LIMIT 1""",
                        (name,),
                    ).fetchone()
                if row is None:
                    return None
                return {
                    "name": row[0],
                    "digest": row[1],
                    "size_vram_bytes": row[2],
                    "num_ctx": row[3],
                    "observed_at": row[4],
                }
            finally:
                conn.close()

    # -- learned capacity ceiling (fast down, slow up) -----------------------

    def get_learned_ceiling(self) -> float | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT learned_ceiling_gb FROM ceiling WHERE id = 1"
                ).fetchone()
                if row is None or row[0] is None:
                    return None
                return float(row[0])
            finally:
                conn.close()

    def _write_ceiling(
        self, conn: Any, ceiling_gb: float | None, successes: int, now: float
    ) -> None:
        conn.execute(
            """INSERT OR REPLACE INTO ceiling
               (id, learned_ceiling_gb, successes_above, updated_at)
               VALUES (1, ?, ?, ?)""",
            (ceiling_gb, successes, now),
        )

    def record_load_failure(
        self,
        observed_in_use_gb: float,
        safety_margin_gb: float,
        floor_gb: float,
        now: float | None = None,
    ) -> float:
        """Fast-down: a failure whose admission predicted a fit lowers the
        working ceiling to (observed in-use at failure) minus the safety
        margin, never below the config floor, immediately. Resets the
        slow-up success counter. Returns the new ceiling.
        """
        ts = now if now is not None else time.time()
        candidate = max(float(floor_gb), float(observed_in_use_gb) - float(safety_margin_gb))
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT learned_ceiling_gb FROM ceiling WHERE id = 1"
                ).fetchone()
                current = row[0] if row is not None else None
                new_ceiling = (
                    candidate if current is None else min(float(current), candidate)
                )
                self._write_ceiling(conn, new_ceiling, 0, ts)
                conn.commit()
                return new_ceiling
            finally:
                conn.close()

    def record_load_success(
        self,
        total_in_use_gb: float,
        configured_capacity_gb: float | None,
        now: float | None = None,
    ) -> float | None:
        """Slow-up: a stretch of successes ABOVE the learned ceiling
        relaxes it back toward the configured capacity.

        Increments the consecutive-success counter only when the observed
        total in-use exceeds the learned ceiling (evidence the ceiling is
        too pessimistic); every _CEILING_RELAX_AFTER_SUCCESSES such
        successes raise the ceiling by _CEILING_RELAX_STEP_GB, capped at
        the configured capacity when one is set. Below-ceiling successes
        carry no evidence and change nothing; only a failure resets the
        counter. Returns the (possibly unchanged) ceiling, or None when no
        ceiling is learned.
        """
        ts = now if now is not None else time.time()
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT learned_ceiling_gb, successes_above FROM ceiling"
                    " WHERE id = 1"
                ).fetchone()
                if row is None or row[0] is None:
                    return None
                ceiling = float(row[0])
                successes = int(row[1] or 0)
                if float(total_in_use_gb) <= ceiling:
                    return ceiling
                successes += 1
                if successes >= _CEILING_RELAX_AFTER_SUCCESSES:
                    ceiling += _CEILING_RELAX_STEP_GB
                    if configured_capacity_gb is not None:
                        ceiling = min(ceiling, float(configured_capacity_gb))
                    successes = 0
                self._write_ceiling(conn, ceiling, successes, ts)
                conn.commit()
                return ceiling
            finally:
                conn.close()

    # -- bounded recent-decisions ring ---------------------------------------

    def record_decision(
        self,
        caller: str,
        model: str,
        requested_ctx: int | None,
        admitted_ctx: int | None,
        decision: str,
        reason: str = "",
        ring_size: int = 200,
        ts: float | None = None,
    ) -> None:
        """Append one admission decision and prune the ring by count.

        Bloc 1 is the writer on the admission path; this bloc lands the
        table and the prune so the ring is bounded from day one.
        """
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO decisions
                       (ts, caller, model, requested_ctx, admitted_ctx,
                        decision, reason)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        ts if ts is not None else time.time(),
                        caller,
                        model,
                        requested_ctx,
                        admitted_ctx,
                        decision,
                        reason,
                    ),
                )
                conn.execute(
                    """DELETE FROM decisions WHERE id NOT IN (
                           SELECT id FROM decisions ORDER BY id DESC LIMIT ?
                       )""",
                    (max(1, int(ring_size)),),
                )
                conn.commit()
            finally:
                conn.close()

    def recent_decisions(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT id, ts, caller, model, requested_ctx,
                              admitted_ctx, decision, reason
                       FROM decisions ORDER BY id DESC LIMIT ?""",
                    (max(1, int(limit)),),
                ).fetchall()
                return [
                    {
                        "id": r[0],
                        "ts": r[1],
                        "caller": r[2],
                        "model": r[3],
                        "requested_ctx": r[4],
                        "admitted_ctx": r[5],
                        "decision": r[6],
                        "reason": r[7],
                    }
                    for r in rows
                ]
            finally:
                conn.close()

    def decision_count(self) -> int:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()
                return int(row[0]) if row else 0
            finally:
                conn.close()


# ---------------------------------------------------------------------------
# The governor (measurement only this bloc)
# ---------------------------------------------------------------------------


class ResourceGovernor:
    """Governor core: measures, caches, learns (Bloc 0); admits (Bloc 1).

    Every collaborator is injectable for container-provable tests: the
    warmup (S1), the backend registry (S2), the clock (TTL), the meminfo
    path (S4 RAM), the config and DB paths. Passing ``None`` explicitly
    for warmup/registry means "deliberately absent"; leaving the argument
    unset resolves the production defaults lazily and conditionally.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        db_path: str | Path | None = None,
        warmup: Any = _UNSET,
        registry: Any = _UNSET,
        clock: Callable[[], float] = time.monotonic,
        meminfo_path: str | Path = "/proc/meminfo",
    ):
        self._config = load_config(config_path)
        self._store = AdaptStore(db_path)
        if warmup is _UNSET:
            self._warmup = _default_warmup if MODEL_WARMUP_AVAILABLE else None
        else:
            self._warmup = warmup
        self._registry_override = registry
        self._clock = clock
        self._meminfo_path = meminfo_path
        self._estimator = (
            _VRAMBudgetCalculator() if SPECULATIVE_AVAILABLE else None
        )
        self._cache_lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        self._snapshot: ResourceSnapshot | None = None
        self._pending_attribution: dict[str, int | None] = {}
        self._refresh_in_flight = False
        self._capacity_warning_emitted = False
        # S224 (R-04 wiring): the admission path observes the estop flag;
        # a transition triggers the drain/resume invalidation hooks
        # without editing emergency_stop. None until first observation.
        self._last_estop_seen: bool | None = None
        # S225 (Bloc 2): runtime backpressure state. The refusal-rate
        # window is in-memory by design (a runtime signal, never
        # persisted; the maxlen is a memory bound, the time window is
        # the config key). The remembered keep_alive original does NOT
        # survive a process restart: the warmup re-initialises at its
        # own default, which is the honest restore in that case.
        self._refusal_events: deque = deque(maxlen=512)
        self._pressure_soft_since: float | None = None
        self._keep_alive_original: str | None = None
        self._last_pressure_level: str | None = None
        # S225 queue: waiters block on the condition in bounded slices;
        # the invalidation hooks notify it (capacity may have moved).
        self._queue_cond = threading.Condition()
        self._queue_depth = 0

    # -- public configuration / store accessors ------------------------------

    @property
    def config(self) -> GovernorConfig:
        return self._config

    @property
    def store(self) -> AdaptStore:
        return self._store

    # -- snapshot API (Section 3 cache contract) ------------------------------

    def refresh(self, force: bool = False) -> ResourceSnapshot:
        """Build (or return) the snapshot. Synchronous.

        With ``force=False`` a fresh cached snapshot is returned as-is;
        a stale or missing one is rebuilt. Building never raises into the
        caller: every source section degrades individually (Section 3.1).
        """
        now = self._clock()
        with self._cache_lock:
            snap = self._snapshot
        if not force and snap is not None and not snap.is_stale(now):
            return snap
        with self._refresh_lock:
            # Re-check under the build lock: a concurrent refresh may have
            # just produced a fresh snapshot.
            now = self._clock()
            with self._cache_lock:
                snap = self._snapshot
            if not force and snap is not None and not snap.is_stale(now):
                return snap
            built = self._build_snapshot()
            with self._cache_lock:
                self._snapshot = built
            return built

    def get_snapshot(self) -> ResourceSnapshot:
        """Fresh-or-rebuild, synchronous (the simple consumer path)."""
        return self.refresh(force=False)

    def get_snapshot_fast(self) -> ResourceSnapshot:
        """The Bloc 1 admission fast-path primitive (Section 3).

        Returns the cached snapshot immediately -- even when stale, the
        current decision uses the cached values conservatively -- and
        triggers at most one asynchronous background refresh. Only a
        first-ever call (no cache at all) builds synchronously.
        """
        now = self._clock()
        with self._cache_lock:
            snap = self._snapshot
        if snap is None:
            return self.refresh(force=True)
        if snap.is_stale(now):
            self._spawn_background_refresh()
        return snap

    def _spawn_background_refresh(self) -> None:
        with self._cache_lock:
            if self._refresh_in_flight:
                return
            self._refresh_in_flight = True
        thread = threading.Thread(
            target=self._background_refresh,
            name="resource-governor-refresh",
            daemon=True,
        )
        thread.start()

    def _background_refresh(self) -> None:
        try:
            self.refresh(force=True)
        except Exception as exc:  # pragma: no cover - refresh never raises
            logger.debug("Background snapshot refresh failed: %s", exc)
        finally:
            with self._cache_lock:
                self._refresh_in_flight = False

    # -- eager invalidation hooks (callable; Bloc 1 wires the callers) --------

    def invalidate_on_load(
        self, model: str, requested_num_ctx: int | None = None
    ) -> None:
        """An admitted load happened: drop the cache and register the
        model for post-load cost attribution at the next fresh ps view."""
        with self._cache_lock:
            self._pending_attribution[model] = requested_num_ctx
            self._snapshot = None
        self._notify_queue()

    def invalidate_on_evict(self, model: str | None = None) -> None:
        with self._cache_lock:
            self._snapshot = None
        self._notify_queue()

    def invalidate_on_estop_drain(self) -> None:
        with self._cache_lock:
            self._snapshot = None
        self._notify_queue()

    def invalidate_on_resume(self) -> None:
        with self._cache_lock:
            self._snapshot = None
        self._notify_queue()

    def _notify_queue(self) -> None:
        """Wake queued admissions (S225): the world may have moved.

        Called by every invalidation hook AFTER the cache lock is
        released (strictly sequential locking, never nested), so a
        waiter woken here re-runs admission against a rebuilt view.
        The estop drain notification is what actively releases waiters
        to refusal: their re-admission honours the flag first.
        """
        with self._queue_cond:
            self._queue_cond.notify_all()

    # -- estimation API --------------------------------------------------------

    def resolve_kv_coefficient(self, model: str | None) -> float:
        """Per-model KV coefficient (S259).

        Resolution order: an exact entry in ``kv_override_models``, else
        the LONGEST ``kv_override_families`` key contained in the
        lowercased name, else the global ``kv_coefficient`` -- the
        fail-secure default for any model the tables do not cover (an
        unknown model is never under-budgeted). Matching is
        case-insensitive; the tables hold lowercase keys by load-time
        normalisation.
        """
        if not model:
            return self._config.kv_coefficient
        name = str(model).lower()
        exact = self._config.kv_override_models.get(name)
        if exact is not None:
            return exact
        best: float | None = None
        best_len = -1
        for family, coeff in self._config.kv_override_families.items():
            if family and family in name and len(family) > best_len:
                best, best_len = coeff, len(family)
        if best is not None:
            return best
        return self._config.kv_coefficient

    def estimate_kv_cache_gb(
        self, num_ctx: int | None, model: str | None = None
    ) -> float:
        """KV-cache increment as a function of the requested num_ctx.

        ``kv_coefficient`` is GiB per 1024 tokens (DI-4); 0 tokens or an
        unset request cost nothing. S259: the optional ``model`` kwarg
        routes through :meth:`resolve_kv_coefficient`; without it the
        global coefficient applies, byte-compatible with every
        pre-existing caller.
        """
        if not num_ctx or num_ctx <= 0:
            return 0.0
        return (float(num_ctx) / 1024.0) * self.resolve_kv_coefficient(model)

    def estimate_model_vram_gb(
        self, model: str, digest: str | None = None
    ) -> tuple[float | None, str]:
        """Best-available weight-cost estimate for one model, with basis.

        Order: live S1 observation (when the cached snapshot holds the
        model with a positive size_vram) > learned cost (the adapt store)
        > the S3 static table via registry metadata > GGUF file size as a
        floor > (None, "unknown") -- never "too large" (Section 3.1).
        """
        with self._cache_lock:
            snap = self._snapshot
        if snap is not None:
            for view in snap.loaded:
                if view.name == model and view.size_vram_bytes > 0:
                    return view.size_vram_gb, "observed"

        learned = self._store.get_model_cost(model, digest)
        if learned is not None and learned.get("size_vram_bytes"):
            return learned["size_vram_bytes"] / _BYTES_PER_GIB, "learned"

        registry = self._resolve_registry()
        if registry is not None:
            try:
                backends = list(registry.backends())
            except Exception as exc:
                logger.debug("Registry backends() failed: %s", exc)
                backends = []
            for backend in backends:
                est, basis = self._estimate_from_backend(backend, model)
                if est is not None:
                    return est, basis
        return None, "unknown"

    def resolve_weights_override(
        self, model: str | None
    ) -> float | None:
        """Per-model weights-residency override in GiB (S261).

        Resolution order: an exact entry in ``weights_override_models``,
        else the LONGEST ``weights_override_families`` key contained in
        the lowercased name, else None -- which leaves
        :meth:`estimate_model_vram_gb` as the answer, so an unknown
        model prices exactly as today (fail-secure: the table can only
        replace the estimate for models the operator named, never
        invent a budget for the rest). Matching is case-insensitive;
        the tables hold lowercase keys by load-time normalisation.
        """
        if not model:
            return None
        name = str(model).lower()
        exact = self._config.weights_override_models.get(name)
        if exact is not None:
            return exact
        best: float | None = None
        best_len = -1
        for family, gib in self._config.weights_override_families.items():
            if family and family in name and len(family) > best_len:
                best, best_len = gib, len(family)
        return best

    # -- internals -------------------------------------------------------------

    def _resolve_registry(self) -> Any:
        if self._registry_override is not _UNSET:
            return self._registry_override
        if INFERENCE_BACKEND_AVAILABLE and _get_backend_registry is not None:
            try:
                return _get_backend_registry()
            except Exception as exc:
                logger.debug("Backend registry unavailable: %s", exc)
        return None

    def _estimate_from_backend(
        self, backend: Any, model: str
    ) -> tuple[float | None, str]:
        """S3-then-file-size estimation from one backend's metadata."""
        try:
            info = backend.model_info(model)
        except Exception as exc:
            logger.debug("model_info(%s) failed: %s", model, exc)
            return None, "unknown"
        if info is None:
            return None, "unknown"

        params_b = _parse_parameter_size_b(getattr(info, "parameter_size", None))
        if params_b > 0 and self._estimator is not None:
            quant = getattr(info, "quantization_level", None) or "Q4_K_M"
            try:
                return (
                    float(self._estimator.estimate_model_vram(params_b, str(quant))),
                    "static_table",
                )
            except Exception as exc:
                logger.debug("Static VRAM estimate failed for %s: %s", model, exc)

        path = getattr(info, "path", None)
        if path:
            try:
                p = Path(path)
                if p.is_file():
                    return p.stat().st_size / _BYTES_PER_GIB, "file_size"
            except Exception:
                pass
        gb = _gb_from_size_string(getattr(info, "size", None))
        if gb is not None:
            return gb, "file_size"
        return None, "unknown"

    def _read_s1(self) -> tuple[list[LoadedModelView], bool]:
        """The S1 read: loaded set through the consumed warmup seam."""
        warmup = self._warmup
        if warmup is None:
            return [], False
        if not _s1_backend_reachable(warmup):
            return [], False
        try:
            models = warmup.get_loaded_models()
        except Exception as exc:
            logger.debug("S1 ps view failed: %s", exc)
            return [], False
        if not isinstance(models, list):
            return [], False
        views: list[LoadedModelView] = []
        for m in models:
            try:
                views.append(
                    LoadedModelView(
                        name=str(getattr(m, "name", "unknown")),
                        size_vram_bytes=int(getattr(m, "size_vram", 0) or 0),
                        expires_at=getattr(m, "expires_at", None),
                        context_length=getattr(m, "context_length", None),
                        digest=getattr(m, "digest", None),
                    )
                )
            except Exception as exc:
                logger.debug("Skipping malformed ps entry: %s", exc)
        return views, True

    def _attribute_pending(self, loaded: list[LoadedModelView]) -> None:
        """DI-9: write learned costs for pending post-load attributions."""
        with self._cache_lock:
            if not self._pending_attribution:
                return
            pending = dict(self._pending_attribution)
        for view in loaded:
            if view.name in pending and view.size_vram_bytes > 0:
                try:
                    self._store.record_model_cost(
                        view.name,
                        view.digest,
                        view.size_vram_bytes,
                        pending[view.name],
                    )
                except Exception as exc:
                    logger.debug(
                        "Cost attribution failed for %s: %s", view.name, exc
                    )
                with self._cache_lock:
                    self._pending_attribution.pop(view.name, None)

    def _read_s2(
        self, s1_names: set[str]
    ) -> tuple[list[BackendResidentView], bool, bool]:
        """The S2 read: backend-resident models not in the S1 view.

        Returns (views, s2_answered, s3_used). Only backends exposing the
        in-process ``_loaded_models`` dict idiom contribute entries; the
        Ollama backend's loaded set IS the S1 view and is never double
        counted.
        """
        registry = self._resolve_registry()
        if registry is None:
            return [], False, False
        try:
            backends = list(registry.backends())
        except Exception as exc:
            logger.debug("S2 registry read failed: %s", exc)
            return [], False, False
        views: list[BackendResidentView] = []
        s3_used = False
        for backend in backends:
            resident = getattr(backend, "_loaded_models", None)
            if not isinstance(resident, dict):
                continue
            backend_name = str(getattr(backend, "name", "unknown"))
            for name in list(resident.keys()):
                if name in s1_names:
                    continue
                learned = self._store.get_model_cost(name)
                if learned is not None and learned.get("size_vram_bytes"):
                    views.append(
                        BackendResidentView(
                            name=name,
                            backend=backend_name,
                            estimated_gb=learned["size_vram_bytes"] / _BYTES_PER_GIB,
                            basis="learned",
                        )
                    )
                    continue
                est, basis = self._estimate_from_backend(backend, name)
                if basis == "static_table":
                    s3_used = True
                views.append(
                    BackendResidentView(
                        name=name,
                        backend=backend_name,
                        estimated_gb=est,
                        basis=basis,
                    )
                )
        return views, True, s3_used

    def _build_snapshot(self) -> ResourceSnapshot:
        now = self._clock()
        sources: list[str] = []

        loaded, s1_answered = self._read_s1()
        if s1_answered:
            sources.append("S1")
            self._attribute_pending(loaded)

        resident, s2_answered, s3_used = self._read_s2(
            {view.name for view in loaded}
        )
        if s2_answered:
            sources.append("S2")
        if s3_used:
            sources.append("S3")

        configured = self._config.total_vram_gb
        learned_ceiling = None
        try:
            learned_ceiling = self._store.get_learned_ceiling()
        except Exception as exc:
            logger.debug("Learned ceiling read failed: %s", exc)
        if configured is not None:
            sources.append("S4-capacity-config")
        if learned_ceiling is not None:
            sources.append("S4-capacity-learned")

        if configured is not None and learned_ceiling is not None:
            capacity: float | None = min(configured, learned_ceiling)
            capacity_source = "config+learned"
        elif configured is not None:
            capacity = configured
            capacity_source = "config"
        elif learned_ceiling is not None:
            capacity = learned_ceiling
            capacity_source = "learned"
        else:
            capacity = None
            capacity_source = "unknown"

        ram_mb = 0.0
        try:
            ram_mb = _read_available_ram_mb(self._meminfo_path)
        except Exception as exc:  # pragma: no cover - the reader never raises
            logger.debug("S4 RAM read failed: %s", exc)
        if ram_mb > 0:
            sources.append("S4-ram")

        in_use = sum(v.size_vram_gb for v in loaded) + sum(
            (v.estimated_gb or 0.0) for v in resident
        )

        if capacity is None:
            vram_status = "disabled_capacity_unknown"
            available: float | None = None
            if not self._capacity_warning_emitted:
                logger.warning(
                    "Resource governor: total VRAM capacity unknown "
                    "(total_vram_gb is null and no ceiling is learned); the "
                    "VRAM half of measurement reports disabled and any "
                    "future admission stays fail-open (spec Section 3.1). "
                    "Set total_vram_gb in resource_governor.yaml on the "
                    "host."
                )
                self._capacity_warning_emitted = True
        else:
            vram_status = "ok"
            available = max(0.0, capacity - in_use)

        return ResourceSnapshot(
            taken_at=now,
            ttl_s=self._config.snapshot_ttl_s,
            loaded=loaded,
            backend_resident=resident,
            capacity_gb=capacity,
            capacity_source=capacity_source,
            vram_in_use_gb=in_use,
            vram_available_gb=available,
            vram_status=vram_status,
            ram_available_mb=ram_mb,
            sources=sources,
        )

    # -- learning passthroughs (rules land now; Bloc 1 calls them) -------------

    def record_load_failure(self, observed_in_use_gb: float) -> float:
        """Fast-down the learned ceiling after a failed load (Section 3.2)."""
        new_ceiling = self._store.record_load_failure(
            observed_in_use_gb,
            self._config.safety_margin_gb,
            self._config.ceiling_floor_gb,
        )
        self.invalidate_on_evict()
        logger.info(
            "Learned VRAM ceiling lowered to %.2f GB after a load failure",
            new_ceiling,
        )
        return new_ceiling

    def record_load_success(self, total_in_use_gb: float) -> float | None:
        """Feed a successful load into the slow-up rule (Section 3.2)."""
        return self._store.record_load_success(
            total_in_use_gb, self._config.total_vram_gb
        )

    def record_decision(
        self,
        caller: str,
        model: str,
        requested_ctx: int | None,
        admitted_ctx: int | None,
        decision: str,
        reason: str = "",
    ) -> None:
        """Append to the bounded recent-decisions ring (Bloc 1's writer)."""
        self._store.record_decision(
            caller,
            model,
            requested_ctx,
            admitted_ctx,
            decision,
            reason,
            ring_size=self._config.decisions_ring_size,
        )

    # -- Bloc 2: runtime backpressure -- the pressure signal (Section 5) ------

    def pressure_state(self) -> dict[str, Any]:
        """The R-02 pressure signal, the shape the Bloc 4 status API
        will surface.

        Level is the max of two contributions: in_use over EFFECTIVE
        capacity (the snapshot's capacity_gb already folds the learned
        ceiling when lower, Section 3.2) against the config soft/hard
        thresholds, and the bounded refusal-rate window (resource
        refusals only; estop refusals are not a resource signal).
        Reading the state also applies the sustained-pressure
        keep_alive policy (override and restore through the warmup's
        existing settable property, Section 5 step 1).
        """
        return self._pressure_from_snapshot(self.get_snapshot_fast())

    # -- Bloc 3: R-03 limit management -- the advisory seat (Section 6) -------

    def ollama_limits_advisory(self) -> dict[str, Any]:
        """The R-03 external-Ollama advisory in the status-API shape.

        Thin delegation to :func:`compute_ollama_limits_advisory` with
        this governor's config; the seat the Bloc 4 status surface
        reads. The startup security checklist consumes the pure
        function directly (advisory-only in all modes, never blocking
        startup -- the S145 precedent).
        """
        return compute_ollama_limits_advisory(self._config)

    def _pressure_from_snapshot(
        self, snapshot: ResourceSnapshot
    ) -> dict[str, Any]:
        cfg = self._config
        effective = snapshot.capacity_gb
        ratio: float | None = None
        ratio_level = "none"
        if effective is not None and effective > 0:
            ratio = snapshot.vram_in_use_gb / effective
            if ratio >= cfg.pressure_hard_threshold:
                ratio_level = "hard"
            elif ratio >= cfg.pressure_soft_threshold:
                ratio_level = "soft"
        refusal_rate, refusals, decisions = self._refusal_window_stats()
        refusal_level = "none"
        if (
            decisions >= _REFUSAL_RATE_MIN_DECISIONS
            and refusal_rate >= _REFUSAL_RATE_SOFT
        ):
            refusal_level = "soft"
        order = {"none": 0, "soft": 1, "hard": 2}
        level = (
            ratio_level
            if order[ratio_level] >= order[refusal_level]
            else refusal_level
        )
        if level != self._last_pressure_level:
            # Debounced by construction: logged on level change only.
            logger.info(
                "Resource pressure level %s -> %s (ratio=%s,"
                " refusal_rate=%.2f over %d decision(s))",
                self._last_pressure_level or "unset",
                level,
                f"{ratio:.2f}" if ratio is not None else "n/a",
                refusal_rate,
                decisions,
            )
            self._last_pressure_level = level
        self._apply_pressure_policy(level)
        return {
            "level": level,
            "ratio": round(ratio, 4) if ratio is not None else None,
            "effective_capacity_gb": effective,
            "in_use_gb": round(snapshot.vram_in_use_gb, 3),
            "soft_threshold": cfg.pressure_soft_threshold,
            "hard_threshold": cfg.pressure_hard_threshold,
            "refusal_rate": round(refusal_rate, 4),
            "refusals_in_window": refusals,
            "decisions_in_window": decisions,
            "refusal_window_s": cfg.pressure_refusal_window_s,
            "keep_alive_overridden": self._keep_alive_original is not None,
        }

    def _refusal_window_stats(self) -> tuple[float, int, int]:
        """(rate, refusals, decisions) over the bounded config window.

        The window is pruned on read (governor clock, fake-clock
        testable). Estop refusals never enter the deque; the disabled
        passthrough is unrecorded and therefore never counted.
        """
        now = self._clock()
        window = max(0.0, self._config.pressure_refusal_window_s)
        with self._cache_lock:
            while self._refusal_events and (
                self._refusal_events[0][0] < now - window
            ):
                self._refusal_events.popleft()
            events = list(self._refusal_events)
        decisions = len(events)
        refusals = sum(1 for _, refused in events if refused)
        rate = (refusals / decisions) if decisions else 0.0
        return rate, refusals, decisions

    def _apply_pressure_policy(self, level: str) -> None:
        """Section 5 step 1, the sustained half: under soft-or-worse
        pressure persisting for pressure_sustain_s, write the warmup's
        keep_alive ONCE through its existing settable property
        (remembering the original read back just before); restore the
        remembered original at the first clear observation. One-way by
        construction: the warmup is never modified to know the
        governor and its keepalive thread is never touched. The
        remembered original does not survive a process restart (the
        warmup re-initialises at its own default). Never raises.
        """
        now = self._clock()
        cfg = self._config
        with self._cache_lock:
            if level in ("soft", "hard"):
                if self._pressure_soft_since is None:
                    self._pressure_soft_since = now
                    return
                sustained = (
                    now - self._pressure_soft_since
                    >= max(0.0, cfg.pressure_sustain_s)
                )
                if not sustained or self._keep_alive_original is not None:
                    return
                warmup = self._warmup
                if warmup is None:
                    return
                try:
                    original = getattr(warmup, "keep_alive", None)
                    if (
                        isinstance(original, str)
                        and original
                        and original != cfg.pressure_keep_alive
                    ):
                        warmup.keep_alive = cfg.pressure_keep_alive
                        self._keep_alive_original = original
                        logger.info(
                            "Sustained pressure: warmup keep_alive %s -> %s"
                            " (restored when pressure clears)",
                            original,
                            cfg.pressure_keep_alive,
                        )
                except Exception as exc:
                    logger.debug(
                        "Pressure keep_alive write failed open: %s", exc
                    )
                return
            # Level none: clear the sustain timer and restore once.
            self._pressure_soft_since = None
            if self._keep_alive_original is None:
                return
            original = self._keep_alive_original
            self._keep_alive_original = None
            warmup = self._warmup
            if warmup is None:
                return
            try:
                warmup.keep_alive = original
                logger.info(
                    "Pressure cleared: warmup keep_alive restored to %s",
                    original,
                )
            except Exception as exc:
                logger.debug(
                    "Pressure keep_alive restore failed open: %s", exc
                )

    # -- Bloc 1: the admission gate (Section 4) -------------------------------

    def admit(
        self,
        model: str,
        requested_ctx: int | None = None,
        caller: str = "chat",
        extra_models: list[str] | None = None,
        digest: str | None = None,
    ) -> AdmissionDecision:
        """R-01: does (model, requested num_ctx) fit the machine right now?

        Contract (spec Sections 4.2-4.5, implemented verbatim):

        - R-04 first: the emergency-stop flag is honoured BEFORE any fit
          math, through the existing seams only (is_stopped(), the refusal
          built from refusal_payload()); the flag transition observed here
          is the drain/resume invalidation wiring.
        - Disabled by config: an honest, deliberately UNRECORDED
          passthrough admit (a disabled governor decides nothing).
        - cost = weights + kv(ctx): zero weight cost when the model is
          already resident in the cached S1 view (admission reduces to the
          ctx check); an unknown estimate is never "too large" (3.1);
          ``extra_models`` folds companion weight costs into the same
          decision (the speculative draft+verify pair, Section 8).
        - budget = capacity - in_use + evictable_now - safety_margin,
          where evictable_now sums loaded models idle past the config
          threshold (derived from the snapshot's expirations); a fit
          reached only through evictable_now is granted CONDITIONAL on
          eviction (the eviction act itself is Bloc 2; Ollama's own LRU
          carries it meanwhile, the Section 12 posture).
        - The requested ctx is clamped to the model's context window
          (ModelLimits stays the authority), then stepped down the config
          ladder to the per-caller floor; callers without a floor
          (benchmark, AGT, direct) are never downsized.
        - Capacity unknown: the VRAM half fails open (admit, the 3.1
          arbitration) while the RAM half still applies (known weight cost
          exceeding MemAvailable refuses).
        - Every decision is recorded in the ring; num_gpu stays None
          (conservative); under soft-or-worse pressure (Section 5) an
          admitted decision carries the keep_alive override the funnels
          apply for that call.
        """
        ticket_id = uuid.uuid4().hex[:12]

        # 4.5 / R-04: the stopped flag comes before everything else.
        estop = _resolve_emergency_stop()
        stopped = False
        if estop is not None:
            try:
                stopped = bool(estop.is_stopped())
            except Exception as exc:
                logger.debug("Estop flag read failed open: %s", exc)
        self._observe_estop_transition(stopped)
        if stopped:
            payload: dict[str, Any] = {
                "error": "emergency_stopped",
                "message": (
                    "Emergency stop is engaged: new work is refused until"
                    " resume."
                ),
            }
            try:
                payload = dict(estop.refusal_payload())
            except Exception as exc:
                logger.debug("Estop refusal payload read failed: %s", exc)
            decision = AdmissionDecision(
                admitted=False,
                model=model,
                num_ctx=None,
                action="refuse",
                reason="emergency_stopped",
                ticket_id=ticket_id,
                caller=caller,
                requested_ctx=requested_ctx,
                is_estop=True,
                payload=payload,
            )
            self._record_admission(decision)
            return decision

        if not self._config.enabled:
            return AdmissionDecision(
                admitted=True,
                model=model,
                num_ctx=None,
                action="admit",
                reason="governor_disabled",
                ticket_id=ticket_id,
                caller=caller,
                requested_ctx=requested_ctx,
            )

        snapshot = self.get_snapshot_fast()
        provenance = list(snapshot.sources)

        # S225 (Bloc 2): the pressure signal rides every admission; a
        # soft-or-worse level fills the decision's keep_alive override
        # (Section 5, escalation step 1) when capacity is known. The
        # same read drives the sustained-write/restore policy.
        pressure = self._pressure_from_snapshot(snapshot)
        ka_override: str | None = (
            self._config.pressure_keep_alive
            if (
                pressure.get("level") in ("soft", "hard")
                and snapshot.capacity_gb is not None
            )
            else None
        )

        loaded_names = {
            v.name for v in snapshot.loaded if v.size_vram_bytes > 0
        }
        already_loaded = model in loaded_names
        load_expected = not already_loaded

        # Weights cost (4.2): zero when already resident; unknown is never
        # "too large" (3.1) and contributes zero to the fit.
        if already_loaded:
            weights_gb: float | None = 0.0
        else:
            weights_gb, _basis = self.estimate_model_vram_gb(model, digest)
            # S261: an operator-named weights-residency override (the
            # MoE active-params gap) replaces the estimate; absent, the
            # estimator answer above stands -- today's pricing.
            weights_override = self.resolve_weights_override(model)
            if weights_override is not None:
                weights_gb = weights_override

        extra_gb = 0.0
        for extra in extra_models or []:
            if not extra or extra == model or extra in loaded_names:
                continue
            load_expected = True
            extra_est, _eb = self.estimate_model_vram_gb(extra)
            # S261: the same override seam for the extra models.
            extra_override = self.resolve_weights_override(extra)
            if extra_override is not None:
                extra_est = extra_override
            if extra_est is not None:
                extra_gb += extra_est

        effective_ctx = self._clamp_ctx(model, requested_ctx)
        known_weights = (0.0 if weights_gb is None else weights_gb) + extra_gb

        def _cost(ctx: int | None) -> float:
            return known_weights + self.estimate_kv_cache_gb(ctx, model=model)

        if snapshot.capacity_gb is None:
            # 3.1: the VRAM half fails open; the RAM half still applies.
            ram_mb = snapshot.ram_available_mb
            if ram_mb > 0.0 and known_weights * 1024.0 > ram_mb:
                decision = AdmissionDecision(
                    admitted=False,
                    model=model,
                    num_ctx=None,
                    action="refuse",
                    reason="ram_insufficient",
                    provenance=provenance,
                    ticket_id=ticket_id,
                    caller=caller,
                    requested_ctx=requested_ctx,
                    shortfall_gb=round(known_weights - ram_mb / 1024.0, 3),
                )
            else:
                decision = AdmissionDecision(
                    admitted=True,
                    model=model,
                    num_ctx=effective_ctx,
                    action="admit",
                    reason="capacity_unknown_fail_open",
                    provenance=provenance,
                    ticket_id=ticket_id,
                    caller=caller,
                    requested_ctx=requested_ctx,
                    load_expected=load_expected,
                )
            self._record_admission(decision)
            return decision

        in_use = snapshot.vram_in_use_gb
        margin = self._config.safety_margin_gb
        evictable = self._evictable_now_gb(snapshot)
        budget_unconditional = snapshot.capacity_gb - in_use - margin
        budget_with_eviction = budget_unconditional + evictable

        def _fit(ctx: int | None) -> bool | None:
            """True: fits now. False: fits only after eviction. None: no."""
            cost = _cost(ctx)
            if cost <= budget_unconditional:
                return True
            if evictable > 0.0 and cost <= budget_with_eviction:
                return False
            return None

        candidates: list[int | None] = [effective_ctx]
        floor = self._config.ctx_floor.get(caller)
        if effective_ctx is not None and floor is not None:
            steps = sorted(
                {
                    int(s)
                    for s in self._config.ctx_ladder
                    if floor <= int(s) < effective_ctx
                },
                reverse=True,
            )
            candidates.extend(steps)

        for index, ctx in enumerate(candidates):
            verdict = _fit(ctx)
            if verdict is None:
                continue
            conditional = verdict is False
            action = "downsize" if index > 0 else "admit"
            reason_parts = []
            if action == "downsize":
                reason_parts.append("ctx_laddered_to_fit")
            elif (
                effective_ctx is not None
                and requested_ctx is not None
                and effective_ctx < requested_ctx
            ):
                reason_parts.append("clamped_to_model_limit")
            if conditional:
                reason_parts.append("conditional_on_eviction")
            if not reason_parts:
                reason_parts.append("fits")
            decision = AdmissionDecision(
                admitted=True,
                model=model,
                num_ctx=ctx,
                keep_alive=ka_override,
                action=action,
                reason="+".join(reason_parts),
                provenance=provenance,
                ticket_id=ticket_id,
                caller=caller,
                requested_ctx=requested_ctx,
                load_expected=load_expected,
                conditional_on_eviction=conditional,
            )
            self._record_admission(decision)
            return decision

        minimal_cost = _cost(candidates[-1])
        decision = AdmissionDecision(
            admitted=False,
            model=model,
            num_ctx=None,
            action="refuse",
            reason="vram_insufficient",
            provenance=provenance,
            ticket_id=ticket_id,
            caller=caller,
            requested_ctx=requested_ctx,
            shortfall_gb=round(
                max(0.0, minimal_cost - budget_with_eviction), 3
            ),
        )
        self._record_admission(decision)
        return decision

    def _observe_estop_transition(self, stopped: bool) -> None:
        """R-04 invalidation wiring without editing emergency_stop:

        the admission path observes the flag; a False->True transition is
        the drain (invalidate_on_estop_drain), True->False the resume
        (invalidate_on_resume). The first observation only seeds state.
        """
        with self._cache_lock:
            previous = self._last_estop_seen
            self._last_estop_seen = stopped
        if previous is None or previous == stopped:
            return
        if stopped:
            self.invalidate_on_estop_drain()
        else:
            self.invalidate_on_resume()

    def _clamp_ctx(
        self, model: str, requested_ctx: int | None
    ) -> int | None:
        """Clamp the requested context to the model's window (spec 4.2).

        ModelLimits stays the authority: the window is read through
        context_manager.get_model_limits, resolved lazily and fail-open
        (no clamp when the seam is unavailable or errors).
        """
        if requested_ctx is None or requested_ctx <= 0:
            return None
        cm = _resolve_context_manager()
        if cm is None:
            return int(requested_ctx)
        try:
            limits = cm.get_model_limits(model)
            window = int(getattr(limits, "context_window", 0) or 0)
        except Exception as exc:
            logger.debug("ModelLimits clamp unavailable: %s", exc)
            return int(requested_ctx)
        if window > 0:
            return min(int(requested_ctx), window)
        return int(requested_ctx)

    def _evictable_now_gb(self, snapshot: ResourceSnapshot) -> float:
        """Summed size_vram of loaded models idle past the threshold (4.2).

        Delegates to _evictable_candidates (S225) so the fit math and
        the targeted eviction read the SAME definition of evictable.
        """
        return sum(
            size for _name, _idle, size in self._evictable_candidates(snapshot)
        )

    def _evictable_candidates(
        self, snapshot: ResourceSnapshot
    ) -> list[tuple[str, float, float]]:
        """(name, idle_s, size_gb) for every loaded model idle past the
        threshold, sorted oldest-idle FIRST (the Section 5 eviction
        order).

        Idle time is derived from the snapshot's expirations: a model's
        ``expires_at`` is last activity plus the warmup keep_alive, so
        idle = keep_alive_s - (expires_at - now), compared on the WALL
        clock (expirations are wall-clock stamps; the snapshot's monotonic
        ``taken_at`` is deliberately not used). Anything that cannot be
        coerced or computed counts as NOT evictable (conservative).
        """
        threshold = self._config.idle_evict_threshold_s
        if threshold is None or threshold < 0:
            return []
        keep_alive_s = _parse_duration_s(
            getattr(self._warmup, "keep_alive", None)
        )
        if keep_alive_s is None:
            return []
        now = time.time()
        candidates: list[tuple[str, float, float]] = []
        for view in snapshot.loaded:
            if view.size_vram_bytes <= 0:
                continue
            expires = _coerce_epoch_s(view.expires_at)
            if expires is None:
                continue
            idle_s = keep_alive_s - (expires - now)
            if idle_s >= threshold:
                candidates.append((view.name, idle_s, view.size_vram_gb))
        candidates.sort(key=lambda c: c[1], reverse=True)
        return candidates

    # -- Bloc 2: targeted eviction (Section 5, honouring conditional grants) --

    def evict_model(
        self,
        model: str,
        trigger: str = "manual",
        ticket_id: str | None = None,
        needed_gb: float | None = None,
    ) -> bool:
        """Per-model eviction through the backends' narrowed primitives.

        Duck-typed ``unload_model(name)`` on every registered backend
        (the Ollama generate(keep_alive=0) idiom narrowed to one model;
        LlamaCppBackend's existing method). A success invalidates the
        snapshot (invalidate_on_evict) and appends to the signed audit
        chain OFF the hot path. Every failure path is fail-open: a
        False return means Ollama's own LRU carries the pressure (the
        Section 12 posture). This is also the surface the Bloc 4
        POST /api/governor/evict will call.
        """
        registry = self._resolve_registry()
        if registry is None:
            return False
        try:
            backends = list(registry.backends())
        except Exception as exc:
            logger.debug("Registry backends read failed: %s", exc)
            return False
        for backend in backends:
            unload = getattr(backend, "unload_model", None)
            if not callable(unload):
                continue
            try:
                if unload(model):
                    self.invalidate_on_evict(model)
                    self._audit_eviction_async(
                        model, trigger, ticket_id, needed_gb
                    )
                    return True
            except Exception as exc:
                logger.debug(
                    "unload_model(%s) failed on %s: %s",
                    model,
                    getattr(backend, "name", "backend"),
                    exc,
                )
        return False

    def _honour_conditional_eviction(
        self, decision: AdmissionDecision
    ) -> None:
        """Act on a conditional-on-eviction grant just before its load.

        Recomputes the shortfall against the current cached view (the
        snapshot has typically moved since the grant), walks the
        idle-past-threshold candidates oldest-idle FIRST and evicts
        ONLY as many as the shortfall needs. Every path fails open:
        any miss or error leaves the admitted call untouched and
        Ollama's own LRU carries it (Section 12). Never raises.
        """
        try:
            snapshot = self.get_snapshot_fast()
            if snapshot.capacity_gb is None:
                return
            loaded_names = {
                v.name for v in snapshot.loaded if v.size_vram_bytes > 0
            }
            weights = 0.0
            if decision.model not in loaded_names:
                est, _basis = self.estimate_model_vram_gb(decision.model)
                if est is not None:
                    weights = est
            cost = weights + self.estimate_kv_cache_gb(decision.num_ctx)
            budget = (
                snapshot.capacity_gb
                - snapshot.vram_in_use_gb
                - self._config.safety_margin_gb
            )
            needed = cost - budget
            if needed <= 0.0:
                return
            freed = 0.0
            for name, _idle, size_gb in self._evictable_candidates(snapshot):
                if name == decision.model:
                    continue
                if self.evict_model(
                    name,
                    trigger="conditional_admission",
                    ticket_id=decision.ticket_id,
                    needed_gb=round(needed, 3),
                ):
                    freed += size_gb
                    if freed >= needed:
                        break
        except Exception as exc:
            logger.debug(
                "Conditional eviction honour failed open: %s", exc
            )

    def _audit_eviction_async(
        self,
        model: str,
        trigger: str,
        ticket_id: str | None,
        needed_gb: float | None,
    ) -> None:
        """Append the eviction to the signed audit chain OFF the hot
        path (a short-lived daemon thread; evictions are rare). The
        established chain_log idiom (the emergency_stop _chain
        precedent): lazy import, never raises, best-effort.
        """

        def _append() -> None:
            try:
                from opti_oignon.signed_audit_log import chain_log

                chain_log(
                    event_type="resource_governor",
                    source="resource_governor",
                    action="evict_model",
                    severity="INFO",
                    model=model,
                    trigger=trigger,
                    ticket_id=ticket_id,
                    needed_gb=needed_gb,
                )
            except Exception as exc:
                logger.debug("Eviction audit append failed: %s", exc)

        try:
            threading.Thread(
                target=_append,
                name="governor-evict-audit",
                daemon=True,
            ).start()
        except Exception as exc:
            logger.debug("Eviction audit thread failed: %s", exc)

    # -- Bloc 2: the bounded opt-in queue (Section 5) -------------------------

    def admit_or_wait(
        self,
        model: str,
        requested_ctx: int | None = None,
        caller: str = "benchmark",
        extra_models: list[str] | None = None,
        digest: str | None = None,
    ) -> AdmissionDecision:
        """admit() with the Section 5 bounded queue for enrolled callers.

        The escalation contract verbatim: a caller NOT enrolled in
        queue.enabled_per_caller (the shipped default enrolls nobody)
        gets plain admit() semantics -- chat and pipeline additionally
        never call this entry, so the interactive path stays
        refuse-by-default at the call site (D3). An enrolled caller
        whose admission was refused waits, bounded in depth and wait:
        beyond either bound the entry resolves to the caller's existing
        refusal semantics (the refusal decision is returned). Every
        wake re-runs admission -- the estop is re-honoured FIRST by
        construction, so a drain releases waiters to refusal (the
        drain's invalidation notify wakes them immediately) and no
        queued entry can outlive it. Waiters hold no lock while
        waiting; the wait is sliced so an injected fake clock drives
        the deadline in container tests.
        """
        decision = self.admit(
            model,
            requested_ctx,
            caller=caller,
            extra_models=extra_models,
            digest=digest,
        )
        if decision.admitted or decision.is_estop:
            return decision
        if not self._config.queue_enabled_per_caller.get(caller, False):
            return decision
        with self._queue_cond:
            if self._queue_depth >= max(0, self._config.queue_depth):
                logger.debug(
                    "Queue depth bound reached; %s refusal stands for %s",
                    caller,
                    model,
                )
                return decision
            self._queue_depth += 1
        try:
            try:
                # Ring visibility of the enqueue (the 4.4 "queue" action).
                self.record_decision(
                    caller, model, requested_ctx, None, "queue", "enqueued"
                )
            except Exception as exc:
                logger.debug("Queue ring write failed: %s", exc)
            deadline = self._clock() + max(0.0, self._config.queue_wait_s)
            last = decision
            while True:
                remaining = deadline - self._clock()
                if remaining <= 0.0:
                    return last
                with self._queue_cond:
                    self._queue_cond.wait(
                        timeout=min(remaining, _QUEUE_WAIT_SLICE_S)
                    )
                last = self.admit(
                    model,
                    requested_ctx,
                    caller=caller,
                    extra_models=extra_models,
                    digest=digest,
                )
                if last.admitted or last.is_estop:
                    return last
        finally:
            with self._queue_cond:
                self._queue_depth -= 1

    @property
    def queue_depth(self) -> int:
        """Current number of queued admissions (the Bloc 4 status field)."""
        with self._queue_cond:
            return self._queue_depth

    def _record_admission(self, decision: AdmissionDecision) -> None:
        """Ring write for every recorded decision (4.4); never raises.

        S225: the same seam feeds the in-memory refusal-rate window
        (resource decisions only -- an estop refusal is not a resource
        signal and never enters it).
        """
        if not decision.is_estop:
            try:
                with self._cache_lock:
                    self._refusal_events.append(
                        (self._clock(), not decision.admitted)
                    )
            except Exception as exc:
                logger.debug("Refusal window append failed: %s", exc)
        try:
            self.record_decision(
                decision.caller,
                decision.model,
                decision.requested_ctx,
                decision.num_ctx,
                decision.action,
                decision.reason,
            )
        except Exception as exc:
            logger.debug("Decision ring write failed: %s", exc)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_governor: ResourceGovernor | None = None
_governor_lock = threading.Lock()


def get_resource_governor(
    config_path: str | Path | None = None,
    db_path: str | Path | None = None,
) -> ResourceGovernor:
    """Return the module-level governor, creating it on first use."""
    global _governor
    if _governor is not None:
        return _governor
    with _governor_lock:
        if _governor is None:
            _governor = ResourceGovernor(config_path=config_path, db_path=db_path)
        return _governor


def reset_resource_governor() -> None:
    """Test hook: drop the module-level singleton."""
    global _governor
    with _governor_lock:
        _governor = None


# ---------------------------------------------------------------------------
# Bloc 1: the mechanical-seam gate (4.1/4.4, consumed by inference_backend)
# ---------------------------------------------------------------------------


def backend_admission_gate(
    model: str, options: dict | None = None
) -> None:
    """The internal hook body behind the four generate/stream heads.

    A matching ticket means the funnel already decided: account the load
    (the invalidate_on_load wiring, once per ticket) and stand down. A
    ticketless call gets the fast cached admit-or-refuse backstop with
    default semantics (caller "direct", the mechanical backstop for the
    Section 8 residual), raising the typed GovernorRefusal on a positive
    refusal only. The caller (inference_backend) wraps this in its own
    fail-open handling; a disabled governor stands down entirely.
    """
    governor = get_resource_governor()
    if not governor.config.enabled:
        return
    ticket = get_active_ticket()
    if ticket is not None and ticket.model == model:
        if ticket.admitted and ticket.load_expected:
            # S225: act on a conditional grant just before its load
            # (oldest-idle first, only as much as the shortfall needs;
            # fail-open to Ollama's own LRU, Section 12).
            if ticket.conditional_on_eviction:
                governor._honour_conditional_eviction(ticket)
            governor.invalidate_on_load(model, ticket.num_ctx)
            ticket.load_expected = False
        return
    requested: int | None = None
    if isinstance(options, dict):
        raw = options.get("num_ctx")
        if isinstance(raw, int) and raw > 0:
            requested = raw
    decision = governor.admit(model, requested, caller="direct")
    if not decision.admitted:
        raise GovernorRefusal(decision)
    if decision.load_expected:
        if decision.conditional_on_eviction:
            governor._honour_conditional_eviction(decision)
        governor.invalidate_on_load(model, decision.num_ctx)
