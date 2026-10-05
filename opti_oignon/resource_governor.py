#!/usr/bin/env python3
"""Resource Governor: measurement, admission, backpressure, limits.

Measurement: a cached
ResourceSnapshot assembled from ranked, individually-optional sources with
honest provenance; the measure-and-adapt store; the YAML
config loader. The admission layer adds the gate on
top: admit() and the AdmissionDecision ticket (Section 4.4), the 4.2 fit
math on the cached snapshot (get_snapshot_fast), the per-caller ctx ladder
and floors (4.3, benchmark/AGT never downsized), the R-04 emergency-stop
honour (4.5: the flag checked FIRST through the existing seams only, the
refusal built from refusal_payload()), the thread-local ticket pass-through
(ticket_scope, the gate arbitration) and the mechanical backend gate
(backend_admission_gate) consumed by the four generate/stream heads.
The second layer adds runtime backpressure (Section 5, the escalation order
verbatim): the pressure signal (in_use over effective capacity against the
config soft/hard thresholds, plus a bounded refusal-rate window), the
per-decision keep_alive override under soft pressure with the
sustained-write-then-restore discipline on the warmup's existing settable
property (one-way: warmup never knows the governor; its keepalive thread
is never stopped here), the targeted per-model eviction honouring
conditional-on-eviction admissions (audit-chained off the hot path,
degrading to Ollama's own LRU on any failure, the Section 12 posture),
and the bounded opt-in queue (per-caller enrollment, depth and wait
bounds, re-admission on wake, the estop never bypassed). The third layer
adds limit management (Section 6, R-03): the pure spawn-path env
construction (build_ollama_spawn_env, the contract for any future
Ollama spawner -- none exists in-app today), the external-Ollama
advisory (compute_ollama_limits_advisory and the governor method,
consumed by the startup security checklist through the standing
advisory-only precedent: never blocking startup in any mode), and the
optional, off-by-default process-wide rlimits applier
(apply_llamacpp_rlimits, consumed by the llama.cpp load seam BEFORE the
first in-process load). The API/frontend
surfaces are implemented separately (routes_governor.py, GovernorPanel.svelte).

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
  is not moved and not duplicated; the contracts on its home module keep
  holding), plus the KV-cache increment implemented HERE as a function of
  the requested num_ctx (the config-tunable ``kv_coefficient``).
- S4: capacity and host memory -- the VRAM capacity is the CONFIGURED
  value (``total_vram_gb``) when one is written down; null asks the
  machine: the capacity is then the sum of the cards the hardware profile
  selects (hardware_profile.py: every NVIDIA card nvidia-smi lists, AMD
  and Intel cards from DRM sysfs). On those cards the memory other
  programs hold -- the cards' used memory less what the engines declare --
  is deducted, and the safety margin is kept on each card. A learned
  ceiling, when one is recorded, lowers the capacity (Section 3.2). Host
  RAM comes from /proc/meminfo (MemTotal and MemAvailable; 0.0 is
  unknown, never a guess). That reader is a deliberate LOCAL EQUIVALENT of
  the profile's: this module stays loadable on its own, in a window
  without the profile, and a contract holds the two readers to the same
  answers. The kernel's pressure stall information, read through the
  profile, sizes the RAM a split leaves to the rest of the machine.

Design decisions (arbitrated):

- DI-2: the default DB path follows the benchmark ResultsStore precedent
  (``opti_oignon/data/resource_governor.db``); ``db_path`` injectable.
- DI-4: ``kv_coefficient`` is GiB of KV cache per 1024 tokens of the
  requested num_ctx (layers folded into one conservative, deliberately
  high-side coefficient; refined later by measure-and-adapt).
- DI-5: the TTL cache exposes refresh() (synchronous build),
  get_snapshot() (stale -> synchronous refresh) and get_snapshot_fast()
  (returns the cached snapshot even when stale and triggers a single
  background refresh -- the primitive the admission fast path
  consumes; the current decision uses the cached values conservatively).
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
  invalidate. All four are now wired to callers: admission (load),
  eviction, and estop observation (drain and resume).

Conservative defaults and fail-open (Section 3.1): capacity unknown
(configured null, no card the profile can read, AND no learned ceiling)
-> the VRAM half reports
vram_status="disabled_capacity_unknown" with a logged warning and the RAM
half still applies; an unknown model is never treated as too large; a
source erroring is the same as a source absent (log at debug, degrade to
the next source, never raise into the request path). No audit-chain append
happens anywhere in this module (the chain is reserved for evictions,
config changes and ceiling-learning surfacing, all outside the
measurement layer and off the hot path).

Kerckhoffs: nothing here is secret; the measurement chain, the learning
rules and the config surface are fully described. The store holds derived,
regenerable state only (ATREST disposition: single-user, pending-scoping,
backup excluded).
"""

from __future__ import annotations

import itertools
import logging
import math
import os
import re
import stat
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

# The warmup seam and the VRAM estimator are Forge features the governor
# builds in its constructor; they are imported there, so that importing the
# governor costs neither of them (the core-boundary guard holds this).

try:
    from opti_oignon.inference_backend import (
        get_backend_registry as _get_backend_registry,
    )

    INFERENCE_BACKEND_AVAILABLE = True
except Exception:
    _get_backend_registry = None
    INFERENCE_BACKEND_AVAILABLE = False


# ---------------------------------------------------------------------------
# Paths and module constants
# ---------------------------------------------------------------------------

def _load_default_warmup():
    """The warmup seam, imported when a governor is built; None when absent."""
    try:
        from opti_oignon.model_warmup import model_warmup
    except Exception:  # noqa: BLE001 - absence is the documented fallback
        return None
    return model_warmup


def _load_vram_estimator():
    """The VRAM estimator, imported when a governor is built; None when absent.

    Reuse BY IMPORT: estimate_model_vram() reads the _VRAM_PER_BILLION_PARAMS
    table in its home module. The table is not moved and not duplicated here.
    """
    try:
        from opti_oignon.speculative_decoding import VRAMBudgetCalculator
    except Exception:  # noqa: BLE001 - absence is the documented fallback
        return None
    return VRAMBudgetCalculator()


_CONFIG_DIR = Path(__file__).parent / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "resource_governor.yaml"
_DATA_DIR = Path(__file__).parent / "data"
_DEFAULT_DB_PATH = _DATA_DIR / "resource_governor.db"

# Ceiling-learning relax knobs (DI-8): module constants, not config keys,
# so the Section 10 config contract is not extended without a spec touch.
_CEILING_RELAX_AFTER_SUCCESSES = 5
_CEILING_RELAX_STEP_GB = 1.0

# Backpressure constants. The queue waits in bounded real-time slices so
# a fake injected clock can drive deadline math in container tests while
# notify-based wakes stay immediate. The refusal-rate rule raises the
# pressure level to AT LEAST soft when at least
# _REFUSAL_RATE_MIN_DECISIONS recorded decisions fall inside the config
# window and the refused fraction reaches _REFUSAL_RATE_SOFT; the rate
# alone never reaches hard.
_QUEUE_WAIT_SLICE_S = 0.5
_REFUSAL_RATE_SOFT = 0.5
_REFUSAL_RATE_MIN_DECISIONS = 3

# Who asks: the admission classes, highest first. Interactive: a person is
# waiting on the answer. User: asked by a person, not watched. Background:
# asked by nobody. A caller the configuration does not name is a user, never
# the background.
ADMISSION_CLASSES = ("interactive", "user", "background")
_CLASS_RANK = {name: rank for rank, name in enumerate(ADMISSION_CLASSES)}
_INTERACTIVE = "interactive"
_DEFAULT_CLASS = "user"
_BACKGROUND = "background"
# The refusals no wait can lift: a caller refused by one of them is answered
# at once, never queued. The card unreadable, the cost unknown, or no context
# to price stay so however long the caller waits.
_FINAL_REFUSALS = frozenset(
    {"background_capacity_unknown", "background_cost_unknown", "background_ctx_unknown"}
)
# How many times its size a file's parse takes in memory, by extension, with
# a "default" for any other: the shipped file's threads.background block
# holds the same table. Estimates, not measurements: owed to the machine.
_PARSE_EXPANSION = {"default": 4.0, ".pdf": 10.0, ".docx": 30.0, ".doc": 30.0, ".xlsx": 50.0, ".xls": 50.0}
_EXTENSION = re.compile(r"\.[A-Za-z0-9]+")
_DEFAULT_CALLER_CLASSES = {
    "chat": "interactive",
    "pipeline": "interactive",
    "benchmark": "user",
    "agent_eval": "user",
    "direct": "user",
    "warmup": "background",
    "index": "background",
    "tuner": "background",
    "reverie": "background",
}

_BYTES_PER_GIB = 1024.0 ** 3

# The two orders a split admission can follow (the offload.prefer key).
_OFFLOAD_PREFER = ("context", "speed")

# The natures of model an engine may declare (InferenceBackend.cost_model):
# a token generator keeps a KV cache that grows with the context; an encoder
# or a predictor need not.
_MODEL_KINDS = ("generator", "encoder", "predictor")

# Bytes per cached key or value element: f16, the KV cache type the engines
# use unless told otherwise. A quantized cache holds less, so pricing every
# model at f16 errs on the side of the larger cost.
_KV_BYTES_PER_ELEMENT = 2

# Sentinel distinguishing "not passed" from an explicit None injection.
_UNSET: Any = object()

_SIZE_STR_RE = re.compile(r"^([\d.]+)\s*(GB|MB|B)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _read_meminfo_mb(meminfo_path: str | Path = "/proc/meminfo") -> tuple[float, float]:
    """(MemAvailable, MemTotal) in MiB; a field that cannot be read is 0.0.

    The local equivalent of hardware_profile.read_meminfo, kept here so the
    governor loads on its own (a contract holds the two to the same answers
    on the same files). A 0.0 MemAvailable means the RAM half of the
    snapshot is unknown and must never exclude anything; a 0.0 MemTotal
    leaves the adaptive reserve at its ceiling.
    """
    fields = {"MemTotal:": 0.0, "MemAvailable:": 0.0}
    try:
        text = Path(meminfo_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return 0.0, 0.0
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in fields:
            try:
                value = float(parts[1])
            except ValueError:
                continue
            if math.isfinite(value) and value >= 0.0:
                fields[parts[0]] = value / 1024.0  # kB -> MiB
    return fields["MemAvailable:"], fields["MemTotal:"]


def _parse_parameter_size_b(value: Any) -> float:
    """Parse a parameter-size label ("7B", "3.2b", 7.0) to billions.

    Returns 0.0 when missing or unparseable (fail-open: an unknown size is
    never treated as too large -- the idiom restated by Section 3.1).
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


def _positive_int(value: Any) -> int | None:
    """A strictly positive integer, or None; a boolean is not a count."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _kv_geometry_from_metadata(meta: Any) -> dict[str, Any] | None:
    """The KV cache geometry a model's metadata names, or None.

    ``meta`` is the key/value mapping Ollama's model information reports or a
    GGUF header holds; both use the GGUF key names. Bytes per token = the KV
    heads summed over the layers x (key length + value length) x the element
    size. An unreported key or value length is the embedding width over the
    attention heads, a model that names no KV head count keeps one per
    attention head, and a per-layer KV head list is summed. Anything else
    missing or malformed answers None, and the flat coefficient stands.
    ``kv_bytes_per_layer`` holds each layer's share of the bytes per token,
    in layer order: a split carries each layer's KV with it.
    """
    if not isinstance(meta, Mapping):
        return None
    arch = meta.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        return None
    layers = _positive_int(meta.get(f"{arch}.block_count"))
    if layers is None:
        return None
    heads = meta.get(f"{arch}.attention.head_count")
    kv_heads = meta.get(f"{arch}.attention.head_count_kv", heads)
    if isinstance(kv_heads, (list, tuple)):
        if len(kv_heads) != layers or any(
            isinstance(h, bool) or not isinstance(h, int) or h < 0 for h in kv_heads
        ):
            return None
        layer_heads = tuple(kv_heads)
    else:
        per_layer = _positive_int(kv_heads)
        if per_layer is None:
            return None
        layer_heads = (per_layer,) * layers
    total_kv_heads = sum(layer_heads)
    key_len = _positive_int(meta.get(f"{arch}.attention.key_length"))
    value_len = _positive_int(meta.get(f"{arch}.attention.value_length"))
    if key_len is None or value_len is None:
        width = _positive_int(meta.get(f"{arch}.embedding_length"))
        n_heads = _positive_int(heads)
        if width is None or n_heads is None or width % n_heads:
            return None
        key_len = key_len or width // n_heads
        value_len = value_len or width // n_heads
    per_head = (key_len + value_len) * _KV_BYTES_PER_ELEMENT
    per_token = total_kv_heads * per_head
    if per_token <= 0:
        return None
    return {
        "layers": layers,
        "kv_bytes_per_token": per_token,
        "kv_bytes_per_layer": tuple(h * per_head for h in layer_heads),
    }


def _kv_geometry_from_gguf(path: Any) -> dict[str, Any] | None:
    """The KV geometry a GGUF file's header names, or None.

    The header reader is the model manager's, imported only when a file is
    named, so importing the governor costs nothing; a file it cannot read
    holds no geometry.
    """
    if not path:
        return None
    try:
        from opti_oignon.model_manager import parse_gguf_header

        return _kv_geometry_from_metadata(parse_gguf_header(path).metadata)
    except Exception as exc:  # noqa: BLE001 - an unreadable header is no geometry
        logger.debug("GGUF header of %s unreadable: %s", path, exc)
        return None


# An Ollama blob: content addressed, named after its digest.
_OLLAMA_BLOB = re.compile(r"sha256-[0-9a-f]{64}")
# Tensor tables kept per governor, by file identity: a handful of models.
_TABLE_CACHE_SIZE = 16


def _model_files(info: Any) -> list[str]:
    """The model files an engine's model information names, in its order.

    The file the engine loads (``path``, as llama.cpp names it), then each
    blob the modelfile's FROM lines name (Ollama): an absolute path, already
    in normal form, to a file called ``sha256-<64 hex>`` in a ``blobs``
    directory. Any other path is not a model file and is never opened.
    """
    files: list[str] = []
    path = getattr(info, "path", None)
    if isinstance(path, str) and path:
        files.append(path)
    extra = getattr(info, "extra", None)
    modelfile = extra.get("modelfile") if isinstance(extra, dict) else None
    if not isinstance(modelfile, str):
        return files
    for line in modelfile.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or parts[0].upper() != "FROM":
            continue
        target = parts[1].strip()
        parent, name = os.path.split(target)
        if (
            os.path.isabs(target)
            and os.path.normpath(target) == target
            and os.path.basename(parent) == "blobs"
            and _OLLAMA_BLOB.fullmatch(name)
        ):
            files.append(target)
    return files


def _read_tensor_table(path: str) -> Any:
    """The tensor table of the GGUF file at ``path``, or None when refused.

    The reader is the model manager's, imported only when a split is priced,
    so importing the governor costs nothing.
    """
    try:
        from opti_oignon.model_manager import read_gguf_tensors

        return read_gguf_tensors(path)
    except Exception as exc:  # noqa: BLE001 - a table that cannot be read places nothing
        logger.debug("Tensor table of %s unreadable: %s", path, exc)
        return None


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
    per 1024 tokens}.

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
    weights-residency GiB}.

    The kv coercer mirrored as a sibling so the kv path stays
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


def _optional_above(
    section: Mapping,
    key: str,
    default: float | None,
    low: float,
    inclusive: bool,
) -> float | None:
    """``section[key]`` as a finite number above ``low``, or None; else ``default``.

    ``low`` itself is allowed when ``inclusive``. An absent key is the
    default, silently; a null is None; anything else out of range is the
    default with a warning naming it.
    """
    if key not in section:
        return default
    raw = section.get(key)
    if raw is None:
        return None
    try:
        value = None if isinstance(raw, bool) else float(raw)
    except (TypeError, ValueError):
        value = None
    if value is not None and math.isfinite(value) and (value >= low if inclusive else value > low):
        return value
    logger.warning(
        "split_speed.%s %r is not a number %s %s; keeping %s",
        key,
        raw,
        "at or above" if inclusive else "above",
        low,
        default,
    )
    return default


def _bounded_float(
    section: Mapping,
    key: str,
    default: float,
    low: float,
    high: float | None,
    section_name: str = "offload",
    low_open: bool = False,
) -> float:
    """``section[key]`` as a number within [low, high], else ``default``.

    No upper bound when ``high`` is None, but never infinite; ``low_open``
    excludes ``low`` itself. An absent key is the default, silently; a
    present one that is not a number, NaN, or out of range is the default
    with a warning naming it.
    """
    if key not in section:
        return default
    raw = section.get(key)
    try:
        value = None if isinstance(raw, bool) else float(raw)
    except (TypeError, ValueError):
        value = None
    if value is not None and (low < value if low_open else low <= value) and (
        value <= high if high is not None else value < float("inf")
    ):
        return value
    logger.warning(
        "%s.%s %r is not a number in %s%s, %s]; keeping %s",
        section_name,
        key,
        raw,
        "(" if low_open else "[",
        low,
        "inf" if high is None else high,
        default,
    )
    return default


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
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GovernorConfig:
    """Section 10 keys with spec defaults.

    The measurement path consumes the measurement subset (enabled, total_vram_gb,
    safety_margin_gb, snapshot_ttl_s, kv_coefficient, ceiling_floor_gb,
    decisions_ring_size); admission, backpressure, the queue, the limits and
    the offload policy consume the rest.
    ``enabled`` gates admission; measurement itself stays available
    regardless.
    """

    enabled: bool = True
    total_vram_gb: float | None = None
    safety_margin_gb: float = 1.5
    snapshot_ttl_s: float = 2.0
    # GiB of KV cache per 1024 tokens of requested num_ctx (DI-4).
    kv_coefficient: float = 0.5
    # Dynamic context quantum: when enabled, the admitted num_ctx lands
    # on the ctx_ladder (a between-steps request rounds UP so a growing
    # conversation keeps one stable quantum), capped by a live ceiling
    # derived from the VRAM left after weights through the KV
    # coefficient. Unknown capacity caps the quantum to the conservative
    # value below instead of passing the raw request through -- the
    # admission itself stays fail-open; only the quantum is fail-secure.
    dynamic_ctx_enabled: bool = False
    dynamic_ctx_unknown_ceiling: int = 8192
    # Per-model KV overrides (resolution: exact model name >
    # longest matching family substring > the global coefficient above,
    # which stays the fail-secure answer for any model neither table
    # names). Keys are normalised to lowercase at load.
    kv_override_models: dict[str, float] = field(default_factory=dict)
    kv_override_families: dict[str, float] = field(default_factory=dict)
    # Per-model weights-residency overrides in GiB (the MoE gap:
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
    # Partial offload: a model the GPU cannot hold, even after evicting idle
    # models, is admitted split between VRAM and system RAM when the two
    # together hold it. ``offload_prefer`` orders the attempts: "context"
    # splits at the requested context before stepping down the ladder,
    # "speed" steps down on the GPU alone first and splits last. A split
    # must leave at least ``offload_min_gpu_share`` of the cost on the GPU
    # (0.0 to 1.0) and never uses the reserve: the last
    # ``offload_ram_reserve_gb`` of available RAM when that is a number (0.0
    # or more), or, when it is None, a reserve sized from the machine --
    # ``ram_reserve_fraction`` of the total RAM, within the floor and the
    # ceiling, multiplied by ``ram_reserve_pressure_factor`` while the
    # kernel reports memory pressure (``host_pressure_*``: entered at or
    # above the enter mark, left below the exit mark). The shipped file
    # leaves it None; a file that does not name it keeps the fixed 4.0.
    offload_enabled: bool = True
    offload_prefer: str = "context"
    offload_min_gpu_share: float = 0.0
    offload_ram_reserve_gb: float | None = 4.0
    ram_reserve_fraction: float = 0.0625
    ram_reserve_floor_gb: float = 2.0
    ram_reserve_ceiling_gb: float = 8.0
    ram_reserve_pressure_factor: float = 2.0
    # The speed of a split placed layer by layer: the bandwidths, in GB/s,
    # at which the GPU (the slowest card a split may take) and the system RAM
    # are read, and the largest slowdown against the GPU alone a split may
    # bring. None is unknown; a speed that cannot be told never refuses.
    split_gpu_bandwidth_gbs: float | None = None
    split_ram_bandwidth_gbs: float | None = None
    split_max_slowdown: float | None = None
    host_pressure_enabled: bool = True
    host_pressure_memory_enter: float = 10.0
    host_pressure_memory_exit: float = 5.0
    # Admission's context shaping and the backpressure, queue and limits keys:
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
    # Backpressure: soft-or-worse pressure must persist this long before
    # the governor writes the warmup keep_alive through its settable
    # property; the original is restored at the first clear observation.
    pressure_sustain_s: float = 60.0
    # Backpressure: the bounded window the refusal-rate signal reads.
    pressure_refusal_window_s: float = 60.0
    queue_enabled_per_caller: dict[str, bool] = field(default_factory=dict)
    queue_depth: int = 2
    queue_wait_s: float = 30.0
    # Who asks: each caller's admission class (ADMISSION_CLASSES). Per
    # class: whether a refused caller waits in the queue when
    # queue.enabled_per_caller does not name it, and the class's own depth
    # and wait (queue.depth and queue.wait_s for a class that names none).
    # Within a class a waiter is passed by later callers that fit at most
    # ``queue_max_bypass`` times.
    caller_classes: dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_CALLER_CLASSES))
    class_queued: dict[str, bool] = field(
        default_factory=lambda: {"interactive": False, "user": False, "background": True}
    )
    class_depth: dict[str, int] = field(default_factory=lambda: {"background": 8})
    class_wait_s: dict[str, float] = field(default_factory=lambda: {"background": 120.0})
    queue_max_bypass: int = 2
    # The background splits a model between VRAM and RAM only when allowed.
    background_allow_split: bool = False
    # The background gate holds every background admission while an
    # interactive call is in flight (an entry held longer than
    # ``background_gate_in_flight_max_s`` is a leak and stops counting), or
    # admitted and not yet held (for at most
    # ``background_gate_admitted_grace_s``), or while a caller of a higher
    # class waits in the queue, and while the CPU pressure other programs
    # suffer (some avg10, percent) is at or above the enter mark, until it
    # falls below the exit mark. A load admitted and never seen by the loaded
    # view stops counting against the background after
    # ``background_gate_pending_load_max_s``.
    background_gate_enabled: bool = True
    background_gate_in_flight_max_s: float = 900.0
    background_gate_admitted_grace_s: float = 10.0
    background_gate_pending_load_max_s: float = 600.0
    background_gate_cpu_enter: float = 10.0
    background_gate_cpu_exit: float = 5.0
    # The CPU threads an engine computes with when the plan computes on the
    # CPU (a split, or a machine with no card): the machine's physical cores
    # less a reserve left to the user's programs (``threads_reserve_fraction``
    # of them, rounded up, within the floor and the ceiling), never an SMT
    # sibling; no more than the fastest class with ``threads_fast_cores_only``;
    # a model ``threads_models`` names takes its own count. The server's own
    # affinity and cgroup quota bound only what computes in its process
    # (``server_threads``). Off, no decision carries one and every engine
    # picks its own.
    threads_enabled: bool = True
    threads_reserve_fraction: float = 0.125
    threads_reserve_floor: int = 1
    threads_reserve_ceiling: int = 4
    threads_fast_cores_only: bool = False
    threads_models: dict[str, int] = field(default_factory=dict)
    # How many of the counts kept in the store the status lists, newest first.
    threads_status_limit: int = 50
    # The background's own budget (background_pool): worker processes that
    # each enter SCHED_IDLE, the idle I/O class and the CPUs of the cores
    # outside the reserve as they start; one per such core, never more than
    # ``threads_background_max_workers`` nor the quota less the reserve;
    # ``threads_background_in_flight`` tasks queued per worker; closed after
    # ``threads_background_idle_s`` without a task. A background caller the
    # admission holds asks again after ``threads_background_held_retry_s``.
    # ``threads_background_parse_expansion`` says how many times its size a
    # file's parse takes in memory, by extension ("default" for the others),
    # so a job sends no more files at once than the memory room holds.
    # Off, background work runs on the thread that asks for it.
    threads_background_enabled: bool = True
    threads_background_max_workers: int = 4
    threads_background_in_flight: int = 2
    threads_background_idle_s: float = 120.0
    threads_background_held_retry_s: float = 30.0
    threads_background_parse_expansion: dict[str, float] = field(default_factory=lambda: dict(_PARSE_EXPANSION))
    rlimits_enabled: bool = False
    rlimits_as_gb: float | None = None
    rlimits_data_gb: float | None = None
    ollama_max_loaded_models: int | None = None
    ollama_num_parallel: int | None = None
    ollama_max_queue: int | None = None
    ollama_spawn_applies: bool = True
    ollama_external_advisory: bool = True

    def class_of(self, caller: str | None) -> str:
        """The admission class of ``caller``; one not named is a user."""
        named = self.caller_classes.get(str(caller)) if caller is not None else None
        return named if named in _CLASS_RANK else _DEFAULT_CLASS


def _load_classes(cfg: GovernorConfig, classes: Mapping) -> None:
    """Merge the ``classes`` block over the defaults.

    ``callers`` may add a caller or move one to another class; a class
    outside ADMISSION_CLASSES, or no name at all (a list, a mapping), is
    refused with a warning and the caller keeps its class. Each class may
    name ``queued``, ``depth`` and ``wait_s``; the background also
    ``allow_split``.
    """
    callers = classes.get("callers")
    if isinstance(callers, dict):
        merged = dict(cfg.caller_classes)
        for caller, klass in callers.items():
            if isinstance(klass, str) and klass in _CLASS_RANK:
                merged[str(caller)] = klass
            else:
                logger.warning(
                    "classes.callers.%s %r is not one of %s; keeping %s",
                    caller,
                    klass,
                    ", ".join(ADMISSION_CLASSES),
                    cfg.class_of(str(caller)),
                )
        cfg.caller_classes = merged
    elif callers is not None:
        logger.warning("classes.callers is not a mapping; the caller table stands")
    queued = dict(cfg.class_queued)
    depth = dict(cfg.class_depth)
    wait = dict(cfg.class_wait_s)
    for klass in ADMISSION_CLASSES:
        block = classes.get(klass)
        if block is None:
            continue
        if not isinstance(block, dict):
            logger.warning("classes.%s is not a mapping; its defaults stand", klass)
            continue
        if "queued" in block:
            queued[klass] = _as_bool(block.get("queued"), queued.get(klass, False))
        if "depth" in block:
            depth[klass] = max(0, _as_int(block.get("depth"), depth.get(klass, cfg.queue_depth)))
        if "wait_s" in block:
            wait[klass] = _bounded_float(
                block, "wait_s", wait.get(klass, cfg.queue_wait_s), 0.0, None, f"classes.{klass}"
            )
        if klass == _BACKGROUND and "allow_split" in block:
            cfg.background_allow_split = _as_bool(block.get("allow_split"), cfg.background_allow_split)
    cfg.class_queued = queued
    cfg.class_depth = depth
    cfg.class_wait_s = wait


def _whole(section: Mapping, key: str, default: int) -> int:
    """``section[key]`` as a whole number at or above zero, else ``default``
    with a warning naming it; an absent key is the default, silently."""
    if key not in section:
        return default
    raw = section.get(key)
    if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
        return raw
    logger.warning("threads.%s %r is not a whole number at or above 0; keeping %s", key, raw, default)
    return default


def _load_threads(cfg: GovernorConfig, threads: Mapping) -> None:
    """The ``threads`` block onto ``cfg``: each key out of its range is
    warned and its default kept; a ceiling under the floor keeps both."""
    cfg.threads_enabled = _as_bool(threads.get("enabled"), cfg.threads_enabled)
    if "reserve_fraction" in threads:
        raw = threads.get("reserve_fraction")
        value = None if isinstance(raw, bool) else _finite_or_none(raw)
        if value is not None and 0.0 <= value < 1.0:
            cfg.threads_reserve_fraction = value
        else:
            logger.warning(
                "threads.reserve_fraction %r is not a number in [0, 1); keeping %s", raw, cfg.threads_reserve_fraction
            )
    floor = _whole(threads, "reserve_floor", cfg.threads_reserve_floor)
    ceiling = _whole(threads, "reserve_ceiling", cfg.threads_reserve_ceiling)
    if ceiling < floor:
        logger.warning(
            "threads.reserve_ceiling %s is under reserve_floor %s; keeping %s and %s",
            ceiling, floor, cfg.threads_reserve_floor, cfg.threads_reserve_ceiling,
        )
    else:
        cfg.threads_reserve_floor, cfg.threads_reserve_ceiling = floor, ceiling
    cfg.threads_fast_cores_only = _as_bool(threads.get("fast_cores_only"), cfg.threads_fast_cores_only)
    if "models" in threads:
        named = threads.get("models")
        if isinstance(named, dict):
            kept: dict[str, int] = {}
            for model, count in named.items():
                if isinstance(model, str) and model and isinstance(count, int) and not isinstance(count, bool) and count >= 1:
                    kept[model] = count
                else:
                    logger.warning("threads.models %r: %r is not a thread count of at least 1; ignored", model, count)
            cfg.threads_models = kept
        elif named is not None:
            logger.warning("threads.models is not a mapping of model names to thread counts; ignored")
    if "status_limit" in threads:
        raw = threads.get("status_limit")
        if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 1:
            cfg.threads_status_limit = raw
        else:
            logger.warning(
                "threads.status_limit %r is not a whole number of at least 1; keeping %s", raw, cfg.threads_status_limit
            )
    if "background" in threads:
        _load_background(cfg, threads.get("background"))


def _load_background(cfg: GovernorConfig, block: Any) -> None:
    """The ``threads.background`` block onto ``cfg``: each key out of its
    range is warned by name and its default kept."""
    if not isinstance(block, Mapping):
        logger.warning("threads.background %r is not a mapping; keeping its defaults", block)
        return
    cfg.threads_background_enabled = _as_bool(block.get("enabled"), cfg.threads_background_enabled)
    for key, attr in (
        ("max_workers", "threads_background_max_workers"),
        ("in_flight_per_worker", "threads_background_in_flight"),
    ):
        if key not in block:
            continue
        raw = block.get(key)
        if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 1:
            setattr(cfg, attr, raw)
        else:
            logger.warning(
                "threads.background.%s %r is not a whole number at or above 1; keeping %s", key, raw, getattr(cfg, attr)
            )
    for key, attr in (
        ("idle_shutdown_s", "threads_background_idle_s"),
        ("held_retry_s", "threads_background_held_retry_s"),
    ):
        if key not in block:
            continue
        raw = block.get(key)
        value = None if isinstance(raw, bool) else _finite_or_none(raw)
        if value is not None and value >= 0.0:
            setattr(cfg, attr, value)
        else:
            logger.warning(
                "threads.background.%s %r is not a number at or above 0; keeping %s", key, raw, getattr(cfg, attr)
            )
    if "parse_expansion" in block:
        cfg.threads_background_parse_expansion = _parse_expansion(block.get("parse_expansion"))


def _parse_expansion(raw: Any) -> dict[str, float]:
    """``threads.background.parse_expansion`` laid over the shipped factors:
    each key "default" or an extension (a dot, then letters or digits), each
    factor a number at or above 1; any other entry is warned by name and
    dropped, and a value that is not a mapping keeps the shipped factors."""
    factors = dict(_PARSE_EXPANSION)
    if not isinstance(raw, Mapping):
        logger.warning("threads.background.parse_expansion %r is not a mapping; keeping its defaults", raw)
        return factors
    for key, value in raw.items():
        name = str(key)
        number = None if isinstance(value, bool) else _finite_or_none(value)
        if (name != "default" and not _EXTENSION.fullmatch(name)) or number is None or number < 1.0:
            logger.warning(
                'threads.background.parse_expansion.%s %r is not a factor at or above 1 for "default" or an '
                "extension; dropped",
                name,
                value,
            )
            continue
        factors[name.lower()] = float(number)
    return factors


def _finite_or_none(raw: Any) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


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

    dyn = raw.get("dynamic_ctx")
    if isinstance(dyn, dict):
        cfg.dynamic_ctx_enabled = _as_bool(
            dyn.get("enabled"), cfg.dynamic_ctx_enabled
        )
        cfg.dynamic_ctx_unknown_ceiling = _as_int(
            dyn.get("unknown_ctx_ceiling"), cfg.dynamic_ctx_unknown_ceiling
        )
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
        cfg.queue_max_bypass = max(0, _as_int(queue.get("max_bypass"), cfg.queue_max_bypass))

    classes = raw.get("classes")
    if isinstance(classes, dict):
        _load_classes(cfg, classes)
    elif classes is not None:
        logger.warning("classes is not a mapping; the class defaults stand")

    gate = raw.get("background_gate")
    if isinstance(gate, dict):
        cfg.background_gate_enabled = _as_bool(gate.get("enabled"), cfg.background_gate_enabled)
        cfg.background_gate_in_flight_max_s = _bounded_float(
            gate, "in_flight_max_s", cfg.background_gate_in_flight_max_s, 0.0, None, "background_gate",
            low_open=True,
        )
        cfg.background_gate_admitted_grace_s = _bounded_float(
            gate, "admitted_grace_s", cfg.background_gate_admitted_grace_s, 0.0, None, "background_gate"
        )
        cfg.background_gate_pending_load_max_s = _bounded_float(
            gate, "pending_load_max_s", cfg.background_gate_pending_load_max_s, 0.0, None, "background_gate",
            low_open=True,
        )
        enter = _bounded_float(
            gate, "cpu_enter_some_avg10", cfg.background_gate_cpu_enter, 0.0, 100.0, "background_gate"
        )
        leave = _bounded_float(
            gate, "cpu_exit_some_avg10", cfg.background_gate_cpu_exit, 0.0, 100.0, "background_gate"
        )
        if leave > enter:
            logger.warning(
                "background_gate.cpu_exit_some_avg10 %s is above cpu_enter_some_avg10 %s; keeping %s and %s",
                leave,
                enter,
                cfg.background_gate_cpu_exit,
                cfg.background_gate_cpu_enter,
            )
            enter, leave = cfg.background_gate_cpu_enter, cfg.background_gate_cpu_exit
        cfg.background_gate_cpu_enter = enter
        cfg.background_gate_cpu_exit = leave
    elif gate is not None:
        logger.warning("background_gate is not a mapping; its defaults stand")

    threads = raw.get("threads")
    if isinstance(threads, dict):
        _load_threads(cfg, threads)
    elif threads is not None:
        logger.warning("threads is not a mapping; its defaults stand")

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

    offload = raw.get("offload")
    if isinstance(offload, dict):
        cfg.offload_enabled = _as_bool(offload.get("enabled"), cfg.offload_enabled)
        prefer = offload.get("prefer", cfg.offload_prefer)
        if prefer in _OFFLOAD_PREFER:
            cfg.offload_prefer = prefer
        else:
            logger.warning(
                "offload.prefer %r is not one of %s; keeping %s",
                prefer,
                ", ".join(_OFFLOAD_PREFER),
                cfg.offload_prefer,
            )
        cfg.offload_min_gpu_share = _bounded_float(
            offload, "min_gpu_share", cfg.offload_min_gpu_share, 0.0, 1.0
        )
        if "ram_reserve_gb" in offload and offload.get("ram_reserve_gb") is None:
            # null: the reserve is sized from the machine (ram_reserve).
            cfg.offload_ram_reserve_gb = None
        else:
            cfg.offload_ram_reserve_gb = _bounded_float(
                offload, "ram_reserve_gb", cfg.offload_ram_reserve_gb, 0.0, None
            )
    elif offload is not None:
        logger.warning("offload is not a mapping; the offload defaults stand")

    reserve = raw.get("ram_reserve")
    if isinstance(reserve, dict):
        fraction = _bounded_float(
            reserve, "fraction", cfg.ram_reserve_fraction, 0.0, 1.0, "ram_reserve"
        )
        floor = _bounded_float(
            reserve, "floor_gb", cfg.ram_reserve_floor_gb, 0.0, None, "ram_reserve"
        )
        ceiling = _bounded_float(
            reserve, "ceiling_gb", cfg.ram_reserve_ceiling_gb, 0.0, None, "ram_reserve"
        )
        factor = _bounded_float(
            reserve,
            "pressure_factor",
            cfg.ram_reserve_pressure_factor,
            1.0,
            None,
            "ram_reserve",
        )
        if floor > ceiling:
            logger.warning(
                "ram_reserve.floor_gb %s is above ceiling_gb %s; keeping %s and %s",
                floor,
                ceiling,
                cfg.ram_reserve_floor_gb,
                cfg.ram_reserve_ceiling_gb,
            )
            floor, ceiling = cfg.ram_reserve_floor_gb, cfg.ram_reserve_ceiling_gb
        cfg.ram_reserve_fraction = fraction
        cfg.ram_reserve_floor_gb = floor
        cfg.ram_reserve_ceiling_gb = ceiling
        cfg.ram_reserve_pressure_factor = factor
    elif reserve is not None:
        logger.warning("ram_reserve is not a mapping; the reserve defaults stand")

    speed = raw.get("split_speed")
    if isinstance(speed, dict):
        cfg.split_gpu_bandwidth_gbs = _optional_above(
            speed, "gpu_bandwidth_gbs", cfg.split_gpu_bandwidth_gbs, 0.0, False
        )
        cfg.split_ram_bandwidth_gbs = _optional_above(
            speed, "ram_bandwidth_gbs", cfg.split_ram_bandwidth_gbs, 0.0, False
        )
        cfg.split_max_slowdown = _optional_above(
            speed, "max_slowdown", cfg.split_max_slowdown, 1.0, True
        )
    elif speed is not None:
        logger.warning("split_speed is not a mapping; the speed is unknown")

    host = raw.get("host_pressure")
    if isinstance(host, dict):
        cfg.host_pressure_enabled = _as_bool(
            host.get("enabled"), cfg.host_pressure_enabled
        )
        enter = _bounded_float(
            host,
            "memory_enter_some_avg10",
            cfg.host_pressure_memory_enter,
            0.0,
            100.0,
            "host_pressure",
        )
        leave = _bounded_float(
            host,
            "memory_exit_some_avg10",
            cfg.host_pressure_memory_exit,
            0.0,
            100.0,
            "host_pressure",
        )
        if leave > enter:
            logger.warning(
                "host_pressure.memory_exit_some_avg10 %s is above"
                " memory_enter_some_avg10 %s; keeping %s and %s",
                leave,
                enter,
                cfg.host_pressure_memory_exit,
                cfg.host_pressure_memory_enter,
            )
            enter, leave = cfg.host_pressure_memory_enter, cfg.host_pressure_memory_exit
        cfg.host_pressure_memory_enter = enter
        cfg.host_pressure_memory_exit = leave
    elif host is not None:
        logger.warning("host_pressure is not a mapping; its defaults stand")

    return cfg


# ---------------------------------------------------------------------------
# R-03 limit management (Section 6)
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
    today (verified by review -- emergency_stop only warms up an
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

    Returns the status-API shape the governor status route reuses. status
    is one of "not_configured" | "match" | "mismatch" | "unknown";
    mixed observations resolve mismatch > unknown > match; a visible
    value that does not parse as an integer counts as a mismatch.
    Advisory-only by contract: the consumer (the startup security
    checklist) never blocks startup on it, in any mode (the standing
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
    """One S1 entry: a model the Ollama ps view reports as loaded.

    ``size_bytes`` is the total the engine reports for the model, VRAM and
    system RAM together, 0 when it reports none. A model split between the
    two shows a total above its ``size_vram_bytes``; one held in RAM alone
    shows a total and no VRAM at all.
    """

    name: str
    size_vram_bytes: int = 0
    expires_at: float | None = None
    context_length: int | None = None
    digest: str | None = None
    size_bytes: int = 0

    @property
    def size_vram_gb(self) -> float:
        return self.size_vram_bytes / _BYTES_PER_GIB if self.size_vram_bytes else 0.0

    @property
    def ram_bytes(self) -> int:
        """The part held outside VRAM; 0 when the engine reports no total."""
        return max(0, self.size_bytes - self.size_vram_bytes)

    @property
    def resident(self) -> bool:
        """Loaded, wherever it sits: on the GPU, split, or in RAM alone."""
        return self.size_vram_bytes > 0 or self.size_bytes > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size_vram_bytes": self.size_vram_bytes,
            "size_vram_gb": round(self.size_vram_gb, 3),
            "expires_at": self.expires_at,
            "context_length": self.context_length,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
            "ram_bytes": self.ram_bytes,
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

    ``vram_available_gb`` is RAW capacity minus in-use minus what other
    programs hold: the safety margin is deliberately NOT subtracted here;
    applying it belongs to the admission fit computation (spec Section
    4.2), once per selected card. ``sources`` is the honest provenance list
    naming exactly which read paths contributed. ``taken_at`` is on the
    governor's clock (monotonic by default). ``devices`` are the cards the
    capacity was summed from (none when it was configured or injected),
    ``vram_others_gb`` the memory other programs hold on them (the cards'
    used memory less what the engines declare, never below zero; its
    reading is ``vram_used_age_s`` old), and ``host_pressure`` the kernel's
    pressure stall information, None where it could not be read.
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
    ram_total_mb: float = 0.0
    devices: list[dict[str, Any]] = field(default_factory=list)
    device_count: int = 0
    vram_others_gb: float = 0.0
    vram_used_age_s: float | None = None
    vram_others_carried: bool = False
    host_pressure: dict[str, Any] | None = None
    memory_pressure_active: bool = False
    # The capacity is unknown because the profile knows the machine has no
    # card, not because a card could not be read.
    cards_absent: bool = False

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
            "ram_total_mb": round(self.ram_total_mb, 1),
            "devices": [dict(d) for d in self.devices],
            "device_count": self.device_count,
            "vram_others_gb": round(self.vram_others_gb, 3),
            "vram_used_age_s": (
                round(self.vram_used_age_s, 3)
                if self.vram_used_age_s is not None
                else None
            ),
            "vram_others_carried": self.vram_others_carried,
            "cards_absent": self.cards_absent,
            "host_pressure": (
                {
                    resource: {kind: dict(values) for kind, values in lines.items()}
                    for resource, lines in self.host_pressure.items()
                }
                if self.host_pressure is not None
                else None
            ),
            "memory_pressure_active": self.memory_pressure_active,
        }


# ---------------------------------------------------------------------------
# The admission ticket (Section 4.4) and the typed refusal
# ---------------------------------------------------------------------------


@dataclass
class AdmissionDecision:
    """The admission ticket (spec Section 4.4).

    The first nine fields are the contract shape ({admitted, model,
    num_ctx, num_gpu or None, keep_alive override or None, action in
    {admit, downsize, refuse, queue}, reason, snapshot provenance, ticket
    id}); the trailing fields are internal companions (accounting,
    testability, payload capture) and not part of the minimum surface.
    num_gpu is the number of layers a split puts on the GPU when the model's
    file says what each layer weighs, and the count a split load pinned when
    the model is resident; else None, and the engine places the layers.
    Ollama is told it only as ``ollama_layers`` allows. A split names itself
    in the partial offload companions: the share of the cost on the GPU, the
    GiB on the GPU and in RAM, the layers on the GPU
    when their count is known (from the file, else in proportion to the
    cost), and how many times slower than on the GPU alone the split is
    expected to run, when the bandwidths are known. A refusal reached after
    a split was priced also names the RAM shortfall. keep_alive carries the
    soft-pressure override when the pressure signal fills it.
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
    # -- partial offload companions (None when no split was priced) ---------
    gpu_share: float | None = None
    vram_cost_gb: float | None = None
    ram_cost_gb: float | None = None
    gpu_layers: int | None = None
    ram_shortfall_gb: float | None = None
    expected_slowdown: float | None = None
    # How many sequences Ollama keeps a KV cache for at once, as the operator
    # names it (ollama_limits.num_parallel), or None unnamed. A split prices
    # the KV of one sequence, so Ollama is told num_gpu only while it keeps
    # one (``ollama_layers``); llama.cpp in process always keeps one.
    num_parallel: int | None = None
    # -- what the load was charged, and by whom it will be served ------------
    # ``engine`` is the backend that will serve the call, when the registry
    # or the caller can say; ``cost_gb`` the GiB charged for the admitted
    # context (weights, any declared per-request state, and the KV cache
    # when the model keeps one); ``credit_gb`` the GiB a reload of a
    # resident model frees first. The eviction a conditional grant plans
    # prices the load with these, exactly as the admission did.
    engine: str | None = None
    cost_gb: float | None = None
    credit_gb: float = 0.0
    # -- who asked: the caller's admission class, and why the background gate
    # held a background admission (None when it did not) --------------------
    admission_class: str = _DEFAULT_CLASS
    held_by: str | None = None
    # -- the CPU threads the engine computes with: ``threads`` for the tokens
    # it makes (Ollama's num_thread, llama.cpp's n_threads), ``threads_batch``
    # for the prompt it reads (llama.cpp's n_threads_batch), and where they
    # come from ("plan", "override", or "pinned" to a resident or a pending
    # load); all None when the load does not compute on the CPU, or the CPUs
    # cannot be read, and the engine picks its own --------------------------
    threads: int | None = None
    threads_batch: int | None = None
    threads_source: str | None = None
    # -- where the CPU computes, as exactly as the governor knows it: "cpu"
    # on a machine with no card, "split:<n>" for a split that puts n layers
    # on the GPU; None where the card holds the model whole, or where a
    # split's layers are not counted. A thread count measured for a model is
    # kept and read for its engine and this placement alone --------------
    placement: str | None = None

    @property
    def partial_offload(self) -> bool:
        """Admitted split between VRAM and system RAM."""
        return (
            self.admitted
            and self.gpu_share is not None
            and self.gpu_share < 1.0
        )

    def ollama_layers(self, options: Mapping[str, Any] | None) -> int | None:
        """The layer count Ollama may be told as num_gpu for a call with
        ``options``, or None, and Ollama places the layers itself.

        Only a count priced for the KV Ollama will keep: while it keeps one
        sequence (``num_parallel`` 1), for a call at the context the count was
        priced for, which names no num_gpu of its own. The engine head that
        sends it and the gate that pins it both ask here.
        """
        layers = self.num_gpu
        if isinstance(layers, bool) or not isinstance(layers, int) or layers < 0:
            return None
        if type(self.num_parallel) is not int or self.num_parallel != 1:
            return None
        sent = options if isinstance(options, Mapping) else {}
        if "num_gpu" in sent:
            return None
        if sent.get("num_ctx") != self.num_ctx:
            return None
        return layers

    def ollama_threads(self, options: Mapping[str, Any] | None) -> int | None:
        """The thread count Ollama may be told as num_thread for a call with
        ``options``, or None: the decision's own, unless the call names its
        own. Ollama reloads a resident model for any num_thread other than
        the one it was loaded with, so the engine heads that send it and the
        gate that pins it all ask here."""
        threads = self.threads
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            return None
        sent = options if isinstance(options, Mapping) else {}
        if "num_thread" in sent:
            return None
        return threads

    def refusal_payload(self) -> dict[str, Any]:
        """The honest refusal body, mirroring the estop idiom (D3).

        The estop case returns the payload captured at decision time from
        emergency_stop.refusal_payload() (spec 4.5), and so does a refusal
        reached after a split was priced, which names both shortfalls, or
        an engine's refusal of a split; the resource case names the model,
        the shortfall and the options.
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
            "gpu_share": self.gpu_share,
            "vram_cost_gb": self.vram_cost_gb,
            "ram_cost_gb": self.ram_cost_gb,
            "gpu_layers": self.gpu_layers,
            "ram_shortfall_gb": self.ram_shortfall_gb,
            "expected_slowdown": self.expected_slowdown,
            "admission_class": self.admission_class,
            "held_by": self.held_by,
            "threads": self.threads,
            "threads_batch": self.threads_batch,
            "threads_source": self.threads_source,
            "placement": self.placement,
        }


@dataclass(eq=False)
class _Waiter:
    """One caller waiting in the admission queue: its class, its place in
    the order of arrival, and how many later callers have passed it."""

    admission_class: str
    seq: int
    bypassed: int = 0


@dataclass
class _PendingLoad:
    """A load admitted that the loaded view does not show yet: the model, the
    context it loads at, what it adds to VRAM and to RAM, the extra models
    loading with it, the class it was admitted for, since when, and when the
    call that held its ticket released it (None while held, or never held)."""

    model: str
    num_ctx: int | None
    vram_gb: float
    ram_gb: float
    extras: tuple[str, ...]
    admission_class: str
    since: float
    released_at: float | None = None
    # The CPU threads the load was admitted with, and where it computes: a
    # call joining it carries them.
    threads: int | None = None
    threads_batch: int | None = None
    placement: str | None = None


@dataclass
class _SharedPressure:
    """What the pressure readings leave behind them, one object for a
    governor and every governor a reload builds from it (_LIVE_STATE), so a
    configuration write never reopens a held gate nor forgets a keep_alive
    to restore: the side of the memory hysteresis and of the background
    gate's CPU hysteresis (the readings are taken again, and the marks of
    the file judge them), since when soft-or-worse pressure has lasted, and
    the warm-up keep_alive a sustained pressure replaced. Read and written
    only under the cache lock, which a reload shares too: each governor
    builds its snapshot under a lock of its own, so both may judge at once."""

    memory_pressure_active: bool = False
    cpu_held: bool = False
    pressure_soft_since: float | None = None
    keep_alive_original: str | None = None


class GovernorRefusal(RuntimeError):
    """Typed refusal raised by the mechanical backend gate (4.1/4.4)."""

    def __init__(self, decision: AdmissionDecision):
        super().__init__(
            decision.refusal_payload().get(
                "message", "resource admission refused"
            )
        )
        self.decision = decision


def _offload_refusal_payload(
    model: str,
    vram_shortfall: float,
    ram_shortfall: float,
    share: float,
    min_share: float,
    *,
    no_layer_fits: bool = False,
    slowdown: float | None = None,
    max_slowdown: float | None = None,
) -> dict[str, Any]:
    """The refusal body once a split was priced: what each placement lacked."""
    message = (
        f"Not enough resources to load {model}: short by"
        f" {vram_shortfall:.1f} GB of VRAM on the GPU alone"
    )
    if ram_shortfall > 0.0:
        message += (
            f", and by {ram_shortfall:.1f} GB of RAM split between the GPU"
            " and system RAM"
        )
    if share < min_share:
        message += (
            f"; split, the GPU would hold {share:.0%} of it, under the"
            f" {min_share:.0%} minimum"
        )
    if no_layer_fits:
        message += "; split, not one of its layers fits the VRAM free now"
    if slowdown is not None and max_slowdown is not None:
        message += (
            f"; split, it would run {slowdown:.1f} times slower than on the"
            f" GPU alone, over the {max_slowdown:.1f} allowed"
        )
    message += ". Evict idle models, pick a smaller model, or lower the context."
    return {
        "error": "resource_admission_refused",
        "message": message,
        "model": model,
        "shortfall_gb": vram_shortfall,
        "ram_shortfall_gb": ram_shortfall,
        "options": [
            "evict idle models",
            "pick a smaller model",
            "lower context",
        ],
    }


def _refuse_split(governor: Any, decision: AdmissionDecision, engine: str) -> None:
    """Refuse a split admission the calling engine cannot run; never returns.

    Recorded like any refusal, so the ring shows the engine's refusal next
    to the admission it overrules, and raised as the typed refusal every
    head already lets through.
    """
    vram = decision.vram_cost_gb or 0.0
    ram = decision.ram_cost_gb or 0.0
    refusal = AdmissionDecision(
        admitted=False,
        model=decision.model,
        action="refuse",
        reason="partial_offload_unsupported",
        provenance=list(decision.provenance),
        ticket_id=decision.ticket_id,
        caller=decision.caller,
        requested_ctx=decision.requested_ctx,
        gpu_share=decision.gpu_share,
        vram_cost_gb=decision.vram_cost_gb,
        ram_cost_gb=decision.ram_cost_gb,
        gpu_layers=decision.gpu_layers,
        admission_class=decision.admission_class,
        payload={
            "error": "resource_admission_refused",
            "message": (
                f"{decision.model} fits only split between the GPU and system"
                f" RAM ({vram:.1f} GB on the GPU, {ram:.1f} GB in RAM), and"
                f" {engine} cannot split it: serve it through Ollama, which"
                " splits it itself, or pick a smaller model or a shorter"
                " context."
            ),
            "model": decision.model,
            "vram_cost_gb": decision.vram_cost_gb,
            "ram_cost_gb": decision.ram_cost_gb,
            "options": [
                "serve it through Ollama",
                "pick a smaller model",
                "lower context",
            ],
        },
    )
    try:
        governor._record_admission(refusal)
    except Exception as exc:
        logger.debug("Split refusal record failed: %s", exc)
    # The load the admission counted on will not happen.
    end = getattr(governor, "end_pending_load", None)
    if callable(end):
        end(decision.ticket_id)
    raise GovernorRefusal(refusal)


_ticket_local = threading.local()


def get_active_ticket() -> AdmissionDecision | None:
    """The calling thread's active admission ticket, if a funnel set one."""
    return getattr(_ticket_local, "ticket", None)


def set_active_ticket(decision: AdmissionDecision | None) -> None:
    """Set the thread-local ticket the backend gate will see (4.4); the
    governor counts it in flight until it is released."""
    _ticket_local.ticket = decision
    _note_held(decision)


def clear_active_ticket() -> None:
    """Drop the thread-local ticket."""
    _ticket_local.ticket = None
    _note_held(None)


def _note_held(decision: AdmissionDecision | None) -> None:
    """Tell the governor, when there is one, what this thread now holds."""
    governor = _governor
    if governor is None:
        return
    try:
        governor.note_held(decision)
    except Exception as exc:
        logger.debug("In-flight registry update failed open: %s", exc)


@contextmanager
def ticket_scope(decision: AdmissionDecision | None):
    """Hold an admission ticket around a backend call (Section 4.4).

    The pass-through mechanism, as arbitrated: a thread
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
    (fast down, slow up, config floor), the bounded recent-decisions
    ring (schema and prune-by-count land here; the admission path writes
    the rows), and the CPU thread counts a tuner measured and kept, per
    model, engine and placement, with the fingerprint of the machine they
    were measured on.
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

                    CREATE TABLE IF NOT EXISTS thread_optima (
                        model TEXT NOT NULL,
                        engine TEXT NOT NULL,
                        placement TEXT NOT NULL,
                        fingerprint TEXT NOT NULL,
                        threads INTEGER NOT NULL,
                        threads_batch INTEGER NOT NULL,
                        tg REAL NOT NULL,
                        base_tg REAL NOT NULL,
                        measured_at REAL NOT NULL,
                        PRIMARY KEY (model, engine, placement)
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
        """Persist an observed per-model cost (supersedes statics).

        The ``size_vram_bytes`` column keeps its name and holds what the
        model takes as loaded: the total its engine reports when it reports
        one (a split model's VRAM and RAM together), else its VRAM.
        """
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

    # -- thread counts a tuner measured and kept ------------------------------

    def record_thread_optimum(
        self,
        model: str,
        engine: str,
        placement: str,
        fingerprint: str,
        threads: int,
        threads_batch: int,
        tg: float,
        base_tg: float,
        measured_at: float | None = None,
    ) -> None:
        """Keep the thread counts measured for ``model`` served by ``engine``
        at ``placement``, replacing what was kept there before."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO thread_optima
                       (model, engine, placement, fingerprint, threads,
                        threads_batch, tg, base_tg, measured_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        model,
                        engine,
                        placement,
                        fingerprint,
                        int(threads),
                        int(threads_batch),
                        float(tg),
                        float(base_tg),
                        measured_at if measured_at is not None else time.time(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def get_thread_optimum(self, model: str, engine: str, placement: str) -> dict[str, Any] | None:
        """The counts kept for exactly (``model``, ``engine``, ``placement``),
        or None. A row that does not hold together -- a count that is not a
        whole number of one or more, a rate that is not a positive finite
        number, no fingerprint -- is ignored rather than trusted: the file
        is written by this process, but read back from a disk anyone may
        have edited."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """SELECT fingerprint, threads, threads_batch, tg, base_tg,
                              measured_at
                       FROM thread_optima
                       WHERE model = ? AND engine = ? AND placement = ?""",
                    (model, engine, placement),
                ).fetchone()
            finally:
                conn.close()
        if row is None:
            return None
        return _held_optimum(model, engine, placement, *row)

    def thread_optima(self, limit: int) -> list[dict[str, Any]]:
        """The counts kept, newest first, at most ``limit`` of them. A row
        that does not hold together is passed over, as get_thread_optimum
        ignores it, and takes no place in the list."""
        kept: list[dict[str, Any]] = []
        if limit < 1:
            return kept
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT model, engine, placement, fingerprint, threads,
                              threads_batch, tg, base_tg, measured_at
                       FROM thread_optima
                       ORDER BY measured_at DESC"""
                )
                for row in rows:
                    held = _held_optimum(*row)
                    if held is not None:
                        kept.append(held)
                        if len(kept) >= limit:
                            break
            finally:
                conn.close()
        return kept

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

        The admission path is the writer; this method implements the
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
# The background's own budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackgroundPlan:
    """How many background workers, on which CPUs, and why.

    ``cpus`` empty leaves each worker the CPUs it inherits; ``reserved`` are
    the CPUs of the cores left to the user's programs; ``source`` is "plan",
    "disabled" (no worker: background work runs on the thread that asks) or
    "unknown" (the CPUs could not be read). ``parse_expansion`` says how many
    times its size a file's parse takes in memory, by extension, with a
    "default" for the others.
    """

    workers: int
    cpus: tuple[int, ...]
    reserved: tuple[int, ...]
    in_flight: int
    idle_s: float
    held_retry_s: float
    source: str
    parse_expansion: dict[str, float] = field(default_factory=dict)


def _reserve_order(core: Any) -> tuple:
    """A core's place in the reserve: the fastest class first, then the
    highest rank its source reads (a core with no rank after every ranked
    one), then the lowest CPU."""
    rank = getattr(core, "rank", None)
    read = isinstance(rank, (int, float)) and not isinstance(rank, bool) and math.isfinite(rank)
    cpus = tuple(getattr(core, "cpus", ()) or ())
    return (getattr(core, "perf_class", 0), 0 if read else 1, -rank if read else 0.0, cpus[0] if cpus else 0)


def _held_optimum(
    model: Any, engine: Any, placement: Any, fingerprint: Any, threads: Any, threads_batch: Any, tg: Any,
    base_tg: Any, measured_at: Any,
) -> dict[str, Any] | None:
    """A kept thread count as the store returns it, or None when the row
    does not hold together: a key that is not a name, a count that is not a
    whole number of one or more, a rate that is not a positive finite
    number, no fingerprint. The file is written by this process, but read
    back from a disk anyone may have edited."""
    keys_hold = all(isinstance(k, str) and k for k in (model, engine, placement))
    counts_hold = all(isinstance(n, int) and not isinstance(n, bool) and n >= 1 for n in (threads, threads_batch))
    rates_hold = all(
        isinstance(r, (int, float)) and not isinstance(r, bool) and math.isfinite(r) and r > 0 for r in (tg, base_tg)
    )
    if not keys_hold or not counts_hold or not rates_hold or not isinstance(fingerprint, str) or not fingerprint:
        logger.debug("thread_optima row for %s/%s/%s does not hold together; ignored", model, engine, placement)
        return None
    return {
        "model": model,
        "engine": engine,
        "placement": placement,
        "fingerprint": fingerprint,
        "threads": threads,
        "threads_batch": threads_batch,
        "tg": float(tg),
        "base_tg": float(base_tg),
        "measured_at": measured_at,
    }


def refusal_is_final(decision: Any) -> bool:
    """Whether ``decision`` refuses for good: the emergency stop, or a
    refusal no wait can lift (the card unreadable, the cost or the context
    unknown). An admission is no refusal."""
    if decision is None or getattr(decision, "admitted", False):
        return False
    return bool(getattr(decision, "is_estop", False)) or getattr(decision, "reason", "") in _FINAL_REFUSALS


# ---------------------------------------------------------------------------
# The governor
# ---------------------------------------------------------------------------


class ResourceGovernor:
    """Governor core: measures, caches, learns; admits.

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
        vram_probe: Any = _UNSET,
        hardware: Any = _UNSET,
    ):
        self._config = load_config(config_path)
        self._store = AdaptStore(db_path)
        # What a reload builds the governor again from (reload_resource_governor).
        self._paths = (config_path, db_path)
        self._vram_probe_arg = vram_probe
        if warmup is _UNSET:
            self._warmup = _load_default_warmup()
        else:
            self._warmup = warmup
        self._registry_override = registry
        self._clock = clock
        self._meminfo_path = meminfo_path
        # Consulted only when no capacity is configured. An injected probe
        # (a callable answering the total in MiB, or None for none) is the
        # whole of the capacity reading, and no card is read beside it;
        # left unset, the capacity is the hardware profile's placement.
        self._probe_injected = vram_probe is not _UNSET
        self._vram_probe = None if vram_probe is _UNSET else vram_probe
        # The hardware profile: injected (an object, or None for none), or
        # resolved lazily at the first snapshot, so importing the governor
        # still spawns no subprocess and touches no device.
        self._hardware_arg = hardware
        self._hardware_resolved: Any = _UNSET
        # What the pressure readings leave behind them (_SharedPressure).
        self._shared = _SharedPressure()
        # What other programs held on the cards at the last reading that
        # paired with the engines' holdings: carried across a load or an
        # eviction until the cards are read again.
        self._others_carry: float | None = None
        self._estimator = _load_vram_estimator()
        self._cache_lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        self._snapshot: ResourceSnapshot | None = None
        # KV geometries found per model (kv_geometry); dropped at a load.
        self._geometry: dict[str, dict[str, Any]] = {}
        # Tensor tables by file identity (path, size, mtime_ns), the latest
        # last; a table that was refused is kept as None, so a damaged file
        # is read once until it changes.
        self._tables: dict[tuple[str, int, int], Any] = {}
        # The layer count each split load pinned, and whether the loaded view
        # has shown the model since (only then does leaving it clear the pin).
        self._pins: dict[str, list[Any]] = {}
        # The CPU threads each load was told, [threads, threads_batch, seen],
        # ended the same way as the layer pins.
        self._thread_pins: dict[str, list[Any]] = {}
        self._pending_attribution: dict[str, int | None] = {}
        self._refresh_in_flight = False
        self._capacity_warning_emitted = False
        # The admission path observes the estop flag;
        # a transition triggers the drain/resume invalidation hooks
        # without editing emergency_stop. None until first observation.
        self._last_estop_seen: bool | None = None
        # Runtime backpressure state. The refusal-rate
        # window is in-memory by design (a runtime signal, never
        # persisted; the maxlen is a memory bound, the time window is
        # the config key). The remembered keep_alive original (_shared)
        # survives a reload but NOT a process restart: the warmup
        # re-initialises at its own default, which is the honest restore
        # in that case.
        self._refusal_events: deque = deque(maxlen=512)
        self._last_pressure_level: str | None = None
        # Queue: waiters block on the condition in bounded slices;
        # the invalidation hooks notify it (capacity may have moved). Its
        # lock also guards the calls in flight (thread -> the ticket it
        # holds, and since when) and the class each resident was loaded for
        # ([class, seen in the loaded view since]).
        self._queue_cond = threading.Condition()
        self._waiters: list[_Waiter] = []
        self._queue_seq = itertools.count(1)
        self._in_flight: dict[int, tuple[AdmissionDecision, float]] = {}
        self._owners: dict[str, list[Any]] = {}
        # Under the same lock: the interactive admissions no thread holds
        # yet (ticket -> the decision, since when), and the loads admitted
        # and not yet seen (ticket -> _PendingLoad) with the count of loads
        # ever claimed (a one-item list, so a governor a reload builds shares
        # it), which a background decision reads before it starts and checks
        # before it claims its own. Background decisions are taken one at a
        # time; a retry in the queue records nothing (_quiet).
        self._admitted: dict[str, tuple[AdmissionDecision, float]] = {}
        self._pending_loads: dict[str, _PendingLoad] = {}
        self._pending_version = [0]
        self._background_lock = threading.Lock()
        self._quiet = threading.local()
        # The background gate's last reading of the CPU pressure other
        # programs suffer, (taken at, reading); whether it holds the gate is
        # in _shared.
        self._others_cpu: tuple[float, dict[str, Any] | None] | None = None

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
            # Only once the snapshot is stored: a background decision that
            # read the loads before it ends one reads this snapshot or a
            # newer one, which counts the load's memory itself.
            self._settle_pending_loads(built)
            return built

    def get_snapshot(self) -> ResourceSnapshot:
        """Fresh-or-rebuild, synchronous (the simple consumer path)."""
        return self.refresh(force=False)

    def get_snapshot_fast(self) -> ResourceSnapshot:
        """The admission fast-path primitive (Section 3).

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

    # -- eager invalidation hooks (callable; now wired to their callers) -----

    def invalidate_on_load(
        self, model: str, requested_num_ctx: int | None = None
    ) -> None:
        """An admitted load happened: drop the cache and register the
        model for post-load cost attribution at the next fresh ps view.
        The model's KV geometry is read again too: a load may bring other
        bytes under the same name. A load ends the layer count a previous
        split load pinned; a split load pins its own after this. It also
        ends the class the model was loaded for: the engine gate names the
        new one (note_loaded_by)."""
        with self._cache_lock:
            self._pending_attribution[model] = requested_num_ctx
            self._geometry.pop(model, None)
            self._pins.pop(model, None)
            self._thread_pins.pop(model, None)
            self._snapshot = None
        with self._queue_cond:
            self._owners.pop(model, None)
        self._invalidate_card_reading()
        self._notify_queue()

    def invalidate_on_evict(self, model: str | None = None) -> None:
        with self._cache_lock:
            # An eviction ends the model's pin; one that names no model may
            # have unloaded any of them.
            if model:
                self._pins.pop(model, None)
                self._thread_pins.pop(model, None)
            else:
                self._pins.clear()
                self._thread_pins.clear()
            self._snapshot = None
        with self._queue_cond:
            if model:
                self._owners.pop(model, None)
            else:
                self._owners.clear()
        self._invalidate_card_reading()
        self._notify_queue()

    def invalidate_on_estop_drain(self) -> None:
        with self._cache_lock:
            self._snapshot = None
        self._invalidate_card_reading()
        self._notify_queue()

    def invalidate_on_resume(self) -> None:
        with self._cache_lock:
            self._snapshot = None
        self._invalidate_card_reading()
        self._notify_queue()

    def _invalidate_card_reading(self) -> None:
        """The engines' holdings changed: ask the profile for a fresh reading.

        Called after the cache lock is released. Only a profile already in
        use is told; none is resolved for it.
        """
        hardware = (
            self._hardware_arg
            if self._hardware_arg is not _UNSET
            else self._hardware_resolved
        )
        if hardware is None or hardware is _UNSET:
            return
        invalidate = getattr(hardware, "invalidate_used", None)
        if invalidate is None:
            return
        try:
            invalidate()
        except Exception as exc:
            logger.debug("Card reading invalidation failed: %s", exc)

    def _notify_queue(self) -> None:
        """Wake queued admissions: the world may have moved.

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
        """Per-model KV coefficient, in GiB per 1024 tokens.

        Resolution order: an exact entry in ``kv_override_models``, else
        the LONGEST ``kv_override_families`` key contained in the
        lowercased name, else what the model's own geometry gives
        (:meth:`kv_geometry`), else the global ``kv_coefficient`` -- the
        fail-secure default for any model neither the tables nor its
        engine describe (an unknown model is never under-budgeted). An
        operator's word wins over the computed figure. Matching is
        case-insensitive; the tables hold lowercase keys by load-time
        normalisation.
        """
        override = self._kv_override(model)
        if override is not None:
            return override
        return self._coefficient_from(self.kv_geometry(model))

    def _kv_override(self, model: str | None) -> float | None:
        """The operator's KV coefficient for ``model``: exact, then family."""
        if not model:
            return None
        name = str(model).lower()
        exact = self._config.kv_override_models.get(name)
        if exact is not None:
            return exact
        best: float | None = None
        best_len = -1
        for family, coeff in self._config.kv_override_families.items():
            if family and family in name and len(family) > best_len:
                best, best_len = coeff, len(family)
        return best

    def _coefficient_from(self, geometry: dict[str, Any] | None) -> float:
        """GiB per 1024 tokens for a geometry; the flat coefficient for none."""
        if geometry is None:
            return self._config.kv_coefficient
        return geometry["kv_bytes_per_token"] * 1024.0 / _BYTES_PER_GIB

    def kv_geometry(
        self, model: str | None, engine: str | None = None
    ) -> dict[str, Any] | None:
        """The model's KV cache geometry, as its engine describes it.

        ``{"layers", "kv_bytes_per_token", "kv_bytes_per_layer"}`` or None.
        The backend that will
        serve the model is asked first (``_serving_backend``), then the
        others in their registration order, for the model's information:
        the metadata mapping Ollama reports first, else the header of the
        GGUF file the engine names. A geometry found is kept until the
        model is loaded again; none found is asked again at the next call,
        so an engine that was not answering yet is not taken at its
        silence. Never raises.
        """
        if not model:
            return None
        with self._cache_lock:
            known = self._geometry.get(model)
        if known is not None:
            return known
        for backend in self._backend_order(model, engine):
            try:
                info = backend.model_info(model)
            except Exception as exc:
                logger.debug("model_info(%s) failed: %s", model, exc)
                continue
            if info is None:
                continue
            extra = getattr(info, "extra", None)
            meta = extra.get("model_info") if isinstance(extra, dict) else None
            geometry = _kv_geometry_from_metadata(meta)
            if geometry is None:
                geometry = _kv_geometry_from_gguf(getattr(info, "path", None))
            if geometry is not None:
                with self._cache_lock:
                    self._geometry[model] = geometry
                return geometry
        return None

    def tensor_table(self, model: str | None, engine: str | None = None) -> Any:
        """The tensor table of the file that serves ``model``, or None.

        The backend that will serve the model is asked first, then the others
        in their registration order, for the file it loads: llama.cpp names
        it, Ollama's modelfile names its blob (``_model_files``). A file whose
        architecture is not the model's, a vision projector, is set aside, and
        so is a table without blocks. Never raises.
        """
        if not model:
            return None
        for backend in self._backend_order(model, engine):
            try:
                info = backend.model_info(model)
            except Exception as exc:
                logger.debug("model_info(%s) failed: %s", model, exc)
                continue
            if info is None:
                continue
            extra = getattr(info, "extra", None)
            meta = extra.get("model_info") if isinstance(extra, dict) else None
            arch = meta.get("general.architecture") if isinstance(meta, Mapping) else None
            for path in _model_files(info):
                table = self._table_at(path)
                if table is None or not table.blocks:
                    continue
                if isinstance(arch, str) and arch and table.architecture != arch:
                    continue
                return table
        return None

    def _table_at(self, path: str) -> Any:
        """The tensor table of the regular file at ``path``, by its identity.

        Read once per identity (path, size, modification time) and again
        when the file changes; a refused table is kept as None until then.
        """
        try:
            info = os.lstat(path)
        except (OSError, ValueError):
            return None
        if not stat.S_ISREG(info.st_mode):
            return None
        key = (path, info.st_size, info.st_mtime_ns)
        with self._cache_lock:
            if key in self._tables:
                table = self._tables.pop(key)
                self._tables[key] = table
                return table
        table = _read_tensor_table(path)
        with self._cache_lock:
            self._tables[key] = table
            while len(self._tables) > _TABLE_CACHE_SIZE:
                self._tables.pop(next(iter(self._tables)))
        return table

    def pin_layers(self, model: str, layers: int, num_ctx: int | None) -> None:
        """A split load of ``model`` at ``num_ctx`` put ``layers`` on the GPU.

        The decisions for the resident model at that same context carry the
        same count, so the engine keeps the model as it is rather than load
        it again; at another context the count was not priced, and is not
        carried.
        """
        with self._cache_lock:
            self._pins[model] = (layers, num_ctx, False)

    def pinned_layers(self, model: str | None, num_ctx: int | None) -> int | None:
        """The layer count a split load of ``model`` pinned at ``num_ctx``."""
        with self._cache_lock:
            pin = self._pins.get(model) if model else None
        if pin is None or num_ctx is None or pin[1] != num_ctx:
            return None
        return pin[0]

    def pin_threads(self, model: str, threads: int, threads_batch: int | None) -> None:
        """A load of ``model`` was told ``threads`` CPU threads for its tokens
        (and ``threads_batch`` for its prompt).

        The decisions for the resident model carry the same counts, whatever
        the plan says by then: Ollama reloads a resident model for any
        num_thread other than the one it was loaded with. The pin ends with
        the resident, as the layer pins do. A count pinned again for the
        same resident (a call sent its own) keeps the placement of the load
        and whether the loaded view has shown the model.
        """
        with self._cache_lock:
            old = self._thread_pins.get(model)
            seen, placement = (old[2], old[3]) if old is not None else (False, None)
            self._thread_pins[model] = [threads, threads_batch, seen, placement]

    def pinned_threads(self, model: str | None) -> tuple[int, int | None] | None:
        """The (threads, threads_batch) a load of ``model`` was told, or None."""
        with self._cache_lock:
            pin = self._thread_pins.get(model) if model else None
        return None if pin is None else (pin[0], pin[1])

    def pin_placement(self, model: str, placement: str | None) -> None:
        """The load of ``model`` whose CPU threads are pinned computes at
        ``placement``: the decisions for the resident say it, so a count
        measured there is read for it. Nothing is pinned without a count."""
        with self._cache_lock:
            pin = self._thread_pins.get(model)
            if pin is not None:
                pin[3] = placement

    def pinned_placement(self, model: str | None) -> str | None:
        """The placement of the load whose threads are pinned for ``model``."""
        with self._cache_lock:
            pin = self._thread_pins.get(model) if model else None
        return None if pin is None else pin[3]

    def _cpu_topology(self) -> Any:
        """The CPU topology the hardware profile reads for this process (its
        affinity, its quota), or None."""
        hardware = self._hardware()
        read = getattr(hardware, "cpu_topology", None) if hardware is not None else None
        if not callable(read):
            return None
        try:
            return read()
        except Exception as exc:
            logger.debug("CPU topology read failed: %s", exc)
            return None

    def _machine_topology(self) -> Any:
        """The machine's CPU topology, as an engine that computes in a
        process of its own may run on it (the profile's machine view: every
        online CPU, no quota), or None; a profile with no such view answers
        with this process's own."""
        hardware = self._hardware()
        read = getattr(hardware, "machine_cpu_topology", None) if hardware is not None else None
        if not callable(read):
            return self._cpu_topology()
        try:
            return read()
        except Exception as exc:
            logger.debug("Machine CPU topology read failed: %s", exc)
            return None

    def thread_reserve(self, physical: int) -> int:
        """The physical cores a plan leaves to the user's programs.

        ``threads_reserve_fraction`` of ``physical``, rounded up, held
        between the floor and the ceiling, and never every core.
        """
        cfg = self._config
        count = math.ceil(cfg.threads_reserve_fraction * physical)
        count = max(cfg.threads_reserve_floor, min(cfg.threads_reserve_ceiling, count))
        return max(0, min(count, physical - 1))

    def plan_threads(
        self, model: str | None, engine: str | None = None, placement: str | None = None
    ) -> tuple[int | None, int | None, str]:
        """(threads, threads_batch, source) for a load of ``model`` that
        computes on the CPU, served by ``engine`` at ``placement``.

        A model ``threads_models`` names takes its count ("override").
        Otherwise the machine's physical cores, less the reserve: never an
        SMT sibling, since the tokens a core makes are bound by its memory
        and its execution units, which its siblings share; never more than
        the fastest class with ``threads_fast_cores_only``; and never under
        one ("plan"). The count is told to an engine that computes in a
        process of its own (Ollama), which neither the server's affinity nor
        its cgroup quota binds; an engine in the server's own process is
        held to ``server_threads``. The prompt gets the same count. A count
        a tuner measured and kept
        for exactly this model, engine and placement, on this machine, is
        planned instead ("measured"), unless it is past the plan of the CPUs
        read now: a measurement never takes the reserve. The plan is the
        model's, whatever class loads it: the count is pinned to the
        resident. None with "disabled" when the plan is off, "unknown" when
        the CPUs cannot be read: the engine picks its own.
        """
        cfg = self._config
        if not cfg.threads_enabled:
            return None, None, "disabled"
        named = cfg.threads_models.get(model) if model else None
        if named is not None:
            return named, named, "override"
        count = self._plan_count()
        if count is None:
            return None, None, "unknown"
        if model and engine and placement:
            measured = self._measured_threads(model, engine, placement, count)
            if measured is not None:
                return measured[0], measured[1], "measured"
        return count, count, "plan"

    def _plan_count(self) -> int | None:
        """The plan's own count (``plan_threads``), whatever any model's
        override or measurement says, on the machine's CPUs: no quota of the
        server's applies; None when the CPUs cannot be read."""
        return self._count_on(self._machine_topology(), quota=False)

    def server_threads(self) -> int | None:
        """The most CPU threads a computation in the server's own process
        may take (llama.cpp in process): the plan's rule on the server's own
        CPUs -- the physical cores of its affinity less the reserve, no more
        than the fastest class's with ``threads_fast_cores_only`` -- never
        past its cgroup quota, floored, since a thread the quota throttles
        stalls every other at the engine's barriers, and never under one.
        None when the plan is off or the CPUs cannot be read."""
        if not self._config.threads_enabled:
            return None
        return self._count_on(self._cpu_topology(), quota=True)

    def _count_on(self, topology: Any, *, quota: bool) -> int | None:
        """The physical cores of ``topology`` less the reserve, within the
        fastest class when asked, within its quota when ``quota``, and never
        under one; None when it names no core."""
        physical = getattr(topology, "physical", None)
        if isinstance(physical, bool) or not isinstance(physical, int) or physical < 1:
            return None
        count = physical - self.thread_reserve(physical)
        if self._config.threads_fast_cores_only:
            fast = sum(1 for core in getattr(topology, "cores", ()) if getattr(core, "perf_class", None) == 0)
            if fast > 0:
                count = min(count, fast)
        limit = getattr(topology, "quota_cpus", None) if quota else None
        if isinstance(limit, (int, float)) and not isinstance(limit, bool) and math.isfinite(limit) and limit > 0:
            count = min(count, int(math.floor(limit)))
        return max(1, count)

    def _measured_threads(self, model: str, engine: str, placement: str, cap: int) -> tuple[int, int] | None:
        """The counts kept for (model, engine, placement) when they were
        measured on this machine and fit within ``cap``; None otherwise,
        and when the store cannot answer."""
        fingerprint = self.cpu_fingerprint()
        if fingerprint is None:
            return None
        try:
            row = self._store.get_thread_optimum(model, engine, placement)
        except Exception as exc:
            logger.debug("Measured thread count for %s unreadable: %s", model, exc)
            return None
        if row is None or row["fingerprint"] != fingerprint:
            return None
        if row["threads"] > cap or row["threads_batch"] > cap:
            return None
        return row["threads"], row["threads_batch"]

    def cpu_fingerprint(self) -> str | None:
        """What a thread count is measured on: the machine's architecture
        and the shape of its cores, as the plan reads them (how many, the
        SMT siblings of each, their classes and where the classes come
        from), hashed; None when the CPUs cannot be read. A count measured
        on another shape is not this machine's, and is not planned."""
        topology = self._machine_topology()
        cores = tuple(getattr(topology, "cores", None) or ())
        if not cores:
            return None
        import hashlib

        uname = getattr(os, "uname", None)
        shape = [uname().machine if callable(uname) else "", str(getattr(topology, "class_source", ""))]
        shape += [f"{len(getattr(core, 'cpus', ()))}:{getattr(core, 'perf_class', '')}" for core in cores]
        return hashlib.sha256("|".join(shape).encode("utf-8")).hexdigest()[:16]

    def thread_candidates(self, model: str | None, fractions: Any) -> tuple[int | None, list[int], str]:
        """The thread counts a tuner's sweep measures for ``model``:
        (base, candidates, source).

        The base is the plan's count ("plan"), the count a load takes while
        nothing is measured, and the sweep's baseline. The candidates are the
        base, each of ``fractions`` (each above 0 and under 1) of it rounded
        up, and the fastest class's core count, each held to [1, base],
        deduplicated, ascending: never past the plan, so no measurement
        takes the reserve. A model the file names
        keeps its count, its only candidate ("override"). The plan off or
        the CPUs unreadable: (None, [], "disabled" or "unknown").
        """
        threads, _batch, source = self.plan_threads(model)
        if threads is None:
            return None, [], source
        if source != "plan":
            return threads, [threads], source
        counts = {threads}
        for fraction in fractions or ():
            if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
                continue
            if math.isfinite(fraction) and 0 < fraction < 1:
                counts.add(math.ceil(fraction * threads))
        cores = getattr(self._machine_topology(), "cores", None) or ()
        fast = sum(1 for core in cores if getattr(core, "perf_class", None) == 0)
        if fast > 0:
            counts.add(fast)
        return threads, sorted({max(1, min(threads, count)) for count in counts}), "plan"

    def record_thread_optimum(
        self,
        model: str,
        engine: str,
        placement: str | None,
        *,
        threads: int,
        threads_batch: int,
        tg: float,
        base_tg: float,
    ) -> bool:
        """Keep the counts a tuner measured for ``model`` served by
        ``engine`` at ``placement``, with this machine's fingerprint; True
        when written. Nothing is written for no model, engine or placement,
        a count that is not a whole number from one to the plan of the CPUs
        read now (a measurement never takes the reserve), a rate that is not
        a positive finite number, or CPUs that cannot be read."""
        if not model or not engine or not placement:
            return False
        cap = self._plan_count()
        fingerprint = self.cpu_fingerprint()
        if cap is None or fingerprint is None:
            return False
        for count in (threads, threads_batch):
            if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= cap:
                return False
        for rate in (tg, base_tg):
            if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
                return False
        self._store.record_thread_optimum(
            model, engine, placement, fingerprint, threads, threads_batch, float(tg), float(base_tg)
        )
        return True

    def plan_background(self) -> BackgroundPlan:
        """The background workers' budget, apart from any engine's plan.

        The reserve is the plan's own count of physical cores
        (thread_reserve), taken from the top of the reserve's order, so the
        cores the user's programs are scheduled on first stay theirs. The
        workers get every CPU of the other cores, SMT siblings included
        (they yield the CPU to anything else), and one worker per such core,
        within the file's ceiling and the cgroup quota less the reserve (a
        worker charges the quota the server's threads draw on), never under
        one ("plan"). Off, no worker ("disabled"). CPUs unreadable, one
        worker on the CPUs it inherits ("unknown").
        """
        cfg = self._config
        common = {
            "in_flight": cfg.threads_background_in_flight,
            "idle_s": cfg.threads_background_idle_s,
            "held_retry_s": cfg.threads_background_held_retry_s,
            "parse_expansion": dict(cfg.threads_background_parse_expansion),
        }
        if not cfg.threads_background_enabled:
            return BackgroundPlan(workers=0, cpus=(), reserved=(), source="disabled", **common)
        topology = self._cpu_topology()
        cores = tuple(getattr(topology, "cores", None) or ())
        if not cores:
            return BackgroundPlan(workers=1, cpus=(), reserved=(), source="unknown", **common)
        reserve = self.thread_reserve(len(cores))
        ordered = sorted(cores, key=_reserve_order)
        kept = ordered[reserve:]
        workers = min(cfg.threads_background_max_workers, len(kept))
        quota = getattr(topology, "quota_cpus", None)
        if isinstance(quota, (int, float)) and not isinstance(quota, bool) and math.isfinite(quota) and quota > 0:
            workers = min(workers, int(math.floor(quota)) - reserve)
        return BackgroundPlan(
            workers=max(1, workers),
            cpus=tuple(sorted(cpu for core in kept for cpu in core.cpus)),
            reserved=tuple(sorted(cpu for core in ordered[:reserve] for cpu in core.cpus)),
            source="plan",
            **common,
        )

    def threads_state(self) -> dict[str, Any]:
        """The CPU threads for the status surface, each value the call the
        admissions make: the plan's own count and its source (plan_threads
        with no model), the machine's physical cores, the reserve
        (thread_reserve), the fast-cores switch, the server's cgroup quota
        and its own cap (server_threads), the background's budget
        (plan_background) with the CPUs it leaves to the user, the count and
        placement pinned to each resident, and the counts kept in the store,
        newest first, each said measured on the CPUs read now or not. A
        reading that fails makes this section unavailable, not the status."""
        try:
            return self._threads_state()
        except Exception as exc:
            logger.debug("Threads state unavailable: %s", exc)
            return {"available": False}

    def _threads_state(self) -> dict[str, Any]:
        cfg = self._config
        topology = self._machine_topology()
        server = self._cpu_topology()
        physical = getattr(topology, "physical", None)
        if isinstance(physical, bool) or not isinstance(physical, int) or physical < 1:
            physical = None
        threads, threads_batch, source = self.plan_threads(None)
        background = self.plan_background()
        with self._cache_lock:
            held = sorted((model, list(pin)) for model, pin in self._thread_pins.items())
        pins = [
            {"model": model, "threads": pin[0], "threads_batch": pin[1], "placement": pin[3], "seen": bool(pin[2])}
            for model, pin in held
        ]
        fingerprint = self.cpu_fingerprint()
        shown = ("model", "engine", "placement", "threads", "threads_batch", "tg", "base_tg", "measured_at")
        measured = []
        for row in self._store.thread_optima(cfg.threads_status_limit):
            entry = {key: row[key] for key in shown}
            entry["this_machine"] = fingerprint is not None and row["fingerprint"] == fingerprint
            measured.append(entry)
        return {
            "available": True,
            "enabled": bool(cfg.threads_enabled),
            "plan": {"threads": threads, "threads_batch": threads_batch, "source": source},
            "physical": physical,
            "reserve": self.thread_reserve(physical) if physical is not None else None,
            "fast_cores_only": bool(cfg.threads_fast_cores_only),
            "quota_cpus": getattr(server, "quota_cpus", None) if server is not None else None,
            "server_threads": self.server_threads(),
            "background": {
                "enabled": bool(cfg.threads_background_enabled),
                "workers": background.workers,
                "cpus": list(background.cpus),
                "reserved": list(background.reserved),
                "source": background.source,
                "in_flight": background.in_flight,
                "idle_s": background.idle_s,
                "held_retry_s": background.held_retry_s,
            },
            "pins": pins,
            "measured": measured,
        }

    def caller_class(self, caller: str | None) -> str:
        """The admission class the file gives ``caller``; an unnamed one is
        a user."""
        return _current_governor(self)._config.class_of(caller)

    def _ollama_sequences(self) -> int | None:
        """How many sequences Ollama keeps a KV cache for at once, as the
        operator names it (ollama_limits.num_parallel); None unnamed, or
        named as no parallelism at all."""
        named = self._config.ollama_num_parallel
        if isinstance(named, bool) or not isinstance(named, int) or named < 1:
            return None
        return named

    def _release_pins(self, loaded: list[LoadedModelView]) -> None:
        """Drop the pin of a model the loaded view has shown and shows no more.

        A view that has not shown the model yet only lags its load.
        """
        names = {view.name for view in loaded if view.resident}
        with self._cache_lock:
            for model, (layers, num_ctx, seen) in list(self._pins.items()):
                if model in names:
                    self._pins[model] = (layers, num_ctx, True)
                elif seen:
                    del self._pins[model]
            for model, pin in list(self._thread_pins.items()):
                if model in names:
                    pin[2] = True
                elif pin[2]:
                    del self._thread_pins[model]
        # The class a model was loaded for ends the same way.
        with self._queue_cond:
            for model, owner in list(self._owners.items()):
                if model in names:
                    owner[1] = True
                elif owner[1]:
                    del self._owners[model]

    def estimate_kv_cache_gb(
        self, num_ctx: int | None, model: str | None = None
    ) -> float:
        """KV-cache increment as a function of the requested num_ctx.

        ``kv_coefficient`` is GiB per 1024 tokens (DI-4); 0 tokens or an
        unset request cost nothing. The optional ``model`` kwarg
        routes through :meth:`resolve_kv_coefficient`; without it the
        global coefficient applies, byte-compatible with every
        pre-existing caller.
        """
        if not num_ctx or num_ctx <= 0:
            return 0.0
        return (float(num_ctx) / 1024.0) * self.resolve_kv_coefficient(model)

    def estimate_model_vram_gb(
        self, model: str, digest: str | None = None, engine: str | None = None
    ) -> tuple[float | None, str]:
        """Best-available weight-cost estimate for one model, with basis.

        Order: live S1 observation (when the cached snapshot holds the
        model with a positive size_vram) > learned cost (the adapt store)
        > the S3 static table via registry metadata, the serving backend
        asked first > GGUF file size as a floor > (None, "unknown") --
        never "too large" (Section 3.1).
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

        for backend in self._backend_order(model, engine):
            est, basis = self._estimate_from_backend(backend, model)
            if est is not None:
                return est, basis
        return None, "unknown"

    def resolve_weights_override(
        self, model: str | None
    ) -> float | None:
        """Per-model weights-residency override in GiB.

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

    def _serving_backend(self, model: str | None, engine: str | None = None) -> Any:
        """The backend that will serve ``model``, or None when none can be told.

        An engine the caller names wins, and a name no backend carries
        serves nothing. Otherwise the registry's cached resolution answers
        (``cached_backend``: the funnels' last ``resolve_backend`` for the
        model, read without a health check or a probe, so admission calls
        no engine for it); before the first resolution, or from a registry
        without that cache, nothing answers and the backends are asked in
        their registration order, as before. Never raises.
        """
        registry = self._resolve_registry()
        if registry is None or not model:
            return None
        if engine:
            try:
                for backend in registry.backends():
                    if str(getattr(backend, "name", "")) == engine:
                        return backend
            except Exception as exc:
                logger.debug("Registry backends() failed: %s", exc)
            return None
        cached = getattr(registry, "cached_backend", None)
        if not callable(cached):
            return None
        try:
            return cached(model)
        except Exception as exc:
            logger.debug("cached_backend(%s) failed: %s", model, exc)
            return None

    def _backend_order(self, model: str | None, engine: str | None = None) -> list[Any]:
        """Every registered backend, the one that will serve ``model`` first."""
        registry = self._resolve_registry()
        if registry is None:
            return []
        try:
            backends = list(registry.backends())
        except Exception as exc:
            logger.debug("Registry backends() failed: %s", exc)
            return []
        serving = self._serving_backend(model, engine)
        if serving is None:
            return backends
        return [serving] + [b for b in backends if b is not serving]

    def _declared_cost(self, backend: Any, model: str | None) -> dict[str, Any] | None:
        """What ``backend`` declares one request to ``model`` costs, or None.

        None -- every engine today -- is a token generator, priced by the
        governor. A declaration that does not hold together (an unknown
        kind, a negative, non-finite or non-numeric figure, a KV flag that
        is not a boolean, a generator declared without a KV cache) is set
        aside for the same answer.
        """
        declare = getattr(backend, "cost_model", None) if backend is not None else None
        if not callable(declare) or not model:
            return None
        try:
            raw = declare(model)
        except Exception as exc:
            logger.debug("cost_model(%s) failed: %s", model, exc)
            return None
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            logger.debug("cost_model(%s) is not a mapping; ignored", model)
            return None
        kind = raw.get("kind", "generator")
        weights = raw.get("weights_gb")
        state = raw.get("state_gb", 0.0)
        kv = raw.get("kv", kind == "generator")

        def _figure(value: Any) -> bool:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
            try:
                return math.isfinite(value) and value >= 0.0
            except (OverflowError, ValueError):
                return False

        if (
            kind not in _MODEL_KINDS
            or not isinstance(kv, bool)
            or (kind == "generator" and not kv)
            or not _figure(state)
            or (weights is not None and not _figure(weights))
        ):
            logger.debug("cost_model(%s) does not hold together; ignored: %r", model, raw)
            return None
        return {
            "kind": kind,
            "weights_gb": float(weights) if weights is not None else None,
            "state_gb": float(state),
            "kv": kv,
        }

    def _load_cost_gb(self, model: str, num_ctx: int | None, snapshot: Any) -> float:
        """The GiB a load of ``model`` at ``num_ctx`` costs, priced as admit
        prices it: the operator's override over a declared figure over the
        estimate, nothing for a resident model, its own KV coefficient when
        it keeps a KV cache, and any declared per-request state."""
        resident = {v.name for v in getattr(snapshot, "loaded", []) if v.resident}
        declared = self._declared_cost(self._serving_backend(model), model)
        weights = 0.0
        if model not in resident:
            estimate, _basis = self.estimate_model_vram_gb(model)
            if declared is not None and declared["weights_gb"] is not None:
                estimate = declared["weights_gb"]
            override = self.resolve_weights_override(model)
            if override is not None:
                estimate = override
            weights = estimate or 0.0
        kv = 0.0
        if declared is None or declared["kv"]:
            kv = self.estimate_kv_cache_gb(num_ctx, model)
        state = declared["state_gb"] if declared is not None else 0.0
        return weights + kv + state

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
                        size_bytes=int(getattr(m, "size", 0) or 0),
                    )
                )
            except Exception as exc:
                logger.debug("Skipping malformed ps entry: %s", exc)
        return views, True

    def _attribute_pending(self, loaded: list[LoadedModelView]) -> None:
        """DI-9: write learned costs for pending post-load attributions.

        The cost learned is the total the engine reports for the model when
        it reports one: a split model's VRAM part alone would price its next
        load below what it takes.
        """
        with self._cache_lock:
            if not self._pending_attribution:
                return
            pending = dict(self._pending_attribution)
        for view in loaded:
            if view.name in pending and view.resident:
                try:
                    self._store.record_model_cost(
                        view.name,
                        view.digest,
                        view.size_bytes or view.size_vram_bytes,
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
    ) -> tuple[list[BackendResidentView], bool, bool, list[str]]:
        """The S2 read: backend-resident models not in the S1 view.

        Returns (views, s2_answered, s3_used, unread). Only backends exposing
        the in-process ``_loaded_models`` dict idiom contribute entries; the
        Ollama backend's loaded set IS the S1 view and is never double
        counted. ``unread`` names every backend that exposes no such set, so
        the snapshot can say where it did not look.
        """
        registry = self._resolve_registry()
        if registry is None:
            return [], False, False, []
        try:
            backends = list(registry.backends())
        except Exception as exc:
            logger.debug("S2 registry read failed: %s", exc)
            return [], False, False, []
        views: list[BackendResidentView] = []
        s3_used = False
        unread: list[str] = []
        for backend in backends:
            resident = getattr(backend, "_loaded_models", None)
            if not isinstance(resident, dict):
                # A backend the governor cannot read is NAMED, not skipped.
                # Skipping it made "no resident models here" and "never
                # looked here" the same snapshot, and an admission decided
                # on that snapshot could be confidently wrong about a card
                # the external server had already filled.
                unread.append(str(getattr(backend, "name", "unknown")))
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
        return views, True, s3_used, unread

    def _probe_capacity_gb(self) -> float | None:
        """Total VRAM in GiB from the probe, or None when it cannot say.

        Only a strictly positive reading becomes a capacity. The collector
        reports -1.0 for anything it could not read, and a probe that raises
        is an absent reading too: in both cases the capacity stays unknown,
        which keeps the VRAM half fail-open. Turning either into 0.0 would
        declare a card with no memory and refuse everything.
        """
        if self._vram_probe is None:
            return None
        try:
            total_mb = float(self._vram_probe())
        except Exception as exc:
            logger.debug("VRAM probe failed: %s", exc)
            return None
        if total_mb <= 0.0:
            return None
        return total_mb / 1024.0

    def _build_snapshot(self) -> ResourceSnapshot:
        now = self._clock()
        sources: list[str] = []

        loaded, s1_answered = self._read_s1()
        if s1_answered:
            sources.append("S1")
            self._attribute_pending(loaded)
            self._release_pins(loaded)

        resident, s2_answered, s3_used, unread = self._read_s2(
            {view.name for view in loaded}
        )
        if s2_answered:
            sources.append("S2")
        if s3_used:
            sources.append("S3")
        # Where the governor did NOT look is part of the provenance too.
        for backend_name in unread:
            sources.append(f"S2-unread:{backend_name}")

        configured = self._config.total_vram_gb
        hardware = self._hardware()
        # The probe is a fallback, not an override: an operator who wrote a
        # figure down is not overruled by a sensor, and on a configured host
        # no card is read at all.
        placement = None
        probed = None
        if configured is None:
            if self._probe_injected:
                probed = self._probe_capacity_gb()
            elif hardware is not None:
                placement = self._read_placement(hardware)
                if placement is not None and (placement.capacity_mib or 0.0) > 0.0:
                    probed = placement.capacity_mib / 1024.0
        declared = configured if configured is not None else probed
        declared_source = "config" if configured is not None else "probe"

        learned_ceiling = None
        try:
            learned_ceiling = self._store.get_learned_ceiling()
        except Exception as exc:
            logger.debug("Learned ceiling read failed: %s", exc)
        if configured is not None:
            sources.append("S4-capacity-config")
        elif probed is not None:
            sources.append("S4-capacity-probe")
        if learned_ceiling is not None:
            sources.append("S4-capacity-learned")

        if declared is not None and learned_ceiling is not None:
            capacity: float | None = min(declared, learned_ceiling)
            capacity_source = f"{declared_source}+learned"
        elif declared is not None:
            capacity = declared
            capacity_source = declared_source
        elif learned_ceiling is not None:
            capacity = learned_ceiling
            capacity_source = "learned"
        else:
            capacity = None
            capacity_source = "unknown"

        ram_mb = ram_total_mb = 0.0
        try:
            ram_mb, ram_total_mb = _read_meminfo_mb(self._meminfo_path)
        except Exception as exc:  # pragma: no cover - the reader never raises
            logger.debug("S4 RAM read failed: %s", exc)
        if ram_mb > 0:
            sources.append("S4-ram")

        in_use = sum(v.size_vram_gb for v in loaded) + sum(
            (v.estimated_gb or 0.0) for v in resident
        )

        # The cards the capacity was summed from, and the memory other
        # programs hold on them: what the cards report used, less what the
        # engines declare -- only when the reading was taken since the
        # engines last loaded or released a model. An older reading set
        # against today's holdings would count a model just loaded as
        # someone else's absence, or one just evicted as someone else's
        # memory: what others held at the last paired reading is carried
        # instead, until the cards are read again. An engine that declares
        # more than the card shows leaves nobody else's memory, never less;
        # a reading too old to describe the cards leaves it unknown.
        devices: list[dict[str, Any]] = []
        others = 0.0
        used_age: float | None = None
        carried = False
        if placement is not None and probed is not None:
            devices = [d.to_dict() for d in placement.devices]
            if placement.used_mib is not None:
                if getattr(placement, "used_current", True):
                    others = max(0.0, placement.used_mib / 1024.0 - in_use)
                    self._others_carry = others
                elif self._others_carry is not None:
                    others = self._others_carry
                    carried = True
                used_age = placement.used_age_s
            else:
                self._others_carry = None

        host_pressure = None
        if hardware is not None:
            try:
                host_pressure = hardware.pressure()
            except Exception as exc:
                logger.debug("Host pressure read failed: %s", exc)
        memory_pressure = self._update_memory_pressure(host_pressure)

        # A capacity unknown because the machine has no card, as the profile
        # knows it, is no unreadable card: the background then loads into
        # the RAM alone.
        cards_absent = False
        if capacity is None and hardware is not None:
            absent = getattr(hardware, "cards_absent", None)
            if callable(absent):
                try:
                    cards_absent = absent() is True
                except Exception as exc:
                    logger.debug("Card presence unreadable: %s", exc)

        if capacity is None:
            vram_status = "disabled_capacity_unknown"
            available: float | None = None
            if not self._capacity_warning_emitted:
                logger.warning(
                    "Resource governor: total VRAM capacity unknown "
                    "(total_vram_gb is null, the hardware profile reads no "
                    "card and no ceiling is learned); the VRAM half of "
                    "measurement reports disabled and any future admission "
                    "stays fail-open (spec Section 3.1). Set total_vram_gb "
                    "in resource_governor.yaml, or name the cards in "
                    "hardware_profile.yaml, on the host."
                )
                self._capacity_warning_emitted = True
        else:
            vram_status = "ok"
            available = max(0.0, capacity - in_use - others)

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
            ram_total_mb=ram_total_mb,
            devices=devices,
            device_count=len(devices),
            vram_others_gb=others,
            vram_used_age_s=used_age,
            vram_others_carried=carried,
            host_pressure=host_pressure,
            memory_pressure_active=memory_pressure,
            cards_absent=cards_absent,
        )

    # -- the machine: cards, pressure, the RAM reserve ------------------------

    def _hardware(self) -> Any:
        """The hardware profile, or None when there is none to ask.

        An injected one (or an injected None) answers. An injected VRAM
        probe stands alone: no card is read beside it. Otherwise the
        process-wide profile is resolved once, lazily; a window without the
        profile module is a governor without one.
        """
        if self._hardware_arg is not _UNSET:
            return self._hardware_arg
        if self._probe_injected:
            return None
        if self._hardware_resolved is _UNSET:
            try:
                from opti_oignon.hardware_profile import get_hardware_profile

                self._hardware_resolved = get_hardware_profile()
            except Exception as exc:
                logger.debug("Hardware profile unavailable: %s", exc)
                self._hardware_resolved = None
        return self._hardware_resolved

    def _read_placement(self, hardware: Any) -> Any:
        try:
            return hardware.placement()
        except Exception as exc:
            logger.debug("Hardware placement read failed: %s", exc)
            return None

    def _update_memory_pressure(self, host_pressure: Any) -> bool:
        """Whether the kernel reports memory pressure, with hysteresis.

        Entered when the share of time some task stalled on memory over the
        last ten seconds reaches the enter mark, left only once it falls
        below the exit mark, so a reading that wavers between the two does
        not flip the reserve from one admission to the next. Unknown, or
        the signal disabled, is no pressure.
        """
        cfg = self._config
        some = None
        if cfg.host_pressure_enabled and isinstance(host_pressure, dict):
            memory = host_pressure.get("memory") or {}
            some = (memory.get("some") or {}).get("avg10")
        with self._cache_lock:
            if not isinstance(some, (int, float)) or isinstance(some, bool):
                self._shared.memory_pressure_active = False
            elif some >= cfg.host_pressure_memory_enter:
                self._shared.memory_pressure_active = True
            elif some < cfg.host_pressure_memory_exit:
                self._shared.memory_pressure_active = False
            return self._shared.memory_pressure_active

    def _vram_margin_gb(self, snapshot: Any) -> float:
        """The safety margin, kept on every card the capacity was summed from."""
        count = getattr(snapshot, "device_count", 0) or 0
        return self._config.safety_margin_gb * max(1, int(count))

    def ram_reserve_state(self, snapshot: Any) -> dict[str, Any]:
        """The RAM a split leaves to the rest of the machine, and why.

        A number in ``offload.ram_reserve_gb`` is that number, whatever the
        machine and its pressure. None sizes it: ``ram_reserve_fraction`` of
        the total RAM, held between the floor and the ceiling (the ceiling
        when the total cannot be read), multiplied by the pressure factor
        while the kernel reports memory pressure, and never more than the
        machine holds.
        """
        cfg = self._config
        known = getattr(snapshot, "host_pressure", None) is not None
        fixed = cfg.offload_ram_reserve_gb
        if fixed is not None:
            return {
                "mode": "fixed",
                "gb": fixed,
                "base_gb": fixed,
                "pressure_applied": False,
                "pressure_known": known,
            }
        total_gb = (getattr(snapshot, "ram_total_mb", 0.0) or 0.0) / 1024.0
        if total_gb > 0.0:
            base = min(
                cfg.ram_reserve_ceiling_gb,
                max(cfg.ram_reserve_floor_gb, cfg.ram_reserve_fraction * total_gb),
            )
        else:
            base = cfg.ram_reserve_ceiling_gb
        applied = cfg.host_pressure_enabled and bool(
            getattr(snapshot, "memory_pressure_active", False)
        )
        reserve = base * cfg.ram_reserve_pressure_factor if applied else base
        if total_gb > 0.0:
            reserve = min(reserve, total_gb)
        return {
            "mode": "adaptive",
            "gb": round(reserve, 3),
            "base_gb": round(base, 3),
            "pressure_applied": applied,
            "pressure_known": known,
        }

    def effective_ram_reserve_gb(self, snapshot: Any) -> float:
        """The RAM reserve a split priced on ``snapshot`` leaves untouched."""
        return float(self.ram_reserve_state(snapshot)["gb"])

    def hardware_state(self) -> dict[str, Any]:
        """The hardware profile's view for the status surface."""
        hardware = self._hardware()
        if hardware is None:
            return {"available": False}
        try:
            state = dict(hardware.to_dict())
        except Exception as exc:
            logger.debug("Hardware profile view failed: %s", exc)
            return {"available": False}
        state["available"] = True
        return state

    # -- learning passthroughs (the rules hold; no production path calls them yet) --

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
        """Append to the bounded recent-decisions ring (the admission path's writer)."""
        self._store.record_decision(
            caller,
            model,
            requested_ctx,
            admitted_ctx,
            decision,
            reason,
            ring_size=self._config.decisions_ring_size,
        )

    # -- Runtime backpressure -- the pressure signal (Section 5) --------------

    def pressure_state(self) -> dict[str, Any]:
        """The R-02 pressure signal, the shape the status API
        surfaces.

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

    # -- R-03 limit management -- the advisory seat (Section 6) ---------------

    def ollama_limits_advisory(self) -> dict[str, Any]:
        """The R-03 external-Ollama advisory in the status-API shape.

        Thin delegation to :func:`compute_ollama_limits_advisory` with
        this governor's config; the seat the status surface
        reads. The startup security checklist consumes the pure
        function directly (advisory-only in all modes, never blocking
        startup -- the standing precedent).
        """
        return compute_ollama_limits_advisory(self._config)

    def _pressure_from_snapshot(
        self, snapshot: ResourceSnapshot
    ) -> dict[str, Any]:
        cfg = self._config
        # What other programs hold on the cards is not the governor's to
        # use: the effective capacity is what is left of it.
        others = getattr(snapshot, "vram_others_gb", 0.0) or 0.0
        effective = snapshot.capacity_gb
        if effective is not None:
            effective = max(0.0, effective - others)
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
        with self._cache_lock:
            overridden = self._shared.keep_alive_original is not None
        return {
            "level": level,
            "ratio": round(ratio, 4) if ratio is not None else None,
            "effective_capacity_gb": effective,
            "in_use_gb": round(snapshot.vram_in_use_gb, 3),
            "others_gb": round(others, 3),
            "soft_threshold": cfg.pressure_soft_threshold,
            "hard_threshold": cfg.pressure_hard_threshold,
            "refusal_rate": round(refusal_rate, 4),
            "refusals_in_window": refusals,
            "decisions_in_window": decisions,
            "refusal_window_s": cfg.pressure_refusal_window_s,
            "keep_alive_overridden": overridden,
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
        shared = self._shared
        with self._cache_lock:
            if level in ("soft", "hard"):
                if shared.pressure_soft_since is None:
                    shared.pressure_soft_since = now
                    return
                sustained = (
                    now - shared.pressure_soft_since
                    >= max(0.0, cfg.pressure_sustain_s)
                )
                if not sustained or shared.keep_alive_original is not None:
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
                        shared.keep_alive_original = original
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
            shared.pressure_soft_since = None
            if shared.keep_alive_original is None:
                return
            original = shared.keep_alive_original
            shared.keep_alive_original = None
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

    # -- The admission gate (Section 4) ---------------------------------------

    def admit(
        self,
        model: str,
        requested_ctx: int | None = None,
        caller: str = "chat",
        extra_models: list[str] | None = None,
        digest: str | None = None,
        engine: str | None = None,
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
          eviction (the eviction act itself is handled by evict_model;
          Ollama's own LRU carries it meanwhile, the Section 12 posture).
        - The requested ctx is clamped to the model's context window
          (ModelLimits stays the authority), then stepped down the config
          ladder to the per-caller floor; callers without a floor
          (benchmark, AGT, direct) are never downsized.
        - Capacity unknown: the VRAM half fails open (admit, the 3.1
          arbitration) while the RAM half still applies (known weight cost
          exceeding MemAvailable refuses).
        - Partial offload: a cost the GPU cannot hold even after eviction
          is admitted split between VRAM and system RAM when the VRAM free
          now and the RAM above the reserve hold it together. A split never
          waits on an eviction, never leaves the GPU less than the minimum
          share, and is never priced against an unreadable RAM. The prefer
          key orders the attempts: "context" splits each ctx before stepping
          down the ladder, "speed" steps down on the GPU alone and splits at
          the last step. A refusal reached after a split was priced names
          both shortfalls.
        - The KV cost of a ctx comes from the model's own geometry when its
          engine describes it (kv_geometry), under any operator override.
        - The engine that will serve (named by the caller, else resolved by
          the registry) is asked first, and may declare what a request
          costs (InferenceBackend.cost_model): a model without a KV cache
          is charged none and is never stepped down for memory, a declared
          per-request state is charged once, declared weights replace the
          estimate and an operator override replaces both.
        - A resident model asked no more context than it holds is admitted
          at the context it holds and charged nothing; asked more, it is a
          reload, its own memory credited and its whole cost charged; its
          loaded context unknown, it is charged the KV of the call as
          before.
        - A split is placed layer by layer when the file that serves the
          model says what each layer weighs (tensor_table): the last layers
          that fit the VRAM free now, each with its KV, go to the GPU with
          the cost the tensors and the KV do not explain; num_gpu is their
          count, and the cost is never priced under the tensors. A split
          slower than split_speed allows does not hold. Without a plan (no
          table, no context told, or weights the operator or the engine
          names) the split is the even one and num_gpu stays None. A
          resident model's decision at the context its split load used
          carries the count that load pinned.
        - Every decision is recorded in the ring; under soft-or-worse
          pressure (Section 5) an admitted decision carries the keep_alive
          override the funnels apply for that call.
        - Who asks: every decision carries the caller's admission class. The
          background never evicts: the background gate may hold it first;
          it counts no eviction credit, so it is never granted on condition
          of an eviction; it never reloads a resident for more context (a
          resident whose context the dynamic stage or the caller's floor
          accepts serves it instead), never splits unless the file allows
          it, and is refused a load whose fit the free memory cannot show
          (a card unreadable, the cost unknown, no context to price), since
          the engine's own LRU would then evict for it. A call on a resident
          at the context it holds loads nothing and stays admitted.
        - The background prices a load against the memory free now less
          every load admitted, of any class, that the loaded view does not
          show yet (pending_loads); a call on a model whose load is pending
          joins that load at its context and loads nothing. Its decisions
          are taken one at a time, and one that sees a load admitted
          meanwhile refuses itself, by a reason waiting lifts. A background
          load told no context is priced, and loaded, at the context the
          interactive class was last admitted at for the model, else the
          model's output reserve, else the ladder's smallest step. On a
          machine the profile knows has no card it loads into the RAM free
          above the reserve. The foreground prices as before.
        """
        klass = self._config.class_of(caller)
        if klass == _BACKGROUND:
            with self._background_lock:
                decision = self._decide(
                    model, requested_ctx, caller, extra_models, digest, engine, self._pending_view()
                )
        else:
            decision = self._decide(model, requested_ctx, caller, extra_models, digest, engine, None)
        if decision.admitted and klass == _INTERACTIVE:
            # Held from its admission: the thread that holds the ticket may
            # take it a moment later, and note_held hands it over.
            now = self._clock()
            with self._queue_cond:
                self._live_admitted(now)
                self._admitted[decision.ticket_id] = (decision, now)
        return decision

    def _decide(
        self,
        model: str,
        requested_ctx: int | None,
        caller: str,
        extra_models: list[str] | None,
        digest: str | None,
        engine: str | None,
        view: tuple[int, list[_PendingLoad]] | None,
    ) -> AdmissionDecision:
        """The body of admit(). ``view`` is the background's reading of the
        loads admitted and not yet seen, (the count of loads ever claimed,
        the loads); None for the other classes, which deduct nothing."""
        ticket_id = uuid.uuid4().hex[:12]
        klass = self._config.class_of(caller)
        background = klass == _BACKGROUND
        pending = view[1] if view is not None else []
        # A model loads once, however many callers asked for it: each model
        # counts once, at the most any of its loads adds.
        largest: dict[str, tuple[float, float]] = {}
        for p in pending:
            vram, ram = largest.get(p.model, (0.0, 0.0))
            largest[p.model] = (max(vram, p.vram_gb), max(ram, p.ram_gb))
        pending_vram = sum(vram for vram, _ in largest.values())
        pending_ram = sum(ram for _, ram in largest.values())

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
                admission_class=klass,
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
                admission_class=klass,
            )

        if background:
            hold = self._background_hold()
            if hold is not None:
                decision = AdmissionDecision(
                    admitted=False,
                    model=model,
                    num_ctx=None,
                    action="refuse",
                    reason=f"background_held:{hold}",
                    ticket_id=ticket_id,
                    caller=caller,
                    requested_ctx=requested_ctx,
                    admission_class=klass,
                    held_by=hold,
                )
                self._record_admission(decision)
                return decision

        snapshot = self.get_snapshot_fast()
        provenance = list(snapshot.sources)

        # The pressure signal rides every admission; a
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

        # The engine that will serve, and what it declares one request costs:
        # None, every engine today, is a token generator.
        serving = self._serving_backend(model, engine)
        engine_name = (str(getattr(serving, "name", "")) or None) if serving is not None else None
        declared = self._declared_cost(serving, model)
        kv_charged = declared["kv"] if declared is not None else True
        state_gb = declared["state_gb"] if declared is not None else 0.0

        # Resident wherever it sits: a model split between VRAM and RAM, or
        # held in RAM alone, loads nothing more either -- unless the call
        # asks more context than it holds (below).
        resident_view = next(
            (v for v in snapshot.loaded if v.name == model and v.resident), None
        )
        loaded_names = {v.name for v in snapshot.loaded if v.resident}
        already_loaded = resident_view is not None
        load_expected = not already_loaded

        def _full_weights() -> float | None:
            # A declared figure replaces the estimate; an operator-named
            # weights-residency override (the MoE active-params gap)
            # replaces both; absent, the estimator answer stands.
            estimate, _basis = self.estimate_model_vram_gb(model, digest, engine=engine)
            if declared is not None and declared["weights_gb"] is not None:
                estimate = declared["weights_gb"]
            override = self.resolve_weights_override(model)
            return override if override is not None else estimate

        # Weights cost (4.2): zero when already resident; unknown is never
        # "too large" (3.1) and contributes zero to the fit.
        weights_gb: float | None = 0.0 if already_loaded else _full_weights()

        extra_gb = 0.0
        extras_pending = False
        extras_unknown = False
        loading_extras: list[str] = []
        for extra in extra_models or []:
            if not extra or extra == model or extra in loaded_names:
                continue
            load_expected = True
            extras_pending = True
            if extra not in loading_extras:
                loading_extras.append(extra)
            extra_est, _eb = self.estimate_model_vram_gb(extra)
            # The same override seam for the extra models.
            extra_override = self.resolve_weights_override(extra)
            if extra_override is not None:
                extra_est = extra_override
            if extra_est is not None:
                extra_gb += extra_est
            else:
                extras_unknown = True

        effective_ctx = self._clamp_ctx(model, requested_ctx)

        # The model's geometry and KV coefficient, resolved once for every
        # candidate below (an engine is asked at most once per decision); a
        # model without a KV cache needs neither.
        geometry = self.kv_geometry(model, engine=engine) if kv_charged else None
        kv_override = self._kv_override(model)
        kv_coefficient = (
            kv_override
            if kv_override is not None
            else self._coefficient_from(geometry)
        )

        def _kv(ctx: int | None) -> float:
            if not kv_charged or not ctx or ctx <= 0:
                return 0.0
            return (float(ctx) / 1024.0) * kv_coefficient

        # A resident model whose loaded context is known holds that context.
        # Asked no more, it loads nothing and costs nothing, whatever the
        # card's occupancy, which it is itself part of; it is admitted at
        # the context it holds, since another can make the engine reload.
        # Asked more, it is a reload: the engine frees all it holds, VRAM
        # and RAM, then allocates the whole cost -- weights taken as what it
        # holds less the KV of its loaded context, unless an operator or an
        # engine names them.
        credit = 0.0
        ram_credit = 0.0
        loaded_ctx = resident_view.context_length if resident_view is not None else None
        holds = already_loaded and isinstance(loaded_ctx, int) and loaded_ctx > 0
        main_kv = kv_charged
        reload = False

        def _holds_decision(action: str, reason: str, at: Any = _UNSET, joins: Any = _UNSET) -> AdmissionDecision:
            # Served at the context the resident holds, with the CPU threads
            # its load was told and where that load computes; or at ``at``,
            # the context of a pending load the call joins, with the threads
            # and the placement that load was admitted with (``joins``):
            # Ollama would reload the model for any other count.
            ctx = loaded_ctx if at is _UNSET else at
            if joins is _UNSET:
                pinned = self.pinned_threads(model)
                told = None if pinned is None else (pinned[0], pinned[1], self.pinned_placement(model))
            else:
                told = joins
            held = AdmissionDecision(
                admitted=True,
                model=model,
                num_ctx=ctx,
                num_gpu=self.pinned_layers(model, ctx),
                keep_alive=ka_override,
                action=action,
                reason=reason,
                provenance=provenance,
                ticket_id=ticket_id,
                caller=caller,
                requested_ctx=requested_ctx,
                load_expected=False,
                engine=engine_name,
                num_parallel=self._ollama_sequences(),
                cost_gb=0.0,
                admission_class=klass,
            )
            if told is not None and told[0] is not None:
                held.threads, held.threads_batch, held.threads_source = told[0], told[1], "pinned"
                held.placement = told[2]
            self._record_admission(held)
            return held

        def _carry_threads(decision: AdmissionDecision, on_cpu: bool) -> None:
            # A resident served as it is keeps the threads its load was told,
            # and says where that load computes; a load computes with the
            # plan for where it computes when it computes on the CPU, and
            # otherwise the engine picks its own.
            if already_loaded and not reload:
                told = self.pinned_threads(model)
                if told is not None:
                    decision.threads, decision.threads_batch, decision.threads_source = told[0], told[1], "pinned"
                    decision.placement = self.pinned_placement(model)
                return
            if not on_cpu:
                return
            if not decision.partial_offload:
                decision.placement = "cpu"
            elif isinstance(decision.gpu_layers, int) and not isinstance(decision.gpu_layers, bool):
                decision.placement = f"split:{decision.gpu_layers}"
            threads, threads_batch, source = self.plan_threads(model, engine_name, decision.placement)
            if threads is not None:
                decision.threads, decision.threads_batch, decision.threads_source = threads, threads_batch, source

        def _refuse(reason: str) -> AdmissionDecision:
            refused = AdmissionDecision(
                admitted=False,
                model=model,
                num_ctx=None,
                action="refuse",
                reason=reason,
                provenance=provenance,
                ticket_id=ticket_id,
                caller=caller,
                requested_ctx=requested_ctx,
                engine=engine_name,
                admission_class=klass,
            )
            self._record_admission(refused)
            return refused

        if holds and (effective_ctx is None or effective_ctx <= loaded_ctx):
            if not extras_pending:
                return _holds_decision("admit", "fits_resident")
            # Only the extra models load: the resident keeps its context.
            effective_ctx = loaded_ctx
            main_kv = False
        elif holds:
            reload = True
            load_expected = True
            override = self.resolve_weights_override(model)
            if override is not None:
                weights_gb = override
            elif declared is not None and declared["weights_gb"] is not None:
                weights_gb = declared["weights_gb"]
            else:
                total_gb = (
                    resident_view.size_bytes or resident_view.size_vram_bytes
                ) / _BYTES_PER_GIB
                weights_gb = max(0.0, total_gb - _kv(loaded_ctx))
            credit = resident_view.size_vram_gb
            ram_credit = resident_view.ram_bytes / _BYTES_PER_GIB

        if background and load_expected and not already_loaded:
            joined = next((p for p in pending if p.model == model), None)
            if joined is not None:
                # The model is loading for an admission the loaded view does
                # not show yet: the call joins that load at its context, and
                # waits for it when it asks more.
                if effective_ctx is None or (joined.num_ctx is not None and effective_ctx <= joined.num_ctx):
                    return _holds_decision(
                        "admit",
                        "fits_pending",
                        joined.num_ctx,
                        (joined.threads, joined.threads_batch, joined.placement),
                    )
                return _refuse("background_load_pending")
            if main_kv and effective_ctx is None:
                # Told no context, the engine would load at its own: the load
                # is priced, and loaded, at one the governor names.
                effective_ctx = self._clamp_ctx(model, self._background_ctx(model))

        known_weights = (0.0 if weights_gb is None else weights_gb) + extra_gb

        # The dynamic quantum runs AFTER the model-window clamp: the
        # clamp answers what the model can hold, the stage answers what
        # the machine should allocate. The pre-stage value keeps the
        # clamp reason honest below. A reload sizes it with the memory it
        # frees, and a quantum at or below the held context is no reload.
        # The background sizes it with the loads not yet seen taken.
        _clamped_ctx = effective_ctx
        _dyn_applied = False
        if main_kv:
            effective_ctx, _dyn_applied = self._dynamic_ctx_stage(
                effective_ctx, model, snapshot, known_weights + state_gb - credit + pending_vram, kv_coefficient
            )
            if reload and (effective_ctx is None or effective_ctx <= loaded_ctx) and not extras_pending:
                return _holds_decision("admit", "fits_resident")
        if reload and background:
            # A reload frees the resident first: the background is served by
            # the resident at its own context where the caller's floor allows
            # it, and otherwise waits for the resident to leave.
            floor = self._config.ctx_floor.get(caller)
            if floor is not None and loaded_ctx >= floor and not extras_pending:
                return _holds_decision("downsize", "ctx_laddered_to_fit+fits_resident")
            return _refuse("background_never_reloads")

        def _cost(ctx: int | None) -> float:
            return known_weights + state_gb + (_kv(ctx) if main_kv else 0.0)

        if background and load_expected:
            # The background loads only into memory known to be free.
            if snapshot.capacity_gb is None and not getattr(snapshot, "cards_absent", False):
                return _refuse("background_capacity_unknown")
            if weights_gb is None or extras_unknown:
                return _refuse("background_cost_unknown")
            if main_kv and not effective_ctx:
                return _refuse("background_ctx_unknown")

        if snapshot.capacity_gb is None:
            # 3.1: the VRAM half fails open; the RAM half still applies,
            # less what a reload frees there. A background load gets here
            # only on a machine with no card: the whole of its cost goes to
            # the RAM free above the reserve, less the loads not yet seen.
            ram_mb = snapshot.ram_available_mb
            ram_only = background and load_expected
            if ram_only:
                ram_need = _cost(effective_ctx) - ram_credit
                ram_free = ram_mb / 1024.0 - self.effective_ram_reserve_gb(snapshot) - pending_ram
                short = ram_mb <= 0.0 or ram_need > ram_free
            else:
                ram_need = known_weights - ram_credit
                ram_free = ram_mb / 1024.0
                short = ram_mb > 0.0 and ram_need > ram_free
            if short:
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
                    shortfall_gb=round(ram_need - ram_free, 3),
                    engine=engine_name,
                    admission_class=klass,
                )
            else:
                if ram_only:
                    reason = "fits_ram+ctx_capped" if _dyn_applied else "fits_ram"
                else:
                    reason = "capacity_unknown_fail_open+ctx_capped" if _dyn_applied else "capacity_unknown_fail_open"
                decision = AdmissionDecision(
                    admitted=True,
                    model=model,
                    num_ctx=effective_ctx,
                    action="admit",
                    reason=reason,
                    provenance=provenance,
                    ticket_id=ticket_id,
                    caller=caller,
                    requested_ctx=requested_ctx,
                    load_expected=load_expected,
                    engine=engine_name,
                    cost_gb=round(_cost(effective_ctx), 3),
                    credit_gb=round(credit, 3),
                    admission_class=klass,
                )
                # On a machine with no card the CPU computes the whole model;
                # with a card of unknown memory, where is unknown.
                _carry_threads(decision, bool(getattr(snapshot, "cards_absent", False)))
                # Where the engine puts a foreground load is unknown here:
                # it counts against both memories. The background's is RAM.
                taken = max(0.0, _cost(effective_ctx) - ram_credit)
                if not self._claim_load(decision, view, 0.0 if ram_only else taken, taken, loading_extras):
                    return _refuse("background_pending_changed")
            self._record_admission(decision)
            return decision

        in_use = snapshot.vram_in_use_gb
        others = getattr(snapshot, "vram_others_gb", 0.0) or 0.0
        margin = self._vram_margin_gb(snapshot)
        # A model is never its own eviction candidate: a reload credits its
        # memory instead. The background evicts nothing.
        evictable = 0.0 if background else sum(
            size
            for name, _idle, size in self._evictable_candidates(snapshot)
            if name != model
        )
        # The background also leaves what the loads not yet seen will take.
        budget_unconditional = snapshot.capacity_gb - in_use - others - margin + credit - pending_vram
        budget_with_eviction = budget_unconditional + evictable
        # The RAM a split may use; None when no split is priced at all.
        split_allowed = not background or self._config.background_allow_split
        ram_budget = self._offload_ram_budget_gb(snapshot) if split_allowed else None
        if ram_budget is not None:
            ram_budget += ram_credit - pending_ram
        min_share = self._config.offload_min_gpu_share

        def _fit(ctx: int | None) -> bool | None:
            """True: fits now. False: fits only after eviction. None: no."""
            cost = _cost(ctx)
            if cost <= budget_unconditional:
                return True
            if evictable > 0.0 and cost <= budget_with_eviction:
                return False
            return None

        def _placement(cost: float) -> tuple[float, float]:
            """(GiB on the GPU, GiB in RAM) of a split: the GPU takes what
            is free NOW -- a split never waits on an eviction."""
            gpu = max(0.0, min(cost, budget_unconditional))
            return gpu, cost - gpu

        # A split placed layer by layer, when the file that serves the model
        # says what each layer weighs: the table is read once a split is
        # priced, and only for a load of the model itself (a resident that
        # makes room for extra models places nothing). An operator's weights
        # override or an engine's declared weights replace the file's (the
        # residency of a mixture of experts whose experts stay in RAM), so no
        # layer count can be told from the file then. Without a plan, the
        # split is the even one.
        figured = self.resolve_weights_override(model) is not None or (
            declared is not None and declared["weights_gb"] is not None
        )
        main_loads = (not already_loaded or reload) and not figured
        tables: list[Any] = []
        too_slow: list[float] = []
        gpu_speed = self._config.split_gpu_bandwidth_gbs
        ram_speed = self._config.split_ram_bandwidth_gbs
        max_slowdown = self._config.split_max_slowdown

        def _layer_kv(ctx: int | None, layers: int) -> list[float]:
            """Each layer's KV at ``ctx``, in GiB: its share of the KV priced,
            from the geometry, or an even share under an operator override."""
            kv = _kv(ctx) if main_kv else 0.0
            shares = None
            if geometry is not None and kv_override is None:
                shares = geometry.get("kv_bytes_per_layer")
            if shares and len(shares) == layers and sum(shares) > 0:
                whole = float(sum(shares))
                return [kv * share / whole for share in shares]
            return [kv / layers] * layers

        def _plan(ctx: int | None) -> tuple[float, float, float, int, float | None] | None:
            """(GiB on the GPU, GiB in RAM, total GiB, layers on the GPU,
            expected slowdown or None) of the split placed layer by layer at
            ``ctx``; None without a table whose blocks are 0 to N-1, or for a
            model with a KV cache whose context is not told: the engine would
            hold its own default context's KV on layers priced without it."""
            if main_kv and not ctx:
                return None
            if not tables:
                tables.append(self.tensor_table(model, engine=engine) if main_loads else None)
            table = tables[0]
            if table is None:
                return None
            layers = len(table.blocks)
            if not layers or list(table.blocks) != list(range(layers)):
                return None
            weights = [table.blocks[i] / _BYTES_PER_GIB for i in range(layers)]
            kv = _layer_kv(ctx, layers)
            tensors = table.total_bytes / _BYTES_PER_GIB
            kv_total = sum(kv)
            # The model's own weights are never priced under its tensors;
            # extra models (a draft) and a declared state come on top.
            main_weights = 0.0 if weights_gb is None else weights_gb
            total = max(main_weights, tensors) + extra_gb + state_gb + kv_total
            # What the tensors and the KV do not explain -- the compute
            # buffers, the estimate's margin, a draft, a declared state --
            # stays on the GPU; the layers are then taken from the last, as
            # the engines place them, while they fit what is free now.
            gpu = total - tensors - kv_total
            count = 0
            for index in reversed(range(layers)):
                step = weights[index] + kv[index]
                if gpu + step > budget_unconditional:
                    break
                gpu += step
                count += 1
            slowdown = None
            if gpu_speed and ram_speed and count:
                # The bytes read per token: each layer and its KV, and the
                # output head, which a split leaves in RAM.
                first = layers - count
                on_gpu = sum(weights[i] + kv[i] for i in range(first, layers))
                in_ram = sum(weights[i] + kv[i] for i in range(first))
                in_ram += table.output_bytes / _BYTES_PER_GIB
                read = on_gpu + in_ram
                if read > 0.0:
                    slowdown = (on_gpu / gpu_speed + in_ram / ram_speed) / (read / gpu_speed)
            return gpu, total - gpu, total, count, slowdown

        def _split(ctx: int | None) -> tuple[float, float, float, int | None, float | None] | None:
            """The placement of a split that holds at ``ctx``, else None:
            (GiB on the GPU, GiB in RAM, total GiB, layers on the GPU when
            placed layer by layer, expected slowdown when known)."""
            if ram_budget is None:
                return None
            plan = _plan(ctx)
            if plan is not None:
                gpu, ram, total, count, slowdown = plan
                if not count or ram <= 0.0 or ram > ram_budget or gpu / total < min_share:
                    return None
                if slowdown is not None and max_slowdown is not None and slowdown > max_slowdown:
                    too_slow.append(slowdown)
                    return None
                return plan
            cost = _cost(ctx)
            gpu, ram = _placement(cost)
            if ram <= 0.0 or ram > ram_budget:
                return None
            if gpu / cost < min_share:
                return None
            return gpu, ram, cost, None, None

        candidates: list[int | None] = [effective_ctx]
        floor = self._config.ctx_floor.get(caller)
        # Without a KV cache a shorter context costs no less: no ladder.
        if effective_ctx is not None and floor is not None and main_kv:
            steps = sorted(
                {
                    int(s)
                    for s in self._config.ctx_ladder
                    if floor <= int(s) < effective_ctx
                },
                reverse=True,
            )
            candidates.extend(steps)
        # Below the context a resident model holds, a reload buys nothing:
        # the model at its own context is the last step instead, where the
        # caller's floor allows it (a caller without a floor is never
        # downsized).
        resident_step = False
        if reload:
            candidates = [c for c in candidates if c is None or c > loaded_ctx]
            resident_step = floor is not None and loaded_ctx >= floor

        # The order of the attempts. Each ctx is tried on the GPU alone
        # first (now, then after eviction). "context" splits it before
        # stepping down; "speed" steps down the whole ladder on the GPU
        # alone and splits only at the last step, where the GPU holds the
        # largest share -- if no split holds there, none holds above. A
        # resident model's own context comes after the GPU-alone steps.
        last = len(candidates) - 1
        if self._config.offload_prefer == "speed":
            attempts = [(i, ctx, "gpu") for i, ctx in enumerate(candidates)]
            if resident_step:
                attempts.append((last + 1, loaded_ctx, "resident"))
            attempts.append((last, candidates[last], "split"))
        else:
            attempts = [
                (i, ctx, mode)
                for i, ctx in enumerate(candidates)
                for mode in ("gpu", "split")
            ]
            if resident_step:
                attempts.append((last + 1, loaded_ctx, "resident"))

        for index, ctx, mode in attempts:
            if mode == "resident":
                return _holds_decision("downsize", "ctx_laddered_to_fit+fits_resident")
            placement: tuple[float, float, float, int | None, float | None] | None = None
            if mode == "split":
                placement = _split(ctx)
                if placement is None:
                    continue
                conditional = False
            else:
                verdict = _fit(ctx)
                if verdict is None:
                    continue
                conditional = verdict is False
            action = "downsize" if index > 0 else "admit"
            reason_parts = []
            if action == "downsize":
                reason_parts.append("ctx_laddered_to_fit")
            elif (
                _clamped_ctx is not None
                and requested_ctx is not None
                and _clamped_ctx < requested_ctx
            ):
                reason_parts.append("clamped_to_model_limit")
            if _dyn_applied:
                reason_parts.append("ctx_quantized")
            if conditional:
                reason_parts.append("conditional_on_eviction")
            if placement is not None:
                reason_parts.append("partial_offload")
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
                engine=engine_name,
                num_parallel=self._ollama_sequences(),
                cost_gb=round(_cost(ctx), 3),
                credit_gb=round(credit, 3),
                admission_class=klass,
            )
            if placement is not None:
                gpu, ram, total, count, slowdown = placement
                share = gpu / (gpu + ram)
                decision.gpu_share = share
                decision.vram_cost_gb = round(gpu, 3)
                decision.ram_cost_gb = round(ram, 3)
                if count is not None:
                    decision.num_gpu = count
                    decision.gpu_layers = count
                    decision.cost_gb = round(total, 3)
                    if slowdown is not None:
                        decision.expected_slowdown = round(slowdown, 3)
                elif geometry is not None:
                    decision.gpu_layers = int(share * geometry["layers"])
            if already_loaded and not reload and decision.num_gpu is None:
                # The resident model stays as its split load placed it.
                decision.num_gpu = self.pinned_layers(model, ctx)
            # A split computes its RAM share on the CPU.
            _carry_threads(decision, placement is not None)
            if placement is not None:
                vram_taken, ram_taken = placement[0] - credit, placement[1] - ram_credit
            else:
                vram_taken, ram_taken = _cost(ctx) - credit, 0.0
            if not self._claim_load(decision, view, max(0.0, vram_taken), max(0.0, ram_taken), loading_extras):
                return _refuse("background_pending_changed")
            self._record_admission(decision)
            return decision

        minimal_cost = _cost(candidates[-1])
        vram_shortfall = round(max(0.0, minimal_cost - budget_with_eviction), 3)
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
            shortfall_gb=vram_shortfall,
            engine=engine_name,
            cost_gb=round(minimal_cost, 3),
            credit_gb=round(credit, 3),
            admission_class=klass,
        )
        if ram_budget is not None:
            # A split was priced and none held: name what it lacked, from the
            # layer-by-layer placement when the model's file gave one.
            plan = _plan(candidates[-1])
            if plan is not None:
                gpu, ram, total, count, _slowdown = plan
                share = gpu / total if total > 0.0 else 1.0
                # The plan's total, never under the model's own tensors, is
                # what the GPU alone would have had to hold.
                vram_shortfall = round(max(0.0, total - budget_with_eviction), 3)
                decision.shortfall_gb = vram_shortfall
                decision.cost_gb = round(total, 3)
            else:
                gpu, ram = _placement(minimal_cost)
                share = gpu / minimal_cost if minimal_cost > 0.0 else 1.0
                count = None
            ram_shortfall = round(max(0.0, ram - ram_budget), 3)
            reason_parts = ["vram_insufficient"]
            if ram_shortfall > 0.0:
                reason_parts.append("ram_insufficient")
            if share < min_share:
                reason_parts.append("gpu_share_below_minimum")
            if count == 0:
                reason_parts.append("no_layer_fits")
            if too_slow:
                reason_parts.append("split_too_slow")
            decision.reason = "+".join(reason_parts)
            decision.ram_shortfall_gb = ram_shortfall
            decision.payload = _offload_refusal_payload(
                model,
                vram_shortfall,
                ram_shortfall,
                share,
                min_share,
                no_layer_fits=count == 0,
                slowdown=min(too_slow) if too_slow else None,
                max_slowdown=max_slowdown,
            )
        self._record_admission(decision)
        return decision

    def _offload_ram_budget_gb(self, snapshot: Any) -> float | None:
        """The system RAM a split may use, or None when none is priced.

        MemAvailable less the effective reserve (ram_reserve_state). Offload
        off, or the RAM unreadable (the snapshot then reads 0), means no
        split: a split priced against RAM nobody measured would be a guess,
        so the refusal stands instead.
        """
        if not self._config.offload_enabled:
            return None
        ram_mb = getattr(snapshot, "ram_available_mb", 0.0) or 0.0
        if ram_mb <= 0.0:
            return None
        return ram_mb / 1024.0 - self.effective_ram_reserve_gb(snapshot)

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

    def _ladder_quantize_up(self, ctx: int) -> int:
        """The smallest ladder step at or above ``ctx``; above the top,
        the top itself -- the quantum never leaves the ladder."""
        steps = sorted(int(s) for s in self._config.ctx_ladder if int(s) > 0)
        if not steps:
            return int(ctx)
        for step in steps:
            if ctx <= step:
                return step
        return steps[-1]

    def _dynamic_ctx_stage(
        self,
        effective_ctx: int | None,
        model: str | None,
        snapshot: ResourceSnapshot,
        known_weights_gb: float,
        kv_coefficient: float | None = None,
    ) -> tuple[int | None, bool]:
        """Ladder-quantized context under a live ceiling: (value, applied).

        Off, or without a request, the input passes through untouched.
        On, the request rounds UP to its ladder step so a growing
        conversation keeps one stable quantum, capped by a ceiling: the
        configured conservative value when capacity is unknown (the
        admission itself stays deliberately fail-open; only the quantum
        is capped), else the highest step whose KV estimate fits the
        VRAM left after weights and margin. A ceiling below the lowest
        step still answers the lowest step: refusing is the fit math's
        job, never this stage's. ``kv_coefficient`` is the one admission
        already resolved for the model; without it the stage resolves it.
        """
        if not self._config.dynamic_ctx_enabled or not effective_ctx:
            return effective_ctx, False
        quantized = self._ladder_quantize_up(int(effective_ctx))
        steps = sorted(int(s) for s in self._config.ctx_ladder if int(s) > 0)
        if snapshot.capacity_gb is None:
            ceiling = int(self._config.dynamic_ctx_unknown_ceiling)
        else:
            kv_budget_gb = (
                snapshot.capacity_gb
                - snapshot.vram_in_use_gb
                - (getattr(snapshot, "vram_others_gb", 0.0) or 0.0)
                - self._vram_margin_gb(snapshot)
                - known_weights_gb
            )
            coeff = (
                kv_coefficient
                if kv_coefficient is not None
                else self.resolve_kv_coefficient(model)
            )
            tokens = 0
            if coeff > 0 and kv_budget_gb > 0:
                tokens = int((kv_budget_gb / coeff) * 1024.0)
            fitting = [s for s in steps if s <= tokens]
            if fitting:
                ceiling = max(fitting)
            else:
                ceiling = steps[0] if steps else quantized
        value = min(quantized, ceiling)
        return value, value != int(effective_ctx)

    def _evictable_now_gb(self, snapshot: ResourceSnapshot) -> float:
        """Summed size_vram of loaded models idle past the threshold (4.2).

        Delegates to _evictable_candidates so the fit math and
        the targeted eviction read the SAME definition of evictable.
        """
        return sum(
            size for _name, _idle, size in self._evictable_candidates(snapshot)
        )

    def _evictable_candidates(
        self, snapshot: ResourceSnapshot
    ) -> list[tuple[str, float, float]]:
        """(name, idle_s, size_gb) for every eviction candidate, in the
        Section 5 eviction order: first every resident only the background
        loaded with no call in flight on it, whatever its idle time; then
        every other loaded model idle past the threshold. Each group is
        sorted oldest-idle FIRST.

        Idle time is derived from the snapshot's expirations: a model's
        ``expires_at`` is last activity plus the warmup keep_alive, so
        idle = keep_alive_s - (expires_at - now), compared on the WALL
        clock (expirations are wall-clock stamps; the snapshot's monotonic
        ``taken_at`` is deliberately not used). Anything that cannot be
        coerced or computed counts as NOT evictable (conservative) -- except
        a resident of the background, which needs no idle time and counts
        as idle for none when it has none.
        """
        with self._queue_cond:
            busy = {d.model for d in self._live_in_flight(self._clock())}
            guests = {
                name
                for name, (owner, _seen) in self._owners.items()
                if owner == _BACKGROUND and name not in busy
            }
        threshold = self._config.idle_evict_threshold_s
        keep_alive_s = _parse_duration_s(
            getattr(self._warmup, "keep_alive", None)
        )
        now = time.time()
        first: list[tuple[str, float, float]] = []
        candidates: list[tuple[str, float, float]] = []
        for view in snapshot.loaded:
            if view.size_vram_bytes <= 0:
                continue
            expires = _coerce_epoch_s(view.expires_at)
            idle_s = (
                keep_alive_s - (expires - now)
                if keep_alive_s is not None and expires is not None
                else None
            )
            if view.name in guests:
                first.append((view.name, idle_s if idle_s is not None else 0.0, view.size_vram_gb))
            elif idle_s is not None and threshold is not None and 0 <= threshold <= idle_s:
                candidates.append((view.name, idle_s, view.size_vram_gb))
        first.sort(key=lambda c: c[1], reverse=True)
        candidates.sort(key=lambda c: c[1], reverse=True)
        return first + candidates

    # -- Targeted eviction (Section 5, honouring conditional grants) ---------

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
        Section 12 posture). This is also the surface the
        POST /api/governor/evict route calls.
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
        ONLY as many as the shortfall needs. The load is priced exactly as
        the admission priced it (the grant's cost and the memory a reload
        frees); a grant the admission did not price is priced its way.
        Every path fails open: any miss or error leaves the admitted call
        untouched and Ollama's own LRU carries it (Section 12). Never
        raises.
        """
        try:
            snapshot = self.get_snapshot_fast()
            if snapshot.capacity_gb is None:
                return
            cost = getattr(decision, "cost_gb", None)
            credit = getattr(decision, "credit_gb", 0.0) or 0.0
            if cost is None:
                cost = self._load_cost_gb(decision.model, decision.num_ctx, snapshot)
                credit = 0.0
            if credit and decision.model not in {v.name for v in snapshot.loaded if v.resident}:
                # The model left since the grant: a reload frees nothing now.
                credit = 0.0
            cost -= credit
            budget = (
                snapshot.capacity_gb
                - snapshot.vram_in_use_gb
                - (getattr(snapshot, "vram_others_gb", 0.0) or 0.0)
                - self._vram_margin_gb(snapshot)
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

    # -- The bounded opt-in queue (Section 5) ---------------------------------

    def admit_or_wait(
        self,
        model: str,
        requested_ctx: int | None = None,
        caller: str = "benchmark",
        extra_models: list[str] | None = None,
        digest: str | None = None,
        wait_s: float | None = None,
        engine: str | None = None,
        cancel: Any = None,
    ) -> AdmissionDecision:
        """admit() with the Section 5 bounded priority queue.

        ``engine`` names the backend that will serve the call, as admit()
        takes it (a tuner's trial names the engine it measures).

        ``cancel`` is the caller's own cancel (an event): once it is set, a
        caller about to wait does not enter the queue, and a waiter leaves
        it at its next wake, its place given back, both refused "cancelled"
        -- the work they would serve is gone.

        Who waits: a caller queue.enabled_per_caller names, as it names it;
        any other as its class does (classes.<class>.queued: the shipped
        file enrolls the background only). A caller that does not wait gets
        plain admit() semantics -- chat and pipeline additionally never call
        this entry, so the interactive path stays refuse-by-default at the
        call site (D3).

        The order: class first (interactive, user, background), then
        arrival. A caller may try for admission only while no waiter of a
        higher class waits and no waiter of its own class ahead of it has
        been passed ``queue.max_bypass`` times; every try through this entry
        passes each waiter of its class ahead of it once, charged in the
        same step as the check (two callers trying at once never pass a
        waiter past its allowance) and given back when the try is refused.
        A caller that may not try waits without trying, so callers that fit
        cannot starve one that does not, and a lower class never takes
        memory a higher one waits for. A background waiter tries only while
        the background gate is open.

        The bounds: each class has its own depth (classes.<class>.depth,
        else queue.depth) and wait (classes.<class>.wait_s, else
        queue.wait_s); ``wait_s`` may shorten the caller's wait, never
        lengthen it, and the deadline is set as the caller enters the
        queue. At the depth bound the caller's refusal stands, and a refusal
        no wait can lift (the card unreadable, the cost or the context
        unknown) is answered at once; at the end of the wait the caller gets
        the last refusal admission gave it, or, if it never tried, one that
        says why it waited. A waiter's retries are silent: the ring and the
        refusal window keep its first try, its entry and its outcome. Every
        wake honours the estop FIRST, whether the waiter may try or not, so
        a drain releases every waiter to refusal (the drain's invalidation
        notify wakes them immediately) and no queued entry can outlive it.
        Waiters hold no lock while waiting; the wait is sliced so an
        injected fake clock drives the deadline in container tests.

        A reload keeps the queue and judges by the file it read: a caller
        still holding the governor it replaced is answered by the current
        one, and a waiter's turn, its gate and its refusal are the current
        one's (its deadline stays).
        """
        current = _current_governor(self)
        if current is not self:
            return current.admit_or_wait(
                model,
                requested_ctx,
                caller=caller,
                extra_models=extra_models,
                digest=digest,
                wait_s=wait_s,
                engine=engine,
                cancel=cancel,
            )
        cfg = self._config
        klass = cfg.class_of(caller)
        named = cfg.queue_enabled_per_caller.get(caller)
        queued = bool(named) if named is not None else bool(cfg.class_queued.get(klass, False))

        def _admit() -> AdmissionDecision:
            # A reload builds a governor that keeps this one's queue: a
            # waiter asks whichever of the two is current.
            return _current_governor(self).admit(
                model,
                requested_ctx,
                caller=caller,
                extra_models=extra_models,
                digest=digest,
                engine=engine,
            )

        def _final(decision: AdmissionDecision) -> bool:
            return decision.is_estop or decision.reason in _FINAL_REFUSALS

        last: AdmissionDecision | None = None
        if queued:
            with self._queue_cond:
                passed = self._take_turn(klass, None)
        else:
            passed = []
        if passed is not None:
            last = _admit()
            if last.admitted:
                return last
            self._give_back(passed)
            if not queued or _final(last):
                return last
        if _cancelled(cancel):
            return self._queue_refusal(model, requested_ctx, caller, klass, "cancelled")
        with self._queue_cond:
            depth = max(0, cfg.class_depth.get(klass, cfg.queue_depth))
            waiting = sum(1 for w in self._waiters if w.admission_class == klass)
            waiter = None
            if waiting < depth:
                bound = cfg.class_wait_s.get(klass, cfg.queue_wait_s)
                if wait_s is not None:
                    bound = min(bound, wait_s)
                # Set as the caller enters the queue: nothing slower that
                # follows (the ring write) lengthens the wait.
                deadline = self._clock() + max(0.0, bound)
                waiter = self._enqueue(klass)
        if waiter is None:
            logger.debug("Queue depth bound reached; %s refusal stands for %s", caller, model)
            return last if last is not None else self._queue_refusal(model, requested_ctx, caller, klass, "queue_full")
        unrecorded = False
        cancelled = False
        try:
            try:
                # Ring visibility of the enqueue (the 4.4 "queue" action).
                self.record_decision(
                    caller, model, requested_ctx, None, "queue", "enqueued"
                )
            except Exception as exc:
                logger.debug("Queue ring write failed: %s", exc)
            while True:
                remaining = deadline - self._clock()
                if remaining <= 0.0:
                    break
                with self._queue_cond:
                    self._queue_cond.wait(
                        timeout=min(remaining, _QUEUE_WAIT_SLICE_S)
                    )
                if self._estop_engaged():
                    return _admit()
                if _cancelled(cancel):
                    cancelled = True
                    break
                if klass == _BACKGROUND and _current_governor(self)._background_hold() is not None:
                    continue
                with self._queue_cond:
                    passed = _current_governor(self)._take_turn(klass, waiter)
                if passed is None:
                    continue
                self._quiet.on = True
                try:
                    last = _admit()
                finally:
                    self._quiet.on = False
                unrecorded = True
                if last.admitted or _final(last):
                    _current_governor(self)._record_admission(last)
                    return last
                self._give_back(passed)
        finally:
            self._dequeue(waiter)
        if cancelled:
            return _current_governor(self)._queue_refusal(model, requested_ctx, caller, klass, "cancelled")
        if last is not None:
            if unrecorded:
                _current_governor(self)._record_admission(last)
            return last
        current = _current_governor(self)
        hold = current._background_hold() if klass == _BACKGROUND else None
        reason = f"background_held:{hold}" if hold is not None else "queue_wait_expired"
        return current._queue_refusal(model, requested_ctx, caller, klass, reason, held_by=hold)

    def _may_try(self, klass: str, waiter: _Waiter | None) -> bool:
        """Whether a caller of ``klass`` may try for admission now: ``waiter``
        in the queue, or a newcomer (None), behind every waiter of its
        class. No waiter of a higher class may wait, and no waiter of its
        own class ahead of it may have used up its passes. Called under the
        queue lock."""
        rank = _CLASS_RANK[klass]
        allowance = max(0, self._config.queue_max_bypass)
        for other in self._waiters:
            if other is waiter:
                continue
            other_rank = _CLASS_RANK[other.admission_class]
            if other_rank < rank:
                return False
            ahead = waiter is None or other.seq < waiter.seq
            if other_rank == rank and ahead and other.bypassed >= allowance:
                return False
        return True

    def _charge_bypass(self, klass: str, waiter: _Waiter | None) -> list[_Waiter]:
        """A try through the queue's entry passes every waiter of its class
        ahead of it: each is charged one pass, and returned. Called under the
        queue lock."""
        charged: list[_Waiter] = []
        for other in self._waiters:
            if other is waiter or other.admission_class != klass:
                continue
            if waiter is None or other.seq < waiter.seq:
                other.bypassed += 1
                charged.append(other)
        return charged

    def _take_turn(self, klass: str, waiter: _Waiter | None) -> list[_Waiter] | None:
        """None when a caller of ``klass`` may not try now; otherwise the
        waiters its try passes, each charged in the same step as the check.
        Called under the queue lock."""
        if not self._may_try(klass, waiter):
            return None
        return self._charge_bypass(klass, waiter)

    def _give_back(self, passed: list[_Waiter]) -> None:
        """A try that was refused passed nobody: its passes are given back."""
        with self._queue_cond:
            for other in passed:
                other.bypassed = max(0, other.bypassed - 1)

    def _enqueue(self, klass: str) -> _Waiter:
        """A waiter of ``klass`` at the end of the order of arrival."""
        with self._queue_cond:
            waiter = _Waiter(admission_class=klass, seq=next(self._queue_seq))
            self._waiters.append(waiter)
            return waiter

    def _dequeue(self, waiter: _Waiter) -> None:
        """``waiter`` leaves the queue; those behind it may now try."""
        with self._queue_cond:
            if waiter in self._waiters:
                self._waiters.remove(waiter)
            self._queue_cond.notify_all()

    def _queue_refusal(
        self,
        model: str,
        requested_ctx: int | None,
        caller: str,
        klass: str,
        reason: str,
        held_by: str | None = None,
    ) -> AdmissionDecision:
        """The refusal of a caller the queue kept from trying, recorded."""
        decision = AdmissionDecision(
            admitted=False,
            model=model,
            num_ctx=None,
            action="refuse",
            reason=reason,
            ticket_id=uuid.uuid4().hex[:12],
            caller=caller,
            requested_ctx=requested_ctx,
            admission_class=klass,
            held_by=held_by,
        )
        self._record_admission(decision)
        return decision

    def _estop_engaged(self) -> bool:
        """Whether the emergency stop is engaged, through its own seam."""
        estop = _resolve_emergency_stop()
        if estop is None:
            return False
        try:
            return bool(estop.is_stopped())
        except Exception as exc:
            logger.debug("Estop flag read failed open: %s", exc)
            return False

    @property
    def queue_depth(self) -> int:
        """Current number of queued admissions, every class (the
        status-API field)."""
        with self._queue_cond:
            return len(self._waiters)

    def queue_by_class(self) -> dict[str, int]:
        """How many callers wait in the queue, by class."""
        counts = dict.fromkeys(ADMISSION_CLASSES, 0)
        with self._queue_cond:
            for waiter in self._waiters:
                counts[waiter.admission_class] += 1
        return counts

    # -- Who asks: the calls in flight, the residents' classes, the gate -------

    def note_held(self, decision: AdmissionDecision | None) -> None:
        """The calling thread now holds ``decision`` (None: it holds none).

        Fed by set_active_ticket and clear_active_ticket, so by ticket_scope
        and the funnels' hold and release; only an admitted ticket is a call
        in flight, and an interactive admission held is no longer counted as
        admitted only. Holding a ticket on a model loaded for a lower class
        makes the model the holder's class: the background's guest is the
        foreground's own once the foreground uses it. A release marks the
        load the ticket admitted, if it is still pending, as ended by its
        call, and wakes the queue, since the background gate may open.
        """
        ident = threading.get_ident()
        now = self._clock()
        with self._queue_cond:
            released = self._in_flight.pop(ident, None)
            if decision is not None and decision.admitted:
                klass = getattr(decision, "admission_class", _DEFAULT_CLASS)
                self._in_flight[ident] = (decision, now)
                self._admitted.pop(decision.ticket_id, None)
                owner = self._owners.get(decision.model)
                if owner is not None and _CLASS_RANK.get(klass, 1) < _CLASS_RANK.get(owner[0], 1):
                    owner[0] = klass
            if released is not None:
                ticket = released[0].ticket_id
                load = self._pending_loads.get(ticket)
                still_held = decision is not None and decision.ticket_id == ticket
                if load is not None and load.released_at is None and not still_held:
                    load.released_at = now
                self._queue_cond.notify_all()

    def note_loaded_by(self, model: str, admission_class: str) -> None:
        """A load of ``model`` admitted for ``admission_class`` is about to
        happen (the engine gate accounts it): the model is that class's
        until it leaves the loaded view or a higher class holds a ticket on
        it."""
        klass = admission_class if admission_class in _CLASS_RANK else _DEFAULT_CLASS
        with self._queue_cond:
            self._owners[model] = [klass, False]

    # -- The loads admitted and not yet seen -----------------------------------

    def _expire_pending_loads(self, now: float) -> None:
        """Drop each load admitted longer ago than
        background_gate.pending_load_max_s without the loaded view showing
        it: a load that never happened, or a signal lost, with a warning.
        Called under the queue lock."""
        bound = self._config.background_gate_pending_load_max_s
        for ticket, load in list(self._pending_loads.items()):
            if now - load.since > bound:
                del self._pending_loads[ticket]
                logger.warning(
                    "Load of %s admitted %.0f s ago for the %s class never showed in the loaded view, past"
                    " the %.0f s bound: no longer counted against the background",
                    load.model,
                    now - load.since,
                    load.admission_class,
                    bound,
                )

    def _pending_view(self) -> tuple[int, list[_PendingLoad]]:
        """What a background decision reads before it starts: the count of
        loads ever claimed and the loads not yet seen. Read before the
        snapshot, so a load the snapshot does not show yet is never missed
        by both."""
        with self._queue_cond:
            self._expire_pending_loads(self._clock())
            return self._pending_version[0], list(self._pending_loads.values())

    def _claim_load(
        self,
        decision: AdmissionDecision,
        view: tuple[int, list[_PendingLoad]] | None,
        vram_gb: float,
        ram_gb: float,
        extras: list[str],
    ) -> bool:
        """Register the load ``decision`` admits, what it adds to VRAM and to
        RAM, until the loaded view shows it. A background decision taken on
        ``view`` claims it only if no load was claimed since it read the
        view; otherwise it claims nothing and False is returned. A decision
        that loads nothing claims nothing."""
        if not decision.load_expected:
            return True
        with self._queue_cond:
            if view is not None and self._pending_version[0] != view[0]:
                return False
            self._pending_version[0] += 1
            self._pending_loads[decision.ticket_id] = _PendingLoad(
                model=decision.model,
                num_ctx=decision.num_ctx,
                vram_gb=vram_gb,
                ram_gb=ram_gb,
                extras=tuple(extras),
                admission_class=decision.admission_class,
                since=self._clock(),
                threads=decision.threads,
                threads_batch=decision.threads_batch,
                placement=decision.placement,
            )
            return True

    def _settle_pending_loads(self, snapshot: ResourceSnapshot) -> None:
        """End each pending load ``snapshot`` accounts for: shown, with its
        extra models, at its context (the view now counts its memory), or
        released by its call before the view was taken, shown or not: the
        engine answers a call only once its model is loaded, so a view taken
        after the release counts whatever the load left resident (nothing
        when it failed, or is gone already). A view the engine did not
        answer proves nothing. Called once ``snapshot`` is stored."""
        if "S1" not in snapshot.sources:
            return
        shown = {view.name: view.context_length for view in snapshot.loaded if view.resident}
        with self._queue_cond:
            for ticket, load in list(self._pending_loads.items()):
                held_ctx = shown.get(load.model)
                seen = (
                    load.model in shown
                    and all(extra in shown for extra in load.extras)
                    and (load.num_ctx is None or held_ctx is None or held_ctx == load.num_ctx)
                )
                gone = load.released_at is not None and load.released_at <= snapshot.taken_at
                if seen or gone:
                    del self._pending_loads[ticket]

    def end_pending_load(self, ticket_id: str) -> None:
        """The load ``ticket_id`` admitted will not happen: the engine gate
        refused it."""
        with self._queue_cond:
            self._pending_loads.pop(ticket_id, None)

    def pending_loads(self) -> list[dict[str, Any]]:
        """The loads admitted and not yet seen, oldest first (the status)."""
        now = self._clock()
        with self._queue_cond:
            self._expire_pending_loads(now)
            loads = sorted(self._pending_loads.values(), key=lambda load: load.since)
            return [
                {
                    "model": load.model,
                    "admission_class": load.admission_class,
                    "num_ctx": load.num_ctx,
                    "vram_gb": round(load.vram_gb, 3),
                    "ram_gb": round(load.ram_gb, 3),
                    "age_s": round(now - load.since, 3),
                    "released": load.released_at is not None,
                }
                for load in loads
            ]

    def _background_ctx(self, model: str) -> int | None:
        """The context a background load told none is priced and loaded at:
        the one the interactive class was last admitted at for ``model`` (the
        decisions ring, which outlives a restart), else the model's output
        reserve (the context a chat asks before its prompt), else the
        ladder's smallest step; None without any of them."""
        try:
            rows = self._store.recent_decisions(self._config.decisions_ring_size)
        except Exception as exc:
            logger.debug("Decisions ring unreadable for a background context: %s", exc)
            rows = []
        for row in rows:
            ctx = row.get("admitted_ctx")
            if row.get("model") != model or row.get("decision") not in ("admit", "downsize"):
                continue
            if isinstance(ctx, int) and ctx > 0 and self._config.class_of(row.get("caller")) == _INTERACTIVE:
                return ctx
        cm = _resolve_context_manager()
        if cm is not None:
            try:
                reserve = int(getattr(cm.get_model_limits(model), "max_output", 0) or 0)
            except Exception as exc:
                logger.debug("Output reserve unavailable for a background context: %s", exc)
                reserve = 0
            if reserve > 0:
                return reserve
        steps = [int(step) for step in self._config.ctx_ladder if int(step) > 0]
        return min(steps) if steps else None

    def _live_admitted(self, now: float) -> bool:
        """Whether an interactive admission no thread holds yet is younger
        than background_gate.admitted_grace_s; older ones are dropped: a
        call that admits without ever taking its ticket. Called under the
        queue lock."""
        grace = self._config.background_gate_admitted_grace_s
        for ticket, (_decision, since) in list(self._admitted.items()):
            if now - since >= grace:
                del self._admitted[ticket]
        return bool(self._admitted)

    def _live_in_flight(self, now: float) -> list[AdmissionDecision]:
        """The tickets held now. An entry held longer than
        background_gate.in_flight_max_s is a leak: it is dropped, with a
        warning, and holds nothing. Called under the queue lock."""
        bound = self._config.background_gate_in_flight_max_s
        live: list[AdmissionDecision] = []
        for ident, (decision, since) in list(self._in_flight.items()):
            if now - since > bound:
                del self._in_flight[ident]
                logger.warning(
                    "Admission ticket %s (%s, %s) held %.0f s, past the %.0f s bound: no longer counted in flight",
                    decision.ticket_id,
                    decision.caller,
                    decision.model,
                    now - since,
                    bound,
                )
                continue
            live.append(decision)
        return live

    def in_flight_by_class(self) -> dict[str, int]:
        """How many calls are in flight, by class."""
        counts = dict.fromkeys(ADMISSION_CLASSES, 0)
        with self._queue_cond:
            live = self._live_in_flight(self._clock())
        for decision in live:
            klass = getattr(decision, "admission_class", _DEFAULT_CLASS)
            counts[klass if klass in counts else _DEFAULT_CLASS] += 1
        return counts

    def _others_cpu_reading(self) -> dict[str, Any] | None:
        """The CPU pressure other programs suffer, as the hardware profile
        reads it (others_cpu_pressure), at most once per snapshot_ttl_s:
        only the background pays the reading. Each fresh reading moves the
        gate's hysteresis: held at or above the enter mark, released below
        the exit mark. None, and no hold, when nothing can read it."""
        now = self._clock()
        with self._cache_lock:
            cached = self._others_cpu
        if cached is not None and now - cached[0] <= self._config.snapshot_ttl_s:
            return cached[1]
        hardware = self._hardware()
        read = getattr(hardware, "others_cpu_pressure", None) if hardware is not None else None
        reading = None
        if callable(read):
            try:
                reading = read()
            except Exception as exc:
                logger.debug("CPU pressure of other programs unreadable: %s", exc)
        some = reading.get("some_avg10") if isinstance(reading, dict) else None
        with self._cache_lock:
            self._others_cpu = (now, reading)
            if not isinstance(some, (int, float)) or isinstance(some, bool):
                self._shared.cpu_held = False
            elif some >= self._config.background_gate_cpu_enter:
                self._shared.cpu_held = True
            elif some < self._config.background_gate_cpu_exit:
                self._shared.cpu_held = False
        return reading

    def _background_hold(self) -> str | None:
        """Why the background gate holds a background admission now, or None.

        In order: an interactive call in flight, an interactive admission no
        thread holds yet (within its grace), an interactive caller waiting in
        the queue, a user caller waiting in it (the background never takes
        memory a higher class waits for, whatever entry it comes through),
        the CPU pressure other programs suffer. A gate switched off never
        holds, and a pressure nobody can read holds nothing.
        """
        if not self._config.background_gate_enabled:
            return None
        now = self._clock()
        with self._queue_cond:
            live = self._live_in_flight(now)
            if any(getattr(d, "admission_class", None) == _INTERACTIVE for d in live):
                return "interactive_in_flight"
            if self._live_admitted(now):
                return "interactive_admitted"
            if any(w.admission_class == _INTERACTIVE for w in self._waiters):
                return "interactive_queued"
            if any(w.admission_class != _BACKGROUND for w in self._waiters):
                return "user_queued"
        self._others_cpu_reading()
        with self._cache_lock:
            held = self._shared.cpu_held
        return "cpu_pressure" if held else None

    def background_memory_short(self) -> str | None:
        """Why background work should keep one task at a time now, or None.

        Parsing a document can take many times its size in memory, and the
        tasks of a background pool parse at once: while the kernel reports
        memory pressure (the hysteresis the RAM reserve reads) the answer is
        "memory_pressure"; while the RAM available is under the reserve a
        split leaves the rest of the machine, "ram_under_reserve". RAM that
        cannot be read is no reason.
        """
        snapshot = self.get_snapshot_fast()
        if getattr(snapshot, "memory_pressure_active", False):
            return "memory_pressure"
        total_mb = getattr(snapshot, "ram_total_mb", 0.0) or 0.0
        available_mb = getattr(snapshot, "ram_available_mb", None)
        if total_mb <= 0.0 or isinstance(available_mb, bool) or not isinstance(available_mb, (int, float)):
            return None
        if available_mb < self.effective_ram_reserve_gb(snapshot) * 1024.0:
            return "ram_under_reserve"
        return None

    def background_memory_room(self) -> int | None:
        """The memory background work may take now, in bytes: the RAM
        available less the reserve a split leaves the machine, never under
        zero, and none while the kernel reports memory pressure; None where
        the RAM cannot be read, which then bounds nothing. A job charges the
        estimated parse of each file it sends against it."""
        snapshot = self.get_snapshot_fast()
        total_mb = getattr(snapshot, "ram_total_mb", 0.0) or 0.0
        available_mb = getattr(snapshot, "ram_available_mb", None)
        if total_mb <= 0.0 or isinstance(available_mb, bool) or not isinstance(available_mb, (int, float)):
            return None
        if getattr(snapshot, "memory_pressure_active", False):
            return 0
        room_mb = available_mb - self.effective_ram_reserve_gb(snapshot) * 1024.0
        return int(max(0.0, room_mb) * 1024 * 1024)

    def background_gate_state(self) -> dict[str, Any]:
        """The background gate for the status surface: on or off, open or
        held and why, and the pressure reading it last saw."""
        reason = self._background_hold()
        reading = self._others_cpu_reading()
        with self._cache_lock:
            active = self._shared.cpu_held
        return {
            "enabled": bool(self._config.background_gate_enabled),
            "state": "held" if reason is not None else "open",
            "reason": reason,
            "pressure": None if reading is None else {**reading, "active": active},
        }

    def scheduling_state(self) -> dict[str, Any]:
        """Who is in flight and who waits, by class, the background gate,
        and the loads admitted and not yet seen (the status-API section)."""
        return {
            "in_flight": self.in_flight_by_class(),
            "queued": self.queue_by_class(),
            "background_gate": self.background_gate_state(),
            "pending_loads": self.pending_loads(),
        }

    def _record_admission(self, decision: AdmissionDecision) -> None:
        """Ring write for every recorded decision (4.4); never raises.

        The same seam feeds the in-memory refusal-rate window
        (resource decisions only -- an estop refusal is not a resource
        signal and never enters it, and neither does any decision of the
        background, whose refusals are its manners, not the machine's
        pressure: they must not change what the foreground is granted; nor
        a refusal a caller's own cancel made). A waiter's retry is not
        recorded at all: admit_or_wait records its entry in the queue and
        its outcome.
        """
        if getattr(self._quiet, "on", False):
            return
        if (
            not decision.is_estop
            and decision.admission_class != _BACKGROUND
            and decision.reason != "cancelled"
        ):
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


# What a governor built again from its files keeps of the one it replaces:
# its queue (the condition, the waiters, their order of arrival), the calls in
# flight and the interactive admissions not yet held, the class each resident
# was loaded for, the loads not yet seen and their count, the lock the
# background's decisions take, the quiet flag of the queue's retries, and,
# with the lock that guards them, the layer counts split loads pinned and the
# CPU thread counts loads were told, the loads waiting for their cost to be learned, the refusal window and what the
# pressure readings leave behind them (_SharedPressure). Each is one object
# both governors hold, so what a call still running on the replaced one
# writes is not lost.
_LIVE_STATE = (
    "_queue_cond",
    "_waiters",
    "_queue_seq",
    "_in_flight",
    "_admitted",
    "_owners",
    "_pending_loads",
    "_pending_version",
    "_background_lock",
    "_quiet",
    "_cache_lock",
    "_pins",
    "_thread_pins",
    "_pending_attribution",
    "_refusal_events",
    "_shared",
)


def reload_resource_governor() -> ResourceGovernor:
    """Build the module-level governor again from its files, at once, with
    the same paths and seams, and hand it the live state of the one it
    replaces (_LIVE_STATE): the configuration route's reload. Dropping the
    governor instead would forget a call in flight mid-call (opening the
    background gate under it), unmark the background's guests, leave the
    waiters and the loads not yet seen on an instance nobody asks, and
    forget the keep_alive a sustained pressure replaced (never restored)."""
    global _governor
    with _governor_lock:
        old = _governor
        if old is None:
            _governor = ResourceGovernor()
            return _governor
        config_path, db_path = old._paths
        new = ResourceGovernor(
            config_path=config_path,
            db_path=db_path,
            warmup=old._warmup,
            registry=old._registry_override,
            clock=old._clock,
            meminfo_path=old._meminfo_path,
            vram_probe=old._vram_probe_arg,
            hardware=old._hardware_arg,
        )
        for name in _LIVE_STATE:
            setattr(new, name, getattr(old, name))
        _governor = new
    # The new configuration may let a waiter in at once.
    new._notify_queue()
    return new


def _cancelled(cancel: Any) -> bool:
    """Whether a caller's cancel (an event, or None) is set; a cancel that
    cannot be read cancels nothing."""
    if cancel is None:
        return False
    try:
        return bool(cancel.is_set())
    except Exception:  # noqa: BLE001 - an unreadable cancel cancels nothing
        return False


def _current_governor(governor: ResourceGovernor) -> ResourceGovernor:
    """The governor a reload built from ``governor``, when there is one (it
    shares the queue), else ``governor`` itself."""
    current = _governor
    if current is not None and current is not governor and current._queue_cond is governor._queue_cond:
        return current
    return governor


# ---------------------------------------------------------------------------
# The mechanical-seam gate (4.1/4.4, consumed by inference_backend)
# ---------------------------------------------------------------------------


def _account_load(
    governor: Any, model: str, decision: AdmissionDecision, options: dict | None
) -> None:
    """The load a decision admits is about to happen: account it, mark the
    model with the class it is loaded for, then pin the layer count a split
    placed, so the calls that follow keep it. Only a count Ollama is told is
    pinned, asked as the engine head asks (``AdmissionDecision.ollama_layers``);
    otherwise the engine places the layers itself, and nothing is pinned.
    The CPU threads the load is told are pinned the same way: the call's own
    num_thread when it names one, else the decision's
    (``AdmissionDecision.ollama_threads``); with them, where the load
    computes (``AdmissionDecision.placement``)."""
    governor.invalidate_on_load(model, decision.num_ctx)
    note = getattr(governor, "note_loaded_by", None)
    if callable(note):
        note(model, getattr(decision, "admission_class", _DEFAULT_CLASS))
    layers = decision.ollama_layers(options) if decision.partial_offload else None
    if layers is not None:
        governor.pin_layers(model, layers, decision.num_ctx)
    pin = getattr(governor, "pin_threads", None)
    if not callable(pin):
        return
    own = options.get("num_thread") if isinstance(options, Mapping) else None
    if isinstance(own, int) and not isinstance(own, bool) and own >= 1:
        pin(model, own, own)
    else:
        threads = decision.ollama_threads(options)
        if threads is None:
            return
        pin(model, threads, decision.threads_batch)
    place = getattr(governor, "pin_placement", None)
    if callable(place):
        place(model, getattr(decision, "placement", None))


def note_own_threads(model: str, options: Mapping[str, Any] | None) -> None:
    """A call sent ``model`` its own num_thread through an engine that
    applies a count per call and reloads a resident model for any count
    other than the one it holds (Ollama): that count is the one the resident
    holds from then on, so it is the one pinned, with the placement of its
    load (``ResourceGovernor.pin_threads``), and the next decisions carry
    it. A load's own count was pinned as the load was accounted
    (``_account_load``), and a call that sends none changes nothing. No
    governor yet, or a disabled one: nothing to tell."""
    own = options.get("num_thread") if isinstance(options, Mapping) else None
    if isinstance(own, bool) or not isinstance(own, int) or own < 1:
        return
    governor = _governor
    if governor is None or not governor.config.enabled:
        return
    pinned = getattr(governor, "pinned_threads", None)
    pin = getattr(governor, "pin_threads", None)
    if not callable(pinned) or not callable(pin):
        return
    if pinned(model) != (own, own):
        pin(model, own, own)


def running_governor() -> ResourceGovernor | None:
    """The governor this process runs, or None when none has started:
    reading what only a running governor holds (a resident's pinned thread
    count) never starts one, with its store and its refresh."""
    return _governor


def server_threads() -> int | None:
    """The most CPU threads a computation in the server's own process may
    take (``ResourceGovernor.server_threads``), for an engine that runs
    there (llama.cpp in process); None with no governor, a disabled one, or
    one that cannot say."""
    governor = _governor
    if governor is None or not governor.config.enabled:
        return None
    read = getattr(governor, "server_threads", None)
    if not callable(read):
        return None
    return read()


def backend_admission_gate(
    model: str, options: dict | None = None, unsplittable: str | None = None
) -> AdmissionDecision | None:
    """The internal hook body behind the four generate/stream heads.

    A matching ticket means the funnel already decided: account the load
    (the invalidate_on_load wiring, once per ticket) and stand down. A
    ticketless call gets the fast cached admit-or-refuse backstop with
    default semantics (caller "direct", the mechanical backstop for the
    Section 8 residual), raising the typed GovernorRefusal on a positive
    refusal only. The caller (inference_backend) wraps this in its own
    fail-open handling; a disabled governor stands down entirely. Returns
    the decision acted on, the ticket or the backstop's, so the engine can
    load what it admits (num_ctx, num_gpu); None when disabled.

    ``unsplittable`` names the calling engine when it cannot place a model's
    layers itself. A split admission that carries no layer count, from the
    ticket or the backstop, is then refused by name before any eviction,
    accounting or load, instead of failing for memory inside the engine; a
    split that carries one is loaded with it.
    """
    governor = get_resource_governor()
    if not governor.config.enabled:
        return None
    ticket = get_active_ticket()
    if ticket is not None and ticket.model == model:
        if _uncounted_split(ticket, unsplittable):
            _refuse_split(governor, ticket, unsplittable)
        if ticket.admitted and ticket.load_expected:
            # Act on a conditional grant just before its load
            # (oldest-idle first, only as much as the shortfall needs;
            # fail-open to Ollama's own LRU, Section 12).
            if ticket.conditional_on_eviction:
                governor._honour_conditional_eviction(ticket)
            _account_load(governor, model, ticket, options)
            ticket.load_expected = False
        return ticket
    requested: int | None = None
    if isinstance(options, dict):
        raw = options.get("num_ctx")
        if isinstance(raw, int) and raw > 0:
            requested = raw
    decision = governor.admit(model, requested, caller="direct")
    if not decision.admitted:
        raise GovernorRefusal(decision)
    if _uncounted_split(decision, unsplittable):
        _refuse_split(governor, decision, unsplittable)
    if decision.load_expected:
        if decision.conditional_on_eviction:
            governor._honour_conditional_eviction(decision)
        _account_load(governor, model, decision, options)
    return decision


def _uncounted_split(decision: AdmissionDecision, unsplittable: str | None) -> bool:
    """A split load the calling engine cannot place: no layer count to load."""
    return bool(
        unsplittable
        and decision.partial_offload
        and decision.load_expected
        and decision.num_gpu is None
    )
