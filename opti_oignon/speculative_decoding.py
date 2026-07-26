#!/usr/bin/env python3
"""
SPECULATIVE DECODING ENGINE -- OPTI-OIGNON
=======================================================

Pairs a small draft model with the main model for 2-5x token generation
speedup via llama.cpp's native speculative decoding flags. Zero quality
loss -- the main model verifies every draft token.

This module is separate from speculative.py (prompt-level
draft-verify pattern). Here we leverage llama.cpp's --draft-max /
--draft-min / -md flags for hardware-level speculative decoding.

Only available when using the llama.cpp backend. Entirely opt-in.

Architecture:
    SpeculativeConfig       -- dataclass holding user config
    DraftModelSelector      -- auto-detect compatible draft models
    VRAMBudgetCalculator    -- estimate if main + draft fit in VRAM
    SpeculativeDecodingManager -- orchestrate config, selection, stats

Additions:
    AcceptanceRecord        -- per-request acceptance data
    AcceptanceStats.history -- rolling deque of per-request records
    parse_llamacpp_log_line -- extract acceptance data from llama.cpp logs

Addition:
    build_llama_server_command -- the pure argv materialisation that
    finally wires this module's config to the external llama-server
    (consumed through inference_backend.LlamaServerBackend; the process
    is launched host-side per INFERENCE_PERF_S259.md, never spawned
    here).
"""

import collections
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path(__file__).parent / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "speculative_decoding.yaml"
_RESULTS_PATH = Path(__file__).parent.parent / "data" / "speculative_stats.json"

# Rough VRAM estimates per billion parameters by quantization level (GB).
# These are approximate; real usage depends on context length and batch size.
_VRAM_PER_BILLION_PARAMS: dict[str, float] = {
    "Q2_K": 0.40,
    "Q3_K_S": 0.45,
    "Q3_K_M": 0.48,
    "Q3_K_L": 0.50,
    "Q4_0": 0.55,
    "Q4_K_S": 0.55,
    "Q4_K_M": 0.58,
    "Q4_K_L": 0.60,
    "Q5_0": 0.65,
    "Q5_K_S": 0.65,
    "Q5_K_M": 0.68,
    "Q6_K": 0.78,
    "Q8_0": 1.00,
    "F16": 2.00,
    "F32": 4.00,
    "BF16": 2.00,
}

# Default family compatibility (loaded from YAML, this is fallback).
_DEFAULT_FAMILY_COMPAT: dict[str, list[dict]] = {
    "llama3": [{"family": "llama3", "max_params_b": 3}],
    "llama": [{"family": "llama", "max_params_b": 3}],
    "qwen": [{"family": "qwen", "max_params_b": 3}],
    "qwen2": [{"family": "qwen2", "max_params_b": 3}],
    "deepseek": [{"family": "deepseek", "max_params_b": 3}],
    "gemma": [{"family": "gemma", "max_params_b": 2}],
    "phi": [{"family": "phi", "max_params_b": 3}],
    "mistral": [{"family": "mistral", "max_params_b": 3}],
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding via llama.cpp."""

    enabled: bool = False
    draft_model: str = ""
    draft_max: int = 16
    draft_min: int = 5
    draft_gpu_layers: int = 99
    auto_select_draft: bool = True

    def validate(self) -> list[str]:
        """Return a list of validation error messages (empty = valid)."""
        errors: list[str] = []
        if self.draft_max < 1:
            errors.append("draft_max must be >= 1")
        if self.draft_min < 1:
            errors.append("draft_min must be >= 1")
        if self.draft_min > self.draft_max:
            errors.append("draft_min must be <= draft_max")
        if self.draft_gpu_layers < -1:
            errors.append("draft_gpu_layers must be >= -1")
        return errors

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "enabled": self.enabled,
            "draft_model": self.draft_model,
            "draft_max": self.draft_max,
            "draft_min": self.draft_min,
            "draft_gpu_layers": self.draft_gpu_layers,
            "auto_select_draft": self.auto_select_draft,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpeculativeConfig":
        """Create from dict, ignoring unknown keys."""
        known = {
            "enabled", "draft_model", "draft_max", "draft_min",
            "draft_gpu_layers", "auto_select_draft",
        }
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class DraftCandidate:
    """A candidate draft model with compatibility info."""

    name: str
    path: str = ""
    family: str = ""
    parameter_size_b: float = 0.0
    quantization: str = ""
    estimated_vram_gb: float = 0.0
    compatibility_score: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "name": self.name,
            "path": self.path,
            "family": self.family,
            "parameter_size_b": self.parameter_size_b,
            "quantization": self.quantization,
            "estimated_vram_gb": self.estimated_vram_gb,
            "compatibility_score": self.compatibility_score,
        }


@dataclass
class AcceptanceRecord:
    """Single per-request acceptance record."""

    timestamp: float = 0.0
    draft_tokens: int = 0
    accepted_tokens: int = 0
    acceptance_rate: float = 0.0
    speedup_factor: float = 1.0
    request_id: str = ""

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "timestamp": round(self.timestamp, 3),
            "draft_tokens": self.draft_tokens,
            "accepted_tokens": self.accepted_tokens,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "speedup_factor": round(self.speedup_factor, 2),
            "request_id": self.request_id,
        }


# Maximum per-request records kept in memory.
_MAX_ACCEPTANCE_HISTORY = 200


@dataclass
class AcceptanceStats:
    """Tracks speculative decoding acceptance rate statistics."""

    total_draft_tokens: int = 0
    accepted_tokens: int = 0
    total_runs: int = 0
    last_acceptance_rate: float = 0.0
    last_speedup_factor: float = 1.0
    last_updated: float = 0.0

    # Per-request history for real-time monitoring.
    _history: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=_MAX_ACCEPTANCE_HISTORY),
        repr=False,
    )

    @property
    def overall_acceptance_rate(self) -> float:
        """Overall acceptance rate across all runs."""
        if self.total_draft_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.total_draft_tokens

    def record_run(
        self,
        draft_tokens: int,
        accepted: int,
        speedup: float = 1.0,
        request_id: str = "",
    ) -> None:
        """Record stats from a single speculative decoding run."""
        self.total_draft_tokens += draft_tokens
        self.accepted_tokens += accepted
        self.total_runs += 1
        self.last_acceptance_rate = (
            accepted / draft_tokens if draft_tokens > 0 else 0.0
        )
        self.last_speedup_factor = speedup
        self.last_updated = time.time()

        # Append per-request record to history.
        self._history.append(AcceptanceRecord(
            timestamp=self.last_updated,
            draft_tokens=draft_tokens,
            accepted_tokens=accepted,
            acceptance_rate=self.last_acceptance_rate,
            speedup_factor=speedup,
            request_id=request_id,
        ))

    def get_history(self, last_n: int = 0) -> list[dict]:
        """Get per-request acceptance history.

        Args:
            last_n: If > 0, return only the last N records.
                If 0, return all available records.

        Returns:
            List of AcceptanceRecord dicts, oldest first.
        """
        records = list(self._history)
        if last_n > 0:
            records = records[-last_n:]
        return [r.to_dict() for r in records]

    def get_rolling_acceptance_rate(self, last_n: int = 10) -> float:
        """Calculate rolling acceptance rate over the last N runs.

        Returns 0.0 if no history is available.
        """
        recent = list(self._history)[-last_n:] if self._history else []
        if not recent:
            return 0.0
        total_draft = sum(r.draft_tokens for r in recent)
        total_accepted = sum(r.accepted_tokens for r in recent)
        if total_draft == 0:
            return 0.0
        return total_accepted / total_draft

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "total_draft_tokens": self.total_draft_tokens,
            "accepted_tokens": self.accepted_tokens,
            "total_runs": self.total_runs,
            "overall_acceptance_rate": round(self.overall_acceptance_rate, 4),
            "last_acceptance_rate": round(self.last_acceptance_rate, 4),
            "last_speedup_factor": round(self.last_speedup_factor, 2),
            "last_updated": self.last_updated,
            "history_size": len(self._history),
            "rolling_acceptance_rate": round(
                self.get_rolling_acceptance_rate(10), 4
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AcceptanceStats":
        """Create from dict."""
        return cls(
            total_draft_tokens=data.get("total_draft_tokens", 0),
            accepted_tokens=data.get("accepted_tokens", 0),
            total_runs=data.get("total_runs", 0),
            last_acceptance_rate=data.get("last_acceptance_rate", 0.0),
            last_speedup_factor=data.get("last_speedup_factor", 1.0),
            last_updated=data.get("last_updated", 0.0),
        )


# ---------------------------------------------------------------------------
# VRAM Budget Calculator
# ---------------------------------------------------------------------------

class VRAMBudgetCalculator:
    """Estimate VRAM usage for main + draft model pairs."""

    def __init__(
        self,
        total_vram_gb: float = 24.0,
        safety_margin_gb: float = 1.5,
    ):
        self._total_vram_gb = total_vram_gb
        self._safety_margin_gb = safety_margin_gb

    @property
    def available_vram_gb(self) -> float:
        """Usable VRAM after safety margin."""
        return max(0.0, self._total_vram_gb - self._safety_margin_gb)

    def estimate_model_vram(
        self,
        parameter_size_b: float,
        quantization: str = "Q4_K_M",
    ) -> float:
        """Estimate VRAM in GB for a model.

        Args:
            parameter_size_b: Parameter count in billions.
            quantization: Quantization level (e.g. Q4_K_M, Q8_0, F16).

        Returns:
            Estimated VRAM in GB.
        """
        quant_upper = quantization.upper().replace("-", "_")
        per_billion = _VRAM_PER_BILLION_PARAMS.get(quant_upper, 0.58)
        return parameter_size_b * per_billion

    def check_fit(
        self,
        main_params_b: float,
        main_quant: str,
        draft_params_b: float,
        draft_quant: str,
    ) -> dict:
        """Check if main + draft fit within VRAM budget.

        Returns:
            Dict with 'fits', 'main_vram_gb', 'draft_vram_gb',
            'total_vram_gb', 'available_vram_gb', 'headroom_gb'.
        """
        main_vram = self.estimate_model_vram(main_params_b, main_quant)
        draft_vram = self.estimate_model_vram(draft_params_b, draft_quant)
        total = main_vram + draft_vram
        available = self.available_vram_gb
        headroom = available - total

        return {
            "fits": headroom >= 0,
            "main_vram_gb": round(main_vram, 2),
            "draft_vram_gb": round(draft_vram, 2),
            "total_vram_gb": round(total, 2),
            "available_vram_gb": round(available, 2),
            "headroom_gb": round(headroom, 2),
        }


# ---------------------------------------------------------------------------
# Draft Model Selector
# ---------------------------------------------------------------------------

class DraftModelSelector:
    """Auto-detect compatible draft models from installed models.

    Uses family compatibility mapping to find suitable small draft
    models that share the same tokenizer family as the main model.
    """

    def __init__(
        self,
        family_compat: dict[str, list[dict]] | None = None,
        vram_calculator: VRAMBudgetCalculator | None = None,
    ):
        self._family_compat = family_compat or _DEFAULT_FAMILY_COMPAT
        self._vram_calc = vram_calculator or VRAMBudgetCalculator()

    def find_compatible_drafts(
        self,
        main_model_family: str,
        main_model_params_b: float,
        main_model_quant: str,
        available_models: list[dict],
    ) -> list[DraftCandidate]:
        """Find draft models compatible with the given main model.

        Args:
            main_model_family: Family string of the main model
                (e.g. "llama3", "qwen2").
            main_model_params_b: Main model parameter count in billions.
            main_model_quant: Main model quantization level.
            available_models: List of dicts with keys: name, family,
                parameter_size_b, quantization, path.

        Returns:
            Sorted list of DraftCandidate (best first).
        """
        family_lower = (main_model_family or "").lower().strip()

        # Find matching family rules.
        compat_rules: list[dict] = []
        for key, rules in self._family_compat.items():
            if family_lower.startswith(key.lower()):
                compat_rules = rules
                break

        if not compat_rules:
            logger.debug(
                "No compatibility rules for family '%s'", main_model_family
            )
            return []

        candidates: list[DraftCandidate] = []

        for model in available_models:
            model_family = (model.get("family") or "").lower().strip()
            model_params = _parse_param_size(model.get("parameter_size_b"))
            model_quant = model.get("quantization") or "Q4_K_M"
            model_name = model.get("name", "")
            model_path = model.get("path", "")

            # Skip if same size or larger than main model.
            if model_params >= main_model_params_b:
                continue

            # Check against compatibility rules.
            for rule in compat_rules:
                rule_family = rule.get("family", "").lower()
                max_params = rule.get("max_params_b", 3)

                if (
                    model_family.startswith(rule_family)
                    and model_params <= max_params
                    and model_params > 0
                ):
                    # Compute a compatibility score.
                    # Prefer: same family > smaller > fits VRAM.
                    score = 1.0

                    # Same exact family prefix gets a bonus.
                    if model_family == family_lower:
                        score += 2.0

                    # Prefer larger drafts (closer to main) within limits.
                    if max_params > 0:
                        score += (model_params / max_params) * 1.0

                    # Penalize if it won't fit in VRAM.
                    budget = self._vram_calc.check_fit(
                        main_model_params_b, main_model_quant,
                        model_params, model_quant,
                    )
                    if not budget["fits"]:
                        score *= 0.1  # Heavy penalty but don't exclude.

                    estimated_vram = self._vram_calc.estimate_model_vram(
                        model_params, model_quant,
                    )

                    candidates.append(DraftCandidate(
                        name=model_name,
                        path=model_path,
                        family=model_family,
                        parameter_size_b=model_params,
                        quantization=model_quant,
                        estimated_vram_gb=round(estimated_vram, 2),
                        compatibility_score=round(score, 3),
                    ))
                    break  # One match per model is enough.

        # Sort by score descending (best first).
        candidates.sort(key=lambda c: c.compatibility_score, reverse=True)
        return candidates

    def auto_select(
        self,
        main_model_family: str,
        main_model_params_b: float,
        main_model_quant: str,
        available_models: list[dict],
    ) -> DraftCandidate | None:
        """Pick the single best draft model for the main model.

        Returns None if no compatible draft is found.
        """
        drafts = self.find_compatible_drafts(
            main_model_family, main_model_params_b,
            main_model_quant, available_models,
        )
        return drafts[0] if drafts else None


# ---------------------------------------------------------------------------
# Speculative Decoding Manager (singleton)
# ---------------------------------------------------------------------------

class SpeculativeDecodingManager:
    """Orchestrates speculative decoding config, draft selection, and stats.

    This is the main entry point for the speculative decoding feature.
    It manages the SpeculativeConfig, auto-selects draft models, builds
    llama.cpp CLI flags, and tracks acceptance rate stats.
    """

    def __init__(self, config_path: str | None = None, stats_path: str | None = None):
        self._config = SpeculativeConfig()
        self._family_compat = dict(_DEFAULT_FAMILY_COMPAT)
        self._vram_budget_cfg: dict = {}
        self._stats = AcceptanceStats()
        self._lock = threading.RLock()
        # Stats path is injectable so tests can use an isolated temp file
        # instead of the shared data/speculative_stats.json (which otherwise
        # leaks state across pytest invocations). Production behavior is
        # unchanged: stats_path=None falls back to the module default.
        self._stats_path = Path(stats_path) if stats_path else _RESULTS_PATH
        self._load_config(config_path)

    def _load_config(self, config_path: str | None = None) -> None:
        """Load configuration from YAML."""
        p = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        if not p.is_file():
            logger.debug("No speculative_decoding.yaml found at %s", p)
            return

        try:
            with open(p, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning("Failed to load speculative_decoding.yaml: %s", exc)
            return

        # Load main section.
        sd_cfg = raw.get("speculative_decoding", {})
        if isinstance(sd_cfg, dict):
            self._config = SpeculativeConfig.from_dict(sd_cfg)

        # Load family compatibility.
        fc = raw.get("family_compatibility")
        if isinstance(fc, dict):
            self._family_compat = fc

        # Load VRAM budget.
        vb = raw.get("vram_budget")
        if isinstance(vb, dict):
            self._vram_budget_cfg = vb

        # Load persisted stats.
        self._load_stats()

        logger.info(
            "Speculative decoding config loaded: enabled=%s, draft_model=%s",
            self._config.enabled, self._config.draft_model or "(auto)",
        )

    def _load_stats(self) -> None:
        """Load persisted acceptance stats from disk."""
        if not self._stats_path.is_file():
            return
        try:
            with open(self._stats_path, encoding="utf-8") as f:
                data = json.load(f)
            self._stats = AcceptanceStats.from_dict(data)
        except Exception as exc:
            logger.debug("Failed to load speculative stats: %s", exc)

    def _save_stats(self) -> None:
        """Persist acceptance stats to disk."""
        try:
            self._stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._stats_path, "w", encoding="utf-8") as f:
                json.dump(self._stats.to_dict(), f, indent=2)
        except Exception as exc:
            logger.debug("Failed to save speculative stats: %s", exc)

    # -- Public API --

    @property
    def config(self) -> SpeculativeConfig:
        """Current configuration (read-only copy)."""
        with self._lock:
            return SpeculativeConfig(**self._config.to_dict())

    @property
    def stats(self) -> AcceptanceStats:
        """Current acceptance stats (read-only copy)."""
        with self._lock:
            return AcceptanceStats.from_dict(self._stats.to_dict())

    def update_config(self, updates: dict) -> SpeculativeConfig:
        """Update configuration with partial dict.

        Args:
            updates: Partial config dict (e.g. {"enabled": True}).

        Returns:
            Updated SpeculativeConfig.

        Raises:
            ValueError: If validation fails after applying updates.
        """
        with self._lock:
            merged = {**self._config.to_dict(), **updates}
            new_cfg = SpeculativeConfig.from_dict(merged)
            errors = new_cfg.validate()
            if errors:
                raise ValueError(
                    f"Invalid speculative decoding config: {'; '.join(errors)}"
                )
            self._config = new_cfg
            logger.info(
                "Speculative decoding config updated: %s", new_cfg.to_dict()
            )
            return SpeculativeConfig(**new_cfg.to_dict())

    def get_status(self) -> dict:
        """Get full status including config, stats, and availability."""
        with self._lock:
            return {
                "config": self._config.to_dict(),
                "stats": self._stats.to_dict(),
                "available": True,
                "backend_required": "llama_cpp",
            }

    def record_acceptance(
        self,
        draft_tokens: int,
        accepted: int,
        speedup: float = 1.0,
        request_id: str = "",
    ) -> None:
        """Record acceptance stats from a decoding run."""
        with self._lock:
            self._stats.record_run(
                draft_tokens, accepted, speedup,
                request_id=request_id,
            )
            self._save_stats()

    def get_acceptance_history(self, last_n: int = 50) -> list[dict]:
        """Get per-request acceptance history.

        Args:
            last_n: Number of recent records to return (0 = all).

        Returns:
            List of AcceptanceRecord dicts, oldest first.
        """
        with self._lock:
            return self._stats.get_history(last_n=last_n)

    def get_rolling_acceptance_rate(self, window: int = 10) -> float:
        """Get rolling acceptance rate over last N requests."""
        with self._lock:
            return self._stats.get_rolling_acceptance_rate(last_n=window)

    def process_log_line(self, line: str) -> bool:
        """Parse a llama.cpp server log line for acceptance data.

        When using llama-server with speculative decoding, it logs
        acceptance stats like:
            "draft accepted X/Y tokens (Z%)"
            "speculative: accepted X, drafted Y"

        Args:
            line: A single line from llama.cpp server stderr/stdout.

        Returns:
            True if the line contained acceptance data that was recorded.
        """
        parsed = parse_llamacpp_log_line(line)
        if parsed is not None:
            self.record_acceptance(
                draft_tokens=parsed["draft_tokens"],
                accepted=parsed["accepted_tokens"],
                speedup=parsed.get("speedup", 1.0),
            )
            return True
        return False

    def reset_stats(self) -> None:
        """Clear acceptance stats."""
        with self._lock:
            self._stats = AcceptanceStats()
            self._save_stats()

    def build_llama_cpp_flags(self) -> list[str]:
        """Build llama.cpp CLI flags for speculative decoding.

        Returns:
            List of CLI flag strings to append to the llama.cpp
            server command, or empty list if disabled.
        """
        with self._lock:
            if not self._config.enabled:
                return []

            if not self._config.draft_model:
                logger.warning(
                    "Speculative decoding enabled but no draft model set"
                )
                return []

            flags: list[str] = [
                "-md", self._config.draft_model,
                "--draft-max", str(self._config.draft_max),
                "--draft-min", str(self._config.draft_min),
                "-ngld", str(self._config.draft_gpu_layers),
            ]
            return flags

    def get_draft_selector(self) -> DraftModelSelector:
        """Create a DraftModelSelector with current config."""
        vram_total = self._vram_budget_cfg.get("default_total_gb", 24.0)
        vram_margin = self._vram_budget_cfg.get("safety_margin_gb", 1.5)
        calc = VRAMBudgetCalculator(
            total_vram_gb=vram_total,
            safety_margin_gb=vram_margin,
        )
        return DraftModelSelector(
            family_compat=self._family_compat,
            vram_calculator=calc,
        )

    def get_vram_calculator(self) -> VRAMBudgetCalculator:
        """Create a VRAMBudgetCalculator with current config."""
        vram_total = self._vram_budget_cfg.get("default_total_gb", 24.0)
        vram_margin = self._vram_budget_cfg.get("safety_margin_gb", 1.5)
        return VRAMBudgetCalculator(
            total_vram_gb=vram_total,
            safety_margin_gb=vram_margin,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_manager: SpeculativeDecodingManager | None = None
_init_lock = threading.Lock()


def get_speculative_decoding_manager(
    config_path: str | None = None,
) -> SpeculativeDecodingManager:
    """Get or create the module-level singleton manager."""
    global _manager
    if _manager is not None:
        return _manager
    with _init_lock:
        if _manager is not None:
            return _manager
        _manager = SpeculativeDecodingManager(config_path=config_path)
        return _manager


def reset_manager() -> None:
    """Reset the singleton (for testing)."""
    global _manager
    with _init_lock:
        _manager = None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _parse_param_size(value: Any) -> float:
    """Parse a parameter size value to float (billions).

    Accepts: float, int, or string like '7B', '3.2b', '70b', '1.5B'.
    Returns 0.0 if unparseable.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip().upper()
        match = re.match(r"^([\d.]+)\s*B?$", value)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
    return 0.0


# ---------------------------------------------------------------------------
# Log parser for llama.cpp speculative decoding
# ---------------------------------------------------------------------------

# Regex patterns for llama.cpp speculative decoding log lines.
# Pattern 1: "draft accepted 12/16 tokens (75.00%)"
_RE_DRAFT_ACCEPTED = re.compile(
    r"draft\s+accepted\s+(\d+)\s*/\s*(\d+)\s*tokens",
    re.IGNORECASE,
)

# Pattern 2: "speculative: accepted 12, drafted 16"
_RE_SPECULATIVE_ACCEPTED = re.compile(
    r"speculative:\s*accepted\s+(\d+)\s*,\s*drafted\s+(\d+)",
    re.IGNORECASE,
)

# Pattern 3: "n_drafted = 16, n_accept = 12"
_RE_N_DRAFTED = re.compile(
    r"n_drafted\s*=\s*(\d+)\s*,\s*n_accept\s*=\s*(\d+)",
    re.IGNORECASE,
)

# Pattern 4: speedup info "speculative decoding speedup: 2.3x"
_RE_SPEEDUP = re.compile(
    r"speculative.*speedup[:\s]+([\d.]+)\s*x",
    re.IGNORECASE,
)


def parse_llamacpp_log_line(line: str) -> dict | None:
    """Parse a llama.cpp server log line for speculative acceptance data.

    Supports several log formats emitted by llama-server when running
    with speculative decoding (--draft-max / -md flags).

    Args:
        line: A single log line from llama.cpp server output.

    Returns:
        A dict with keys 'draft_tokens', 'accepted_tokens', and
        optionally 'speedup', or None if no acceptance data found.
    """
    if not line:
        return None

    # Try each pattern.
    m = _RE_DRAFT_ACCEPTED.search(line)
    if m:
        accepted = int(m.group(1))
        drafted = int(m.group(2))
        speedup = _extract_speedup(line)
        return {
            "draft_tokens": drafted,
            "accepted_tokens": accepted,
            "speedup": speedup,
        }

    m = _RE_SPECULATIVE_ACCEPTED.search(line)
    if m:
        accepted = int(m.group(1))
        drafted = int(m.group(2))
        speedup = _extract_speedup(line)
        return {
            "draft_tokens": drafted,
            "accepted_tokens": accepted,
            "speedup": speedup,
        }

    m = _RE_N_DRAFTED.search(line)
    if m:
        drafted = int(m.group(1))
        accepted = int(m.group(2))
        speedup = _extract_speedup(line)
        return {
            "draft_tokens": drafted,
            "accepted_tokens": accepted,
            "speedup": speedup,
        }

    return None


def _extract_speedup(line: str) -> float:
    """Extract speedup factor from a log line, default 1.0."""
    m = _RE_SPEEDUP.search(line)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return 1.0


def build_llama_server_command(
    model_path: str,
    config: SpeculativeConfig,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    n_ctx: int | None = None,
    flash_attn: bool = False,
    type_k: str | None = None,
    type_v: str | None = None,
) -> list[str]:
    """Materialise the llama-server argv from a SpeculativeConfig.

    PURE by contract: no filesystem reads, no process spawning, no
    state -- the same inputs always answer the same argv. Launching the
    server is host-side (INFERENCE_PERF_S259.md); the running process is
    consumed through inference_backend.LlamaServerBackend.

    Draft posture: an enabled config with a draft model emits the same
    flag quad as SpeculativeDecodingManager.build_llama_cpp_flags
    (-md / --draft-max / --draft-min / -ngld). An enabled config WITHOUT
    a draft model emits no draft flags at all: that is the MTP
    self-draft posture -- the draft lives inside the model file and the
    server applies it natively, so nothing external is wired (real
    acceptance rates are host-verified per the runbook). A disabled
    config likewise emits the bare server command.

    Validation is loud, never guessed: an empty model path or a config
    whose validate() reports errors raises ValueError.
    """
    if not model_path or not str(model_path).strip():
        raise ValueError("model_path must be a non-empty path")
    errors = config.validate()
    if errors:
        raise ValueError("invalid SpeculativeConfig: " + "; ".join(errors))

    cmd: list[str] = [
        "llama-server",
        "-m", str(model_path),
        "--host", str(host),
        "--port", str(int(port)),
    ]
    if n_ctx:
        cmd += ["-c", str(int(n_ctx))]
    if flash_attn:
        cmd.append("--flash-attn")
    if type_k:
        cmd += ["--cache-type-k", str(type_k)]
    if type_v:
        cmd += ["--cache-type-v", str(type_v)]
    if config.enabled and config.draft_model:
        cmd += [
            "-md", str(config.draft_model),
            "--draft-max", str(config.draft_max),
            "--draft-min", str(config.draft_min),
            "-ngld", str(config.draft_gpu_layers),
        ]
    return cmd
