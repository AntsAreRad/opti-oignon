#!/usr/bin/env python3
"""Guarded loader for the agent configuration (S176, Theme 3 / Odysseus Core).

Reads ``opti_oignon/agent/config.yaml`` and exposes the agent's configurable
parameters (ODYSSEUS_SPEC.md Section 5): the loop round cap, the verifier cap,
the teacher thresholds and model, and the per-mode tool exposure. The laptop-
lite preset is represented and selectable.

The loader is guarded so it is importlib-isolatable and behaviour never depends
on the file being present: ``yaml`` is imported lazily inside a try/except, and
if PyYAML or the file is unavailable the built-in ``CONFIG_DEFAULTS`` are used.
Those defaults match the S175 reference values (round cap 20, verifier cap 2)
and the teacher reference thresholds, so a missing config degrades to the same
secure behaviour rather than to something weaker.

There is one process-level cached config with ``reset_agent_config()`` for test
isolation.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Module conventions (Theme 3).
checkpoint_before_apply = True
FEATURE_AVAILABLE = True

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

# Reference defaults, mirroring the S175 loop reference and the teacher
# reference thresholds. Used verbatim when the file or PyYAML is unavailable.
CONFIG_DEFAULTS: dict[str, Any] = {
    "loop": {
        "round_cap": 20,
        "verifier_cap": 2,
        "verify": False,
    },
    "teacher": {
        "enabled": True,
        "failure_threshold": 2,
        "on_verifier_fail": True,
        "on_max_rounds": True,
        "on_model_error": True,
        "teacher_model": "qwen3:32b",
        "student_model": "qwen3:4b",
        "timeout": 120.0,
    },
    "tools": {
        "daily": ["bash", "view", "create_file", "str_replace", "web_search", "manage_memory", "manage_skills"],
        "bulbe": ["bash", "view", "create_file", "str_replace"],
    },
}

# Hard bounds, mirroring the loop's own clamps so config can never widen them.
_ROUND_CAP_MIN = 1
_ROUND_CAP_MAX = 1000
_VERIFIER_CAP_MIN = 1
_VERIFIER_CAP_MAX = 2

LAPTOP_LITE = "laptop_lite"


@dataclass
class AgentConfig:
    """The resolved agent configuration for a deployment (or preset)."""

    round_cap: int = 20
    verifier_cap: int = 2
    verify: bool = False
    teacher: dict[str, Any] = field(default_factory=dict)
    daily_tools: tuple[str, ...] = ()
    bulbe_tools: tuple[str, ...] = ()
    preset: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def teacher_policy(self) -> Any:
        """Build a ``teacher.EscalationPolicy`` from the teacher config.

        The import is lazy so the loader stays isolatable: it loads without
        ``teacher`` present, and only constructs a policy on demand.
        """
        from opti_oignon.agent.teacher import EscalationPolicy

        return EscalationPolicy.from_dict(self.teacher)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_cap": self.round_cap,
            "verifier_cap": self.verifier_cap,
            "verify": self.verify,
            "teacher": dict(self.teacher),
            "daily_tools": list(self.daily_tools),
            "bulbe_tools": list(self.bulbe_tools),
            "preset": self.preset,
        }


# YAML loading (guarded)


def _load_yaml() -> Any:
    """Return the ``yaml`` module, or None when PyYAML is unavailable."""
    try:
        import yaml

        return yaml
    except Exception:  # pragma: no cover - defensive guard
        return None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto a copy of ``base``."""
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config_data(path: str | Path | None = None) -> dict[str, Any]:
    """Load the raw config mapping, merged onto the defaults.

    A missing file or absent PyYAML yields the defaults (a copy). A malformed
    file is logged and falls back to the defaults; this function never raises.
    """
    defaults = copy.deepcopy(CONFIG_DEFAULTS)  # a private deep copy, not shared
    target = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not target.exists():
        return defaults
    yaml = _load_yaml()
    if yaml is None:
        return defaults
    try:
        with open(target, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        logger.warning("Agent config unreadable at %s; using defaults", target)
        return defaults
    if not isinstance(data, dict):
        return defaults
    return _deep_merge(defaults, data)


# Coercion helpers


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(value, hi))


def _as_tuple(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return default


def _apply_preset(data: dict[str, Any], preset: str | None) -> dict[str, Any]:
    if not preset:
        return data
    presets = data.get("presets")
    if not isinstance(presets, dict) or preset not in presets:
        return data
    override = presets.get(preset)
    if not isinstance(override, dict):
        return data
    return _deep_merge(data, override)


def _build(data: dict[str, Any], preset: str | None) -> AgentConfig:
    loop = data.get("loop") if isinstance(data.get("loop"), dict) else {}
    teacher = data.get("teacher") if isinstance(data.get("teacher"), dict) else {}
    tools = data.get("tools") if isinstance(data.get("tools"), dict) else {}

    round_cap = _clamp(
        _as_int(loop.get("round_cap"), CONFIG_DEFAULTS["loop"]["round_cap"]),
        _ROUND_CAP_MIN,
        _ROUND_CAP_MAX,
    )
    verifier_cap = _clamp(
        _as_int(loop.get("verifier_cap"), CONFIG_DEFAULTS["loop"]["verifier_cap"]),
        _VERIFIER_CAP_MIN,
        _VERIFIER_CAP_MAX,
    )
    verify = bool(loop.get("verify", CONFIG_DEFAULTS["loop"]["verify"]))

    daily = _as_tuple(tools.get("daily"), tuple(CONFIG_DEFAULTS["tools"]["daily"]))
    bulbe = _as_tuple(tools.get("bulbe"), tuple(CONFIG_DEFAULTS["tools"]["bulbe"]))

    return AgentConfig(
        round_cap=round_cap,
        verifier_cap=verifier_cap,
        verify=verify,
        teacher=dict(teacher),
        daily_tools=daily,
        bulbe_tools=bulbe,
        preset=preset,
        raw=data,
    )


def load_config(path: str | Path | None = None, *, preset: str | None = None) -> AgentConfig:
    """Load and resolve the agent config, optionally applying a named preset."""
    data = load_config_data(path)
    data = _apply_preset(data, preset)
    return _build(data, preset)


def available_presets(path: str | Path | None = None) -> list[str]:
    """The preset names declared in the config (empty when none / on error)."""
    data = load_config_data(path)
    presets = data.get("presets")
    if isinstance(presets, dict):
        return sorted(presets.keys())
    return []


_CONFIG: AgentConfig | None = None


def get_agent_config() -> AgentConfig:
    """The process-level agent config, loaded once from the default path."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG


def set_agent_config(config: AgentConfig) -> None:
    """Install a config (e.g. a preset-resolved one) process-wide."""
    global _CONFIG
    _CONFIG = config


def reset_agent_config() -> None:
    """Drop the cached config so tests do not leak state across runs."""
    global _CONFIG
    _CONFIG = None
