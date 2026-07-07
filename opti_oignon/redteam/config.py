#!/usr/bin/env python3
"""
Red Team Configuration — load & validate config/redteam.yaml.

Provides RedTeamConfig dataclass and load_redteam_config() helper.
"""

__all__ = ["RedTeamConfig", "SchedulerConfig", "load_redteam_config"]

import logging
import os
import re
from dataclasses import dataclass, field
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

logger = logging.getLogger(__name__)

_TIME_RE = re.compile(r"^\d{2}:\d{2}$")


def _is_loopback_url(url: str) -> bool:
    """Return True only if url addresses the local host.

    The hostname must be the literal "localhost" or an address in a
    loopback range (127.0.0.0/8, ::1). No DNS resolution is performed:
    an unresolved hostname is treated as non-local and refused, so the
    check itself never reaches the network.
    """
    host = (urlparse(url).hostname or "").strip()
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _assert_loopback(url: str) -> None:
    """Raise ValueError unless url addresses the local host."""
    if not _is_loopback_url(url):
        raise ValueError(
            "Ollama endpoint must be on the local host (loopback); "
            f"refusing non-local URL: {url!r}"
        )


@dataclass
class SchedulerConfig:
    """Configuration for automated security scheduling (S158)."""

    enabled: bool = True
    interval: str = "weekly"
    quiet_hours_start: str = "00:00"
    quiet_hours_end: str = "06:00"
    auto_accept_suggestions: bool = False
    dep_audit_interval: str = "weekly"
    dep_severity_threshold: str = "high"

    def __post_init__(self) -> None:
        """Validate scheduler configuration values."""
        if self.interval not in VALID_INTERVALS:
            raise ValueError(
                f"Invalid scheduler interval: {self.interval!r} "
                f"(valid: {sorted(VALID_INTERVALS)})"
            )
        if self.dep_audit_interval not in VALID_INTERVALS:
            raise ValueError(
                f"Invalid dep_audit_interval: {self.dep_audit_interval!r} "
                f"(valid: {sorted(VALID_INTERVALS)})"
            )
        if self.dep_severity_threshold not in VALID_SEVERITIES:
            raise ValueError(
                f"Invalid dep_severity_threshold: {self.dep_severity_threshold!r} "
                f"(valid: {sorted(VALID_SEVERITIES)})"
            )
        for label, value in [
            ("quiet_hours_start", self.quiet_hours_start),
            ("quiet_hours_end", self.quiet_hours_end),
        ]:
            if not _TIME_RE.match(value):
                raise ValueError(
                    f"Invalid {label} format: {value!r} (expected HH:MM)"
                )
            hh, mm = int(value[:2]), int(value[3:])
            if hh > 23 or mm > 59:
                raise ValueError(
                    f"Invalid {label} value: {value!r} "
                    f"(hours 00-23, minutes 00-59)"
                )

    @property
    def interval_seconds(self) -> int:
        """Return interval in seconds for timer scheduling."""
        if self.interval == "daily":
            return 86400
        if self.interval == "weekly":
            return 604800
        # on-deploy has no fixed interval
        return 0

    @property
    def dep_audit_interval_seconds(self) -> int:
        """Return dep audit interval in seconds."""
        if self.dep_audit_interval == "daily":
            return 86400
        if self.dep_audit_interval == "weekly":
            return 604800
        return 0

# All supported attack categories
VALID_CATEGORIES = frozenset({
    "prompt_injection",
    "jailbreak",
    "rag_poisoning",
    "data_exfiltration",
    "tool_hijack",
    "delimiter_escape",
    "off_topic",
    "encoding_bypass",
})

# All supported strategies
VALID_STRATEGIES = frozenset({
    "none",
    "base64_encode",
    "rot13",
    "leetspeak",
    "multilingual",
    "roleplay",
    "few_shot",
    "payload_splitting",
    "char_swap",
})

# All supported targets
VALID_TARGETS = frozenset({
    "rag_sanitizer",
    "rag_augmenter",
    "search_sanitizer",
    "pii_sanitizer",
    "sandbox",
    "chat",
})

# Scheduler interval options
VALID_INTERVALS = frozenset({"daily", "weekly", "on-deploy"})

# Dependency audit severity thresholds
VALID_SEVERITIES = frozenset({"low", "medium", "high", "critical"})

# Default config path
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_DEFAULT_CONFIG_PATH = _DEFAULT_CONFIG_DIR / "redteam.yaml"


@dataclass
class RedTeamConfig:
    """Validated red team configuration."""

    enabled: bool = True
    model: str = "llama3.2"
    ollama_url: str = "http://127.0.0.1:11434"

    # Attack generation
    categories: list[str] = field(default_factory=lambda: list(VALID_CATEGORIES))
    attacks_per_category: int = 10
    min_attack_length: int = 10
    max_attack_length: int = 2000
    batch_size: int = 5
    seed_fallback: bool = True
    seed_file: str = "data/redteam_seeds.json"

    # Strategies
    strategies: list[str] = field(default_factory=lambda: ["none", "base64_encode", "rot13"])
    strategy_chains: list[list[str]] = field(default_factory=list)

    # Targets
    targets: list[str] = field(default_factory=lambda: list(VALID_TARGETS))

    # Scoring
    bypass_threshold: float = 0.7
    flag_threshold: float = 0.3

    # Output
    output_dir: str = "data/redteam_results"
    save_attacks: bool = True
    save_results: bool = True

    # Per-category toggles
    category_toggles: dict[str, bool] = field(default_factory=dict)

    # S158: scheduler configuration
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # The red team must never reach off the local host.
        _assert_loopback(self.ollama_url)

        # Validate categories
        invalid_cats = set(self.categories) - VALID_CATEGORIES
        if invalid_cats:
            raise ValueError(f"Invalid attack categories: {invalid_cats}")

        # Validate strategies
        invalid_strats = set(self.strategies) - VALID_STRATEGIES
        if invalid_strats:
            raise ValueError(f"Invalid strategies: {invalid_strats}")

        # Validate strategy chains
        for chain in self.strategy_chains:
            invalid_chain = set(chain) - VALID_STRATEGIES
            if invalid_chain:
                raise ValueError(f"Invalid strategies in chain: {invalid_chain}")

        # Validate targets
        invalid_targets = set(self.targets) - VALID_TARGETS
        if invalid_targets:
            raise ValueError(f"Invalid targets: {invalid_targets}")

        # Validate numeric ranges
        if self.attacks_per_category < 1:
            raise ValueError("attacks_per_category must be >= 1")
        if self.min_attack_length < 1:
            raise ValueError("min_attack_length must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not 0.0 <= self.bypass_threshold <= 1.0:
            raise ValueError("bypass_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.flag_threshold <= 1.0:
            raise ValueError("flag_threshold must be between 0.0 and 1.0")

        # Validate category toggles
        invalid_toggles = set(self.category_toggles.keys()) - VALID_CATEGORIES
        if invalid_toggles:
            raise ValueError(f"Invalid category toggle keys: {invalid_toggles}")


def load_redteam_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> RedTeamConfig:
    """Load red team configuration from YAML file.

    Parameters
    ----------
    config_path : str or Path or None
        Path to redteam.yaml. Uses default if None.
    overrides : dict or None
        Key-value overrides applied after file loading.

    Returns
    -------
    RedTeamConfig
        Validated configuration.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    raw: dict[str, Any] = {}
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            logger.info("Loaded red team config from %s", path)
        except Exception as exc:
            logger.warning("Failed to load %s: %s — using defaults", path, exc)
    else:
        logger.info("No red team config at %s — using defaults", path)

    # Apply overrides
    if overrides:
        raw.update(overrides)

    # Environment variable override for enabled flag
    env_flag = os.environ.get("OPTI_REDTEAM_ENABLED")
    if env_flag is not None:
        raw["enabled"] = env_flag.lower() in ("1", "true", "yes")

    # Build config from raw dict
    known_fields = {f.name for f in RedTeamConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw.items() if k in known_fields}

    # S158: convert nested scheduler dict to SchedulerConfig
    if "scheduler" in filtered and isinstance(filtered["scheduler"], dict):
        sched_known = {f.name for f in SchedulerConfig.__dataclass_fields__.values()}
        sched_filtered = {k: v for k, v in filtered["scheduler"].items() if k in sched_known}
        filtered["scheduler"] = SchedulerConfig(**sched_filtered)

    return RedTeamConfig(**filtered)
