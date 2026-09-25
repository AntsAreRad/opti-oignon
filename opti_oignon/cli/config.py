#!/usr/bin/env python3
"""
CLI configuration loader -- Opti-Oignon.

Reads and writes CLI settings from ``~/.config/opti-oignon/cli.yaml``.

Supported keys
--------------
- ``api_url``       : Base URL of the running backend (default ``http://localhost:8001``)
- ``default_model`` : Model name to use when ``-m`` is not specified (default ``None`` = smart router)
- ``output_format`` : One of ``text``, ``json``, ``markdown`` (default ``text``)
- ``color``         : Enable ANSI color output (default ``True``; respects ``NO_COLOR`` env)
- ``timeout``       : HTTP request timeout in seconds (default ``120``)
- ``animations``    : Draw the wait animation of ``oo chat`` on stderr (default ``True``;
                      off anyway when stderr is not a terminal, with ``NO_COLOR``,
                      ``--no-color`` or ``TERM=dumb``)
- ``animation_interval_ms`` : Time between two frames, 50 to 1000 (default ``150``)
- ``animation_delay_ms``    : Wait before the first frame, 100 to 5000 (default ``400``)
- ``animation_stop_ms``     : Longest the chat waits to erase a frame, 10 to 1000 (default ``100``)

An animation value that cannot be read falls back to its own default and
leaves every other setting as it is; an unreadable ``animations`` is off.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

# Default config directory following XDG
_XDG_CONFIG = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
CONFIG_DIR = Path(_XDG_CONFIG) / "opti-oignon"
CONFIG_FILE = CONFIG_DIR / "cli.yaml"

VALID_OUTPUT_FORMATS = ("text", "json", "markdown")
DEFAULT_API_URL = "http://localhost:8001"
DEFAULT_TIMEOUT = 120

# The wait animation's timings: key -> (lowest, highest, default), in milliseconds.
ANIMATION_RANGES = {
    "animation_interval_ms": (50, 1000, 150),
    "animation_delay_ms": (100, 5000, 400),
    "animation_stop_ms": (10, 1000, 100),
}
_SWITCH_ON = ("true", "1", "yes", "on")
_SWITCH_OFF = ("false", "0", "no", "off")


def parse_switch(value) -> bool | None:
    """A YAML boolean or one of the usual spellings; None when it is neither."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _SWITCH_ON:
        return True
    if text in _SWITCH_OFF:
        return False
    return None


def parse_animation_ms(key: str, value) -> int | None:
    """An integer within the key's range; None for anything else, a boolean included."""
    low, high, _ = ANIMATION_RANGES[key]
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if not isinstance(value, int) or not low <= value <= high:
        return None
    return value


@dataclass
class CLIConfig:
    """Runtime configuration for the ``oo`` CLI."""

    api_url: str = DEFAULT_API_URL
    default_model: str | None = None
    output_format: str = "text"
    color: bool = True
    timeout: int = DEFAULT_TIMEOUT
    animations: bool = True
    animation_interval_ms: int = ANIMATION_RANGES["animation_interval_ms"][2]
    animation_delay_ms: int = ANIMATION_RANGES["animation_delay_ms"][2]
    animation_stop_ms: int = ANIMATION_RANGES["animation_stop_ms"][2]

    def __post_init__(self) -> None:
        # Normalise trailing slash
        self.api_url = self.api_url.rstrip("/")
        # Validate output format
        if self.output_format not in VALID_OUTPUT_FORMATS:
            self.output_format = "text"
        # Respect NO_COLOR environment variable
        if os.environ.get("NO_COLOR") is not None:
            self.color = False

    # -- URLs ---------------------------------------------------------------

    @property
    def ws_base(self) -> str:
        """Return the WebSocket base URL derived from ``api_url``."""
        base = self.api_url
        if base.startswith("https://"):
            return "wss://" + base[len("https://"):]
        if base.startswith("http://"):
            return "ws://" + base[len("http://"):]
        return "ws://" + base

    # -- Persistence --------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a plain dict for YAML output."""
        data: dict = {"api_url": self.api_url, "output_format": self.output_format,
                      "color": self.color, "timeout": self.timeout,
                      "animations": self.animations,
                      "animation_interval_ms": self.animation_interval_ms,
                      "animation_delay_ms": self.animation_delay_ms,
                      "animation_stop_ms": self.animation_stop_ms}
        if self.default_model is not None:
            data["default_model"] = self.default_model
        return data

    def save(self, path: Path | None = None) -> Path:
        """Write current configuration to *path* (default CONFIG_FILE)."""
        dest = path or CONFIG_FILE
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, default_flow_style=False, sort_keys=False)
        return dest


def load_config(path: Path | None = None) -> CLIConfig:
    """Load CLI configuration, falling back to defaults if the file is absent."""
    src = path or CONFIG_FILE
    if not src.exists():
        return CLIConfig()
    try:
        with open(src, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            return CLIConfig()
        return CLIConfig(
            api_url=str(raw.get("api_url", DEFAULT_API_URL)),
            default_model=raw.get("default_model"),
            output_format=str(raw.get("output_format", "text")),
            color=bool(raw.get("color", True)),
            timeout=int(raw.get("timeout", DEFAULT_TIMEOUT)),
            **_animation_settings(raw),
        )
    except Exception:
        return CLIConfig()


def _animation_settings(raw: dict) -> dict:
    """The animation keys of a loaded file, each read alone; never raises."""
    out = {}
    if "animations" in raw:
        out["animations"] = parse_switch(raw["animations"]) is True
    for key, (_, _, default) in ANIMATION_RANGES.items():
        if key in raw:
            value = parse_animation_ms(key, raw[key])
            out[key] = default if value is None else value
    return out
