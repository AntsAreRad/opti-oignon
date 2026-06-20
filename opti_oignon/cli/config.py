#!/usr/bin/env python3
"""
CLI configuration loader -- Opti-Oignon S122.

Reads and writes CLI settings from ``~/.config/opti-oignon/cli.yaml``.

Supported keys
--------------
- ``api_url``       : Base URL of the running backend (default ``http://localhost:8001``)
- ``default_model`` : Model name to use when ``-m`` is not specified (default ``None`` = smart router)
- ``output_format`` : One of ``text``, ``json``, ``markdown`` (default ``text``)
- ``color``         : Enable ANSI color output (default ``True``; respects ``NO_COLOR`` env)
- ``timeout``       : HTTP request timeout in seconds (default ``120``)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

# Default config directory following XDG
_XDG_CONFIG = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
CONFIG_DIR = Path(_XDG_CONFIG) / "opti-oignon"
CONFIG_FILE = CONFIG_DIR / "cli.yaml"

VALID_OUTPUT_FORMATS = ("text", "json", "markdown")
DEFAULT_API_URL = "http://localhost:8001"
DEFAULT_TIMEOUT = 120


@dataclass
class CLIConfig:
    """Runtime configuration for the ``oo`` CLI."""

    api_url: str = DEFAULT_API_URL
    default_model: Optional[str] = None
    output_format: str = "text"
    color: bool = True
    timeout: int = DEFAULT_TIMEOUT

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
                      "color": self.color, "timeout": self.timeout}
        if self.default_model is not None:
            data["default_model"] = self.default_model
        return data

    def save(self, path: Optional[Path] = None) -> Path:
        """Write current configuration to *path* (default CONFIG_FILE)."""
        dest = path or CONFIG_FILE
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, default_flow_style=False, sort_keys=False)
        return dest


def load_config(path: Optional[Path] = None) -> CLIConfig:
    """Load CLI configuration, falling back to defaults if the file is absent."""
    src = path or CONFIG_FILE
    if not src.exists():
        return CLIConfig()
    try:
        with open(src, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            return CLIConfig()
        return CLIConfig(
            api_url=str(raw.get("api_url", DEFAULT_API_URL)),
            default_model=raw.get("default_model"),
            output_format=str(raw.get("output_format", "text")),
            color=bool(raw.get("color", True)),
            timeout=int(raw.get("timeout", DEFAULT_TIMEOUT)),
        )
    except Exception:
        return CLIConfig()
