#!/usr/bin/env python3
"""Token counting with an honest split between exact and estimated.

Every token figure this project has ever produced was a character-ratio
estimate: chars-per-token calibrated by model family, refined by code
detection. Useful, cheap -- and silently approximate. This module keeps
that estimator as the floor and adds the one thing it could never offer:
a count from a real tokenizer, clearly labelled as such.

The exact path speaks to a llama.cpp server's tokenize endpoint through a
caller-injectable transport. The transport is the seam: production wires a
loopback HTTP round trip built from the backends configuration, tests seed
a scripted callable, and nothing in between is simulated. The heuristic
path delegates to the existing family-calibrated estimator and degrades to
a plain character ratio when even that module is absent.

One promise must never bend: a count is labelled ``exact`` only when the
tokenizer actually answered. Disabled, unreachable, timed out, malformed
-- every failure of the exact path falls back to an estimate that says so.
A rough number under an honest label is recoverable; a right-looking
number under a false label poisons every measurement built on top of it.

Counting is off the exact path by default (``enabled: false``), in which
case no network round trip of any kind is attempted and every result is
the estimate the rest of the codebase already relies on. Mode-free by
construction: the only traffic the default transport can ever emit is a
loopback request to the same local inference server the backends already
use, identical in Daily and in Bulbe.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "config" / "token_counting.yaml"
_BACKENDS_CONFIG_PATH = Path(__file__).parent / "config" / "backends.yaml"
_DEFAULT_ENDPOINT = "http://127.0.0.1:8080"
_DEFAULT_TIMEOUT_S = 0.5

METHOD_EXACT = "exact"
METHOD_ESTIMATED = "estimated"

SOURCE_TOKENIZE = "llama_server_tokenize"
SOURCE_FAMILY = "family_calibration"
SOURCE_CHAR_RATIO = "char_ratio"
SOURCE_EMPTY = "empty_input"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _load_config(path: Path | None = None) -> dict[str, Any]:
    """Load the counting config, tolerating an absent or unreadable file.

    Args:
        path: Optional override path. Defaults to config/token_counting.yaml.

    Returns:
        Dict with the known keys filled from the file when present, from
        the safe defaults otherwise.
    """
    defaults: dict[str, Any] = {
        "enabled": False,
        "endpoint": "",
        "timeout_s": _DEFAULT_TIMEOUT_S,
    }
    target = path or _CONFIG_PATH
    try:
        import yaml

        if target.exists():
            with open(target, encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
            if isinstance(loaded, dict):
                for key in defaults:
                    if key in loaded:
                        defaults[key] = loaded[key]
    except Exception as exc:
        logger.debug("Token counting config unreadable, using defaults: %s", exc)
    return defaults


def _resolve_endpoint(cfg: dict[str, Any]) -> str:
    """Resolve the tokenize base URL from the narrowest source that answers.

    Order: the counting config's own ``endpoint``, then the
    ``llama_server.host`` entry of the backends configuration (the single
    place the external server address already lives), then the
    conventional local default.
    """
    own = str(cfg.get("endpoint") or "").strip()
    if own:
        return own.rstrip("/")
    try:
        import yaml

        if _BACKENDS_CONFIG_PATH.exists():
            with open(_BACKENDS_CONFIG_PATH, encoding="utf-8") as fh:
                backends = yaml.safe_load(fh) or {}
        else:
            backends = {}
        server_cfg = backends.get("llama_server") if isinstance(backends, dict) else None
        if isinstance(server_cfg, dict):
            host = str(server_cfg.get("host") or "").strip()
            if host:
                return host.rstrip("/")
    except Exception as exc:
        logger.debug("Backends config unreadable for endpoint resolution: %s", exc)
    return _DEFAULT_ENDPOINT


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

def _build_http_transport(
    endpoint: str,
    timeout_s: float,
) -> Callable[[str, dict[str, Any] | None], Any]:
    """Build the default loopback transport: one guarded round trip per call.

    Mirrors the llama-server backend's request discipline: stdlib only,
    a tight timeout, and a RuntimeError on any failure -- unreachable,
    non-JSON, or otherwise -- so the caller can fall back honestly.
    """
    base = endpoint.rstrip("/")

    def _transport(path: str, payload: dict[str, Any] | None = None) -> Any:
        url = f"{base}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read()
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise RuntimeError(
                f"tokenize endpoint unreachable at {base}: {exc}"
            ) from exc
        try:
            return json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise RuntimeError(
                f"tokenize endpoint returned a non-JSON body from {path}"
            ) from exc

    return _transport


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TokenCount:
    """One count with its provenance. ``method`` is the honesty bit.

    Attributes:
        tokens: The count.
        method: ``exact`` only when a real tokenizer answered; otherwise
            ``estimated``.
        source: Where the number came from (tokenize endpoint, family
            calibration, bare character ratio, or the empty-input
            short-circuit).
    """

    tokens: int
    method: str
    source: str


def _estimate(text: str, model: str) -> TokenCount:
    """The heuristic floor: family calibration, then a bare character ratio."""
    try:
        from .context_manager import estimate_tokens_calibrated

        return TokenCount(
            int(estimate_tokens_calibrated(text, model)),
            METHOD_ESTIMATED,
            SOURCE_FAMILY,
        )
    except Exception:
        return TokenCount(max(1, len(text) // 4), METHOD_ESTIMATED, SOURCE_CHAR_RATIO)


# ---------------------------------------------------------------------------
# Counter
# ---------------------------------------------------------------------------

class TokenCounter:
    """Counts tokens, saying plainly which counts are real.

    Args:
        config: Optional config dict (see ``_load_config`` for the keys).
            When omitted, the YAML file is read.
        transport: Optional transport callable ``(path, payload) -> parsed
            JSON`` used for the exact path. When omitted and the exact
            path is enabled, the default loopback HTTP transport is built
            from the resolved endpoint. An injected transport is still
            gated by ``enabled``: a disabled counter never calls it.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        transport: Callable[[str, dict[str, Any] | None], Any] | None = None,
    ):
        self._config = dict(config) if config is not None else _load_config()
        try:
            self._timeout_s = max(0.05, float(self._config.get("timeout_s", _DEFAULT_TIMEOUT_S)))
        except (TypeError, ValueError):
            self._timeout_s = _DEFAULT_TIMEOUT_S
        self._endpoint = _resolve_endpoint(self._config)
        self._enabled = bool(self._config.get("enabled", False))
        if transport is not None:
            self._transport: Callable[[str, dict[str, Any] | None], Any] | None = transport
        elif self._enabled:
            self._transport = _build_http_transport(self._endpoint, self._timeout_s)
        else:
            self._transport = None

    @property
    def exact_enabled(self) -> bool:
        """Whether the exact path may be attempted at all."""
        return self._enabled and self._transport is not None

    @property
    def endpoint(self) -> str:
        """The resolved tokenize base URL (informational)."""
        return self._endpoint

    # -- counting ----------------------------------------------------------

    def count(self, text: str, model: str = "") -> TokenCount:
        """Count one text, exact when the tokenizer answers, estimated otherwise.

        Empty input short-circuits to a zero labelled ``estimated``: no
        tokenizer was consulted, so no exactness is claimed.
        """
        if not text:
            return TokenCount(0, METHOD_ESTIMATED, SOURCE_EMPTY)
        if self.exact_enabled:
            exact = self._count_exact(text)
            if exact is not None:
                return exact
        return _estimate(text, model)

    def count_messages(
        self,
        messages: list[dict[str, Any]] | None,
        model: str = "",
    ) -> TokenCount:
        """Aggregate count over message contents.

        Counts the content fields only -- the chat-template overhead a
        server adds around them is not included; the host-side calibration
        notes quantify that margin. The aggregate is labelled exact only
        when every counted part was exact, and the first exact failure
        stops asking the transport, so a dead server is paid for once,
        not once per message.
        """
        total = 0
        any_part = False
        all_exact = True
        exact_open = self.exact_enabled
        fallback_source = SOURCE_FAMILY
        for msg in messages or []:
            content = str((msg or {}).get("content", "") or "")
            if not content:
                continue
            any_part = True
            part: TokenCount | None = None
            if exact_open:
                part = self._count_exact(content)
                if part is None:
                    exact_open = False
            if part is None:
                part = _estimate(content, model)
                all_exact = False
                fallback_source = part.source
            total += part.tokens
        if not any_part:
            return TokenCount(0, METHOD_ESTIMATED, SOURCE_EMPTY)
        if all_exact:
            return TokenCount(total, METHOD_EXACT, SOURCE_TOKENIZE)
        return TokenCount(total, METHOD_ESTIMATED, fallback_source)

    # -- exact path --------------------------------------------------------

    def _count_exact(self, text: str) -> TokenCount | None:
        """One tokenize round trip; None on any failure, never an exception."""
        if self._transport is None:
            return None
        try:
            payload = self._transport("/tokenize", {"content": text})
        except Exception as exc:
            logger.debug("Exact token count unavailable, estimating: %s", exc)
            return None
        tokens = payload.get("tokens") if isinstance(payload, dict) else None
        if not isinstance(tokens, list):
            logger.debug("Tokenize endpoint answered an unexpected shape; estimating")
            return None
        return TokenCount(len(tokens), METHOD_EXACT, SOURCE_TOKENIZE)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_counter: TokenCounter | None = None


def get_token_counter() -> TokenCounter:
    """Return the shared counter, building it from configuration on first use."""
    global _counter
    if _counter is None:
        _counter = TokenCounter()
    return _counter


def reset_token_counter() -> None:
    """Drop the shared counter so the next resolution reloads configuration."""
    global _counter
    _counter = None
