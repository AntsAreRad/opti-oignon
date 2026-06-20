#!/usr/bin/env python3
"""
opti_oignon.middleware.csp -- Content Security Policy middleware (S155).

Provides a strict, nonce-based CSP that:
- Generates a unique nonce per request for inline script authorization
- Restricts connect-src to localhost origins only
- Blocks eval(), inline scripts without nonce, and framing
- Supports report-only mode (default) and enforcement mode
- Includes a violation reporting endpoint

The nonce is stored on request.state.csp_nonce so templates and
frontend responses can reference it.

Configuration is loaded from config/security.yaml under the 'csp' key.
If absent, secure defaults are used.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Optional

logger = logging.getLogger(__name__)

checkpoint_before_apply = True

# ---------------------------------------------------------------------------
# CSP configuration
# ---------------------------------------------------------------------------

_DEFAULT_CSP_CONFIG = {
    "enabled": True,
    "report_only": True,
    "nonce_length": 24,
    "report_uri": "/api/csp-report",
    "max_stored_reports": 500,
    "directives": {
        "default-src": "'self'",
        "script-src": "'self'",
        "style-src": "'self' 'unsafe-inline'",
        "img-src": "'self' data:",
        "font-src": "'self'",
        "connect-src": "'self' http://localhost:* http://127.0.0.1:* ws://localhost:* ws://127.0.0.1:*",
        "frame-ancestors": "'none'",
        "base-uri": "'self'",
        "form-action": "'self'",
        "object-src": "'none'",
        "upgrade-insecure-requests": "",
    },
}


@dataclass
class CSPConfig:
    """Validated CSP configuration."""

    enabled: bool = True
    report_only: bool = True
    nonce_length: int = 24
    report_uri: str = "/api/csp-report"
    max_stored_reports: int = 500
    directives: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.nonce_length < 16:
            self.nonce_length = 16
        if self.nonce_length > 64:
            self.nonce_length = 64
        if self.max_stored_reports < 0:
            self.max_stored_reports = 0
        if self.max_stored_reports > 10000:
            self.max_stored_reports = 10000
        if not self.directives:
            self.directives = dict(_DEFAULT_CSP_CONFIG["directives"])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CSPConfig":
        """Create config from a dictionary (e.g. parsed YAML)."""
        directives = data.get("directives", {})
        if not isinstance(directives, dict):
            directives = {}
        merged = dict(_DEFAULT_CSP_CONFIG["directives"])
        merged.update(directives)
        return cls(
            enabled=bool(data.get("enabled", True)),
            report_only=bool(data.get("report_only", True)),
            nonce_length=int(data.get("nonce_length", 24)),
            report_uri=str(data.get("report_uri", "/api/csp-report")),
            max_stored_reports=int(data.get("max_stored_reports", 500)),
            directives=merged,
        )

    @classmethod
    def default(cls) -> "CSPConfig":
        """Return default configuration."""
        return cls.from_dict(_DEFAULT_CSP_CONFIG)


def load_csp_config() -> CSPConfig:
    """Load CSP configuration from config/security.yaml.

    Falls back to defaults if the file is missing or the 'csp' section
    is absent.
    """
    try:
        import yaml
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "security.yaml"
        )
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
            csp_section = raw.get("csp", {})
            if isinstance(csp_section, dict) and csp_section:
                return CSPConfig.from_dict(csp_section)
    except Exception as exc:
        logger.debug("Failed to load CSP config: %s", exc)
    return CSPConfig.default()


# ---------------------------------------------------------------------------
# Nonce generation
# ---------------------------------------------------------------------------

def generate_nonce(length: int = 24) -> str:
    """Generate a cryptographically secure random nonce.

    Uses secrets.token_urlsafe which draws from os.urandom.
    The result is base64url-encoded, so the actual byte entropy
    is (length * 3/4).
    """
    return secrets.token_urlsafe(length)


# ---------------------------------------------------------------------------
# CSP header builder
# ---------------------------------------------------------------------------

def build_csp_header(directives: dict[str, str], nonce: str,
                     report_uri: Optional[str] = None) -> str:
    """Build a CSP header string from directives, injecting the nonce.

    The nonce is added to script-src. If script-src is not present,
    it is created from default-src.
    """
    parts = []
    for directive, value in directives.items():
        # Inject nonce into script-src
        if directive == "script-src":
            nonce_token = f"'nonce-{nonce}'"
            if nonce_token not in value:
                value = f"{value} {nonce_token}"
            parts.append(f"{directive} {value}")
        elif directive == "upgrade-insecure-requests" and value == "":
            # Valueless directive
            parts.append(directive)
        else:
            parts.append(f"{directive} {value}")

    if report_uri:
        parts.append(f"report-uri {report_uri}")

    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Violation report storage
# ---------------------------------------------------------------------------

@dataclass
class CSPViolationReport:
    """A single CSP violation report."""

    timestamp: float
    document_uri: str = ""
    referrer: str = ""
    violated_directive: str = ""
    effective_directive: str = ""
    original_policy: str = ""
    blocked_uri: str = ""
    source_file: str = ""
    line_number: int = 0
    column_number: int = 0
    status_code: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "document_uri": self.document_uri,
            "referrer": self.referrer,
            "violated_directive": self.violated_directive,
            "effective_directive": self.effective_directive,
            "blocked_uri": self.blocked_uri,
            "source_file": self.source_file,
            "line_number": self.line_number,
            "column_number": self.column_number,
            "status_code": self.status_code,
        }


class CSPReportStore:
    """Thread-safe in-memory store for CSP violation reports."""

    def __init__(self, max_reports: int = 500) -> None:
        self._max = max_reports
        self._reports: deque[CSPViolationReport] = deque(maxlen=max_reports)
        self._lock = Lock()
        self._total_received: int = 0

    def add(self, report: CSPViolationReport) -> None:
        """Add a violation report."""
        with self._lock:
            self._reports.append(report)
            self._total_received += 1

    def get_all(self) -> list[dict[str, Any]]:
        """Return all stored reports as dicts."""
        with self._lock:
            return [r.to_dict() for r in self._reports]

    def get_recent(self, count: int = 50) -> list[dict[str, Any]]:
        """Return the most recent N reports."""
        with self._lock:
            items = list(self._reports)[-count:]
            return [r.to_dict() for r in items]

    @property
    def total_received(self) -> int:
        """Total number of reports received (including evicted)."""
        with self._lock:
            return self._total_received

    @property
    def stored_count(self) -> int:
        """Number of currently stored reports."""
        with self._lock:
            return len(self._reports)

    def clear(self) -> int:
        """Clear all stored reports. Returns count cleared."""
        with self._lock:
            count = len(self._reports)
            self._reports.clear()
            return count


# Module-level singleton
_report_store: Optional[CSPReportStore] = None
_store_lock = Lock()


def get_report_store(max_reports: int = 500) -> CSPReportStore:
    """Get or create the singleton report store."""
    global _report_store
    with _store_lock:
        if _report_store is None:
            _report_store = CSPReportStore(max_reports=max_reports)
        return _report_store


# ---------------------------------------------------------------------------
# Starlette middleware
# ---------------------------------------------------------------------------

try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response

    STARLETTE_AVAILABLE = True
except ImportError:
    STARLETTE_AVAILABLE = False
    BaseHTTPMiddleware = object  # type: ignore[misc,assignment]
    Request = object  # type: ignore[misc,assignment]
    Response = object  # type: ignore[misc,assignment]


class CSPMiddleware(BaseHTTPMiddleware):
    """Content Security Policy middleware with per-request nonce.

    Generates a cryptographic nonce for each request and builds a
    strict CSP header. The nonce is stored at request.state.csp_nonce.

    In report-only mode (default), violations are logged but not blocked.
    """

    def __init__(self, app: Any, config: Optional[CSPConfig] = None) -> None:
        if not STARLETTE_AVAILABLE:
            raise RuntimeError("starlette is required for CSPMiddleware")
        super().__init__(app)
        self._config = config or load_csp_config()
        self._store = get_report_store(self._config.max_stored_reports)
        mode = "report-only" if self._config.report_only else "enforce"
        logger.info("CSP middleware initialized (mode=%s)", mode)

    @property
    def config(self) -> CSPConfig:
        """Current CSP configuration."""
        return self._config

    @property
    def report_store(self) -> CSPReportStore:
        """Access the violation report store."""
        return self._store

    async def dispatch(self, request: Any, call_next: Any) -> Any:
        """Process request: generate nonce, add CSP header to response."""
        if not self._config.enabled:
            return await call_next(request)

        # Generate per-request nonce
        nonce = generate_nonce(self._config.nonce_length)

        # Store nonce on request state for template access
        request.state.csp_nonce = nonce

        response = await call_next(request)

        # Build CSP header
        csp_value = build_csp_header(
            self._config.directives,
            nonce,
            report_uri=self._config.report_uri,
        )

        # Apply as report-only or enforced
        if self._config.report_only:
            response.headers["Content-Security-Policy-Report-Only"] = csp_value
        else:
            response.headers["Content-Security-Policy"] = csp_value

        # Always set nonce header for frontend consumption
        response.headers["X-CSP-Nonce"] = nonce

        return response


# ---------------------------------------------------------------------------
# FastAPI route for violation reports
# ---------------------------------------------------------------------------

def parse_csp_report(body: bytes) -> Optional[CSPViolationReport]:
    """Parse a CSP violation report from the request body.

    Browsers send reports as JSON with a 'csp-report' key.
    """
    try:
        data = json.loads(body)
        report_data = data.get("csp-report", data)
        return CSPViolationReport(
            timestamp=time.time(),
            document_uri=str(report_data.get("document-uri", "")),
            referrer=str(report_data.get("referrer", "")),
            violated_directive=str(report_data.get("violated-directive", "")),
            effective_directive=str(report_data.get("effective-directive", "")),
            original_policy=str(report_data.get("original-policy", "")),
            blocked_uri=str(report_data.get("blocked-uri", "")),
            source_file=str(report_data.get("source-file", "")),
            line_number=int(report_data.get("line-number", 0)),
            column_number=int(report_data.get("column-number", 0)),
            status_code=int(report_data.get("status-code", 0)),
        )
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("Failed to parse CSP violation report: %s", exc)
        return None


try:
    from fastapi import APIRouter, Request as FARequest
    from fastapi.responses import JSONResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

if FASTAPI_AVAILABLE:
    csp_router = APIRouter(tags=["security"])

    @csp_router.post("/api/csp-report")
    async def receive_csp_report(request: FARequest) -> JSONResponse:
        """Receive and store a CSP violation report from the browser."""
        body = await request.body()
        report = parse_csp_report(body)
        if report is None:
            return JSONResponse(
                {"error": "Invalid report format"},
                status_code=400,
            )
        store = get_report_store()
        store.add(report)
        logger.info(
            "CSP violation: directive=%s blocked=%s source=%s:%d",
            report.violated_directive,
            report.blocked_uri,
            report.source_file,
            report.line_number,
        )
        return JSONResponse({"status": "received"}, status_code=204)

    @csp_router.get("/api/csp-reports")
    async def list_csp_reports(count: int = 50) -> JSONResponse:
        """List recent CSP violation reports (admin endpoint)."""
        store = get_report_store()
        reports = store.get_recent(min(count, 200))
        return JSONResponse({
            "total_received": store.total_received,
            "stored": store.stored_count,
            "reports": reports,
        })

    @csp_router.delete("/api/csp-reports")
    async def clear_csp_reports() -> JSONResponse:
        """Clear all stored CSP violation reports."""
        store = get_report_store()
        cleared = store.clear()
        return JSONResponse({"cleared": cleared})
else:
    csp_router = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

CSP_AVAILABLE = STARLETTE_AVAILABLE
