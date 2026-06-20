#!/usr/bin/env python3
"""
Web Search Hardware Kill Switch for Opti-Oignon (S126).

When the kill switch is engaged:
  1. The ``WebSearcher`` singleton is destroyed
  2. ``ddgs`` / ``duckduckgo_search`` are purged from ``sys.modules``
  3. The search code path is removed from memory entirely -- not just flagged

Re-enabling requires the same multi-factor ceremony as mode degradation
(visual code + password + 2FA + cooldown).  In **Bulbe mode**, search
CANNOT be re-enabled at all (hardcoded restriction).

Circuit breaker: if >= 3 injection attempts are detected within 10 minutes,
search is automatically disabled.  Re-enabling requires the full ceremony.

Domain allowlist: when search is active, only results from approved domains
are passed through.  The allowlist is server-enforced (not LLM-side).

Security derives from module unloading and the ceremony gate, not from
obscurity.  An attacker who reads this code cannot re-enable search
without the encryption key and human physical presence (Kerckhoffs).
"""

from __future__ import annotations

import logging
import secrets
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Circuit breaker defaults
DEFAULT_INJECTION_THRESHOLD = 3
DEFAULT_INJECTION_WINDOW = 600  # 10 minutes

# Re-enable ceremony cooldown
REENABLE_COOLDOWN_SECONDS = 300  # 5 minutes

# Modules to purge when search is killed
_SEARCH_MODULES = frozenset({
    "ddgs",
    "duckduckgo_search",
    "duckduckgo_search.exceptions",
    "duckduckgo_search.ddgs",
    "duckduckgo_search.duckduckgo_search",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class KillSwitchState:
    """Current state of the search kill switch."""
    search_enabled: bool = True
    killed_at: float = 0.0
    killed_by: str = ""
    kill_reason: str = ""
    circuit_breaker_tripped: bool = False
    injection_count: int = 0
    reenable_pending: bool = False
    reenable_request_id: str = ""
    reenable_requested_at: float = 0.0
    reenable_visual_code: str = ""


@dataclass
class DomainAllowlist:
    """Server-enforced domain allowlist for search results."""
    enabled: bool = False
    domains: list[str] = field(default_factory=list)

    def is_allowed(self, url: str) -> bool:
        """Check if a URL's domain is in the allowlist."""
        if not self.enabled or not self.domains:
            return True  # No allowlist = all allowed
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            hostname = parsed.hostname or ""
            for domain in self.domains:
                if hostname == domain or hostname.endswith(f".{domain}"):
                    return True
            return False
        except Exception:
            return False


# ---------------------------------------------------------------------------
# SearchKillSwitch
# ---------------------------------------------------------------------------

class SearchKillSwitch:
    """Manages the web search kill switch.

    The kill switch physically removes the search module from memory
    rather than using a boolean flag.  This prevents any code path
    from accidentally or maliciously invoking search.
    """

    def __init__(self) -> None:
        self._state = KillSwitchState()
        self._injection_timestamps: list[float] = []
        self._domain_allowlist = DomainAllowlist()
        self._config = self._load_config()
        self._apply_config()

    def _load_config(self) -> dict[str, Any]:
        """Load kill switch config from security.yaml."""
        try:
            import yaml
            config_path = (
                Path(__file__).resolve().parent / "config" / "security.yaml"
            )
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as fh:
                    cfg = yaml.safe_load(fh) or {}
                return cfg.get("search_killswitch", {})
        except Exception:
            pass
        return {}

    def _apply_config(self) -> None:
        """Apply configuration values."""
        allowlist_cfg = self._config.get("domain_allowlist", {})
        if isinstance(allowlist_cfg, dict):
            self._domain_allowlist.enabled = allowlist_cfg.get("enabled", False)
            self._domain_allowlist.domains = allowlist_cfg.get("domains", [])

    # -- Kill / restore ------------------------------------------------------

    @property
    def is_enabled(self) -> bool:
        """Whether web search is currently enabled."""
        return self._state.search_enabled

    @property
    def is_killed(self) -> bool:
        """Whether the kill switch has been engaged."""
        return not self._state.search_enabled

    def kill(
        self,
        user_id: str = "system",
        reason: str = "manual",
    ) -> dict[str, Any]:
        """Engage the kill switch.  Destroys the search module.

        Adding security (killing search) requires no ceremony.
        """
        if self.is_killed:
            return {
                "success": True,
                "already_killed": True,
                "message": "Search already disabled",
            }

        # 1. Destroy the WebSearcher singleton
        try:
            from opti_oignon import web_search
            if hasattr(web_search, "web_searcher"):
                searcher = web_search.web_searcher
                # Call cleanup if available
                if hasattr(searcher, "close"):
                    try:
                        searcher.close()
                    except Exception:
                        pass
                web_search.web_searcher = None  # type: ignore[assignment]
                web_search.web_search_engine = None  # type: ignore[assignment]
            # Remove the convenience functions
            web_search.DDGS_AVAILABLE = False
        except ImportError:
            pass

        # 2. Purge search modules from sys.modules
        purged = []
        for mod_name in list(sys.modules.keys()):
            top = mod_name.split(".")[0]
            if top in ("ddgs", "duckduckgo_search"):
                del sys.modules[mod_name]
                purged.append(mod_name)

        # 3. Update state
        self._state.search_enabled = False
        self._state.killed_at = time.time()
        self._state.killed_by = user_id
        self._state.kill_reason = reason

        # Audit
        try:
            from opti_oignon.security_mode import _audit_log
            _audit_log(
                "search_killswitch_engaged",
                severity="WARNING",
                user_id=user_id,
                reason=reason,
                modules_purged=purged,
            )
        except Exception:
            pass

        logger.warning(
            "Search kill switch ENGAGED by %s (reason: %s). "
            "Purged %d modules.",
            user_id, reason, len(purged),
        )

        return {
            "success": True,
            "message": "Search disabled and modules purged from memory",
            "modules_purged": purged,
        }

    def request_reenable(self, user_id: str) -> dict[str, Any]:
        """Start the re-enable ceremony.

        In Bulbe mode, this always fails (search cannot be re-enabled).
        """
        # Check Bulbe mode
        try:
            from opti_oignon.security_mode import is_bulbe
            if is_bulbe():
                return {
                    "success": False,
                    "error": "bulbe_mode",
                    "message": (
                        "Web search cannot be re-enabled in Bulbe mode. "
                        "Switch to Daily mode first."
                    ),
                }
        except ImportError:
            pass

        if self.is_enabled:
            return {
                "success": True,
                "pending": False,
                "message": "Search is already enabled",
            }

        # Generate visual code for ceremony
        code = "".join(
            [str(secrets.randbelow(10)) for _ in range(6)]
        )
        request_id = secrets.token_urlsafe(16)
        now = time.time()

        self._state.reenable_pending = True
        self._state.reenable_request_id = request_id
        self._state.reenable_requested_at = now
        self._state.reenable_visual_code = code

        return {
            "success": True,
            "pending": True,
            "request_id": request_id,
            "cooldown_seconds": REENABLE_COOLDOWN_SECONDS,
            "expires_at": now + REENABLE_COOLDOWN_SECONDS + 60,
            # visual_code NOT in this response (DOM only)
        }

    def get_reenable_visual_code(self) -> str | None:
        """Return the visual code for DOM injection (not API)."""
        if self._state.reenable_pending:
            return self._state.reenable_visual_code
        return None

    def confirm_reenable(
        self,
        request_id: str,
        visual_code: str,
        user_id: str,
    ) -> dict[str, Any]:
        """Confirm the re-enable ceremony.

        Password and 2FA are verified by the caller (API route).
        """
        import hmac as _hmac

        if not self._state.reenable_pending:
            return {
                "success": False,
                "error": "no_pending_request",
                "message": "No pending re-enable request",
            }

        # Check Bulbe mode again (could have changed)
        try:
            from opti_oignon.security_mode import is_bulbe
            if is_bulbe():
                self._state.reenable_pending = False
                return {
                    "success": False,
                    "error": "bulbe_mode",
                    "message": "Cannot re-enable search in Bulbe mode",
                }
        except ImportError:
            pass

        # Verify request_id
        if not _hmac.compare_digest(request_id, self._state.reenable_request_id):
            return {
                "success": False,
                "error": "invalid_request",
                "message": "Invalid request ID",
            }

        # Verify cooldown
        elapsed = time.time() - self._state.reenable_requested_at
        if elapsed < REENABLE_COOLDOWN_SECONDS:
            remaining = REENABLE_COOLDOWN_SECONDS - elapsed
            return {
                "success": False,
                "error": "cooldown_active",
                "message": f"Cooldown active. {remaining:.0f}s remaining.",
            }

        # Verify visual code
        if not _hmac.compare_digest(visual_code, self._state.reenable_visual_code):
            return {
                "success": False,
                "error": "invalid_code",
                "message": "Invalid confirmation code",
            }

        # Re-enable search
        self._state.search_enabled = True
        self._state.reenable_pending = False
        self._state.circuit_breaker_tripped = False
        self._injection_timestamps.clear()

        try:
            from opti_oignon.security_mode import _audit_log
            _audit_log(
                "search_killswitch_disengaged",
                severity="CRITICAL",
                user_id=user_id,
            )
        except Exception:
            pass

        logger.warning("Search kill switch DISENGAGED by %s", user_id)

        return {
            "success": True,
            "message": "Search re-enabled. Module will reload on next use.",
        }

    def cancel_reenable(self) -> dict[str, Any]:
        """Cancel a pending re-enable request."""
        self._state.reenable_pending = False
        self._state.reenable_request_id = ""
        self._state.reenable_visual_code = ""
        return {"success": True, "message": "Re-enable request cancelled"}

    # -- Circuit breaker -----------------------------------------------------

    def record_injection(self, details: str = "") -> dict[str, Any]:
        """Record a detected search injection attempt.

        If the threshold is exceeded, auto-kill search.
        """
        now = time.time()
        threshold = self._config.get(
            "circuit_breaker_threshold", DEFAULT_INJECTION_THRESHOLD
        )
        window = self._config.get(
            "circuit_breaker_window", DEFAULT_INJECTION_WINDOW
        )

        self._injection_timestamps.append(now)
        # Clean old entries
        self._injection_timestamps = [
            t for t in self._injection_timestamps
            if now - t < window
        ]
        self._state.injection_count = len(self._injection_timestamps)

        try:
            from opti_oignon.security_mode import _audit_log
            _audit_log(
                "search_injection_detected",
                severity="WARNING",
                details=details,
                count_in_window=len(self._injection_timestamps),
                threshold=threshold,
            )
        except Exception:
            pass

        if len(self._injection_timestamps) >= threshold:
            self._state.circuit_breaker_tripped = True
            result = self.kill(user_id="circuit_breaker", reason="injection_threshold")
            result["circuit_breaker_tripped"] = True
            logger.critical(
                "Circuit breaker TRIPPED: %d injections in %ds. "
                "Search auto-disabled.",
                len(self._injection_timestamps), window,
            )
            return result

        return {
            "tripped": False,
            "count": len(self._injection_timestamps),
            "threshold": threshold,
        }

    # -- Domain allowlist ----------------------------------------------------

    @property
    def domain_allowlist(self) -> DomainAllowlist:
        return self._domain_allowlist

    def set_domain_allowlist(
        self, enabled: bool, domains: list[str] | None = None
    ) -> None:
        """Update the domain allowlist."""
        self._domain_allowlist.enabled = enabled
        if domains is not None:
            self._domain_allowlist.domains = domains

    def filter_results(self, results: list[Any]) -> list[Any]:
        """Filter search results through the domain allowlist.

        Each result must have a .url or ['url'] attribute.
        """
        if not self._domain_allowlist.enabled:
            return results

        filtered = []
        for r in results:
            url = getattr(r, "url", None)
            if url is None and isinstance(r, dict):
                url = r.get("url", "")
            if url and self._domain_allowlist.is_allowed(url):
                filtered.append(r)
        return filtered

    # -- Status --------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Return full kill switch status for the API."""
        return {
            "search_enabled": self._state.search_enabled,
            "killed_at": self._state.killed_at if self.is_killed else None,
            "killed_by": self._state.killed_by if self.is_killed else None,
            "kill_reason": self._state.kill_reason if self.is_killed else None,
            "circuit_breaker_tripped": self._state.circuit_breaker_tripped,
            "injection_count": self._state.injection_count,
            "reenable_pending": self._state.reenable_pending,
            "domain_allowlist": {
                "enabled": self._domain_allowlist.enabled,
                "domain_count": len(self._domain_allowlist.domains),
                "domains": self._domain_allowlist.domains,
            },
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

search_killswitch = SearchKillSwitch()
