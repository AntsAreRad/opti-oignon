#!/usr/bin/env python3
"""
Web search kill switch for Opti-Oignon.

The switch is a recorded state. Engaging it writes
``data/.search_killswitch.json`` (mode 0600, replaced atomically), and every
check reads that record again, so an engaged switch holds across a restart
and is seen by every process that shares ``data/``. It is read at every check
by the web searcher's request gate, the capability manifest, the chat
executor and the Bulbe middleware. A record that cannot be read -- malformed,
another version, a link, a directory, a FIFO, or larger than 64 KiB -- reads
engaged. A missing record reads as never engaged: anyone who can write
``data/`` can delete it and so re-enable search, the same exposure as the
security mode's lockfile, and a missing ``data/`` (an unmounted volume) reads
as never engaged too.

Engaging needs no ceremony. When the record cannot be written, the switch
stays engaged in this process, says the state will not survive a restart,
and the next kill retries the write. Only the re-enable ceremony may record
"not engaged"; every other writer reads the record again under the lock and
keeps an engaged record engaged. A process that could not record its kill
stays engaged until its own ceremony, a restart, or a later write of its
own that records the engaged state; until then it holds even after another
process records a re-enable.

Re-enabling needs the visual code and the cooldown, checked here, and the
administrator's password, checked by the API route, which refuses when it
cannot read the auth manager. The visual code is served over the API, so it
proves an API session, not physical presence. The 2FA code the route accepts
is not verified. In Bulbe mode re-enabling is refused.

The domain allowlist is recorded in the same file. The web searcher applies
it, and feeds the circuit breaker, on every real search, cached results
included. An enabled allowlist naming no domain passes nothing. Entries are
normalised to host names, and one that is not a host name is refused by
name; a URL a browser would read differently from the host it names never
passes an enabled allowlist. The circuit breaker engages the switch after
three searches whose results carried a detected injection within ten
minutes (both configurable).
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import stat
import tempfile
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    import fcntl
except ImportError:  # pragma: no cover - not a POSIX system: the thread lock alone
    fcntl = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Circuit breaker defaults
DEFAULT_INJECTION_THRESHOLD = 3
DEFAULT_INJECTION_WINDOW = 600  # 10 minutes

# Re-enable ceremony cooldown
REENABLE_COOLDOWN_SECONDS = 300  # 5 minutes

# The recorded state, beside the security mode's lockfile. Read at call time,
# never at import: loading this module touches nothing under data/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STATE_PATH = _PROJECT_ROOT / "data" / ".search_killswitch.json"
_STATE_VERSION = 1
_STATE_MAX_BYTES = 64 * 1024

# What a host name in the allowlist may be made of, once normalised.
_HOST_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789.-")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class KillSwitchState:
    """What this process holds besides the record: the ceremony and the breaker."""
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
        """Whether a result URL passes the allowlist.

        A disabled allowlist passes everything; an enabled one naming no
        domain passes nothing. A URL carrying a backslash, whitespace or a
        control character, a scheme other than http or https, or userinfo
        never passes: a browser would read it differently from the host
        name parsed here.
        """
        if not self.enabled:
            return True
        if not self.domains:
            return False
        try:
            from urllib.parse import urlparse

            text = str(url)
            if "\\" in text or any(
                ch.isspace() or unicodedata.category(ch) in ("Cc", "Cf") for ch in text
            ):
                return False
            parsed = urlparse(text)
            if parsed.scheme.lower() not in ("http", "https"):
                return False
            if "@" in parsed.netloc:
                return False
            hostname = parsed.hostname or ""
            for domain in self.domains:
                if hostname == domain or hostname.endswith(f".{domain}"):
                    return True
            return False
        except Exception:
            return False


def _normalise_domains(domains: Any) -> tuple[list[str], list[str]]:
    """Host names from allowlist entries, and the entries refused.

    An entry is lowercased and stripped; a leading scheme, anything from the
    first slash, a leading ``*.`` and a trailing dot are removed. An entry
    that is then empty is dropped; one that still carries a character a host
    name cannot hold is refused, as it was given. Duplicates are dropped and
    the order kept.
    """
    kept: list[str] = []
    refused: list[str] = []
    for raw in list(domains or []):
        entry = str(raw).strip().lower()
        for scheme in ("http://", "https://"):
            if entry.startswith(scheme):
                entry = entry[len(scheme):]
                break
        entry = entry.split("/", 1)[0]
        if entry.startswith("*."):
            entry = entry[2:]
        entry = entry.strip(".")
        if not entry:
            continue
        if not set(entry) <= _HOST_CHARS:
            refused.append(str(raw))
            continue
        if entry not in kept:
            kept.append(entry)
    return kept, refused


def _allowlist_in(record: dict) -> DomainAllowlist | None:
    """The allowlist a well-formed record carries, or None when it is malformed."""
    value = record.get("domain_allowlist")
    if not isinstance(value, dict):
        return None
    enabled = value.get("enabled")
    domains = value.get("domains")
    if not isinstance(enabled, bool) or not isinstance(domains, list):
        return None
    if not all(isinstance(d, str) for d in domains):
        return None
    return DomainAllowlist(enabled=enabled, domains=list(domains))


# ---------------------------------------------------------------------------
# SearchKillSwitch
# ---------------------------------------------------------------------------

class SearchKillSwitch:
    """Manages the web search kill switch as a recorded state."""

    def __init__(self, state_path: Path | None = None) -> None:
        self._state = KillSwitchState()
        self._injection_timestamps: list[float] = []
        self._state_path = state_path
        self._lock = threading.RLock()
        # Engaged here while the record could not say so.
        self._latched = False
        self._cache: tuple[tuple, tuple[bool, dict | None]] | None = None
        self._warned: set[tuple] = set()
        self._config = self._load_config()
        self._config_allowlist = DomainAllowlist()
        self._apply_config()

    def _load_config(self) -> dict[str, Any]:
        """Load kill switch config from security.yaml."""
        try:
            import yaml
            config_path = (
                Path(__file__).resolve().parent / "config" / "security.yaml"
            )
            if config_path.exists():
                with open(config_path, encoding="utf-8") as fh:
                    cfg = yaml.safe_load(fh) or {}
                return cfg.get("search_killswitch", {})
        except Exception:
            pass
        return {}

    def _apply_config(self) -> None:
        """The allowlist that applies while nothing is recorded."""
        allowlist_cfg = self._config.get("domain_allowlist", {})
        if isinstance(allowlist_cfg, dict):
            self._config_allowlist.enabled = bool(allowlist_cfg.get("enabled", False))
            self._config_allowlist.domains = _normalise_domains(
                allowlist_cfg.get("domains", [])
            )[0]

    # -- The record ----------------------------------------------------------

    def _path(self) -> Path:
        return Path(self._state_path or _STATE_PATH)

    def _unreadable(self, key: tuple, path: Path, reason: str) -> tuple[bool, None]:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(
                "Search kill switch: the record at %s cannot be read (%s); "
                "search stays disabled.", path, reason,
            )
        return True, None

    def _read_record(self) -> tuple[bool, dict | None]:
        """Read the record: (engaged, the record or None).

        A missing record was never engaged. Anything that is not a regular
        file of at most 64 KiB holding a version-1 record with a boolean
        ``engaged`` reads engaged. No link is followed.
        """
        path = self._path()
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            return False, None
        except OSError as exc:
            return self._unreadable(("lstat", exc.__class__.__name__), path, exc.__class__.__name__)
        key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
        if not stat.S_ISREG(st.st_mode):
            return self._unreadable(key, path, "not a regular file")
        if st.st_size > _STATE_MAX_BYTES:
            return self._unreadable(key, path, "larger than 64 KiB")
        cached = self._cache
        if cached is not None and cached[0] == key:
            return cached[1]
        result = self._parse(path, st, key)
        self._cache = (key, result)
        return result

    def _parse(self, path: Path, st: os.stat_result, key: tuple) -> tuple[bool, dict | None]:
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError as exc:
            return self._unreadable(key, path, exc.__class__.__name__)
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (st.st_dev, st.st_ino):
                return self._unreadable(key, path, "replaced while read")
            data = b""
            while len(data) <= _STATE_MAX_BYTES:
                chunk = os.read(fd, _STATE_MAX_BYTES + 1 - len(data))
                if not chunk:
                    break
                data += chunk
        except OSError as exc:
            return self._unreadable(key, path, exc.__class__.__name__)
        finally:
            os.close(fd)
        if len(data) > _STATE_MAX_BYTES:
            return self._unreadable(key, path, "larger than 64 KiB")
        try:
            record = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RecursionError):
            return self._unreadable(key, path, "not a JSON record")
        if not isinstance(record, dict):
            return self._unreadable(key, path, "not a JSON object")
        version = record.get("version")
        if type(version) is not int or version != _STATE_VERSION:
            return self._unreadable(key, path, "another version")
        if not isinstance(record.get("engaged"), bool):
            return self._unreadable(key, path, "engaged is not a boolean")
        return record["engaged"], record

    def _applied(self, engaged: bool, record: dict | None) -> DomainAllowlist:
        """The allowlist that applies, one rule.

        The record's when it is readable and holds a well-formed one; the
        configuration's when there is no record; enabled with no domain --
        nothing passes -- when the record is unreadable or its allowlist is
        malformed.
        """
        if record is not None:
            return _allowlist_in(record) or DomainAllowlist(enabled=True, domains=[])
        if engaged:
            return DomainAllowlist(enabled=True, domains=[])
        return DomainAllowlist(
            enabled=self._config_allowlist.enabled,
            domains=list(self._config_allowlist.domains),
        )

    def _commit(self, build: Callable[[bool, dict | None], dict]) -> bool:
        """Write the record ``build`` returns from the record as it stands.

        Called under the thread lock. The record is read again after the
        cross-process lock is held, written to a unique temporary file in the
        same directory and moved into place. Returns False when anything
        could not be written; nothing is changed then.
        """
        path = self._path()
        tmp: str | None = None
        lock_fd: int | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            lock_fd = os.open(
                str(path) + ".lock", os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600,
            )
            if fcntl is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            self._cache = None
            record = build(*self._read_record())
            fd, tmp = tempfile.mkstemp(
                dir=path.parent, prefix=".search_killswitch.", suffix=".tmp",
            )
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(record, fh, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
            tmp = None
            return True
        except OSError as exc:
            logger.warning(
                "Search kill switch: the state could not be recorded at %s (%s).",
                path, exc.__class__.__name__,
            )
            return False
        finally:
            if tmp is not None:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            if lock_fd is not None:
                os.close(lock_fd)
            self._cache = None

    @staticmethod
    def _make_record(
        *,
        engaged: bool,
        allowlist: DomainAllowlist,
        killed_at: float | None = None,
        killed_by: str | None = None,
        kill_reason: str | None = None,
        breaker: bool = False,
    ) -> dict[str, Any]:
        return {
            "version": _STATE_VERSION,
            "engaged": bool(engaged),
            "killed_at": killed_at,
            "killed_by": killed_by,
            "kill_reason": kill_reason,
            "circuit_breaker_tripped": bool(breaker),
            "domain_allowlist": {
                "enabled": bool(allowlist.enabled),
                "domains": list(allowlist.domains),
            },
        }

    def _keep_engaged(self, engaged: bool, record: dict | None) -> dict[str, Any]:
        """The record a writer other than the ceremony leaves.

        Engaged stays engaged: the record's state, or this process's latch,
        whichever says engaged. Only the ceremony records "not engaged".
        """
        still = self._latched or engaged
        if record is not None and record["engaged"] == still:
            return dict(record)
        latched = self._latched
        return self._make_record(
            engaged=still,
            allowlist=self._applied(engaged, record),
            killed_at=self._state.killed_at if latched else None,
            killed_by=self._state.killed_by if latched else None,
            kill_reason=self._state.kill_reason if latched else None,
            breaker=self._state.circuit_breaker_tripped if latched else False,
        )

    # -- Kill / restore ------------------------------------------------------

    def is_killed(self) -> bool:
        """Whether the kill switch is engaged: recorded, unreadable, or latched here."""
        return self._latched or self._read_record()[0]

    def is_enabled(self) -> bool:
        """Whether web search may run."""
        return not self.is_killed()

    def kill(
        self,
        user_id: str = "system",
        reason: str = "manual",
    ) -> dict[str, Any]:
        """Engage the kill switch and record it.

        Adding security (killing search) requires no ceremony. When the record
        cannot be written, search stays disabled in this process and the next
        kill retries the write.
        """
        with self._lock:
            engaged, record = self._read_record()
            if engaged and record is not None and not self._latched:
                return {
                    "success": True,
                    "already_killed": True,
                    "persisted": True,
                    "message": "Search already disabled",
                }

            now = time.time()
            self._latched = True
            self._state.killed_at = now
            self._state.killed_by = user_id
            self._state.kill_reason = reason

            def build(on_disk: bool, current: dict | None) -> dict[str, Any]:
                return self._make_record(
                    engaged=True,
                    allowlist=self._applied(on_disk, current),
                    killed_at=now,
                    killed_by=user_id,
                    kill_reason=reason,
                    breaker=self._state.circuit_breaker_tripped,
                )

            persisted = self._commit(build)
            if persisted:
                # The record holds the state now.
                self._latched = False

        try:
            from opti_oignon.security_mode import _audit_log
            _audit_log(
                "search_killswitch_engaged",
                severity="WARNING",
                user_id=user_id,
                reason=reason,
                persisted=persisted,
            )
        except Exception:
            pass

        logger.warning(
            "Search kill switch ENGAGED by %s (reason: %s); recorded: %s.",
            user_id, reason, persisted,
        )

        if persisted:
            message = "Search disabled; the state is recorded and holds across restarts."
        else:
            message = (
                "Search disabled in this process; the state could not be recorded "
                "and will not survive a restart."
            )
        return {"success": True, "persisted": persisted, "message": message}

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

        if self.is_enabled():
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
        """Confirm the re-enable ceremony and record it.

        Password and 2FA are the caller's (API route). This is the only
        writer that records "not engaged"; when that cannot be recorded,
        search stays disabled.
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

        with self._lock:
            def build(on_disk: bool, current: dict | None) -> dict[str, Any]:
                return self._make_record(
                    engaged=False,
                    allowlist=self._applied(on_disk, current),
                )

            if not self._commit(build):
                return {
                    "success": False,
                    "error": "not_recorded",
                    "message": "The re-enable could not be recorded; search stays disabled.",
                }
            self._latched = False
            self._state.reenable_pending = False
            self._state.circuit_breaker_tripped = False
            self._state.killed_at = 0.0
            self._state.killed_by = ""
            self._state.kill_reason = ""
            self._state.injection_count = 0
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
            "message": "Search re-enabled.",
        }

    def cancel_reenable(self) -> dict[str, Any]:
        """Cancel a pending re-enable request."""
        self._state.reenable_pending = False
        self._state.reenable_request_id = ""
        self._state.reenable_visual_code = ""
        return {"success": True, "message": "Re-enable request cancelled"}

    # -- Circuit breaker -----------------------------------------------------

    def record_injection(self, details: str = "") -> dict[str, Any]:
        """Record a search whose results carried a detected injection.

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
        """The allowlist that applies now, read from the record."""
        return self._applied(*self._read_record())

    def set_domain_allowlist(
        self, enabled: bool, domains: list[str] | None = None
    ) -> dict[str, Any]:
        """Record the domain allowlist.

        Entries are normalised to host names and those that are not host
        names are refused by name. The switch's engaged state is kept as the
        record has it. Returns ``{"persisted", "domains", "refused"}``; when
        nothing could be recorded, the previous allowlist still applies.
        """
        with self._lock:
            if domains is None:
                normalised, refused = list(self.domain_allowlist.domains), []
            else:
                normalised, refused = _normalise_domains(domains)
            wanted = DomainAllowlist(enabled=bool(enabled), domains=normalised)

            def build(on_disk: bool, current: dict | None) -> dict[str, Any]:
                kept = self._keep_engaged(on_disk, current)
                kept["domain_allowlist"] = {
                    "enabled": wanted.enabled,
                    "domains": list(wanted.domains),
                }
                return kept

            persisted = self._commit(build)
            if persisted and self._latched:
                # The record now says engaged: it holds what the latch held.
                self._latched = False
        return {"persisted": persisted, "domains": normalised, "refused": refused}

    def filter_results(self, results: list[Any]) -> list[Any]:
        """Filter search results through the domain allowlist that applies.

        Each result must have a .url or ['url'] attribute.
        """
        allowlist = self.domain_allowlist
        if not allowlist.enabled:
            return results

        filtered = []
        for r in results:
            url = getattr(r, "url", None)
            if url is None and isinstance(r, dict):
                url = r.get("url", "")
            if url and allowlist.is_allowed(url):
                filtered.append(r)
        return filtered

    # -- Status --------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Return full kill switch status for the API."""
        engaged, record = self._read_record()
        latched = self._latched
        killed = latched or engaged
        if latched or record is None:
            killed_at = self._state.killed_at if latched else None
            killed_by = self._state.killed_by if latched else None
            kill_reason = self._state.kill_reason if latched else None
            tripped = self._state.circuit_breaker_tripped if latched else False
        else:
            killed_at = record.get("killed_at") if killed else None
            killed_by = record.get("killed_by") if killed else None
            kill_reason = record.get("kill_reason") if killed else None
            tripped = record.get("circuit_breaker_tripped") is True
        allowlist = self._applied(engaged, record)
        return {
            "search_enabled": not killed,
            "killed_at": killed_at,
            "killed_by": killed_by,
            "kill_reason": kill_reason,
            "circuit_breaker_tripped": tripped,
            "injection_count": self._state.injection_count,
            "reenable_pending": self._state.reenable_pending,
            "domain_allowlist": {
                "enabled": allowlist.enabled,
                "domain_count": len(allowlist.domains),
                "domains": list(allowlist.domains),
            },
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

search_killswitch = SearchKillSwitch()
