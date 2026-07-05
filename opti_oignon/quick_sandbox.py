#!/usr/bin/env python3
"""
QUICK SANDBOX - OPTI-OIGNON v2.1.1 (S117)
==========================================

Lightweight orchestrator for chat-integrated sandboxed code execution.

When quick_sandbox mode is enabled, tool calls (execute_code, write_file,
read_file, list_files) in regular chat are transparently redirected to
an isolated sandbox session. The sandbox is auto-created on the first
tool call and auto-expires after a configurable timeout.

This bridges the UX gap between the full Coding Agent (multi-step
sandboxed workflow) and raw unsandboxed tool calls in normal chat.

Architecture:
  - QuickSandboxSession: wraps a SandboxManager session, tracks files
  - QuickSandboxManager: pool of active sessions, cleanup, config

Security:
  - Same bwrap isolation as the Coding Agent sandbox
  - No auto-approve: user must explicitly approve & download files
  - Sessions auto-expire after configurable timeout

Author: Leon
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from opti_oignon.sandbox_manager import (
        SANDBOX_AVAILABLE,
        SandboxManager,
        SandboxSession,
    )
    from opti_oignon.sandbox_manager import (
        sandbox_manager as _default_sandbox_manager,
    )
except ImportError:
    SANDBOX_AVAILABLE = False
    _default_sandbox_manager = None
    SandboxManager = None
    SandboxSession = None

try:
    from opti_oignon.file_tools import (
        FILE_TOOLS_AVAILABLE,
        _handle_sandbox_bash,
        _handle_sandbox_create_file,
        _handle_sandbox_view,
    )
except ImportError:
    FILE_TOOLS_AVAILABLE = False
    _handle_sandbox_bash = None
    _handle_sandbox_view = None
    _handle_sandbox_create_file = None

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class QuickSandboxConfig:
    """Configuration for quick sandbox mode."""
    enabled: bool = False
    auto_destroy_minutes: int = 30
    max_concurrent_quick_sessions: int = 3


def _load_config() -> QuickSandboxConfig:
    """Load quick sandbox config from sandbox.yaml."""
    cfg = QuickSandboxConfig()
    if not YAML_AVAILABLE:
        return cfg
    try:
        import os
        config_path = os.path.join(
            os.path.dirname(__file__), "config", "sandbox.yaml"
        )
        if os.path.isfile(config_path):
            with open(config_path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            qs = data.get("quick_sandbox", {})
            if isinstance(qs, dict):
                cfg.enabled = bool(qs.get("enabled", False))
                cfg.auto_destroy_minutes = int(
                    qs.get("auto_destroy_minutes", 30)
                )
                cfg.max_concurrent_quick_sessions = int(
                    qs.get("max_concurrent_quick_sessions", 3)
                )
    except Exception as exc:
        logger.warning("Failed to load quick_sandbox config: %s", exc)
    return cfg


# ---------------------------------------------------------------------------
# QuickSandboxSession
# ---------------------------------------------------------------------------

class QuickSandboxSession:
    """A lightweight sandbox session for a single chat request.

    Auto-creates a sandbox on first tool call and provides wrapped
    handlers for execute_code, write_file, read_file, list_files
    that redirect to the sandbox transparently.
    """

    def __init__(
        self,
        session_id: str,
        sandbox_mgr: "SandboxManager | None" = None,
        auto_destroy_minutes: int = 30,
        conversation_id: str | None = None,
        existing_sandbox_id: str | None = None,
    ):
        self._session_id = session_id
        self._conversation_id = conversation_id
        self._mgr = sandbox_mgr or _default_sandbox_manager
        self._sandbox_session: SandboxSession | None = None
        # When set, the session adopts this already-existing workspace
        # (typically one bound to the conversation) instead of creating a
        # fresh sandbox. The adopted workspace is never destroyed by this
        # wrapper: the binding layer owns its lifecycle.
        self._existing_sandbox_id = existing_sandbox_id
        self._adopted = False
        # The manager-side identifier all tool calls are routed to: the
        # session's own id, or the adopted workspace id after adoption.
        self._effective_id = session_id
        self._created_at = time.time()
        self._last_activity = time.time()
        self._auto_destroy_seconds = auto_destroy_minutes * 60
        self._files_created: list[str] = []
        # Number of chat turns currently executing against this session.
        # While positive the inactivity timeout does not apply: a long
        # inference between tool calls must never destroy the workspace
        # mid-turn. begin_turn / end_turn bracket the whole generation.
        self._active_turns = 0
        self._lock = threading.Lock()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def conversation_id(self) -> str | None:
        """The conversation that owns this sandbox (None if detached)."""
        return self._conversation_id

    @property
    def bound_sandbox_id(self) -> str | None:
        """The workspace this session adopts (None for an own sandbox)."""
        return self._existing_sandbox_id

    @property
    def effective_sandbox_id(self) -> str:
        """The manager-side id the sandbox API serves this session under.

        The session's own id until adoption, the adopted workspace id
        after. The done metadata must carry this id: the UI lists,
        previews, approves and downloads against it.
        """
        return self._effective_id

    @property
    def active(self) -> bool:
        return (
            self._sandbox_session is not None
            and self._sandbox_session.active
        )

    @property
    def expired(self) -> bool:
        """Whether the session has exceeded its auto-destroy timeout.

        A session with a turn in flight never expires: the inactivity
        window is a between-turns notion (see begin_turn / end_turn).
        """
        if self._active_turns > 0:
            return False
        return (
            time.time() - self._last_activity > self._auto_destroy_seconds
        )

    @property
    def created_at(self) -> float:
        return self._created_at

    @property
    def auto_destroy_minutes(self) -> int:
        """Current inactivity timeout before auto-destroy, in minutes."""
        return int(self._auto_destroy_seconds // 60)

    @property
    def seconds_until_expiry(self) -> float:
        """Seconds of inactivity remaining before auto-destroy (>= 0).

        While a turn is in flight the full window is reported: the
        countdown only starts once the turn closes.
        """
        if self._active_turns > 0:
            return float(self._auto_destroy_seconds)
        remaining = self._auto_destroy_seconds - (
            time.time() - self._last_activity
        )
        return max(0.0, remaining)

    def set_auto_destroy_minutes(self, minutes: int) -> None:
        """Change the inactivity timeout after the session was created.

        The activity clock is reset so the new window starts now, giving a
        predictable fresh countdown (useful to keep a session alive longer).
        """
        if minutes <= 0:
            raise ValueError("auto_destroy_minutes must be positive")
        with self._lock:
            self._auto_destroy_seconds = minutes * 60
            self._last_activity = time.time()

    def begin_turn(self) -> None:
        """Mark a chat turn as executing against this session.

        While at least one turn is in flight the session cannot expire,
        so a long inference between tool calls never loses its workspace.
        Callers must pair every begin_turn with end_turn (try/finally).
        """
        with self._lock:
            self._active_turns += 1
            self._last_activity = time.time()

    def end_turn(self) -> None:
        """Mark the end of a chat turn.

        The inactivity window restarts from the end of the turn. Extra
        calls are harmless: the counter never goes below zero.
        """
        with self._lock:
            if self._active_turns > 0:
                self._active_turns -= 1
            self._last_activity = time.time()

    @property
    def files_created(self) -> list[str]:
        """List of file paths created in this session."""
        return list(self._files_created)

    # Upper bound on adopted file names announced to the model through the
    # tools prompt (large workspaces stay listable via list_files).
    ADOPTED_FILES_ANNOUNCE_CAP = 50

    def _list_adopted_files(self) -> list[str]:
        """Relative paths already present in the adopted workspace."""
        if self._mgr is None:
            return []
        try:
            entries = self._mgr.extract_files(self._effective_id)
        except Exception:
            return []
        names = [
            str(entry.get("path"))
            for entry in entries
            if entry.get("path")
        ]
        return names[: self.ADOPTED_FILES_ANNOUNCE_CAP]

    def _ensure_sandbox(self) -> None:
        """Lazily create the sandbox on first use."""
        if self._sandbox_session is not None and self._sandbox_session.active:
            self._last_activity = time.time()
            return
        if self._mgr is None:
            raise RuntimeError("Sandbox manager not available")
        if self._existing_sandbox_id is not None:
            adopted = self._mgr.get_session(self._existing_sandbox_id)
            if adopted is not None and adopted.active:
                self._sandbox_session = adopted
                self._adopted = True
                self._effective_id = self._existing_sandbox_id
                self._files_created = self._list_adopted_files()
                logger.info(
                    "Quick sandbox adopted bound workspace %s for %s",
                    self._existing_sandbox_id, self._session_id,
                )
                return
            logger.warning(
                "Bound workspace %s unavailable; creating a fresh sandbox "
                "for %s", self._existing_sandbox_id, self._session_id,
            )
        self._sandbox_session = self._mgr.create_sandbox(
            self._session_id, allow_degraded=True
        )
        logger.info(
            "Quick sandbox session created: %s", self._session_id
        )

    def handle_execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: int = 30,
    ) -> str:
        """Execute code inside the sandbox.

        Wraps execute_code by converting it to a sandbox bash command.
        """
        self._ensure_sandbox()
        self._last_activity = time.time()

        # Build a shell command that runs the code
        if language.lower() in ("python", "python3"):
            # Write code to a temp file and execute
            escaped = code.replace("'", "'\\''")
            cmd = f"python3 -c '{escaped}'"
        elif language.lower() in ("r", "rscript"):
            escaped = code.replace("'", "'\\''")
            cmd = f"Rscript -e '{escaped}'"
        elif language.lower() in ("bash", "sh"):
            cmd = code
        else:
            return f"Unsupported language: {language}"

        try:
            result = _handle_sandbox_bash(
                self._effective_id, cmd, timeout,
                _sandbox_manager=self._mgr,
            )
            return result
        except Exception as exc:
            return f"Quick sandbox execution error: {exc}"
        finally:
            # Completion also counts as activity: a call that outlives the
            # window must not leave the session already expired at return.
            self._last_activity = time.time()

    def handle_write_file(self, path: str, content: str) -> str:
        """Write a file inside the sandbox."""
        self._ensure_sandbox()
        self._last_activity = time.time()

        try:
            result = _handle_sandbox_create_file(
                self._effective_id, path, content,
                _sandbox_manager=self._mgr,
            )
            with self._lock:
                if path not in self._files_created:
                    self._files_created.append(path)
            return result
        except Exception as exc:
            return f"Quick sandbox write error: {exc}"
        finally:
            self._last_activity = time.time()

    def handle_read_file(self, path: str) -> str:
        """Read a file from inside the sandbox."""
        self._ensure_sandbox()
        self._last_activity = time.time()

        try:
            result = _handle_sandbox_view(
                self._effective_id, path, 0, 0,
                _sandbox_manager=self._mgr,
            )
            return result
        except Exception as exc:
            return f"Quick sandbox read error: {exc}"
        finally:
            self._last_activity = time.time()

    def handle_list_files(self, path: str = ".") -> str:
        """List files inside the sandbox."""
        self._ensure_sandbox()
        self._last_activity = time.time()

        try:
            result = _handle_sandbox_view(
                self._effective_id, path, 0, 0,
                _sandbox_manager=self._mgr,
            )
            return result
        except Exception as exc:
            return f"Quick sandbox list error: {exc}"
        finally:
            self._last_activity = time.time()

    def get_sandbox_files(self) -> list[dict[str, Any]]:
        """Get the list of files in the sandbox for the UI."""
        if not self.active or self._mgr is None:
            return []
        try:
            return self._mgr.extract_files(self._effective_id)
        except Exception:
            return []

    def destroy(self) -> bool:
        """Destroy the sandbox session and clean up."""
        if self._mgr is None or self._sandbox_session is None:
            return False
        if self._adopted:
            # The bound workspace belongs to the binding layer; only the
            # wrapper is dropped, the workspace and its files survive.
            self._sandbox_session = None
            logger.info(
                "Quick sandbox wrapper detached; bound workspace %s "
                "preserved", self._effective_id,
            )
            return True
        try:
            result = self._mgr.destroy_sandbox(self._session_id)
            self._sandbox_session = None
            logger.info(
                "Quick sandbox session destroyed: %s", self._session_id
            )
            return result
        except Exception as exc:
            logger.warning(
                "Quick sandbox destroy failed: %s: %s",
                self._session_id, exc,
            )
            return False


# ---------------------------------------------------------------------------
# QuickSandboxManager
# ---------------------------------------------------------------------------

class QuickSandboxManager:
    """Pool of active quick sandbox sessions.

    Manages session lifecycle, expiry cleanup, and provides the
    entry point for tool routing.
    """

    def __init__(
        self,
        sandbox_mgr: "SandboxManager | None" = None,
        config: QuickSandboxConfig | None = None,
    ):
        self._mgr = sandbox_mgr or _default_sandbox_manager
        self._config = config or _load_config()
        self._sessions: dict[str, QuickSandboxSession] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._config.enabled = value

    @property
    def config(self) -> QuickSandboxConfig:
        return self._config

    @property
    def available(self) -> bool:
        """Whether quick sandbox can operate (dependencies present)."""
        return (
            SANDBOX_AVAILABLE
            and FILE_TOOLS_AVAILABLE
            and self._mgr is not None
        )

    def get_or_create_session(
        self, request_id: str | None = None,
        bound_sandbox_id: str | None = None,
    ) -> QuickSandboxSession:
        """Get an existing session or create a new one for a request.

        Args:
            request_id: A unique ID for the chat request (conversation_id
                or a generated UUID). Used as the session key.
            bound_sandbox_id: An existing workspace explicitly bound to
                the conversation. When provided, the session adopts that
                workspace instead of creating a fresh sandbox; a live
                session adopting a different workspace (or none) is
                replaced so a rebinding takes effect immediately.

        Returns:
            A QuickSandboxSession ready for tool calls.

        Raises:
            RuntimeError: If max concurrent sessions exceeded or
                sandbox unavailable.
        """
        if not self.available:
            raise RuntimeError(
                "Quick sandbox not available (missing dependencies)"
            )

        rid = request_id or f"qs-{uuid.uuid4().hex[:12]}"

        with self._lock:
            # Return existing active session (unless a binding points it
            # at a different workspace: the binding takes priority).
            existing = self._sessions.get(rid)
            if existing is not None and not existing.expired:
                if (
                    bound_sandbox_id is not None
                    and existing.bound_sandbox_id != bound_sandbox_id
                ):
                    existing.destroy()
                    del self._sessions[rid]
                    logger.info(
                        "Quick sandbox rebinding %s to workspace %s",
                        rid, bound_sandbox_id,
                    )
                else:
                    return existing

            # Clean up expired session if it exists
            existing = self._sessions.get(rid)
            if existing is not None and existing.expired:
                existing.destroy()
                del self._sessions[rid]

            # Check concurrent limit
            active_count = sum(
                1 for s in self._sessions.values() if not s.expired
            )
            if active_count >= self._config.max_concurrent_quick_sessions:
                raise RuntimeError(
                    f"Maximum concurrent quick sandbox sessions reached "
                    f"({self._config.max_concurrent_quick_sessions})"
                )

            # Create new session
            session = QuickSandboxSession(
                session_id=rid,
                sandbox_mgr=self._mgr,
                auto_destroy_minutes=self._config.auto_destroy_minutes,
                conversation_id=request_id,
                existing_sandbox_id=bound_sandbox_id,
            )
            self._sessions[rid] = session
            return session

    def get_session(self, session_id: str) -> QuickSandboxSession | None:
        """Get an existing session by ID."""
        with self._lock:
            return self._sessions.get(session_id)

    def destroy_session(self, session_id: str) -> bool:
        """Destroy a specific quick sandbox session."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        return session.destroy()

    def cleanup_expired(self) -> int:
        """Destroy all expired sessions. Returns count destroyed."""
        to_destroy: list[str] = []
        with self._lock:
            for sid, session in self._sessions.items():
                if session.expired:
                    to_destroy.append(sid)

        count = 0
        for sid in to_destroy:
            if self.destroy_session(sid):
                count += 1
        return count

    def set_session_auto_destroy(
        self, session_id: str, minutes: int
    ) -> bool:
        """Change the auto-destroy timeout of an existing session.

        Returns True if the session exists and was updated, False otherwise.
        Raises ValueError if minutes is not positive.
        """
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            return False
        session.set_auto_destroy_minutes(minutes)
        return True

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all active quick sandbox sessions."""
        with self._lock:
            result = []
            for sid, session in self._sessions.items():
                result.append({
                    "session_id": sid,
                    "active": session.active,
                    "expired": session.expired,
                    "created_at": session.created_at,
                    "files_created": session.files_created,
                    "auto_destroy_minutes": session.auto_destroy_minutes,
                    "seconds_until_expiry": session.seconds_until_expiry,
                })
            return result

    def active_session_count(self) -> int:
        """Count of active (non-expired) sessions."""
        with self._lock:
            return sum(
                1 for s in self._sessions.values()
                if not s.expired
            )

    def get_status(self) -> dict[str, Any]:
        """Get quick sandbox status for API responses."""
        return {
            "enabled": self._config.enabled,
            "available": self.available,
            "auto_destroy_minutes": self._config.auto_destroy_minutes,
            "max_concurrent_sessions": (
                self._config.max_concurrent_quick_sessions
            ),
            "active_sessions": self.active_session_count(),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

quick_sandbox_manager = QuickSandboxManager()

QUICK_SANDBOX_AVAILABLE = (
    SANDBOX_AVAILABLE
    and FILE_TOOLS_AVAILABLE
)
