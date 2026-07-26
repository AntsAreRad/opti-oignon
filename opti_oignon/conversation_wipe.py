#!/usr/bin/env python3
"""
Conversation RAM Wipe for Opti-Oignon.

Provides best-effort zeroing of in-memory conversation data (messages,
system prompts, tool call results, embedding buffers) to reduce the
window during which sensitive text resides in process memory.

Architecture
------------

``ConversationWipeManager`` is a module-level singleton that maintains a
registry of in-memory buffers associated with conversation IDs.  When a
conversation is closed, archived, or manually wiped, the manager:

  1. Iterates over all registered buffers for that conversation.
  2. For each buffer, zeroes the underlying memory using ``ctypes.memset``
     on CPython string/bytes internals (best-effort).
  3. Replaces the Python reference with a sentinel so the GC can reclaim.
  4. Logs a ``conversation_wipe`` event to the audit chain.

In **Bulbe mode**, ``bulbe_wipe_per_turn`` triggers an automatic wipe
after every LLM response so that no conversation history accumulates
in RAM beyond the current exchange.

Scope: RAM vs disk
--------------------------------

By default a wipe is **RAM-only**: it zeroes in-memory buffers but does NOT
delete the persisted conversation rows on disk. Conversations are persisted
through the conversation manager (SQLCipher-encrypted at rest in Bulbe mode), so
the on-disk copy survives a wipe and remains protected by encryption. Deleting
the persisted rows is a separate operation (``delete_conversation``). The wipe
endpoints expose an opt-in ``purge_disk`` flag for a *full wipe* that performs
both the RAM zeroing and the on-disk row deletion; it is off by default so the
emergency wipe is non-destructive to stored history unless explicitly requested.

Limitations (documented for transparency)
------------------------------------------

- CPython's garbage collector may retain copies of Python ``str`` and
  ``bytes`` objects that were interned or referenced elsewhere.
- Short strings (<= 256 chars by default) may be interned by CPython
  and cannot be reliably zeroed.
- The ``ctypes.memset`` approach is CPython-specific and version-fragile.
- This is **defense-in-depth**, not a guarantee.  For true memory
  isolation, use a separate process per conversation with ``mlock``.

Configuration (security.yaml)
------------------------------

.. code-block:: yaml

   hardening:
     auto_wipe_on_close: true
     bulbe_wipe_per_turn: true
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import sys
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Low-level memory zeroing (reuses pattern from secure_bytes.py)
# ---------------------------------------------------------------------------

_libc: Any = None
_HAS_MEMSET = False

try:
    if sys.platform == "win32":
        _libc = ctypes.cdll.msvcrt
    else:
        libc_name = ctypes.util.find_library("c")
        if libc_name:
            _libc = ctypes.CDLL(libc_name, use_errno=True)
    if _libc and hasattr(_libc, "memset"):
        _libc.memset.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
        ]
        _libc.memset.restype = ctypes.c_void_p
        _HAS_MEMSET = True
except Exception as exc:
    logger.debug("conversation_wipe: failed to load libc memset: %s", exc)


# A genuine per-conversation secret reaching the wipe path is held only by the
# registry entry and a few locals -- a reference count in the single digits.
# Interned string literals, single-character strings, and cached single-byte
# ``bytes`` are shared across the whole interpreter: on Python 3.12+ they are
# immortal (PEP 683) with a frozen sentinel refcount (~4.3e9), and on 3.10/3.11
# they carry large finite counts. Zeroing such an object's buffer in place would
# corrupt every other holder process-wide (e.g. the literal "data" used as a
# dict key elsewhere). We therefore refuse to scrub any ``str``/``bytes`` that is
# not exclusively ours; refusing is the safe failure -- a secret may linger, but
# the running interpreter is never corrupted. Mutable ``bytearray`` secrets are
# unaffected and are always scrubbed.
_MAX_EXCLUSIVE_REFCOUNT = 50


def _is_exclusively_owned(obj: Any) -> bool:
    """True when ``obj`` is held only by the wipe path, so zeroing it is safe.

    Shared/interned/immortal immutables fail this test and must not be zeroed in
    place. Errs on the side of refusing (returns False) on any uncertainty.
    """
    try:
        return sys.getrefcount(obj) <= _MAX_EXCLUSIVE_REFCOUNT
    except Exception:  # pragma: no cover - defensive
        return False


def _zero_string(s: str) -> bool:
    """Best-effort zero of a Python str object's internal buffer.

    CPython stores str data as a compact ASCII or UCS buffer.
    We attempt to overwrite it via ctypes.  This is inherently
    unsafe and CPython-specific, so we only do it for a string we
    exclusively own (see ``_is_exclusively_owned``); shared or interned
    strings are left untouched to avoid corrupting other holders.

    Returns True if the memset was attempted, False otherwise.
    """
    if not _HAS_MEMSET or not isinstance(s, str) or len(s) == 0:
        return False
    if not _is_exclusively_owned(s):
        return False
    try:
        # CPython compact ASCII: data follows the PyASCIIObject header.
        # sys.getsizeof gives the full size; the data area starts at
        # id(s) + (sys.getsizeof(s) - len(s) - 1) approximately.
        # A simpler heuristic: use the id + struct offset.
        total_size = sys.getsizeof(s)
        data_len = len(s)
        # The data bytes sit at the end of the object.
        buf_start = id(s) + total_size - data_len - 1
        _libc.memset(buf_start, 0, data_len)
        return True
    except Exception:
        return False


def _zero_bytes(b: bytes) -> bool:
    """Best-effort zero of a Python bytes object's internal buffer.

    Only zeroes a bytes object we exclusively own; shared/cached bytes (e.g.
    single-byte singletons) are left untouched to avoid corrupting other holders.
    """
    if not _HAS_MEMSET or not isinstance(b, bytes) or len(b) == 0:
        return False
    if not _is_exclusively_owned(b):
        return False
    try:
        buf_addr = id(b) + sys.getsizeof(b"") - 1
        _libc.memset(buf_addr, 0, len(b))
        return True
    except Exception:
        return False


def _zero_object(obj: Any) -> int:
    """Attempt to zero an object.  Returns count of fields zeroed."""
    zeroed = 0
    if isinstance(obj, str):
        if _zero_string(obj):
            zeroed += 1
    elif isinstance(obj, bytes):
        if _zero_bytes(obj):
            zeroed += 1
    elif isinstance(obj, bytearray):
        for i in range(len(obj)):
            obj[i] = 0
        zeroed += 1
    elif isinstance(obj, dict):
        for key in list(obj.keys()):
            zeroed += _zero_object(obj[key])
            obj[key] = None
        obj.clear()
        zeroed += 1
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            zeroed += _zero_object(item)
            obj[i] = None
        obj.clear()
        zeroed += 1
    return zeroed


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WipeResult:
    """Result of a conversation wipe operation."""
    conversation_id: str
    buffers_wiped: int = 0
    fields_zeroed: int = 0
    success: bool = True
    error: str = ""
    timestamp: float = field(default_factory=time.time)
    memset_available: bool = _HAS_MEMSET
    # Set when the opt-in full wipe also purged persisted rows.
    disk_purged: bool = False
    rows_deleted: int = 0


@dataclass
class HardeningStatus:
    """Combined hardening status for the GET endpoint."""
    conversation_wipe_available: bool = False
    auto_wipe_on_close: bool = True
    bulbe_wipe_per_turn: bool = True
    active_conversations: int = 0
    total_registered_buffers: int = 0
    memset_available: bool = _HAS_MEMSET
    swap_status: dict[str, Any] = field(default_factory=dict)
    ollama_log_status: dict[str, Any] = field(default_factory=dict)
    network_status: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ConversationWipeManager singleton
# ---------------------------------------------------------------------------

class ConversationWipeManager:
    """Manages registration and wiping of in-memory conversation buffers.

    Usage::

        manager = ConversationWipeManager()
        manager.register_buffer("conv-123", message_list)
        manager.register_buffer("conv-123", system_prompt_str)
        # ... later ...
        result = manager.wipe("conv-123")
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # conversation_id -> list of (weak_ref_or_strong_ref, label)
        self._buffers: dict[str, list[tuple[Any, str]]] = {}
        self._config = self._load_config()

    # -- Configuration -----------------------------------------------------

    @staticmethod
    def _load_config() -> dict[str, Any]:
        """Load hardening config from security.yaml."""
        try:
            import yaml
            config_path = os.path.join(
                os.path.dirname(__file__), "config", "security.yaml"
            )
            if os.path.isfile(config_path):
                with open(config_path, encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
                return data.get("hardening", {})
        except Exception as exc:
            logger.debug("conversation_wipe: config load failed: %s", exc)
        return {}

    def reload_config(self) -> None:
        """Reload configuration from disk."""
        self._config = self._load_config()

    @property
    def auto_wipe_on_close(self) -> bool:
        """Whether conversations are auto-wiped on close/archive."""
        return self._config.get("auto_wipe_on_close", True)

    @property
    def bulbe_wipe_per_turn(self) -> bool:
        """Whether Bulbe mode wipes after every LLM response."""
        return self._config.get("bulbe_wipe_per_turn", True)

    # -- Buffer Registration -----------------------------------------------

    def register_buffer(
        self, conversation_id: str, obj: Any, label: str = ""
    ) -> None:
        """Track an in-memory object for later wiping.

        For mutable objects (list, dict, bytearray) we store a direct
        reference.  For immutable objects (str, bytes) we store the
        object directly since weak references to str/bytes are not
        supported by CPython.
        """
        with self._lock:
            if conversation_id not in self._buffers:
                self._buffers[conversation_id] = []

            # Try weakref for mutable objects; fall back to strong ref.
            try:
                ref = weakref.ref(obj)
                self._buffers[conversation_id].append((ref, label or "weakref"))
            except TypeError:
                # str, bytes, etc. do not support weakref
                self._buffers[conversation_id].append((obj, label or "strong"))

    def _resolve_buffers(self, conversation_id: str) -> list[Any]:
        """Resolve all live buffer references for a conversation."""
        result = []
        entries = self._buffers.get(conversation_id, [])
        for ref_or_obj, _label in entries:
            if callable(ref_or_obj) and hasattr(ref_or_obj, "__callback__"):
                # It is a weakref
                obj = ref_or_obj()
                if obj is not None:
                    result.append(obj)
            elif isinstance(ref_or_obj, weakref.ref):
                obj = ref_or_obj()
                if obj is not None:
                    result.append(obj)
            else:
                result.append(ref_or_obj)
        return result

    # -- Wipe Operations ---------------------------------------------------

    def wipe(self, conversation_id: str, *, purge_disk: bool = False) -> WipeResult:
        """Wipe all registered buffers for a conversation.

        Zeroes memory (best-effort), removes from registry, and logs
        to the audit chain.

        By default this is RAM-only -- it does NOT delete the
        persisted (SQLCipher-encrypted in Bulbe) conversation rows on disk.
        Pass ``purge_disk=True`` to opt into a full wipe that also deletes the
        persisted rows via the conversation manager (best-effort; the RAM wipe
        still succeeds if the disk purge is unavailable).
        """
        result = WipeResult(conversation_id=conversation_id)

        with self._lock:
            buffers = self._resolve_buffers(conversation_id)
            result.buffers_wiped = len(buffers)

            for obj in buffers:
                try:
                    result.fields_zeroed += _zero_object(obj)
                except Exception as exc:
                    logger.debug(
                        "conversation_wipe: error zeroing buffer: %s", exc
                    )

            # Remove from registry
            self._buffers.pop(conversation_id, None)

        # Audit chain entry
        self._chain_log_wipe(conversation_id, result)

        logger.info(
            "Wiped conversation %s: %d buffers, %d fields zeroed",
            conversation_id, result.buffers_wiped, result.fields_zeroed,
        )

        if purge_disk:
            if self._purge_disk(conversation_id):
                result.disk_purged = True
                result.rows_deleted = 1
                logger.warning(
                    "Full wipe: purged persisted rows for conversation %s",
                    conversation_id,
                )
        return result

    def wipe_all(self, *, purge_disk: bool = False) -> list[WipeResult]:
        """Emergency wipe: zero all registered conversations.

        RAM-only by default. With ``purge_disk=True`` this also
        deletes ALL persisted conversation rows from disk (not only those with a
        registered RAM buffer), returning a WipeResult per purged conversation.
        """
        with self._lock:
            conv_ids = list(self._buffers.keys())

        results = []
        for cid in conv_ids:
            results.append(self.wipe(cid))  # RAM only; disk handled uniformly below

        if purge_disk:
            purged_ids = self._purge_disk_all()
            ram_wiped = {r.conversation_id for r in results}
            for r in results:
                if r.conversation_id in purged_ids:
                    r.disk_purged = True
                    r.rows_deleted = 1
            for cid in purged_ids:
                if cid not in ram_wiped:
                    results.append(
                        WipeResult(
                            conversation_id=cid,
                            buffers_wiped=0,
                            disk_purged=True,
                            rows_deleted=1,
                        )
                    )
            logger.warning(
                "Full emergency wipe: purged %d persisted conversations from disk",
                len(purged_ids),
            )

        # Log the emergency wipe
        self._chain_log_emergency(len(results))

        logger.warning("Emergency wipe: cleared %d conversations", len(results))
        return results

    @staticmethod
    def _get_conversation_manager() -> Any | None:
        """Lazy-load the conversation manager singleton for disk purges.

        Lazy + best-effort: the wipe module is otherwise standalone, so this
        coupling is deferred to call time and degrades gracefully when the
        conversation layer is unavailable.
        """
        try:
            from opti_oignon.conversation import conversation_manager

            return conversation_manager
        except Exception as exc:  # pragma: no cover - import-time degradation
            logger.warning(
                "conversation_wipe: conversation manager unavailable for disk "
                "purge: %s",
                exc,
            )
            return None

    def _purge_disk(self, conversation_id: str) -> bool:
        """Delete the persisted rows for one conversation (best-effort)."""
        mgr = self._get_conversation_manager()
        if mgr is None:
            return False
        try:
            return bool(mgr.delete_conversation(conversation_id))
        except Exception as exc:
            logger.warning(
                "conversation_wipe: disk purge failed for %s: %s",
                conversation_id, exc,
            )
            return False

    def _purge_disk_all(self) -> set[str]:
        """Delete every persisted conversation from disk (best-effort).

        Paginates through the conversation store deleting as it goes; the
        progress guard prevents an infinite loop if a batch cannot be deleted.
        """
        mgr = self._get_conversation_manager()
        if mgr is None:
            return set()
        purged: set[str] = set()
        try:
            batch = 500
            while True:
                convs = mgr.list_conversations(limit=batch, offset=0)
                if not convs:
                    break
                progressed = False
                for c in convs:
                    cid = getattr(c, "id", None)
                    if cid and mgr.delete_conversation(cid):
                        purged.add(cid)
                        progressed = True
                if not progressed:
                    break
        except Exception as exc:
            logger.warning("conversation_wipe: bulk disk purge error: %s", exc)
        return purged

    def on_conversation_close(self, conversation_id: str) -> WipeResult | None:
        """Hook: called when a conversation is closed or archived.

        Wipes only if auto_wipe_on_close is enabled.
        """
        if not self.auto_wipe_on_close:
            return None
        return self.wipe(conversation_id)

    def on_bulbe_response(self, conversation_id: str) -> WipeResult | None:
        """Hook: called after every LLM response in Bulbe mode.

        Wipes the conversation buffers so no history accumulates.
        """
        if not self.bulbe_wipe_per_turn:
            return None
        try:
            from opti_oignon.security_mode import is_bulbe
            if not is_bulbe():
                return None
        except ImportError:
            return None
        return self.wipe(conversation_id)

    # -- Status ------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return current wipe manager status."""
        with self._lock:
            active = len(self._buffers)
            total_buffers = sum(
                len(entries) for entries in self._buffers.values()
            )
        return {
            "available": True,
            "auto_wipe_on_close": self.auto_wipe_on_close,
            "bulbe_wipe_per_turn": self.bulbe_wipe_per_turn,
            "active_conversations": active,
            "total_registered_buffers": total_buffers,
            "memset_available": _HAS_MEMSET,
        }

    def get_registered_conversations(self) -> list[str]:
        """Return list of conversation IDs with registered buffers."""
        with self._lock:
            return list(self._buffers.keys())

    # -- Audit Chain Integration -------------------------------------------

    @staticmethod
    def _chain_log_wipe(conversation_id: str, result: WipeResult) -> None:
        """Log a conversation wipe event to the audit chain."""
        try:
            from opti_oignon.signed_audit_log import chain_log
            chain_log(
                event_type="conversation_wipe",
                source="conversation_wipe",
                action=f"Wiped conversation {conversation_id}",
                severity="INFO",
                conversation_id=conversation_id,
                buffers_wiped=result.buffers_wiped,
                fields_zeroed=result.fields_zeroed,
                memset_available=result.memset_available,
            )
        except ImportError:
            pass

    @staticmethod
    def _chain_log_emergency(count: int) -> None:
        """Log an emergency wipe-all event."""
        try:
            from opti_oignon.signed_audit_log import chain_log
            chain_log(
                event_type="conversation_wipe_all",
                source="conversation_wipe",
                action=f"Emergency wipe: cleared {count} conversations",
                severity="WARNING",
                conversations_wiped=count,
            )
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

import os  # noqa: E402  (deferred to avoid circular with config)

conversation_wipe_manager = ConversationWipeManager()

# Convenience aliases
register_buffer = conversation_wipe_manager.register_buffer
wipe_conversation = conversation_wipe_manager.wipe
wipe_all_conversations = conversation_wipe_manager.wipe_all
on_conversation_close = conversation_wipe_manager.on_conversation_close
on_bulbe_response = conversation_wipe_manager.on_bulbe_response

CONVERSATION_WIPE_AVAILABLE = True
