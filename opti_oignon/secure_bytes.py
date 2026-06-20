#!/usr/bin/env python3
"""
Key Memory Protection for Opti-Oignon (S126).

Provides ``SecureBytes``, a wrapper around sensitive key material that:

  1. Locks the memory page via ``mlock()`` to prevent swapping to disk
  2. Zeros the buffer via ``memset(0)`` on ``__del__`` or explicit wipe
  3. Hides content from ``repr()`` / ``str()`` / logging
  4. Registers a ``SIGTERM`` handler to wipe all tracked keys on shutdown

This is defense-in-depth: even if an attacker gains read access to
process memory after shutdown, the key material is zeroed.

Platform support:
  - Linux: full ``mlock()`` + ``memset()`` via ctypes
  - macOS: same via ctypes (libc)
  - Windows/other: graceful degradation (no mlock, still zeroes on del)

Usage::

    from opti_oignon.secure_bytes import SecureBytes

    key = SecureBytes(os.urandom(32))
    # Use key.as_bytes() for crypto operations
    raw = key.as_bytes()
    # When done:
    key.wipe()  # or let __del__ handle it
"""

from __future__ import annotations

import atexit
import ctypes
import ctypes.util
import logging
import os
import signal
import sys
import weakref
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Platform-specific libc bindings
# ---------------------------------------------------------------------------

_libc: Any = None
_HAS_MLOCK = False
_HAS_MEMSET = False

try:
    if sys.platform == "win32":
        _libc = ctypes.cdll.msvcrt
    else:
        libc_name = ctypes.util.find_library("c")
        if libc_name:
            _libc = ctypes.CDLL(libc_name, use_errno=True)

    if _libc:
        # mlock(addr, len) -> int
        if hasattr(_libc, "mlock"):
            _libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            _libc.mlock.restype = ctypes.c_int
            _HAS_MLOCK = True

        # munlock(addr, len) -> int
        if hasattr(_libc, "munlock"):
            _libc.munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            _libc.munlock.restype = ctypes.c_int

        # memset(s, c, n) -> void*
        if hasattr(_libc, "memset"):
            _libc.memset.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
            ]
            _libc.memset.restype = ctypes.c_void_p
            _HAS_MEMSET = True

except Exception as exc:
    logger.debug("Failed to load libc for memory protection: %s", exc)


# ---------------------------------------------------------------------------
# Global registry of live SecureBytes instances (weak references)
# ---------------------------------------------------------------------------

_live_keys: list[weakref.ref] = []


def _wipe_all_keys() -> None:
    """Wipe all tracked SecureBytes instances.

    Called on SIGTERM, SIGINT, and atexit.
    """
    count = 0
    for ref in _live_keys:
        obj = ref()
        if obj is not None and not obj._wiped:
            obj.wipe()
            count += 1
    _live_keys.clear()
    if count > 0:
        logger.info("Wiped %d encryption keys from memory", count)


def _sigterm_handler(signum: int, frame: Any) -> None:
    """SIGTERM/SIGINT handler: wipe all keys before exit."""
    _wipe_all_keys()
    # Re-raise with default handler
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


# Register handlers
try:
    signal.signal(signal.SIGTERM, _sigterm_handler)
except (OSError, ValueError):
    pass  # Not main thread or signal not available

try:
    # Only register SIGINT handler if we are in the main thread
    import threading
    if threading.current_thread() is threading.main_thread():
        _original_sigint = signal.getsignal(signal.SIGINT)
        def _sigint_handler(signum: int, frame: Any) -> None:
            _wipe_all_keys()
            if callable(_original_sigint) and _original_sigint not in (
                signal.SIG_DFL, signal.SIG_IGN
            ):
                _original_sigint(signum, frame)
            else:
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)
        signal.signal(signal.SIGINT, _sigint_handler)
except (OSError, ValueError):
    pass

atexit.register(_wipe_all_keys)


# ---------------------------------------------------------------------------
# SecureBytes
# ---------------------------------------------------------------------------

class SecureBytes:
    """A secure container for sensitive key material.

    Features:
      - ``mlock()`` pins memory to prevent swap-to-disk
      - ``memset(0)`` on ``__del__`` or ``wipe()``
      - ``__repr__`` returns ``'<SecureBytes [REDACTED]>'``
      - Tracked in global registry for SIGTERM cleanup
      - Supports context manager protocol

    Usage::

        key = SecureBytes(raw_key_bytes)
        try:
            do_crypto(key.as_bytes())
        finally:
            key.wipe()

        # Or as context manager:
        with SecureBytes(raw_key_bytes) as key:
            do_crypto(key.as_bytes())
    """

    __slots__ = ("_buf", "_length", "_wiped", "_mlocked", "__weakref__")

    def __init__(self, data: bytes | bytearray) -> None:
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("SecureBytes requires bytes or bytearray")

        # Store as mutable bytearray so we can zero it
        self._buf = bytearray(data)
        self._length = len(data)
        self._wiped = False
        self._mlocked = False

        # Zero the source if it was a bytearray (caller's copy)
        if isinstance(data, bytearray):
            _zero_bytearray(data)

        # Lock memory page
        if _HAS_MLOCK and self._length > 0:
            try:
                addr = (ctypes.c_char * self._length).from_buffer(self._buf)
                result = _libc.mlock(ctypes.addressof(addr), self._length)
                if result == 0:
                    self._mlocked = True
                else:
                    # mlock can fail if RLIMIT_MEMLOCK is too low
                    logger.debug(
                        "mlock() failed (errno=%d). Key may be swappable.",
                        ctypes.get_errno(),
                    )
            except Exception as exc:
                logger.debug("mlock() exception: %s", exc)

        # Register in global tracker
        _live_keys.append(weakref.ref(self))

    def as_bytes(self) -> bytes:
        """Return the key material as bytes.

        Raises RuntimeError if already wiped.
        """
        if self._wiped:
            raise RuntimeError("SecureBytes has been wiped")
        return bytes(self._buf)

    def wipe(self) -> None:
        """Zero the key material immediately.

        After calling wipe(), ``as_bytes()`` will raise RuntimeError.
        """
        if self._wiped:
            return

        if self._length > 0:
            # Zero via memset if available (bypasses Python optimizations)
            if _HAS_MEMSET:
                try:
                    addr = (ctypes.c_char * self._length).from_buffer(self._buf)
                    _libc.memset(ctypes.addressof(addr), 0, self._length)
                except Exception:
                    # Fallback: Python-level zeroing
                    _zero_bytearray(self._buf)
            else:
                _zero_bytearray(self._buf)

            # Unlock memory
            if self._mlocked and _libc and hasattr(_libc, "munlock"):
                try:
                    addr = (ctypes.c_char * self._length).from_buffer(self._buf)
                    _libc.munlock(ctypes.addressof(addr), self._length)
                except Exception:
                    pass

        self._wiped = True

    @property
    def length(self) -> int:
        """Length of the key material in bytes."""
        return self._length

    @property
    def is_wiped(self) -> bool:
        """Whether the key material has been zeroed."""
        return self._wiped

    @property
    def is_mlocked(self) -> bool:
        """Whether the memory page is locked (non-swappable)."""
        return self._mlocked

    def __len__(self) -> int:
        return self._length

    def __repr__(self) -> str:
        if self._wiped:
            return "<SecureBytes [WIPED]>"
        return "<SecureBytes [REDACTED]>"

    def __str__(self) -> str:
        return repr(self)

    def __bool__(self) -> bool:
        return not self._wiped and self._length > 0

    def __del__(self) -> None:
        self.wipe()

    def __enter__(self) -> "SecureBytes":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.wipe()

    # Prevent accidental exposure
    def __bytes__(self) -> bytes:
        raise TypeError(
            "Cannot implicitly convert SecureBytes to bytes. "
            "Use .as_bytes() explicitly."
        )

    def __hash__(self) -> int:
        raise TypeError("SecureBytes is not hashable (security)")

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SecureBytes):
            if self._wiped or other._wiped:
                return False
            # Constant-time comparison
            import hmac
            return hmac.compare_digest(self._buf, other._buf)
        return NotImplemented


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zero_bytearray(buf: bytearray) -> None:
    """Zero a bytearray in place (Python-level fallback)."""
    for i in range(len(buf)):
        buf[i] = 0


def wipe_bytes_object(data: bytes) -> None:
    """Best-effort zeroing of an immutable bytes object.

    This uses ctypes to overwrite the internal buffer of a bytes
    object.  This is inherently unsafe and CPython-specific, but
    it is the best we can do for immutable bytes objects that
    contained key material.

    Call this when you receive key bytes from an external source
    and want to zero them after copying into SecureBytes.
    """
    if not isinstance(data, bytes) or len(data) == 0:
        return
    if not _HAS_MEMSET:
        return
    try:
        # CPython bytes internal buffer offset
        # This is fragile and version-specific
        buf_addr = id(data) + sys.getsizeof(bytes()) - 1
        _libc.memset(buf_addr, 0, len(data))
    except Exception:
        pass  # Best effort only


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def secure_key_from_bytes(data: bytes) -> SecureBytes:
    """Create a SecureBytes from raw bytes and zero the original.

    This is the recommended way to wrap key material:
    it creates the SecureBytes and then attempts to zero
    the source bytes object.
    """
    secure = SecureBytes(data)
    wipe_bytes_object(data)
    return secure


def get_platform_info() -> dict[str, Any]:
    """Return platform memory protection capabilities."""
    return {
        "mlock_available": _HAS_MLOCK,
        "memset_available": _HAS_MEMSET,
        "libc_loaded": _libc is not None,
        "platform": sys.platform,
        "tracked_keys": sum(1 for r in _live_keys if r() is not None),
    }


# ---------------------------------------------------------------------------
# S131: Swap / Hibernation Protection
# ---------------------------------------------------------------------------

@dataclass
class SwapCheckResult:
    """Result of swap encryption check."""
    swap_enabled: bool = False
    encrypted: bool = False
    devices: list[dict[str, str]] = field(default_factory=list)
    safe: bool = True
    error: str = ""
    platform_supported: bool = True


def check_swap_encrypted() -> SwapCheckResult:
    """Check whether swap is enabled and, if so, whether it uses dm-crypt.

    This is Linux-specific (reads ``/proc/swaps`` and checks for
    dm-crypt backing devices).  On non-Linux platforms, returns a
    graceful no-op result.

    A system is considered *safe* if:
      - swap is disabled entirely, OR
      - all swap devices are backed by dm-crypt (LUKS / encrypted swap).

    Returns
    -------
    SwapCheckResult
        Structured result with per-device details.
    """
    result = SwapCheckResult()

    if sys.platform != "linux":
        result.platform_supported = False
        result.safe = True  # Cannot check, assume OK
        return result

    # Parse /proc/swaps
    proc_swaps = "/proc/swaps"
    if not os.path.isfile(proc_swaps):
        result.error = "/proc/swaps not found"
        return result

    try:
        with open(proc_swaps, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except PermissionError:
        result.error = "Permission denied reading /proc/swaps"
        return result

    # First line is the header; skip it
    swap_lines = [ln.strip() for ln in lines[1:] if ln.strip()]
    if not swap_lines:
        # No swap enabled
        result.swap_enabled = False
        result.safe = True
        return result

    result.swap_enabled = True
    all_encrypted = True

    for line in swap_lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        device = parts[0]
        swap_type = parts[1]
        device_info = {
            "device": device,
            "type": swap_type,
            "encrypted": False,
        }

        # Check if the device is backed by dm-crypt
        # dm-crypt devices live under /dev/dm-* or /dev/mapper/*
        is_dm = (
            device.startswith("/dev/dm-")
            or device.startswith("/dev/mapper/")
        )

        if is_dm:
            device_info["encrypted"] = True
        else:
            # Check if the device is a zram device (compressed RAM, no disk)
            if device.startswith("/dev/zram"):
                device_info["encrypted"] = True  # zram is RAM-only, safe
                device_info["type"] = "zram"
            else:
                all_encrypted = False

        result.devices.append(device_info)

    result.encrypted = all_encrypted
    result.safe = all_encrypted
    return result


def swap_startup_check() -> None:
    """Run at application startup: log a warning if swap is insecure.

    In Bulbe mode with ``require_encrypted_swap: true``, this raises
    a RuntimeError to prevent startup with unencrypted swap.
    """
    result = check_swap_encrypted()

    if not result.platform_supported:
        logger.debug("Swap check: non-Linux platform, skipping")
        return

    if not result.swap_enabled:
        logger.info("Swap check: no swap enabled (safe)")
        return

    if result.safe:
        logger.info(
            "Swap check: swap enabled with encryption (%d devices)",
            len(result.devices),
        )
        return

    # Swap is enabled without full encryption
    unencrypted = [
        d["device"] for d in result.devices if not d.get("encrypted")
    ]
    logger.critical(
        "SWAP SECURITY WARNING: Unencrypted swap detected on: %s. "
        "Sensitive data (keys, prompts) may be written to disk. "
        "Use encrypted swap, zram, or disable swap entirely.",
        ", ".join(unencrypted),
    )

    # Check if Bulbe mode requires encrypted swap
    try:
        import yaml as _yaml
        config_path = os.path.join(
            os.path.dirname(__file__), "config", "security.yaml"
        )
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as fh:
                data = _yaml.safe_load(fh) or {}
            hardening = data.get("hardening", {})
            require_encrypted = hardening.get("require_encrypted_swap", True)
        else:
            require_encrypted = True
    except Exception:
        require_encrypted = True

    # Check security mode
    try:
        from opti_oignon.security_mode import is_bulbe
        if is_bulbe() and require_encrypted:
            raise RuntimeError(
                "Bulbe mode requires encrypted swap. "
                f"Unencrypted devices: {', '.join(unencrypted)}. "
                "Either encrypt swap, use zram, or set "
                "hardening.require_encrypted_swap: false in security.yaml."
            )
    except ImportError:
        pass
