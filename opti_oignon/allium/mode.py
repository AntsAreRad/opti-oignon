"""The security mode as the componion reads it: Daily or Bulbe, fail-closed.

``live_mode()`` answers ``"daily"`` only when the platform's mode manager
says exactly ``"daily"``; anything else -- another word, a stray space, an
exception, a module that cannot be imported -- is ``"bulbe"``.

The manager keeps the mode in a per-process cache that only a transition in
the same process clears, so a long-running server would never see a mode
switched from the command line. Before each read, the mode file and the
lockfile are stat-ed; when either changed since the last read (or cannot be
named), the cache is invalidated and the manager reads again. Re-reading
only on a change keeps the cost of a key derivation off every action.

The store reads the mode once per public action and passes the value down.
"""

import os

checkpoint_before_apply = True

_seen = {"stamp": None}


def _stat(path):
    """``(mtime_ns, size, inode)`` of a file, or ``"absent"``."""
    try:
        st = os.stat(path)
    except (OSError, TypeError, ValueError):
        return "absent"
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def live_mode():
    """``"daily"`` or ``"bulbe"``; ``"bulbe"`` whenever the mode cannot be read exactly."""
    try:
        from opti_oignon import security_mode as sm

        paths = (getattr(sm, "_SECURITY_YAML", None), getattr(sm, "_LOCKFILE_PATH", None))
        stamp = tuple(_stat(p) for p in paths)
        if None in paths or stamp != _seen["stamp"]:
            sm.security_mode_manager.invalidate_cache()
            _seen["stamp"] = stamp
        mode = sm.security_mode_manager.get_current_mode()
    except Exception:  # noqa: BLE001 - a mode that cannot be read is Bulbe
        return "bulbe"
    return "daily" if mode == "daily" else "bulbe"
