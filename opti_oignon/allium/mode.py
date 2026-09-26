"""The security mode as the componion reads it: Daily or Bulbe, fail-closed.

``live_reading()`` answers ``"daily"`` or ``"bulbe"`` when the platform's
mode manager says exactly that word, and ``"unknown"`` for anything else --
another word, a stray space, an exception, a module that cannot be imported.
``live_mode()`` answers ``"daily"`` only for a reading of ``"daily"`` and
``"bulbe"`` otherwise: a mode that cannot be read takes Bulbe's rules. The
garden tells the two apart only to say which it was.

The platform's manager keeps the mode in a per-process cache that only a
transition in the same process clears, so a long-running server would never
see a mode switched from the command line. The garden reads through a
manager of its own: before each read, the mode file and the lockfile are
stat-ed, and when either changed since the last read (or cannot be named) a
new private ``SecurityModeManager`` is made, which reads the files again --
the same files, the same HMAC check, failing secure to Bulbe on a mismatch.
The platform's shared manager is never touched: a look never makes a server
re-read, let alone change, its own security mode. Re-reading only on a
change keeps the cost of a key derivation off every action. The stamp and
its manager are kept as one pair, stored in one assignment, so a thread
never pairs a new stamp with an old manager.

One write is the platform's own, and a look can cause it: a new private
manager that finds the two files disagreeing, or the lockfile's HMAC
failing, records it as tamper evidence -- one entry in the auth store's
audit log and one in the signed audit chain -- once per change of the
files, never once per reading. Nothing is written in the being's store.
The server's own manager keeps the mode it read until a transition in its
process, so a server whose files were changed from outside keeps the
policies of the mode it read, while its garden, reading the files, says the
mode they now give.

The store reads the mode once per public action and passes the value down.
"""

import os

checkpoint_before_apply = True

# ``(stamp of the two files, the private manager that read them)``, replaced when the stamp changes.
_seen = {"entry": None}


def _stat(path):
    """``(mtime_ns, size, inode)`` of a file, or ``"absent"``."""
    try:
        st = os.stat(path)
    except (OSError, TypeError, ValueError):
        return "absent"
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def live_reading():
    """``"daily"`` or ``"bulbe"`` as the manager says it exactly; ``"unknown"`` whenever it cannot be read so."""
    try:
        from opti_oignon import security_mode as sm

        paths = (getattr(sm, "_SECURITY_YAML", None), getattr(sm, "_LOCKFILE_PATH", None))
        stamp = tuple(_stat(p) for p in paths)
        entry = _seen["entry"]
        if None in paths or entry is None or entry[0] != stamp:
            entry = (stamp, sm.SecurityModeManager())
            _seen["entry"] = entry
        mode = entry[1].get_current_mode()
    except Exception:  # noqa: BLE001 - a mode that cannot be read is unknown
        return "unknown"
    if isinstance(mode, str) and mode in ("daily", "bulbe"):
        return mode
    return "unknown"


def live_mode():
    """``"daily"`` or ``"bulbe"``; ``"bulbe"`` whenever the mode cannot be read exactly."""
    return "daily" if live_reading() == "daily" else "bulbe"
