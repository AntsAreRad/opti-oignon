#!/usr/bin/env python3
"""The run journal: an opt-in record of the user's own agent runs, kept on this machine.

An agent run is a model choosing tool calls for a task the user typed. The
journal keeps, for each run whose task its caller vouches typed, the words
typed and every call the model emitted, in order, with what the gates made of
it and the trust the run stood in when the model chose it: a record for the
user's own study, made on this machine and read by no model. It is off until
the user turns it on (``config/run_journal.yaml``).

* KEPT. One record per run, ``"journal": 1``: the run's identifier (the one
  its write gate names its proposals by), its conversation, model, mode,
  network policy, settings and the tools it was offered; the words typed,
  whole up to ``max_value_chars``, and their SHA-256; when it started (UTC)
  and how long it took, its stop, its rounds and the verifier's verdict.
  Each call: its index, round and ``t_ms`` (from the run's start to when the
  call's round was done), its tool and the format it came in, its arguments
  as the model gave them and the dispatch resolved them (each string cut at
  ``max_value_chars``, the names cut listed, the SHA-256 taken over the
  whole), the gate's outcome (ran, denied, refused, halted, unrunnable,
  error, never dispatched, other) with its reason and whether a person was
  asked, and its trust (``agent.loop.CallTrust``). A subtask and the
  verifier that made a call are runs of their own, ``<id>-t<k>`` (the run's
  k-th subtask) and ``<id>-v``, naming their parent and its round, carrying
  the parent's typed words, which judge their calls; a subtask's rounds and
  stop are not reported by the loop, and are left empty.
* NEVER KEPT. A tool's output, a subtask's answer, the memory block, the
  skills block, the system prompt, the model's prose, the final and the
  verifier's texts, spill files, proposals, user identifiers. A call's
  arguments are the model's own, and may quote what it read.
* THE SWITCH. On only when the file's top level is a mapping whose
  ``enabled`` is the YAML boolean ``true``: a missing file or key, the string
  "true", 1, or a file that does not parse (said once per change of it) is
  off. Off, a run creates and writes nothing and is handed no observer.
  Turning it off is a pause: what is kept stays until its retention, and a
  run that was going when it was turned off is not written.
* RETENTION. One file per UTC day a run started on, ``YYYY-MM-DD.jsonl``,
  under ``run_journal/`` in the data directory; a record is kept less than
  ``retention_days`` days (a whole number from 1 to 30, 30 by default: the
  ceiling of the consent), by the day it started: a run whose day is already
  past it when the run ends -- with one day, a run that goes past midnight
  UTC -- is not written, and is counted. The days past it go at boot, every
  hour while the server runs (``tend``), at a run's start at most once an
  hour, and at each write, deletion, wipe and sealing, on or off. A day
  dated after tomorrow -- a clock that ran ahead -- goes too, and is counted:
  a clock set back past the newest day removes every day. A read only leaves
  out what a pass removes.
* SEALED. With encryption on in ``security.yaml`` (read strictly:
  ``encryption.encryption_toggle``; a file that cannot say is no permission
  to write in clear), or while the running manager holds an enabled key,
  each line is ``JS1:`` and the base64url of the record sealed by
  ``encryption.encrypt_bytes`` under the journal's own subkey of the master
  key (HMAC-SHA256 of ``oo-run-journal-v1``), a fresh nonce for each line.
  The key is the one that loads, or the one a passphrase set up in this
  process holds; the cipher is kept once a key loads. Encryption on with no
  key or no cipher writes nothing; off, a Daily run is written in plaintext
  and a Bulbe one not at all. Plaintext records kept from before are sealed
  at the next pass once encryption is on, and a plaintext line that is no
  record -- one a crash cut -- is removed; nothing is ever unsealed. A line
  sealed under a key that no longer loads is unreadable: counted, and
  removed with its day or by a wipe.
* ON DISK. Each pass holds the directory by its descriptor and takes every
  name relative to it, so a link swapped in for the directory is never
  followed; the directory must be the user's own. It is made 0700 and each
  file 0600 whatever the umask, and both are held to it at each write. No
  file is opened through a link, nor blocks on a pipe, and a run's records
  go out in one append, so a reader sees a run whole or not at all. A tail a
  crash cut short is skipped and counted, and the next append starts on a
  line of its own. A rewrite goes through a temp beside the file, synced,
  moved in place, the directory synced; a temp a crash left is swept. One
  process writes the journal: a second writer is not supported, and a
  reader, which only reads, may be one.
* BOUNDED, NEVER IN THE WAY. A run keeps at most ``max_calls_per_run``
  calls; a run with a record over ``max_record_bytes`` is not written, nor
  one that would take its day past ``max_day_bytes``. A journal that cannot
  record -- busy past half a second, a seal or a write that fails, a bound
  passed -- drops what it held and counts why, said once per reason per
  process (``status``); a run's result, events and tool results are never
  changed by it.
* PRIVACY. Deleting a conversation, here or from a paired device, removes
  its runs and their sub-runs, with every line a loaded key cannot open
  (``forget_conversation``); a user's wipe and the full conversation wipe
  remove every file (``wipe``). A run that was going when its conversation
  was deleted, or the journal wiped, is not written. A sealed line no key
  opens is never dropped for want of one: the deletion is kept in the
  ledger instead, synced -- the SHA-256 of the conversation and the day
  (UTC), never its identifier; the newest day kept when a clock that ran
  ahead filed later runs -- applied by the next pass that has the key, and
  reads skip that conversation's runs of that day and before until then;
  so is a deletion that failed or found the journal busy (kept under
  tomorrow). A day that cannot be rewritten for a deletion is removed whole
  when it held a record the deletion names. The journal is single-user: a
  wipe removes it whole. It is never exported nor backed up. A fact deleted
  from memory or a source withdrawn is not traced into it; a wipe is.
* READ. ``iter_runs`` yields the records ordered by their start, day by day,
  ``trace_lines`` the same runs as lines of the injection bench's trace
  format 1, and ``status`` numbers only: per day, the runs, sub-runs, calls
  and bytes kept, and the failures counted.
"""

from __future__ import annotations

import base64
import contextlib
import errno
import hashlib
import hmac
import json
import logging
import os
import stat
import threading
import time
import weakref
from collections.abc import Iterable, Iterator
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Every new module hardcodes this; it is never overridable.
checkpoint_before_apply = True

# The shape of a record, and of the lines ``trace_lines`` gives.
JOURNAL_FORMAT = 1
TRACE_FORMAT = 1
KINDS = ("run", "task", "verifier")

# A sealed line, and the label of its subkey (distinct from the field, the
# database and the attachment labels by construction).
SEALED = "JS1:"
SUBKEY_LABEL = b"oo-run-journal-v1"

CONFIG_FILE = Path(__file__).resolve().parent / "config" / "run_journal.yaml"
DIRNAME = "run_journal"
# The data directory the journal lives in; None reads the package's
# (``config.DATA_DIR``) when it is first needed.
DATA_DIR: Path | None = None

DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "retention_days": 30,
    "max_calls_per_run": 500,
    "max_value_chars": 16384,
    "max_record_bytes": 262144,
    "max_day_bytes": 8388608,
}
# The whole numbers each bound may take; any other value keeps its default.
BOUNDS: dict[str, tuple[int, int]] = {
    "retention_days": (1, 30),
    "max_calls_per_run": (1, 5000),
    "max_value_chars": (64, 65536),
    "max_record_bytes": (4096, 1048576),
    "max_day_bytes": (65536, 67108864),
}

# How long a run's write and a read wait for the journal; a deletion, a
# wipe and a pass of the server's own, the user's gestures among them, wait
# longer before they give up.
LOCK_WAIT_S = 0.5
GESTURE_WAIT_S = 10.0
# The server tends the journal this often, and a run's start prunes at most
# this often.
PRUNE_INTERVAL_S = 3600.0

_SUFFIX = ".jsonl"
_TEMP = ".tmp-"
_LEDGER_NAME = ".forgotten"
_HEX = frozenset("0123456789abcdef")
_SEALED_B = SEALED.encode("ascii")
_RUN = ("run", None)
_VERIFIER = ("verifier", None)

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_APPEND = os.O_RDWR | os.O_APPEND | os.O_CREAT | _NOFOLLOW | _NONBLOCK | _CLOEXEC
_LEDGER_APPEND = os.O_WRONLY | os.O_APPEND | os.O_CREAT | _NOFOLLOW | _NONBLOCK | _CLOEXEC
_CREATE = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _CLOEXEC
_READ = os.O_RDONLY | _NONBLOCK | _NOFOLLOW | _CLOEXEC
_HELD = os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC

# The gate's reasons, by the outcome a call is kept under; a call that ran is
# "ran" and one never dispatched "not_dispatched", whatever their reason.
_OUTCOMES = {
    "halted": "halted",
    "denied_by_human": "denied",
    "denied": "denied",
    "not_in_allowlist": "refused",
    "no_approval_channel": "refused",
    "not_offered": "refused",
    "not_permitted": "refused",
    "no_channel": "refused",
    "unendorsed": "refused",
    "unshowable": "refused",
    "sandbox_unavailable": "unrunnable",
    "no_executor": "unrunnable",
    "error": "error",
}

# What each counted failure means, said once per process.
_WHY = {
    "lock_timeout": "the journal was busy past its wait, and what it held was dropped",
    "seal_unavailable": "encryption is on, or cannot be read, and no key or cipher can seal a record: nothing was written",
    "seal_failed": "a record could not be sealed: nothing of its run was written",
    "plaintext_refused": "encryption is off and a Bulbe run is never kept in plaintext: nothing was written",
    "record_too_big": "a run with a record over max_record_bytes was not written",
    "day_full": "the day's file is at max_day_bytes: a run was dropped",
    "calls_capped": "a run passed max_calls_per_run: its later calls were dropped",
    "unsafe_path": "something other than the journal's own directory or file stood in its place: refused",
    "write_failed": "a run could not be written",
    "observe_failed": "a call could not be observed",
    "finish_failed": "a run could not be recorded",
    "torn": "a file ends on a line cut short: it was skipped",
    "unreadable": "a line could not be read (sealed under another key, or damaged): it was skipped",
    "read_failed": "a file could not be read",
    "prune_failed": "a day past its retention could not be removed",
    "future_day": "a day dated after tomorrow was removed: the clock had run ahead",
    "reseal_failed": "plaintext records could not be sealed",
    "forget_failed": "the runs of a deleted conversation could not all be removed: the ledger holds the deletion",
    "forget_dropped_day": "a day that could not be rewritten for a deletion was removed whole",
    "past_retention": "a run that started on a day already past the retention was not written",
    "wipe_failed": "the journal could not be wiped whole",
}

_LOCK = threading.Lock()
_COUNTS_LOCK = threading.Lock()
_FAILURES: dict[str, int] = {}
_SAID: set[str] = set()
_SETTINGS: dict[str, Any] = {"entry": None}
# ``plain``: plaintext may be on disk (unknown until a pass has looked);
# ``prune_at``: when a run's start prunes next; ``wiped_at``: when the journal
# was last wiped (monotonic seconds); ``cipher``: the cipher once a key loaded.
_STATE: dict[str, Any] = {"plain": True, "prune_at": 0.0, "wiped_at": -1.0, "cipher": None}
# Conversation -> when it was deleted here (monotonic): a run going then is
# never written. Identifiers only, in memory only, and only while a run of
# that conversation that started before it is still open (``_OPEN``).
_FORGOTTEN: dict[str, float] = {}
# The runs open now: a deletion stamps its conversation only while one of
# them is of it, and a run's end drops the stamps no open run started before.
_OPEN: weakref.WeakSet = weakref.WeakSet()
# The deletions left to apply, held with the ledger on disk: the SHA-256 of
# the conversation -> the day it was deleted (UTC), never its identifier.
_LEDGER: dict[str, str] = {}
# The day a deletion is kept under when its day cannot be read: the last day
# there is, so a damaged entry hides every run of its conversation, never none.
_NO_DAY = "9999-12-31"
_UNPARSED = object()
_INFINITY = float("inf")


class _Unsafe(OSError):
    """Something other than the journal's own directory or file stands where it would be."""


def _now() -> datetime:
    """The time now in UTC: a run's start, and the day the retention counts back from."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Failures: counted by reason, said once per reason
# ---------------------------------------------------------------------------
def _failed(reason: str) -> None:
    with _COUNTS_LOCK:
        _FAILURES[reason] = _FAILURES.get(reason, 0) + 1
        first = reason not in _SAID
        _SAID.add(reason)
    if first:
        logger.warning("run journal: %s (%s); counted in run_journal.status(), said once",
                       _WHY.get(reason, reason), reason)


def _reason(exc: BaseException, default: str) -> str:
    """A failure's reason: a link, a pipe or a stranger in the journal's place is unsafe, anything else ``default``."""
    if isinstance(exc, _Unsafe) or getattr(exc, "errno", None) in (errno.ELOOP, errno.ENXIO):
        return "unsafe_path"
    return default


def failures() -> dict[str, int]:
    """Every failure counted since the process began, by reason."""
    with _COUNTS_LOCK:
        return dict(sorted(_FAILURES.items()))


def reset() -> None:
    """The journal's state as a new process finds it: no count, no cached settings or cipher, no stamp, a prune
    due; the ledger on disk is read again by the next pass."""
    with _COUNTS_LOCK:
        _FAILURES.clear()
        _SAID.clear()
        _LEDGER.clear()
        _FORGOTTEN.clear()
        _OPEN.clear()
    _SETTINGS["entry"] = None
    _STATE.update(plain=True, prune_at=0.0, wiped_at=-1.0, cipher=None)


# ---------------------------------------------------------------------------
# Settings: the switch and the bounds
# ---------------------------------------------------------------------------
def settings(path: str | Path | None = None) -> dict[str, Any]:
    """The journal's settings, from ``path`` or ``config/run_journal.yaml``.

    ``state`` is "on", "off" or "unreadable" and ``enabled`` is exactly
    ``state == "on"``: on only for a top-level mapping whose ``enabled`` is
    the YAML boolean true; a missing file or key and any other value are off;
    a file that does not parse, or whose top level is not a mapping, is
    unreadable, and so off, said once per change of the file. Each bound is
    a whole number within ``BOUNDS``; any other value present is refused by
    name and its default kept. Cached until the file changes (its
    modification time, size and inode), so a run's start costs one stat.
    """
    file = Path(path) if path is not None else Path(CONFIG_FILE)
    try:
        st = file.stat()
        key: tuple = (str(file), st.st_mtime_ns, st.st_size, st.st_ino)
    except OSError:
        key = (str(file), None)
    entry = _SETTINGS["entry"]
    if entry is not None and entry[0] == key:
        return dict(entry[1])
    value = _read_settings(file) if key[1] is not None else dict(DEFAULTS, state="off")
    _SETTINGS["entry"] = (key, value)
    return dict(value)


def _read_settings(file: Path) -> dict[str, Any]:
    value = dict(DEFAULTS, state="off")
    try:
        import yaml

        data = yaml.safe_load(file.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a switch that cannot be read is off, and said to be unreadable
        data = _UNPARSED
    if data is _UNPARSED or not isinstance(data, dict):
        logger.warning("run journal: %s cannot be read: the journal is off", file.name)
        value["state"] = "unreadable"
        return value
    value["enabled"] = data.get("enabled") is True
    value["state"] = "on" if value["enabled"] else "off"
    for name, (low, high) in BOUNDS.items():
        if name not in data:
            continue
        given = data[name]
        if type(given) is int and low <= given <= high:
            value[name] = given
        else:
            logger.warning("run journal: %s %r is not a whole number from %d to %d; %d kept", name, given, low,
                           high, DEFAULTS[name])
    return value


def enabled() -> bool:
    """Whether the switch is on now."""
    return bool(settings()["enabled"])


def _cutoff(retention: int) -> date:
    """The oldest day kept: a record is kept less than ``retention`` days, counted back from today (UTC)."""
    return _now().date() - timedelta(days=retention - 1)


# ---------------------------------------------------------------------------
# The directory, held by its descriptor, and its files
# ---------------------------------------------------------------------------
def _base() -> Path:
    base = DATA_DIR
    if base is None:
        from opti_oignon import config

        base = config.DATA_DIR
    return Path(base).resolve()


@contextlib.contextmanager
def _opened(make: bool = False) -> Iterator[tuple[int, int] | None]:
    """The journal's directory for one pass, held by its descriptor beside the data directory's: ``(directory,
    data directory)``, or None when it is not there.

    Every name a pass reads, writes, replaces or removes is taken relative to
    it, so a link swapped in for the directory once it is held is never
    followed. ``make`` makes it, 0700, and holds it to 0700: a write. A link,
    a file or a directory another user owns in its place is ``_Unsafe``.
    """
    try:
        parent = os.open(_base(), os.O_RDONLY | _DIRECTORY | _CLOEXEC)
    except FileNotFoundError:
        if make:
            raise
        yield None
        return
    try:
        if make:
            with contextlib.suppress(FileExistsError):
                os.mkdir(DIRNAME, 0o700, dir_fd=parent)
        try:
            fd = os.open(DIRNAME, _HELD, dir_fd=parent)
        except FileNotFoundError:
            fd = -1
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                raise _Unsafe(f"{DIRNAME} is not the journal's own directory") from exc
            raise
        if fd < 0:
            yield None
            return
        try:
            st = os.fstat(fd)
            if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
                raise _Unsafe(f"{DIRNAME} is not the journal's own directory")
            if make and stat.S_IMODE(st.st_mode) != 0o700:
                os.fchmod(fd, 0o700)
            yield fd, parent
        finally:
            os.close(fd)
    finally:
        os.close(parent)


def _day_of(name: str) -> date | None:
    """The day a file name holds, ``YYYY-MM-DD.jsonl`` exactly; None for any other name."""
    if not name.endswith(_SUFFIX):
        return None
    try:
        day = date.fromisoformat(name[: -len(_SUFFIX)])
    except ValueError:
        return None
    return day if day.isoformat() + _SUFFIX == name else None


def _is_temp(name: str) -> bool:
    """Whether a name is a rewrite's temp: a dot, a day's file or the ledger, ``.tmp-`` and eight hex digits."""
    head, mark, tail = name.partition(_TEMP)
    return (name.startswith(".") and bool(mark) and (_day_of(head[1:]) is not None or head[1:] == _LEDGER_NAME)
            and len(tail) == 8 and set(tail) <= _HEX)


def _names(fd: int) -> list[str]:
    with os.scandir(fd) as entries:
        return sorted(entry.name for entry in entries)


def _days(fd: int) -> list[tuple[date, str]]:
    """The journal's day files, oldest first; any other file is left alone."""
    return [(day, name) for name in _names(fd) if (day := _day_of(name)) is not None]


def _read(name: str, fd: int) -> bytes:
    handle = os.open(name, _READ, dir_fd=fd)
    try:
        if not stat.S_ISREG(os.fstat(handle).st_mode):
            raise _Unsafe(f"{name} is not a file of the journal's own")
        chunks = []
        while chunk := os.read(handle, 1 << 20):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(handle)


def _split(data: bytes) -> tuple[list[bytes], bool]:
    """A file's whole lines, and whether it ends on a line cut short."""
    lines = data.split(b"\n")
    tail = lines.pop()
    return [line for line in lines if line], bool(tail)


def _write_all(handle: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(handle, view):]


def _owned(handle: int, name: str) -> os.stat_result:
    """A file the journal opened: its own and regular, held to 0600."""
    st = os.fstat(handle)
    if not stat.S_ISREG(st.st_mode):
        raise _Unsafe(f"{name} is not a file of the journal's own")
    if stat.S_IMODE(st.st_mode) != 0o600:
        os.fchmod(handle, 0o600)
    return st


def _append(name: str, data: bytes, limit: int, fd: int) -> str:
    """Append ``data`` to a day's file, whole, never through a link: "" when written, else why not.

    A file that ends on a line cut short gets a line break first, so the cut
    line stays a line of its own; data that would take the file past
    ``limit`` is not written; a write that fails part-way is cut back to the
    file's length before it.
    """
    handle = os.open(name, _APPEND, 0o600, dir_fd=fd)
    try:
        try:
            size = _owned(handle, name).st_size
        except _Unsafe:
            return "unsafe_path"
        if size and os.pread(handle, 1, size - 1) != b"\n":
            data = b"\n" + data
        if size + len(data) > limit:
            return "day_full"
        try:
            _write_all(handle, data)
        except OSError:
            with contextlib.suppress(OSError):
                os.ftruncate(handle, size)
            raise
    finally:
        os.close(handle)
    return ""


def _rewrite(name: str, lines: list[bytes], fd: int) -> None:
    """Replace a file by ``lines``: a temp beside it, 0600, synced, moved in place, the directory synced.

    No line left removes the file.
    """
    if not lines:
        os.unlink(name, dir_fd=fd)
        os.fsync(fd)
        return
    temp = f".{name}{_TEMP}{os.urandom(4).hex()}"
    handle = os.open(temp, _CREATE, 0o600, dir_fd=fd)
    try:
        try:
            _write_all(handle, b"".join(line + b"\n" for line in lines))
            os.fsync(handle)
        finally:
            os.close(handle)
        os.replace(temp, name, src_dir_fd=fd, dst_dir_fd=fd)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp, dir_fd=fd)
        raise
    os.fsync(fd)


@contextlib.contextmanager
def _held(wait: float) -> Iterator[bool]:
    """The journal's lock, waited for ``wait`` seconds: yields whether it was taken."""
    taken = _LOCK.acquire(timeout=wait)
    try:
        yield taken
    finally:
        if taken:
            _LOCK.release()


# ---------------------------------------------------------------------------
# Sealing
# ---------------------------------------------------------------------------
class _Cipher:
    """The journal's lines under its subkey: sealed by the encryption module's own primitives."""

    def __init__(self, encryption: Any, subkey: bytes) -> None:
        self._encryption = encryption
        self._subkey = subkey

    def seal(self, text: str) -> str:
        sealed = self._encryption.encrypt_bytes(self._subkey, text.encode("utf-8"))
        return SEALED + base64.urlsafe_b64encode(sealed).decode("ascii")

    def open(self, line: bytes) -> bytes:
        return self._encryption.decrypt_bytes(self._subkey, base64.urlsafe_b64decode(line[len(SEALED):]))


def _cipher() -> _Cipher | None:
    """The journal's cipher, kept once a key loads, whatever the toggle says; None while none does.

    The key is the one that loads (``get_encryption_key``), or else the one a
    passphrase set up in this process holds: its key file opens only with the
    passphrase, which the running manager does not keep.
    """
    cached = _STATE["cipher"]
    if cached is not None:
        return cached
    try:
        from opti_oignon import encryption

        master = encryption.get_encryption_key()
        if master is None:  # set up from a passphrase in this process: its key file opens only with it
            manager = encryption.get_encryption_manager()
            master = manager._raw_key() if manager.enabled else None
    except Exception:  # noqa: BLE001 - no module, no manager or no key: nothing seals and nothing sealed opens
        return None
    if master is None:
        return None
    raw = master.as_bytes() if hasattr(master, "as_bytes") else bytes(master)
    cipher = _Cipher(encryption, hmac.new(raw, SUBKEY_LABEL, hashlib.sha256).digest())
    _STATE["cipher"] = cipher
    return cipher


def _toggle() -> bool | None:
    """Whether encryption is on: while the running manager holds an enabled key, or as ``security.yaml`` says it,
    read strictly (``encryption.encryption_toggle``); None when it cannot be told -- never read as off."""
    try:
        from opti_oignon import encryption
    except Exception:  # noqa: BLE001 - a toggle that cannot be read is no permission to write in plaintext
        return None
    try:
        if encryption.get_encryption_manager().enabled:
            return True
    except Exception:  # noqa: BLE001 - no manager to ask: the file decides
        logger.debug("run journal: the encryption manager cannot be asked", exc_info=True)
    try:
        return encryption.encryption_toggle()
    except Exception:  # noqa: BLE001 - unknown
        return None


def _machine_mode() -> str:
    """The machine's security mode now, Bulbe when it cannot be read."""
    try:
        from opti_oignon.agent import allowlists

        return allowlists.current_mode()
    except Exception:  # noqa: BLE001 - fail-secure
        return "bulbe"


def _sealing(mode: str | None) -> tuple[_Cipher | None, str]:
    """How a record is written now: ``(cipher, "")`` sealed, ``(None, "")`` in plaintext, ``(None, why)`` not.

    Encryption on: sealed, or nothing when no key loads. Off: plaintext for a
    Daily run on a Daily machine, nothing otherwise. A toggle that cannot be
    told writes nothing.
    """
    toggle = _toggle()
    if toggle is None:
        return None, "seal_unavailable"
    if toggle:
        cipher = _cipher()
        return (cipher, "") if cipher is not None else (None, "seal_unavailable")
    if mode == "daily" and _machine_mode() == "daily":
        return None, ""
    return None, "plaintext_refused"


def _decode(line: bytes, cipher: _Cipher | None) -> dict[str, Any] | None:
    """The record a line holds, sealed or plain; None when it cannot be read as one."""
    try:
        if line.startswith(_SEALED_B):
            if cipher is None:
                return None
            record = json.loads(cipher.open(line))
        elif line.startswith(b"{"):
            record = json.loads(line)
        else:
            return None
    except Exception:  # noqa: BLE001 - another key, or damage: unreadable, never raised
        return None
    return record if isinstance(record, dict) and record.get("journal") == JOURNAL_FORMAT else None


# ---------------------------------------------------------------------------
# The ledger of deletions left to apply
# ---------------------------------------------------------------------------
def _hashed(conversation_id: str) -> str:
    return hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()


def _ledger_on_disk(fd: int) -> dict[str, str]:
    try:
        data = _read(_LEDGER_NAME, fd)
    except FileNotFoundError:
        return {}
    entries: dict[str, str] = {}
    for line in data.split(b"\n"):
        parts = line.decode("ascii", "replace").split()
        if parts and len(parts[0]) == 64 and set(parts[0]) <= _HEX:
            entries[parts[0]] = max(entries.get(parts[0], ""), _day_or_last(parts[1] if len(parts) > 1 else ""))
    return entries


def _day_or_last(text: str) -> str:
    """A ledger entry's day, or the last day there is when it cannot be read: a deletion whose day is lost
    hides every run of its conversation, never none."""
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        return _NO_DAY


def _with_memory(disk: dict[str, str]) -> dict[str, str]:
    """The deletions of the ledger on disk with this process's own."""
    entries = dict(disk)
    with _COUNTS_LOCK:
        for key, at in _LEDGER.items():
            entries[key] = max(entries.get(key, ""), at)
    return entries


def _forgotten(fd: int | None) -> dict[str, str]:
    """Every deletion left to apply, the ledger's on disk and this process's: SHA-256 -> the day it was deleted."""
    return _with_memory(_ledger_on_disk(fd) if fd is not None else {})


def _hidden(record: dict[str, Any], entries: dict[str, str]) -> bool:
    """Whether a deletion left to apply names a record: its conversation, deleted on the day the record started
    or later (the ledger keeps the day, never the time)."""
    at = entries.get(_hashed(str(record.get("conversation_id") or "")))
    return at is not None and str(record.get("started_at") or "")[:10] <= at[:10]


def _deleted_on(fd: int | None) -> str:
    """The day a deletion is kept under: today, or the newest day kept when a clock that ran ahead filed runs
    later -- tomorrow at most, as a pass keeps no later day; tomorrow when the journal could not be listed."""
    today = _now().date()
    ahead = today + timedelta(days=1)
    if fd is None:
        return ahead.isoformat()
    try:
        newest = max((day for day, _name in _days(fd)), default=today)
    except OSError:
        return ahead.isoformat()
    return min(max(newest, today), ahead).isoformat()


def _remember(conversation_id: str, fd: int | None = None) -> None:
    """Keep a deletion to apply later: in memory, and in the ledger on disk, 0600, never through a link.

    The ledger holds the SHA-256 of the conversation and the day it was
    deleted (UTC, ``_deleted_on``), never its identifier. Without the
    journal held (it was busy), the ledger is written beside the pass that
    holds it; this process's copy is the one that counts until a pass
    applies it or writes it down.
    """
    key, at = _hashed(conversation_id), _deleted_on(fd)
    with _COUNTS_LOCK:
        _LEDGER[key] = max(_LEDGER.get(key, ""), at)
    try:
        if fd is not None:
            _note_forgotten(fd, key, at)
            return
        with _opened() as held:
            if held is not None:
                _note_forgotten(held[0], key, at)
    except Exception as exc:  # noqa: BLE001 - counted; this process still holds it
        _failed(_reason(exc, "forget_failed"))


def _note_forgotten(fd: int, key: str, at: str) -> None:
    """Append a deletion to the ledger, synced; a ledger made here is synced into the directory too."""
    try:
        handle, made = os.open(_LEDGER_NAME, _LEDGER_APPEND | os.O_EXCL, 0o600, dir_fd=fd), True
    except FileExistsError:
        handle, made = os.open(_LEDGER_NAME, _LEDGER_APPEND, 0o600, dir_fd=fd), False
    try:
        _owned(handle, _LEDGER_NAME)
        _write_all(handle, f"{key} {at}\n".encode("ascii"))
        os.fsync(handle)
    finally:
        os.close(handle)
    if made:
        os.fsync(fd)


def _ledger_lines(entries: dict[str, str]) -> list[bytes]:
    return [f"{key} {at}".encode("ascii") for key, at in sorted(entries.items())]


def _clear_forgotten(fd: int, applied: dict[str, str]) -> None:
    """Forget the deletions a pass applied: this process's, unless deleted again on a later day, and the ledger on
    disk."""
    with _COUNTS_LOCK:
        for key, at in applied.items():
            if _LEDGER.get(key, "") <= at:
                _LEDGER.pop(key, None)
    with contextlib.suppress(FileNotFoundError):
        os.unlink(_LEDGER_NAME, dir_fd=fd)
        os.fsync(fd)


def _expire_forgotten(fd: int, cutoff: date) -> None:
    """Drop the deletions older than the oldest day kept: every record they name has gone with its day."""
    limit = cutoff.isoformat()
    with _COUNTS_LOCK:
        for key in [key for key, at in _LEDGER.items() if at[:10] < limit]:
            del _LEDGER[key]
    entries = _ledger_on_disk(fd)
    kept = {key: at for key, at in entries.items() if at[:10] >= limit}
    if len(kept) != len(entries):
        _rewrite(_LEDGER_NAME, _ledger_lines(kept), fd)


def _dropped_day(name: str, fd: int) -> bool:
    """Remove a day that could not be rewritten for a deletion, so the deletion wins; whether it went."""
    try:
        os.unlink(name, dir_fd=fd)
        os.fsync(fd)
    except FileNotFoundError:
        return True
    except OSError as exc:
        _failed(_reason(exc, "forget_failed"))
        return False
    _failed("forget_dropped_day")
    return True


def _apply_forgotten(fd: int) -> None:
    """Apply the deletions left to apply: each conversation's records started on the day of its deletion or
    before go, with every line a loaded key cannot open. A sealed line no key opens is kept, and the deletions
    with it, for the next pass with the key, the ledger on disk then holding this process's own too; a pass
    that could read every line clears them. A day that cannot be rewritten goes whole when it held a record a
    deletion names, and is kept otherwise (counted)."""
    disk = _ledger_on_disk(fd)
    entries = _with_memory(disk)
    if not entries:
        return
    cipher = _cipher()
    complete = True
    for _day, name in _days(fd):
        try:
            lines, torn = _split(_read(name, fd))
        except OSError as exc:
            _failed(_reason(exc, "forget_failed"))
            complete = False
            continue
        kept, changed, named = [], torn, False
        for line in lines:
            if cipher is None and line.startswith(_SEALED_B):
                kept.append(line)
                complete = False
                continue
            record = _decode(line, cipher)
            if record is None:
                changed = True
                continue
            if _hidden(record, entries):
                changed = named = True
                continue
            kept.append(line)
        if changed:
            try:
                _rewrite(name, kept, fd)
            except OSError as exc:
                if named:
                    complete = _dropped_day(name, fd) and complete
                else:
                    _failed(_reason(exc, "forget_failed"))
    if complete:
        _clear_forgotten(fd, entries)
    elif entries != disk:
        _rewrite(_LEDGER_NAME, _ledger_lines(entries), fd)


# ---------------------------------------------------------------------------
# Passes: what every write, deletion, wipe, sealing and tending does first
# ---------------------------------------------------------------------------
def _prune_in(fd: int, retention: int) -> int:
    """Remove the days older than the retention, the days after tomorrow, and every rewrite's temp left behind;
    each name refused alone, the others removed. The deletions older than the oldest day kept go with them."""
    today = _now().date()
    cutoff, ahead = _cutoff(retention), today + timedelta(days=1)
    removed = 0
    for name in _names(fd):
        day = _day_of(name)
        if not ((day is not None and (day < cutoff or day > ahead)) or _is_temp(name)):
            continue
        try:
            os.unlink(name, dir_fd=fd)
            removed += 1
        except FileNotFoundError:
            continue
        except OSError as exc:
            _failed(_reason(exc, "prune_failed"))
            continue
        if day is not None and day > ahead:
            _failed("future_day")
    if removed:
        os.fsync(fd)
    _expire_forgotten(fd, cutoff)
    return removed


def _reseal_in(fd: int) -> int:
    """Seal every plaintext record and drop every plaintext line that is no record, a cut one included, once
    encryption is on; the records sealed. -1 when no key loads: ``plain`` stays set, and the next pass tries again."""
    if not _STATE["plain"] or not _toggle():
        return 0
    cipher = _cipher()
    if cipher is None:
        return -1
    sealed, refused = 0, False
    for _day, name in _days(fd):
        try:
            lines, torn = _split(_read(name, fd))
        except OSError as exc:
            _failed(_reason(exc, "reseal_failed"))
            refused = True
            continue
        kept, changed = [], torn
        for line in lines:
            if line.startswith(_SEALED_B):
                kept.append(line)
            elif _decode(line, None) is not None:
                kept.append(cipher.seal(line.decode("utf-8")).encode("ascii"))
                changed, sealed = True, sealed + 1
            else:
                changed = True  # plaintext that is no record: never kept in clear once sealing is on
        if changed:
            try:
                _rewrite(name, kept, fd)
            except OSError as exc:
                _failed(_reason(exc, "reseal_failed"))
                refused = True
    if not refused:
        _STATE["plain"] = False
    return sealed


def _tend(fd: int) -> None:
    """Prune, apply the deletions left to apply, then seal what was kept in plaintext; a failure of each is counted
    and the pass goes on."""
    try:
        _prune_in(fd, settings()["retention_days"])
    except Exception as exc:  # noqa: BLE001 - counted
        _failed(_reason(exc, "prune_failed"))
    try:
        _apply_forgotten(fd)
    except Exception as exc:  # noqa: BLE001 - counted
        _failed(_reason(exc, "forget_failed"))
    try:
        _reseal_in(fd)
    except Exception as exc:  # noqa: BLE001 - counted
        _failed(_reason(exc, "reseal_failed"))


def tend() -> None:
    """Tend the journal now: the days past the retention go, the deletions left to apply are applied, plaintext
    is sealed once encryption is on. The boot's pass and the server's hourly one. Never raises."""
    try:
        with _held(GESTURE_WAIT_S) as taken:
            if not taken:
                _failed("lock_timeout")
                return
            with _opened() as held:
                if held is not None:
                    _tend(held[0])
    except Exception as exc:  # noqa: BLE001 - counted
        _failed(_reason(exc, "prune_failed"))


def prune() -> int:
    """Remove the days past the retention and the temps a rewrite left; the files removed. Never raises."""
    try:
        with _held(LOCK_WAIT_S) as taken:
            if not taken:
                _failed("lock_timeout")
                return 0
            with _opened() as held:
                return 0 if held is None else _prune_in(held[0], settings()["retention_days"])
    except Exception as exc:  # noqa: BLE001 - counted
        _failed(_reason(exc, "prune_failed"))
        return 0


def prune_if_due() -> None:
    """``prune`` when the last one is more than an hour old; until then, no input or output at all."""
    now = time.monotonic()
    if now < _STATE["prune_at"]:
        return
    _STATE["prune_at"] = now + PRUNE_INTERVAL_S
    prune()


# ---------------------------------------------------------------------------
# Recording a run
# ---------------------------------------------------------------------------
def _gate(result: Any) -> dict[str, Any]:
    """What the gates made of a call: its outcome, reason, whether a person was asked, whether it ran."""
    if result is None:
        return {"outcome": "not_dispatched", "reason": None, "asked": False, "executed": False}
    executed = bool(getattr(result, "executed", False))
    reason = str(getattr(result, "reason", "") or "")
    provenance = getattr(result, "provenance", None) or {}
    return {"outcome": "ran" if executed else _OUTCOMES.get(reason, "other"), "reason": reason,
            "asked": bool(provenance.get("asked", False)), "executed": executed}


def _clip(value: Any, limit: int) -> tuple[Any, bool]:
    """``value`` with each string in it cut at ``limit`` characters, and whether one was."""
    if isinstance(value, str):
        return (value[:limit], True) if len(value) > limit else (value, False)
    if isinstance(value, dict):
        out, cut = {}, False
        for key, item in value.items():
            out[key], was = _clip(item, limit)
            cut = cut or was
        return out, cut
    if isinstance(value, (list, tuple)):
        items, cut = [], False
        for item in value:
            kept, was = _clip(item, limit)
            items.append(kept)
            cut = cut or was
        return items, cut
    return value, False


def _digest(arguments: Any) -> str:
    """The SHA-256 of the arguments' canonical JSON: keys sorted, "," and ":", any other value as its text."""
    try:
        text = json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        text = json.dumps(json.loads(json.dumps(arguments, default=str)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strict(value: Any) -> Any:
    """``value`` as JSON holds it: each number JSON cannot hold written as its name, each key as text."""
    if isinstance(value, float) and (value != value or value in (_INFINITY, -_INFINITY)):
        return repr(value)
    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    return value


class Run:
    """One run being recorded: its fields from its start, each call as the loop observes it, written once.

    ``on_call`` is the loop's observer (``agent.loop.run(on_call=...)``): it
    keeps the call, its gate and the trust ``trust`` answers for it, and
    keeps nothing more past ``max_calls_per_run``. ``finish`` writes the run
    and its sub-runs in one append once the loop has returned (``None`` for a
    run whose thread failed), unless the run's conversation was deleted or
    the journal wiped while it went. Neither ever raises.
    """

    def __init__(self, *, run_id: str, typed: str, trust: Any, conversation_id: str, model: str, mode: str,
                 network_policy: str | None, run_settings: dict[str, Any], offered: Iterable[str],
                 bounds: dict[str, Any]) -> None:
        self.run_id = str(run_id)
        self._trust = trust
        self._mode = str(mode or "")
        self._bounds = dict(bounds)
        self._started = _now()
        self._clock = time.monotonic()
        limit = int(self._bounds["max_value_chars"])
        words = str(typed or "")
        self._common = {
            "journal": JOURNAL_FORMAT,
            "surface": "agent",
            "started_at": self._started.isoformat(timespec="milliseconds"),
            "typed": words[:limit],
            "typed_clipped": len(words) > limit,
            "typed_sha256": hashlib.sha256(words.encode("utf-8")).hexdigest(),
            "conversation_id": str(conversation_id or ""),
            "model": str(model or ""),
            "mode": self._mode,
            "network_policy": network_policy,
            "settings": dict(run_settings or {}),
        }
        self._offered = sorted(str(name) for name in offered)
        self._calls: dict[tuple, list[tuple]] = {}
        self._dropped: dict[tuple, int] = {}
        self._parents: dict[Any, int] = {}
        self._kept = 0
        self._done = False
        with _COUNTS_LOCK:
            _OPEN.add(self)

    def on_call(self, event: dict[str, Any]) -> None:
        """Keep one observed call with its gate and its trust; past the bound, count it dropped. Never raises."""
        try:
            trust = self._trust.observe(event) if self._trust is not None else None
            scope = event.get("scope")
            if scope == "task":
                key: tuple = ("task", int(event.get("subrun") or 0))
                self._parents.setdefault(key[1], int(event.get("parent_round") or 0))
            else:
                key = _VERIFIER if scope == "verifier" else _RUN
            if self._kept >= int(self._bounds["max_calls_per_run"]):
                self._dropped[key] = self._dropped.get(key, 0) + 1
                _failed("calls_capped")
                return
            call, result = event.get("call"), event.get("result")
            self._kept += 1
            self._calls.setdefault(key, []).append((
                int(event.get("round") or 0), int((time.monotonic() - self._clock) * 1000),
                str(getattr(call, "name", "") or ""), str(getattr(call, "source", "") or ""),
                dict(getattr(call, "arguments", None) or {}), _gate(result), trust))
        except Exception:  # noqa: BLE001 - an observer never breaks the run; counted
            _failed("observe_failed")

    def finish(self, result: Any) -> bool:
        """Write the run and its sub-runs, once; whether they were written. Never raises."""
        if self._done:
            return False
        self._done = True
        written = self._recorded(result)
        _closed(self)
        return written

    def _recorded(self, result: Any) -> bool:
        try:
            return self._write(result)
        except Exception:  # noqa: BLE001 - a journal never fails a run; counted
            _failed("finish_failed")
            return False

    def _write(self, result: Any) -> bool:
        if not enabled():
            return False  # turned off while the run went: a pause covers the run it found going
        if self._started.date() < _cutoff(settings()["retention_days"]):
            _failed("past_retention")
            return False  # its day is past the retention: the next pass would remove it
        cipher, refusal = _sealing(self._mode)
        if refusal:
            _failed(refusal)
            return False
        lines = []
        for record in self._records(result):
            text = json.dumps(_strict(record), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                              allow_nan=False, default=str)
            if len(text.encode("utf-8")) > int(self._bounds["max_record_bytes"]):
                _failed("record_too_big")
                return False  # a run is written whole or not at all
            if cipher is not None:
                try:
                    text = cipher.seal(text)
                except Exception:  # noqa: BLE001 - never a plaintext line in its place
                    _failed("seal_failed")
                    return False
            lines.append(text)
        data = "".join(line + "\n" for line in lines).encode("utf-8")
        with _held(LOCK_WAIT_S) as taken:
            if not taken:
                _failed("lock_timeout")
                return False
            if self._clock <= max(_FORGOTTEN.get(self._common["conversation_id"], -1.0), _STATE["wiped_at"]):
                return False  # deleted or wiped while it went: nothing of it outlives the gesture
            try:
                with _opened(make=True) as held:
                    _tend(held[0])
                    refusal = _append(self._started.date().isoformat() + _SUFFIX, data,
                                      int(self._bounds["max_day_bytes"]), held[0])
            except Exception as exc:  # noqa: BLE001 - counted
                _failed(_reason(exc, "write_failed"))
                return False
            if refusal:
                _failed(refusal)
                return False
            if cipher is None:
                _STATE["plain"] = True
        return True

    def _records(self, result: Any) -> list[dict[str, Any]]:
        keys = set(self._calls) | set(self._dropped)
        last = {key: max((item[0] for item in self._calls.get(key, ())), default=0) for key in keys}
        if result is None:
            stop, rounds, verifier = "error", last.get(_RUN, 0), None
        else:
            stop = str(getattr(result, "stop_reason", "") or "")
            rounds = int(getattr(result, "rounds", 0) or 0)
            verifier = getattr(result, "verifier", None)
        verdict = getattr(verifier, "verdict", None) if verifier is not None else None
        records = [self._record(_RUN, run_id=self.run_id, kind="run", parent_run_id=None, parent_round=None,
                                offered=self._offered, stop_reason=stop, rounds=rounds, verdict=verdict,
                                duration_ms=int((time.monotonic() - self._clock) * 1000))]
        for key in sorted(key for key in keys if key[0] == "task"):
            records.append(self._record(key, run_id=f"{self.run_id}-t{key[1]}", kind="task",
                                        parent_run_id=self.run_id, parent_round=self._parents.get(key[1]),
                                        offered=None, stop_reason=None, rounds=None, verdict=None,
                                        duration_ms=None))
        if _VERIFIER in keys:
            records.append(self._record(_VERIFIER, run_id=f"{self.run_id}-v", kind="verifier",
                                        parent_run_id=self.run_id, parent_round=rounds, offered=None,
                                        stop_reason=None,
                                        rounds=int(getattr(verifier, "rounds", 0) or 0) or last[_VERIFIER],
                                        verdict=verdict, duration_ms=None))
        return records

    def _record(self, key: tuple, **fields: Any) -> dict[str, Any]:
        limit = int(self._bounds["max_value_chars"])
        calls = []
        for index, (rnd, t_ms, tool, source, arguments, gate, trust) in enumerate(self._calls.get(key, ())):
            kept, clipped = {}, []
            for name, value in arguments.items():
                kept[name], cut = _clip(value, limit)
                if cut:
                    clipped.append(str(name))
            calls.append({"i": index, "round": rnd, "t_ms": t_ms, "tool": tool, "source": source,
                          "arguments": kept, "arguments_sha256": _digest(arguments), "clipped": sorted(clipped),
                          "gate": gate, "trust": trust})
        return {**self._common, **fields, "calls_dropped": self._dropped.get(key, 0), "calls": calls}


def _closed(run: Run) -> None:
    """A run no longer open: the deletion stamps no open run of their conversation started before go."""
    with _COUNTS_LOCK:
        _OPEN.discard(run)
        opened = list(_OPEN)
        for conversation, at in list(_FORGOTTEN.items()):
            if not any(other._common["conversation_id"] == conversation and other._clock <= at for other in opened):
                del _FORGOTTEN[conversation]


def open_run(*, run_id: str, typed: str, trust: Any = None, conversation_id: str = "", model: str = "",
             mode: str = "", network_policy: str | None = None, run_settings: dict[str, Any] | None = None,
             offered: Iterable[str] = ()) -> Run | None:
    """The record of a run about to start, or None while the journal is off.

    Nothing is created or written here: the directory and the day's file
    are made by the run's first write, at its ``finish``. ``trust`` answers
    each observed call's trust (``agent.loop.CallTrust``).
    """
    current = settings()
    if not current["enabled"]:
        return None
    return Run(run_id=run_id, typed=typed, trust=trust, conversation_id=conversation_id, model=model, mode=mode,
               network_policy=network_policy, run_settings=dict(run_settings or {}), offered=offered,
               bounds={name: current[name] for name in BOUNDS})


# ---------------------------------------------------------------------------
# Reading: only reads -- a reader may be another process
# ---------------------------------------------------------------------------
def _listing() -> tuple[list[tuple[date, str]], dict[str, str]]:
    """The days a read may show, oldest first -- those a pass keeps: none past the retention, none after
    tomorrow -- and the deletions left to apply; read under the lock, changing nothing on disk."""
    try:
        with _held(LOCK_WAIT_S) as taken:
            if not taken:
                _failed("lock_timeout")
                return [], {}
            with _opened() as held:
                if held is None:
                    return [], _forgotten(None)
                cutoff = _cutoff(settings()["retention_days"])
                ahead = _now().date() + timedelta(days=1)
                return ([(day, name) for day, name in _days(held[0]) if cutoff <= day <= ahead],
                        _forgotten(held[0]))
    except Exception as exc:  # noqa: BLE001 - counted
        _failed(_reason(exc, "read_failed"))
        return [], {}


def _day_bytes(name: str) -> bytes | None:
    """One day's bytes, read under the lock; None when the day has gone or cannot be read (counted)."""
    try:
        with _held(LOCK_WAIT_S) as taken:
            if not taken:
                _failed("lock_timeout")
                return None
            with _opened() as held:
                return None if held is None else _read(name, held[0])
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001 - counted
        _failed(_reason(exc, "read_failed"))
        return None


def _order(record: dict[str, Any]) -> tuple[str, str]:
    return str(record.get("started_at") or ""), str(record.get("run_id") or "")


def iter_runs(kinds: Iterable[str] = KINDS) -> Iterator[dict[str, Any]]:
    """Every record of the ``kinds`` asked, ordered by start and identifier, one day at a time.

    A run is written when it ends, into the file of the day it started, so
    the order of a day's lines is not the order of its starts: a long run
    that started first comes first. A line that cannot be read is skipped and
    counted, and so is a file's tail cut short; a conversation whose deletion
    is left to apply is skipped; a day past the retention, or after tomorrow,
    is never shown.
    """
    wanted = set(kinds)
    days, forgotten = _listing()
    cipher, looked = None, False
    for _day, name in days:
        data = _day_bytes(name)
        if data is None:
            continue
        lines, torn = _split(data)
        if torn:
            _failed("torn")
        records = []
        for line in lines:
            if not looked and line.startswith(_SEALED_B):
                cipher, looked = _cipher(), True
            record = _decode(line, cipher)
            if record is None:
                _failed("unreadable")
            elif record.get("kind") in wanted and not _hidden(record, forgotten):
                records.append(record)
        records.sort(key=_order)
        yield from records


def trace_line(record: dict[str, Any]) -> dict[str, Any]:
    """A record as a line of the injection bench's trace, format 1: a run alone, by the user, under no plan.

    The bench's setting is the run's own (its mode, its network policy), its
    task the run's identifier; nothing in a real run has a plan's steps, an
    injection, a draw, an unread answer or a verdict of the bench's, which
    are left empty. ``executed``, ``refused`` and ``asked`` are counted from
    the gates; each call keeps its trust; ``started_at`` and the model join.
    """
    calls = list(record.get("calls") or ())
    gates = [call.get("gate") or {} for call in calls]
    refused: dict[str, int] = {}
    for gate in gates:
        if gate.get("executed") or gate.get("outcome") in ("halted", "not_dispatched"):
            continue
        reason = str(gate.get("reason") or "refused")
        refused[reason] = refused.get(reason, 0) + 1
    return {
        "format": TRACE_FORMAT,
        "suite": "journal",
        "arm": "journal",
        "persona": "user",
        "mode_asked": record.get("mode"),
        "policy_asked": record.get("network_policy"),
        "responder": None,
        "run": "alone",
        "draw": 0,
        "mode": record.get("mode"),
        "policy": record.get("network_policy"),
        "task": record.get("run_id"),
        "typed": record.get("typed"),
        "injection": None,
        "kind": None,
        "calls": [{"tool": call.get("tool"), "arguments": call.get("arguments"), "task_steps": [],
                   "injection_steps": [], "trust": call.get("trust")} for call in calls],
        "executed": sum(bool(gate.get("executed")) for gate in gates),
        "refused": dict(sorted(refused.items())),
        "asked": sum(bool(gate.get("asked")) for gate in gates),
        "unread": 0,
        "rounds": record.get("rounds"),
        "stop_reason": record.get("stop_reason"),
        "verdicts": {},
        "model": {"name": record.get("model"), "options": {}},
        "started_at": record.get("started_at"),
    }


def trace_lines(kinds: Iterable[str] = ("run",)) -> Iterator[dict[str, Any]]:
    """The records of the ``kinds`` asked, the runs alone by default, as trace lines (``trace_line``)."""
    for record in iter_runs(kinds):
        yield trace_line(record)


def status() -> dict[str, Any]:
    """The journal in numbers: the switch, the retention, how it writes now, the deletions left to apply, and
    per day the runs, sub-runs, calls, lines unreadable and bytes kept; the failures counted. No word of a record,
    and nothing changed on disk."""
    current = settings()
    days_listed, forgotten = _listing()
    cipher, looked = None, False
    days: dict[str, dict[str, int]] = {}
    for day, name in days_listed:
        data = _day_bytes(name)
        if data is None:
            continue
        lines, torn = _split(data)
        entry = {"runs": 0, "subruns": 0, "calls": 0, "unreadable": int(torn), "bytes": len(data)}
        for line in lines:
            if not looked and line.startswith(_SEALED_B):
                cipher, looked = _cipher(), True
            record = _decode(line, cipher)
            if record is None:
                entry["unreadable"] += 1
                continue
            if _hidden(record, forgotten):
                continue
            entry["runs" if record.get("kind") == "run" else "subruns"] += 1
            entry["calls"] += len(record.get("calls") or ())
        days[day.isoformat()] = entry
    sealing, refusal = _sealing(_machine_mode())
    return {"state": current["state"], "enabled": current["enabled"], "retention_days": current["retention_days"],
            "writes": "sealed" if sealing is not None else (refusal or "plaintext"),
            "forgets_pending": len(forgotten), "days": days, "failures": failures()}


# ---------------------------------------------------------------------------
# Privacy: a deleted conversation, the wipe, encryption turned on
# ---------------------------------------------------------------------------
def forget_conversation(conversation_id: str) -> int:
    """Remove the runs and sub-runs of ``conversation_id`` from every day, with every line a loaded key cannot open.

    A run of it going now is never written. A sealed line no key opens is
    kept, and the deletion kept in the ledger (``_remember``) for the next
    pass with the key; so is a deletion that fails or finds the journal busy.
    A day that cannot be rewritten is removed whole when it held one of the
    conversation's records, and kept otherwise (counted). The runs of no
    conversation ("") are never touched. Answers the records removed now, -1
    when a day could not be read or the deletion not made (counted). Never
    raises.
    """
    wanted = str(conversation_id or "")
    if not wanted:
        return 0
    with _COUNTS_LOCK:
        if any(run._common["conversation_id"] == wanted for run in _OPEN):
            _FORGOTTEN[wanted] = time.monotonic()
    try:
        with _held(GESTURE_WAIT_S) as taken:
            if not taken:
                _failed("lock_timeout")
                _remember(wanted)
                return -1
            with _opened() as held:
                if held is None:
                    return 0
                fd = held[0]
                _tend(fd)
                cipher = _cipher()
                removed, refused, deferred = 0, False, False
                for _day, name in _days(fd):
                    try:
                        lines, torn = _split(_read(name, fd))
                    except OSError as exc:
                        _failed(_reason(exc, "forget_failed"))
                        refused = True
                        continue
                    kept, changed, named = [], torn, False
                    for line in lines:
                        if cipher is None and line.startswith(_SEALED_B):
                            kept.append(line)
                            deferred = True
                            continue
                        record = _decode(line, cipher)
                        if record is None:
                            changed = True
                            continue
                        if record.get("conversation_id") == wanted:
                            removed += 1
                            changed = named = True
                            continue
                        kept.append(line)
                    if changed:
                        try:
                            _rewrite(name, kept, fd)
                        except OSError as exc:
                            if named:
                                refused = not _dropped_day(name, fd) or refused
                            else:
                                _failed(_reason(exc, "forget_failed"))
                if refused or deferred:
                    _remember(wanted, fd)
                return -1 if refused else removed
    except Exception as exc:  # noqa: BLE001 - counted, and kept for the next pass
        _failed(_reason(exc, "forget_failed"))
        _remember(wanted)
        return -1


def wipe() -> dict[str, Any]:
    """Remove every file of the journal, whatever it holds and whoever it was for; what was removed.

    A run going now is never written. The journal is single-user, so a
    user's wipe removes it whole; a file that is not the journal's is left,
    and so is the directory then. A name that cannot be removed is said, and
    the others are removed all the same.
    """
    _STATE["wiped_at"] = time.monotonic()
    try:
        with _held(GESTURE_WAIT_S) as taken:
            if not taken:
                _failed("lock_timeout")
                return {"wiped": False, "reason": "lock_timeout"}
            with _opened() as held:
                if held is None:
                    with _COUNTS_LOCK:
                        _LEDGER.clear()
                    return {"wiped": True, "files": 0}
                fd, parent = held
                removed, refusal = 0, ""
                for name in _names(fd):
                    if _day_of(name) is None and not _is_temp(name) and name != _LEDGER_NAME:
                        continue
                    try:
                        os.unlink(name, dir_fd=fd)
                        removed += 1
                    except FileNotFoundError:
                        continue
                    except OSError as exc:
                        refusal = _reason(exc, "wipe_failed")
                        _failed(refusal)
                with contextlib.suppress(OSError):
                    os.fsync(fd)
                with _COUNTS_LOCK:
                    _LEDGER.clear()
                if refusal:
                    return {"wiped": False, "reason": refusal, "files": removed}
                _STATE["plain"] = False
                try:
                    os.rmdir(DIRNAME, dir_fd=parent)
                except OSError:
                    pass  # a file not the journal's keeps it
                else:
                    with contextlib.suppress(OSError):
                        os.fsync(parent)
                return {"wiped": True, "files": removed}
    except Exception as exc:  # noqa: BLE001 - counted, and said in the answer
        why = _reason(exc, "wipe_failed")
        _failed(why)
        return {"wiped": False, "reason": why}


def on_encryption_enabled() -> int:
    """Seal now every record kept in plaintext, encryption having just been turned on; the records sealed.

    The key just set up is the one taken (the cipher kept is dropped first).
    The days past the retention go first, and the deletions left to apply are
    applied. -1, counted, when no key loads; -1 when the sealing could not be
    made; the next pass tries again either way.
    """
    _STATE["plain"] = True
    _STATE["cipher"] = None
    try:
        with _held(GESTURE_WAIT_S) as taken:
            if not taken:
                _failed("lock_timeout")
                return -1
            with _opened() as held:
                if held is None:
                    return 0
                fd = held[0]
                try:
                    _prune_in(fd, settings()["retention_days"])
                except Exception as exc:  # noqa: BLE001 - counted; the sealing goes on
                    _failed(_reason(exc, "prune_failed"))
                try:
                    _apply_forgotten(fd)
                except Exception as exc:  # noqa: BLE001 - counted; the sealing goes on
                    _failed(_reason(exc, "forget_failed"))
                sealed = _reseal_in(fd)
                if sealed < 0:
                    _failed("seal_unavailable")
                return sealed
    except Exception as exc:  # noqa: BLE001 - counted
        _failed(_reason(exc, "reseal_failed"))
        return -1
