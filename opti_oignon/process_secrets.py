#!/usr/bin/env python3
"""The platform's two secrets in this process: taken out of its environment, held, and kept from its children.

The launchers hand the backend the key file's passphrase (``OPTI_KEYFILE_PASSPHRASE``) and the master key
(``OPTI_ENCRYPTION_KEY``) in its environment. Left there, every child the backend starts would hold them: one
started without ``env=`` is handed the whole environment, and so is each one no ``env=`` can reach -- the
background pool's workers, the multiprocessing resource tracker, the compiler and the linker
``ctypes.util.find_library`` runs, the ``uname -p`` of ``platform.processor()``. And a process of the same user
that shares the process table reads them in ``/proc/<pid>/environ``, which shows the environment a process was
started with, whatever it takes out of ``os.environ`` later.

``take()`` is the first statement of the application's entry (``opti_oignon/api/app.py``), ahead of every import
that starts a process. For each secret it zeroes the value of every entry of that name in the process's initial
environment block -- what ``/proc/<pid>/environ`` reads --, reads the block back there to see each value gone,
takes the name out of ``os.environ``, and holds the value; an empty value is no secret, and leaves too. It also
takes out of ``os.environ`` what a shell runs code from -- ``BASH_ENV``, ``ENV``, ``SHELLOPTS``, ``BASHOPTS``,
``PS4``, and every function exported as ``BASH_FUNC_*`` --, so that no child of the backend, started with an
environment of its own or inheriting this one, is handed them. ``secret(name)`` is the one read of the two names:
the environment's value when it holds one, then the held one. ``child_env()`` is the environment a child is started
with: this process's, without the two secrets and without those shell variables, whatever has put them back.
``withdraw(name)`` zeroes a secret and takes it out without holding it: the launcher and the key file's gate call it
for those they take out of their own environment.

A value is held in memory of its own, locked where the system allows (``status()`` names one it could not lock; a
lock is kept until the process ends, since a page may hold another holder's value), and wiped when the process exits
-- never by a signal's handler, which runs while the rest of the process may still read the key. A process a signal
kills is not wiped: its memory goes with it, and no core is dumped (below). A held value read once it is wiped
raises ``SecretWiped``, never answers "no secret". A zeroed value leaves its length readable: the block keeps one zero
per byte. A worker of the background pool, started after ``take()``, holds neither secret: a task that needs a key
is handed it by this process, never by the environment.

Once ``take()`` holds a value it installs a check, an audit hook (``sys.addaudithook``) that runs for as long as the
interpreter does. A process start Python audits -- ``subprocess``, ``os.exec*``, ``os.posix_spawn*``, ``os.system``,
``pty.spawn``; on POSIX ``os.spawn*`` is a fork, then an exec in the child -- is refused (``PermissionError``) when
the environment it is handed names a secret or what a shell runs code from, or holds a held value in a variable's
name or value, or when its arguments hold one. A variable put into this process's environment through Python
(``os.putenv``, ``os.environ``) is refused on the same terms: a start handed no environment -- one of those, a fork,
or one Python does not audit, multiprocessing's children through ``_posixsubprocess`` -- inherits this environment,
which no Python code of this interpreter can then give a secret. An update stops at the variable it refuses; those
before it are set. A refusal is logged without the value and without where it sat. A value is looked for wherever it
sits; one shorter than 8 bytes is not -- it would match by chance in paths and words, a ``SHLVL=1`` among them --,
nor one a variable of the environment already holds when it is taken -- a passphrase that is the login name, which
every child that inherits or is handed this environment holds anyway, and which one handed a cleared or allowlisted
environment can be handed unseen --, and ``status()`` names both. Hooks are the interpreter's own: a subinterpreter
runs without this one, and the census guard counts one as a start of its own. Code in this process can turn the
check off -- its tables and its state are the module's --: it guards against a mistake, not against code that means
to. The C library called through ctypes, which can start a process or write the environment without Python, is not
checked.

A value that could not be zeroed in the initial block, or that the block still shows once zeroed, is said, never
taken for a wiped one: it is logged, the startup checklist reports it, and the two places where untrusted code runs
beside this process -- a sandbox command without bubblewrap, a plugin started without it -- are refused while it is
there (``unwiped_refusal``). While a secret is held, this process and its children have no core size to dump
(``RLIMIT_CORE`` 0, both limits): a core handler that honours the limit -- the kernel's own file, systemd-coredump --
stores none of the copies the heap still holds, the string the environment held or those a key derivation makes.

Stdlib only, and nothing here starts a process: ``take()`` runs before anything else is imported.
"""

from __future__ import annotations

import atexit
import ctypes
import logging
import os
import sys
import threading
import types

logger = logging.getLogger(__name__)

SECRET_NAMES = ("OPTI_KEYFILE_PASSPHRASE", "OPTI_ENCRYPTION_KEY")
# What a shell started with the environment runs code from, or changes how it runs: bash reads BASH_ENV and imports
# the functions exported as BASH_FUNC_*, and traces every line under an inherited SHELLOPTS with the PS4 it is
# given; ENV and BASHOPTS are read by interactive shells and change what a shell does.
SHELL_HAZARDS = ("BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS", "PS4")
FUNCTION_PREFIX = "BASH_FUNC_"

_lock = threading.Lock()
_held: dict = {}
_unwiped: dict = {}
_taken = False
_wipe_at_exit = []
# What the check reads, replaced whole and never under _lock: the check also runs while _lock is held, for the C
# library a holder calls is audited too.
_watched: tuple = ()
_checking = []
# The process starts Python audits: (where the program's arguments are, where its environment is), None where the
# event carries none -- a start handed no environment inherits this process's. A fork copies this process.
_START_EVENTS = types.MappingProxyType({
    "subprocess.Popen": (1, 3),
    "os.exec": (1, 2),
    "os.posix_spawn": (1, 2),
    "os.spawn": (2, 3),
    "os.system": (0, None),
    "pty.spawn": (0, None),
})
# A variable put into this process's environment, which the starts Python does not audit inherit.
_PUT_EVENT = "os.putenv"
# From this length, in bytes, a held value is looked for; a shorter one is not, and is said.
_LOOKED_FOR_FROM = 8


class SecretWiped(RuntimeError):
    """A secret this process held has been wiped, at exit: never answered as absent."""


def withheld(name: str) -> bool:
    """Whether ``name`` is never handed to a child: one of the two secrets, or what a shell runs code from."""
    return name in SECRET_NAMES or name in SHELL_HAZARDS or name.startswith(FUNCTION_PREFIX)


def _named(name: str) -> None:
    if name not in SECRET_NAMES:
        raise ValueError(f"{name} is not one of the platform's secrets ({', '.join(SECRET_NAMES)})")


def _initial_block() -> tuple:
    """Where this process's initial environment block lies, and that it can be written there.

    The kernel shows its bounds in ``/proc/self/stat`` (env_start and env_end, the 50th and 51st fields, counted
    after the last parenthesis, which closes the command's name), and ``/proc/<pid>/environ`` reads exactly that
    range.
    """
    with open("/proc/self/stat", "rb") as stat:
        fields = stat.read().rsplit(b")", 1)[1].split()
    start, end = int(fields[47]), int(fields[48])
    if not 0 < start < end:
        raise OSError("the kernel shows no initial environment block for this process")
    with open("/proc/self/maps", "rb") as maps:
        for line in maps:
            span, perms = line.split()[:2]
            low, high = (int(bound, 16) for bound in span.split(b"-"))
            if low <= start and end <= high:
                if b"w" not in perms:
                    raise OSError("the initial environment block lies in memory this process cannot write")
                return start, end
    raise OSError("the initial environment block lies in no single mapping of this process")


def _zero(name: str, block: tuple) -> list:
    """Zero, in place, the value of every ``name=`` entry of the initial block: ``[(offset, length)]`` of each value
    zeroed, its offset counted from the block's start."""
    strnlen = ctypes.CDLL(None).strnlen
    strnlen.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    strnlen.restype = ctypes.c_size_t
    prefix = os.fsencode(name) + b"="
    start, end = block
    zeroed, at = [], start
    while at < end:
        length = strnlen(at, end - at)
        if length >= len(prefix) and ctypes.string_at(at, len(prefix)) == prefix:
            ctypes.memset(at + len(prefix), 0, length - len(prefix))
            zeroed.append((at + len(prefix) - start, length - len(prefix)))
        at += length + 1
    return zeroed


def _still_shown(value: str | None, zeroed: list) -> str | None:
    """Why the block ``/proc/self/environ`` shows still holds the value, or None: a value zeroed that does not read
    as zeros there, or the value itself anywhere in it."""
    with open("/proc/self/environ", "rb") as block:
        shown = block.read()
    if any(any(shown[offset:offset + length]) for offset, length in zeroed):
        return "a value it zeroed still reads in /proc/self/environ"
    if value and os.fsencode(value) in shown:
        return "/proc/self/environ still shows the value"
    return None


def withdraw(name: str) -> str | None:
    """Take the secret ``name`` out of this process's environment: its value, or None when it was not there.

    Every entry of that name in the environment the process was started with is zeroed first, whatever its value
    -- a name given twice is zeroed twice --, and the block is read back where ``/proc/<pid>/environ`` reads it. A
    value that cannot be zeroed there, or that still shows, is recorded, said in the log and by
    ``unwiped_refusal()``, and is still taken out of ``os.environ``. A process whose block cannot be read, and whose
    environment holds no value for the name, records nothing: it has nothing known to wipe.
    """
    _named(name)
    value = os.environ.get(name)
    try:
        why = _still_shown(value, _zero(name, _initial_block()))
    except Exception as exc:  # noqa: BLE001 - whatever stops the wipe is said, never taken for one
        why = f"{type(exc).__name__}: {exc}" if value else None
    if why:
        with _lock:
            _unwiped[name] = why
        logger.warning("%s could not be zeroed in this process's initial environment, where a process of the "
                       "same user can read it (%s)", name, why)
    os.environ.pop(name, None)
    return value


class _Held:
    """A held secret: a bytearray locked in memory where the system allows, zeroed by ``wipe()``."""

    __slots__ = ("_buf", "is_wiped", "locked", "common")

    def __init__(self, data: bytearray) -> None:
        self._buf, self.is_wiped, self.locked, self.common = bytearray(data), False, False, False
        data[:] = bytes(len(data))
        if self._buf:
            try:
                libc = ctypes.CDLL(None, use_errno=True)
                libc.mlock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
                address = ctypes.addressof((ctypes.c_char * len(self._buf)).from_buffer(self._buf))
                self.locked = libc.mlock(address, len(self._buf)) == 0
            except (AttributeError, OSError):
                self.locked = False

    def as_bytes(self) -> bytes:
        if self.is_wiped:
            raise RuntimeError("the held value has been wiped")
        return bytes(self._buf)

    def wipe(self) -> None:
        """Zero the value. Its lock is kept: locks do not stack, and the page may hold a live holder's value."""
        if self.is_wiped:
            return
        if self._buf:
            view = (ctypes.c_char * len(self._buf)).from_buffer(self._buf)
            ctypes.memset(ctypes.addressof(view), 0, len(self._buf))
            del view
        self.is_wiped = True

    def found_in(self, data: bytes) -> bool:
        """Whether ``data`` holds this value, compared in place; never for a value shorter than
        ``_LOOKED_FOR_FROM``, nor for one the environment held already when it was taken (``common``)."""
        if self.is_wiped or self.common or len(self._buf) < _LOOKED_FOR_FROM:
            return False
        return data.find(self._buf) != -1


def _wipe_held() -> None:
    with _lock:
        for holder in _held.values():
            holder.wipe()


def _encoded(item) -> bytes | None:
    try:
        return os.fsencode(item)
    except TypeError:
        return None


def _start_refusal(event: str, args: tuple) -> str | None:
    """Why the process start ``event`` must not be made, or None: what part of it holds a secret, or names what a
    shell runs code from."""
    argv_at, env_at = _START_EVENTS[event]
    held = [holder for holder in _watched if not holder.is_wiped]
    env = args[env_at] if env_at is not None and len(args) > env_at else None
    # A start handed no environment inherits this process's, into which no secret can be put (_put_refusal).
    try:
        entries = [] if env is None else list(env.items())
    except (AttributeError, TypeError):
        return "its environment could not be read"
    for name, value in entries:
        key = name.decode("utf-8", "replace") if isinstance(name, bytes) else str(name)
        if withheld(key):
            return f"its environment names {key}"
        encoded, encoded_name = _encoded(value), _encoded(name)
        if encoded is None or encoded_name is None:
            return "its environment could not be read"
        if any(holder.found_in(encoded) or holder.found_in(encoded_name) for holder in held):
            return "its environment holds a secret's value"
    if argv_at is not None and len(args) > argv_at and held:
        argv = args[argv_at]
        items = [argv] if isinstance(argv, (str, bytes, os.PathLike)) else list(argv or ())
        for item in items:
            encoded = _encoded(item)
            if encoded is not None and any(holder.found_in(encoded) for holder in held):
                return "its arguments hold a secret"
    return None


def _put_refusal(event: str, args: tuple) -> str | None:
    """Why a variable must not be put into this process's environment, or None: a secret's name, or what a shell
    runs code from, put back; or a held value as a variable's name or value."""
    key, value = (tuple(args) + (None, None))[:2]
    name = key.decode("utf-8", "replace") if isinstance(key, bytes) else str(key)
    if withheld(name):
        return f"{name} put back in this process's environment"
    encoded_key = _encoded(key)
    encoded = b"" if value is None else _encoded(value)
    if encoded is None or encoded_key is None:
        return "a variable that could not be read put in this process's environment"
    held = [holder for holder in _watched if not holder.is_wiped]
    if any(holder.found_in(encoded) or holder.found_in(encoded_key) for holder in held):
        return "a secret's value put in this process's environment, which every child inherits"
    return None


def _make_check(starts, put, refuse_start, refuse_put, log):
    """The audit hook, its tables and its functions bound as it is made: the module's names rebound later, or cleared
    as the interpreter ends, leave it as it is."""

    def check(event: str, args: tuple) -> None:
        if event in starts:
            refusal = refuse_start
        elif event == put:
            refusal = refuse_put
        else:
            return
        try:
            why = refusal(event, args)
        except Exception as exc:  # noqa: BLE001 - what cannot be read is refused, never let through
            why = f"what it would be handed could not be read ({type(exc).__name__})"
        if why:
            try:
                log.warning("Refused (%s): %s", event, why)
            except Exception:  # noqa: BLE001 - a log that cannot be written stops no refusal
                pass
            raise PermissionError(f"refused ({event}): {why}; a child never holds the platform's secrets")

    return check


def _no_core_dumps() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, OSError, ValueError) as exc:
        logger.warning("core dumps could not be turned off while this process holds a secret (%s)", exc)


def take() -> None:
    """Take both secrets out of this process's environment and hold them, and the shell's variables with them.

    What it cannot do is said and stops nothing: a value it cannot zero in the initial block, or that the block
    still shows, is recorded (``unwiped_refusal``) and held all the same. A second call holds a value for a name it
    does not hold yet. Once a value is held, the check is installed (once): Python can then put no secret's name
    back in the environment, nor a held value under another.
    """
    global _taken, _watched
    given = {}
    for name in SECRET_NAMES:
        value = withdraw(name)
        if value:
            given[name] = value
    # The shell's variables alone: the secrets leave through withdraw(), whose failure must not be covered here.
    for name in [name for name in os.environ if name in SHELL_HAZARDS or name.startswith(FUNCTION_PREFIX)]:
        os.environ.pop(name, None)
    if given:
        with _lock:
            for name, value in given.items():
                if name in _held:
                    # The check refuses a put through os.putenv once a value is held: only a write to posix.environ
                    # itself, which os.environ reads and no child inherits, can have given it again.
                    logger.warning("%s was given again; the value this process holds is kept", name)
                    continue
                _held[name] = _Held(bytearray(os.fsencode(value)))
            if not _wipe_at_exit:
                atexit.register(_wipe_held)
                _wipe_at_exit.append(True)
            # A value the environment holds already, under another name, is no secret from any child.
            words = [os.fsencode(value) for value in os.environ.values()]
            for holder in _held.values():
                holder.common = bool(holder._buf) and any(word.find(holder._buf) != -1 for word in words)
            watched = tuple(_held.values())
        _watched = watched
        if not _checking:
            sys.addaudithook(_make_check(_START_EVENTS, _PUT_EVENT, _start_refusal, _put_refusal, logger))
            _checking.append(True)
        _no_core_dumps()
    _taken = True


def secret(name: str) -> str | None:
    """The secret ``name``: the environment's value when it holds one, else the value ``take()`` holds, else None.

    A held value that has been wiped raises ``SecretWiped``: answered as absent, it would read as "no passphrase"
    and let a key file be written unwrapped, or a store be opened in plaintext.
    """
    _named(name)
    value = os.environ.get(name)
    if value:
        return value
    with _lock:
        held = _held.get(name)
        if held is None:
            return None
        if held.is_wiped:
            raise SecretWiped(f"{name} was held by this process and has been wiped: it is not answered as absent")
        return os.fsdecode(held.as_bytes())


def child_env(extra: dict | None = None) -> dict:
    """The environment a child is started with: this process's, without the platform's secrets and without what
    a shell runs code from, plus ``extra``. ``os.environ`` is left as it is.

    ``extra`` naming a secret or one of those shell variables is refused (``ValueError``): a child that needs a
    secret is handed it through a channel of its own, never the environment.
    """
    extra = dict(extra or {})
    refused = sorted(name for name in extra if withheld(name))
    if refused:
        raise ValueError(f"a child's environment never carries {', '.join(refused)}")
    env = {name: value for name, value in os.environ.items() if not withheld(name)}
    env.update(extra)
    return env


def unwiped_refusal() -> str | None:
    """Why code that runs untrusted beside this process must not start now, or None.

    A secret this process was handed and could not zero in its initial environment can be read there, in
    ``/proc/<pid>/environ``, by any process of the same user that shares the process table, whatever environment
    that process was given.
    """
    with _lock:
        unwiped = sorted(_unwiped.items())
    if not unwiped:
        return None
    named = "; ".join(f"{name}: {why}" for name, why in unwiped)
    return (f"this process could not zero a secret in the environment it was started with ({named}), where code "
            "running beside it could read it: nothing runs here outside bubblewrap until the platform is started "
            "again")


def status() -> dict:
    """What ``take()`` did in this process: whether it ran, the names it holds, those whose memory could not be
    locked, those the check does not look for -- too short (under ``looked_for_from`` bytes), or held by the
    environment already (``common``) --, those it could not zero (with why), and the secret names the environment
    holds now."""
    with _lock:
        return {
            "taken": _taken,
            "held": sorted(_held),
            "unlocked": sorted(name for name, holder in _held.items() if not holder.locked),
            "short": sorted(name for name, holder in _held.items() if len(holder._buf) < _LOOKED_FOR_FROM),
            "common": sorted(name for name, holder in _held.items() if holder.common),
            "looked_for_from": _LOOKED_FOR_FROM,
            "unwiped": dict(_unwiped),
            "in_environment": sorted(name for name in SECRET_NAMES if os.environ.get(name)),
        }
