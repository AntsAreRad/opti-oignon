"""The answers the caches keep, forgotten when what they may have been drawn from is deleted.

The response cache and the semantic cache keep answers by the request and the prompt they were given for -- a prompt
built from the knowledge base's documents, a project's files, the notes, the skills, the conversation's window --, and
no entry says which of those its prompt held. So the deletion of a knowledge-base document or collection, a project,
a note or a skill empties both caches whole, overwritten (each one's ``empty``): an answer drawn from the deleted words
never stays behind them. An answer a request already under way puts after the emptying is refused by the cache
itself, its lookup older than the emptying -- older than the beginning of the turn it belongs to, when it is a later
call of a turn (a pipeline's step, the agentic executor's next pass), whose prompt repeats words drawn before it.

A deletion asks it of its own data folder once a cache's file is there, or once this process loaded a cache, which a
request may be under way on before its first lookup makes the file (the caller checks, so that nothing is loaded to be
asked of a folder that holds none in a process that loaded none); a cache that lives in another folder is not this
store's, and is left. Another process on the same folder -- the server and ``oo chat`` -- is told through the file
itself: each emptying draws the file's generation anew, and a put there reads it in its own write.
Nothing of the package is imported to load this module.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import weakref
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

def _response_cache() -> Any:
    from opti_oignon.response_cache import response_cache

    return response_cache


def _semantic_cache() -> Any:
    from opti_oignon.semantic_cache import semantic_cache

    return semantic_cache


# Each cache: the label the account wipe's answer names it by (``store_inventory``), its file in the data folder, and
# the process's instance, loaded when asked.
CACHES = (
    ("response cache", "response_cache.db", _response_cache),
    ("semantic cache", "semantic_cache.db", _semantic_cache),
)


# The caches' modules, which mark a request when this process has loaded them.
_CACHE_MODULES = ("opti_oignon.response_cache", "opti_oignon.semantic_cache")
# Each cache's file, by its module and the module's instance.
_LOADED_AS = {"response_cache.db": ("opti_oignon.response_cache", "response_cache"),
              "semantic_cache.db": ("opti_oignon.semantic_cache", "semantic_cache")}

# The marks of each turn under way, by the stop every call of the turn shares -- the chat turn's, marked when the route
# opens it and handed to the pipeline runner's steps, to the agentic executor's passes and to each executor call --,
# held weakly: the first call of a turn marks it, and its later calls begin from those marks.
_TURNS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_TURNS_LOCK = threading.Lock()


def begin_cached_request(run: Any = None) -> dict[str, int]:
    """A request begins, before its prompt draws on any store: each cache this process loaded marks it
    (``begin_request``); the marks, which the request keeps and hands back right before each of its puts
    (``declare_cached_request``). A cache that cannot mark is left out: it then marks a lookup when it misses.

    ``run``: the turn the request belongs to (``mark_cached_turn``). The turn is marked if it was not yet, and the
    request begins from the turn's marks -- a pipeline's step, the agentic executor's next pass, whose prompt repeats
    words drawn before it, is judged by the turn's beginning. A cache the turn's marks lack (not loaded when it was
    marked) begins from its first count: every emptying of it came after the turn began."""
    turn = mark_cached_turn(run)
    return _begin(turn or {}, known=turn is not None)


def mark_cached_turn(run: Any) -> dict[str, int] | None:
    """Mark the turn ``run`` in the caches if it was not yet, before anything of it draws on a store; the turn's marks,
    or None when ``run`` is no turn. A turn is known by its stop -- the ``threading.Event`` every call of the turn
    shares; a run with another stop, or none, is no turn. Its marks are each loaded cache's count, read without the
    caches' locks (``emptied``), so marking never waits on a put's disk work -- the chat route marks on its event
    loop; two calls of one turn marking at once take the same marks."""
    stop = getattr(run, "stop", None)
    if not isinstance(stop, threading.Event):
        return None
    with _TURNS_LOCK:
        held = _TURNS.get(stop)
        if held is None:
            held = _TURNS[stop] = _counts()
        return dict(held)


def _counts() -> dict[str, int]:
    """Each loaded cache's count of emptyings, read without its lock."""
    counts: dict[str, int] = {}
    for name in _CACHE_MODULES:
        emptied = getattr(sys.modules.get(name), "emptied", None)
        if callable(emptied):
            try:
                counts[name] = int(emptied())
            except Exception:  # noqa: BLE001 - the mark is a guard, never a reason to fail a turn
                logger.debug("a cache could not mark the turn", exc_info=True)
    return counts


def _begin(held: dict[str, int], *, known: bool) -> dict[str, int]:
    """Each loaded cache begins the request (``begin_request``): from the turn's mark ``held`` for it, from its
    first count when the turn is ``known`` but did not mark it, from now otherwise."""
    marks: dict[str, int] = {}
    for name in _CACHE_MODULES:
        begin = getattr(sys.modules.get(name), "begin_request", None)
        if callable(begin):
            since = held.get(name, 0 if known else None)
            try:
                marks[name] = int(begin() if since is None else begin(since))
            except Exception:  # noqa: BLE001 - the mark is a guard, never a reason to fail a request
                logger.debug("a cache could not mark the request", exc_info=True)
    return marks


def declare_cached_request(marks: dict[str, int], cache: str) -> None:
    """The request whose ``marks`` these are is about to put its answer into ``cache`` (its module's name), on this
    thread: that cache refuses the put when it was emptied since the request began (``declare_request``, spent by the
    put). Only the cache about to be written is told: a mark left in another would outlive the request."""
    for name, mark in marks.items():
        if name != cache:
            continue
        declare = getattr(sys.modules.get(name), "declare_request", None)
        if callable(declare):
            try:
                declare(mark)
            except Exception:  # noqa: BLE001 - an undeclared put falls back to its lookup's mark
                logger.debug("a cache could not take the request's mark", exc_info=True)


def forget_cached_answers(data: Path) -> list[str]:
    """Empty, overwritten, both caches of the folder ``data``: a deletion of what an answer may have been drawn from
    asks it; what stays, named. A cache whose file is not there holds none, and nothing is loaded for it; but a
    request may be under way on one this process loaded, marked before its first lookup makes the file, so its
    emptying is counted. One this process keeps in another folder is another store's, and is left."""
    left: list[str] = []
    for label, filename, held in CACHES:
        path = Path(data) / filename
        if not path.exists():
            left += _count_emptying(label, path)
            continue
        try:
            cache = held()
            if Path(cache._db_path).resolve() != path.resolve():
                continue
            cache.empty()
        except Exception as exc:  # noqa: BLE001 - what stays is said
            logger.warning("The %s could not be emptied after a deletion: %s", label, type(exc).__name__)
            left.append(f"{label} (the answers it keeps not emptied: {type(exc).__name__}; the next deletion or the "
                        "wipe empties it)")
    return left


def _count_emptying(label: str, path: Path) -> list[str]:
    """The cache of the file ``path``, not there yet, emptied when this process loaded it: no answer is held, but the
    emptying is counted (its ``empty``), so a request under way on it is refused. Nothing is loaded for it."""
    module, name = _LOADED_AS[path.name]
    cache = getattr(sys.modules.get(module), name, None)
    # A cache that names no file is no folder's.
    if getattr(cache, "_db_path", None) is None:
        return []
    try:
        # Compared by spelling, never on disk: a cache of another folder -- the process's own, in its data folder --
        # is not looked at.
        if os.path.abspath(cache._db_path) == os.path.abspath(path):
            cache.empty()
    except Exception as exc:  # noqa: BLE001 - what stays is said
        logger.warning("The %s could not count an emptying after a deletion: %s", label, type(exc).__name__)
        return [f"{label} (a request under way on it not refused: {type(exc).__name__})"]
    return []
