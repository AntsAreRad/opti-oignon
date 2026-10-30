"""The answers the caches keep, forgotten when what they may have been drawn from is deleted.

The response cache and the semantic cache keep answers by the request and the prompt they were given for -- a prompt
built from the knowledge base's documents, a project's files, the notes, the skills, the conversation's window --, and
no entry says which of those its prompt held. So the deletion of a knowledge-base document or collection, a project,
a note or a skill empties both caches whole, overwritten (each one's ``empty``): an answer drawn from the deleted words
never stays behind them. An answer a request already under way puts after the emptying is refused by the cache
itself, its lookup older than the emptying.

A deletion asks it of its own data folder once a cache's file is there (the caller checks, so that nothing is loaded
to be asked of a folder that holds none); a cache that lives in another folder is not this store's, and is left.
Nothing of the package is imported to load this module.
"""

from __future__ import annotations

import logging
import sys
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


def begin_cached_request() -> dict[str, int]:
    """A request begins, before its prompt draws on any store: each cache this process loaded marks it
    (``begin_request``); the marks, which the request keeps and hands back right before each of its puts
    (``declare_cached_request``). A cache that cannot mark is left out: it then marks a lookup when it misses."""
    marks: dict[str, int] = {}
    for name in _CACHE_MODULES:
        begin = getattr(sys.modules.get(name), "begin_request", None)
        if callable(begin):
            try:
                marks[name] = int(begin())
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
    asks it; what stays, named. A cache whose file is not there holds none -- a lookup makes it, so no request is
    under way on it either --, and nothing is loaded; one this process keeps in another folder is another store's,
    and is left."""
    left: list[str] = []
    for label, filename, held in CACHES:
        path = Path(data) / filename
        if not path.exists():
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
