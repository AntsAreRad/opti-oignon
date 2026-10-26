"""The trash of the facts: a deleted fact waits there, then is erased for good.

Deleting a fact takes it out of service (``MemoryStore.soft_delete``): it
leaves the window and the vector layer, and its row keeps the text so that it
can be restored. It stays so for ``trash_days`` (``config/memory.yaml``, 30 by
default), counted from its last change -- its deletion -- and then ``tend``
erases it: the row deleted and overwritten, the write-ahead log emptied, the
paired devices told (a tombstone). The user can erase one at once (``erase``)
or all of theirs (``empty``). ``tend`` runs at boot and every hour while the
server runs, as the run journal's retention does, so a fact deleted long ago
is erased at the first start that finds it.

An erasure takes a fact only while it is in the trash: one restored meanwhile
stays.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Every new module hardcodes this; it is never overridable.
checkpoint_before_apply = True

logger = logging.getLogger(__name__)

_CONFIG = Path(__file__).resolve().parent.parent / "config" / "memory.yaml"
_DEFAULTS: dict[str, Any] = {"trash_days": 30}
# The whole numbers each setting may take; anything else keeps the default.
_BOUNDS: dict[str, tuple[int, int]] = {"trash_days": (1, 3650)}
# Seconds between two passes while the server runs.
PRUNE_INTERVAL_S = 3600


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """The trash's settings from ``config/memory.yaml``.

    A missing file or library keeps the defaults; a value that is not a whole
    number within its bounds is refused by name and the default kept.
    """
    config = dict(_DEFAULTS)
    try:
        import yaml

        raw = yaml.safe_load(Path(path or _CONFIG).read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("memory: configuration unreadable (%s); defaults kept", exc)
        return config
    for key, (low, high) in _BOUNDS.items():
        value = raw.get(key) if isinstance(raw, dict) else None
        if type(value) is int and low <= value <= high:
            config[key] = value
        elif value is not None:
            logger.warning("memory: %s %r is not a whole number from %d to %d; %d kept",
                           key, value, low, high, config[key])
    return config


def trash_days() -> int:
    """The days a deleted fact stays in the trash."""
    return int(load_config()["trash_days"])


def _store(store: Any) -> Any:
    if store is not None:
        return store
    from .dedup import get_memory_store

    return get_memory_store()


def erase_at(record: Any, days: int | None = None) -> str | None:
    """When a fact in the trash is erased (ISO-8601 UTC): its last change plus the trash's days; None when its last
    change cannot be read, and the next pass erases it."""
    try:
        changed = datetime.fromisoformat(str(getattr(record, "updated_at", "")))
    except ValueError:
        return None
    if changed.tzinfo is None:
        changed = changed.replace(tzinfo=timezone.utc)
    return (changed + timedelta(days=trash_days() if days is None else days)).isoformat()


def listing(*, store: Any = None, user_id: str | None = None) -> tuple[int, list[tuple[Any, str | None]]]:
    """The trash's days, and the user's facts in the trash, each with when it is erased."""
    days = trash_days()
    held = [record for record in _store(store).list(active_only=False, user_id=user_id) if not record.active]
    return days, [(record, erase_at(record, days)) for record in held]


def _anything_to_do(cutoff: str) -> bool:
    """Whether the facts hold one in the trash since ``cutoff`` or before, or a vector a peer's change left owed,
    read without opening the vector layer; a store whose file is not there holds none, and is not created to be
    asked."""
    from .canonical_store import _default_db_path, get_canonical_store

    if not _default_db_path().exists():
        return False
    canonical = get_canonical_store()
    return bool(canonical.trashed_before(cutoff)) or bool(canonical.owed_vectors())


def tend(*, store: Any = None, now: datetime | None = None) -> int:
    """Erase every fact the trash has held past its days, every user's; how many. Never raises: a fact that stays
    is said, and the next pass erases it. It also settles the vectors a peer's change left owed. With nothing to
    do, the facts' store is not opened whole."""
    try:
        cutoff = ((now or datetime.now(timezone.utc)) - timedelta(days=trash_days())).isoformat()
        if store is None and not _anything_to_do(cutoff):
            return 0
        held = _store(store)
        erased = 0
        for fact_id, _owner in held.trashed_before(cutoff):
            if held.erase(fact_id):
                erased += 1
        if erased:
            held.empty_log(f"{erased} fact(s) erased from the trash")
            logger.info("memory: %d fact(s) erased from the trash, past their days", erased)
        settle = getattr(held, "settle_vectors", None)
        if callable(settle):
            settle()
        return erased
    except Exception as exc:  # noqa: BLE001 - the boot and the hourly pass never break on the trash
        logger.warning("memory: the trash could not be tended (%s: %s); the next pass tries again",
                       type(exc).__name__, exc)
        return 0


def erase(fact_id: str, *, store: Any = None, user_id: str | None = None) -> bool:
    """Erase one of the user's facts in the trash at once; False when it is not theirs, or not in the trash."""
    held = _store(store)
    record = held.get(fact_id, user_id=user_id)
    if record is None or record.active:
        return False
    if not held.erase(fact_id):
        return False
    held.empty_log("a fact erased from the trash")
    return True


def empty(*, store: Any = None, user_id: str | None = None) -> int:
    """Erase every one of the user's facts in the trash at once; how many."""
    held = _store(store)
    erased = sum(1 for record in held.list(active_only=False, user_id=user_id)
                 if not record.active and held.erase(record.id))
    if erased:
        held.empty_log(f"{erased} fact(s) erased from the trash")
    return erased
