"""The move of the legacy memory store into the new store (M3a), and its end.

The memory tab and the ``/extract`` route historically wrote the legacy
``memories.db`` (via ``MemoryManager``), while the tool, the sync apply, and the
auto-capture write the coordinated ``MemoryStore`` (canonical SQLite + vector
layer). This module moves the legacy facts into the new store so the split ends
with a single source of truth -- and with a single copy: a fact the user
deletes in the store must not wait in the old one, where a fallback or a
second migration could bring it back.

It is:

* A MOVE. Without the marker, each old fact is added through
  ``MemoryStore.add`` (its double deduplication merges a fact already there),
  and its old row is deleted once it is in; a fact the user had put out of
  service lands in the trash. A fact whose add fails keeps its row, and the
  marker is written only once the old store is empty: the next run finishes.
* AN END. With the marker -- every store migrated before this, whose first
  migration copied and left the old rows in place -- nothing is added: the
  store holds the old facts and what the user did with them since, and an old
  row cannot be told from a fact the user deleted or wiped. An old fact that
  was out of service and landed in service (its text, the source
  ``legacy-import``, written in the hour that migration's marker closes, never
  adopted since, and no old row in service holding that text) goes to the
  trash; then every old row is deleted.
* OVERWRITTEN. The old store overwrites what it deletes, and its write-ahead
  log is emptied after.
* FAIL-SAFE. It never raises: a failure is logged and reported in the result,
  so it can be called from app startup without risking the boot.

The legacy manager and the store are injectable so the logic is unit tested
without touching the on-disk databases.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Every new module hardcodes this; it is never overridable.
checkpoint_before_apply = True

logger = logging.getLogger(__name__)

_MARKER_NAME = ".legacy_memory_migrated"
_FALLBACK_CATEGORIES = frozenset(
    {"identity", "preference", "fact", "contact", "project", "goal"}
)
_FALLBACK_DEFAULT_CATEGORY = "fact"


def _categories() -> frozenset[str]:
    try:
        from .canonical_store import CATEGORIES

        return frozenset(CATEGORIES)
    except Exception:  # pragma: no cover - canonical store optional at import
        return _FALLBACK_CATEGORIES


def _default_category() -> str:
    try:
        from .canonical_store import DEFAULT_CATEGORY

        return str(DEFAULT_CATEGORY)
    except Exception:  # pragma: no cover
        return _FALLBACK_DEFAULT_CATEGORY


def _default_marker_path() -> Path | None:
    try:
        from ..config import DATA_DIR

        return Path(DATA_DIR) / _MARKER_NAME
    except Exception:  # pragma: no cover
        return None


def _map_category(category: str | None) -> str:
    cat = (category or "").strip().lower()
    return cat if cat in _categories() else _default_category()


def _held(manager: Any) -> list:
    """Every fact the old store holds. The real manager reads them so a failure raises (``held``): a store that could
    not be read is never taken for an empty one. A manager that can only list is read that way. An old store whose
    file is not there holds nothing, and is not created to be asked."""
    path = getattr(manager, "db_path", None)
    if isinstance(path, (str, Path)) and not Path(path).exists():
        return []
    held = getattr(manager, "held", None)
    return list(held() if callable(held) else manager.get_all_facts(active_only=False) or ())


def _erase_old(manager: Any, fact: Any) -> bool:
    """Delete one old row; whether it went. A row that stays is said, and the next run deletes it."""
    try:
        gone = bool(manager.delete_fact(getattr(fact, "id", None)))
    except Exception as exc:  # noqa: BLE001 - one row that stays never stops the others
        logger.warning("an old memory's row could not be deleted (%s: %s); the next run deletes it",
                       type(exc).__name__, exc)
        return False
    if not gone:
        logger.warning("an old memory's row could not be deleted; the next run deletes it")
    return gone


def _move(manager: Any, store: Any, facts: list, result: dict) -> None:
    """Add each old fact to the store, then delete its old row: the move of a store never migrated."""
    for fact in facts:
        text = getattr(fact, "fact", None)
        if isinstance(text, str) and text.strip():
            result["scanned"] += 1
            # An old fact out of service makes no fact the user's: a drawn fact it merges into stays drawn.
            options = {} if getattr(fact, "active", True) else {"claim": False}
            try:
                record, decision = store.add(
                    text,
                    _map_category(getattr(fact, "category", None)),
                    source="legacy-import",
                    **options,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("legacy-import add failed, kept in the old store: %s", exc)
                continue
            if getattr(decision, "action", "add") == "merge":
                # The fact already there keeps its text and its state.
                result["merged"] += 1
            else:
                result["added"] += 1
                if not getattr(fact, "active", True):
                    # Out of service in the old store: in the trash here. A copy that cannot go there is taken
                    # back, and its old row waits for the next run, which adds it again.
                    try:
                        if store.soft_delete(record.id):
                            result["trashed"] += 1
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("an old memory out of service could not be put in the trash (%s); its copy "
                                       "taken back, its old row kept", exc)
                        try:
                            store.hard_delete(record.id)
                            result["added"] -= 1
                        except Exception as again:  # noqa: BLE001
                            logger.warning("the copy of an old memory out of service stays in service (%s)", again)
                        continue
        if _erase_old(manager, fact):
            result["erased"] += 1


def _first_migration(marker_path: Any) -> tuple[datetime, datetime] | None:
    """When the first migration copied the old facts: the hour before its marker was written, give or take a minute;
    None when the marker cannot be read, and nothing is put in the trash on a guess."""
    try:
        written = datetime.fromtimestamp(Path(marker_path).stat().st_mtime, timezone.utc)
    except Exception:  # noqa: BLE001 - a marker that cannot be read dates nothing
        return None
    return written - timedelta(hours=1, minutes=1), written + timedelta(minutes=1)


def _landed_then(record: Any, window: tuple[datetime, datetime]) -> bool:
    """Whether ``record`` was written within ``window``."""
    try:
        created = datetime.fromisoformat(str(getattr(record, "created_at", "")))
    except ValueError:
        return False
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return window[0] <= created <= window[1]


def _retire(manager: Any, store: Any, facts: list, result: dict, marker_path: Any = None) -> None:
    """Delete every old row of a store migrated before, adding nothing. First, an old fact out of service whose copy
    the first migration put in service goes to the trash: never one the old store also keeps in service, nor one
    written outside the hour that migration's marker closes -- written since, or a peer's written another time; a
    peer's copy written in that same hour is taken for this store's -- nor one the user confirmed since."""
    result["scanned"] = len(facts)
    in_service = {getattr(fact, "fact", None) for fact in facts if getattr(fact, "active", True)}
    out_of_service = {getattr(fact, "fact", None) for fact in facts if not getattr(fact, "active", True)} - in_service
    out_of_service.discard(None)
    window = _first_migration(marker_path) if out_of_service and marker_path is not None else None
    if window is not None:
        for record in store.list(active_only=True):
            if (getattr(record, "source", "") == "legacy-import" and getattr(record, "text", None) in out_of_service
                    and getattr(record, "endorsed", None) is None and _landed_then(record, window)
                    and store.soft_delete(record.id)):
                result["trashed"] += 1
    for fact in facts:
        if _erase_old(manager, fact):
            result["erased"] += 1


def migrate_legacy_to_store(
    *,
    manager: Any = None,
    store: Any = None,
    marker_path: Path | None = None,
    force: bool = False,
) -> dict:
    """Move the old store's facts into the store, or end an old store migrated before; never raises.

    Without the marker the facts move (``_move``) and the marker is written once the old store is empty; with it,
    the old rows are deleted and nothing is added (``_retire``). ``force`` stays for the route's call: the marker
    decides, since adding the old facts again would bring back every fact the user deleted since.

    Returns a dict with ``scanned``, ``added``, ``merged``, ``trashed`` (old facts out of service put in the trash),
    ``erased`` (old rows deleted), ``skipped_marker`` (the move skipped for the marker) and ``error`` (None on
    success).
    """
    result: dict[str, Any] = {
        "scanned": 0,
        "added": 0,
        "merged": 0,
        "trashed": 0,
        "erased": 0,
        "skipped_marker": False,
        "error": None,
    }
    try:
        if marker_path is None:
            marker_path = _default_marker_path()
        migrated = marker_path is not None and marker_path.exists()
        result["skipped_marker"] = migrated

        if manager is None:
            from .legacy import memory_manager as manager  # singleton
        if manager is None:
            result["error"] = "manager or store unavailable"
            return result
        facts = _held(manager)
        # The store is opened only for old facts to move or to put back in the trash: an empty old store asks
        # nothing of it at each start.
        if facts and store is None:
            from .dedup import get_memory_store

            store = get_memory_store()
        if facts and store is None:
            result["error"] = "manager or store unavailable"
            return result

        if migrated:
            _retire(manager, store, facts, result, marker_path)
        else:
            _move(manager, store, facts, result)
        if result["erased"]:
            try:
                manager.empty_log()
            except Exception as exc:  # noqa: BLE001 - the log goes at the last connection's close
                logger.warning("the old memory store's write-ahead log could not be emptied (%s)", exc)

        if not migrated and marker_path is not None and not _held(manager):
            try:
                marker_path.parent.mkdir(parents=True, exist_ok=True)
                marker_path.write_text("migrated\n", encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                logger.debug("could not write migration marker: %s", exc)

        logger.info(
            "legacy memory migration: scanned=%d added=%d merged=%d trashed=%d erased=%d",
            result["scanned"],
            result["added"],
            result["merged"],
            result["trashed"],
            result["erased"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("legacy memory migration failed, swallowed: %s", exc)
        result["error"] = str(exc)
    return result


def run_boot_migration() -> dict:
    """Startup adapter for the legacy -> store move (M3a-startup).

    The application lifespan calls this once at boot. It delegates to
    ``migrate_legacy_to_store`` with defaults -- which resolves the data-dir
    marker, moves an old store never migrated and ends one migrated before --
    so once the old store is empty it reads an empty store and does nothing.
    Like ``migrate_legacy_to_store`` it NEVER raises: a migration problem must
    not break the boot. Returns the migration result dict.
    """
    try:
        return migrate_legacy_to_store()
    except Exception as exc:  # noqa: BLE001 - boot must never break on migration
        logger.warning("boot migration adapter failed, swallowed: %s", exc)
        return {
            "scanned": 0,
            "added": 0,
            "merged": 0,
            "trashed": 0,
            "erased": 0,
            "skipped_marker": False,
            "error": str(exc),
        }
