"""One-shot migration of the legacy memory store into the new store (M3a).

The memory tab and the ``/extract`` route historically wrote the legacy
``memories.db`` (via ``MemoryManager``), while the tool, the sync apply, and the
auto-capture write the coordinated ``MemoryStore`` (canonical SQLite + vector
layer). This module copies the legacy facts into the new store so the split ends
with a single source of truth.

It is:

* IDEMPOTENT. Each fact is added through ``MemoryStore.add``, whose double
  deduplication merges a re-run instead of duplicating, so running it twice is
  harmless.
* MARKER-GUARDED. After a successful pass a marker file is written in the data
  directory; a later call returns early unless ``force=True``. The marker is an
  optimisation (it avoids re-scanning every boot), not a correctness
  requirement -- the dedup is the real safety net.
* FAIL-SAFE. It never raises: a migration failure is logged and reported in the
  result, so it can be called from app startup without risking the boot.

The legacy manager and the store are injectable so the logic is unit tested
without touching the on-disk databases.
"""

from __future__ import annotations

import logging
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


def migrate_legacy_to_store(
    *,
    manager: Any = None,
    store: Any = None,
    marker_path: Path | None = None,
    force: bool = False,
) -> dict:
    """Copy legacy facts into the new store. Idempotent; never raises.

    Returns a dict with ``scanned``, ``added``, ``merged``, ``skipped_marker``
    and ``error`` (None on success).
    """
    result: dict[str, Any] = {
        "scanned": 0,
        "added": 0,
        "merged": 0,
        "skipped_marker": False,
        "error": None,
    }
    try:
        if marker_path is None:
            marker_path = _default_marker_path()
        if not force and marker_path is not None and marker_path.exists():
            result["skipped_marker"] = True
            return result

        if manager is None:
            from .legacy import memory_manager as manager  # singleton
        if store is None:
            from .dedup import get_memory_store

            store = get_memory_store()
        if manager is None or store is None:
            result["error"] = "manager or store unavailable"
            return result

        facts = manager.get_all_facts(active_only=False)
        for fact in facts or ():
            text = getattr(fact, "fact", None)
            if not isinstance(text, str) or not text.strip():
                continue
            result["scanned"] += 1
            try:
                _record, decision = store.add(
                    text,
                    _map_category(getattr(fact, "category", None)),
                    source="legacy-import",
                )
                if getattr(decision, "action", "add") == "merge":
                    result["merged"] += 1
                else:
                    result["added"] += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("legacy-import add failed, skipped: %s", exc)

        if marker_path is not None:
            try:
                marker_path.parent.mkdir(parents=True, exist_ok=True)
                marker_path.write_text("migrated\n", encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                logger.debug("could not write migration marker: %s", exc)

        logger.info(
            "legacy memory migration: scanned=%d added=%d merged=%d",
            result["scanned"],
            result["added"],
            result["merged"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("legacy memory migration failed, swallowed: %s", exc)
        result["error"] = str(exc)
    return result


def run_boot_migration() -> dict:
    """Startup adapter for the one-shot legacy -> store migration (M3a-startup).

    The application lifespan calls this once at boot. It delegates to
    ``migrate_legacy_to_store`` with defaults -- which resolves the data-dir
    marker, is idempotent (the store's dedup merges a re-run), and is
    marker-guarded -- so it is a no-op after the first successful pass. Like
    ``migrate_legacy_to_store`` it NEVER raises: a migration problem must not
    break the boot. Returns the migration result dict.
    """
    try:
        return migrate_legacy_to_store()
    except Exception as exc:  # noqa: BLE001 - boot must never break on migration
        logger.warning("boot migration adapter failed, swallowed: %s", exc)
        return {
            "scanned": 0,
            "added": 0,
            "merged": 0,
            "skipped_marker": False,
            "error": str(exc),
        }
