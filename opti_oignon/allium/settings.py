"""The componion's persistence settings: one strict reader for ``persistence`` in ``config/allium.yaml``.

Three keys, each read on its own terms:

* ``require_encryption`` loosens only when it is exactly the YAML boolean
  ``false``. A missing key, the string ``"false"``, ``0``, ``null`` or a file
  that cannot be read all mean ``true``: a birth in clear is never chosen by
  accident.
* ``path`` is where the stores live, relative to the data directory. Absent,
  it is ``"allium"``. Present, it must be a non-empty relative string with no
  ``..`` and no empty part, or it is refused by name -- never replaced in
  silence by a default the person did not choose.
* ``busy_timeout_ms`` is how long a write waits for another writer: an
  integer (not a boolean) in 1..=60000, else 5000, and the fallback is logged.

The settings govern births only; after a birth the being's genesis governs
its files. ``yaml`` is imported when the file is read, the file is found from
this package's location, and the parsed result is kept until the file's
modification time changes.
"""

import logging
from pathlib import Path

checkpoint_before_apply = True

logger = logging.getLogger(__name__)

DEFAULT_PATH = "allium"
DEFAULT_BUSY_TIMEOUT_MS = 5000
BUSY_TIMEOUT_MAX_MS = 60000

_cache = {"key": None, "value": None}


def config_file():
    """``opti_oignon/config/allium.yaml``, found from this package's location."""
    return Path(__file__).resolve().parent.parent.joinpath("config", "allium.yaml")


def _refuse_path(detail):
    from .store import StoreRefused

    return StoreRefused("path", detail)


def _path(raw):
    if not isinstance(raw, str) or raw == "":
        raise _refuse_path("persistence.path is not a non-empty string")
    if raw.startswith("/") or raw.startswith("\\"):
        raise _refuse_path("persistence.path is absolute")
    parts = raw.split("/")
    for part in parts:
        if part == "":
            raise _refuse_path("persistence.path has an empty part")
        if part == "..":
            raise _refuse_path("persistence.path climbs out with '..'")
    return raw


def _busy(raw):
    if isinstance(raw, int) and not isinstance(raw, bool) and 1 <= raw <= BUSY_TIMEOUT_MAX_MS:
        return raw
    logger.warning("persistence.busy_timeout_ms is not an integer in 1..=%d: %d is used",
                   BUSY_TIMEOUT_MAX_MS, DEFAULT_BUSY_TIMEOUT_MS)
    return DEFAULT_BUSY_TIMEOUT_MS


def normalise(section):
    """The ``persistence`` mapping read by the rules above; a bad ``path`` is refused by name.

    ``section`` is what the YAML holds under ``persistence`` (or a mapping a
    caller injects); anything that is not a mapping reads as an empty one.
    """
    if not isinstance(section, dict):
        section = {}
    path = _path(section["path"]) if "path" in section else DEFAULT_PATH
    busy = _busy(section["busy_timeout_ms"]) if "busy_timeout_ms" in section else DEFAULT_BUSY_TIMEOUT_MS
    return {
        "busy_timeout_ms": busy,
        "path": path,
        "require_encryption": section.get("require_encryption") is not False,
    }


def _read(file):
    try:
        import yaml

        data = yaml.safe_load(file.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - an unreadable file reads as the strict defaults
        logger.warning("the componion's settings could not be read: encryption stays required")
        return None
    return data if isinstance(data, dict) else None


def persistence(path=None):
    """The persistence settings, from ``path`` or the package's own YAML file."""
    file = Path(path) if path is not None else config_file()
    try:
        stamp = file.stat().st_mtime_ns
    except OSError:
        stamp = None
    key = (str(file), stamp)
    if _cache["key"] == key:
        return dict(_cache["value"])
    data = _read(file) if stamp is not None else None
    value = normalise(data.get("persistence") if data is not None else None)
    _cache["key"] = key
    _cache["value"] = value
    return dict(value)
