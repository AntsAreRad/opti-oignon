"""The componion's settings: strict readers for ``persistence``, ``laws`` and ``life`` in ``config/allium.yaml``.

``persistence`` has three keys, each read on its own terms:

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
its files.

``laws`` is the proposal for a new being, frozen into its genesis at its
birth: four params (``light.sun_max`` and the soil's ``evap_awake``,
``evap_dormant`` and ``rain_gain``) and three sowing fields
(``seasons.default_hemisphere``, ``seasons.default_band`` and
``weather.mode``). The reader returns each value raw, ``None`` when it is
missing, and never judges it against a law: a present but malformed value is
refused by name where it is used, never replaced in silence. What the reader
cannot read as a value -- a section that is not a mapping, a key the proposal
does not have, a file that does not parse -- reaches the fields it touches
as ``Malformed``, which names it, and is refused there too. A replay never
reads it: a living being's params change only through a law update.

``life`` is machine policy: it decides which caches are kept and when the
recorder notes a clock that was set back, never what a being is, and a
replay never reads it. ``skew_note_min`` (how many minutes behind the latest
fact the wall may read before a ``clock`` fact says so) and the checkpoint
retention ``daily`` and ``weekly`` are integers (not booleans) in
0..=2^53-1, ``monthly`` is a boolean; a malformed value falls back to its
default and the fallback is logged, a missing one reads its default.

``yaml`` is imported when the file is read, the file is found from this
package's location, and each reader keeps its parsed result, in a cache of
its own, until the file's modification time changes.
"""

import logging
from pathlib import Path
from typing import NamedTuple

checkpoint_before_apply = True

logger = logging.getLogger(__name__)

DEFAULT_PATH = "allium"
DEFAULT_BUSY_TIMEOUT_MS = 5000
BUSY_TIMEOUT_MAX_MS = 60000
MAX_INT = (1 << 53) - 1
DEFAULT_SKEW_NOTE_MIN = 5
DEFAULT_CHECKPOINTS = {"daily": 14, "monthly": True, "weekly": 52}

# Each param and each sowing field of the proposal, and where it lives under ``laws`` in the file.
PARAM_KEYS = {"evap_awake": ("soil", "evap_awake"), "evap_dormant": ("soil", "evap_dormant"),
              "rain_gain": ("soil", "rain_gain"), "sun_max": ("light", "sun_max")}
SOWING_KEYS = {"band": ("seasons", "default_band"), "hemisphere": ("seasons", "default_hemisphere"),
               "weather": ("weather", "mode")}
UNREADABLE = "the settings file cannot be read"

_cache = {"key": None, "value": None}
_life_cache = {"key": None, "value": None}
_laws_cache = {"key": None, "value": None}


class Malformed(NamedTuple):
    """A proposal value the reader could not read as one; ``detail`` names it, and it is refused where used."""

    detail: str


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


def _read(file, consequence="encryption stays required"):
    try:
        import yaml

        data = yaml.safe_load(file.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - an unreadable file reads as the strict defaults
        logger.warning("the componion's settings could not be read: %s", consequence)
        return None
    return data if isinstance(data, dict) else None


def _stamped(path):
    """``(file, cache key)``: the file and its modification time, ``None`` when it cannot be read."""
    file = Path(path) if path is not None else config_file()
    try:
        stamp = file.stat().st_mtime_ns
    except OSError:
        stamp = None
    return file, (str(file), stamp)


def persistence(path=None):
    """The persistence settings, from ``path`` or the package's own YAML file."""
    file, key = _stamped(path)
    if _cache["key"] == key:
        return dict(_cache["value"])
    data = _read(file) if key[1] is not None else None
    value = normalise(data.get("persistence") if data is not None else None)
    _cache["key"] = key
    _cache["value"] = value
    return dict(value)


def _count(name, raw, default):
    if isinstance(raw, int) and not isinstance(raw, bool) and 0 <= raw <= MAX_INT:
        return raw
    logger.warning("life.%s is not an integer in 0..=%d: %d is used", name, MAX_INT, default)
    return default


def normalise_life(section):
    """The ``life`` mapping read by the rules above: every value an integer (or ``monthly`` a boolean).

    ``section`` is what the YAML holds under ``life`` (or a mapping a caller
    injects); ``None`` reads as all defaults, and anything else that is not a
    mapping falls back to them with a warning. Never refuses: it is policy.
    """
    if section is None:
        section = {}
    elif not isinstance(section, dict):
        logger.warning("life is not a mapping: its defaults are used")
        section = {}
    skew = section.get("skew_note_min", DEFAULT_SKEW_NOTE_MIN)
    checkpoints = section.get("checkpoints", DEFAULT_CHECKPOINTS)
    if not isinstance(checkpoints, dict):
        logger.warning("life.checkpoints is not a mapping: its defaults are used")
        checkpoints = DEFAULT_CHECKPOINTS
    monthly = checkpoints.get("monthly", DEFAULT_CHECKPOINTS["monthly"])
    if not isinstance(monthly, bool):
        logger.warning("life.checkpoints.monthly is not a boolean: %s is used", DEFAULT_CHECKPOINTS["monthly"])
        monthly = DEFAULT_CHECKPOINTS["monthly"]
    return {
        "checkpoints": {
            "daily": _count("checkpoints.daily", checkpoints.get("daily", DEFAULT_CHECKPOINTS["daily"]),
                            DEFAULT_CHECKPOINTS["daily"]),
            "monthly": monthly,
            "weekly": _count("checkpoints.weekly", checkpoints.get("weekly", DEFAULT_CHECKPOINTS["weekly"]),
                             DEFAULT_CHECKPOINTS["weekly"]),
        },
        "skew_note_min": _count("skew_note_min", skew, DEFAULT_SKEW_NOTE_MIN),
    }


def life(path=None):
    """The life settings, from ``path`` or the package's own YAML file; a file that cannot be read gives the defaults."""
    file, key = _stamped(path)
    if _life_cache["key"] == key:
        value = _life_cache["value"]
    else:
        data = _read(file, "the life defaults are used") if key[1] is not None else None
        value = normalise_life(data.get("life") if data is not None else None)
        _life_cache["key"] = key
        _life_cache["value"] = value
    return {"checkpoints": dict(value["checkpoints"]), "skew_note_min": value["skew_note_min"]}


def _groups():
    """``{group: {field: (section, key)}}``: the params, then the sowing fields."""
    return {"params": PARAM_KEYS, "sowing": SOWING_KEYS}


def _all(value):
    return {group: {name: value for name in keys} for group, keys in _groups().items()}


def normalise_laws(section):
    """The ``laws`` proposal read by the rules above: ``{"params": {...}, "sowing": {...}}``, raw or ``None``.

    ``section`` is what the YAML holds under ``laws`` (or a mapping a caller
    injects). Nothing here is judged against a law; what cannot be read as a
    value is a ``Malformed`` naming it, in every field it touches: a section
    that is not a mapping, in its own fields, and a key the proposal does not
    have, in the fields of its section (every field, for a section it does
    not have).
    """
    out = _all(None)
    if section is None:
        return out
    if not isinstance(section, dict):
        return _all(Malformed("laws is not a mapping"))
    fields = {}
    for group, keys in _groups().items():
        for name, (part, key) in keys.items():
            fields.setdefault(part, []).append((group, name, key))
    for part in sorted(fields):
        value = section.get(part)
        if value is None:
            continue
        if not isinstance(value, dict):
            problem = Malformed(f"{part} is not a mapping")
        else:
            known = [key for _group, _name, key in fields[part]]
            unknown = sorted(str(key) for key in value if key not in known)
            problem = Malformed(f"{part}.{unknown[0]} is not a key of the proposal") if unknown else None
        for group, name, key in fields[part]:
            out[group][name] = problem if problem is not None else value.get(key)
    unknown = sorted(str(part) for part in section if part not in fields)
    if unknown:
        return _all(Malformed(f"{unknown[0]} is not a section of the proposal"))
    return out


def laws(path=None):
    """The ``laws`` proposal, from ``path`` or the package's own YAML file; a missing file proposes nothing.

    A file that exists and does not parse, or whose top level is not a
    mapping, makes every field ``Malformed``: the proposal is never read as
    the law's defaults in silence.
    """
    file, key = _stamped(path)
    if _laws_cache["key"] != key:
        section, readable = None, True
        if key[1] is not None:
            try:
                import yaml

                data = yaml.safe_load(file.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - refused by name where the proposal is used
                data, readable = None, False
            if data is not None and not isinstance(data, dict):
                readable = False
            if readable and data is not None:
                section = data.get("laws")
        _laws_cache["key"] = key
        _laws_cache["value"] = (section, readable)
    section, readable = _laws_cache["value"]
    if not readable:
        return _all(Malformed(UNREADABLE))
    return normalise_laws(section)
