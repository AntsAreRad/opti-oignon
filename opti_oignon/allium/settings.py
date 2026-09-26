"""The componion's settings: strict readers for ``persistence``, ``laws``, ``life`` and ``api`` in ``config/allium.yaml``.

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

``enabled`` is the garden's switch (``switch``): ``"on"`` only when the
file's top level is a mapping whose ``enabled`` is the YAML boolean
``true``; ``"off"`` for a missing file, a missing key or any other value
(the string ``"true"``, ``1``); ``"unreadable"`` for a file that exists and
does not parse, or whose top level is not a mapping -- and a switch that
cannot be read is off, said as such, once per modification of the file.

``api`` is read only for the API's garden routes, never by ``oo garden``
(``api``, an ``Api``): ``hosts`` are the names those routes answer beside
the loopback names, which are always allowed, for the Host and the Origin
of a request -- a YAML list of exact lowercase names or bracketed IPv6
literals, with no port, scheme or wildcard. A list that is not one, holds
one entry that is not such a name, names an unspecified address
(``0.0.0.0``, ``[::]`` and their spellings, which some browsers route to
loopback), or spells an IPv4 address otherwise than as its dotted quad
(``0``, ``0x0``, ``127.1``, which a browser reads as the address itself)
allows the loopback names only, ``()``, and is logged; a missing key is
``()``. The API's router reads the same names by the same rule without
importing this package, and a contract holds the two readers equal.
``python_cap`` and ``native_cap`` bound the engine's work in one request,
with the Python reference and with the native core: integers (not
booleans) in 1..=2^53-1, else 200000 and 5000000, and the fallback is
logged; a missing key reads its default. A file that cannot be read reads
every default. The service never caps a view below one awake day of the
laws the engine carries (``service.api_view_cap``).

``yaml`` is imported when the file is read, the file is found from this
package's location, and each reader keeps its parsed result, in a cache of
its own, until the file changes: its modification time, and for ``switch``
and ``api`` its size and inode too.

Threads. A long-lived process (the API) calls ``switch`` and ``api`` from
many threads at once: each keeps its cache as one ``(key, value)`` pair,
stored in one assignment, so no reader pairs a new key with an old value.
``persistence``, ``life`` and ``laws`` write their key and their value in
two statements: at most one reader sees the value from before a change,
which is what a read made just before the change would have seen.
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
DEFAULT_PYTHON_CAP = 200000
DEFAULT_NATIVE_CAP = 5000000
# A name of ``api.hosts``: an exact lowercase name, or a bracketed IPv6 literal; no port, scheme or wildcard.
HOST_PATTERN = r"[a-z0-9-]+(\.[a-z0-9-]+)*|\[[0-9a-f:]+\]"
DEFAULT_CHECKPOINTS = {"daily": 14, "monthly": True, "weekly": 52}

# Each param and each sowing field of the proposal, and where it lives under ``laws`` in the file.
PARAM_KEYS = {"evap_awake": ("soil", "evap_awake"), "evap_dormant": ("soil", "evap_dormant"),
              "rain_gain": ("soil", "rain_gain"), "sun_max": ("light", "sun_max")}
SOWING_KEYS = {"band": ("seasons", "default_band"), "hemisphere": ("seasons", "default_hemisphere"),
               "weather": ("weather", "mode")}
UNREADABLE = "the settings file cannot be read"
# What a file that does not parse reads as, apart from a file that parses to nothing.
_UNPARSED = object()

_cache = {"key": None, "value": None}
_life_cache = {"key": None, "value": None}
_laws_cache = {"key": None, "value": None}
# ``(key, value)`` pairs, each stored in one assignment: a thread never pairs a new key with an old value.
_switch_cache = {"entry": None}
_api_cache = {"entry": None}


class Malformed(NamedTuple):
    """A proposal value the reader could not read as one; ``detail`` names it, and it is refused where used."""

    detail: str


class Api(NamedTuple):
    """The ``api`` section as read: the names the garden's routes answer beside loopback, and the two caps."""

    hosts: tuple
    python_cap: int
    native_cap: int


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


def _file_key(path):
    """``(file, cache key)``: the file, its modification time, size and inode; ``None`` for them when absent."""
    file = Path(path) if path is not None else config_file()
    try:
        st = file.stat()
    except OSError:
        return file, (str(file), None)
    return file, (str(file), st.st_mtime_ns, st.st_size, st.st_ino)


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


def switch(path=None):
    """The garden's switch in ``path`` or the package's own file: ``"on"``, ``"off"`` or ``"unreadable"``.

    ``"on"`` only for a top-level mapping whose ``enabled`` is the YAML
    boolean ``true`` (``true`` or ``yes``). A missing file or key, and every
    other value, is ``"off"``; a file that exists and does not parse, or
    whose top level is not a mapping, is ``"unreadable"``, logged once per
    modification. Cached until the file changes (its modification time,
    size and inode).
    """
    file, key = _file_key(path)
    entry = _switch_cache["entry"]
    if entry is not None and entry[0] == key:
        return entry[1]
    if key[1] is None:
        value = "off"
    else:
        try:
            import yaml

            data = yaml.safe_load(file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a switch that cannot be read is off, and said to be unreadable
            data = _UNPARSED
        if data is _UNPARSED or not isinstance(data, dict):
            value = "unreadable"
            logger.warning("the componion's settings file cannot be read: the garden is off")
        else:
            value = "on" if data.get("enabled") is True else "off"
    _switch_cache["entry"] = (key, value)
    return value


def enabled(path=None):
    """Whether the garden is on: exactly ``switch(path) == "on"``."""
    return switch(path) == "on"


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


def _api_cap(section, name, default):
    if name not in section:
        return default
    raw = section[name]
    if isinstance(raw, int) and not isinstance(raw, bool) and 1 <= raw <= MAX_INT:
        return raw
    logger.warning("api.%s is not an integer in 1..=%d: %d is used", name, MAX_INT, default)
    return default


def _unspecified(entry):
    """Whether ``entry`` is an address no host has: ``0.0.0.0``, ``[::]``, or ``[::ffff:0.0.0.0]``."""
    import ipaddress

    try:
        address = ipaddress.ip_address(entry[1:-1] if entry.startswith("[") else entry)
    except ValueError:
        return None
    mapped = getattr(address, "ipv4_mapped", None)
    return address.is_unspecified or (mapped is not None and mapped.is_unspecified)


def _other_spelling(entry):
    """Whether ``entry`` is an IPv4 address spelled otherwise than as its dotted quad (``0``, ``0x0``, ``127.1``).

    A browser reads such a name as the address itself (``0`` and ``0x0`` as
    ``0.0.0.0``), and sends the dotted quad in its Host: listed as written,
    it names an address that no request is addressed by, or the unspecified one.
    """
    import socket

    if entry.startswith("["):
        return False
    try:
        packed = socket.inet_aton(entry)
    except (OSError, ValueError):
        return False
    return socket.inet_ntoa(packed) != entry


def _api_hosts(section):
    """The names of ``api.hosts``: ``()`` when the key is missing, and, logged, for anything that is not a list of them.

    A bracketed entry must be an IPv6 literal the ``ipaddress`` module reads.
    """
    import re

    if "hosts" not in section:
        return ()
    raw = section["hosts"]
    if not isinstance(raw, list):
        logger.warning("api.hosts is not a list of names: only the loopback names are allowed")
        return ()
    for entry in raw:
        if not isinstance(entry, str) or not re.fullmatch(HOST_PATTERN, entry):
            logger.warning("api.hosts holds an entry that is not a lowercase name with no port or scheme: "
                           "only the loopback names are allowed")
            return ()
        unspecified = _unspecified(entry)
        if unspecified is None and entry.startswith("["):
            logger.warning("api.hosts holds a bracketed entry that is not an address: only the loopback names "
                           "are allowed")
            return ()
        if unspecified:
            logger.warning("api.hosts names an unspecified address: only the loopback names are allowed")
            return ()
        if _other_spelling(entry):
            logger.warning("api.hosts holds an address that is not written as its dotted quad: only the loopback "
                           "names are allowed")
            return ()
    return tuple(raw)


def normalise_api(section):
    """The ``api`` mapping read by the rules above, as an ``Api``; never refuses, every fallback logged.

    ``section`` is what the YAML holds under ``api`` (or a mapping a caller
    injects); ``None`` reads as all defaults, and anything else that is not
    a mapping falls back to them with a warning.
    """
    if section is None:
        section = {}
    elif not isinstance(section, dict):
        logger.warning("api is not a mapping: its defaults are used")
        section = {}
    return Api(_api_hosts(section), _api_cap(section, "python_cap", DEFAULT_PYTHON_CAP),
               _api_cap(section, "native_cap", DEFAULT_NATIVE_CAP))


def api(path=None):
    """The ``api`` settings (``Api``), from ``path`` or the package's own file; a file that cannot be read: defaults.

    Cached until the file changes (its modification time, size and inode),
    as one pair stored in one assignment.
    """
    file, key = _file_key(path)
    entry = _api_cache["entry"]
    if entry is not None and entry[0] == key:
        return entry[1]
    data = _read(file, "the api defaults are used") if key[1] is not None else None
    value = normalise_api(data.get("api") if data is not None else None)
    _api_cache["entry"] = (key, value)
    return value
