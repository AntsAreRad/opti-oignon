"""The law and table files both engines read: one source each, pinned by digest.

Law files (``laws/<name>.json``), founder pools (``laws/founders_<name>.json``)
and table files (``tables/<name>.json``) are indented on disk, one entry per
line, so a change to them reads as a diff; their digest is the SHA-256 of their canonical OCJ re-emission, so the
indentation never counts. The Rust twin embeds the very same files at build
time; the handshake compares the digests, and a native core built from other
files is not used.

Files are read when asked for, never at import.

Two readers serve the platform alone, and neither is embedded in the twin
nor part of the engine's identity: ``retired()`` lists the prototype laws
retired from the tree (``laws/retired.json``), and ``successor()`` finds the
carried stable law that may follow a given one.
"""

import hashlib
from pathlib import Path

from . import wire

checkpoint_before_apply = True

_HERE = Path(__file__).resolve().parent
LAWS = ("fixture", "v0_1")
FOUNDERS = ("fixture", "v1")
TABLES = ("journal_v1", "phon_v1", "sine_q15_v1", "taboo_fixture_v1", "taboo_v1")
JOURNAL_TABLES = ("journal_v1",)
PHON_TABLES = ("phon_v1",)
TABOO_TABLES = ("taboo_fixture_v1", "taboo_v1")


def _read(path):
    return wire.parse(path.read_bytes(), lenient=True)


def digest(value):
    """SHA-256 hex of the canonical re-emission of a parsed file."""
    return hashlib.sha256(wire.emit(value)).hexdigest()


def law_bytes(name):
    """The raw bytes of law file ``name``; ``KeyError`` for a law this engine does not carry."""
    if name not in LAWS:
        raise KeyError(name)
    return _HERE.joinpath("laws", f"{name}.json").read_bytes()


def law(name):
    """The parsed law file ``name``; ``KeyError`` for a law this engine does not carry."""
    return wire.parse(law_bytes(name), lenient=True)


def founders_bytes(name):
    """The raw bytes of founder pool ``name``; ``KeyError`` for a pool this engine does not carry."""
    if name not in FOUNDERS:
        raise KeyError(name)
    return _HERE.joinpath("laws", f"founders_{name}.json").read_bytes()


def founders(name):
    """The parsed founder pool ``name``."""
    return wire.parse(founders_bytes(name), lenient=True)


def table_bytes(name):
    """The raw bytes of table file ``name``; ``KeyError`` for a table this engine does not carry."""
    if name not in TABLES:
        raise KeyError(name)
    return _HERE.joinpath("tables", f"{name}.json").read_bytes()


def table(name):
    return wire.parse(table_bytes(name), lenient=True)


def sine():
    """The frozen Q15 sine table: 1024 entries, one binary-angle turn."""
    return table("sine_q15_v1")["entries"]


def retired():
    """The retired prototype laws, ``[{"name", "sha256"}]``, from ``laws/retired.json``; platform only.

    An entry that is not a name and a digest is left out: it names no law.
    """
    value = _read(_HERE.joinpath("laws", "retired.json"))
    entries = value.get("retired") if isinstance(value, dict) else None
    out = []
    for entry in entries if isinstance(entries, list) else []:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str) and isinstance(entry.get("sha256"), str):
            out.append({"name": entry["name"], "sha256": entry["sha256"]})
    return out


def successor(name, sha256):
    """The carried stable law that may follow law ``(name, sha256)``, ``{"name", "sha256", "v"}``, or ``None``.

    A successor is what the engine's own migration check accepts: a stable
    law naming this one as the law it follows by the identity migration,
    with a higher version (``ref/lawdata.successor_ok``). A provisional law
    has none. When several are carried, the highest version is taken, then
    the first name. Platform only.
    """
    from .ref import lawdata

    try:
        source = lawdata.life(name)
    except wire.Refused:
        return None
    if source.provisional or source.digest != sha256:
        return None
    found = None
    for candidate in LAWS:
        try:
            life = lawdata.life(candidate)
        except wire.Refused:
            continue
        if not lawdata.successor_ok(life, {"name": name, "sha256": sha256}):
            continue
        if found is None or life.version > found["v"]:
            found = {"name": life.name, "sha256": life.digest, "v": life.version}
    return found
