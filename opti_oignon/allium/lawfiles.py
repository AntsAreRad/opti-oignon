"""The law and table files both engines read: one source each, pinned by digest.

Law files (``laws/<name>.json``), founder pools (``laws/founders_<name>.json``)
and table files (``tables/<name>.json``) are indented on disk, one entry per
line, so a change to them reads as a diff; their digest is the SHA-256 of their canonical OCJ re-emission, so the
indentation never counts. The Rust twin embeds the very same files at build
time; the handshake compares the digests, and a native core built from other
files is not used.

Files are read when asked for, never at import.
"""

import hashlib
from pathlib import Path

from . import wire

checkpoint_before_apply = True

_HERE = Path(__file__).resolve().parent
LAWS = ("fixture", "v0_1")
FOUNDERS = ("fixture", "v1")
TABLES = ("phon_v1", "sine_q15_v1", "taboo_fixture_v1", "taboo_v1")
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
