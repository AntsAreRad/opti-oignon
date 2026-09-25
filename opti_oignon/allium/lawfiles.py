"""The law and table files both engines read: one source each, pinned by digest.

Law files (``laws/<name>.json``) and table files (``tables/<name>.json``)
are indented on disk, one entry per line, so a change to them reads as a
diff; their digest is the SHA-256 of their canonical OCJ re-emission, so the
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
LAWS = ("fixture",)
TABLES = ("sine_q15_v1",)


def _read(path):
    return wire.parse(path.read_bytes(), lenient=True)


def digest(value):
    """SHA-256 hex of the canonical re-emission of a parsed file."""
    return hashlib.sha256(wire.emit(value)).hexdigest()


def law(name):
    """The parsed law file ``name``; ``KeyError`` for a law this engine does not carry."""
    if name not in LAWS:
        raise KeyError(name)
    return _read(_HERE.joinpath("laws", f"{name}.json"))


def table(name):
    if name not in TABLES:
        raise KeyError(name)
    return _read(_HERE.joinpath("tables", f"{name}.json"))


def sine():
    """The frozen Q15 sine table: 1024 entries, one binary-angle turn."""
    return table("sine_q15_v1")["entries"]
